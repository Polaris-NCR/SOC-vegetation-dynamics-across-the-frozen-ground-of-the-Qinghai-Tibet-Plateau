import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import joblib

from Unet_model import ConditionalUNet
from Diffusion import Diffusion
from Regression_Baseline import (
    cal_sam, cal_spectral_gradient,
    compute_metrics, load_band_dataloaders,
)
BAND_HYPERION_COUNTS = {
    'Red':   6,
    'NIR':   5,
    'Blue':  7,
    'Green': 8,
    'SWIR1': 11,
    'SWIR2': 19,
}

# BAND_ORDER = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
BAND_ORDER = ['Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']   
BAND_COND_IDX = {'Blue': 0, 'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR1': 4, 'SWIR2': 5}

# ============================================================
# 1. 空间数据增强（与基线一致）
# ============================================================

def spatial_augment(landsat, hyperion):
    if torch.rand(1) > 0.5:
        landsat = torch.flip(landsat, dims=[-1])
        hyperion = torch.flip(hyperion, dims=[-1])
    if torch.rand(1) > 0.5:
        landsat = torch.flip(landsat, dims=[-2])
        hyperion = torch.flip(hyperion, dims=[-2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        landsat = torch.rot90(landsat, k, dims=[-2, -1])
        hyperion = torch.rot90(hyperion, k, dims=[-2, -1])
    return landsat, hyperion


# ============================================================
# 2. 扩散模型训练器（+SRF 物理约束）
# ============================================================

class PerBandSRFDiffusionTrainer:
    """
    管理一个 Landsat 波段 → Hyperion 波段组的 DDPM 训练全流程。

    模型架构：ConditionalUNet（pix2pix 通道拼接 + 多尺度交叉注意力）
    训练范式：v-prediction MSE (Salimans & Ho, 2022)
             + 辅助损失 (L1 + SAM + 光谱梯度)
             + SRF 物理约束损失
             + SRT 重激活
    验证范式：v-prediction 损失（早停）+ DDIM 加速采样（重建指标）

    v = sqrt(ᾱ_t)·ε − sqrt(1−ᾱ_t)·x₀

    SRF 约束:
        L_SRF = || Σ ω_i · x̂_{0,i} − y ||_1
        确保重建的窄带 Hyperion 经 SRF 降尺度后匹配原始 Landsat 观测，
        为后续生态指数约束提供物理一致性基础。
    """

    def __init__(self, band_name, base_channels=32,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 lr=1e-4, weight_decay=1e-4, device='cuda',
                 ddim_steps=50,
                 lambda_sam=0.1, lambda_grad=0.1, lambda_srf=0.1,
                 max_react_prob=0.25, warmup_react_epochs=10,
                 aux_weight_floor=0.05, pretrained_path=None):
        self.band_name = band_name
        self.cond_idx = BAND_COND_IDX[band_name]
        self.device = device
        self.ddim_steps = ddim_steps
        self.lambda_sam = lambda_sam
        self.lambda_grad = lambda_grad
        self.lambda_srf = lambda_srf
        self.max_react_prob = max_react_prob
        self.warmup_react_epochs = warmup_react_epochs
        self.aux_weight_floor = aux_weight_floor

        n_hyper = BAND_HYPERION_COUNTS[band_name]

        # ---- DDPM 扩散过程（T=1000 cosine 调度 + SRF 权重矩阵） ----
        self.diffusion = Diffusion(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            band=band_name,
            device=device,
            schedule='cosine',
            use_srf=True,
        )
        # SRF 投影矩阵: [1, N_b] → 降尺度后 Landsat 单波段
        self.srf_matrix = self.diffusion.srf_matrix

        # ---- ConditionalUNet ----
        # 输出：预测 v [B, N_hyper, H, W]
        self.model = ConditionalUNet(
            in_channels=n_hyper,
            cond_channels=1,
            out_channels=n_hyper,
            base_channels=base_channels,
        ).to(device)

        # ---- 回归预训练权重初始化（可选） ----
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            reg_state = torch.load(pretrained_path, map_location=device)
            if any(k.startswith('unet.') for k in reg_state.keys()):
                reg_state = {k[5:]: v for k, v in reg_state.items() if k.startswith('unet.')}
            missing, unexpected = self.model.load_state_dict(reg_state, strict=False)
            if missing:
                print(f"  [{band_name}] Pretrained weights loaded from {pretrained_path}")
                print(f"  [{band_name}] Missing keys (random init): "
                      f"{[k for k in missing if 'time_mlp' not in k][:5]}...")
            else:
                print(f"  [{band_name}] Pretrained weights fully loaded from {pretrained_path}")
        elif pretrained_path is not None:
            print(f"  [{band_name}] WARNING: pretrained_path={pretrained_path} not found "
                  f"— training from scratch")

        # ---- 优化器 ----
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # ---- 学习率调度器：Cosine Warm Restarts (SGDR) ----
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6)

        # ---- 历史记录 ----
        self.history = {
            'epoch': [],
            'train_loss': [],      'val_loss': [],
            'train_RMSE': [],      'val_RMSE': [],
            'train_PSNR': [],      'val_PSNR': [],
            'train_SAM': [],       'val_SAM': [],
            'train_SSIM': [],      'val_SSIM': [],
            'train_PCC': [],       'val_PCC': [],
            'train_ERGAS': [],     'val_ERGAS': [],
        }

        m_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{band_name}] Model params: {m_params / 1e6:.2f}M  |  "
              f"Timesteps: {num_timesteps} (cosine, v-prediction, +SRF {lambda_srf})  |  "
              f"SRT: max_prob={max_react_prob}, warmup={warmup_react_epochs}ep")

    def get_react_prob(self, epoch):
        """SRT 重激活概率调度：warmup 期内为 0，之后线性增长至 max_react_prob。"""
        if epoch < self.warmup_react_epochs:
            return 0.0
        elif self.max_react_prob <= 0:
            return 0.0
        else:
            progress = min((epoch - self.warmup_react_epochs) / 30.0, 1.0)
            return self.max_react_prob * progress

    def record_history(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        for k in ['RMSE', 'PSNR', 'SAM', 'SSIM', 'PCC', 'ERGAS']:
            self.history[f'train_{k}'].append(train_metrics[k])
            self.history[f'val_{k}'].append(val_metrics[k])

    def train_epoch(self, train_loader, epoch):
        """训练阶段：v-prediction MSE + 辅助损失 + SRF 约束 + SRT 重激活。"""
        self.model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_l1 = 0.0
        total_sam = 0.0
        total_sgrad = 0.0
        total_srf = 0.0
        total_react = 0.00

        react_count = 0
        p_react = self.get_react_prob(epoch)

        for batch_idx, (landsat_band, hyperion_band) in enumerate(train_loader):
            landsat_band = landsat_band.to(self.device, dtype=torch.float32)
            hyperion_band = hyperion_band.to(self.device, dtype=torch.float32)

            # 归一化到 [-1, 1]
            landsat_norm = landsat_band * 2 - 1
            hyperion_norm = hyperion_band * 2 - 1

            B = landsat_band.size(0)
            T = self.diffusion.num_timesteps

            # ============================================================
            # Path A: v-prediction 训练
            # ============================================================
            t = torch.randint(0, T, (B,), device=self.device).long()
            noise = torch.randn_like(hyperion_norm)
            x_t = self.diffusion.q_sample(hyperion_norm, t, noise)

            # 模型预测 v
            v_pred = self.model(x_t, landsat_norm, t)

            # v 训练目标
            v_target = self.diffusion.v_target(hyperion_norm, noise, t)
            loss_v = F.mse_loss(v_pred, v_target)

            # 辅助损失：从 v̂ 恢复 x̂₀ 进行重建质量监督
            x0_pred = self.diffusion.predict_x0_from_v(x_t, t, v_pred)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            ab_clamped = self.diffusion.alpha_bar[t].detach()
            aux_w = self.aux_weight_floor + (1.0 - self.aux_weight_floor) * ab_clamped
            l1_weight = aux_w.reshape(-1, 1, 1, 1)
            loss_l1 = (torch.abs(x0_pred - hyperion_norm) * l1_weight).mean()

            t_weight = aux_w.mean()
            loss_sam = cal_sam(x0_pred, hyperion_norm) * t_weight
            loss_sgrad = F.l1_loss(
                cal_spectral_gradient(x0_pred),
                cal_spectral_gradient(hyperion_norm)
            ) * t_weight

            # SRF 物理约束：x̂₀ → SRF 降尺度 → 匹配 Landsat
            # srf_matrix: [1, N_b], x0_pred: [B, N_b, H, W]
            # → landsat_sim: [B, 1, H, W]
            x0_denorm = (x0_pred + 1) / 2  # 反归一化到 [0, 1] 用于 SRF 计算
            landsat_sim = torch.einsum('ij, bjhw -> bihw',
                                       self.srf_matrix, x0_denorm)
            loss_srf = F.l1_loss(landsat_sim, landsat_band)

            loss = (loss_v + loss_l1
                    + self.lambda_sam * loss_sam
                    + self.lambda_grad * loss_sgrad
                    + self.lambda_srf * loss_srf)

            # ============================================================
            # Path B: SRT 重激活（v-prediction 版本）
            # ============================================================
            do_react = (p_react > 0 and torch.rand(1).item() < p_react
                        and t.min().item() >= T // 2)
            loss_react = torch.tensor(0.0, device=self.device)

            if do_react:
                with torch.no_grad():
                    x0_detached = x0_pred.detach()
                t2 = torch.randint(10, max(t.min().item(), 11), (B,),
                                   device=self.device).long()
                noise2 = torch.randn_like(hyperion_norm)
                x_t2 = self.diffusion.q_sample(x0_detached, t2, noise2)
                v_pred2 = self.model(x_t2, landsat_norm, t2)
                v_target2 = self.diffusion.v_target(x0_detached, noise2, t2)
                loss_react = F.mse_loss(v_pred2, v_target2)

                loss = loss + loss_react
                total_react += loss_react.item()
                react_count += 1

            # ============================================================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_v_loss += loss_v.item()
            total_l1 += loss_l1.item()
            total_sam += loss_sam.item()
            total_sgrad += loss_sgrad.item()
            total_srf += loss_srf.item()

            if batch_idx % 50 == 0:
                msg = (f"  [{self.band_name}] Epoch {epoch} | Batch {batch_idx} | "
                       f"Loss: {loss.item():.4f} (v={loss_v.item():.4f}, "
                       f"L1={loss_l1.item():.4f}, SAM={loss_sam.item():.4f}, "
                       f"Grad={loss_sgrad.item():.4f}, SRF={loss_srf.item():.4f}")
                if do_react:
                    msg += f", react={loss_react.item():.4f}"
                msg += f") | p_react={p_react:.2f}"
                print(msg)

        n = len(train_loader)
        avg_loss = total_loss / n
        if react_count > 0:
            print(f"  [{self.band_name}] Epoch {epoch} summary: "
                  f"avg_loss={avg_loss:.4f}, v={total_v_loss/n:.4f}, "
                  f"L1={total_l1/n:.4f}, SAM={total_sam/n:.4f}, "
                  f"Grad={total_sgrad/n:.4f}, SRF={total_srf/n:.4f}, "
                  f"react={total_react/react_count:.4f} "
                  f"(×{react_count}/{n} batches)")

        placeholder = {'RMSE': 0.0, 'PSNR': 0.0, 'SAM': 0.0,
                       'SSIM': 0.0, 'PCC': 0.0, 'ERGAS': 0.0}
        return avg_loss, placeholder

    @torch.no_grad()
    def validate(self, val_loader):
        """验证阶段：DDIM 重建指标 + 全损失分量（含 SRF）诊断。"""
        self.model.eval()
        val_loss = 0.0
        val_v_loss = 0.0
        val_l1_loss = 0.0
        val_sam_loss = 0.0
        val_sgrad_loss = 0.0
        val_srf_loss = 0.0
        v_bucket = {'low': 0.0, 'mid': 0.0, 'high': 0.0}
        bucket_counts = {'low': 0, 'mid': 0, 'high': 0}
        all_metrics = {'RMSE': 0.0, 'PSNR': 0.0, 'SAM': 0.0,
                       'SSIM': 0.0, 'PCC': 0.0, 'ERGAS': 0.0}
        T = self.diffusion.num_timesteps
        t_low = T // 3
        t_high = 2 * T // 3

        for landsat_band, hyperion_band in val_loader:
            landsat_band = landsat_band.to(self.device, dtype=torch.float32)
            hyperion_band = hyperion_band.to(self.device, dtype=torch.float32)
            B = landsat_band.size(0)

            landsat_norm = landsat_band * 2 - 1
            hyperion_norm = hyperion_band * 2 - 1

            # ---- DDIM 加速采样（先行，作为早停判据） ----
            pred = self.ddim_sample(landsat_band)
            batch_m = compute_metrics(pred, hyperion_band)
            for k in all_metrics:
                all_metrics[k] += batch_m.get(k, 0.0)

            # ---- v-prediction 损失（诊断用，与 train_epoch 对齐） ----
            t_rand = torch.randint(0, T, (B,), device=self.device).long()
            noise = torch.randn_like(hyperion_norm)
            x_tr = self.diffusion.q_sample(hyperion_norm, t_rand, noise)
            v_pred = self.model(x_tr, landsat_norm, t_rand)
            v_target = self.diffusion.v_target(hyperion_norm, noise, t_rand)

            v_mse_per_sample = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2, 3])
            for i in range(B):
                ti = t_rand[i].item()
                if ti < t_low:
                    v_bucket['low'] += v_mse_per_sample[i].item()
                    bucket_counts['low'] += 1
                elif ti < t_high:
                    v_bucket['mid'] += v_mse_per_sample[i].item()
                    bucket_counts['mid'] += 1
                else:
                    v_bucket['high'] += v_mse_per_sample[i].item()
                    bucket_counts['high'] += 1

            loss_v = v_mse_per_sample.mean()
            x0_pred = self.diffusion.predict_x0_from_v(x_tr, t_rand, v_pred)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            ab_v = self.diffusion.alpha_bar[t_rand].detach()
            aux_wv = self.aux_weight_floor + (1.0 - self.aux_weight_floor) * ab_v
            l1_weight = aux_wv.reshape(-1, 1, 1, 1)
            loss_l1 = (torch.abs(x0_pred - hyperion_norm) * l1_weight).mean()
            t_weight = aux_wv.mean()
            loss_sam = cal_sam(x0_pred, hyperion_norm) * t_weight
            loss_sgrad = F.l1_loss(
                cal_spectral_gradient(x0_pred),
                cal_spectral_gradient(hyperion_norm)
            ) * t_weight

            # SRF 约束
            x0_denorm = (x0_pred + 1) / 2
            landsat_sim = torch.einsum('ij, bjhw -> bihw',
                                       self.srf_matrix, x0_denorm)
            loss_srf = F.l1_loss(landsat_sim, landsat_band)

            loss = (loss_v + loss_l1
                    + self.lambda_sam * loss_sam
                    + self.lambda_grad * loss_sgrad
                    + self.lambda_srf * loss_srf)

            val_loss += loss.item()
            val_v_loss += loss_v.item()
            val_l1_loss += loss_l1.item()
            val_sam_loss += loss_sam.item()
            val_sgrad_loss += loss_sgrad.item()
            val_srf_loss += loss_srf.item()

        n = len(val_loader)
        val_loss /= n
        val_v_loss /= n
        val_l1_loss /= n
        val_sam_loss /= n
        val_sgrad_loss /= n
        val_srf_loss /= n
        for k in all_metrics:
            all_metrics[k] /= n
        bucket_avg = {}
        for bk in ('low', 'mid', 'high'):
            bucket_avg[bk] = v_bucket[bk] / max(bucket_counts[bk], 1)

        print(f"  [{self.band_name}] Val | Loss: {val_loss:.4f} "
              f"(v={val_v_loss:.4f}, L1={val_l1_loss:.4f}, "
              f"SAM={val_sam_loss:.4f}, Grad={val_sgrad_loss:.4f}, "
              f"SRF={val_srf_loss:.4f})")
        print(f"  [{self.band_name}] v-MSE by t-bucket | "
              f"low(t<{t_low}): {bucket_avg['low']:.6f} | "
              f"mid({t_low}≤t<{t_high}): {bucket_avg['mid']:.6f} | "
              f"high(t≥{t_high}): {bucket_avg['high']:.6f}")
        print(f"  [{self.band_name}] DDIM→ RMSE: {all_metrics['RMSE']:.4f} | "
              f"PSNR: {all_metrics['PSNR']:.2f} | "
              f"SAM: {all_metrics['SAM']:.4f} | SSIM: {all_metrics['SSIM']:.4f} | "
              f"PCC: {all_metrics['PCC']:.4f} | ERGAS: {all_metrics['ERGAS']:.4f}")
        return val_loss, all_metrics

    @torch.no_grad()
    def ddim_sample(self, landsat_band, eta=0.0):
        """DDIM accelerated sampling with v-prediction (Salimans & Ho, 2022)."""
        self.model.eval()
        B, _, H, W = landsat_band.shape
        n_hyper = BAND_HYPERION_COUNTS[self.band_name]
        T = self.diffusion.num_timesteps

        landsat_norm = landsat_band * 2 - 1

        stride = max(T // self.ddim_steps, 1)
        times = list(range(T - 1, -1, -stride))
        if times[-1] != 0:
            times.append(0)

        x_t = torch.randn(B, n_hyper, H, W, device=self.device)

        for i in range(len(times) - 1):
            t_curr = times[i]
            t_prev = times[i + 1]

            t_tensor = torch.full((B,), t_curr, device=self.device, dtype=torch.long)

            v_pred = self.model(x_t, landsat_norm, t_tensor)
            x0_pred = self.diffusion.predict_x0_from_v(x_t, t_tensor, v_pred)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            epsilon_pred = self.diffusion.predict_noise_from_v(x_t, t_tensor, v_pred)

            ab_prev = self.diffusion.alpha_bar[t_prev]
            ab_curr = self.diffusion.alpha_bar[t_curr]

            if eta > 0:
                sigma_t = eta * torch.sqrt(
                    (1 - ab_prev) / (1 - ab_curr).clamp(min=1e-8)
                ) * torch.sqrt(
                    (1 - ab_curr / ab_prev.clamp(min=1e-8)).clamp(min=0)
                )
            else:
                sigma_t = torch.zeros_like(ab_prev)

            direction = torch.sqrt((1 - ab_prev - sigma_t ** 2).clamp(min=0))

            ab_prev = ab_prev.reshape(-1, 1, 1, 1)
            direction = direction.reshape(-1, 1, 1, 1)

            x_t = torch.sqrt(ab_prev) * x0_pred + direction * epsilon_pred
            if eta > 0:
                sigma_t = sigma_t.reshape(-1, 1, 1, 1)
                x_t = x_t + sigma_t * torch.randn_like(x_t)

        return (x_t + 1) / 2

    @torch.no_grad()
    def inference(self, landsat_band):
        """DDIM sampling for final test evaluation."""
        return self.ddim_sample(landsat_band)


# ============================================================
# 3. 完整训练主循环
# ============================================================

def train_all_bands_diffusion_srf(
        num_epochs=300, base_channels=32, lr=1e-4,
        weight_decay=1e-4, patience=40, device='cuda',
        data_dir='./data2026', save_dir='./diffusion_SRF_checkpoints',
        num_timesteps=1000, ddim_steps=50,
        lambda_srf=0.1,
        max_react_prob=0.25, warmup_react_epochs=10,
        aux_weight_floor=0.05, pretrained_dir=None):
    os.makedirs(save_dir, exist_ok=True)

    pt_label = pretrained_dir if pretrained_dir else 'scratch'
    print("Loading per-band dataloaders ...")
    print(f'epoch={num_epochs}, patience={patience}, base_channels={base_channels}, '
          f'T={num_timesteps}(cosine, v-prediction, +SRF={lambda_srf}), '
          f'ddim_steps={ddim_steps}, '
          f'SRT(p={max_react_prob}, warmup={warmup_react_epochs}ep), '
          f'AuxFloor={aux_weight_floor}, Pretrained={pt_label}')
    all_loaders = load_band_dataloaders(data_dir)

    trainers = {}
    best_metrics = {}
    stop_epochs = {}

    for band_name in BAND_ORDER:
        print(f"\n{'='*60}")
        print(f"Training [{band_name}] — {BAND_HYPERION_COUNTS[band_name]} Hyperion bands")
        print(f"{'='*60}")

        pretrained_path = None
        if pretrained_dir is not None:
            pt_path = os.path.join(pretrained_dir, f'{band_name}_best.pth')
            if os.path.isfile(pt_path):
                pretrained_path = pt_path

        trainer = PerBandSRFDiffusionTrainer(
            band_name=band_name,
            base_channels=base_channels,
            num_timesteps=num_timesteps,
            ddim_steps=ddim_steps,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            lambda_srf=lambda_srf,
            max_react_prob=max_react_prob,
            warmup_react_epochs=warmup_react_epochs,
            aux_weight_floor=aux_weight_floor,
            pretrained_path=pretrained_path,
        )
        train_ldr = all_loaders[band_name]['train']
        val_ldr = all_loaders[band_name]['val']
        best_val_rmse = float('inf')
        epochs_no_improve = 0
        # CosineAnnealingWarmRestarts(T_0=20, T_mult=2): restarts at 20, 60, 140, 300
        next_restart = 20
        restart_cycle = 20

        for epoch in range(1, num_epochs + 1):
            train_loss, train_m = trainer.train_epoch(train_ldr, epoch)
            val_loss, val_m = trainer.validate(val_ldr)
            trainer.record_history(epoch, train_loss, train_m, val_loss, val_m)

            if val_m['RMSE'] < best_val_rmse:
                best_val_rmse = val_m['RMSE']
                best_metrics[band_name] = val_m
                epochs_no_improve = 0
                torch.save(trainer.model.state_dict(),
                           os.path.join(save_dir, f'{band_name}_best.pth'))
                print(f"  [{band_name}] Best model saved "
                      f"(DDIM-RMSE={best_val_rmse:.5f}, val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"  [{band_name}] Early stopping at epoch {epoch} "
                          f"(DDIM-RMSE no improvement for {patience} epochs)")
                    stop_epochs[band_name] = epoch
                    break

            trainer.scheduler.step()

            if epoch == next_restart:
                epochs_no_improve = 0
                restart_cycle *= 2
                next_restart += restart_cycle
                print(f"  [{band_name}] LR restarted at epoch {epoch} "
                      f"— patience counter reset (next restart: {next_restart})")

        # 保存训练历史
        history_path = os.path.join(save_dir, f'{band_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)
        print(f"  [{band_name}] Training history saved to {history_path}")

        trainers[band_name] = trainer

    # 绘制训练曲线
    plot_diffusion_srf_curves(trainers, save_dir)

    # ========== 测试集评估 ==========
    print(f"\n{'='*60}")
    print("Computing test metrics (DDIM sampling + SRF model) ...")
    test_metrics = {}
    for band_name in BAND_ORDER:
        trainer = trainers[band_name]
        model_path = os.path.join(save_dir, f'{band_name}_best.pth')
        if os.path.exists(model_path):
            trainer.model.load_state_dict(torch.load(model_path, map_location=device))
        trainer.model.eval()

        test_ldr = all_loaders[band_name]['test']
        total = {'RMSE': 0.0, 'PSNR': 0.0, 'SAM': 0.0, 'SSIM': 0.0,
                 'PCC': 0.0, 'ERGAS': 0.0}
        count = 0
        with torch.no_grad():
            for landsat_band, hyperion_band in test_ldr:
                landsat_band = landsat_band.to(device, dtype=torch.float32)
                hyperion_band = hyperion_band.to(device, dtype=torch.float32)
                pred = trainer.inference(landsat_band)
                m = compute_metrics(pred, hyperion_band)
                for k in total:
                    total[k] += m.get(k, 0.0)
                count += 1
                if count % 10 == 0:
                    print(f"  [{band_name}] Test progress: {count}/{len(test_ldr)}")

        for k in total:
            total[k] = float(total[k] / count)
        test_metrics[band_name] = total
        print(f"  [{band_name}] Test: RMSE={total['RMSE']:.4f}, "
              f"PSNR={total['PSNR']:.2f}, ERGAS={total['ERGAS']:.2f}, "
              f"SAM={total['SAM']:.4f}, SSIM={total['SSIM']:.4f}")

    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # ========== 完整 Hyperion 重建评估 ==========
    full_pkl = os.path.join(data_dir, 'test_loader-Full.pkl')
    if os.path.exists(full_pkl):
        print(f"\nComputing full-reconstruction metrics ...")
        full_loader = joblib.load(full_pkl)
        if isinstance(full_loader, dict):
            loader = full_loader['test']
        else:
            loader = full_loader

        total_m = {'RMSE': 0.0, 'PSNR': 0.0, 'SAM': 0.0, 'SSIM': 0.0,
                   'PCC': 0.0, 'ERGAS': 0.0}
        cnt = 0
        for landsat_full, hyperion_full in loader:
            landsat_full = landsat_full.to(device, dtype=torch.float32)
            hyperion_full = hyperion_full.to(device, dtype=torch.float32)
            outputs = []
            for name in BAND_ORDER:
                t = trainers[name]
                landsat_band = landsat_full[:, t.cond_idx:t.cond_idx + 1, :, :]
                pred = t.inference(landsat_band)
                outputs.append(pred.cpu())
            pred_full = torch.cat(outputs, dim=1).to(device)
            m = compute_metrics(pred_full, hyperion_full)
            for k in total_m:
                total_m[k] += m.get(k, 0.0)
            cnt += 1
            if cnt % 10 == 0:
                print(f"  Full reconstruction progress: {cnt}/{len(loader)}")

        for k in total_m:
            total_m[k] = float(total_m[k] / cnt)
        print(f"  Full reconstruction ({cnt} batches): "
              f"RMSE={total_m['RMSE']:.4f}, PSNR={total_m['PSNR']:.2f}, "
              f"ERGAS={total_m['ERGAS']:.2f}")
        with open(os.path.join(save_dir, 'full_reconstruction_metrics.json'), 'w') as f:
            json.dump(total_m, f, indent=2)

    return trainers, test_metrics, best_metrics


def plot_diffusion_srf_curves(trainers, save_dir):
    """绘制 +SRF 扩散模型的训练/验证曲线。"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(BAND_ORDER)))

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    metric_keys = [('loss', 'Loss'), ('RMSE', 'RMSE'), ('PSNR', 'PSNR (dB)'),
                   ('SAM', 'SAM'), ('SSIM', 'SSIM'), ('PCC', 'PCC'),
                   ('ERGAS', 'ERGAS')]

    for idx, (key, ylabel) in enumerate(metric_keys):
        ax = axes[idx]
        for c, name in zip(colors, BAND_ORDER):
            if name in trainers:
                h = trainers[name].history
                ax.plot(h['epoch'], h[f'val_{key}'],
                        color=c, linewidth=1.2, alpha=0.8, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Validation {ylabel}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 摘要
    ax_summary = axes[8]
    ax_summary.axis('off')
    lines = ["Band — Best val ERGAS"]
    for name in BAND_ORDER:
        if name in trainers:
            best_ergas = min(trainers[name].history['val_ERGAS'])
            lines.append(f"  {name}: {best_ergas:.2f}")
    for i, line in enumerate(lines):
        ax_summary.text(0.1, 0.95 - i * 0.06, line, transform=ax_summary.transAxes,
                        fontsize=10, fontfamily='monospace', verticalalignment='top')

    plt.suptitle('DDPM + SRF — Per-Band Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'diffusion_SRF_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    DATA_DIR = './data2026'
    SAVE_DIR = './diffusion_SRF_checkpoints'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data directory: {DATA_DIR}")

    train_all_bands_diffusion_srf(
        num_epochs=200,
        base_channels=32,
        lr=1e-4,
        weight_decay=1e-4,
        patience=40,
        device=device,
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        num_timesteps=1000,
        ddim_steps=50,
        lambda_srf=0.1,
        max_react_prob=0.0,
        warmup_react_epochs=0,
        aux_weight_floor=0.05,
        pretrained_dir=None,
    )
