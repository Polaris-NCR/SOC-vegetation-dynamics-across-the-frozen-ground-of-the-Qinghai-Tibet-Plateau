import math
import numpy as np
import torch
import torch.nn as nn


class Diffusion:
    """
    Diffusion process utilities: defines the forward diffusion schedule and provides
    sampling (reverse diffusion) steps.  Supports both linear and cosine schedules.

    Cosine schedule (Nichol & Dhariwal, 2021) keeps ᾱ_T ≈ 0.01 (vs 4e-5 linear),
    reducing x0 prediction error amplification from ~150× to ~10× at high t.
    """

    def __init__(self, num_timesteps=200, beta_start=1e-4, beta_end=0.02, band='Red',
                 device=None, use_srf=False, schedule='cosine'):
        self.use_srf = use_srf
        self.device = device
        self.num_timesteps = num_timesteps

        if schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar).to(device)

        # SRF projection matrix (only when use_srf=True)
        if use_srf:
            SRF_Response = self._get_srf_response(band)
            SRF_weight = SRF_Response / np.sum(np.array(SRF_Response))
            srf_matrix = torch.tensor(SRF_weight, dtype=torch.float32)
            srf_matrix = srf_matrix.view(1, len(SRF_weight))
            srf_pinv = torch.pinverse(srf_matrix)
            self.srf_matrix = srf_matrix.to(device)
            self.srf_pinv = srf_pinv.to(device)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Cosine noise schedule (Nichol & Dhariwal, 2021).

        Returns betas such that ᾱ_T ≈ 0.006 (cf. linear T=1000: ᾱ_T ≈ 4e-5).
        Clips beta ≤ 0.999 so ᾱ_T never reaches exactly 0.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 1e-5, 0.999)

    @staticmethod
    def _get_srf_response(band):
        if band == 'Red':
            return [0.01100236, 0.9491205, 0.957126455, 0.973884545, 0.94790654, 0.0004666]
        if band == 'NIR':
            return [0.00316, 0.9655386, 0.9552052, 0.95221917, 0.0248636]
        if band == 'Blue':
            return [0.91057168, 0.88934788, 0.90562965, 0.9732924, 0.957390009, 0.99726844, 0.00137877]
        if band == 'Green':
            return [0.058038, 0.95790148, 0.989992, 0.95971264, 0.97255289, 0.9814449, 0.5859786, 0.0001468]
        if band == 'SWIR1':
            return [0.16608812, 0.59041576, 0.9021962, 0.92408152, 0.94512671, 0.954958518, 0.961247391,
                    0.98135547, 0.98036232, 0.6423191, 0.15054]
        if band == 'SWIR2':
            return [0.6909688, 0.88400644, 0.93326308, 0.94967666, 0.94731958, 0.95028218, 0.94007347,
                    0.93499313, 0.93471557, 0.95581502, 0.96255784, 0.96021581, 0.954052, 0.9830669,
                    0.99679732, 0.98245216, 0.96430706, 0.88903696, 0.47219605]
        return []

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion (q): sample x_t from x_0 by adding Gaussian noise at step t.
        :param x0: Original image (Hyperion) [N, C, H, W]
        :param t: Diffusion timestep (Tensor of shape [N]) with values in [0, T-1]
        :param noise: Optional noise tensor to use (otherwise sampled from standard normal).
        :return: Noisy image x_t at time t.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # t = t.to(x0.device)
        t = t.to(self.device)

        # Gather alpha_bar[t] for each batch element
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)  # sqrt(alpha_bar_t)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)  # sqrt(1 - alpha_bar_t)
        # Compute x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        return sqrt_ab * x0 + sqrt_om_ab * noise

    def p_sample_step(self, x_t, t, cond, model):
        """
        One reverse diffusion step p(x_{t-1} | x_t, cond) with spectral projection correction.
        :param x_t: Current noised image at time t [N, C, H, W].
        :param t: Current timestep (int).
        :param cond: Conditioning Landsat image [N, cond_channels, H, W].
        :param model: The conditional UNet model to predict noise.
        :return: x_{t-1} sample after one reverse diffusion step (with spectral correction).
        """
        # Prepare model input timestep (as tensor of shape [N] on same device)
        N = x_t.shape[0]

        # 修复：训练时 t ∈ [0, T-1]（0-indexed），推理时 t ∈ [1, T]（1-indexed）。
        # array 索引用 t-1（0-indexed）正确，但传入模型必须转换为 0-indexed（t-1），
        # 否则正弦时间嵌入会接收到训练中从未见过的值（如 t=1000），导致噪声预测偏移。
        t_model = (t - 1).to(x_t.device).long()  # 转换为 0-indexed 后传入模型
        epsilon_pred = model(x_t, cond, t_model)

        # # Compute predicted x0 from epsilon_pred (reverse of forward process):
        # x0_pred = (x_t - torch.sqrt(1 - self.alpha_bar[t].to(x_t.device)) * epsilon_pred) / torch.sqrt(
        #     self.alpha_bar[t].to(x_t.device))

        # DDPM reverse mean: μ = (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * ε_pred)
        # t is 1-indexed (from sample loop: range(num_timesteps, 0, -1))
        beta_t = self.betas[t - 1].to(x_t.device)
        beta_t = beta_t.reshape(-1, 1, 1, 1).expand(-1, x_t.shape[1], x_t.shape[2], x_t.shape[3])

        alpha_t = self.alphas[t - 1].to(x_t.device)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t = sqrt_alpha_t.reshape(-1, 1, 1, 1).expand(-1, x_t.shape[1], x_t.shape[2], x_t.shape[3])

        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t - 1].to(x_t.device)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.reshape(-1, 1, 1, 1).expand(-1, x_t.shape[1], x_t.shape[2],
                                                                                            x_t.shape[3])

        mu = (1.0 / sqrt_alpha_t) * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * epsilon_pred)

        # Stochastic sampling: add noise for t > 1, deterministic for t = 1
        add_noise_mask = (t > 1).float().reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        x_prev = mu + add_noise_mask * sigma_t * noise
        if self.use_srf:
            landsat_sim = torch.einsum('ij, bjhw -> bihw', self.srf_matrix, x_prev)
            diff = cond - landsat_sim
            hyperion_correction = torch.einsum('ij, bjhw -> bihw', self.srf_pinv, diff)
            x_prev = x_prev + hyperion_correction
        return x_prev

    def sample(self, x_t, model, cond):
        """
        Perform reverse diffusion process to generate a sample from random noise.
        :param model: The model used to predict noise at each step.
        :param cond: Conditioning image (e.g., Landsat data). B,C,H,W
        :param num_samples: The number of samples to generate.
        :return: The generated image after reverse diffusion.
        """
        # Start with random noise
        # x_t = torch.randn(cond.size(0), cond.size(1), cond.size(2), cond.size(3)).to(self.device)
        # t = torch.randint(self.num_timesteps - 1, self.num_timesteps, (cond.size(0),), device=self.device)
        for t in range(self.num_timesteps, 0, -1):
            # t_tensor = torch.randint(0, t, (cond.size(0),), device=self.device).long()
            # 反向传播的过程应该是从T、T-1...一直到1的一个过程
            # 上面的代码是随机从[0, t]之间抽取一个随机数字，而不是按顺序进行反向传播，这样会导致推理过程中逻辑出现混乱
            # 因此应该将其修正为下面格式
            t_tensor = torch.full((cond.size(0),), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample_step(x_t, t_tensor, cond, model)

        # for step in reversed(range(self.num_timesteps)):
        #     # t_tensor = torch.full((cond.size(0),), step, dtype=torch.long, device=self.device)
        #     x_t, _ = self.p_sample_step(x_t, t, cond, model)

        return x_t  # Final generated image

    def predict_x0_from_noise(self, x_t, t, noise_pred):
        """
        根据当前噪声图 x_t 和预测的噪声 noise_pred，反解出原始图像 x0 的估计值。
        根据 DDPM 公式从 x_t 和预测噪声中反解估计的 x0
        公式: x0_hat = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

        修复：对 sqrt(alpha_bar_t) 添加 clamp(min=1e-6)，防止 t 接近 T 时
        分母趋近于 0 导致的数值爆炸（sqrt(ᾱ_T) ≈ 0.01 → 除法放大 100× 误差）。
        """
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)

        # x0_hat = (x_t - sqrt(1-ᾱ_t)*noise) / sqrt(ᾱ_t)
        # 使用 clamp 防止除以接近 0 的值
        return (x_t - sqrt_om_ab * noise_pred) / sqrt_ab.clamp(min=1e-6)

    # ============================================================
    # v-prediction (Salimans & Ho, 2022)
    # ============================================================
    # v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x0
    #
    # 训练目标平滑过渡：
    #   高 t (低 SNR): v ≈ -x0  —— 目标为干净图像结构，训练信号强
    #   低 t (高 SNR): v ≈ noise —— 目标为标准高斯噪声，约束统计特性
    #
    # 推理时从 v 恢复 x0 和 ε 均无误差放大（系数 ≤ 1 处处成立），
    # 解决了 ε-prediction 在低 SNR 区间除以 sqrt(alpha_bar) 导致的数值爆炸。
    # ============================================================

    def v_target(self, x0, noise, t):
        """
        计算 v-prediction 训练目标。
        v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x0

        Args:
            x0:   干净图像 [B, C, H, W]
            noise: 高斯噪声 [B, C, H, W]
            t:     时间步 [B]
        Returns:
            v:     v-prediction 目标 [B, C, H, W]
        """
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return sqrt_ab * noise - sqrt_om_ab * x0

    def predict_x0_from_v(self, x_t, t, v_pred):
        """
        从 v-prediction 恢复 x0 估计值。
        x0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v

        误差放大分析：v_pred 误差乘 sqrt(1-alpha_bar_t) ≤ 1.0，无放大。
        """
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return sqrt_ab * x_t - sqrt_om_ab * v_pred

    def predict_noise_from_v(self, x_t, t, v_pred):
        """
        从 v-prediction 恢复噪声估计值（DDIM 方向项需要）。
        epsilon = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v

        误差放大分析：v_pred 误差乘 sqrt(alpha_bar_t) ≤ 1.0，无放大。
        """
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return sqrt_om_ab * x_t + sqrt_ab * v_pred