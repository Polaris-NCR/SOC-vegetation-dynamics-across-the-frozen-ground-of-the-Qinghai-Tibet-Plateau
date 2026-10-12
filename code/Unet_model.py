import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# ============================================================
# 安全 GroupNorm：确保 group 数能整除通道数
# ============================================================
def safe_group_norm(num_channels, num_groups=32):
    num_groups = min(num_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


# ============================================================
# 正弦时间步嵌入（Vaswani et al., NeurIPS 2017 风格）
# ============================================================
def sinusoidal_time_embedding(timesteps, embedding_dim):
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    t = timesteps.float().view(-1)
    device = t.device
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device))
        * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ============================================================
# 修改 1（P0）：通道拼接条件注入（pix2pix 标准）
# ============================================================
# 原问题：FiLM 通过 cond_feat.mean(dim=(2,3)) 全局平均池化，
# 将 Landsat 条件分支的全部空间信息（边缘、纹理、地类边界）
# 压缩为单个全局向量，模型丢失了"像素级空间对应关系"。
#
# 修改方案：采用 pix2pix (Isola et al., CVPR 2017) 标准的
# 通道拼接 —— 将条件特征与主特征沿通道维拼接后经卷积融合，
# 完整保留条件图像的空间结构。
#
# 顶刊依据：
#   - pix2pix: "Image-to-Image Translation with Conditional
#     Adversarial Networks" (Isola et al., CVPR 2017, 引用 25000+)
#   - F3DUN: "Rethinking 3D-CNN in Hyperspectral Image Super-Resolution"
#     (Remote Sensing, 2023) 同样使用 concat + 1×1 conv 跳连接
# ============================================================

class ResBlock(nn.Module):
    """
    残差块，通道拼接条件注入（pix2pix 标准）。

    将条件特征 cond_feat 与主特征 x 在通道维拼接后经卷积融合，
    完整保留条件图像的空间信息，比 SPADE 更简洁通用。

    文献依据：
      - pix2pix (Isola et al., CVPR 2017): 条件图像拼接注入
      - 原始 UNet (Ronneberger et al., MICCAI 2015): 拼接跳连接
      - 遥感超分 F3DUN (Remote Sens. 2023): concat + 1×1 conv
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_channels, dropout=0.0):
        super(ResBlock, self).__init__()

        combined_in = in_channels + cond_channels

        # ---- 归一化与卷积 ----
        self.norm1 = safe_group_norm(combined_in)
        self.norm2 = safe_group_norm(out_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(combined_in, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # ---- 时间嵌入投影 ----
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # ---- 残差跳连接 ----
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, cond_feat):
        """
        x:         [B, in_C, H, W]  主特征
        time_emb:  [B, time_emb_dim]  时间嵌入
        cond_feat: [B, cond_C, H, W]  条件特征（空间分辨率与 x 一致）
        """
        # 通道拼接：主特征 + 条件特征
        h = torch.cat([x, cond_feat], dim=1)            # [B, in_C+cond_C, H, W]

        # 第一卷积块（条件信息经卷积与主特征融合）
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        h = h + self.time_proj(time_emb)[:, :, None, None]

        # 第二卷积块
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


# ============================================================
# 修改 2（P2）：多尺度交叉注意力块
# ============================================================
# 原问题：原模型仅在瓶颈处使用交叉注意力。对于需要精细
# 空间对齐的光谱超分任务，单尺度注意力不足以恢复 Landsat
# → Hyperion 的逐像素光谱映射关系。
#
# 修改方案：瓶颈处 + 解码器每一级均添加交叉注意力，
# 构成"瓶颈 + 解码器全尺度"的多级空间条件注入。
#
# 顶刊依据：
#   - Multi-scale attention 在多篇图像翻译工作中被验证
#     有效（如 Wang et al., ECCV 2018; Chen et al.,
#     CVPR 2021 "Pre-Trained Image Processing Transformer"）
#   - 遥感超分中空间对齐是关键，多尺度注意力直接强化
#     条件特征与重建特征的逐区域对齐能力
# ============================================================

class CrossAttnBlock(nn.Module):
    """交叉注意力块：主特征 (query) 关注条件特征 (key/value)。"""
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttnBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_x = safe_group_norm(embed_dim)
        self.norm_cond = safe_group_norm(embed_dim)

    def forward(self, x, cond):
        """
        x:    [B, C, H, W]  主特征 (query)
        cond: [B, C, Hc, Wc] 条件特征 (key/value)
        """
        B, C, H, W = x.shape
        x_norm = self.norm_x(x)
        cond_norm = self.norm_cond(cond)

        # 展平空间维度 → [B, N, C]
        x_seq = x_norm.flatten(2).transpose(1, 2)
        cond_seq = cond_norm.flatten(2).transpose(1, 2)

        # 交叉注意力：query = x, key/value = cond
        attn_out, _ = self.attn(x_seq, cond_seq, cond_seq)

        # 恢复空间形状
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return x + attn_out


# ============================================================
# 条件 UNet 主模型
# ============================================================
# 整合以上所有修改，保留与现有训练代码的接口兼容性：
#   forward(x, cond, t) → output
# ============================================================

class ConditionalUNet(nn.Module):

    def __init__(self, in_channels, cond_channels, out_channels,
                 base_channels=64, channel_mult=(1, 2, 4, 8), num_heads=4):
        """
        in_channels:   输入噪声高光谱波段数
        cond_channels: Landsat 条件波段数
        out_channels:  输出重建光谱波段数
        """
        super(ConditionalUNet, self).__init__()

        # ---- 时间嵌入 ----
        self.time_emb_dim = base_channels * 4
        self.time_emb_input_dim = base_channels * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_input_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # ---- 初始投影 ----
        self.conv_in = nn.Conv2d(in_channels, base_channels * channel_mult[0],
                                 kernel_size=3, padding=1)
        self.cond_conv_in = nn.Conv2d(cond_channels, base_channels * channel_mult[0],
                                      kernel_size=3, padding=1)
        self.cond_norm_in = safe_group_norm(base_channels * channel_mult[0])

        num_down = len(channel_mult) - 1  # 下采样次数

        # ---- 编码器（主路径 ResBlock + 下采样） ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(num_down):
            in_ch = base_channels * channel_mult[i]
            out_ch = base_channels * channel_mult[i]
            cond_ch = base_channels * channel_mult[i]
            self.down_blocks.append(ResBlock(in_ch, out_ch, self.time_emb_dim, cond_ch))
            next_ch = base_channels * channel_mult[i + 1]
            self.downsamples.append(nn.Conv2d(out_ch, next_ch, kernel_size=3, stride=2, padding=1))

        # ---- 条件编码器（独立下采样路径，与主路径分辨率同步） ----
        self.cond_downs = nn.ModuleList()
        self.cond_norms = nn.ModuleList()
        for i in range(num_down):
            in_ch = base_channels * channel_mult[i]
            next_ch = base_channels * channel_mult[i + 1]
            self.cond_downs.append(nn.Conv2d(in_ch, next_ch, kernel_size=3, stride=2, padding=1))
            self.cond_norms.append(safe_group_norm(next_ch))

        # ---- 瓶颈（P2：第 1 级交叉注意力） ----
        mid_ch = base_channels * channel_mult[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, self.time_emb_dim, cond_channels=mid_ch)
        self.mid_attn = CrossAttnBlock(mid_ch, num_heads=num_heads)   # 瓶颈交叉注意力
        self.mid_block2 = ResBlock(mid_ch, mid_ch, self.time_emb_dim, cond_channels=mid_ch)

        # ---- 解码器 ----
        # 修改 3（P3）：跳连接由加法改为通道拼接
        # 原问题：h = h + skip_feat（加法）强制逐元素混合，
        # 网络无法选择性使用上采样信息 vs 跳连接信息。
        # 原始 UNet (Ronneberger et al., MICCAI 2015) 使用
        # 拼接而非加法，已成为标准范式。
        # 后果：ResBlock 输入通道数从 out_ch 变为 out_ch*2。
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()    # P2: 解码器级交叉注意力

        for i in reversed(range(num_down)):
            in_ch = base_channels * channel_mult[i + 1]   # 低分辨率特征通道
            out_ch = base_channels * channel_mult[i]       # 目标分辨率特征通道

            # 上采样 + 卷积（DDPM+ / Stable Diffusion 标准，无棋盘伪影）
            self.up_convs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            ))
            # P3: 拼接跳连接后通道数 = out_ch (上采样) + out_ch (跳连接) = out_ch*2
            self.up_blocks.append(
                ResBlock(out_ch * 2, out_ch, self.time_emb_dim, cond_channels=out_ch)
            )
            # P2: 瓶颈 + 解码器中低分辨率级交叉注意力
            # 全分辨率 (i=0, H×H) 跳过：4096² tokens 导致 8+ GB 显存爆炸，
            # 该层级通道拼接条件注入已提供逐像素条件信息
            if i == 0:
                self.up_attns.append(nn.Identity())
            else:
                self.up_attns.append(CrossAttnBlock(out_ch, num_heads=num_heads))

        # ---- 输出投影 ----
        self.conv_out = nn.Conv2d(base_channels * channel_mult[0], out_channels,
                                  kernel_size=3, padding=1)

    def forward(self, x, cond, t):
        """
        x:    噪声高光谱图像 [B, in_channels, H, W]
        cond: Landsat 条件   [B, cond_channels, H, W]
        t:    扩散时间步     [B]
        返回: 预测噪声       [B, out_channels, H, W]
        """
        # 1. 时间嵌入
        t_emb = sinusoidal_time_embedding(t, self.time_emb_input_dim)
        t_emb = self.time_mlp(t_emb)

        # 2. 条件路径编码（多尺度条件特征金字塔）
        cond_feat = self.cond_conv_in(cond)
        cond_feat = self.cond_norm_in(cond_feat)
        cond_feat = F.silu(cond_feat)
        cond_feats = [cond_feat]                        # 存各尺度条件特征
        for conv, norm in zip(self.cond_downs, self.cond_norms):
            cond_feat = conv(cond_feat)
            cond_feat = norm(cond_feat)
            cond_feat = F.silu(cond_feat)
            cond_feats.append(cond_feat)

        # 3. 主路径编码（通道拼接条件注入）
        h = self.conv_in(x)
        hs = []                                         # 存跳连接特征
        for i, resblock in enumerate(self.down_blocks):
            h = resblock(h, t_emb, cond_feats[i])       # 通道拼接条件注入
            hs.append(h)
            h = self.downsamples[i](h)

        # 4. 瓶颈（第 1 级交叉注意力）
        h = self.mid_block1(h, t_emb, cond_feats[-1])
        h = self.mid_attn(h, cond_feats[-1])            # 瓶颈交叉注意力
        h = self.mid_block2(h, t_emb, cond_feats[-1])

        # 5. 解码器（P2 逐级交叉注意力 + P3 拼接跳连接）
        for j, (up_conv, resblock, attn) in enumerate(
                zip(self.up_convs, self.up_blocks, self.up_attns)):
            h = up_conv(h)                              # 上采样
            skip_feat = hs[-(j + 1)]                    # 对应跳连接
            cond_feat = cond_feats[-(j + 2)]            # 对应尺度条件特征

            # P3: 通道拼接替代加法
            h = torch.cat([h, skip_feat], dim=1)
            # ResBlock 输入通道 = out_ch*2，输出通道 = out_ch
            h = resblock(h, t_emb, cond_feat)

            # P2: 解码器交叉注意力（全分辨率级 skip，其余级生效）
            if not isinstance(attn, nn.Identity):
                h = attn(h, cond_feat)

        # 6. 输出
        return self.conv_out(h)
