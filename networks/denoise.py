import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as einops
from einops import rearrange
# --- 新的核心构建模块：(2+1)D 卷积 ---
class FactorizedConv3D(nn.Module):
    """
    将一个3D卷积分解为 2D空间卷积 + 1D时间卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 空间卷积：只在H, W维度上进行
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])
        self.spatial_conv = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=spatial_kernel_size,
                                      stride=spatial_stride,
                                      padding=spatial_padding,
                                      bias=False)

        # 时间卷积：只在D维度上进行
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        self.temporal_conv = nn.Conv3d(out_channels, out_channels,
                                       kernel_size=temporal_kernel_size,
                                       stride=temporal_stride,
                                       padding=temporal_padding,
                                       bias=False)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x

class FactorizedDoubleConv3D(nn.Module):
    """使用 (2+1)D 卷积的 DoubleConv 模块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # 第一个分解卷积
            FactorizedConv3D(in_channels, mid_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            # 第二个分解卷积
            FactorizedConv3D(mid_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# --- U-Net的其他部分（Down, Up, OutConv）可以复用我们之前的设计 ---
# --- 为清晰起见，这里将它们与新的DoubleConv组合起来 ---

class DownSpatial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            FactorizedDoubleConv3D(in_channels, out_channels) # <--- 使用新的高效卷积块
        )
    def forward(self, x): return self.maxpool_conv(x)

class UpSpatial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 上采样部分保持不变，因为其计算量相对较小
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = FactorizedDoubleConv3D(in_channels, out_channels) # <--- 使用新的高效卷积块
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD, diffH, diffW = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3), x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)


# class UNet3D_Factorized(nn.Module):
#     """
#     最终版的轻量化、高性能3D U-Net。
#     核心是采用了(2+1)D卷积来替代标准3D卷积。
#     注意：为了与之前的模型兼容，我们这里只进行空间下采样。
#     """
#     def __init__(self, in_channels, out_channels):
#         super(UNet3D_Factorized, self).__init__()
#         f_maps = [32, 64, 128, 256]
#         self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
#         self.down1 = DownSpatial(f_maps[0], f_maps[1])
#         self.down2 = DownSpatial(f_maps[1], f_maps[2])
#         self.down3 = DownSpatial(f_maps[2], f_maps[3])
#         self.up1 = UpSpatial(f_maps[3], f_maps[2])
#         self.up2 = UpSpatial(f_maps[2], f_maps[1])
#         self.up3 = UpSpatial(f_maps[1], f_maps[0])
#         self.outc = OutConv3D(f_maps[0], out_channels)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         return self.outc(x)


# class UNet3D_Factorized_Lite(nn.Module):
#     """
#     轻量化的最终版3D U-Net。
#     通过减少特征图通道数，使其参数量与 SRDTrans 相当。
#     """
#     def __init__(self, in_channels, out_channels):
#         super(UNet3D_Factorized_Lite, self).__init__()

#         # --- 关键修改：使用与SRDTrans相同的特征图通道数 ---
#         f_maps = [8, 16, 32, 64]

#         self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
#         self.down1 = DownSpatial(f_maps[0], f_maps[1])
#         self.down2 = DownSpatial(f_maps[1], f_maps[2])
#         self.down3 = DownSpatial(f_maps[2], f_maps[3])
#         self.up1 = UpSpatial(f_maps[3], f_maps[2])
#         self.up2 = UpSpatial(f_maps[2], f_maps[1])
#         self.up3 = UpSpatial(f_maps[1], f_maps[0])
#         self.outc = OutConv3D(f_maps[0], out_channels)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         return self.outc(x)


# # --- 新的核心模块：轻量化的注意力瓶颈 ---
# class AttentionBottleneck(nn.Module):
#     def __init__(self, dim, num_heads=4, mlp_ratio=2.):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )

#     def forward(self, x):
#         B, C, D, H, W = x.shape
#         # 将3D特征图展平为序列
#         x_rearranged = rearrange(x, 'b c d h w -> b (d h w) c')

#         # 自注意力
#         attn_input = self.norm(x_rearranged)
#         attn_output, _ = self.attn(attn_input, attn_input, attn_input)
#         x_rearranged = x_rearranged + attn_output # 残差连接

#         # MLP / FeedForward
#         mlp_output = self.mlp(x_rearranged)
#         x_rearranged = x_rearranged + mlp_output # 残差连接

#         # 恢复形状
#         return rearrange(x_rearranged, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

# # --- 最终的混合模型 ---
# class UNet_Attention_Hybrid(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet_Attention_Hybrid, self).__init__()
#         f_maps = [8, 16, 32, 64] # 使用更轻量的通道数

#         self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
#         self.down1 = DownSpatial(f_maps[0], f_maps[1])
#         self.down2 = DownSpatial(f_maps[1], f_maps[2])
#         self.down3 = DownSpatial(f_maps[2], f_maps[3])

#         # --- 使用注意力模块替换最深处的卷积 ---
#         self.bottleneck = AttentionBottleneck(dim=f_maps[3])

#         self.up1 = UpSpatial(f_maps[3], f_maps[2])
#         self.up2 = UpSpatial(f_maps[2], f_maps[1])
#         self.up3 = UpSpatial(f_maps[1], f_maps[0])
#         self.outc = OutConv3D(f_maps[0], out_channels)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)

#         x_bottle = self.bottleneck(x4) # 在瓶颈处应用全局注意力

#         x = self.up1(x_bottle, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         return self.outc(x)


# # --- 新的核心模块：高效的窗口化注意力瓶颈 ---
# class WindowedAttentionBottleneck(nn.Module):
#     def __init__(self, dim, window_size=(2, 8, 8), num_heads=4, mlp_ratio=2.):
#         """
#         - dim: 输入通道数
#         - window_size: (D_win, H_win, W_win) 3D窗口的大小
#         """
#         super().__init__()
#         self.window_size = window_size
#         self.norm = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )

#     def forward(self, x):
#         B, C, D, H, W = x.shape
#         D_win, H_win, W_win = self.window_size

#         # 1. 将特征图重排为不重叠的窗口
#         # (B, C, D, H, W) -> (B, C, D/D_w, D_w, H/H_w, H_w, W/W_w, W_w) -> (B * num_windows, D_w*H_w*W_w, C)
#         x_windows = rearrange(x, 'b c (d d_w) (h h_w) (w w_w) -> (b d h w) (d_w h_w w_w) c',
#                               d_w=D_win, h_w=H_win, w_w=W_win)

#         # 2. 在每个窗口内部独立进行自注意力计算
#         attn_input = self.norm(x_windows)
#         attn_output, _ = self.attn(attn_input, attn_input, attn_input)
#         x_windows = x_windows + attn_output

#         mlp_output = self.mlp(x_windows)
#         x_windows = x_windows + mlp_output

#         # 3. 将窗口恢复到原始的特征图形状
#         # (B * num_windows, D_w*H_w*W_w, C) -> (B, C, D, H, W)
#         x_restored = rearrange(x_windows, '(b d h w) (d_w h_w w_w) c -> b c (d d_w) (h h_w) (w w_w)',
#                                b=B, d=D//D_win, h=H//H_win, w=W//W_win,
#                                d_w=D_win, h_w=H_win, w_w=W_win)

#         return x_restored

# # --- 最终的混合模型，使用新的注意力瓶颈 ---
# class UNet_Attention_Hybrid(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet_Attention_Hybrid, self).__init__()
#         f_maps = [8, 16, 32, 64]

#         self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
#         self.down1 = DownSpatial(f_maps[0], f_maps[1])
#         self.down2 = DownSpatial(f_maps[1], f_maps[2])
#         self.down3 = DownSpatial(f_maps[2], f_maps[3])

#         # --- 使用新的、高效的窗口化注意力瓶颈 ---
#         # 注意：这里的window_size需要能被瓶颈处的D,H,W整除
#         # 假设输入clip_len=5, patch_size=128，瓶颈处D=5, H=16, W=16
#         # 我们可以设置window_size为(5, 8, 8)
#         self.bottleneck = WindowedAttentionBottleneck(dim=f_maps[3], window_size=(5, 8, 8))

#         self.up1 = UpSpatial(f_maps[3], f_maps[2])
#         self.up2 = UpSpatial(f_maps[2], f_maps[1])
#         self.up3 = UpSpatial(f_maps[1], f_maps[0])
#         self.outc = OutConv3D(f_maps[0], out_channels)

#     def forward(self, x):
#         # 确保输入尺寸可以被窗口大小整除
#         # 为简单起见，这里假设输入尺寸是合适的，实际应用中可能需要填充
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x_bottle = self.bottleneck(x4)
#         x = self.up1(x_bottle, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         return self.outc(x)


# --- Part 2: 修正后的 Swin风格注意力瓶颈 ---

class SwinStyleAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size=(0, 0, 0)):
        super().__init__()
        self.dim, self.window_size, self.shift_size, self.num_heads = dim, window_size, shift_size, num_heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(dim * 2.0)), nn.GELU(), nn.Linear(int(dim * 2.0), dim))

    def forward(self, x):
        B, C, D, H, W = x.shape
        D_win, H_win, W_win = self.window_size

        shortcut = x
        x = rearrange(x, 'b c d h w -> b d h w c')

        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        x_windows = rearrange(shifted_x, 'b (d d_w) (h h_w) (w w_w) c -> (b d h w) (d_w h_w w_w) c',
                              d_w=D_win, h_w=H_win, w_w=W_win)

        attn_windows = self.attn(self.norm(x_windows), self.norm(x_windows), self.norm(x_windows))[0]

        # --- 关键修改：为 rearrange 补全所有必需的尺寸参数 ---
        shifted_x = rearrange(attn_windows, '(b d h w) (d_w h_w w_w) c -> b (d d_w) (h h_w) (w w_w) c',
                              b=B, d=D//D_win, h=H//H_win, w=W//W_win,
                              d_w=D_win, h_w=H_win, w_w=W_win) # <-- 补上这三个参数

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        x = rearrange(x, 'b d h w c -> b c d h w')

        x = shortcut + x
        x = x + self.mlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

class SwinStyleBottleneck(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.block1 = SwinStyleAttention(dim, window_size, num_heads, shift_size=(0,0,0))
        shift_size = (window_size[0]//2, window_size[1]//2, window_size[2]//2)
        self.block2 = SwinStyleAttention(dim, window_size, num_heads, shift_size=shift_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# --- Part 3: 最终组装的、更健壮的混合U-Net模型 ---
#
# class UNet_Attention_Hybrid(nn.Module):
#     def __init__(self, in_channels, out_channels, clip_len=5, patch_size=128):
#         super(UNet_Attention_Hybrid, self).__init__()
#         f_maps = [16, 32, 64, 128]

#         self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
#         self.down1 = DownSpatial(f_maps[0], f_maps[1])
#         self.down2 = DownSpatial(f_maps[1], f_maps[2])
#         self.down3 = DownSpatial(f_maps[2], f_maps[3])

#         # 瓶颈处的特征图尺寸
#         bottleneck_d = clip_len
#         bottleneck_h = patch_size // 8 # 经过3次空间下采样
#         bottleneck_w = patch_size // 8

#         # --- 增加断言以保证尺寸兼容性 ---
#         # 暂时只在空间上做窗口化，以适应任意时间深度
#         win_d, win_h, win_w = 1, 4, 4
#         assert bottleneck_h % win_h == 0, f"Bottleneck height ({bottleneck_h}) must be divisible by window height ({win_h})"
#         assert bottleneck_w % win_w == 0, f"Bottleneck width ({bottleneck_w}) must be divisible by window width ({win_w})"
#         assert bottleneck_d % win_d == 0, f"Bottleneck depth ({bottleneck_d}) must be divisible by window depth ({win_d})"

#         self.bottleneck = SwinStyleBottleneck(
#             dim=f_maps[3],
#             window_size=(win_d, win_h, win_w),
#             num_heads=4
#         )

#         self.up1 = UpSpatial(f_maps[3], f_maps[2])
#         self.up2 = UpSpatial(f_maps[2], f_maps[1])
#         self.up3 = UpSpatial(f_maps[1], f_maps[0])
#         self.outc = OutConv3D(f_maps[0], out_channels)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x_bottle = self.bottleneck(x4)
#         x = self.up1(x_bottle, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         return self.outc(x)
class UNet_Attention_Hybrid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Attention_Hybrid, self).__init__()
        f_maps = [8, 16, 32, 64]

        self.inc = FactorizedDoubleConv3D(in_channels, f_maps[0])
        self.down1 = DownSpatial(f_maps[0], f_maps[1])
        self.down2 = DownSpatial(f_maps[1], f_maps[2])
        self.down3 = DownSpatial(f_maps[2], f_maps[3])

        # 临时修复：使用简单卷积替代注意力瓶颈
        self.bottleneck = FactorizedDoubleConv3D(f_maps[3], f_maps[3])

        self.up1 = UpSpatial(f_maps[3], f_maps[2])
        self.up2 = UpSpatial(f_maps[2], f_maps[1])
        self.up3 = UpSpatial(f_maps[1], f_maps[0])
        self.outc = OutConv3D(f_maps[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_bottle = self.bottleneck(x4)
        x = self.up1(x_bottle, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
