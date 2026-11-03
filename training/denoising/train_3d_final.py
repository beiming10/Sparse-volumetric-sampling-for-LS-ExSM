import torch
import os
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import datetime
import numpy as np
from einops import rearrange # 需要安装 einops: pip install einops
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from networks.denoise import UNet_Attention_Hybrid
# 导入带有LeakyReLU的模型和我们刚刚修改好的数据集

from dataset_hsi_3d_final import TrainDataset, ValidDataset
from utils import AverageMeter, save_checkpoint, time2file_name
import tifffile as tiff
# --- 严格复用您成功的2D空间邻居生成函数 ---
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // 2, w // 2)

def generate_mask_triplet(img_batch_2d):
    n, c, h, w = img_batch_2d.shape
    num_blocks = n * (h // 2) * (w // 2)
    total_elements = num_blocks * 4
    mask1, mask2, mask3 = torch.zeros(total_elements, dtype=torch.bool, device=img_batch_2d.device), torch.zeros(total_elements, dtype=torch.bool, device=img_batch_2d.device), torch.zeros(total_elements, dtype=torch.bool, device=img_batch_2d.device)
    idx_pair = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 0, 2], [2, 0, 3], [3, 1, 0], [3, 2, 1]], dtype=torch.int64, device=img_batch_2d.device)
    rand_idx = torch.randint(0, 8, (num_blocks,), device=img_batch_2d.device)
    rd_pair_idx = idx_pair[rand_idx]
    offsets = torch.arange(0, total_elements, 4, dtype=torch.int64, device=img_batch_2d.device).unsqueeze(1)
    rd_pair_idx = rd_pair_idx + offsets
    mask1[rd_pair_idx[:, 0]], mask2[rd_pair_idx[:, 1]], mask3[rd_pair_idx[:, 2]] = True, True, True
    return mask1, mask2, mask3

def generate_subimages_triplet(img_batch_2d, mask):
    n, c, h, w = img_batch_2d.shape
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img_batch_2d.dtype, device=img_batch_2d.device)
    for i in range(c):
        channel = img_batch_2d[:, i:i+1, :, :]
        unfolded = space_to_depth(channel, block_size=2)
        unfolded = unfolded.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i+1, :, :] = unfolded[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

# --- 新的包装函数，忠实地将2D逻辑应用于3D数据 ---
def create_3d_subpatches(clip_3d, mask1, mask2, mask3):
    """将一个3D切片在空间上分解为3个子切片"""
    B, C, D, H, W = clip_3d.shape

    # 1. 将3D切片看作一个批次的2D帧，以便应用2D函数
    frames_2d = rearrange(clip_3d, 'b c d h w -> (b d) c h w')

    # 2. 生成2D子图像
    sub_frames1 = generate_subimages_triplet(frames_2d, mask1)
    sub_frames2 = generate_subimages_triplet(frames_2d, mask2)
    sub_frames3 = generate_subimages_triplet(frames_2d, mask3)

    # 3. 将2D子图像恢复成3D子切片
    sub_clip1 = rearrange(sub_frames1, '(b d) c h w -> b c d h w', b=B)
    sub_clip2 = rearrange(sub_frames2, '(b d) c h w -> b c d h w', b=B)
    sub_clip3 = rearrange(sub_frames3, '(b d) c h w -> b c d h w', b=B)

    return sub_clip1, sub_clip2, sub_clip3


# # --- 即时测试函数 ---
# def run_intermediate_test(model, val_loader, device, clip_len, epoch, output_dir, scale_factor):
#     model.eval()
#     epoch_result_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
#     os.makedirs(epoch_result_dir, exist_ok=True)
#     print(f"\nRunning intermediate test for epoch {epoch}, saving results to {epoch_result_dir}")

#     with torch.no_grad():
#         for i, (noisy_stack_float, dtype_str_tuple) in enumerate(val_loader):
#             noisy_stack_float = noisy_stack_float.squeeze(0).numpy()
#             original_dtype = np.dtype(dtype_str_tuple[0])

#             # 预处理：仅进行缩放，与TrainDataset保持一致
#             noisy_stack_scaled = noisy_stack_float / scale_factor

#             # 滑动窗口推理
#             denoised_stack_scaled = []
#             pad_size = clip_len // 2
#             padded_stack = np.pad(noisy_stack_scaled, ((pad_size, pad_size), (0, 0), (0, 0)), 'reflect')

#             for frame_idx in range(len(noisy_stack_scaled)):
#                 clip_np = padded_stack[frame_idx : frame_idx + clip_len]
#                 input_tensor = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(0).to(device)
#                 output_clip = model(input_tensor)
#                 denoised_frame_scaled = output_clip[:, :, pad_size, :, :].squeeze().cpu().numpy()
#                 denoised_stack_scaled.append(denoised_frame_scaled)
#                 print(f"Processed frame {frame_idx + 1}/{len(noisy_stack_scaled)}")

#             denoised_stack_scaled = np.array(denoised_stack_scaled)

#             # 后处理：仅进行反向缩放
#             denoised_stack_restored = denoised_stack_scaled * scale_factor

#             clip_max = 65535 if original_dtype == np.uint16 else 65535
#             denoised_stack_clipped = np.clip(denoised_stack_restored, 0, clip_max)
#             denoised_stack_final = denoised_stack_clipped.astype(original_dtype)

#             save_name = f"scene_{i:02d}_denoised_epoch{epoch}.tif"
#             output_path = os.path.join(epoch_result_dir, save_name)
#             tiff.imsave(output_path, denoised_stack_final)
#             print(f"Saved denoised result to {output_path}")

# def run_intermediate_test(model, val_loader, device, clip_len, epoch, output_dir, scale_factor, inference_batch_size=4):
#     """
#     在训练过程中执行【优化后的】测试和可视化结果保存。
#     采用批次化推理以大幅提升速度。
#     """
#     model.eval()
#     epoch_result_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
#     os.makedirs(epoch_result_dir, exist_ok=True)
#     print(f"\nRunning intermediate test for epoch {epoch}, saving results to {epoch_result_dir}")

#     with torch.no_grad():
#         for i, (noisy_stack_float, dtype_str_tuple) in enumerate(val_loader):
#             noisy_stack_float = noisy_stack_float.squeeze(0).numpy()
#             original_dtype = np.dtype(dtype_str_tuple[0])

#             # --- 预处理 (保持不变) ---
#             noisy_stack_scaled = noisy_stack_float / scale_factor

#             # --- 优化点 1: 一次性准备所有输入切片 ---
#             all_clips = []
#             pad_size = clip_len // 2
#             padded_stack = np.pad(noisy_stack_scaled, ((pad_size, pad_size), (0, 0), (0, 0)), 'reflect')
#             for frame_idx in range(len(noisy_stack_scaled)):
#                 clip_np = padded_stack[frame_idx : frame_idx + clip_len]
#                 all_clips.append(clip_np)

#             # 将所有切片打包成一个大的Numpy数组，并转换为PyTorch张量
#             all_clips_tensor = torch.from_numpy(np.array(all_clips)).unsqueeze(1) # Shape: [NumFrames, 1, D, H, W]

#             # --- 优化点 2: 创建临时DataLoader进行批次化推理 ---
#             inference_dataset = torch.utils.data.TensorDataset(all_clips_tensor)
#             inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=inference_batch_size)

#             denoised_stack_scaled = []

#             # 使用tqdm显示批次处理进度
#             for batch_clips_tensor, in tqdm(inference_loader, desc=f"Denoising Scene {i}"):
#                 batch_clips_tensor = batch_clips_tensor.to(device)

#                 # 模型现在一次处理一个批次
#                 output_batch_clips = model(batch_clips_tensor)

#                 # 提取每个clip的中心帧
#                 center_frames = output_batch_clips[:, :, pad_size, :, :].squeeze(1).cpu().numpy()
#                 denoised_stack_scaled.append(center_frames)

#             # --- 优化点 3: 收集并拼接所有批次的结果 ---
#             denoised_stack_scaled = np.concatenate(denoised_stack_scaled, axis=0)
#             # log_scale_factor = np.log1p(scale_factor)

#             # # 1. 反向缩放
#             # denoised_stack_log = denoised_stack_scaled * log_scale_factor
#             # # 2. 应用指数变换 exp(p) - 1
#             # denoised_stack_restored = np.expm1(denoised_stack_log)
#             # --- 后处理 (保持不变) ---
#             denoised_stack_restored = denoised_stack_scaled * scale_factor

#             clip_max = 65535 if original_dtype == np.uint16 else 65535
#             denoised_stack_clipped = np.clip(denoised_stack_restored, 0, clip_max)
#             denoised_stack_final = denoised_stack_clipped.astype(original_dtype)

#             # --- 保存结果 ---
#             save_name = f"scene_{i:02d}_denoised_epoch{epoch}.tif"
#             output_path = os.path.join(epoch_result_dir, save_name)
#             tiff.imsave(output_path, denoised_stack_final)
#             print(f"Saved denoised result to {output_path}")
def run_intermediate_test(model, val_loader, device, clip_len, epoch, output_dir, scale_factor, inference_batch_size=4):
    """
    在训练过程中执行【优化后的】测试和可视化结果保存。
    采用批次化推理以大幅提升速度。
    """
    model.eval()
    epoch_result_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_result_dir, exist_ok=True)
    print(f"\nRunning intermediate test for epoch {epoch}, saving results to {epoch_result_dir}")

    with torch.no_grad():
        # 修改这里：接收三个返回值
        for i, (noisy_stack_float, dtype_str_tuple, original_filename_tuple) in enumerate(val_loader):
            noisy_stack_float = noisy_stack_float.squeeze(0).numpy()
            original_dtype = np.dtype(dtype_str_tuple[0])
            # 可以选择使用原始文件名，或者继续使用scene编号
            original_filename = original_filename_tuple[0] if len(original_filename_tuple) > 0 else f"scene_{i:02d}"

            print(f"Processing scene {i}: {original_filename}")

            # 检查并处理数据维度 - 确保变成 [T, H, W] 格式
            if noisy_stack_float.ndim == 4:  # [T, C, H, W]
                if noisy_stack_float.shape[1] == 1:  # C=1，可以压缩
                    noisy_stack_float = noisy_stack_float.squeeze(1)  # 变成 [T, H, W]
                else:
                    raise ValueError(f"Unexpected channel dimension: {noisy_stack_float.shape[1]}")
            elif noisy_stack_float.ndim != 3:
                raise ValueError(f"Unexpected data dimensions: {noisy_stack_float.shape}")

            # --- 预处理 (保持不变) ---
            noisy_stack_scaled = noisy_stack_float / scale_factor

            # --- 优化点 1: 一次性准备所有输入切片 ---
            all_clips = []
            pad_size = clip_len // 2
            padded_stack = np.pad(noisy_stack_scaled, ((pad_size, pad_size), (0, 0), (0, 0)), 'reflect')
            for frame_idx in range(len(noisy_stack_scaled)):
                clip_np = padded_stack[frame_idx : frame_idx + clip_len]
                all_clips.append(clip_np)

            # 将所有切片打包成一个大的Numpy数组，并转换为PyTorch张量
            all_clips_tensor = torch.from_numpy(np.array(all_clips)).unsqueeze(1) # Shape: [NumFrames, 1, D, H, W]

            # --- 优化点 2: 创建临时DataLoader进行批次化推理 ---
            inference_dataset = torch.utils.data.TensorDataset(all_clips_tensor)
            inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=inference_batch_size)

            denoised_stack_scaled = []

            # 使用tqdm显示批次处理进度
            for batch_clips_tensor, in tqdm(inference_loader, desc=f"Denoising Scene {i}"):
                batch_clips_tensor = batch_clips_tensor.to(device)

                # 模型现在一次处理一个批次
                output_batch_clips = model(batch_clips_tensor)

                # 提取每个clip的中心帧
                center_frames = output_batch_clips[:, :, pad_size, :, :].squeeze(1).cpu().numpy()
                denoised_stack_scaled.append(center_frames)

            # --- 优化点 3: 收集并拼接所有批次的结果 ---
            denoised_stack_scaled = np.concatenate(denoised_stack_scaled, axis=0)

            # --- 后处理 (保持不变) ---
            denoised_stack_restored = denoised_stack_scaled * scale_factor

            clip_max = 65535 if original_dtype == np.uint16 else 65535
            denoised_stack_clipped = np.clip(denoised_stack_restored, 0, clip_max)
            denoised_stack_final = denoised_stack_clipped.astype(original_dtype)

            # --- 保存结果 ---
            save_name = f"scene_{i:02d}_denoised_epoch{epoch}.tif"
            output_path = os.path.join(epoch_result_dir, save_name)
            tiff.imsave(output_path, denoised_stack_final)
            print(f"Saved denoised result to {output_path}")
def train_true_n2n(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.outf, time2file_name(str(datetime.datetime.now())))
    os.makedirs(output_dir, exist_ok=True)

    train_data = TrainDataset(data_root=args.data_root, crop_size=args.patch_size, clip_len=args.clip_len, aug=True, scale_factor=args.scale_factor)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_data = ValidDataset(data_root=args.data_root)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 我们的3D模型现在接收的是空间下采样后的输入
    # 注意：模型的输入通道是1，但输出通道也是1。它处理的是子切片。
    model = UNet_Attention_Hybrid(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_fn = nn.MSELoss()  # 混合损失函数
    loss_fn_2 = nn.L1Loss()  # 备用损失函数
    print("Starting training with True Hybrid N2N Strategy...")
    for epoch in range(1, args.end_epoch + 1):
        model.train()
        losses, losses_t, losses_s = AverageMeter(), AverageMeter(), AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.end_epoch}")
        for patch_t, patch_t_plus_1 in pbar:
            patch_t, patch_t_plus_1 = patch_t.to(device), patch_t_plus_1.to(device)
            B, C, D, H, W = patch_t.shape

            # --- 严格遵循您的2D逻辑进行3D扩展 ---
            # 1. 为 t 时刻的切片生成mask
            frames_t_2d = rearrange(patch_t, 'b c d h w -> (b d) c h w')
            mask1, mask2, mask3 = generate_mask_triplet(frames_t_2d)

            # 2. 将t时刻的切片在空间上分解为3个子切片
            sub_t1, sub_t2, sub_t3 = create_3d_subpatches(patch_t, mask1, mask2, mask3)

            # 3. 为 t+1 时刻的切片生成时间监督目标
            frames_t_plus_1_2d = rearrange(patch_t_plus_1, 'b c d h w -> (b d) c h w')
            sub_t_plus_1 = generate_subimages_triplet(frames_t_plus_1_2d, mask1)
            sub_t_plus_1 = rearrange(sub_t_plus_1, '(b d) c h w -> b c d h w', b=B)

            # 4. 前向传播：模型的输入是空间子切片sub_t1
            denoised_output_subpatch = model(sub_t1)

            # 5. 计算空间损失
            loss_spatial = loss_fn(denoised_output_subpatch, sub_t2) + loss_fn(denoised_output_subpatch, sub_t3)

            # 6. 计算时间损失
            loss_temporal = loss_fn(denoised_output_subpatch, sub_t_plus_1)

            # 7. 计算混合总损失
            loss_total = loss_spatial + args.alpha * loss_temporal

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses.update(loss_total.item())
            losses_t.update(loss_temporal.item())
            losses_s.update(loss_spatial.item())
            pbar.set_postfix(loss=f'{losses.avg:.6f}', S=f'{losses_s.avg:.6f}', T=f'{losses_t.avg:.6f}')

        print(f"Epoch {epoch} Summary: Total Loss: {losses.avg:.6f}, Spatial: {losses_s.avg:.6f}, Temporal: {losses_t.avg:.6f}")

        if epoch > 0 and epoch % 30 == 0:
            # 保存检查点并进行中间测试
            save_checkpoint(output_dir, epoch, 0, model, optimizer)
            run_intermediate_test(model, val_loader, device, args.clip_len, epoch, output_dir, args.scale_factor)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train with the True Hybrid N2N Strategy.")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--outf', type=str, default='./outputs/denoise_training/')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--clip_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--end_epoch', type=int, default=600)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--scale_factor', type=float, default=250.0)
    parser.add_argument('--alpha', type=float, default=0.0, help="Weight for the temporal loss component.")

    args = parser.parse_args()
    train_true_n2n(args)
