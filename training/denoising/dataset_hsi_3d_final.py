import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
import random

# 禁用Pillow的图像大小限制，以防TIFF文件过大
Image.MAX_IMAGE_PIXELS = None

class TrainDataset(Dataset):
    """
    训练数据集 (最终版):
    - 对 uint16/uint8 数据进行 [0, 1] 区间缩放。
    - __getitem__ 返回一对时间上相邻的3D视频切片 (patch_t, patch_t_plus_1)，
      用于同时计算时间和空间损失。
    """
    def __init__(self, data_root, crop_size, clip_len=5, aug=True, scale_factor=65535.0):
        super().__init__()
        self.crop_size = crop_size
        self.clip_len = clip_len
        self.aug = aug
        self.scenes = []
        self.scale_factor = scale_factor
        self.log_scale_factor = np.log1p(scale_factor)
        train_path = os.path.join(data_root, 'Train_Hyper/')
        scene_files = sorted(glob.glob(os.path.join(train_path, '*.tif*')))
        print(f"Found {len(scene_files)} scene files in {train_path}")

        for scene_path in scene_files:
            try:
                with Image.open(scene_path) as img_stack:
                    # 确保有足够的帧来构成一对连续的切片
                    if img_stack.n_frames < self.clip_len + 1:
                        continue

                    full_stack_np = np.array([np.array(img_stack.seek(i) or img_stack) for i in range(img_stack.n_frames)], dtype=np.float32)
                    #     # --- 关键修改：应用对数变换 ---
                    # # 1. 应用 log(1+p)
                    # full_stack_np = np.log1p(full_stack_np)
                    # 2. 再进行线性缩放
                    full_stack_np = full_stack_np / self.scale_factor

                    scene_frames = [np.expand_dims(frame, axis=0) for frame in full_stack_np]
                    self.scenes.append(scene_frames)
            except Exception as e:
                print(f"Could not load or process file {scene_path}. Error: {e}")

        # 索引现在指向一对连续切片的起始位置
        self.clip_pairs = []
        for scene_idx, scene in enumerate(self.scenes):
            # 一个有N帧的场景可以构成 N - clip_len 个有效的切片对起始点
            for frame_idx in range(len(scene) - self.clip_len):
                self.clip_pairs.append((scene_idx, frame_idx))

        print(f"Successfully loaded {len(self.scenes)} scenes.")
        print(f"Total number of valid temporal clip pairs for training: {len(self.clip_pairs)}")

    def augment_3d_pair(self, clip1, clip2):
        # 对一对3D切片应用相同的空间随机变换
        rot_k = random.randint(0, 3)
        v_flip = random.random() > 0.5
        h_flip = random.random() > 0.5

        def transform(tensor):
            tensor = torch.rot90(tensor, k=rot_k, dims=[-2, -1])
            if v_flip:
                tensor = torch.flip(tensor, dims=[-2])
            if h_flip:
                tensor = torch.flip(tensor, dims=[-1])
            return tensor

        return transform(clip1), transform(clip2)

    def __getitem__(self, idx):
        scene_idx, start_frame_idx = self.clip_pairs[idx]

        clip_frames_t = self.scenes[scene_idx][start_frame_idx : start_frame_idx + self.clip_len]
        clip_frames_t_plus_1 = self.scenes[scene_idx][start_frame_idx + 1 : start_frame_idx + self.clip_len + 1]

        _, h, w = clip_frames_t[0].shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        patch_clip_t = [frame[:, top:top + self.crop_size, left:left + self.crop_size] for frame in clip_frames_t]
        patch_clip_t_plus_1 = [frame[:, top:top + self.crop_size, left:left + self.crop_size] for frame in clip_frames_t_plus_1]

        clip_tensor_t = torch.from_numpy(np.array(patch_clip_t).transpose(1, 0, 2, 3))
        clip_tensor_t_plus_1 = torch.from_numpy(np.array(patch_clip_t_plus_1).transpose(1, 0, 2, 3))

        if self.aug:
            clip_tensor_t, clip_tensor_t_plus_1 = self.augment_3d_pair(clip_tensor_t, clip_tensor_t_plus_1)

        return clip_tensor_t.contiguous(), clip_tensor_t_plus_1.contiguous()

    def __len__(self):
        return len(self.clip_pairs)


# class ValidDataset(Dataset):
#     """
#     验证数据集 (最终版):
#     - 加载原始数据，不进行缩放。
#     - __getitem__ 返回 (float32格式的数据, 原始数据类型字符串)，以完美兼容DataLoader和后续的测试函数。
#     """
#     def __init__(self, data_root):
#         super().__init__()
#         self.validation_scenes = []
#         self.original_dtypes = []

#         valid_path = os.path.join(data_root, 'Valid_Hyper/')
#         scene_files = sorted(glob.glob(os.path.join(valid_path, '*.tif*')))
#         print(f"\nFound {len(scene_files)} validation scene files in {valid_path}")

#         for scene_path in scene_files:
#             try:
#                 with Image.open(scene_path) as img_stack:
#                     self.original_dtypes.append(np.array(img_stack).dtype)
#                     scene_data = [np.array(img_stack.seek(i) or img_stack) for i in range(img_stack.n_frames)]
#                     self.validation_scenes.append(np.array(scene_data))
#             except Exception as e:
#                 print(f"Could not load or process validation file {scene_path}. Error: {e}")

#         print(f"Total number of validation scenes: {len(self.validation_scenes)}")

#     def __getitem__(self, idx):
#         raw_data = self.validation_scenes[idx]
#         original_dtype_str = str(self.original_dtypes[idx])
#         return raw_data.astype(np.float32), original_dtype_str

#     def __len__(self):
#         return len(self.validation_scenes)

class ValidDataset(Dataset):
    """
    验证数据集 (最终版):
    - 与TrainDataset保持一致的数据存储方式
    - 每个数据项对应一个完整的场景，返回整个场景的所有帧
    """
    def __init__(self, data_root, scale_factor=65535.0, use_subdir=True):
        super().__init__()
        # 懒加载：只保存文件路径列表，避免一次性加载所有文件到内存
        self.scene_paths = []  # 存储文件路径
        self.scene_filenames = []  # 存储对应的文件名
        self.scene_dtypes = []  # 存储原始数据类型
        self.scale_factor = scale_factor

        # use_subdir=True 时优先查找 Valid_Hyper/ 子目录，若不存在则直接在 data_root 下查找
        # use_subdir=False 时直接在 data_root 下查找，忽略子目录
        if use_subdir:
            valid_subpath = os.path.join(data_root, 'Valid_Hyper')
            if os.path.isdir(valid_subpath):
                valid_path = valid_subpath
            else:
                valid_path = data_root
        else:
            valid_path = data_root
        allowed_extensions = ['tif', 'tiff']
        scene_files = []
        for ext in allowed_extensions:
            scene_files.extend(glob.glob(os.path.join(valid_path, f'*.{ext}')))
        scene_files.sort()

        print(f"\nFound {len(scene_files)} validation scene files in {valid_path}")

        # 预读每个文件的原始数据类型（不加载完整数据），用于后续保存时恢复类型
        for scene_path in scene_files:
            original_filename = os.path.basename(scene_path)
            try:
                with Image.open(scene_path) as img_stack:
                    img_stack.seek(0)
                    original_dtype = np.array(img_stack).dtype
                self.scene_paths.append(scene_path)
                self.scene_filenames.append(original_filename)
                self.scene_dtypes.append(original_dtype)
            except Exception as e:
                print(f"Could not load or process file {scene_path}. Error: {e}")

        print(f"Total number of validation scenes: {len(self.scene_paths)}")

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        original_filename = self.scene_filenames[idx]
        original_dtype = self.scene_dtypes[idx]

        # 懒加载：在 __getitem__ 时才加载文件
        scene_frames = []
        with Image.open(scene_path) as img_stack:
            for i in range(img_stack.n_frames):
                img_stack.seek(i)
                frame_np = np.array(img_stack)
                # 转换为float32
                frame_np = frame_np.astype(np.float32)
                # 添加通道维度，与TrainDataset保持一致
                frame_np = np.expand_dims(frame_np, axis=0)  # [H, W] -> [1, H, W]
                scene_frames.append(frame_np)

        # 将帧列表转换为numpy数组: [T, C, H, W]
        scene_stack = np.array(scene_frames)

        # 转换为连续数组
        scene_stack = np.ascontiguousarray(scene_stack, dtype=np.float32)

        return scene_stack.copy(), str(original_dtype), original_filename

    def __len__(self):
        return len(self.scene_paths)
