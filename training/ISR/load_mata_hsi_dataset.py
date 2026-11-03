from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
import h5py
import random
from scipy.io import loadmat
try:
    from tifffile import imread
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False
    print("Warning: tifffile not available. TIFF support disabled.")

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, aug = True, h= 420, w =420, stride=8,
                 use_tiff=False, hyper_frames=4, less_frames=2, step=4, scalefactor=65535.0, background_value=0.0):
        """
        Args:
            data_root: 数据根目录
            crop_size: 裁剪尺寸
            aug: 是否数据增强
            h, w: 图像尺寸
            stride: 裁剪步长
            use_tiff: 是否使用TIFF堆栈模式
            hyper_frames: TIFF模式下hyper数据的帧数
            less_frames: TIFF模式下less数据的帧数
            step: TIFF模式下滑动窗口步长
            scalefactor: 归一化比例因子 (默认65535.0用于16位数据)
            background_value: 背景值，用于数据预处理 (默认0.0)
        """
        self.crop_size = crop_size
        self.hypers = [] # hyperspectral(8 channels)
        self.lcs = [] # less channels spectrum(3 channels)
        self.aug = aug
        self.h = h
        self.w = w
        self.stride = stride
        self.use_tiff = use_tiff
        self.hyper_frames = hyper_frames
        self.less_frames = less_frames
        self.step = step
        self.scalefactor = scalefactor
        self.background_value = background_value
        self.patch_per_line = (w-crop_size)//stride+1 # (w - crop_size) means the number of pixels that can be cropped from the width of the image
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.tiff_stacks = []  # TIFF模式：原始uint16堆栈（内存优化，避免float32常驻）
        self.pairs = []        # TIFF模式：(stack_idx, start_frame) 数据对索引

        if use_tiff:
            self._load_tiff_data(data_root)
            # 对于TIFF模式，根据实际数据尺寸重新计算patch数量
            if len(self.tiff_stacks) > 0:
                # 获取第一个数据的尺寸作为参考
                sample_data = self.tiff_stacks[0]
                if sample_data.ndim == 4:  # (T, H, W, C)
                    self.h = sample_data.shape[2]  # height
                    self.w = sample_data.shape[3]  # width
                else:  # (T, H, W)
                    self.h = sample_data.shape[1]  # height
                    self.w = sample_data.shape[2]  # width
                print(f"TIFF图像尺寸: H={self.h}, W={self.w}")
                # 重新计算patch数量
                self.patch_per_line = (self.w-crop_size)//stride+1
                self.patch_per_colum = (self.h-crop_size)//stride+1
                self.patch_per_img = self.patch_per_line*self.patch_per_colum
                print(f"每个图像的patch数量: {self.patch_per_img}")
            self.img_num = len(self.pairs)
        else:
            self._load_mat_data(data_root)
            self.img_num = len(self.hypers)

        self.length = self.patch_per_img * self.img_num
        print(f'length of dataset: {self.length}')

    def _load_tiff_data(self, data_root):
        """加载TIFF堆栈数据（内存优化版：仅保留原始堆栈，归一化延迟到__getitem__）"""
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile package is required for TIFF support")

        tiff_data_path = f'{data_root}/Train/'
        print(f'TIFF data path: {tiff_data_path}')

        # 查找所有TIFF文件
        allowed_extensions = ['tif', 'tiff', 'TIF', 'TIFF']
        tiff_files = glob.glob(os.path.join(tiff_data_path, '*'))
        tiff_list = [f for f in tiff_files if f.split('.')[-1] in allowed_extensions]
        tiff_list.sort()

        print(f'Found {len(tiff_list)} TIFF stack files')

        # 处理每个TIFF堆栈
        for tiff_idx, tiff_path in enumerate(tiff_list):
            print(f'Loading TIFF stack {tiff_idx + 1}: {os.path.basename(tiff_path)}')

            # 读取TIFF堆栈
            tiff_stack = imread(tiff_path)  # 形状应该是 (T, H, W) 或 (T, H, W, C)

            # 确保数据形状正确
            if len(tiff_stack.shape) == 3:  # (T, H, W)
                total_frames = tiff_stack.shape[0]
            elif len(tiff_stack.shape) == 4:  # (T, H, W, C)
                total_frames = tiff_stack.shape[0]
            else:
                raise ValueError(f"Unexpected TIFF stack shape: {tiff_stack.shape}")

            print(f'Total frames in stack: {total_frames}')

            stack_idx = len(self.tiff_stacks)
            self.tiff_stacks.append(tiff_stack)

            pairs_generated = 0
            for start_frame in range(0, total_frames - self.hyper_frames + 1, self.step):
                if start_frame + self.hyper_frames > total_frames:
                    break
                self.pairs.append((stack_idx, start_frame))
                pairs_generated += 1

            print(f'Generated {pairs_generated} data pairs from stack {tiff_idx + 1}')

        print(f'Total pairs: {len(self.pairs)}')

    def _extract_tiff_pair(self, stack_idx, start_frame):
        """从原始堆栈中提取一对(hyper, less)数据并归一化"""
        tiff_stack = self.tiff_stacks[stack_idx]

        # 选择hyper帧（连续的帧）
        hyper_indices = list(range(start_frame, start_frame + self.hyper_frames))

        # 选择less帧（第一帧和最后一帧）
        less_indices = [start_frame, start_frame + self.hyper_frames - 1]

        # 如果需要更多的less帧，可以均匀采样
        if self.less_frames > 2:
            step_size = (self.hyper_frames - 1) / (self.less_frames - 1)
            less_indices = [int(start_frame + i * step_size) for i in range(self.less_frames)]
        elif self.less_frames == 1:
            less_indices = [start_frame]

        # 提取数据
        hyper_data = tiff_stack[hyper_indices]
        less_data = tiff_stack[less_indices]

        # 归一化
        if tiff_stack.dtype == np.uint16:
            hyper_data = (hyper_data.astype(np.float32) - self.background_value) / self.scalefactor
            less_data = (less_data.astype(np.float32) - self.background_value) / self.scalefactor
        elif tiff_stack.dtype == np.uint8:
            hyper_data = (hyper_data.astype(np.float32) - self.background_value) / 255.0
            less_data = (less_data.astype(np.float32) - self.background_value) / 255.0
        else:
            hyper_data = hyper_data.astype(np.float32)
            less_data = less_data.astype(np.float32)

        # 调整维度顺序为 (C, H, W)
        if len(hyper_data.shape) == 4:  # (T, H, W, C)
            hyper_data = np.transpose(hyper_data, (0, 3, 1, 2))  # (T, C, H, W)
            hyper_data = hyper_data.reshape(-1, hyper_data.shape[2], hyper_data.shape[3])
            less_data = np.transpose(less_data, (0, 3, 1, 2))    # (T, C, H, W)
            less_data = less_data.reshape(-1, less_data.shape[2], less_data.shape[3])
        # (T, H, W) 已经是 (C, H, W)，无需调整

        return hyper_data, less_data

    def _load_mat_data(self, data_root):
        """加载原始MAT文件数据"""
        hyper_data_path = f'{data_root}/Train_Hyper/' # hyperspectral data path,字符串格式化
        lc_data_path = f'{data_root}/Train_Less/' # less channels spectrum data path
        # print(f'hyper_data_path: {hyper_data_path}')
        # print(f'lc_data_path: {lc_data_path}')
        # 确保文件夹中的文件是以下后缀名的数据：JPG, PNG, jpeg, mat, h5
        allowed_extensions = ['JPG', 'PNG', 'jpeg', 'png', 'mat', 'h5']
        # no need to read the file name, just read the data in the folder,it might be JPG, PNG, jpeg, mat, h5, etc.
        hyper_files = glob.glob(os.path.join(hyper_data_path, '*'))

        lc_files = glob.glob(os.path.join(lc_data_path, '*'))

        hyper_list = [f for f in hyper_files if f.split('.')[-1] in allowed_extensions]
        lc_list = [f for f in lc_files if f.split('.')[-1] in allowed_extensions]
        hyper_list.sort()
        lc_list.sort()


        print(f'len(hyper) of dataset:{hyper_data_path} {len(hyper_list)}')
        print(f'len(lc) of dataset:{lc_data_path} {len(lc_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            lc_path = lc_list[i]
            if 'mat' in hyper_path:
                mat_data_hyper_train = loadmat(hyper_path)
                hyper = (np.float32(mat_data_hyper_train['hyperScaled']) - self.background_value) / self.scalefactor
                # hyper = np.float32(mat_data_hyper_train['hyperScaled'])/self.scalefactor
                # with h5py.File(hyper_path, 'r') as mat:
                #     hyper = np.float32(np.array(mat['cube_y']))
            else:
                hyper = Image.open(hyper_path).convert('F')
            # hyper = np.transpose(hyper, [0, 2, 1])
            hyper = np.transpose(hyper, [ 2, 0, 1])

            # assert hyper_list[i].split('.')[0] == lc_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'

            # 只使用文件夹名中的数字部分来判断是否是同一场景
            hyper_base = os.path.basename(hyper_path).split('_')[1]
            lc_base = os.path.basename(lc_path).split('_')[1]
            assert hyper_base == lc_base, 'Hyper and RGB come from different scenes.'

            if 'mat' in lc_path:
                mat_data_lc_train = loadmat(lc_path)
                lc = (np.float32(mat_data_lc_train['lessChunk']) - self.background_value) / self.scalefactor
                # with h5py.File(lc_path, 'r') as mat:
                #     lc = np.float32(np.array(mat['cube_x']))
            else:
                lc = Image.open(lc_path).convert('F')
            # lc = np.transpose(lc, [0, 2, 1])
            #如果lc只有一个通道，比如[420,420],记得增加通道维度，变成[420,420,1]
            if lc.ndim == 2:
                lc = np.expand_dims(lc, axis=-1)
            # lc = np.transpose(lc, [0, 2, 1])

            lc = np.transpose(lc, [ 2, 0, 1])
            self.hypers.append(hyper)
            self.lcs.append(lc)
            # mat.close()
            print(f'scene {i+1} is loaded.')

    def augment(self, img, rotTimes, vFlip, hFlip):
        # random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(),axes = (1,2)) # rotate the image, rotTime need radomly choose a number
        # random vertical flip
        for j in range(vFlip):
            img = img[:,:,::-1].copy()
        # random horizontal flip
        for j in range(hFlip):
            img = img[:,::-1,:].copy()
        return img

    def __getitem__(self, idx):
        #idx 是整体的索引，按照patch计数，idx is the index of the whole image
        stride = self.stride
        crop_size = self.crop_size
        img_idx = idx // self.patch_per_img # get the index of the image,img_idx is the index of the image
        patch_idx = idx % self.patch_per_img # get the index of the patch, patch_idx is the index of the patch
        #先从左到右，再从上往下
        h_idx = patch_idx // self.patch_per_line #整除运算列（w）的分块数，得到行的索引
        w_idx = patch_idx % self.patch_per_line#取余运算，得到列的索引

        hyper = None
        lc = None
        if self.use_tiff:
            stack_idx, start_frame = self.pairs[img_idx]
            hyper, lc = self._extract_tiff_pair(stack_idx, start_frame)
        else:
            hyper = self.hypers[img_idx]
            lc = self.lcs[img_idx]
        hyper = hyper[:,h_idx*stride:h_idx*stride+crop_size,w_idx*stride:w_idx*stride+crop_size]
        lc = lc[:,h_idx*stride:h_idx*stride+crop_size,w_idx*stride:w_idx*stride+crop_size]
        rotTimes = random.randint(0,3)
        vFlip = random.randint(0,1)
        hFlip = random.randint(0,1)
        if self.aug:
            hyper = self.augment(hyper,rotTimes,vFlip,hFlip)
            lc = self.augment(lc,rotTimes,vFlip,hFlip)
        return np.ascontiguousarray(hyper),np.ascontiguousarray(lc)

    def __len__(self):
        return self.patch_per_img * self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, use_tiff=False, hyper_frames=4, less_frames=2, step=4, scalefactor=65535.0, background_value=0.0):
        """
        验证数据集
        Args:
            data_root: 数据根目录
            use_tiff: 是否使用TIFF堆栈模式
            hyper_frames: TIFF模式下hyper数据的帧数
            less_frames: TIFF模式下less数据的帧数
            step: TIFF模式下滑动窗口步长
            scalefactor: 归一化比例因子 (默认65535.0用于16位数据)
            background_value: 背景值，用于数据预处理 (默认0.0)
        """
        self.hypers = []
        self.lcs = []
        self.use_tiff = use_tiff
        self.hyper_frames = hyper_frames
        self.less_frames = less_frames
        self.step = step
        self.scalefactor = scalefactor
        self.background_value = background_value
        self.tiff_stacks = []  # TIFF模式：原始uint16堆栈（内存优化）
        self.pairs = []        # TIFF模式：(stack_idx, start_frame) 数据对索引

        if use_tiff:
            self._load_tiff_data(data_root)
        else:
            self._load_mat_data(data_root)

    def _load_tiff_data(self, data_root):
        """加载TIFF堆栈数据（内存优化版：仅保留原始堆栈，归一化延迟到__getitem__）"""
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile package is required for TIFF support")

        tiff_data_path = f'{data_root}/Valid/'
        print(f'Validation TIFF data path: {tiff_data_path}')

        # 查找所有TIFF文件
        allowed_extensions = ['tif', 'tiff', 'TIF', 'TIFF']
        tiff_files = glob.glob(os.path.join(tiff_data_path, '*'))
        tiff_list = [f for f in tiff_files if f.split('.')[-1] in allowed_extensions]
        tiff_list.sort()

        print(f'Found {len(tiff_list)} validation TIFF stack files')

        # 处理每个TIFF堆栈
        for tiff_idx, tiff_path in enumerate(tiff_list):
            print(f'Loading validation TIFF stack {tiff_idx + 1}: {os.path.basename(tiff_path)}')

            # 读取TIFF堆栈
            tiff_stack = imread(tiff_path)

            # 确保数据形状正确
            if len(tiff_stack.shape) == 3:  # (T, H, W)
                total_frames = tiff_stack.shape[0]
            elif len(tiff_stack.shape) == 4:  # (T, H, W, C)
                total_frames = tiff_stack.shape[0]
            else:
                raise ValueError(f"Unexpected TIFF stack shape: {tiff_stack.shape}")

            print(f'Total frames in validation stack: {total_frames}')

            stack_idx = len(self.tiff_stacks)
            self.tiff_stacks.append(tiff_stack)

            pairs_generated = 0
            for start_frame in range(0, total_frames - self.hyper_frames + 1, self.step):
                if start_frame + self.hyper_frames > total_frames:
                    break
                self.pairs.append((stack_idx, start_frame))
                pairs_generated += 1

            print(f'Generated {pairs_generated} validation data pairs from stack {tiff_idx + 1}')

        print(f'Total validation pairs: {len(self.pairs)}')

    def _extract_tiff_pair(self, stack_idx, start_frame):
        """从原始堆栈中提取一对(hyper, less)数据并归一化"""
        tiff_stack = self.tiff_stacks[stack_idx]

        # 选择hyper帧
        hyper_indices = list(range(start_frame, start_frame + self.hyper_frames))

        # 选择less帧
        less_indices = [start_frame, start_frame + self.hyper_frames - 1]
        if self.less_frames > 2:
            step_size = (self.hyper_frames - 1) / (self.less_frames - 1)
            less_indices = [int(start_frame + i * step_size) for i in range(self.less_frames)]
        elif self.less_frames == 1:
            less_indices = [start_frame]

        # 提取数据
        hyper_data = tiff_stack[hyper_indices]
        less_data = tiff_stack[less_indices]

        # 归一化
        if tiff_stack.dtype == np.uint16:
            hyper_data = (hyper_data.astype(np.float32) - self.background_value) / self.scalefactor
            less_data = (less_data.astype(np.float32) - self.background_value) / self.scalefactor
        elif tiff_stack.dtype == np.uint8:
            hyper_data = (hyper_data.astype(np.float32) - self.background_value) / 255.0
            less_data = (less_data.astype(np.float32) - self.background_value) / 255.0
        else:
            hyper_data = hyper_data.astype(np.float32)
            less_data = less_data.astype(np.float32)

        # 调整维度顺序
        if len(hyper_data.shape) == 4:  # (T, H, W, C)
            hyper_data = np.transpose(hyper_data, (0, 3, 1, 2))
            hyper_data = hyper_data.reshape(-1, hyper_data.shape[2], hyper_data.shape[3])
            less_data = np.transpose(less_data, (0, 3, 1, 2))
            less_data = less_data.reshape(-1, less_data.shape[2], less_data.shape[3])
        # (T, H, W) 已经是 (C, H, W)，无需调整

        return hyper_data, less_data

    def _load_mat_data(self, data_root):
        """加载原始MAT文件数据"""
        hyper_data_path = f'{data_root}/Valid_Hyper/'
        lc_data_path = f'{data_root}/Valid_Less/'
        allowed_extensions = ['JPG', 'PNG', 'jpeg', 'png', 'mat', 'h5']
        hyper_files = glob.glob(os.path.join(hyper_data_path, '*'))
        lc_files = glob.glob(os.path.join(lc_data_path, '*'))
        hyper_list = [f for f in hyper_files if f.split('.')[-1] in allowed_extensions]
        lc_list = [f for f in lc_files if f.split('.')[-1] in allowed_extensions]
        hyper_list.sort()
        lc_list.sort()
        print(f'len(hyper_valid) of dataset:{hyper_data_path} {len(hyper_list)}')
        print(f'len(lc_valid) of dataset:{lc_data_path} {len(lc_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            lc_path = lc_list[i]
            if 'mat' in hyper_path:
                mat_data_hyper_valid = loadmat(hyper_path)
                hyper = (np.float32(mat_data_hyper_valid['hyperScaled']) - self.background_value) / self.scalefactor  # 归一化到0-1之间
                # with h5py.File(hyper_path, 'r') as mat:
                #     hyper = np.float32(np.array(mat['cube_8']))
            else:
                hyper = Image.open(hyper_path).convert('F')
            # hyper = np.transpose(hyper, [0, 2, 1])
            hyper = np.transpose(hyper, [ 2, 0, 1])

            # assert hyper_list[i].split('.')[0] == lc_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'

            hyper_base = os.path.basename(hyper_path).split('_')[1]
            lc_base = os.path.basename(lc_path).split('_')[1]
            assert hyper_base == lc_base, 'Hyper and RGB come from different scenes.'

            if 'mat' in lc_path:
                mat_data_lc_valid = loadmat(lc_path)
                lc = (np.float32(mat_data_lc_valid['lessChunk']) - self.background_value) / self.scalefactor  # 归一化到0-1之间
                # with h5py.File(lc_path, 'r') as mat:
                #     lc = np.float32(np.array(mat['cube_x']))
            else:
                lc = Image.open(lc_path).convert('F')
            if lc.ndim == 2:
                lc = np.expand_dims(lc, axis=-1)
            # lc = np.transpose(lc, [0, 2, 1])
            lc = np.transpose(lc, [ 2, 0, 1])
            self.hypers.append(hyper)
            self.lcs.append(lc)
            # mat.close()
            print(f'validation scene {i+1} is loaded.')

    def __getitem__(self, idx):
        if self.use_tiff:
            stack_idx, start_frame = self.pairs[idx]
            hyper, lc = self._extract_tiff_pair(stack_idx, start_frame)
        else:
            hyper = self.hypers[idx]
            lc = self.lcs[idx]
        return np.ascontiguousarray(hyper), np.ascontiguousarray(lc)

    def __len__(self):
        return len(self.pairs) if self.use_tiff else len(self.hypers)

#test dataloader

# TIFF模式测试
# data_root = 'f:/3DVolumeReducerTriple/train_code/dataset_tiff/'
# crop_size = 128
# train_data = TrainDataset(data_root, crop_size, aug=True, h=420, w=420, stride=8,
#                          use_tiff=True, hyper_frames=4, less_frames=2, step=4)
# print(f"TIFF Train dataset length: {len(train_data)}")
# if len(train_data) > 0:
#     hyper, lc = train_data[0]
#     print(f"TIFF Train data shape: {hyper.shape}, {lc.shape}")

# valid_data = ValidDataset(data_root, use_tiff=True, hyper_frames=4, less_frames=2, step=4)
# print(f"TIFF Validation dataset length: {len(valid_data)}")
# if len(valid_data) > 0:
#     hyper, lc = valid_data[0]
#     print(f"TIFF Validation data shape: {hyper.shape}, {lc.shape}")

# MAT模式测试（原始）
# crop_size = 128
# train_data = TrainDataset(data_root, crop_size, aug=True, h=420, w=420, stride=8, use_tiff=False)
# print(f"MAT Train dataset length: {len(train_data)}")
# hyper, lc = train_data[0]
# print(f"MAT Train data shape: {hyper.shape}, {lc.shape}")
# valid_data = ValidDataset(data_root, use_tiff=False)
# print(f"MAT Validation dataset length: {len(valid_data)}")
# hyper, lc = valid_data[0]
# print(f"MAT Validation data shape: {hyper.shape}, {lc.shape}")
