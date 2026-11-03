import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from load_mata_hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime

parser = argparse.ArgumentParser(description="Spectral Expansion Toolbox")
parser.add_argument('--method', type=str, default='fsr', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./outputs/isr_training/', help='path log files')
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU device ID')
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--use_simple_loss', action='store_true', help='Use simple L1 loss instead of composite loss')
# TIFF mode parameters
parser.add_argument('--use_tiff', action='store_true', help='Use TIFF stack mode for training')
parser.add_argument('--hyper_frames', type=int, default=4, help='Number of frames for hyper data in TIFF mode')
parser.add_argument('--less_frames', type=int, default=2, help='Number of frames for less data in TIFF mode')
parser.add_argument('--step', type=int, default=4, help='Step size for sliding window in TIFF mode')
parser.add_argument('--scalefactor', type=float, default=250.0, help='Scale factor for normalization (default: 65535.0 for 16-bit)')
parser.add_argument('--background_value', type=float, default=110.0, help='Background value for normalization (default: 0.0)')

opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 设置随机种子以确保实验的可重复性
import random
import numpy as np
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)  # 多GPU情况下
# 确保CuDNN使用确定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load dataset
print("\nloading dataset ...")
if opt.use_tiff:
    print("使用TIFF数据集模式")
    # 对于TIFF模式，不指定固定的h和w，让数据集自动检测
    train_data = TrainDataset(
        data_root=opt.data_root,
        crop_size=opt.patch_size,
        aug=True,
        stride=opt.stride,
        use_tiff=True,
        hyper_frames=opt.hyper_frames,
        less_frames=opt.less_frames,
        step=opt.step,
        scalefactor=opt.scalefactor,
        background_value=opt.background_value
    )
    val_data = ValidDataset(
        data_root=opt.data_root,
        use_tiff=True,
        hyper_frames=opt.hyper_frames,
        less_frames=opt.less_frames,
        step=opt.step,
        scalefactor=opt.scalefactor,
        background_value=opt.background_value
    )
    print(f"TIFF模式参数: hyper_frames={opt.hyper_frames}, less_frames={opt.less_frames}, step={opt.step}, scalefactor={opt.scalefactor}, background_value={opt.background_value}")
else:
    print("使用MAT数据集模式")
    train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, h=512, w=256, aug=True, stride=opt.stride, scalefactor=opt.scalefactor, background_value=opt.background_value)
    val_data = ValidDataset(data_root=opt.data_root, scalefactor=opt.scalefactor, background_value=opt.background_value)

print(f"Iteration per epoch: {len(train_data)}")
print("Validation set samples: ", len(val_data))

# 创建数据加载器
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()

# 引入新的损失函数
from architecture.losses import CompositeLoss, CharbonnierLoss, EdgeLoss
criterion_composite = CompositeLoss(charbonnier_weight=1.0, edge_weight=0.1, tv_weight=1e-4).cuda()
criterion_charbonnier = CharbonnierLoss().cuda()
criterion_edge = EdgeLoss().cuda()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# 输出训练配置信息
print(f"\n======== 训练配置 ========")
print(f"  方法: {method}")
print(f"  数据根目录: {opt.data_root}")
print(f"  数据模式: {'TIFF' if opt.use_tiff else 'MAT'}")
if opt.use_tiff:
    print(f"  TIFF参数:")
    print(f"    - hyper_frames: {opt.hyper_frames}")
    print(f"    - less_frames: {opt.less_frames}")
    print(f"    - step: {opt.step}")
print(f"  归一化因子: {opt.scalefactor}")
print(f"  背景值: {opt.background_value}")
print(f"  随机种子: {opt.seed}")
print(f"  批大小: {opt.batch_size}")
print(f"  初始学习率: {opt.init_lr}")
print(f"  训练轮数: {opt.end_epoch}")
print(f"  补丁大小: {opt.patch_size}")
print(f"  步长: {opt.stride}")
print(f"  GPU ID: {opt.gpu_id}")
print(f"  输出目录: {opt.outf}")
print(f"  损失函数: {'简单L1损失' if opt.use_simple_loss else '复合损失(Charbonnier+Edge)'}")
print(f"  模型参数数量: {sum(param.numel() for param in model.parameters())}")
print("=" * 30)

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_mse.cuda()
    criterion_l1.cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # cudnn.benchmark设置为True会自动寻找最高效的算法，可能影响可重复性
    # 但设置为False会影响性能，这里保持与之前的设置一致
    cudnn.benchmark = False
    iteration = 0
    record_mrae_loss = 1000
    record_rmse_loss = 1000
    record_l1_loss = 1000
    record_mse_loss = 1000

    print("开始训练...")
    print(f"总迭代次数: {total_iteration}")
    print(f"每轮迭代次数: {per_epoch_iteration}")

    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        for i, (labels, images) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            # print('labels max:', labels.max())
            # print('labels min:', labels.min())
            # 检查输入数据是否包含无效值
            if torch.isnan(images).any() or torch.isnan(labels).any():
                print("Input contains NaN values")
                continue
            if torch.isinf(images).any() or torch.isinf(labels).any():
                print("Input contains Inf values")
                continue
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            # 为不同的帧分配不同的权重
            # 输入包含frame0和frame3，所以这两帧的权重较小
            # frame1和frame2是需要额外生成的帧，权重较大
            frame_weights = torch.tensor([0.01, 3.0, 3.0, 0.01], device=labels.device)
            frame_weights = frame_weights.view(-1, 1, 1)  # 形状: (4, 1, 1)

            # 使用element-wise损失计算，并为不同帧分配权重
            import torch.nn.functional as F

            # MSE损失：element-wise计算，然后乘以帧权重
            # mse_per_element = F.mse_loss(output, labels, reduction='none')
            # mse_weighted = mse_per_element * frame_weights
            # loss_mse = mse_weighted.mean()
            # 先把loss_mse写成0，但是能和loss_l1做加法运算
            loss_mse = torch.tensor(0.0, device=labels.device)

            # 根据use_simple_loss标志选择损失函数
            if opt.use_simple_loss:
                # 使用简单的L1损失
                l1_per_element = F.l1_loss(output, labels, reduction='none')
                l1_weighted = l1_per_element * frame_weights
                loss_l1 = l1_weighted.mean()
                total_loss = loss_mse + loss_l1
                loss_charbonnier = torch.tensor(0.0, device=labels.device)
                loss_edge = torch.tensor(0.0, device=labels.device)
            else:
                # 使用CompositeLoss代替L1损失，保持帧权重功能
                # CompositeLoss会返回总损失和各分量损失
                # 首先计算基础复合损失
                composite_loss, loss_components = criterion_composite(output, labels)

                # 为复合损失应用帧权重
                # 我们使用类似Charbonnier的element-wise计算方式来应用帧权重
                charbonnier_per_element = torch.sqrt(torch.pow(output - labels, 2) + 1e-6)
                charbonnier_weighted = charbonnier_per_element * frame_weights
                loss_charbonnier = charbonnier_weighted.mean()

                # 如果需要，我们也可以为边缘损失应用帧权重
                # 计算边缘损失的element-wise值
                edge_loss_per_element = criterion_edge(output, labels)
                edge_weighted = edge_loss_per_element * frame_weights
                loss_edge = edge_weighted.mean()

                # 组合带权重的Charbonnier损失和边缘损失
                # 这里的权重可以根据需要调整
                total_loss = loss_charbonnier + 0.1 * loss_edge + loss_mse
                loss_l1 = torch.tensor(0.0, device=labels.device)  # 用于保持日志输出的一致性

            loss = total_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f,train_mse=%.9f,train_charbonnier=%.9f,train_edge=%.9f'
                      % (iteration, total_iteration, lr, losses.avg, loss_mse, loss_charbonnier, loss_edge))
            if iteration % 10000 == 0:
                print(f'\n=== 第 {iteration//1000} 轮验证 ===')
                rmse_loss, psnr_loss, mse_loss, l1_loss = validate(val_loader, model, opt.outf, iteration, opt.scalefactor)
                print(f'验证结果 - RMSE: {rmse_loss:.6f}, PSNR: {psnr_loss:.6f}, MSE: {mse_loss:.6f}, L1: {l1_loss:.6f}')

                # Save model
                if  rmse_loss < record_rmse_loss or iteration % 10000 == 0:
                    print(f'保存模型到 {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if rmse_loss < record_rmse_loss:
                        record_rmse_loss = rmse_loss
                        print(f'新的最佳RMSE: {record_rmse_loss:.6f}')

                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, rmse_loss, psnr_loss))
                print('=' * 50)

            # 如果达到每轮训练的迭代次数，重新开始
            if (iteration % per_epoch_iteration == 0):
                break

    return 0

# Validate
def validate(val_loader, model, outf, iteration, scalefactor):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_l1 = AverageMeter()
    losses_mse = AverageMeter()

    # 创建保存图像的文件夹
    img_save_path = os.path.join(outf, f'val_images_iter_{iteration}')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    # 清理旧的val_images目录（只保留当前这次），避免磁盘累积导致No space left
    try:
        import shutil
        old_dirs = [d for d in os.listdir(outf) if d.startswith('val_images_iter_') and os.path.join(outf, d) != img_save_path]
        for d in old_dirs:
            old_path = os.path.join(outf, d)
            shutil.rmtree(old_path, ignore_errors=True)
    except Exception:
        pass

    # 图像保存间隔（每10个样本保存一个）
    save_interval = 10

    for i, (target, input_) in enumerate(val_loader):
        input_ = input_.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input_)
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_l1 = criterion_l1(output, target)
            loss_mse = criterion_mse(output, target)

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_l1.update(loss_l1.data)
        losses_mse.update(loss_mse.data)

        # 按固定间隔保存图像
        if i % save_interval == 0:
            # 保存输入、目标和输出图像
            import numpy as np
            import tifffile as tiff

            # 转换为numpy数组并恢复原始尺度
            input_np = input_.cpu().numpy()[0] * scalefactor + opt.background_value  # 取第一个样本
            target_np = target.cpu().numpy()[0] * scalefactor + opt.background_value
            output_np = output.cpu().numpy()[0] * scalefactor + opt.background_value

            # 保存所有通道/帧
            def save_all_frames(img_np, filename_prefix):
                """保存所有帧/通道，保持原始数据类型
                img_np shape: (channels/frames, height, width)
                """
                if img_np.ndim == 3:
                    # 多通道/帧数据
                    num_frames = img_np.shape[0]
                    for frame_idx in range(num_frames):
                        frame = img_np[frame_idx]
                        # 使用TIFF格式保存，用uint16类型
                        tiff.imwrite(f"{filename_prefix}_frame_{frame_idx}.tif", frame.astype(np.uint16))
                else:
                    # 单通道数据
                    tiff.imwrite(f"{filename_prefix}.tif", img_np.astype(np.uint16))

            # 保存图像
            save_all_frames(input_np, os.path.join(img_save_path, f'input_iter_{iteration}_batch_{i}'))
            save_all_frames(target_np, os.path.join(img_save_path, f'target_iter_{iteration}_batch_{i}'))
            save_all_frames(output_np, os.path.join(img_save_path, f'output_iter_{iteration}_batch_{i}'))

    return losses_rmse.avg, losses_psnr.avg, losses_mse.avg, losses_l1.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)
