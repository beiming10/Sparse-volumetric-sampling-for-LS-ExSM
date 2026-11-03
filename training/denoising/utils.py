from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__() # super() 调用nn.Module父类继承。
    # MRAE（平均相对绝对误差）：计算预测值和真实值之间的相对绝对误差，并取其平均值
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        # 防止label中出现0值，导致除数为0
        label = torch.where(label == 0, torch.full_like(label, 1e-6), label)
        error = torch.abs(outputs - label) / label
        # print('outputs max', outputs.max())
        # print('outputs min', outputs.min())
        # print('label max', label.max())
        # print('label min', label.min())
        # 打印error的shape
        # print('error shape:', error.shape)
        mrae = torch.mean(error.reshape(-1)) # view(-1)将tensor展平成一维
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()
    # RMSE（均方根误差）：计算预测值和真实值之间的平方误差，并取其平方根
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.pow(outputs - label, 2)
        rmse = torch.sqrt(torch.mean(error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1000):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        # 将输入的数据归一化到[0,1]之间
        Itrue = im_true.clamp(0.,6.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0.,6.).mul_(data_range).reshape(N, C * H * W)
        mse = nn.MSELoss(reduction='none')
        mse = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. *torch.log((data_range ** 2) / mse) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    return year + month + day +'_'+ hour + minute + second

def record_loss(loss_csv, epoch, iteration, epoch_time, lr ,train_loss, test_loss):
    loss_csv.write('%d,%d,%s,%f,%f,%f\n' % (epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close()
