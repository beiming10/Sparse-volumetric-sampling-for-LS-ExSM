import torch
import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    """
    N层判别器，用于GAN训练
    直接输入单张图像，输出真假判断
    """
    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        # 设置初始归一化层参数
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # 卷积核大小
        padw = 1  # 填充大小
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # 添加中间层
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 输出层
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN判别器，适用于Pix2Pix。
    输入为条件对（输入图像和目标图像拼接）。
    输出为每个patch的真/假概率。
    """
    def __init__(self, in_channels=2, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, input, target):
        # input: 低帧/低光谱，target: 高帧/高光谱
        x = torch.cat([input, target], dim=1)
        return self.model(x)


def weights_init_normal(m):
    """
    初始化网络权重为正态分布
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 导入已移至文件顶部
