import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRGANLosses:
    """
    FSR模型专用的GAN损失函数类
    为FSR网络提供结合回归损失和对抗损失的综合损失计算
    """
    def __init__(self, regression_weight=1.0, adversarial_weight=0.0, use_wgan=False, lambda_gp=10.0, start_adversarial_epoch=0, frame_weights=None):
        """
        Args:
            regression_weight: 回归损失的权重
            adversarial_weight: 对抗损失的权重
            use_wgan: 是否使用WGAN损失
            lambda_gp: WGAN-GP的梯度惩罚系数
            start_adversarial_epoch: 开始使用对抗损失的起始轮次
            frame_weights: 不同帧的权重列表，用于加权回归损失
        """
        self.regression_weight = regression_weight
        self.adversarial_weight = adversarial_weight
        self.use_wgan = use_wgan
        self.lambda_gp = lambda_gp
        self.start_adversarial_epoch = start_adversarial_epoch
        self.frame_weights = frame_weights

        # 标准GAN使用BCELoss
        if not use_wgan:
            self.criterion_gan = nn.BCEWithLogitsLoss()
        else:
            # WGAN-GP不使用损失函数，而是直接优化判别器输出
            self.criterion_gan = None

    def generator_loss(self, generated_output, target, discriminator_pred_fake, current_epoch=0):
        """
        计算生成器损失

        Args:
            generated_output: 生成器生成的输出
            target: 目标真实数据
            discriminator_pred_fake: 判别器对生成数据的预测
            current_epoch: 当前训练轮次

        Returns:
            tuple: (总损失, 回归损失, 对抗损失)
        """
        # 计算回归损失 (MSE)
        if self.frame_weights is not None:
            # 应用帧权重
            weight_tensor = torch.tensor(self.frame_weights, device=generated_output.device).view(-1, 1, 1)
            mse_loss_per_pixel = F.mse_loss(generated_output, target, reduction='none')
            # 对每个帧的损失进行加权
            weighted_mse = mse_loss_per_pixel * weight_tensor
            regression_loss = weighted_mse.mean()
        else:
            regression_loss = F.mse_loss(generated_output, target)

        # 根据当前轮次决定是否使用对抗损失
        adv_weight = self.get_adversarial_weight(current_epoch)

        # 计算对抗损失
        if not self.use_wgan:
            # 标准GAN损失
            real_label = torch.ones_like(discriminator_pred_fake)
            adversarial_loss = self.criterion_gan(discriminator_pred_fake, real_label)
        else:
            # WGAN损失
            adversarial_loss = -torch.mean(discriminator_pred_fake)

        # 总损失 = 回归损失 + 对抗损失 * 对抗权重
        total_loss = self.regression_weight * regression_loss + adv_weight * adversarial_loss

        return total_loss, regression_loss, adversarial_loss

    def discriminator_loss(self, discriminator, discriminator_pred_real, discriminator_pred_fake, real_input, fake_input, current_epoch=0):
        """
        计算判别器损失

        Args:
            discriminator: 判别器模型实例
            discriminator_pred_real: 判别器对真实数据的预测
            discriminator_pred_fake: 判别器对生成数据的预测
            real_input: 真实输入数据（用于WGAN-GP的梯度惩罚）
            fake_input: 生成的虚假数据（用于WGAN-GP的梯度惩罚）
            current_epoch: 当前训练轮次

        Returns:
            判别器总损失
        """
        # 根据当前轮次决定是否使用对抗损失
        adv_weight = self.get_adversarial_weight(current_epoch)

        if adv_weight <= 0:
            # 如果不使用对抗损失，返回0
            return torch.tensor(0.0, device=discriminator_pred_real.device)

        if not self.use_wgan:
            # 标准GAN损失
            real_label = torch.ones_like(discriminator_pred_real)
            fake_label = torch.zeros_like(discriminator_pred_fake)

            loss_real = self.criterion_gan(discriminator_pred_real, real_label)
            loss_fake = self.criterion_gan(discriminator_pred_fake, fake_label)

            # 判别器总损失
            total_loss = (loss_real + loss_fake) * 0.5 * adv_weight
        else:
            # WGAN-GP损失
            loss_real = -torch.mean(discriminator_pred_real)
            loss_fake = torch.mean(discriminator_pred_fake)

            # 添加梯度惩罚
            gradient_penalty = self.gradient_penalty(discriminator, real_input, fake_input)

            # 判别器总损失
            total_loss = (loss_real + loss_fake + self.lambda_gp * gradient_penalty) * adv_weight

        return total_loss

    def gradient_penalty(self, discriminator, real_data, fake_data):
        """
        计算WGAN-GP的梯度惩罚

        Args:
            discriminator: 判别器模型实例
            real_data: 真实数据样本
            fake_data: 生成的虚假数据样本

        Returns:
            梯度惩罚损失
        """
        # 随机采样alpha
        alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        # 创建插值样本
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # 计算判别器对插值样本的输出
        disc_interpolates = discriminator(interpolates)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 梯度惩罚项
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def get_adversarial_weight(self, current_epoch):
        """
        根据当前轮次获取对抗损失权重
        从start_adversarial_epoch开始线性增长到设定的权重值
        """
        if current_epoch < self.start_adversarial_epoch:
            return 0.0
        elif self.adversarial_weight <= 0:
            return 0.0
        else:
            return self.adversarial_weight


class GANLosses(nn.Module):
    """
    混合损失函数，包含回归损失和对抗损失
    支持动态切换机制，可以在训练过程中调整损失权重
    """
    def __init__(self, regression_weight=1.0, adversarial_weight=0.001,
                 start_adversarial_epoch=0, ramp_up_epochs=10,
                 regression_type='l1', use_gradient_penalty=False):
        """
        Args:
            regression_weight: 回归损失的初始权重
            adversarial_weight: 对抗损失的初始权重
            start_adversarial_epoch: 开始引入对抗损失的轮数
            ramp_up_epochs: 对抗损失权重的线性增长轮数
            regression_type: 回归损失类型 ('l1', 'mse', 'l1_mse')
            use_gradient_penalty: 是否使用梯度惩罚（WGAN-GP）
        """
        super(GANLosses, self).__init__()

        # 损失权重和切换参数
        self.regression_weight = regression_weight
        self.adversarial_weight = adversarial_weight
        self.start_adversarial_epoch = start_adversarial_epoch
        self.ramp_up_epochs = ramp_up_epochs
        self.use_gradient_penalty = use_gradient_penalty

        # 当前轮数
        self.current_epoch = 0

        # 初始化回归损失函数
        if regression_type == 'l1':
            self.regression_criterion = nn.L1Loss()
        elif regression_type == 'mse':
            self.regression_criterion = nn.MSELoss()
        elif regression_type == 'l1_mse':
            self.regression_criterion = lambda x, y: nn.L1Loss()(x, y) + nn.MSELoss()(x, y)
        else:
            raise ValueError(f"不支持的回归损失类型: {regression_type}")

        # 初始化对抗损失函数 (BCEWithLogitsLoss 或 MSELoss)
        self.adversarial_criterion = nn.BCELoss()

    def update_epoch(self, epoch):
        """
        更新当前轮数，用于动态调整损失权重
        """
        self.current_epoch = epoch

    def get_adversarial_weight(self):
        """
        根据当前轮数计算对抗损失的权重
        实现线性增长的ramp-up机制
        """
        if self.current_epoch < self.start_adversarial_epoch:
            return 0.0

        if self.current_epoch < self.start_adversarial_epoch + self.ramp_up_epochs:
            # 线性增长阶段
            progress = (self.current_epoch - self.start_adversarial_epoch) / self.ramp_up_epochs
            return self.adversarial_weight * progress

        # 完全权重阶段
        return self.adversarial_weight

    def generator_loss(self, fake_output, real_target, discriminator_pred_fake):
        """
        计算生成器的总损失

        Args:
            fake_output: 生成器生成的输出
            real_target: 真实目标
            discriminator_pred_fake: 判别器对生成输出的预测

        Returns:
            总损失及各部分损失
        """
        # 计算回归损失
        regression_loss = self.regression_criterion(fake_output, real_target)

        # 计算对抗损失 (生成器希望判别器认为生成的样本是真实的)
        batch_size = fake_output.size(0)
        real_labels = torch.ones_like(discriminator_pred_fake)
        adversarial_loss = self.adversarial_criterion(discriminator_pred_fake, real_labels)

        # 获取当前对抗损失权重
        adv_weight = self.get_adversarial_weight()

        # 总损失
        total_loss = self.regression_weight * regression_loss + adv_weight * adversarial_loss

        return {
            'total': total_loss,
            'regression': regression_loss,
            'adversarial': adversarial_loss,
            'adversarial_weight': adv_weight
        }

    def discriminator_loss(self, discriminator_pred_real, discriminator_pred_fake):
        """
        计算判别器的总损失

        Args:
            discriminator_pred_real: 判别器对真实样本的预测
            discriminator_pred_fake: 判别器对生成样本的预测

        Returns:
            判别器损失
        """
        batch_size = discriminator_pred_real.size(0)

        # 真实样本的标签和损失
        real_labels = torch.ones_like(discriminator_pred_real)
        real_loss = self.adversarial_criterion(discriminator_pred_real, real_labels)

        # 生成样本的标签和损失
        fake_labels = torch.zeros_like(discriminator_pred_fake)
        fake_loss = self.adversarial_criterion(discriminator_pred_fake, fake_labels)

        # 判别器总损失
        total_loss = (real_loss + fake_loss) * 0.5

        return {
            'total': total_loss,
            'real': real_loss,
            'fake': fake_loss
        }

    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        计算梯度惩罚 (用于WGAN-GP)

        Args:
            discriminator: 判别器模型
            real_samples: 真实样本
            fake_samples: 生成样本

        Returns:
            梯度惩罚损失
        """
        # 随机权重
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)

        # 在真实样本和生成样本之间插值
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        # 判别器对插值样本的预测
        d_interpolates = discriminator(interpolates)

        # 创建目标张量
        fake = torch.ones_like(d_interpolates).to(real_samples.device)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 计算梯度范数
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def discriminator_loss_with_gp(self, discriminator, real_samples, fake_samples, lambda_gp=10.0):
        """
        计算带有梯度惩罚的判别器损失 (WGAN-GP)

        Args:
            discriminator: 判别器模型
            real_samples: 真实样本
            fake_samples: 生成样本
            lambda_gp: 梯度惩罚的权重

        Returns:
            判别器总损失
        """
        # 判别器对真实样本和生成样本的预测
        d_real = discriminator(real_samples)
        d_fake = discriminator(fake_samples.detach())  # 不计算生成器的梯度

        # WGAN损失: 最大化真实样本得分，最小化生成样本得分
        wgan_loss = -torch.mean(d_real) + torch.mean(d_fake)

        # 计算梯度惩罚
        gp = self.gradient_penalty(discriminator, real_samples, fake_samples)

        # 总损失
        total_loss = wgan_loss + lambda_gp * gp

        return {
            'total': total_loss,
            'wgan': wgan_loss,
            'gradient_penalty': gp
        }

    def generator_loss_wgan(self, fake_output, real_target, discriminator_pred_fake):
        """
        计算WGAN中生成器的损失

        Args:
            fake_output: 生成器生成的输出
            real_target: 真实目标
            discriminator_pred_fake: 判别器对生成输出的预测

        Returns:
            总损失及各部分损失
        """
        # 计算回归损失
        regression_loss = self.regression_criterion(fake_output, real_target)

        # 计算对抗损失 (WGAN版本)
        adversarial_loss = -torch.mean(discriminator_pred_fake)

        # 获取当前对抗损失权重
        adv_weight = self.get_adversarial_weight()

        # 总损失
        total_loss = self.regression_weight * regression_loss + adv_weight * adversarial_loss

        return {
            'total': total_loss,
            'regression': regression_loss,
            'adversarial': adversarial_loss,
            'adversarial_weight': adv_weight
        }

# 辅助函数：计算不同的评估指标
class EvaluationMetrics:
    """
    评估指标集合，用于模型性能评估
    """
    @staticmethod
    def mse(pred, target):
        """计算均方误差"""
        return F.mse_loss(pred, target)

    @staticmethod
    def rmse(pred, target):
        """计算均方根误差"""
        return torch.sqrt(F.mse_loss(pred, target))

    @staticmethod
    def l1(pred, target):
        """计算L1损失"""
        return F.l1_loss(pred, target)

    @staticmethod
    def mrae(pred, target, epsilon=1e-8):
        """计算平均相对绝对误差"""
        # 避免除以零
        relative_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        return torch.mean(relative_error)

    @staticmethod
    def psnr(pred, target, data_range=1.0):
        """计算峰值信噪比"""
        mse_value = F.mse_loss(pred, target)
        if mse_value == 0:
            return float('inf')
        return 10 * torch.log10((data_range ** 2) / mse_value)


# ================ 新增损失函数 ================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier损失（平滑L1损失的变体）
    对小误差使用L2损失，对大误差使用L1损失，既有L2的平滑性又有L1的鲁棒性
    不容易产生平均多解的问题，有助于保持细节
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon)
        return torch.mean(loss)

class EdgeLoss(nn.Module):
    """
    边缘损失
    通过计算预测图像和真实图像的边缘差异来增强边缘细节
    有助于解决生成结果发糊的问题
    """
    def __init__(self):
        super().__init__()
        # Sobel边缘检测算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # 初始化Sobel权重
        sobel_x_weight = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_weight = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_x.weight.data = sobel_x_weight
        self.sobel_y.weight.data = sobel_y_weight

        # 设置为不学习的参数
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def get_edge(self, x):
        """计算图像的边缘图"""
        if x.size(1) > 1:
            # 多通道情况，对每个通道计算边缘后求和
            edge_x = 0
            edge_y = 0
            for i in range(x.size(1)):
                edge_x += self.sobel_x(x[:, i:i+1])
                edge_y += self.sobel_y(x[:, i:i+1])
        else:
            # 单通道情况
            edge_x = self.sobel_x(x)
            edge_y = self.sobel_y(x)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge

    def forward(self, pred, target):
        pred_edge = self.get_edge(pred)
        target_edge = self.get_edge(target)
        return F.l1_loss(pred_edge, target_edge)

class TotalVariationLoss(nn.Module):
    """
    总变分损失
    保持局部平滑但增强边缘，有助于减少模糊并保持细节
    """
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        # 计算水平方向的差异
        dx = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        # 计算垂直方向的差异
        dy = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return self.weight * (dx + dy)

class PerceptualLoss(nn.Module):
    """
    感知损失
    基于预训练模型的特征提取，考虑图像的语义信息而非像素级差异
    有助于生成更清晰、更符合人类视觉的结果
    """
    def __init__(self, feature_extractor=None):
        super().__init__()
        if feature_extractor is None:
            # 使用预训练的VGG16作为默认特征提取器
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features[:16].eval()
            # 设置为不学习
            for param in vgg.parameters():
                param.requires_grad = False
            self.feature_extractor = vgg
        else:
            self.feature_extractor = feature_extractor

        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        # 如果输入是单通道，转换为三通道以适应VGG
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        return self.loss(pred_features, target_features)

class CompositeLoss(nn.Module):
    """
    复合损失
    结合Charbonnier损失、边缘损失和总变分损失
    用于解决发糊问题，保持细节和边缘清晰度
    """
    def __init__(self, charbonnier_weight=1.0, edge_weight=0.1, tv_weight=1e-4):
        super().__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.tv_loss = TotalVariationLoss(weight=tv_weight)

        self.charbonnier_weight = charbonnier_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        charbonnier = self.charbonnier_loss(pred, target)
        edge = self.edge_loss(pred, target)
        tv = self.tv_loss(pred)

        total_loss = (self.charbonnier_weight * charbonnier +
                      self.edge_weight * edge +
                      tv)
        return total_loss, {
            'charbonnier': charbonnier,
            'edge': edge,
            'tv': tv
        }
