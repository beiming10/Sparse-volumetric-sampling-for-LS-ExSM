import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .Fan import FANLayer, FANLayerGated
from .pix2pix_util import NLayerDiscriminator, weights_init_normal
def _no_grad_trunc_normal_(tensor, mean, std, a , b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale = 1.0, mode = 'fan_in', distribution = 'normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std = math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3*variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(Tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn # 在下面定义了feedforward，由prenorm调用，fn是一个函数
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias = False, padding = 1, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = (kernal_size//2), bias = bias)

def shift_back(inputs, step=2):
    [ba, nC, h, w] = inputs.shape
    down_sample = 256//row
    step =float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)# q k v 线形层
        self.to_k = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias = False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)) # 用于缩放 head,
        self.proj = nn.Linear(dim_head * heads, dim, bias = True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias = False, groups = dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias = False, groups = dim),
        )
        self.dim = dim

    def forward(self, x_in):
        b , h, w, c = x_in.shape
        x = x_in.reshape(b, h*w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q , k , v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q_inp, k_inp, v_inp))

        #q:b,heads,hw,c
        q= q.transpose(-2,-1) #交换q的倒数第二个维度和最后一个维度
        k = k.transpose(-2,-1)
        v = v.transpose(-2,-1)
        q = F.normalize(q, dim = -1, p = 2)#对查询和健进行归一化，维度取倒数第一个，p=2表示二范数
        k = F.normalize(k, dim = -1, p = 2)
        attn =(k @ q.transpose(-2,-1))  #计算注意力矩阵
        attn = attn * self.rescale
        attn = attn.softmax(dim = -1)
        x = attn @ v #计算注意力矩阵和值的乘积 b , heads, hw, c
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h*w , self.dim_head * self.num_heads)
        out_c = self.proj(x).view(b, h, w, c)
        out_p =self.pos_emb(v_inp.reshape(b,h,w,c).permute(0,3,1,2)).permute(0,2,3,1)

        out = out_c + out_p
        out = out_c
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias = False),
            GELU(),
            nn.Conv2d(dim * mult, dim*mult,3,  1, 1, bias = False, groups = dim*mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias = False)
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class FeedForwardWithFAN(nn.Module):
    def __init__(self, dim, mult=4, p_ratio=0.25, activation='gelu', gated=True):
        super().__init__()
        self.embedding = nn.Linear(dim, int(dim * mult))  # 扩展到 hidden_dim
        self.layers = nn.ModuleList()

        # 添加 FANLayer 层
        for _ in range(2):  # 这里假设 FAN 层数为 2，可根据需求调整
            if gated:
                self.layers.append(FANLayerGated(int(dim * mult), int(dim * mult), gated=True))
            else:
                self.layers.append(FANLayer(int(dim * mult), int(dim * mult)))

        # 添加普通线性层
        self.reduce_linear = nn.Linear(int(dim * mult), dim)  # 恢复到原始通道数

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.view(b * h * w, c)  # 展平空间维度
        x = self.embedding(x)     # 初始线性变换

        # 应用 FANLayer
        for layer in self.layers:
            x = layer(x)

        x = self.reduce_linear(x)  # 恢复到原始维度
        x = x.view(b, h, w, -1)    # 恢复空间形状
        return x


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head= dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        x= x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class MSABWithFAN(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks, p_ratio=0.25, activation='gelu', gated=True):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 通道自注意力机制
                PreNorm(dim, FeedForwardWithFAN(dim=dim, p_ratio=p_ratio, activation=activation, gated=gated))  # FAN 前向传播网络
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # 自注意力模块
            x = ff(x) + x    # FAN 前向网络
        return x.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]


class MST(nn.Module):
    def __init__(self, in_dim=8, out_dim=8, dim=8, stage=2,num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim =dim
        self.stage = stage

        #input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias = False)

        #encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        #BottleNeck
        self.bottleneck = MSAB(
            dim = dim_stage, dim_head = dim, heads = dim_stage//dim, num_blocks=num_blocks[-1])

        #decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        #output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias = False)

        ### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        #embedding
        fea = self.embedding(x)

        #encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        #bottleneck
        fea = self.bottleneck(fea)

        #decoder
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlock(fea)

        #Mapping
        out = self.mapping(fea) + x

        return out


class MSTWithFAN(nn.Module):
    def __init__(self, in_dim=8, out_dim=8, dim=8, stage=2, num_blocks=[2, 4, 4], p_ratio=0.25, activation='gelu', gated=True):
        super(MSTWithFAN, self).__init__()
        self.dim = dim
        self.stage = stage

        # input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSABWithFAN(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # BottleNeck
        self.bottleneck = MSABWithFAN(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSABWithFAN(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        ### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # embedding
        fea = self.embedding(x)

        # encoder
        fea_encoder = []
        for (MSABWithFAN, FeaDownSample) in self.encoder_layers:
            fea = MSABWithFAN(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # bottleneck
        fea = self.bottleneck(fea)

        # decoder
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlock(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class FSR(nn.Module):
    def __init__(self, in_channels=2, out_channels=4, n_feat=8, stage=3):
        super(FSR, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        print(f'FSR: {in_channels} -> {n_feat}')
        modules_body = [MST(dim=n_feat, stage=3, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        print(f'FSR: {n_feat} -> {n_feat}')
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        print(f'FSR: {n_feat} -> {out_channels}')
        # 用于将 conv_in 的输出通道数映射到 out_channels，以便残差相加
        self.skip_conv = nn.Conv2d(n_feat, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x_feat = self.conv_in(x)
        h = self.body(x_feat)
        h = self.conv_out(h)
        # 将 skip 分支的通道数对齐到 out_channels 再相加
        h += self.skip_conv(x_feat)
        return h[:, :, :h_inp, :w_inp]


class FSRGAN(nn.Module):
    """
    FSR模型的GAN版本，结合FSR生成器和判别器
    用于引入对抗学习来提高生成数据的质量
    """
    def __init__(self, generator_in_channels=2, generator_out_channels=4,
                 n_feat=8, stage=3, use_fan=False,
                 ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Args:
            generator_in_channels: 生成器输入通道数
            generator_out_channels: 生成器输出通道数
            n_feat: 生成器内部特征通道数
            stage: 生成器内部MST模块数量
            use_fan: 是否使用FAN增强的生成器
            ndf: 判别器初始特征通道数
            n_layers: 判别器层数量
            norm_layer: 判别器使用的归一化层
        """
        super(FSRGAN, self).__init__()

        # 初始化生成器
        if use_fan:
            # 使用带FAN增强的生成器
            self.generator = FSRWithFAN(
                in_channels=generator_in_channels,
                out_channels=generator_out_channels,
                n_feat=n_feat,
                stage=stage
            )
        else:
            # 使用普通生成器
            self.generator = FSR(
                in_channels=generator_in_channels,
                out_channels=generator_out_channels,
                n_feat=n_feat,
                stage=stage
            )

        # 初始化判别器
        # 判别器的输入通道数应该与生成器的输出通道数相同
        self.discriminator = NLayerDiscriminator(
            input_nc=generator_out_channels,
            ndf=ndf,
            n_layers=n_layers,
            norm_layer=norm_layer
        )

        print(f"FSRGAN 初始化完成:")
        print(f"  - 生成器: {'FSRWithFAN' if use_fan else 'FSR'} (in_channels={generator_in_channels}, out_channels={generator_out_channels})")
        print(f"  - 判别器: NLayerDiscriminator (in_channels={generator_out_channels}, ndf={ndf}, n_layers={n_layers})")

    def forward(self, x):
        """
        前向传播仅通过生成器
        判别器在训练过程中单独使用
        """
        return self.generator(x)

    def initialize_weights(self):
        """
        初始化模型权重
        """
        # 对判别器进行权重初始化
        self.discriminator.apply(weights_init_normal)
        print("FSRGAN 判别器权重初始化完成")
class FSRWithFAN_old(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, n_feat=16, stage=3):
        super(FSRWithFAN, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        print(f'FSR: {in_channels} -> {n_feat}')
        modules_body = [MSTWithFAN(dim=8, stage=2, num_blocks=[1, 1, 1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        print(f'FSR: {n_feat} -> {n_feat}')
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        print(f'FSR: {n_feat} -> {out_channels}')
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        # print(h.shape)
        # print(x.shape)
        h += x
        return h[:, :, :h_inp, :w_inp]


class FSRWithFAN(nn.Module):
    def __init__(self, in_channels=2, out_channels=4, n_feat=8, stage=3):
        """
        Args:
            in_channels: 输入图像的通道数。
            out_channels: 输出图像的通道数。
            n_feat: conv_in 输出的特征通道数（同时也为 body 的通道数）。
            stage: body 层中 MSTWithFAN 模块的个数。
        """
        super(FSRWithFAN, self).__init__()
        self.stage = stage

        # ---------------------------
        # Enhanced conv_in 模块（可视为 encoder）
        # ---------------------------
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True)
        )
        print(f'FSR: {in_channels} -> {n_feat} (enhanced conv_in)')

        # ---------------------------
        # Body 层：多层 MSTWithFAN 模块
        # ---------------------------
        modules_body = [MSTWithFAN(dim=n_feat, stage=3, num_blocks=[1, 1, 1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        print(f'FSR: {n_feat} -> {n_feat} (body)')

        # ---------------------------
        # Enhanced conv_out 模块（可视为 decoder）
        # ---------------------------
        self.conv_out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=False)
        )
        print(f'FSR: {n_feat} -> {out_channels} (enhanced conv_out)')

        # ---------------------------
        # 投影层，用于将 conv_in 的输出投影到 out_channels 以匹配 skip connection
        # ---------------------------
        self.skip_conv = nn.Conv2d(n_feat, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 [B, C, H, W]
        Returns:
            输出张量，形状 [B, out_channels, H, W]
        """
        B, C, H_inp, W_inp = x.shape

        # 为保证空间尺寸满足后续要求，这里做了 pad 操作（使得 H,W 可被 8 整除）
        hb, wb = 8, 8
        pad_h = (hb - H_inp % hb) % hb
        pad_w = (wb - W_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # 特征提取（encoder）
        x_feat = self.conv_in(x)  # 输出形状: [B, n_feat, H, W]

        # 特征处理（body 层）
        h = self.body(x_feat)

        # 特征解码（decoder）
        h = self.conv_out(h)     # 输出形状: [B, out_channels, H, W]

        # 通过投影层对 skip 分支进行通道匹配，然后残差相加
        x_skip = self.skip_conv(x_feat)  # 将 [B, n_feat, H, W] 映射为 [B, out_channels, H, W]
        h = h + x_skip

        # 去掉 pad 部分，恢复原始尺寸
        h = h[:, :, :H_inp, :W_inp]
        return h
