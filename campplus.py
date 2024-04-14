import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.cbam = CBAM(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(self.shortcut(x).shape)

        # out = self.cbam(out)
        out += self.shortcut(x)

        out = F.relu(out)
        return out
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()



        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, 8)

        self.downsample = downsample

        self.projection = nn.Sequential()

        self.projection = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(128),
            )
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        # residual = self.conv4(residual)
        if out.shape[1] != residual.shape[1]:
            residual = self.projection(residual)  # 假设projection是一个将通道数从32映射到128的1x1卷积层
        out += residual
        out = self.relu(out)
        # print(out.shape)
        # print('11111111')
        # print(residual.shape)
        shape2 = out.shape
        # out = out.reshape(shape2[0],shape2[2], shape2[3])

        return out






class SEBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.projection = nn.Sequential()

        self.projection = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(64),
        )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if out.shape[1] != residual.shape[1]:
            residual = self.projection(residual)  # 假设projection是一个将通道数从32映射到128的1x1卷积层

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y






class FCM(nn.Module):
    def __init__(self,
                 block=Bottle2neck,
                 num_blocks=[1, 2],
                 m_channels=32,
                 feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)#基础卷积为一维卷积，加了一个bn
        self.bn1 = nn.BatchNorm2d(m_channels)

        # self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=1)


        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=1)

        self.conv2 = nn.Conv2d(m_channels*4, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)#bottle2neck
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)



    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        # print('out')
        # print(out.shape)
        out = self.layer1(out)
        # shape1 = out.shape
        # out = out.reshape(shape1[0], shape1[1]*2, shape1[2])
        out = self.layer2(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 embd_dim=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=input_size)
        channels = self.head.out_channels
        self.embd_dim = embd_dim

        self.xvector = nn.Sequential(
            OrderedDict([('tdnn', TDNNLayer(channels,
                                            init_channels,
                                            5,
                                            stride=2,
                                            dilation=1,
                                            padding=-1,
                                            config_str=config_str)),
                         ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                # dilation) in enumerate(zip((8, 16, 10), (5, 5, 5), (1, 2, 2))):
                dilation) in enumerate(zip((4, 8, 3), (3, 5, 3), (1, 2, 2))):#pam=2
                # dilation) in enumerate(zip((6, 12, 3), (3, 5, 3), (1, 2, 2))):
                # dilation) in enumerate(zip((3, 4, 3), (3, 3, 3), (1, 2, 2))):#pam=1.5
                # dilation) in enumerate(zip((1, 1, 1), (3, 3, 3), (1, 2, 2))):#pam=1.07
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module('transit%d' % (i + 1),
                                    TransitLayer(channels,
                                                 channels // 2,
                                                 bias=False,
                                                 config_str=config_str))
            channels //= 2

        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module('dense', DenseLayer(channels * 2, embd_dim, config_str='batchnorm_'))
        # 分类层
        self.fc = nn.Linear(embd_dim, num_class)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # print('xxxxxx')
        # print(x.shape)
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        x = self.fc(x)
        return x
class CBAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CBAM, self).__init__()

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc2(self.relu(self.fc1(x)))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)  # 添加通道注意力

        # 空间注意力
        spatial_avg_out = torch.mean(channel_attention, dim=1, keepdim=True)
        spatial_max_out, _ = torch.max(channel_attention, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([spatial_avg_out, spatial_max_out], dim=1)))  # 添加空间注意力

        # 特征图与注意力相乘
        out = x * channel_attention * spatial_attention

        return out
