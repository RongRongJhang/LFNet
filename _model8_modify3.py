import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from ecb import ECB  # 假設 ECB 已定義

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x * self.gamma + self.beta

class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=64):
        super(ShuffleAttention, self).__init__()
        self.channels = channels
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channels // (2 * groups), channels // (2 * groups))
        self._init_weights()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # 通道注意力
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # 空間注意力
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # 拼接並混洗
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out

    def _init_weights(self):
        init.kaiming_uniform_(self.cweight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.sweight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.cbias, 1)
        init.constant_(self.sbias, 1)

class MSEFBlock(nn.Module):
    def __init__(self, filters, groups=64):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attention = ShuffleAttention(filters, groups=groups)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.attention(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MSEFBlock(num_filters)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x), inplace=True)
        x2 = self.activation(self.conv2(x1), inplace=True)
        x3 = self.activation(self.conv3(x2), inplace=True)
        x = self.bottleneck(x3)
        x = self.up3(x)
        x = self.refine3(x + x2)
        x = self.up2(x)
        x = self.refine2(x + x1)
        return x
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LaaFNet(nn.Module):
    def __init__(self, filters=128, groups=64):  # 修改 filters 為 128
        super(LaaFNet, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.msef = MSEFBlock(filters, groups=groups)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters//2, 3, 3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):
        denoised = self.denoiser(inputs)
        enhanced = self.msef(denoised)
        output = self.final_conv(enhanced)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)