import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ecb import ECB

# model7_modify5_fix2 (只留 denoiser + msef)

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

class TruncatedShuffleAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(TruncatedShuffleAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.group_channels = channels // reduction
        
        self.groups = max(1, channels // self.group_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.weight = nn.Parameter(torch.zeros(1, self.group_channels, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.group_channels, 1, 1))
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        y = self.avg_pool(x)
        y = y.reshape(b * self.groups, self.group_channels, 1, 1)
        y = self.sigmoid(y * self.weight + self.bias)
        y = y.reshape(b, c, 1, 1)
        
        out = x * y
        
        return out

    def _init_weights(self):
        init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')

class EFBlock(nn.Module):
    def __init__(self, filters):
        super(EFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attention = TruncatedShuffleAttention(filters)
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
        self.bottleneck = EFBlock(num_filters)
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
    def __init__(self, filters=48):
        super(LaaFNet, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.efblock = EFBlock(filters)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters//2, 3, 3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):
        denoised = self.denoiser(inputs)
        enhanced = self.efblock(denoised)
        output = self.final_conv(enhanced)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)