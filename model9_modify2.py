import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ecb import ECB

# model9_modify 去掉 U-Net 後的 EFBlock

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
    def __init__(self, channels, groups=8):  # 48/8=6, 6*2=12 (符合48通道)
        super(ShuffleAttention, self).__init__()
        
        self.groups = groups  # 使用8組，這樣每組會有6個通道(48/8=6)
        split_channels = channels // (2 * groups)  # 48/(2*8)=3
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention parameters (每組3個通道)
        self.cweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        
        # Spatial attention parameters (每組3個通道)
        self.sweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(split_channels, split_channels)
        
        # Initialize weights
        self._init_weights()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w) # flatten
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape and split channels (48->8組，每組6通道)
        x = x.reshape(b * self.groups, -1, h, w)  # [b*8, 6, h, w]
        x_0, x_1 = x.chunk(2, dim=1)  # 各[b*8, 3, h, w]
        
        # Channel attention
        xn = self.avg_pool(x_0)  # [b*8, 3, 1, 1]
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        
        # Spatial attention
        xs = self.gn(x_1)  # [b*8, 3, h, w]
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        
        # Concatenate and shuffle
        out = torch.cat([xn, xs], dim=1)  # [b*8, 6, h, w]
        out = out.reshape(b, -1, h, w)  # [b, 48, h, w]
        out = self.channel_shuffle(out, 2)  # 在兩個子組間shuffle
        
        return out

    def _init_weights(self):
        init.kaiming_uniform_(self.cweight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.sweight, a=0, mode='fan_in', nonlinearity='relu')

class EFBlock(nn.Module):
    def __init__(self, filters):
        super(EFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attention = ShuffleAttention(filters)
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
        # self.efblock = EFBlock(filters)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters//2, 3, 3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):
        denoised = self.denoiser(inputs)
        # enhanced = self.efblock(denoised)
        output = self.final_conv(denoised)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)