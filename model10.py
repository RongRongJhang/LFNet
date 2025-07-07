import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# U-Net

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def forward(self, x):
        return self.double_conv(x)

    def _init_weights(self):
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        return self.conv(x)

    def _init_weights(self):
        init.kaiming_normal_(self.conv.weight, a=0.2, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

class LaaFNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(LaaFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output

    def _init_weights(self):
        # 權重初始化已在各模塊中處理（DoubleConv 和 OutConv）
        pass

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint_sequential(self.inc, segments=1)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential(self.down1, segments=1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential(self.down2, segments=1)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential(self.down3, segments=1)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential(self.down4, segments=1)
        self.up1 = torch.utils.checkpoint.checkpoint_sequential(self.up1, segments=1)
        self.up2 = torch.utils.checkpoint.checkpoint_sequential(self.up2, segments=1)
        self.up3 = torch.utils.checkpoint.checkpoint_sequential(self.up3, segments=1)
        self.up4 = torch.utils.checkpoint.checkpoint_sequential(self.up4, segments=1)
        self.outc = torch.utils.checkpoint.checkpoint_sequential(self.outc, segments=1)