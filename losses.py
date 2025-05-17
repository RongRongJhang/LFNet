import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.fft import fft2, fftshift
from pytorch_msssim import ms_ssim
from options import option

# loss3_modify4

opt = option().parse_args()

class EnhancedColorBrightnessLoss(nn.Module):
    def __init__(self):
        super(EnhancedColorBrightnessLoss, self).__init__()
        
    def forward(self, output, target):
        # 色彩豐富度指標
        output_rgb = output.permute(0, 2, 3, 1)
        rg = output_rgb[..., 0] - output_rgb[..., 1]
        yb = 0.5 * (output_rgb[..., 0] + output_rgb[..., 1]) - output_rgb[..., 2]
        std_rg = torch.std(rg, dim=[1, 2])
        std_yb = torch.std(yb, dim=[1, 2])
        colorfulness = torch.sqrt(std_rg**2 + std_yb**2)
        color_loss = -colorfulness.mean()  # 我們希望最大化色彩豐富度
        
        # 亮度保持
        brightness_diff = torch.abs(output.mean() - target.mean())
        
        return 0.1 * color_loss + 0.05 * brightness_diff

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36]  # 使用更深層特徵
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))  # 加入 eps 避免除零
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_true_hist /= y_true_hist.sum() + 1e-8  # 避免除零
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()
        
    def forward(self, y_true, y_pred):
        # Compute FFT
        y_true_fft = fftshift(fft2(y_true, norm='ortho'))
        y_pred_fft = fftshift(fft2(y_pred, norm='ortho'))
        
        # Magnitude and phase loss
        mag_loss = F.l1_loss(torch.abs(y_true_fft), torch.abs(y_pred_fft))
        phase_loss = F.l1_loss(torch.angle(y_true_fft), torch.angle(y_pred_fft))
        
        return 0.3 * mag_loss + 0.1 * phase_loss

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.color_brightness_loss = EnhancedColorBrightnessLoss()
        self.freq_loss = FrequencyDomainLoss()
        
        self.alpha1 = opt.smooth_weight  # Smooth L1
        self.alpha2 = opt.perc_weight    # 感知損失(降低)
        self.alpha3 = opt.hist_weight    # 直方圖損失
        self.alpha4 = opt.ssim_weight    # MS-SSIM
        self.alpha5 = opt.psnr_weight    # PSNR
        self.alpha6 = opt.color_weight   # 色彩/亮度損失(提高)
        self.alpha7 = opt.freq_weight    # 頻域損失(新增)

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_bright_l = self.color_brightness_loss(y_true, y_pred)
        freq_l = self.freq_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l + 
                     self.alpha2 * perc_l + 
                     self.alpha3 * hist_l + 
                     self.alpha4 * ms_ssim_l + 
                     self.alpha5 * psnr_l + 
                     self.alpha6 * color_bright_l +
                     self.alpha7 * freq_l)
        
        return torch.mean(total_loss)