import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
from options import option

opt = option().parse_args()

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

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        
        self.alpha1 = opt.smooth_weight  # Smooth L1
        self.alpha2 = opt.perc_weight    # 感知損失(降低)
        self.alpha3 = opt.ssim_weight    # MS-SSIM
        self.alpha4 = opt.psnr_weight    # PSNR

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l + 
                      self.alpha2 * perc_l + 
                      self.alpha3 * ms_ssim_l + 
                      self.alpha4 * psnr_l)
        
        return torch.mean(total_loss)