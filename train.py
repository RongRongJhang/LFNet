import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import LaaFNet
from losses import CombinedLoss
from options import option
from util import save_valid_output
from eval.measure import metrics
from dataloader import create_paired_dataloaders

# e.g.
# python train.py --lol_v1

opt = option().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

def main():
    train_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.lol_v1:
        train_low = opt.data_train_lol_v1
        train_high = opt.data_traingt_lol_v1
        valid_low = opt.data_val_lol_v1
        valid_high = opt.data_valgt_lol_v1
    if opt.lolv2_real:
        train_low = opt.data_train_lolv2_real
        train_high = opt.data_traingt_lolv2_real
        valid_low = opt.data_val_lolv2_real
        valid_high = opt.data_valgt_lolv2_real
    if opt.lolv2_syn:
        train_low = opt.data_train_lolv2_syn
        train_high = opt.data_traingt_lolv2_syn
        valid_low = opt.data_val_lolv2_syn
        valid_high = opt.data_valgt_lolv2_syn

    train_loader, valid_loader = create_paired_dataloaders(train_low, train_high, valid_low, valid_high, 
                                                           crop_size=opt.cropSize, batch_size=opt.batchSize, num_workers=opt.threads)
    
    print(f'Train loader: {len(train_loader)}; Valid loader: {len(valid_loader)}')

    model = LaaFNet().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.nEpochs)
    scaler = GradScaler(enabled=opt.gpu_mode)

    best_psnr = 0
    best_ssim = 0
    best_lpips = 1
    
    print('Training started.')
    for epoch in range(opt.nEpochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        if opt.lol_v1:
            output_folder = opt.results_folder + 'LOLdataset/train/output/'
            label_folder = opt.data_valgt_lol_v1
            weights_path = opt.results_folder + 'LOLdataset/train/checkpoints/'
            file_path = opt.results_folder + 'LOLdataset/train/metrics.md'
        if opt.lolv2_real:
            output_folder = opt.results_folder + 'LOLv2_real/train/output/'
            label_folder = opt.data_valgt_lolv2_real
            weights_path = opt.results_folder + 'LOLv2_real/train/checkpoints/'
            file_path = opt.results_folder + 'LOLv2_real/train/metrics.md'
        if opt.lolv2_syn:
            output_folder = opt.results_folder + 'LOLv2_syn/train/output/'
            label_folder = opt.data_valgt_lolv2_syn
            weights_path = opt.results_folder + 'LOLv2_syn/train/checkpoints/'
            file_path = opt.results_folder + 'LOLv2_syn/train/metrics.md'

        save_valid_output(model, valid_loader, device, output_folder)

        # GT mean 表示 GroundTruth 圖像做過 Gamma 校正
        avg_psnr, avg_ssim, avg_lpips = metrics(output_folder + '*.png', label_folder, use_GT_mean=True)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{opt.nEpochs}, LR: {current_lr:.4f}, Loss: {avg_train_loss:.6f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')

        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), weights_path + 'best_psnr_model.pth')
            print(f'Saving model with PSNR: {best_psnr:.4f}')

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(model.state_dict(), weights_path + 'best_ssim_model.pth')
            print(f'Saving model with SSIM: {best_ssim:.4f}')
        
        if avg_lpips < best_lpips:
            best_lpips = avg_lpips
            torch.save(model.state_dict(), weights_path + 'best_lpips_model.pth')
            print(f'Saving model with LPIPS: {best_lpips:.4f}')
        

        os.environ['TZ']='Asia/Taipei'  
        time.tzset()
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        file_exists = os.path.exists(file_path)

        with open(file_path, "a") as f:
            if not file_exists:
                f.write("|   Timestemp   |   Epoch   |    LR    |   Loss   |   PSNR   |   SSIM   |   LPIPS   |\n")
                f.write("|---------------|-----------|----------|----------|----------|----------|-----------|\n")
            
            f.write(f"|   {now}   | {epoch + 1} | {current_lr:.4f} | {avg_train_loss:.6f} |  {avg_psnr:.3f}  |  {avg_ssim:.3f}  |  {avg_lpips:.3f}  |\n")


if __name__ == '__main__':  
    main()