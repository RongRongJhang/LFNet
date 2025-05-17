import os
import time
import torch
from datetime import datetime
from options import option
from model import LaaFNet
from util import save_valid_output
from eval.measure import metrics
from dataloader import create_paired_dataloaders

'''
訓練完後驗證 the best PSNR, SSIM and LPIPS 的結果
'''
# e.g.
# python valid.py --lol_v1 --best_PSNR

opt = option().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if opt.lol_v1:
        valid_low = opt.data_val_lol_v1
        valid_high = opt.data_valgt_lol_v1
        weights_path = opt.results_folder + 'LOLdataset/train/checkpoints/'
        output_folder = opt.results_folder + 'LOLdataset/valid/output/'
        label_folder = opt.data_valgt_lol_v1
        file_path = opt.results_folder + 'LOLdataset/valid/metrics.md'
    elif opt.lolv2_real:
        valid_low = opt.data_val_lolv2_real
        valid_high = opt.data_valgt_lolv2_real
        weights_path = opt.results_folder + 'LOLv2_real/train/checkpoints/'
        output_folder = opt.results_folder + 'LOLv2_real/valid/output/'
        label_folder = opt.data_valgt_lolv2_real
        file_path = opt.results_folder + 'LOLv2_real/valid/metrics.md'
    elif opt.lolv2_syn:
        valid_low = opt.data_val_lolv2_syn
        valid_high = opt.data_valgt_lolv2_syn
        weights_path = opt.results_folder + 'LOLv2_syn/train/checkpoints/'
        output_folder = opt.results_folder + 'LOLv2_syn/valid/output/'
        label_folder = opt.data_valgt_lolv2_syn
        file_path = opt.results_folder + 'LOLv2_syn/valid/metrics.md'
    
    if opt.best_PSNR:
        weights_path = weights_path + 'best_psnr_model.pth'
    elif opt.best_SSIM:
        weights_path = weights_path + 'best_ssim_model.pth'
    elif opt.best_LPIPS:
        weights_path = weights_path + 'best_lpips_model.pth'

    _, valid_loader = create_paired_dataloaders(None, None, valid_low, valid_high, crop_size=None, batch_size=1, num_workers=1)

    print(f'Valid loader: {len(valid_loader)}')
    
    model = LaaFNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    save_valid_output(model, valid_loader, device, output_folder)

    avg_psnr, avg_ssim, avg_lpips = metrics(output_folder + '*.png', label_folder, use_GT_mean=True)
    print(f'Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')

    os.environ['TZ']='Asia/Taipei'  
    time.tzset()
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("| Timestemp | PSNR | SSIM | LPIPS |\n")
            f.write("|-----------|------|------|-------|\n")
        
        f.write(f"| {now} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")


if __name__ == '__main__':  
    main()