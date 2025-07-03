import os
import time
import torch
from options import option
from datetime import datetime
from model import LaaFNet
from util import save_test_output
from eval.measure_unpaired import metrics as metrics_niqu
from dataloader import create_unpaired_dataloaders

'''
訓練完後用 Unpaired Datasets 來測試結果
'''
# e.g.
# python test.py --lol_v1 --best_PSNR --DICM

opt = option().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if opt.lol_v1:
        weights_path = opt.results_folder + 'LOLdataset/train/checkpoints/'
        file_path = opt.results_folder + 'LOLdataset/test/metrics.md'
        paired = 'LOLdataset/'
    elif opt.lolv2_real:
        weights_path = opt.results_folder + 'LOLv2_real/train/checkpoints/'
        file_path = opt.results_folder + 'LOLv2_real/test/metrics.md'
        paired = 'LOLv2_real/'
    elif opt.lolv2_syn:
        weights_path = opt.results_folder + 'LOLv2_syn/train/checkpoints/'
        file_path = opt.results_folder + 'LOLv2_syn/test/metrics.md'
        paired = 'LOLv2_syn/'

    if opt.best_3_metrics:
        weights_path = weights_path + 'best_3_metrics.pth'
        metrics_dir = 'best_3_metrics/'
    elif opt.best_2_metrics:
        weights_path = weights_path + 'best_2_metrics.pth'
        metrics_dir = 'best_2_metrics/'
    elif opt.best_PSNR:
        weights_path = weights_path + 'best_psnr_model.pth'
        metrics_dir = 'best_PSNR/'
    elif opt.best_SSIM:
        weights_path = weights_path + 'best_ssim_model.pth'
        metrics_dir = 'best_SSIM/'
    elif opt.best_LPIPS:
        weights_path = weights_path + 'best_lpips_model.pth'
        metrics_dir = 'best_LPIPS/'

    if opt.DICM:
        test_low = opt.data_DICM
        unpaired = 'DICM/'
        fnex = '*.jpg'
    elif opt.LIME:
        test_low = opt.data_LIME
        unpaired = 'LIME/'
        fnex = '*.bmp'
    elif opt.MEF:
        test_low = opt.data_MEF
        unpaired = 'MEF/'
        fnex = '*.png'
    elif opt.NPE:
        test_low = opt.data_NPE
        unpaired = 'NPE/'
        fnex = '*.jpg'
    elif opt.VV:
        test_low = opt.data_VV
        unpaired = 'VV/'
        fnex = '*.jpg'
    
    output_folder = opt.results_folder + paired + 'test/output/' + metrics_dir + unpaired

    # os.makedirs 可以遞迴地建立多層資料夾, exist_ok=True 確保即使資料夾已經存在，程式也不會報錯。
    os.makedirs(output_folder, exist_ok=True)

    test_loader = create_unpaired_dataloaders(test_low)

    print(f'Test loader: {len(test_loader)}')

    model = LaaFNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    model.eval()
    save_test_output(model, test_loader, device, output_folder)

    avg_niqe, avg_brisque, avg_loe = metrics_niqu(output_folder + fnex, test_low + '/')
    # print(f'Test NIQE: {avg_niqe.item():.4f}, BRISQUE: {avg_brisque.item():.4f}')
    print(f'Test NIQE: {avg_niqe:.4f}, BRISQUE: {avg_brisque:.4f}, LOE:  {avg_loe:.4f}')

    os.environ['TZ']='Asia/Taipei'  
    time.tzset()
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("| Timestemp |   NIQE   |   BRISQUE   |    LOE    |\n")
            f.write("|-----------|----------|-------------|-----------|\n")
        
        # f.write(f"| {now} | {avg_niqe.item():.4f} | {avg_brisque.item():.4f} |\n")
        f.write(f"| {now} | {avg_niqe:.4f} | {avg_brisque:.4f} | {avg_loe:.4f} |\n")

if __name__ == '__main__':
    main()