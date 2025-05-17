import argparse

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='LaaFNet')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')

    # train input
    parser.add_argument('--data_train_lol_v1'       , type=str, default='data/LOLdataset/our485/low')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='data/LOLv2/Real_captured/Train/Low')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='data/LOLv2/Synthetic/Train/Low')

    # train ground truth
    parser.add_argument('--data_traingt_lol_v1'       , type=str, default='data/LOLdataset/our485/high')
    parser.add_argument('--data_traingt_lolv2_real'   , type=str, default='data/LOLv2/Real_captured/Train/Normal')
    parser.add_argument('--data_traingt_lolv2_syn'    , type=str, default='data/LOLv2/Synthetic/Train/Normal')

    # validation input
    parser.add_argument('--data_val_lol_v1'         , type=str, default='data/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='data/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='data/LOLv2/Synthetic/Test/Low')

    # validation ground truth
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='data/LOLdataset/eval15/high')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='data/LOLv2/Real_captured/Test/Normal')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='data/LOLv2/Synthetic/Test/Normal')

    # unpaired datasets for test
    parser.add_argument('--data_DICM'  , type=str, default='data/DICM')
    parser.add_argument('--data_LIME'  , type=str, default='data/LIME')
    parser.add_argument('--data_MEF'   , type=str, default='data/MEF')
    parser.add_argument('--data_NPE'   , type=str, default='data/NPE')
    parser.add_argument('--data_VV'    , type=str, default='data/VV')

    # loss weights
    parser.add_argument('--smooth_weight', type=float, default=1.0)
    parser.add_argument('--perc_weight', type=float, default=0.1)
    parser.add_argument('--hist_weight',  type=float, default=0.05)
    parser.add_argument('--ssim_weight',  type=float, default=0.8)
    parser.add_argument('--psnr_weight',  type=float, default=0.005)
    parser.add_argument('--color_weight',  type=float, default=0.15)
    parser.add_argument('--freq_weight',  type=float, default=0.05)

    # best model
    parser.add_argument('--best_PSNR', action='store_true', help='output dataset best_PSNR')
    parser.add_argument('--best_SSIM', action='store_true', help='output dataset best_SSIM')
    parser.add_argument('--best_LPIPS', action='store_true', help='output dataset best_LPIPS')

    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default=True)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)

    # choose which dataset you want to test, please only set one "True"
    parser.add_argument('--DICM', type=bool, default=True)
    parser.add_argument('--LIME', type=bool, default=False)
    parser.add_argument('--MEF', type=bool, default=False)
    parser.add_argument('--NPE', type=bool, default=False)
    parser.add_argument('--VV', type=bool, default=False)

    # Google Drive results folder
    parser.add_argument('--results_folder', default='/content/drive/MyDrive/LFNet/results/', help='Location to save results')

    return parser