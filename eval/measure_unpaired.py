import glob
import torch
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from niqe_utils import *
from options import option
import numpy as np
import os
# import pyiqa

opt = option().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def metrics(im_dir):
#     avg_niqe = 0
#     n = 0
#     avg_brisque = 0
        
#     for item in tqdm(sorted(glob.glob(im_dir))):
#         n += 1
        
#         im1 = Image.open(item).convert('RGB')

#         niqe_metric = pyiqa.create_metric('niqe').to(device)
#         brisquqe_metric = pyiqa.create_metric('brisque').to(device)

#         niqe_score = niqe_metric(im1)
#         brisquqe_score = brisquqe_metric(im1)
        
#         avg_niqe += niqe_score
#         avg_brisque += brisquqe_score

#         torch.cuda.empty_cache()
    
#     avg_brisque = avg_brisque / n
#     avg_niqe = avg_niqe / n
#     return avg_niqe, avg_brisque

def calculate_loe(img_low, img_enhanced, downsample_size=50):
    # 將輸入圖像轉換為 uint8 格式
    img1 = img_low.astype(np.uint8)
    img2 = img_enhanced.astype(np.uint8)

    # Step 1: 計算亮度(取 RGB 三通道最大值)
    img1_r, img1_g, img1_b = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    img1_l = np.maximum(np.maximum(img1_r, img1_g), img1_b)
    
    img2_r, img2_g, img2_b = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    img2_l = np.maximum(np.maximum(img2_r, img2_g), img2_b)

    # Step 2: 獲取圖像尺寸並計算縮放比例
    m, n = img1_l.shape
    ratio = downsample_size / min(m, n)
    M = int(round(m * ratio))
    N = int(round(n * ratio))

    # Step 3: 下採樣以降低計算複雜度
    # 在行和列方向均勻採樣
    sample_rows = np.round(np.linspace(0, m - 1, M)).astype(int)
    sample_cols = np.round(np.linspace(0, n - 1, N)).astype(int)

    # 對亮度圖像進行下採樣
    img1_l_ds = img1_l[sample_rows][:, sample_cols]
    img2_l_ds = img2_l[sample_rows][:, sample_cols]

    # 計算 LOE
    error = 0
    for i in range(M):
        for j in range(N):
            map_img1_order = img1_l_ds >= img1_l_ds[i, j]
            map_img2_order = img2_l_ds >= img2_l_ds[i, j]
            map_error = np.logical_xor(map_img1_order, map_img2_order)
            error += np.sum(map_error)
    
    # 計算最終的 LOE 值
    loe = error / (M * N)
    
    return loe


def metrics(im_dir, test_low):
    n = 0
    avg_niqe = 0
    avg_brisque = 0
    avg_loe = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        name = os.path.basename(item)
        
        im1 = Image.open(item).convert('RGB')
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        
        avg_brisque += score_brisque
        avg_niqe += score_niqe

        im2 = Image.open(test_low + name).convert('RGB')
        im2 = np.array(im2)
        score_loe = calculate_loe(im2, im1)
        avg_loe += score_loe

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    avg_loe = avg_loe / n
    
    return avg_niqe, avg_brisque, avg_loe