import glob
import torch
from tqdm import tqdm
from PIL import Image
# import imquality.brisque as brisque
# from niqe_utils import *
from options import option
import pyiqa

opt = option().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        
        im1 = Image.open(item).convert('RGB')

        niqe_metric = pyiqa.create_metric('niqe').to(device)
        brisquqe_metric = pyiqa.create_metric('brisque').to(device)

        niqe_score = niqe_metric(im1)
        brisquqe_score = brisquqe_metric(im1)
        
        avg_niqe += niqe_score
        avg_brisque += brisquqe_score

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

# def metrics(im_dir):
#     avg_niqe = 0
#     n = 0
#     avg_brisque = 0
        
#     for item in tqdm(sorted(glob.glob(im_dir))):
#         n += 1
        
#         im1 = Image.open(item).convert('RGB')
#         score_brisque = brisque.score(im1) 
#         im1 = np.array(im1)
#         score_niqe = calculate_niqe(im1)
        
#         avg_brisque += score_brisque
#         avg_niqe += score_niqe

#         torch.cuda.empty_cache()
    
#     avg_brisque = avg_brisque / n
#     avg_niqe = avg_niqe / n
#     return avg_niqe, avg_brisque