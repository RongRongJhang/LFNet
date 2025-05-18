import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from niqe_utils import *
import argparse
from options import option

opt = option().parse_args()

def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        
        im1 = Image.open(item).convert('RGB')
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        
        avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

if __name__ == '__main__':

    if opt.DICM:
        im_dir = opt.results_folder + 'testing/output/DICM/*.jpg'

    elif opt.LIME:
        im_dir = opt.results_folder + 'testing/output/LIME/*.bmp'

    elif opt.MEF:
        im_dir = opt.results_folder + 'testing/output/MEF/*.png'

    elif opt.NPE:
        im_dir = opt.results_folder + 'testing/output/NPE/*.jpg'

    elif opt.VV:
        im_dir = opt.results_folder + 'testing/output/VV/*.jpg'


    avg_niqe, avg_brisque = metrics(im_dir)
    print(avg_niqe)
    print(avg_brisque)