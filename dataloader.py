import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from options import option

opt = option().parse_args()

def transform1():
    return Compose([ToTensor()])

def transform2(size):
    return Compose([
        RandomCrop((size, size)),
        ToTensor(),
    ])

# Data augmentation
def transform3(size):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])
        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        seed = np.random.randint(1111)
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed)
            low_image = self.transform(low_image)
            random.seed(seed)
            torch.manual_seed(seed)         
            high_image = self.transform(high_image)

        return low_image, high_image, self.low_images[idx]


class UnpairedDataset(Dataset):
    def __init__(self, low_dir, transform=None):
        self.low_dir = low_dir
        self.transform = transform
        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])

    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        low_image = Image.open(low_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            factor = 8
            h, w = low_image.shape[1], low_image.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            low_image = F.pad(low_image.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)

        return low_image, self.low_images[idx]

def create_paired_dataloaders(train_low, train_high, valid_low, valid_high, crop_size, batch_size, num_workers):
    train_loader = None
    valid_loader = None
    
    if train_low and train_high:
        # train_dataset = PairedDataset(train_low, train_high, transform=transform2(crop_size))
        train_dataset = PairedDataset(train_low, train_high, transform=transform3(crop_size))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if valid_low and valid_high:
        valid_dataset = PairedDataset(valid_low, valid_high, transform=transform1())
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader


def create_unpaired_dataloaders(test_low):
    test_loader = None

    if test_low:
        test_dataset = UnpairedDataset(test_low, transform=transform1())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return test_loader