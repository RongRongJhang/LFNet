import os
import torch
from PIL import Image
from torchvision.utils import save_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def save_valid_output(model, valid_loader, device, output_folder):
    model.eval()
    with torch.no_grad():
        for low, high, name in valid_loader:
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)
            filename = name[0] if not name[0].endswith('.png') else name[0]
            save_path = os.path.join(output_folder, filename)
            save_image(output, save_path)
    torch.cuda.empty_cache()

def save_test_output(model, test_loader, device, output_folder):
    model.eval()
    with torch.no_grad():
        for low, name in test_loader:
            low = low.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)
            # filename = name[0] if is_image_file(name[0]) else None
            filename = name[0] if not name[0].endswith('.png') else name[0]
            save_path = os.path.join(output_folder, filename)
            save_image(output, save_path)