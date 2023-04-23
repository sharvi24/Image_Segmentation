import os
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SeepDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(self.root_dir, 'train_images_256')
        self.masks_dir = os.path.join(self.root_dir, 'train_masks_256')
        self.images = os.listdir(self.images_dir)
        self.masks = os.listdir(self.masks_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        mask = transform(mask)
        return image, mask
