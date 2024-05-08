import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Class to handle patchified data."""
    def __init__(self, image_paths, label_paths, transform=None):
        self.images = image_paths
        self.masks = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask = Image.open(self.images[idx]), Image.open(self.masks[idx])
        img, mask = np.array(img), np.array(mask)
        if self.transform:
            aug_data = self.transform(image=img, mask=mask)
            img, mask = aug_data['image'], aug_data['mask']
        img, mask = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(-1).permute(2, 0, 1)
        return img, mask
