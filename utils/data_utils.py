"""
File contains utility functions to get dataloaders and config file.
"""
import os
from typing import Tuple, Dict

import yaml
from albumentations import Compose, VerticalFlip, HorizontalFlip, RandomRotate90
from loguru import logger
import numpy as np
from torch.utils.data import DataLoader
from yaml import SafeLoader

from datasets.patch_dataset import PatchDataset
from utils.plot_utils import plot_batch


def get_config(path='config.yml') -> Dict:
    """Read config file from repository and return it."""
    with open(path, 'r') as f:
        config = yaml.load(f, SafeLoader)
    return config


def get_dataloaders(
        path_to_dir: str, batch_size: int,
        train_size: float = 0.7, val_size: float = 0.2, test_size: float = 0.1) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split image-mask patches into 3 groups: train, val and test. """
    assert round(train_size + val_size + test_size, 2) == 1.0, \
        f"Wrong split coefficients: {train_size + val_size + test_size} is not equal to 1.0"
    patch_masks_dir = os.path.join(path_to_dir, 'mask')
    mask_patches = [os.path.join(patch_masks_dir, file_name) for file_name in os.listdir(patch_masks_dir)]
    mask_patches.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    patch_image_dir = os.path.join(path_to_dir, 'image')
    image_patches = [os.path.join(patch_image_dir, file_name) for file_name in os.listdir(patch_image_dir)]
    image_patches.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    n = len(image_patches)
    train_number = int(train_size * n)
    val_number = int((train_size + val_size) * n)
    train_images, val_images, test_images = np.split(image_patches, [train_number, val_number])
    train_masks, val_masks, test_masks = np.split(mask_patches, [train_number, val_number])
    transforms = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])
    train_patch_ds = PatchDataset(image_paths=train_images, label_paths=train_masks, transform=transforms)
    val_patch_ds = PatchDataset(image_paths=val_images, label_paths=val_masks)
    test_patch_ds = PatchDataset(image_paths=test_images, label_paths=test_masks)
    logger.info(f"\nTrain samples: {len(train_patch_ds)}"
                f"\nVal samples: {len(val_patch_ds)}"
                f"\nTest samples: {len(test_patch_ds)}")
    train_dataloader = DataLoader(
        train_patch_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True)
    val_dataloader = DataLoader(
        val_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=True)
    test_dataloader = DataLoader(
        test_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, *_ = get_dataloaders(
        path_to_dir='D:\\Hepatocyte\\patches\\steatosis_128\\', batch_size=4,
        train_size=1.0, val_size=0.0, test_size=0.0)
    batch = next(iter(data_loader))
    plot_batch(batch)
