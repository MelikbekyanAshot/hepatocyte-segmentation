"""
File contains utility functions to get dataloaders and config file.
"""
import os
from typing import Tuple, Dict

import torch
import yaml
from albumentations import Compose, VerticalFlip, HorizontalFlip, RandomRotate90
from loguru import logger
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from yaml import SafeLoader

from datasets.patch_dataset import PatchDataset
from utils.plot_utils import plot_batch


def get_config(path='config.yml') -> Dict:
    """Read config file from repository and return it."""
    with open(path, 'r') as f:
        config = yaml.load(f, SafeLoader)
    return config


def dump_config(config: Dict, path: str):
    """Create or update config file."""
    with open(path, 'w') as f:
        yaml.dump(config, f)


def get_dataloaders(root_path: str, batch_size: int, val_size: float = 0.2) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split image-mask patches into 2 groups: train and val.
    Data directories structure:
        sample_number:
            full_sample - contains full sized image, mask and txt file with presented labels.
            json - json file with labels.
            patches:
                images - image tiles.
                masks - mask tiles.
    """
    sample_folders = os.listdir(root_path)
    train_val_mask_patch_paths, train_val_image_patch_paths = [], []
    test_mask_patch_paths, test_image_patch_paths = [], []

    for folder in sample_folders:
        mask_dir = os.path.join(root_path, folder, 'patches', 'masks')
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

        image_dir = os.path.join(root_path, folder, 'patches', 'images')
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        if folder == '7939_20_310320201319_7':
            # This is temporally crunch for testing
            for patch_path in os.listdir(mask_dir):
                test_mask_patch_paths.append(os.path.join(mask_dir, patch_path))
            for patch_path in os.listdir(image_dir):
                test_image_patch_paths.append(os.path.join(image_dir, patch_path))
        else:
            for patch_path in os.listdir(mask_dir):
                train_val_mask_patch_paths.append(os.path.join(mask_dir, patch_path))

            for patch_path in os.listdir(image_dir):
                train_val_image_patch_paths.append(os.path.join(image_dir, patch_path))

    train_val_mask_patch_paths = sorted(train_val_mask_patch_paths)
    train_val_image_patch_paths = sorted(train_val_image_patch_paths)

    test_mask_patch_paths = sorted(test_mask_patch_paths)
    test_image_patch_paths = sorted(test_image_patch_paths)

    train_images, val_images, train_masks, val_masks = \
        train_test_split(train_val_image_patch_paths, train_val_mask_patch_paths, test_size=val_size)

    transforms = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])
    train_patch_ds = PatchDataset(image_paths=train_images, label_paths=train_masks, transform=transforms)
    val_patch_ds = PatchDataset(image_paths=val_images, label_paths=val_masks)
    test_patch_ds = PatchDataset(image_paths=test_image_patch_paths, label_paths=test_mask_patch_paths)
    logger.info(f"\nTrain samples: {len(train_patch_ds)}"
                f"\nVal samples: {len(val_patch_ds)}"
                f"\nTest samples: {len(test_patch_ds)}")
    num_workers = 0 if os.name == 'nt' else os.cpu_count() - 1  # set 0 for Windows
    train_dataloader = DataLoader(
        train_patch_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(
        val_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(
        test_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, *_ = get_dataloaders(root_path='D:\\Hepatocyte', batch_size=16, val_size=0.1)
    batch = next(iter(data_loader))
    plot_batch(batch)
