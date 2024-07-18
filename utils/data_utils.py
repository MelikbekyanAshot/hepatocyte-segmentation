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


def get_dataloaders(
        root_path: str, batch_size: int,
        train_size: float = 0.8, val_size: float = 0.2) \
        -> Tuple[DataLoader, DataLoader]:
    """Split image-mask patches into 2 groups: train and val.
    Data directories structure:
        sample_number:
            full_sample - contains full sized image, mask and txt file with presented labels.
            json - json file with labels.
            patches:
                images - image tiles.
                masks - mask tiles.
    """
    assert round(train_size + val_size, 2) == 1.0, \
        f"Wrong split coefficients: {train_size + val_size} is not equal to 1.0"
    sample_folders = os.listdir(root_path)

    # mask_patch_paths = []
    # for folder in sample_folders:
    #     cur_dir = os.path.join(root_path, folder, 'patches', 'masks')
    #     cur_paths = []
    #     for mask_patch_path in os.listdir(cur_dir):
    #         cur_paths.append(os.path.join(cur_dir, mask_patch_path))
    #     cur_paths.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    #     mask_patch_paths.extend(cur_paths)
    #
    # image_patch_paths = []
    # for folder in sample_folders:
    #     cur_dir = os.path.join(root_path, folder, 'patches', 'images')
    #     cur_paths = []
    #     for image_patch_path in os.listdir(cur_dir):
    #         cur_paths.append(os.path.join(cur_dir, image_patch_path))
    #     cur_paths.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    #     image_patch_paths.extend(cur_paths)

    mask_patch_paths, image_patch_paths = [], []

    for folder in sample_folders:
        mask_dir = os.path.join(root_path, folder, 'patches', 'masks')
        image_dir = os.path.join(root_path, folder, 'patches', 'images')

        for patch_path in os.listdir(mask_dir):
            mask_patch_paths.append(os.path.join(mask_dir, patch_path))

        for patch_path in os.listdir(image_dir):
            image_patch_paths.append(os.path.join(image_dir, patch_path))

    train_images, val_images, train_masks, val_masks = \
        train_test_split(image_patch_paths, mask_patch_paths, test_size=0.2)

    transforms = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])
    train_patch_ds = PatchDataset(image_paths=train_images, label_paths=train_masks, transform=transforms)
    val_patch_ds = PatchDataset(image_paths=val_images, label_paths=val_masks)
    logger.info(f"\nTrain samples: {len(train_patch_ds)}"
                f"\nVal samples: {len(val_patch_ds)}")
    num_workers = 0 if os.name == 'nt' else os.cpu_count() - 1  # set 0 for Windows
    train_dataloader = DataLoader(
        train_patch_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(
        val_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True)
    return train_dataloader, val_dataloader


def split_mask(mask: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Split original 2D-mask array into multichannel array.
    Input size: [B, H, W], output size: [B, N, H, W].

    Args:
        mask - original 2D mask.
        n_channels - number of channels to extract.

    Returns:
        splitted_mask - multichannel mask with separate channel for every type of cell.
    """
    B, _, H, W = mask.size()
    splitted_mask = torch.zeros((B, n_channels, H, W), dtype=torch.float)

    for i in range(n_channels):
        splitted_mask[:, i] = (mask == i).squeeze() * i

    return splitted_mask


def merge_mask(mask: torch.Tensor):
    """Merge multichannel mask into 2D-array.
    Input size: [B, N, H, W], output size: [B, H, W].

    Args:
        mask (torch.Tensor) - multichannel mask.

    Returns:
        merged_msak (torch,Tensor) - 2D mask.
    """
    merged_mask = mask.sum(axis=1)
    return merged_mask


if __name__ == '__main__':
    data_loader, *_ = get_dataloaders(
        root_path='D:\\Hepatocyte\\patches\\steatosis_128\\', batch_size=4,
        train_size=1.0, val_size=0.0, test_size=0.0)
    batch = next(iter(data_loader))
    plot_batch(batch)
