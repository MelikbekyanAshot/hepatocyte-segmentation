"""
File contains utility functions to get dataloaders and config file.
"""
import os
from typing import Tuple, Dict, List, Optional

import albumentations as A
import yaml
from loguru import logger
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


def get_dataloaders_from_folders(
        train_folders: List[str], val_folders: List[str], test_folders: List[str],
        root_path: str, patches_path: str, batch_size: int, train_transforms: Optional[A.Compose]) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split image-mask patches into 2 groups: train and val.
    Data directories structure:
        root_dir:
            code_number:
                full_sample - contains full sized image, mask and txt file with presented labels.
                json - json file with labels.
                patches:
                    images - image tiles.
                    masks - mask tiles.

    Args:
        train_folders (List[str]) - paths to folders for train.
        val_folders (List[str]) - paths to folders for val.
        test_folders (List[str]) - paths to folders for test.
        root_path (str) - path to directory with data.
        patches_path (str) - name of directory with patches (for experiments purpose).
        batch_size (int) - number of samples in batch.
        train_transforms (A.Compose) - augmentations for train dataloader.

    Returns:
        train_dataloader, val_dataloader, test_dataloader (DataLoader, DataLoader, DataLoader) - dataloaders.
    """
    num_workers = 0 if os.name == 'nt' else os.cpu_count() - 1  # set 0 for Windows

    train_img_patches, train_mask_patches = [], []
    for folder in train_folders:
        image_dir = os.path.join(root_path, folder, patches_path, 'images')
        mask_dir = os.path.join(root_path, folder, patches_path, 'masks')
        for patch_path in os.listdir(image_dir):
            train_img_patches.append(os.path.join(image_dir, patch_path))
        for patch_path in os.listdir(mask_dir):
            train_mask_patches.append(os.path.join(mask_dir, patch_path))
    train_img_patches = sorted(train_img_patches)
    train_mask_patches = sorted(train_mask_patches)
    train_patch_ds = PatchDataset(
        image_paths=train_img_patches, label_paths=train_mask_patches, transform=train_transforms)
    train_dataloader = DataLoader(
        train_patch_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)

    val_img_patches, val_mask_patches = [], []
    for folder in val_folders:
        image_dir = os.path.join(root_path, folder, patches_path, 'images')
        mask_dir = os.path.join(root_path, folder, patches_path, 'masks')
        for patch_path in os.listdir(image_dir):
            val_img_patches.append(os.path.join(image_dir, patch_path))
        for patch_path in os.listdir(mask_dir):
            val_mask_patches.append(os.path.join(mask_dir, patch_path))
    val_img_patches = sorted(val_img_patches)
    val_mask_patches = sorted(val_mask_patches)
    val_patch_ds = PatchDataset(
        image_paths=val_img_patches, label_paths=val_mask_patches)
    val_dataloader = DataLoader(
        val_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True)

    test_img_patches, test_mask_patches = [], []
    for folder in test_folders:
        image_dir = os.path.join(root_path, folder, patches_path, 'images')
        mask_dir = os.path.join(root_path, folder, patches_path, 'masks')
        for patch_path in os.listdir(image_dir):
            test_img_patches.append(os.path.join(image_dir, patch_path))
        for patch_path in os.listdir(mask_dir):
            test_mask_patches.append(os.path.join(mask_dir, patch_path))
    test_img_patches = sorted(test_img_patches)
    test_mask_patches = sorted(test_mask_patches)
    test_patch_ds = PatchDataset(
        image_paths=test_img_patches, label_paths=test_mask_patches)
    test_dataloader = DataLoader(
        test_patch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True)

    logger.info(f"\nTrain samples: {len(train_patch_ds)}"
                f"\nVal samples: {len(val_patch_ds)}"
                f"\nTest samples: {len(test_patch_ds)}")

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    root = 'D:\\Hepatocyte_full'
    folders = set(os.listdir(root))
    train_folders = ['7939_20_310320201319_7', '7939_20_310320201319_16']
    val_folders = ['7939_20_310320201319_4']
    test_folders = ['7939_20_310320201319_3']

    transforms = A.Compose([
        A.RGBShift(),
        A.Blur(),
        A.GaussNoise(),
        A.Flip(),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomSizedCrop(min_max_height=(512 - 100, 512 + 100), width=256, height=256, p=0.5)
    ])

    train_dl, val_dl, test_dl = get_dataloaders_from_folders(
        train_folders=list(train_folders),
        val_folders=list(val_folders),
        test_folders=list(test_folders),
        root_path=root,
        patches_path='patches',
        batch_size=4,
        train_transforms=transforms
    )

    iterator = iter(val_dl)
    while True:
        batch = next(iterator)
        plot_batch(batch)
