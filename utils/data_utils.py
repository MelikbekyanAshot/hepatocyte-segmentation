import os

import yaml
from yaml import SafeLoader

from albumentations import Compose, VerticalFlip, HorizontalFlip, RandomRotate90
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.patch_dataset import PatchDataset


def get_dataloaders(config):
    patch_masks_dir = config['PATH']['PATCHES']['STEATOSIS']['MASK']
    mask_patches = [os.path.join(patch_masks_dir, file_name) for file_name in os.listdir(patch_masks_dir)]
    mask_patches.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    patch_image_dir = config['PATH']['PATCHES']['STEATOSIS']['IMAGE']
    image_patches = [os.path.join(patch_image_dir, file_name) for file_name in os.listdir(patch_image_dir)]
    image_patches.sort(key=lambda p: int(p[p.rfind('_') + 1:p.rfind('.')]))
    train_images, val_images, train_masks, val_masks = \
        train_test_split(image_patches, mask_patches, test_size=0.2)
    transforms = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])
    train_patch_ds = PatchDataset(image_paths=train_images, label_paths=train_masks, transform=transforms)
    val_patch_ds = PatchDataset(image_paths=val_images, label_paths=val_masks)
    train_dataloader = DataLoader(
        train_patch_ds, batch_size=config['TRAIN']['BATCH_SIZE'], shuffle=True,
        num_workers=0, drop_last=True)
    val_dataloader = DataLoader(
        val_patch_ds, batch_size=config['TRAIN']['BATCH_SIZE'], shuffle=False,
        num_workers=0, drop_last=True)
    return train_dataloader, val_dataloader


def get_config():
    with open('config.yml', 'r') as f:
        config = yaml.load(f, SafeLoader)
    return config
