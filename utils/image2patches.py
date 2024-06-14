import os
from typing import List, Dict

import numpy as np
from PIL import Image
from patchify import patchify
from torch.utils.data import Dataset
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None


class FullImageDataset(Dataset):
    """Class to handle full images."""
    def __init__(self, image_paths: List[str], mask_paths: List[str], label_paths: List[str], sample_names: List[str],
                 patch_width: int, patch_height: int, patch_step: int, label2idx: Dict):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.label_paths = label_paths
        self.sample_names = sample_names
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_step = patch_step
        self.label2idx = label2idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img, mask = Image.open(self.image_paths[idx]), Image.open(self.mask_paths[idx])  # [10K, 10K], [10K, 10K]
        img, mask = np.array(img), np.array(mask)  # [10K, 10K, 3], [10K, 10K]
        img, mask = self.__crop(img, mask)  # [<10K, <10K, 3], [<10K, <10K]
        mask = self.__replace_values(mask, idx)  # [<10K, <10K, 3], [<10K, <10K]
        img, mask = self.__tiles(img, mask)  # [N_ROWS, N_COLS, PATCH_HEIGHT, PATCH_WIDTH, N_CHANNELS]
        img, mask = self.__filter_tiles(img, mask)  # [N_SAMPLES, PATCH_HEIGHT, PATCH_WIDTH, N_CHANNELS]
        return {'image': img, 'mask': mask, 'sample_name': self.sample_names[idx]}

    def __crop(self, img, mask):
        """Crop empty places from image and mask."""
        nonzero_indices = np.nonzero(mask)
        min_row_idx = np.min(nonzero_indices[0])
        max_row_idx = np.max(nonzero_indices[0])
        min_col_idx = np.min(nonzero_indices[1])
        max_col_idx = np.max(nonzero_indices[1])
        mask = mask[min_row_idx:max_row_idx, min_col_idx:max_col_idx]
        img = img[min_row_idx:max_row_idx, min_col_idx:max_col_idx]
        return img, mask

    def __tiles(self, image, mask):
        """Split big image into small patches."""
        img_patches = patchify(
            image,
            patch_size=(self.patch_width, self.patch_height, 3),
            step=self.patch_width // 4).squeeze()
        mask_patches = patchify(
            mask,
            patch_size=(self.patch_width, self.patch_height),
            step=self.patch_width // 4).squeeze()
        return img_patches, mask_patches

    def __filter_tiles(self, img_tiles, mask_tiles):
        """Keep only non-empty masks and images."""
        nrow, ncol, *_ = img_tiles.shape
        img_with_label, mask_with_label = [], []
        for i in range(nrow):
            for j in range(ncol):
                mask_patch = mask_tiles[i][j]
                # borders = (mask_patch[0, :] + mask_patch[:, 0] + mask_patch[-1, :] + mask_patch[:, -1])
                if mask_patch.sum().item() > 0:  # and borders.sum() == 0:
                    img_with_label.append(img_tiles[i][j])
                    mask_with_label.append(mask_patch)
        return np.array(img_with_label), np.array(mask_with_label)

    def __replace_values(self, mask, idx):
        """Convert local labels into global ones."""
        with open(self.label_paths[idx], encoding='utf-8') as f:
            labels = [line.replace('\n', '') for line in f.readlines()]
        local_idx2label = dict(enumerate(labels))
        mapping = []
        for i, label in local_idx2label.items():
            mapping.append((i, self.label2idx.get(label, 0)))
        new_mask = np.zeros_like(mask)
        for old, new in mapping:
            new_mask[mask == old] = new
        return new_mask


def convert_dataset(dataset, root_path):
    for ds_batch in tqdm(dataset):
        print(f"Converting {ds_batch['sample_name']}")
        k = 0
        for image, mask in zip(ds_batch['image'], ds_batch['mask']):
            img = Image.fromarray(image, 'RGB')
            img.save(os.path.join(root_path, ds_batch['sample_name'], 'patches', 'images', f'image_{k}.png'))

            mask = Image.fromarray(mask)
            mask.save(os.path.join(root_path, ds_batch['sample_name'], 'patches', 'masks', f'mask_{k}.png'))
            k += 1
        print(f"Converted {ds_batch['sample_name']}")
        print()


if __name__ == '__main__':
    root = os.path.abspath(r'D:\Hepatocyte')
    sample_names = os.listdir(root)
    folders = [os.path.join(root, folder_name, 'full_sample') for folder_name in os.listdir(root)]
    img_paths = [os.path.join(folder, 'img.png') for folder in folders]
    mask_paths = [os.path.join(folder, 'label.png') for folder in folders]
    label_paths = [os.path.join(folder, 'label_names.txt') for folder in folders]
    patch_width = patch_height = 256
    global_mapping = {
        'background': 0,
        'balloon_dystrophy': 1,
        'inclusion': 2,
        'non_nuclei': 3,
        'relatively_normal': 4,
        'steatosis': 5,
        'mesenchymal_cells': 6
    }
    full_img_ds = FullImageDataset(
        image_paths=img_paths, mask_paths=mask_paths, label_paths=label_paths, sample_names=sample_names,
        patch_width=patch_width, patch_height=patch_height, patch_step=patch_width // 4,
        label2idx=global_mapping
    )
    convert_dataset(full_img_ds, os.path.abspath('D:\\Hepatocyte'))