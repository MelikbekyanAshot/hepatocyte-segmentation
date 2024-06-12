import math
from typing import Union, Tuple

from PIL import Image
import numpy as np
import torch
from patchify import patchify, unpatchify
from tqdm.auto import tqdm


class Pipeline:
    """Class for end-to-end segmentation. Full image to full mask.

    Attributes:
        model (torch.nn.Module) - segmentation model.
        patch_size (Tuple[int, int, int]) - size of patch in [H, W, C] format.
        batch_size (int) - batch size.
    """
    def __init__(self, model: torch.nn.Module,
                 patch_size: Tuple[int, int, int],
                 batch_size: int = 8):
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.batch_size = batch_size

    def segment(self, image_path: str):
        image = Image.open(image_path).resize((8192, 8192))
        image = np.array(image)
        print(image.shape)
        patches = patchify(image=image, patch_size=self.patch_size, step=self.patch_size[0] // 2).squeeze()
        print(patches.shape)
        flatten_patches = torch.from_numpy(patches.reshape((-1, 3, 128, 128)))
        print(flatten_patches.shape)
        mask_patches = self.predict(flatten_patches)
        print(mask_patches.shape)
        mask = unpatchify(mask_patches.reshape(patches.shape[:-1]), (8192, 8192))
        print(mask.shape)
        return mask

    def predict(self, patches):
        predicts = []
        for i in tqdm(range(0, math.ceil(len(patches) / self.batch_size))):
            batch = patches[i * self.batch_size:(i + 1) * self.batch_size]
            n_samples = batch.shape[0]
            batch = batch\
                .reshape(n_samples, *self.patch_size[::-1])\
                .float()
            with torch.no_grad():
                predicts.extend(self.model(batch))
        predicts = np.stack(predicts)
        return predicts
