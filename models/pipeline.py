import math
import os
from typing import Union, Tuple

from PIL import Image
import numpy as np
import torch
from patchify import patchify, unpatchify
from tqdm.auto import tqdm
import wandb

from utils.data_utils import get_config
from utils.model_utils import get_model

CONFIG = get_config(path=r'C:\Users\melik\PycharmProjects\hepatocyte-segmentation\config.yml')
WANDB_CONFIG = CONFIG['WANDB']


class Pipeline:
    """Class for end-to-end segmentation. Full image to full mask.

    Attributes:
        model (torch.nn.Module) - segmentation model.
        patch_size (Union[int, Tuple[int, int]]) - size of patch to split original image.
        mask_size (Union[int, Tuple[int, int]]) - size of resulting mask.
        batch_size (int) - batch size.
    """
    def __init__(self, model: torch.nn.Module,
                 patch_size: Union[int, Tuple[int, int]],
                 mask_size: Union[int, Tuple[int, int]],
                 batch_size: int = 8,
                 log_results: bool = False):
        self.model = model
        self.model.eval()
        self.patch_size = patch_size if isinstance(patch_size, Tuple) else (patch_size, patch_size)
        self.mask_size = mask_size if isinstance(mask_size, Tuple) else (mask_size, mask_size)
        self.batch_size = batch_size
        self.log_results = log_results

    def segment(self, image: np.ndarray):
        print(image.shape, self.patch_size, self.patch_size[0])
        image_patches = patchify(image=image, patch_size=(*self.patch_size, 3), step=self.patch_size[0]).squeeze()
        flatten_image_patches = torch.from_numpy(image_patches.reshape((-1, 3, *self.patch_size)))
        predicted_mask_patches = self.predict(flatten_image_patches)
        predicted_grid_patches = predicted_mask_patches.reshape(image_patches.shape[:-1])
        mask = unpatchify(predicted_grid_patches, self.mask_size)
        if self.log_results:
            self.log(image, mask)
        return mask

    def predict(self, patches):
        predicts = []
        for i in tqdm(range(0, math.ceil(len(patches) / self.batch_size))):
            batch = patches[i * self.batch_size:(i + 1) * self.batch_size]
            n_samples = batch.shape[0]
            batch = batch\
                .reshape(n_samples, 3, *self.patch_size)\
                .float()
            with torch.no_grad():
                logits_mask = self.model(batch).float()
                prob_mask = logits_mask.sigmoid()
                pred_mask = prob_mask.argmax(dim=1, keepdim=True)
                predicts.extend(pred_mask)
        predicts = np.stack(predicts)
        return predicts

    def log(self, image, mask):
        class_set = wandb.Classes(
            [{'name': cls_name, 'id': idx} for idx, cls_name in WANDB_CONFIG['IDX2LABEL'].items()])
        masked_image = wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": mask,
                    "class_labels":  WANDB_CONFIG['IDX2LABEL']},
            },
            classes=class_set
        )
        wandb.log({'full_sample': masked_image})


if __name__ == '__main__':
    model = get_model(
        architecture='pspnet',
        encoder_name='resnet18',
        output_classes=7)
    mask_size = (2048, 2048)
    pipe = Pipeline(model=model, patch_size=(128, 128), mask_size=mask_size)
    path = os.path.abspath(r"D:\Hepatocyte\10432_19_0\full_sample\img.png")
    image = Image.open(path).resize(mask_size)
    image = np.array(image)
    mask = pipe.segment(image)
    pil_mask = Image.fromarray(mask, mode='RGB')
    pil_mask.save('10432_19_0_predicted.png')
