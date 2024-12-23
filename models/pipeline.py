import math
import os
import time
from abc import ABC
from typing import Union, Tuple, List

import numpy as np
import streamlit as st
import torch
from PIL import Image
from patchify import patchify, unpatchify
from stqdm import stqdm
from tqdm.auto import tqdm

from utils.colors import Color
from utils.data_utils import get_config

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
                 patch_size: Union[int, Tuple[int, int]] = 512,
                 mask_size: Union[int, Tuple[int, int]] = 8192,
                 batch_size: int = 8,
                 log_results: bool = False):
        self.model = model
        self.model.eval()
        self.patch_size = patch_size if isinstance(patch_size, Tuple) else (patch_size, patch_size)
        self.mask_size = mask_size if isinstance(mask_size, Tuple) else (mask_size, mask_size)
        self.batch_size = batch_size
        self.log_results = log_results

    def run(self, wsi: np.ndarray, n_jobs: int = 1, merge_predicted: bool = True):
        non_empty_image_patches, non_empty_indexes, total_patches_number = self._extract_info(wsi)
        mask_patches = self._segment_patches(n_jobs, non_empty_image_patches)
        if not merge_predicted:
            return mask_patches
        wsi_mask = self._build_wsi_mask(mask_patches, non_empty_indexes, total_patches_number)
        wsi_mask = self._colorify_mask(wsi_mask)
        return wsi_mask

    def _segment_patches(self, n_jobs, non_empty_image_patches):
        if n_jobs == 1:
            mask_patches = self._predict(non_empty_image_patches)
        else:
            raise NotImplementedError
        return mask_patches

    def _extract_info(self, wsi):
        image_patches = patchify(
            image=wsi,
            patch_size=(*self.patch_size, 3),
            step=self.patch_size[0]).squeeze()
        image_patches = list(enumerate(image_patches.reshape((-1, 3, *self.patch_size))))
        total_patches_number = len(image_patches)
        non_empty_image_patches_info = [(i, patch) for i, patch in image_patches if patch.mean().item() < 200]
        non_empty_indexes = [i for i, patch in non_empty_image_patches_info]
        non_empty_image_patches = [patch for i, patch in non_empty_image_patches_info]
        non_empty_image_patches = torch.from_numpy(np.array(non_empty_image_patches))
        return non_empty_image_patches, non_empty_indexes, total_patches_number

    def _build_wsi_mask(self, mask_patches: List[np.ndarray], patches_indexes: List[int], total_number: int):
        mask_patches = list(mask_patches)
        patches = []
        background_patch = np.zeros((512, 512), dtype=np.int64)
        for idx in range(total_number):
            if idx in patches_indexes:
                patches.append(mask_patches.pop(0))
            else:
                patches.append(background_patch)
        patches = np.array(patches)
        patches.resize((16, 16, 512, 512))
        wsi_mask = unpatchify(patches, self.mask_size)
        return wsi_mask

    def _colorify_mask(self, grayscale_mask: np.array):
        id2color = dict(enumerate(Color))
        """Convert grayscale mask with 0...N values to specific colors."""
        color_mask = Image.new('RGBA', grayscale_mask.shape)
        for x in range(grayscale_mask.shape[0]):
            for y in range(grayscale_mask.shape[1]):
                color_mask.putpixel((y, x), id2color[grayscale_mask[x, y]].value)
        return color_mask

    def _predict(self, patches):
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

    def _blend_image_mask(self, image: Image, mask: Image) -> Image:
        st.write(image.size, mask.size)
        blended = Image.blend(
            image.convert('RGBA'),
            mask.convert('RGBA'),
            alpha=0.3
        )
        return blended

    def _segment_parallel(self, image: np.ndarray, n_jobs: int):
        raise NotImplementedError

    def log(self, image, pred_mask, gt_mask):
        raise NotImplementedError


class StreamlitPipeline(Pipeline, ABC):
    """Adjusted to Streamlit interface."""
    def run(self, wsi: Image, n_jobs: int = 1, merge_predicted: bool = True):
        with st.status('Обработка WSI', expanded=True):
            np_wsi = np.array(wsi)
            non_empty_image_patches, non_empty_indexes, total_patches_number = self._extract_info(np_wsi)
            st.write('Сегментирование')
            mask_patches = self._segment_patches(n_jobs, non_empty_image_patches).squeeze()
            wsi_mask = self._build_wsi_mask(mask_patches, non_empty_indexes, total_patches_number)
            st.write('Окрашивание изображения')
            wsi_mask = self._colorify_mask(wsi_mask)
            result = self._blend_image_mask(wsi, wsi_mask)
            return result

    def _predict(self, patches):
        predicts = []
        for i in stqdm(range(0, math.ceil(len(patches) / self.batch_size))):
            batch = patches[i * self.batch_size:(i + 1) * self.batch_size]
            n_samples = batch.shape[0]
            batch = batch \
                .reshape(n_samples, 3, *self.patch_size) \
                .float()
            with torch.no_grad():
                logits_mask = self.model(batch).float()
                prob_mask = logits_mask.sigmoid()
                pred_mask = prob_mask.argmax(dim=1, keepdim=True)
                predicts.extend(pred_mask)
        predicts = np.stack(predicts)
        return predicts

    def _build_wsi_mask(self, mask_patches: List[np.ndarray], patches_indexes: List[int], total_number: int):
        mask_patches = list(mask_patches)
        patches = []
        background_patch = np.zeros((512, 512), dtype=np.int64)
        for idx in stqdm(range(total_number)):
            if idx in patches_indexes:
                patches.append(mask_patches.pop(0))
            else:
                patches.append(background_patch)
        patches = np.array(patches)
        patches.resize((16, 16, 512, 512))
        wsi_mask = unpatchify(patches, self.mask_size)
        return wsi_mask

    def _colorify_mask(self, grayscale_mask: np.array):
        id2color = dict(enumerate(Color))
        """Convert grayscale mask with 0...N values to specific colors."""
        color_mask = Image.new('RGBA', grayscale_mask.shape)
        for x in stqdm(range(grayscale_mask.shape[0])):
            for y in range(grayscale_mask.shape[1]):
                color_mask.putpixel((y, x), id2color[grayscale_mask[x, y]].value)
        return color_mask


if __name__ == '__main__':
    model = torch.jit.load('../weights/unet-resnet101_scripted.pth')
    mask_size = (8192, 8192)
    pipe = Pipeline(
        model=model,
        batch_size=4,
        patch_size=(512, 512),
        mask_size=mask_size,
        log_results=False
    )
    samples = ['7939_20_310320201319_4', '7939_20_310320201319_7', '7939_20_310320201319_10']
    for sample in samples:
        root_dir = os.path.abspath(f'D:\\Hepatocyte_full\\{sample}\\full_sample')
        img_path = os.path.join(root_dir, 'img.png')
        img = Image.open(img_path).resize(mask_size)
        img = np.array(img)
        start = time.perf_counter()
        predicted_mask = pipe.run(img)
        end = time.perf_counter()
        print(end - start)
        break
    # composite = np.array(blend_image_mask(img, predicted_mask))
    # plt.imsave(os.path.join(root_dir, 'predicted.png'), predicted_mask, cmap='gray')

    # gt_mask_path = os.path.join(root_path, 'label.png')
    # gt_mask = Image.open(gt_mask_path).resize(mask_size)
    # gt_mask = np.array(gt_mask)
    # try:
    #     compute_metrics(gt_mask, predicted_mask, mode='multiclass', num_classes=7)
    # except:
    #     print('Error')

    # image = Image.fromarray(image)
    # mask = Image.fromarray(predicted_mask)
    # composite = Image.alpha_composite(
    #     image.convert(mode='RGBA'),
    #     mask
    # )
    # composite.save('blended_predict.png')
