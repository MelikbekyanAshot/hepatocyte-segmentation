from typing import Optional, Dict, Union

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from utils.colors import Color


def smooth_mask(mask, kernel_size=(5, 5)):
    mask = cv2.GaussianBlur(mask, kernel_size, 0)
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def blend_image_mask(image: Image, mask: Image) -> Image:
    blended = Image.blend(image, mask, 0.3)
    return blended


def pil_to_pt(image: Image) -> torch.Tensor:
    np_image = np.array(image)
    tensor_image = torch.Tensor(np_image) \
        .unsqueeze(0) \
        .permute(0, 3, 1, 2)
    return tensor_image


def segment_patch(model: torch.nn.Module, patch: Image) -> Dict[str, Union[np.ndarray]]:
    pt_image = pil_to_pt(patch)
    predict = model(pt_image) \
        .argmax(dim=1) \
        .squeeze() \
        .detach() \
        .numpy() \
        .astype(np.uint8)
    colored_predict = colorify_mask(predict)
    blended = blend_image_mask(patch.convert('RGBA'), colored_predict.convert('RGBA'))
    return {'raw_predict': predict, 'colored_predict': colored_predict, 'blended_predict': blended}


def colorify_mask(grayscale_mask: np.array, id2color: Optional[Dict] = None) -> Image:
    if not id2color:
        id2color = dict(enumerate(Color))
    """Convert grayscale mask with 0...N values to specific colors."""
    color_mask = Image.new('RGBA', grayscale_mask.shape)
    for x in range(grayscale_mask.shape[0]):
        for y in range(grayscale_mask.shape[1]):
            color_mask.putpixel((y, x), id2color[grayscale_mask[x, y]].value)
    return color_mask


def extract_layer(matrix, label):
    layer_mask = np.isin(matrix, [label])
    res_matrix = np.zeros_like(matrix)
    res_matrix[layer_mask] = matrix[layer_mask]
    return res_matrix
