"""Utility components for web application."""
from enum import Enum

import numpy as np
import torch
from PIL import Image


class Color(Enum):
    RED = (255, 0, 0, 128)
    GREEN = (0, 255, 0, 128)
    BLUE = (0, 0, 255, 128)
    YELLOW = (255, 255, 0, 128)
    PURPLE = (128, 0, 128, 128)
    ORANGE = (255, 165, 0, 128)


IDX2LABEL = {
    0: 'background',
    1: 'hepatocyte_balloon_dystrophy',
    2: 'hepatocyte_inclusion',
    3: 'hepatocyte_non_nuclei',
    4: 'hepatocyte_relatively_normal',
    5: 'hepatocyte_steatosis',
    6: 'mesenchymal_cells'
}


def colorify_mask(grayscale_mask: np.array):
    """Convert grayscale mask with 0...N values to specific colors."""
    color_mask = Image.new('RGBA', (512, 512))
    for x in range(grayscale_mask.shape[0]):
        for y in range(grayscale_mask.shape[1]):
            if grayscale_mask[x, y] == 1:
                color_mask.putpixel((y, x), Color.RED.value)
            elif grayscale_mask[x, y] == 2:
                color_mask.putpixel((y, x), Color.GREEN.value)
            elif grayscale_mask[x, y] == 3:
                color_mask.putpixel((y, x), Color.BLUE.value)
            elif grayscale_mask[x, y] == 4:
                color_mask.putpixel((y, x), Color.YELLOW.value)
            elif grayscale_mask[x, y] == 5:
                color_mask.putpixel((y, x), Color.PURPLE.value)
            elif grayscale_mask[x, y] == 6:
                color_mask.putpixel((y, x), Color.ORANGE.value)
    return color_mask


def blend_image_mask(image: Image, mask: Image):
    composite = Image.alpha_composite(
        image.convert(mode='RGBA'),
        mask
    )
    return composite


def pil_to_pt(image: Image) -> torch.Tensor:
    np_image = np.array(image)
    tensor_image = torch.Tensor(np_image) \
        .unsqueeze(0) \
        .permute(0, 3, 1, 2)
    return tensor_image


def segment_image(model: torch.nn.Module, image: torch.Tensor):
    predict = model(image) \
        .argmax(dim=1) \
        .squeeze() \
        .detach() \
        .numpy() \
        .astype(np.uint8)
    return predict


def extract_layer(matrix, label):
    layer_mask = np.isin(matrix, [label])
    res_matrix = np.zeros_like(matrix)
    res_matrix[layer_mask] = matrix[layer_mask]
    return res_matrix
