"""Utility components for web application."""
from enum import Enum
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


class Color(Enum):
    TRANSPARENT = (0, 0, 0, 0)
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


ID2COLOR = {
    0: Color.TRANSPARENT,
    1: Color.RED,
    2: Color.GREEN,
    3: Color.BLUE,
    4: Color.YELLOW,
    5: Color.PURPLE,
    6: Color.ORANGE
}


def colorify_mask(grayscale_mask: np.array):
    """Convert grayscale mask with 0...N values to specific colors."""
    color_mask = Image.new('RGBA', (512, 512))
    for x in range(grayscale_mask.shape[0]):
        for y in range(grayscale_mask.shape[1]):
            color_mask.putpixel((y, x), ID2COLOR[grayscale_mask[x, y]].value)
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


@lru_cache(maxsize=1)
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


def generate_color_circle(color):
    html_code = f"""
<head>
    <style>
        .red-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ff0000;
        }}
        .green-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #00ff00;
        }}
        .blue-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #0000ff;
        }}
        .yellow-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ffff00;
        }}
        .purple-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #800080;
        }}
        .orange-circle {{
            padding: 2px 11px;
            border-radius: 100%;
            background-color: #ffa500;
        }}
    </style>
</head>
<body>
    <span class="{color}-circle"></span>
</body>
</html>"""
    return html_code
