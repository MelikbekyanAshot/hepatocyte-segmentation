"""
File contains utility functions to create model.
"""
from functools import lru_cache
from typing import Optional, Type, Dict

import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus,\
    Segformer, DPT, UPerNet
from segmentation_models_pytorch.base import SegmentationModel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR


def get_model(architecture: str, **kwargs: Dict) -> torch.nn.Module:
    """Build segmentation model from segmentation-models-pytorch package.

    Args:
        architecture (str) - neural network type.

    Returns:
        model (torch.nn.Module) - neural network for segmentation.

    Raises:
         KeyError: if given architecture is not supported.
         KeyError: if architecture with given backbone is not supported.
         KeyError: if backbone with given encoder_weights is not supported.
    """
    model_mapping = {
        'unet': Unet,
        'unet++': UnetPlusPlus,
        'manet': MAnet,
        'linknet': Linknet,
        'fpn': FPN,
        'pspnet': PSPNet,
        'pan': PAN,
        'deeplabv3': DeepLabV3,
        'deeplabv3+': DeepLabV3Plus,
        'segformer': Segformer,
        'dpt': DPT,
        'upernet': UPerNet
    }
    model = model_mapping.get(architecture.lower(), None)
    if model is None:
        raise KeyError(f"Wrong architecture name '{architecture}', "
                       f"supported architectures: {list(model_mapping.keys())}'")
    return model(**kwargs)


def get_optimizer(name: str):
    opt_mapping = {
        'Adam': Adam,
        'AdamW': AdamW
    }
    optimizer = opt_mapping.get(name, None)
    if optimizer is None:
        raise KeyError(f'Optimizer {name} is not found!')
    return optimizer


def get_scheduler(name: str) -> Optional[Type[LRScheduler]]:
    schedulers = {
        'cosine_annealing': CosineAnnealingLR,
        'step': StepLR,
        'exponential': ExponentialLR
    }
    scheduler = schedulers.get(name, None)
    return scheduler


def save_model(model: SegmentationModel):
    torch.jit.save(model, f'weights/{model.name}.pt')


@lru_cache(maxsize=1)
def load_model(path):
    model = torch.jit.load(path)
    return model
