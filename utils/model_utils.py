"""
File contains utility functions to create model.
"""
from typing import Optional, Type, Dict

import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, JaccardLoss, FocalLoss, TverskyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR


def get_model(architecture: str, encoder_name: str, encoder_weights: Optional[str] = None, output_classes: int = 1) \
        -> torch.nn.Module:
    """Build segmentation model from segmentation-models-pytorch package.

    Args:
        architecture (str) - neural network type.
        encoder_name (str) - classification model that will be used as encoder to extract features from image.
        encoder_weights (Optional[str]) - pretrained weights, None for random initialization.
        output_classes (int) - number of classes to segment, including background.

    Returns:
        model (torch.nn.Module) - neural network for segmentation.

    Raises:
         KeyError: if given architecture is not supported.
         KeyError: if architecture with given backbone is not supported.
         KeyError: if backbone with given encoder_weights is not supported.
    """
    kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'classes': output_classes
    }
    model_mapping = {
        'unet': Unet,
        'unet++': UnetPlusPlus,
        'manet': MAnet,
        'linknet': Linknet,
        'fpn': FPN,
        'pspnet': PSPNet,
        'pan': PAN,
        'deeplabv3': DeepLabV3,
        'deeplabv3+': DeepLabV3Plus
    }
    model = model_mapping.get(architecture.lower(), None)
    if model is None:
        raise KeyError(f"Wrong architecture name '{architecture}', "
                       f"supported architectures: {list(model_mapping.keys())}'")
    return model(**kwargs)


def get_loss(function: str, kwargs: Dict):
    loss_mapping = {
        'dice': DiceLoss,
        'soft_cse': SoftCrossEntropyLoss,
        'jaccard': JaccardLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss
    }
    loss_fn = loss_mapping.get(function, None)
    if loss_fn is None:
        raise KeyError(f'Loss function {function} is not found!')
    return loss_fn(**kwargs)


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


def load_model(path):
    model = torch.jit.load(path)
    return model
