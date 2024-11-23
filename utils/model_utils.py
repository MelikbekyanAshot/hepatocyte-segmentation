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

from utils.custom_losses import GeneralizedDiceLoss


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
        'gen_dice': GeneralizedDiceLoss,
        'soft_cse': SoftCrossEntropyLoss,
        'jaccard': JaccardLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss
    }
    loss_fn = loss_mapping.get(function, None)
    if loss_fn is None:
        raise KeyError(f'Loss function {function} is not found!')
    if function == 'gen_dice':
        return loss_fn()
    if function == 'tversky':
        return TverskyLoss(mode=kwargs.get('mode'), alpha=kwargs.get('alpha'), beta=kwargs.get('beta'))
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
