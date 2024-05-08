import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3, PSPNet, MAnet, PAN
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, JaccardLoss, FocalLoss
from torch.optim import Adam, AdamW


def get_model(architecture: str, encoder_name: str, encoder_weights=None, output_classes=1):
    kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'classes': output_classes}
    model_mapping = {
        'Unet': Unet,
        'Unet++': UnetPlusPlus,
        'MANet': MAnet,
        'PSPNet': PSPNet,
        'PAN': PAN,
        'DeepLabV3': DeepLabV3
    }
    model = model_mapping.get(architecture, None)(**kwargs)
    if model is None:
        raise KeyError(f'Model {architecture} with {encoder_name} backbone is not found!')
    return model


def get_loss(function: str, mode: str):
    print(mode)
    loss_mapping = {
        'dice': DiceLoss(mode=mode, from_logits=True),
        'soft_cse': SoftCrossEntropyLoss(),
        'jaccard': JaccardLoss(mode=mode, from_logits=True),
        'focal': FocalLoss(mode=mode)
    }
    loss_fn = loss_mapping.get(function, None)
    if loss_fn is None:
        raise KeyError(f'Loss function {function} is not found!')
    return loss_fn


def get_optimizer(name: str):
    opt_mapping = {
        'Adam': Adam,
        'AdamW': AdamW
    }
    optimizer = opt_mapping.get(name, None)
    if optimizer is None:
        raise KeyError(f'Optimizer {name} is not found!')
    return optimizer


def save_model(model):
    script = model.to_torchscript()
    torch.jit.save(script, f'weights/{model.model.name}.pt')


def load_model(path):
    model = torch.jit.load(path)
    return model
