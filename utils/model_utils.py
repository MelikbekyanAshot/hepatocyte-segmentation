from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE
from torch.optim import Adam


def get_model(name: str):
    model_mapping = {
        'u-resnet34': Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=1),
        'u-resnet50': Unet(
            encoder_name='resnet50',
            encoder_weights=None,
            classes=1)
    }
    model = model_mapping.get(name, None)
    if model is None:
        raise KeyError(f'Model {name} is not found!')
    return model


def get_loss(name: str):
    loss_mapping = {
        'dice': DiceLoss(mode=BINARY_MODE, from_logits=True)
    }
    loss_fn = loss_mapping.get(name, None)
    if loss_fn is None:
        raise KeyError(f'Loss function {name} is not found!')
    return loss_fn


def get_optimizer(name: str):
    opt_mapping = {
        'Adam': Adam
    }
    optimizer = opt_mapping.get(name, None)
    if optimizer is None:
        raise KeyError(f'Optimizer {name} is not found!')
    return optimizer
