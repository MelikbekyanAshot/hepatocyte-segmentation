from typing import List, Union

import torch
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, JaccardLoss, FocalLoss, TverskyLoss, \
    MULTICLASS_MODE
from torch import nn


class BoundaryDoULoss(nn.Module):
    """Source: https://github.com/sunfan-bvb/BoundaryDoULoss/blob/main/TransUNet/utils.py"""
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        kernel.to(self.device)
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1))
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0).to(self.device),
                padding=1)
        Y = Y.cpu() * target.cpu()
        # Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  # We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs.to(self.device)
        target.to(self.device)
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.squeeze(1))

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


class CombinedLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, *args, **kwargs):
        return sum(loss(*args, **kwargs) for loss in self.losses)


def get_loss(functions: Union[str, List[str]], **kwargs) -> torch.nn.Module:
    loss_mapping = {
        'dice': lambda: DiceLoss(
            mode=kwargs.get('mode')
        ),
        'soft_cse': lambda: SoftCrossEntropyLoss(
            smooth_factor=kwargs.get('smooth_factor')
        ),
        'jaccard': lambda: JaccardLoss(
            mode=kwargs.get('mode')
        ),
        'focal': lambda: FocalLoss(
            mode=kwargs.get('mode'),
            alpha=kwargs.get('alpha'),
            gamma=kwargs.get('gamma', 2.0)
        ),
        'tversky': lambda: TverskyLoss(
            mode=kwargs.get('mode', MULTICLASS_MODE),
            alpha=kwargs.get('alpha', 0.5),
            beta=kwargs.get('beta', 0.5),
            gamma=kwargs.get('gamma', 1.0)
        ),
        'boundary': lambda: BoundaryDoULoss(
            n_classes=len(kwargs.get('class_weights'))
        ),
    }
    # if function not in loss_mapping:
    #     raise KeyError(f'Loss function {function} is not found!')
    # loss_fn = loss_mapping[function]()
    # return loss_fn
    # Приведение к списку, если передали одну строку
    if isinstance(functions, str):
        functions = [functions]

    for fn in functions:
        if fn not in loss_mapping:
            raise KeyError(f'Loss function "{fn}" is not found!')

    losses = [loss_mapping[fn]() for fn in functions]
    return CombinedLoss(losses) if len(losses) > 1 else losses[0]
