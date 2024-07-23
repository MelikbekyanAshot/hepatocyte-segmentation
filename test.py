import torch
from pytorch_lightning import Trainer, seed_everything
from torchinfo import summary

import wandb
from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config

seed_everything(123)


if __name__ == '__main__':
    config = get_config()
    *_, test_dl = get_dataloaders(
        root_path=config['PATH'], patches_path='patches',
        batch_size=config['TRAIN']['BATCH_SIZE'])
    seg_model = SegmentationModel()
    seg_model.model.load_state_dict(torch.load('weights/deeplabv3-resnet50-5ep_state_dict.pth'), strict=False)
    trainer = Trainer()
    wandb.login()
    wandb.init(
        project='hepatocyte-segmentation',
        name=f"test-{config['TRAIN']['MODEL']['architecture']}-{config['TRAIN']['MODEL']['encoder_name']}",
        config={
            **config['TRAIN']['MODEL'],
            **config['TRAIN']['LOSS'],
            'trainable_params': summary(seg_model.model).trainable_params,
            'optimizer': config['TRAIN']['OPTIMIZER'],
            'lr': config['TRAIN']['LEARNING_RATE'],
            'n_epochs': config['TRAIN']['N_EPOCHS'],
            'batch_size': config['TRAIN']['BATCH_SIZE'],
        }
    )
    trainer.test(model=seg_model, dataloaders=test_dl)
    wandb.finish()
