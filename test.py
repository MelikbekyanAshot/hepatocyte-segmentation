import os

import wandb
from pytorch_lightning import Trainer, seed_everything
from torchinfo import summary

from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config
from utils.model_utils import save_model

seed_everything(123)


if __name__ == '__main__':
    config = get_config()
    *_, test_dl = get_dataloaders(
        path_to_dir=os.path.dirname(config['PATH']['PATCHES']['STEATOSIS']['IMAGE']),
        batch_size=config['TRAIN']['BATCH_SIZE'])
    seg_model = SegmentationModel()
    trainer = Trainer()
    wandb.login()
    wandb.init(
        project='hepatocyte-segmentation',
        name=f"test-{seg_model.model.name}",
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
