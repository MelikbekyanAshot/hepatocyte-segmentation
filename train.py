import os
import time

from pytorch_lightning import Trainer, seed_everything
from torchinfo import summary

import wandb
from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config

seed_everything(1234)


if __name__ == '__main__':
    config = get_config()
    train_config = config['TRAIN']
    path_to_patches = config['PATH']['PATCHES']
    wandb_config = config['WANDB']
    train_dl, val_dl = get_dataloaders(
        path_to_dir=path_to_patches,
        batch_size=train_config['BATCH_SIZE'])
    seg_model = SegmentationModel()
    trainer = Trainer(
        max_epochs=train_config['N_EPOCHS'])
    wandb.login()
    wandb.init(
        project=wandb_config['PROJECT'],
        name=wandb_config['NAME'] or f"{seg_model.model.name} ({time.asctime()})",
        config={
            **train_config['MODEL'],
            **train_config['LOSS'],
            'trainable_params': summary(seg_model.model).trainable_params,
            'optimizer': train_config['OPTIMIZER'],
            'lr': train_config['LEARNING_RATE'],
            'n_epochs': train_config['N_EPOCHS'],
            'batch_size': train_config['BATCH_SIZE'],
        }
    )
    trainer.fit(
        model=seg_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl)
    trainer.test(model=seg_model, dataloaders=val_dl)
    folders = os.listdir('lightning_logs/')
    cur_folder = sorted(folders, key=lambda f: int(f[f.rfind('_') + 1:]))[-1]
    wandb.save(
        f'lightning_logs/{cur_folder}/checkpoints/*.ckpt'
    )
    wandb.finish()
