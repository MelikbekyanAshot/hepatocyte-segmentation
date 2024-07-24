import os
import time
import random

import torch.jit
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torchinfo import summary

import wandb
from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config

seed_everything(42, workers=True)


if __name__ == '__main__':
    config = get_config()
    train_config = config['TRAIN']
    root = config['PATH']
    folders = os.listdir(root)
    test_folders = random.sample(folders, 3)
    logger.info(f"Test folders: {test_folders}")
    train_dl, val_dl, test_dl = get_dataloaders(
        root_path=root,
        train_val_folders=set(folders).difference(test_folders),
        test_folders=test_folders,
        patches_path='patches_st_norm_nn_mc',
        batch_size=train_config['BATCH_SIZE'],
        val_size=0.1)
    seg_model = SegmentationModel()
    wandb_config = config['WANDB']
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
    trainer = Trainer(
        max_epochs=train_config['N_EPOCHS'],
        fast_dev_run=True,
        accumulate_grad_batches=4,
        benchmark=True,
        callbacks=[EarlyStopping(monitor='Val/Loss')]
    )
    trainer.fit(
        model=seg_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
    trainer.test(
        model=seg_model,
        dataloaders=test_dl
    )

    # Save jit weights
    try:
        jit_weights_file_name = f"{wandb_config['NAME']}_scripted.pth"
        dummy_input = torch.randn(train_config['BATCH_SIZE'], 3, 128, 128)
        with torch.no_grad():
            traced_cell = torch.jit.trace(seg_model.model, dummy_input)
        torch.jit.save(traced_cell, jit_weights_file_name)
        wandb.save(jit_weights_file_name)
    except:
        logger.error("Can't save jit-weights")

    # Save state dict
    try:
        state_dict_weights = f"{wandb_config['NAME']}_state_dict.pth"
        torch.save(seg_model.state_dict(), state_dict_weights)
        wandb.save(state_dict_weights)
    except:
        logger.error("Can't save state dict")

    wandb.finish()
