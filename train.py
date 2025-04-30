import os

import albumentations as A
from pytorch_lightning import Trainer, seed_everything
from torchinfo import summary

import wandb
from models.base_model import SegmentationModel
from utils.data_utils import get_config, get_dataloaders_from_folders

seed_everything(42, workers=True)


if __name__ == '__main__':
    # Build dataloaders
    config = get_config()
    train_config = config['TRAIN']
    root = config['PATH']['ROOT']
    patches = config['PATH']['PATCHES']
    folders = set(os.listdir(root))
    train_folders = ['7939_20_310320201319_7', '7939_20_310320201319_10']
    val_folders = ['7939_20_310320201319_4']
    test_folders = ['7939_20_310320201319_4']
    transforms = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ])
    train_dl, val_dl, test_dl = get_dataloaders_from_folders(
        train_folders=list(train_folders),
        val_folders=list(val_folders),
        test_folders=list(test_folders),
        root_path=root,
        patches_path=patches,
        batch_size=train_config['BATCH_SIZE'],
        train_transforms=transforms
    )

    # Train and validate model
    seg_model = SegmentationModel(config)
    wandb.login()
    wandb.init(
        project='hepatocyte-segmentation',
        name=f"{config['WANDB']['NAME']}",
        config={
            'architecture': config['TRAIN']['MODEL']['architecture'],
            **config['TRAIN']['MODEL']['kwargs'],
            'function': config['TRAIN']['LOSS']['function'],
            **config['TRAIN']['LOSS']['kwargs'],
            'trainable_params': summary(seg_model.model).trainable_params,
            'optimizer': config['TRAIN']['OPTIMIZER'],
            'lr': config['TRAIN']['LEARNING_RATE'],
            'n_epochs': config['TRAIN']['N_EPOCHS'],
            'batch_size': config['TRAIN']['BATCH_SIZE'],
        }
    )
    trainer = Trainer(
        max_epochs=train_config['N_EPOCHS'],
        accumulate_grad_batches=4,
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

    wandb.finish()
