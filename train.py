import wandb
from pytorch_lightning import Trainer, seed_everything
from torchinfo import summary

from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config


seed_everything()


if __name__ == '__main__':
    config = get_config()
    train_dl, val_dl = get_dataloaders(config)
    seg_model = SegmentationModel()
    trainer = Trainer(
        max_epochs=config['TRAIN']['N_EPOCHS'])
    wandb.login()
    wandb.init(
        project='hepatocyte-segmentation',
        name=seg_model.model.name,
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
    trainer.fit(
        model=seg_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl)
    wandb.finish()
