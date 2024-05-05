import wandb
from pytorch_lightning import Trainer

from models.base_model import SegmentationModel
from utils.data_utils import get_dataloaders, get_config


if __name__ == '__main__':
    config = get_config()
    train_dl, val_dl = get_dataloaders(config)
    wandb.login()
    info = {**config['TRAIN']}
    wandb.init()
    model = SegmentationModel()
    trainer = Trainer(
        max_epochs=config['TRAIN']['N_EPOCHS'])
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl)
    wandb.finish()
