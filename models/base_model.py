import wandb
from pytorch_lightning import LightningModule

from metrics.metrics import compute_metrics
from utils.data_utils import get_config
from utils.model_utils import get_model, get_loss, get_optimizer


config = get_config()


class SegmentationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model(config['TRAIN']['MODEL_NAME'])
        self.loss_fn = get_loss(config['TRAIN']['LOSS_FN'])
        self.optimizer = get_optimizer(config['TRAIN']['OPTIMIZER'])

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.float()
        logits_mask = self.forward(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, mask, 'Train')
        wandb.log({"Train/Loss": loss.item()})
        wandb.log(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.float()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, mask, 'Val')
        wandb.log({"Val/Loss": loss.item()})
        wandb.log(metrics)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, image, batch_idx):
        logits_mask = self.model(image.float())
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        return pred_mask

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=config['TRAIN']['LEARNING_RATE'])
        return optimizer
