import wandb
from pytorch_lightning import LightningModule

from metrics.metrics import compute_metrics
from utils.data_utils import get_config
from utils.model_utils import get_model, get_loss, get_optimizer


config = get_config()


class SegmentationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model(**config['TRAIN']['MODEL'])
        self.loss_fn = get_loss(**config['TRAIN']['LOSS'])
        self.optimizer = get_optimizer(config['TRAIN']['OPTIMIZER'])
        self.test_table = None

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

    def on_test_epoch_start(self) -> None:
        self.test_table = wandb.Table(columns=['sample'])

    def test_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.float()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, mask, 'Test')
        wandb.log({'Test/Loss': loss.item()})
        wandb.log(metrics)
        self.__add_images_to_table(image, mask, pred_mask)

    def __add_images_to_table(self, images, masks, preds):
        idx2label = {0: 'background', 1: 'steatosis'}
        class_set = wandb.Classes(
            [{'name': cls_name, 'id': idx} for idx, cls_name in idx2label.items()])
        for image, mask, pred in zip(images, masks, preds):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy().squeeze()
            pred = pred.detach().cpu().numpy().squeeze()
            masked_image = wandb.Image(
                image,
                masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": idx2label},
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": idx2label}},
                classes=class_set)
            self.test_table.add_data(masked_image)

    def on_test_epoch_end(self) -> None:
        wandb.log({f"Gallery/{self.model.name}": self.test_table})

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
