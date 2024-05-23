import wandb
from pytorch_lightning import LightningModule

from metrics.metrics import compute_metrics
from utils.data_utils import get_config, split_mask, merge_mask
from utils.model_utils import get_model, get_loss, get_optimizer

CONFIG = get_config()
TRAIN_CONFIG = CONFIG['TRAIN']
MODEL_CONFIG = TRAIN_CONFIG['MODEL']
OUTPUT_CLASSES = MODEL_CONFIG['output_classes']
LOSS_CONFIG = TRAIN_CONFIG['LOSS']
OPT_CONFIG = TRAIN_CONFIG['OPTIMIZER']
LR_CONFIG = TRAIN_CONFIG['LEARNING_RATE']
WANDB_CONFIG = CONFIG['WANDB']


class SegmentationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model(**MODEL_CONFIG)
        self.loss_fn = get_loss(**LOSS_CONFIG)
        self.optimizer = get_optimizer(OPT_CONFIG)
        self.test_table = None

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.forward(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, split_mask(mask, OUTPUT_CLASSES), 'Train')
        wandb.log({"Train/Loss": loss.item()})
        wandb.log(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, split_mask(mask, OUTPUT_CLASSES), 'Val')
        wandb.log({"Val/Loss": loss.item()})
        wandb.log(metrics)

    def on_test_epoch_start(self) -> None:
        self.test_table = wandb.Table(columns=['sample'])

    def test_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        metrics = compute_metrics(pred_mask, split_mask(mask, OUTPUT_CLASSES), 'Test')
        wandb.log({'Test/Loss': loss.item()})
        wandb.log(metrics)
        self.__add_images_to_table(image, mask, pred_mask)

    def __add_images_to_table(self, images, masks, preds):
        class_set = wandb.Classes(
            [{'name': cls_name, 'id': idx} for idx, cls_name in WANDB_CONFIG['IDX2LABEL'].items()])
        for image, mask, pred in zip(images, masks, preds):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy().squeeze()
            pred = merge_mask(pred.detach().cpu().numpy().squeeze())
            masked_image = wandb.Image(
                image,
                masks={
                    'predictions': {
                        'mask_data': pred,
                        'class_labels': WANDB_CONFIG['IDX2LABEL']},
                    'ground_truth': {
                        'mask_data': mask,
                        'class_labels': WANDB_CONFIG['IDX2LABEL']}},
                classes=class_set)
            self.test_table.add_data(masked_image)

    def on_test_epoch_end(self) -> None:
        wandb.log({f"Gallery/{WANDB_CONFIG['NAME'] or self.model.name}": self.test_table})

    def predict_step(self, image, batch_idx):
        logits_mask = self.model(image.float())
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        return pred_mask

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=LR_CONFIG)
        return optimizer
