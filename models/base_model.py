import numpy as np
import wandb
from pytorch_lightning import LightningModule

from metrics.metrics import compute_metrics
from utils.data_utils import get_config
from utils.model_utils import get_model, get_loss, get_optimizer

CONFIG = get_config()
TRAIN_CONFIG = CONFIG['TRAIN']
MODEL_CONFIG = TRAIN_CONFIG['MODEL']
OUTPUT_CLASSES = MODEL_CONFIG['output_classes']
LOSS_CONFIG = TRAIN_CONFIG['LOSS']
MODE = LOSS_CONFIG['mode']
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
        self.train_epoch_loss = []
        self.train_epoch_f1 = []
        self.train_epoch_iou = []
        self.val_epoch_loss = []
        self.val_epoch_f1 = []
        self.val_epoch_iou = []
        self.test_gt_labels = []
        self.test_pred_labels = []

    def forward(self, img):
        return self.model(img)

    def on_train_epoch_start(self) -> None:
        self.train_epoch_loss = []
        self.train_epoch_f1 = []
        self.train_epoch_iou = []

    def training_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.forward(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(pred_mask, mask, mode=MODE, num_classes=OUTPUT_CLASSES)
        self.log_batch_results(loss, metrics, 'Train')
        return loss

    def on_train_epoch_end(self) -> None:
        mean_loss = np.mean(self.train_epoch_loss)
        mean_f1 = np.mean(self.train_epoch_f1)
        mean_iou = np.mean(self.train_epoch_iou)
        wandb.log({'Train/EpochLoss': mean_loss, 'Train/EpochF1': mean_f1, 'Train/EpochIoU': mean_iou})

    def on_validation_epoch_start(self) -> None:
        self.val_epoch_loss = []
        self.val_epoch_f1 = []
        self.val_epoch_iou = []

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(pred_mask, mask, mode=MODE, num_classes=OUTPUT_CLASSES)
        self.log_batch_results(loss, metrics, 'Val')

    def on_validation_epoch_end(self) -> None:
        mean_loss = np.mean(self.val_epoch_loss)
        mean_f1 = np.mean(self.val_epoch_f1)
        mean_iou = np.mean(self.val_epoch_iou)
        wandb.log({'Val/EpochLoss': mean_loss, 'Val/EpochF1': mean_f1, 'Val/EpochIoU': mean_iou})

    def on_test_epoch_start(self) -> None:
        self.test_table = wandb.Table(columns=['sample'])
        self.test_gt_labels = []
        self.test_pred_labels = []

    def test_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(pred_mask, mask, mode=MODE, num_classes=OUTPUT_CLASSES)
        wandb.log({'Test/Loss': loss.item()})
        wandb.log(metrics.to_dict('Test'))
        self.test_gt_labels.extend(mask.ravel().numpy())
        self.test_pred_labels.extend(pred_mask.ravel().numpy())
        self.__add_images_to_table(image, mask, pred_mask)

    def __add_images_to_table(self, images, masks, preds):
        class_set = wandb.Classes(
            [{'name': cls_name, 'id': idx} for idx, cls_name in WANDB_CONFIG['IDX2LABEL'].items()])
        for image, mask, pred in zip(images, masks, preds):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy().squeeze()
            pred = pred.detach().cpu().numpy().squeeze()
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
        class_names = [label.replace('hepatocyte_', '') for label in WANDB_CONFIG['IDX2LABEL'].values()]
        wandb.log({
            'conf_matrix': wandb.plot.confusion_matrix(
                probs=None, y_true=self.test_gt_labels, preds=self.test_pred_labels,
                class_names=class_names, title='Confusion matrix'
            )
        })

    def predict_step(self, image, batch_idx):
        logits_mask = self.model(image.float())
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        return pred_mask

    def log_batch_results(self, loss, metrics, mode):
        wandb.log({f"{mode}/Loss": loss.item()})
        wandb.log(metrics.to_dict(mode=mode))
        if mode == 'Train':
            self.train_epoch_loss.append(loss.item())
            self.train_epoch_f1.append(metrics.F1)
            self.train_epoch_iou.append(metrics.IoU)
        elif mode == 'Val':
            self.val_epoch_loss.append(loss.item())
            self.val_epoch_f1.append(metrics.F1)
            self.val_epoch_iou.append(metrics.IoU)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=LR_CONFIG)
        return optimizer
