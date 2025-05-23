from typing import Dict

import numpy as np
import torch
import wandb
from loguru import logger
from pytorch_lightning import LightningModule

from metrics.metrics import compute_metrics
from utils.model_utils import get_model, get_optimizer, get_scheduler
from utils.loss_utils import get_loss


class SegmentationModel(LightningModule):
    """Class for segmentation model's training."""
    def __init__(
            self,
            config: Dict
    ):
        super().__init__()
        self.config = config
        self.model = get_model(
            architecture=config['TRAIN']['MODEL']['architecture'],
            **config['TRAIN']['MODEL']['kwargs'])
        self.loss_fn = get_loss(
            functions=config['TRAIN']['LOSS']['function'],
            **config['TRAIN']['LOSS']['kwargs']
        )
        self.scheduler = get_scheduler(name=config['TRAIN']['SCHEDULER']['function'])
        self.optimizer = get_optimizer(name=config['TRAIN']['OPTIMIZER'])
        self.test_table = None
        self.train_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}
        self.val_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}
        self.test_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}
        self.test_gt_labels = []
        self.test_pred_labels = []

    def forward(self, img):
        return self.model(img)

    def on_train_epoch_start(self) -> None:
        self.train_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}

    def training_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long().squeeze()
        logits_mask = self.forward(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(
            pred_mask, mask,
            mode=self.config['TRAIN']['LOSS']['kwargs']['mode'],
            num_classes=self.config['TRAIN']['MODEL']['kwargs']['classes'],
            reduction=self.config['TRAIN']['METRICS']['reduction']
        )
        self.log_batch_results(loss.item(), metrics, mode='Train')
        return loss

    def on_train_epoch_end(self) -> None:
        self.__log_epoch_mean_metrics(
            mode='Train', loss=self.train_history['loss'],
            f1_score=self.train_history['f1_score'], iou=self.train_history['iou'],
            precision=self.train_history['precision'], recall=self.train_history['recall']
        )

    def on_validation_epoch_start(self) -> None:
        self.val_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long().squeeze()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(
            pred_mask, mask,
            mode=self.config['TRAIN']['LOSS']['kwargs']['mode'],
            num_classes=self.config['TRAIN']['MODEL']['kwargs']['classes']
        )
        self.log_batch_results(loss.item(), metrics, mode='Val')

    def on_validation_epoch_end(self) -> None:
        self.__log_epoch_mean_metrics(
            mode='Val', loss=self.val_history['loss'],
            f1_score=self.val_history['f1_score'], iou=self.val_history['iou'],
            precision=self.val_history['precision'], recall=self.val_history['recall']
        )
        self.jit_save()

    def on_test_epoch_start(self) -> None:
        self.test_table = wandb.Table(columns=['sample'])
        self.test_history = {
            'loss': [], 'f1_score': [], 'iou': [],
            'precision': [], 'recall': []}
        self.test_gt_labels = []
        self.test_pred_labels = []

    def test_step(self, batch, batch_idx):
        image, mask = batch
        image, mask = image.float(), mask.long().squeeze()
        logits_mask = self.model(image).float()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        metrics = compute_metrics(
            pred_mask, mask,
            mode=self.config['TRAIN']['LOSS']['kwargs']['mode'],
            num_classes=self.config['TRAIN']['MODEL']['kwargs']['classes']
        )
        self.log_batch_results(loss.item(), metrics, mode='Test')
        self.test_gt_labels.extend(mask.ravel().cpu().numpy())
        self.test_pred_labels.extend(pred_mask.ravel().cpu().numpy())
        self.__add_images_to_table(image, mask, pred_mask)

    def __add_images_to_table(self, images, masks, preds):
        class_set = wandb.Classes(
            [{'name': cls_name, 'id': idx} for idx, cls_name in self.config['WANDB']['IDX2LABEL'].items()])
        for image, mask, pred in zip(images, masks, preds):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy().squeeze()
            pred = pred.detach().cpu().numpy().squeeze()
            masked_image = wandb.Image(
                image,
                masks={
                    'predictions': {
                        'mask_data': pred,
                        'class_labels': self.config['WANDB']['IDX2LABEL']},
                    'ground_truth': {
                        'mask_data': mask,
                        'class_labels': self.config['WANDB']['IDX2LABEL']}},
                classes=class_set)
            self.test_table.add_data(masked_image)

    def on_test_epoch_end(self) -> None:
        self.__log_epoch_mean_metrics(
            mode='Test', loss=self.test_history['loss'],
            f1_score=self.test_history['f1_score'], iou=self.test_history['iou'],
            precision=self.test_history['precision'], recall=self.test_history['recall']
        )

        wandb.log({f"Gallery/{self.config['WANDB']['NAME']}": self.test_table})
        class_names = [label.replace('hepatocyte_', '') for label in self.config['WANDB']['IDX2LABEL'].values()]
        wandb.log({
            'conf_matrix': wandb.plot.confusion_matrix(
                probs=None, y_true=self.test_gt_labels, preds=self.test_pred_labels,
                class_names=class_names, title='Confusion matrix'
            )
        })

    def __log_epoch_mean_metrics(self, mode: str, loss, f1_score, iou, precision, recall):
        mean_loss = np.mean(loss)
        mean_f1 = np.mean(f1_score)
        mean_iou = np.mean(iou)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        wandb.log({
            f'{mode}/EpochLoss': mean_loss,
            f'{mode}/EpochF1': mean_f1,
            f'{mode}/EpochIoU': mean_iou,
            f'{mode}/EpochPrecision': mean_precision,
            f'{mode}/EpochRecall': mean_recall,
            'epoch': self.current_epoch}
        )

    def predict_step(self, image, batch_idx):
        logits_mask = self.model(image.float())
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask.argmax(dim=1, keepdim=True)
        return pred_mask

    def log_batch_results(self, loss: float, metrics, mode):
        wandb.log({f"{mode}/Loss": loss})
        wandb.log(metrics.to_dict(mode=mode))
        if mode == 'Train':
            self.train_history['loss'].append(loss)
            self.train_history['f1_score'].append(metrics.F1)
            self.train_history['iou'].append(metrics.IoU)
            self.train_history['precision'].append(metrics.precision)
            self.train_history['recall'].append(metrics.recall)
        elif mode == 'Val':
            self.val_history['loss'].append(loss)
            self.val_history['f1_score'].append(metrics.F1)
            self.val_history['iou'].append(metrics.IoU)
            self.val_history['precision'].append(metrics.precision)
            self.val_history['recall'].append(metrics.recall)
        elif mode == 'Test':
            self.test_history['loss'].append(loss)
            self.test_history['f1_score'].append(metrics.F1)
            self.test_history['iou'].append(metrics.IoU)
            self.test_history['precision'].append(metrics.precision)
            self.test_history['recall'].append(metrics.recall)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.config['TRAIN']['LEARNING_RATE'])
        if self.scheduler:
            scheduler = self.scheduler(optimizer, **self.config['TRAIN']['SCHEDULER']['kwargs'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer

    def jit_save(self):
        try:
            jit_weights_file_name = f"{self.config['WANDB']['NAME']}-{self.current_epoch}epoch-scripted.pth"
            dummy_input = torch.randn(
                self.config['TRAIN']['BATCH_SIZE'],
                3,
                self.config['TRAIN']['PATCH_SIZE'],
                self.config['TRAIN']['PATCH_SIZE']
            ).cuda()
            with torch.no_grad():
                traced_cell = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(traced_cell, jit_weights_file_name)
            wandb.save(jit_weights_file_name)
        except Exception as e:
            logger.error(f"Can't save jit weights:\n{e}")
