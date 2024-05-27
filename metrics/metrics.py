from dataclasses import dataclass, fields

from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score


@dataclass
class Metrics:
    F1: float
    IoU: float

    def to_dict(self, mode):
        return {f"{mode}/{metric.name}": getattr(self, metric.name) for metric in fields(self)}


def compute_metrics(output, target, mode: str, num_classes: int) -> Metrics:
    target = target.round().long()
    tp, fp, fn, tn = get_stats(output, target, mode=mode, num_classes=num_classes)
    iou = iou_score(tp, fp, fn, tn, reduction="macro")
    f1 = f1_score(tp, fp, fn, tn, reduction="macro")
    return Metrics(F1=f1, IoU=iou)
