from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score


def compute_metrics(output, target, mode='Val'):
    """"""
    target = target.round().long()
    tp, fp, fn, tn = get_stats(output, target, mode='binary', threshold=0.5)  # TODO fix hardcode 'mode'
    iou = iou_score(tp, fp, fn, tn, reduction="micro")
    f1 = f1_score(tp, fp, fn, tn, reduction="micro")
    return {f'{mode}/iou_score': iou.item(), f'{mode}/f1_score': f1.item()}
