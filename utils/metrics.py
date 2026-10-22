"""仅通过 SMP 计算水体前景统计量和全局验证指标。"""

import segmentation_models_pytorch as smp
import torch


def batch_statistics(logits, masks, threshold=0.5):
    prob = torch.sigmoid(logits.float())
    pred = (prob > threshold).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred, masks.long(), mode="binary")
    # 先对样本求和；训练入口继续跨 batch 累加这四个整数。
    return torch.stack([value.sum() for value in (tp, fp, fn, tn)])


def compute_metrics(totals):
    tp, fp, fn, tn = (value.reshape(1, 1) for value in totals)
    functions = {
        "iou": smp.metrics.iou_score,
        "f1": smp.metrics.f1_score,
        "precision": smp.metrics.precision,
        "recall": smp.metrics.recall,
    }
    return {
        name: function(tp, fp, fn, tn, reduction="micro").item()
        for name, function in functions.items()
    }
