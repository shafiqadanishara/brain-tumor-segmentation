import torch


def compute_metrics(pred_logits, target, smooth=1e-6):
    """
    Compute DSC, Precision, Sensitivity, Specificity, Accuracy per channel (WT, TC, ET).

    Args:
        pred_logits : (B, 3, D, H, W) raw logits
        target      : (B, 3, D, H, W) binary float
    Returns:
        dict with per-channel and mean values
    """
    pred = (torch.sigmoid(pred_logits) > 0.5).float()

    metrics   = {}
    dsc_list  = []
    prec_list = []
    sens_list = []
    spec_list = []
    acc_list  = []

    for c, name in enumerate(["WT", "TC", "ET"]):
        p = pred[:, c].contiguous().view(-1)
        t = target[:, c].contiguous().view(-1)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tn = ((1 - p) * (1 - t)).sum()

        dsc  = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        prec = (tp + smooth) / (tp + fp + smooth)
        sens = (tp + smooth) / (tp + fn + smooth)
        spec = (tn + smooth) / (tn + fp + smooth)
        acc  = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

        metrics[f"dsc_{name}"]  = dsc.item()
        metrics[f"prec_{name}"] = prec.item()
        metrics[f"sens_{name}"] = sens.item()
        metrics[f"spec_{name}"] = spec.item()
        metrics[f"acc_{name}"]  = acc.item()

        dsc_list.append(dsc.item())
        prec_list.append(prec.item())
        sens_list.append(sens.item())
        spec_list.append(spec.item())
        acc_list.append(acc.item())

    metrics["dsc_mean"]  = sum(dsc_list)  / 3
    metrics["prec_mean"] = sum(prec_list) / 3
    metrics["sens_mean"] = sum(sens_list) / 3
    metrics["spec_mean"] = sum(spec_list) / 3
    metrics["acc_mean"]  = sum(acc_list)  / 3

    return metrics


def accumulate_metrics(running, new):
    """Add new batch metric values into running totals."""
    for k, v in new.items():
        running[k] = running.get(k, 0.0) + v
    return running


def average_metrics(running, n):
    """Divide all running totals by n to get epoch averages."""
    return {k: v / n for k, v in running.items()}