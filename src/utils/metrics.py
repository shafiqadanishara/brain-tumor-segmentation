import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def hd95_single(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    # both empty -> perfect match
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    # one empty -> worst case
    if pred.sum() == 0 or target.sum() == 0:
        return 373.0   # approx diagonal of 128^3 cube

    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)

    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    dist_pred_to_target = dt_target[pred_surface]
    dist_target_to_pred = dt_pred[target_surface]

    all_dists = np.concatenate([dist_pred_to_target, dist_target_to_pred])

    return float(np.percentile(all_dists, 95))


def compute_metrics(pred_logits, target, smooth=1e-6):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()

    metrics = {}

    dsc_list = []
    hd_list = []
    prec_list = []
    sens_list = []
    spec_list = []
    acc_list = []

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

        # HD95 per sample then average batch
        hd_vals = []
        for b in range(pred.shape[0]):
            pb = pred[b, c].cpu().numpy()
            tb = target[b, c].cpu().numpy()
            hd_vals.append(hd95_single(pb, tb))
        hd95 = float(np.mean(hd_vals))

        metrics[f"dsc_{name}"]  = dsc.item()
        metrics[f"hd95_{name}"] = hd95
        metrics[f"prec_{name}"] = prec.item()
        metrics[f"sens_{name}"] = sens.item()
        metrics[f"spec_{name}"] = spec.item()
        metrics[f"acc_{name}"]  = acc.item()

        dsc_list.append(dsc.item())
        hd_list.append(hd95)
        prec_list.append(prec.item())
        sens_list.append(sens.item())
        spec_list.append(spec.item())
        acc_list.append(acc.item())

    metrics["dsc_mean"]  = sum(dsc_list) / 3
    metrics["hd95_mean"] = sum(hd_list) / 3
    metrics["prec_mean"] = sum(prec_list) / 3
    metrics["sens_mean"] = sum(sens_list) / 3
    metrics["spec_mean"] = sum(spec_list) / 3
    metrics["acc_mean"]  = sum(acc_list) / 3

    return metrics


def accumulate_metrics(running, new):
    for k, v in new.items():
        running[k] = running.get(k, 0.0) + v
    return running


def average_metrics(running, n):
    return {k: v / n for k, v in running.items()}