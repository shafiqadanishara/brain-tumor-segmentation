import torch
import torch.nn.functional as F


def bce_dice_loss(pred_logits, target, smooth=1e-6):
    """
    Combined BCE + Soft Dice loss for multi-channel binary masks.

    Args:
        pred_logits : (B, 3, D, H, W) raw logits
        target      : (B, 3, D, H, W) binary float — channels: WT, TC, ET
    Returns:
        scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)

    pred     = torch.sigmoid(pred_logits)
    pred_f   = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_f = target.contiguous().view(target.shape[0], target.shape[1], -1)

    intersection = (pred_f * target_f).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (
        pred_f.sum(dim=2) + target_f.sum(dim=2) + smooth
    )

    return 0.5 * bce + 0.5 * (1.0 - dice.mean())