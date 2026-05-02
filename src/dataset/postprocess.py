# =========================
# src/utils/postprocess.py
# Standardized to (D,H,W)
# =========================
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resize_mask(mask, target_shape):
    """
    mask: (D,H,W)
    target_shape: (D,H,W)
    """
    factors = (
        target_shape[0] / mask.shape[0],
        target_shape[1] / mask.shape[1],
        target_shape[2] / mask.shape[2],
    )
    out = zoom(mask.astype(np.float32), factors, order=0)
    return (out > 0.5).astype(np.uint8)


def restore_to_original(pred_chw, original_shape, bbox):
    """
    pred_chw: (3,D,H,W)
    original_shape: (D,H,W)
    bbox: (d0,d1,h0,h1,w0,w1)

    returns:
        (3,D,H,W)
    """
    d0, d1, h0, h1, w0, w1 = bbox
    crop_shape = (d1 - d0, h1 - h0, w1 - w0)

    restored = np.zeros((3,) + tuple(original_shape), dtype=np.uint8)

    for c in range(3):
        resized = resize_mask(pred_chw[c], crop_shape)
        restored[c, d0:d1, h0:h1, w0:w1] = resized

    return restored


def save_nifti(arr, affine, path):
    nii = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(nii, str(path))