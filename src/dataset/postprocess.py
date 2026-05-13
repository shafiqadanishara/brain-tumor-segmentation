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


def restore_to_original(pred_chw, original_t1):
    """
    pred_chw : (3,D,H,W) prediction in 128³ space
    original_t1 : original T1 volume in (D,H,W)

    returns:
        restored segmentation in original MRI space
    """

    # --------------------------------------
    # Recompute SAME ROI crop as preprocessing
    # --------------------------------------

    mid = original_t1.shape[0] // 2
    slice_img = original_t1[mid]

    coords = np.where(slice_img > 0)

    h1, h2 = coords[0].min(), coords[0].max() + 1
    w1, w2 = coords[1].min(), coords[1].max() + 1

    # crop shape used BEFORE resize
    crop_shape = (
        original_t1.shape[0],
        h2 - h1,
        w2 - w1
    )

    # --------------------------------------
    # Restore
    # --------------------------------------

    restored = np.zeros(
        (3,) + tuple(original_t1.shape),
        dtype=np.uint8
    )

    for c in range(3):

        resized = resize_mask(
            pred_chw[c],
            crop_shape
        )

        restored[c, :, h1:h2, w1:w2] = resized

    return restored


def save_nifti(arr, affine, path):
    nii = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(nii, str(path))