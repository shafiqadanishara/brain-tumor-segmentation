# src/utils/postprocess.py

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resize_mask(mask, target_shape):
    """
    Resize 3D binary mask using nearest-neighbor interpolation.
    mask: (D,H,W)
    """
    factors = [
        target_shape[0] / mask.shape[0],
        target_shape[1] / mask.shape[1],
        target_shape[2] / mask.shape[2],
    ]
    out = zoom(mask.astype(np.float32), factors, order=0)
    return (out > 0.5).astype(np.uint8)


def restore_to_original(pred_chw, original_shape, bbox):
    """
    pred_chw: (3,D,H,W) in resized training space
    original_shape: (X,Y,Z)
    bbox: (x0,x1,y0,y1,z0,z1)

    returns:
        full volume (3,X,Y,Z)
    """
    x0, x1, y0, y1, z0, z1 = bbox
    crop_shape = (x1 - x0, y1 - y0, z1 - z0)

    restored = np.zeros((3,) + tuple(original_shape), dtype=np.uint8)

    for c in range(3):
        resized = resize_mask(pred_chw[c], crop_shape)
        restored[c, x0:x1, y0:y1, z0:z1] = resized

    return restored


def save_nifti(arr, affine, path):
    nii = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(nii, str(path))