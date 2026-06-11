# =========================
# src/utils/postprocess.py
# Standardized to (D,H,W)
# FIXED: proper bbox-based restore + correct NIfTI save
# =========================
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resize_mask(mask, target_shape):
    """
    mask        : (D,H,W) binary
    target_shape: (D,H,W) target
    """
    factors = (
        target_shape[0] / mask.shape[0],
        target_shape[1] / mask.shape[1],
        target_shape[2] / mask.shape[2],
    )
    out = zoom(mask.astype(np.float32), factors, order=0)
    return (out > 0.5).astype(np.uint8)


def restore_to_original(pred_chw, original_t1, bbox=None):
    """
    Restore prediction dari 128³ space ke original MRI space.

    Args:
        pred_chw   : (3, D, H, W) — binary prediction (WT, TC, ET)
        original_t1: (D, H, W)    — original T1 volume untuk referensi shape
        bbox       : [d0, d1, h0, h1, w0, w1] dari meta["bbox"] dataset
                     Kalau None, fallback ke crop berbasis middle slice
                     (kurang akurat di axis D)

    Returns:
        restored: (3, D, H, W) uint8 — segmentasi di original space
    """
    orig_shape = original_t1.shape  # (D, H, W)

    restored = np.zeros((3,) + tuple(orig_shape), dtype=np.uint8)

    if bbox is not None:
        # ---- Cara akurat: pakai bbox yang disimpan saat preprocessing ----
        d0, d1, h0, h1, w0, w1 = int(bbox[0]), int(bbox[1]), \
                                   int(bbox[2]), int(bbox[3]), \
                                   int(bbox[4]), int(bbox[5])

        crop_shape = (d1 - d0, h1 - h0, w1 - w0)

        for c in range(3):
            resized = resize_mask(pred_chw[c], crop_shape)
            restored[c, d0:d1, h0:h1, w0:w1] = resized

    else:
        # ---- Fallback: estimasi crop dari middle slice (seperti preprocessing) ----
        # CATATAN: ini tidak akurat di axis D karena preprocessing resize full D
        mid = orig_shape[0] // 2
        slice_img = original_t1[mid]

        coords = np.where(slice_img > 0)
        h1_f, h2_f = int(coords[0].min()), int(coords[0].max()) + 1
        w1_f, w2_f = int(coords[1].min()), int(coords[1].max()) + 1

        # D tidak di-crop di preprocessing, hanya H dan W
        crop_shape = (orig_shape[0], h2_f - h1_f, w2_f - w1_f)

        for c in range(3):
            resized = resize_mask(pred_chw[c], crop_shape)
            restored[c, :, h1_f:h2_f, w1_f:w2_f] = resized

    return restored


def pred_to_label_map(pred_chw):
    """
    Convert 3-channel binary prediction ke single-channel label map.
    Mengikuti konvensi BraTS 2023:
        0 = background
        1 = NCR  (Necrotic Core)     → ET ⊂ TC
        2 = ED   (Edema)             → WT - TC
        3 = ET   (Enhancing Tumor)   → ET

    Args:
        pred_chw: (3, D, H, W) — channel order: WT, TC, ET

    Returns:
        label_map: (D, H, W) uint8
    """
    D, H, W = pred_chw.shape[1:]
    label_map = np.zeros((D, H, W), dtype=np.uint8)

    # WT - TC = Edema (label 2)
    wt_mask = pred_chw[0] > 0.5
    tc_mask = pred_chw[1] > 0.5
    et_mask = pred_chw[2] > 0.5

    label_map[wt_mask & ~tc_mask] = 2  # Edema
    label_map[tc_mask & ~et_mask] = 1  # NCR
    label_map[et_mask]            = 3  # ET (paling prioritas)

    return label_map


def save_nifti(arr, affine, path):
    """
    Save array sebagai NIfTI.
    arr: bisa (D,H,W) label map ATAU (3,D,H,W) multi-channel
    Untuk 3D Slicer, gunakan label map (D,H,W) agar bisa
    di-convert ke segmentation dengan warna per label.
    """
    # Kalau input (3,D,H,W), convert ke label map dulu
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = pred_to_label_map(arr)

    # NIfTI convention: (H,W,D)
    if arr.ndim == 3:
        arr_nifti = arr.transpose(1, 2, 0)
    else:
        arr_nifti = arr

    nii = nib.Nifti1Image(arr_nifti.astype(np.float32), affine)
    nib.save(nii, str(path))


def save_nifti_multichannel(pred_chw, affine, path):
    """
    Simpan tiap channel (WT, TC, ET) sebagai file NIfTI terpisah.
    Berguna untuk visualisasi per-region di 3D Slicer.

    Args:
        pred_chw: (3, D, H, W)
        affine  : affine matrix
        path    : base path (e.g. "pred.nii.gz")
                  akan jadi pred_WT.nii.gz, pred_TC.nii.gz, pred_ET.nii.gz
    """
    from pathlib import Path
    base = Path(str(path).replace(".nii.gz", ""))
    names = ["WT", "TC", "ET"]

    for c, name in enumerate(names):
        channel = pred_chw[c].transpose(1, 2, 0)  # (H,W,D)
        nii = nib.Nifti1Image(channel.astype(np.float32), affine)
        nib.save(nii, str(base) + f"_{name}.nii.gz")