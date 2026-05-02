# =========================
# src/dataset/preprocessing.py
# Standardized to (D,H,W) volumes
# =========================
import numpy as np
from scipy.ndimage import zoom


def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def crop_nonzero(image, mask):
    """
    image: (C,D,H,W)
    mask : (D,H,W)
    """
    ref = image[3]
    non_zero = ref != 0
    coords = np.array(np.where(non_zero))

    if coords.shape[1] == 0:
        return image, mask

    min_d, min_h, min_w = coords.min(axis=1)
    max_d, max_h, max_w = coords.max(axis=1) + 1

    image = image[:, min_d:max_d, min_h:max_h, min_w:max_w]
    mask = mask[min_d:max_d, min_h:max_h, min_w:max_w]

    return image, mask


def pad_to_shape(image, mask, target_shape=(160, 160, 160)):
    """
    image: (C,D,H,W)
    mask : (D,H,W)
    target_shape: (D,H,W)
    """
    C, D, H, W = image.shape
    td, th, tw = target_shape

    start_d = max((D - td) // 2, 0)
    start_h = max((H - th) // 2, 0)
    start_w = max((W - tw) // 2, 0)

    image = image[
        :,
        start_d:start_d + min(D, td),
        start_h:start_h + min(H, th),
        start_w:start_w + min(W, tw)
    ]

    mask = mask[
        start_d:start_d + min(D, td),
        start_h:start_h + min(H, th),
        start_w:start_w + min(W, tw)
    ]

    C, D, H, W = image.shape

    pad_d = max(td - D, 0)
    pad_h = max(th - H, 0)
    pad_w = max(tw - W, 0)

    pd1, pd2 = pad_d // 2, pad_d - pad_d // 2
    ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
    pw1, pw2 = pad_w // 2, pad_w - pad_w // 2

    image = np.pad(
        image,
        ((0, 0), (pd1, pd2), (ph1, ph2), (pw1, pw2)),
        mode="constant"
    )

    mask = np.pad(
        mask,
        ((pd1, pd2), (ph1, ph2), (pw1, pw2)),
        mode="constant"
    )

    return image, mask


def crop_roi_t1(t1, t1ce, t2, flair, seg):
    """
    Input single volumes: (D,H,W)
    ROI based on middle depth slice
    """
    mid = t1.shape[0] // 2
    slice_img = t1[mid, :, :]

    coords = np.where(slice_img > 0)
    h1, h2 = coords[0].min(), coords[0].max() + 1
    w1, w2 = coords[1].min(), coords[1].max() + 1

    t1 = t1[:, h1:h2, w1:w2]
    t1ce = t1ce[:, h1:h2, w1:w2]
    t2 = t2[:, h1:h2, w1:w2]
    flair = flair[:, h1:h2, w1:w2]
    seg = seg[:, h1:h2, w1:w2]

    return t1, t1ce, t2, flair, seg


def crop_depth(t1, t1ce, t2, flair, seg):
    """
    Crop depth axis first
    """
    t1 = t1[13:141, :, :]
    t1ce = t1ce[13:141, :, :]
    t2 = t2[13:141, :, :]
    flair = flair[13:141, :, :]
    seg = seg[13:141, :, :]
    return t1, t1ce, t2, flair, seg


def resize_3d(volume, target_shape=(128, 128, 128), order=1):
    """
    volume: (D,H,W)
    """
    D, H, W = volume.shape
    td, th, tw = target_shape

    factors = (td / D, th / H, tw / W)
    return zoom(volume, factors, order=order)