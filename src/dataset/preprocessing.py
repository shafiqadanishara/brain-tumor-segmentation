import numpy as np
from scipy.ndimage import zoom

#normalisasi sederhana (z-score)
def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

#cropping non-zero
def crop_nonzero(image, mask):
    ref = image[3]
    non_zero = ref != 0
    coords = np.array(np.where(non_zero))

    if coords.shape[1] == 0:
        return image, mask
    
    min_h, min_w, min_d = coords.min(axis=1)
    max_h, max_w, max_d = coords.max(axis=1)

    image_cropped = image[:,
                          min_h:max_h+1,
                          min_w:max_w+1,
                          min_d:max_d+1]
    
    mask_cropped = mask[
        min_h:max_h+1,
        min_w:max_w+1,
        min_d:max_d+1
    ]

    return image_cropped, mask_cropped

#padding
def pad_to_shape(image, mask, target_shape=(160,160,160)):
    C, H, W, D = image.shape
    th, tw, td = target_shape

    # =========================
    # CROP JIKA LEBIH BESAR
    # =========================
    start_h = max((H - th) // 2, 0)
    start_w = max((W - tw) // 2, 0)
    start_d = max((D - td) // 2, 0)

    image = image[:, 
                  start_h:start_h+min(H, th),
                  start_w:start_w+min(W, tw),
                  start_d:start_d+min(D, td)]

    mask = mask[
        start_h:start_h+min(H, th),
        start_w:start_w+min(W, tw),
        start_d:start_d+min(D, td)
    ]

    # update size setelah crop
    C, H, W, D = image.shape

    # =========================
    # PAD JIKA KURANG
    # =========================
    pad_h = max(th - H, 0)
    pad_w = max(tw - W, 0)
    pad_d = max(td - D, 0)

    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2

    image = np.pad(
        image,
        ((0,0), (pad_h1,pad_h2), (pad_w1,pad_w2), (pad_d1,pad_d2)),
        mode='constant'
    )

    mask = np.pad(
        mask,
        ((pad_h1,pad_h2), (pad_w1,pad_w2), (pad_d1,pad_d2)),
        mode='constant'
    )

    return image, mask

#ROI crop
def crop_roi_t1(t1, t1ce, t2, flair, seg):
    """
    ROI cropping based on T1 middle slice
    """

    # ambil middle slice (≈77)
    mid_slice = t1.shape[2] // 2
    slice_img = t1[:, :, mid_slice]

    # cari area non-zero di slice ini
    coords = np.where(slice_img > 0)

    x1, x2 = coords[0].min(), coords[0].max()
    y1, y2 = coords[1].min(), coords[1].max()

    # crop semua modality
    t1 = t1[x1:x2, y1:y2, :]
    t1ce = t1ce[x1:x2, y1:y2, :]
    t2 = t2[x1:x2, y1:y2, :]
    flair = flair[x1:x2, y1:y2, :]
    seg = seg[x1:x2, y1:y2, :]

    return t1, t1ce, t2, flair, seg

#depth cropping
def crop_depth(t1, t1ce, t2, flair, seg):
    # 155 → 128
    t1 = t1[:, :, 13:141]
    t1ce = t1ce[:, :, 13:141]
    t2 = t2[:, :, 13:141]
    flair = flair[:, :, 13:141]
    seg = seg[:, :, 13:141]

    return t1, t1ce, t2, flair, seg

#resize
def resize_3d(volume, target_shape=(128,128,128), order=1):
    H, W, D = volume.shape

    th, tw, td = target_shape

    zoom_factors = (th/H, tw/W, td/D)

    return zoom(volume, zoom_factors, order=order)