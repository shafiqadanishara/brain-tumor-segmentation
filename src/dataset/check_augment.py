import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    affine_transform,
    map_coordinates,
    gaussian_filter
)

from src.dataset.preprocessing import (
    crop_roi_t1,
    resize_3d,
    normalize
)

# =====================================================
# CONFIG
# =====================================================

CASE_DIR = r"data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000"

# =====================================================
# LOAD
# =====================================================

files = os.listdir(CASE_DIR)

t1_file    = next(f for f in files if "-t1n" in f.lower())
t1ce_file  = next(f for f in files if "-t1c" in f.lower())
t2_file    = next(f for f in files if "-t2w" in f.lower())
flair_file = next(f for f in files if "-t2f" in f.lower())
seg_file   = next(f for f in files if "-seg" in f.lower())

t1 = np.transpose(
    nib.load(os.path.join(CASE_DIR, t1_file)).get_fdata(),
    (2,0,1)
)

t1ce = np.transpose(
    nib.load(os.path.join(CASE_DIR, t1ce_file)).get_fdata(),
    (2,0,1)
)

t2 = np.transpose(
    nib.load(os.path.join(CASE_DIR, t2_file)).get_fdata(),
    (2,0,1)
)

flair = np.transpose(
    nib.load(os.path.join(CASE_DIR, flair_file)).get_fdata(),
    (2,0,1)
)

seg = np.transpose(
    nib.load(os.path.join(CASE_DIR, seg_file)).get_fdata(),
    (2,0,1)
)

# =====================================================
# SAME PREPROCESS AS TRAINING
# =====================================================

t1, t1ce, t2, flair, seg = crop_roi_t1(
    t1,
    t1ce,
    t2,
    flair,
    seg
)

t1ce = resize_3d(t1ce)
seg  = resize_3d(seg, order=0)

t1ce = normalize(t1ce)

# =====================================================
# TAKE CENTER SLICE AFTER PREPROCESS
# =====================================================

z = t1ce.shape[0] // 2

img0 = t1ce[z]
seg0 = seg[z]

# =====================================================
# FLIP
# =====================================================

img_flip = np.fliplr(img0)
seg_flip = np.fliplr(seg0)

# =====================================================
# ROTATION
# =====================================================

angle = 12

theta = np.deg2rad(angle)

c = np.cos(theta)
s = np.sin(theta)

matrix = np.array([
    [c, -s],
    [s,  c]
])

center = np.array(img0.shape) / 2

offset = center - matrix @ center

img_rot = affine_transform(
    img0,
    matrix,
    offset=offset,
    order=1,
    mode="nearest"
)

seg_rot = affine_transform(
    seg0,
    matrix,
    offset=offset,
    order=0,
    mode="nearest"
)

# =====================================================
# ELASTIC
# =====================================================

H, W = img0.shape

alpha = 20
sigma = 4

dx = gaussian_filter(
    np.random.randn(H, W),
    sigma
) * alpha

dy = gaussian_filter(
    np.random.randn(H, W),
    sigma
) * alpha

x, y = np.meshgrid(
    np.arange(W),
    np.arange(H)
)

coords = [
    np.clip(y + dy, 0, H - 1),
    np.clip(x + dx, 0, W - 1)
]

img_elastic = map_coordinates(
    img0,
    coords,
    order=1
)

seg_elastic = map_coordinates(
    seg0,
    coords,
    order=0
)

# =====================================================
# BRIGHTNESS
# =====================================================

img_bright = img0 + 0.1

# =====================================================
# GAMMA
# =====================================================

tmp = img0.copy()

tmp = (tmp - tmp.min()) / (
    tmp.max() - tmp.min() + 1e-8
)

gamma = 1.8

img_gamma = np.power(tmp, gamma)

# =====================================================
# PLOT
# =====================================================

titles = [
    "Original",
    "Flip",
    "Rotation",
    "Elastic",
    "Brightness",
    "Gamma"
]

images = [
    img0,
    img_flip,
    img_rot,
    img_elastic,
    img_bright,
    img_gamma
]

masks = [
    seg0,
    seg_flip,
    seg_rot,
    seg_elastic,
    seg0,
    seg0
]

fig, axes = plt.subplots(
    2,
    6,
    figsize=(20,7)
)

for i in range(6):

    axes[0, i].imshow(
        images[i],
        cmap="gray"
    )

    axes[0, i].set_title(
        titles[i],
        fontsize=12
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        masks[i],
        cmap="jet",
        vmin=0,
        vmax=3
    )

    axes[1, i].axis("off")

axes[0,0].set_ylabel(
    "Preprocessed MRI",
    fontsize=14
)

axes[1,0].set_ylabel(
    "Segmentation Mask",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "augmentation_after_preprocessing.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()