import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from src.dataset.preprocessing import (
    crop_roi_t1,
    resize_3d,
    normalize
)

# ==========================================
# CONFIG
# ==========================================

CASE_DIR = r"data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000"

# ==========================================
# LOAD FILES
# ==========================================

files = os.listdir(CASE_DIR)

t1_file    = next(f for f in files if "-t1n" in f.lower())
t1ce_file  = next(f for f in files if "-t1c" in f.lower())
t2_file    = next(f for f in files if "-t2w" in f.lower())
flair_file = next(f for f in files if "-t2f" in f.lower())
seg_file   = next(f for f in files if "-seg" in f.lower())

# ==========================================
# LOAD MRI
# ==========================================

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

# ==========================================
# SAME PREPROCESSING AS TRAINING
# ==========================================

t1, t1ce, t2, flair, seg = crop_roi_t1(
    t1,
    t1ce,
    t2,
    flair,
    seg
)

t1ce = resize_3d(t1ce)

# ==========================================
# NORMALIZATION
# ==========================================

t1ce_norm = normalize(t1ce)

# ==========================================
# STATISTICS (FULL VOLUME)
# ==========================================

print("===== FULL VOLUME =====")

print("\nBefore Normalization")
print("Mean :", t1ce.mean())
print("Std  :", t1ce.std())

print("\nAfter Normalization")
print("Mean :", t1ce_norm.mean())
print("Std  :", t1ce_norm.std())
print("Min  :", t1ce_norm.min())
print("Max  :", t1ce_norm.max())

# ==========================================
# KDE (BRAIN VOXELS ONLY)
# ==========================================

mask = t1ce > 0

before = t1ce[mask]
after  = t1ce_norm[mask]

# sampling biar KDE cepat
N = 50000

if len(before) > N:
    before = np.random.choice(before, N, replace=False)

if len(after) > N:
    after = np.random.choice(after, N, replace=False)

# KDE
kde_before = gaussian_kde(before)
kde_after  = gaussian_kde(after)

x_before = np.linspace(
    np.percentile(before, 1),
    np.percentile(before, 99),
    1000
)

x_after = np.linspace(
    np.percentile(after, 1),
    np.percentile(after, 99),
    1000
)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12,5))

# BEFORE
plt.subplot(1,2,1)

plt.plot(
    x_before,
    kde_before(x_before),
    linewidth=3
)

plt.title("Before Z-score Normalization")
plt.xlabel("Intensity")
plt.ylabel("Density")

# AFTER
plt.subplot(1,2,2)

plt.plot(
    x_after,
    kde_after(x_after),
    linewidth=3
)

plt.title("After Z-score Normalization")
plt.xlabel("Normalized Intensity")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    "zscore_normalization_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()