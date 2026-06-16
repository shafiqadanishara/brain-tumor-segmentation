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

CASE_DIR = (
    "data/raw/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    "BraTS-GLI-00000-000"
)

# ==========================================
# LOAD MRI
# ==========================================

files = os.listdir(CASE_DIR)

t1_file    = next(f for f in files if "-t1n" in f.lower())
t1ce_file  = next(f for f in files if "-t1c" in f.lower())
t2_file    = next(f for f in files if "-t2w" in f.lower())
flair_file = next(f for f in files if "-t2f" in f.lower())
seg_file   = next(f for f in files if "-seg" in f.lower())


def load_vol(path):
    return np.transpose(
        nib.load(path).get_fdata(),
        (2, 0, 1)
    )


t1 = load_vol(
    os.path.join(CASE_DIR, t1_file)
)

t1ce = load_vol(
    os.path.join(CASE_DIR, t1ce_file)
)

t2 = load_vol(
    os.path.join(CASE_DIR, t2_file)
)

flair = load_vol(
    os.path.join(CASE_DIR, flair_file)
)

seg = load_vol(
    os.path.join(CASE_DIR, seg_file)
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
# BEFORE / AFTER
# ==========================================

before = t1ce.copy()

after = normalize(t1ce)

# ==========================================
# BRAIN MASK
# ==========================================

brain_mask = before > 0

brain_before = before[brain_mask]
brain_after = after[brain_mask]

# ==========================================
# STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("FULL VOLUME BEFORE NORMALIZATION")
print("=" * 60)

print(f"Mean     : {before.mean():.6f}")
print(f"Std      : {before.std():.6f}")
print(f"Variance : {before.var():.6f}")
print(f"Min      : {before.min():.6f}")
print(f"Max      : {before.max():.6f}")

print("\n" + "=" * 60)
print("FULL VOLUME AFTER NORMALIZATION")
print("=" * 60)

print(f"Mean     : {after.mean():.12f}")
print(f"Std      : {after.std():.12f}")
print(f"Variance : {after.var():.12f}")
print(f"Min      : {after.min():.6f}")
print(f"Max      : {after.max():.6f}")

print("\n" + "=" * 60)
print("BRAIN VOXELS ONLY AFTER NORMALIZATION")
print("=" * 60)

print(f"Mean     : {brain_after.mean():.12f}")
print(f"Std      : {brain_after.std():.12f}")
print(f"Variance : {brain_after.var():.12f}")
print(f"Min      : {brain_after.min():.6f}")
print(f"Max      : {brain_after.max():.6f}")

# ==========================================
# DISTRIBUTION CURVES
# ==========================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

plots = [
    (
        before.flatten(),
        "Before Normalization"
    ),
    (
        after.flatten(),
        "After Z-score (Full Volume)"
    ),
    (
        brain_after,
        "After Z-score (Brain Voxels Only)"
    )
]

for i, (data, title) in enumerate(plots):

    mean = data.mean()
    var = data.var()

    hist, bins = np.histogram(
        data,
        bins=150,
        density=True
    )

    centers = (
        bins[:-1] +
        bins[1:]
    ) / 2

    # distribution curve
    ax[i].plot(
        centers,
        hist,
        linewidth=2,
        label="Distribution"
    )

    # mean
    ax[i].axvline(
        mean,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean:.3f}"
    )

    ax[i].set_title(
        f"{title}\n"
        f"Variance = {var:.3f}"
    )

    ax[i].set_xlabel(
        "Intensity"
    )

    ax[i].set_ylabel(
        "Density"
    )

    ax[i].grid(
        alpha=0.3
    )

    ax[i].legend(
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "zscore_brain_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()