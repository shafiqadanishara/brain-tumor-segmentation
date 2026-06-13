import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

DATA_ROOT = r"data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

N_CASES = 200

MAX_SAMPLE_PER_CASE = 3000
KDE_SAMPLE = 5000

# =====================================================
# STORAGE
# =====================================================

modalities = ["T1", "T1ce", "T2", "FLAIR"]

healthy = {m: [] for m in modalities}
wt      = {m: [] for m in modalities}
tc      = {m: [] for m in modalities}
et      = {m: [] for m in modalities}

# =====================================================
# CASES
# =====================================================

cases = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
])

cases = cases[:N_CASES]

print(f"Using {len(cases)} cases")

# =====================================================
# LOOP
# =====================================================

for case in tqdm(cases):

    case_dir = os.path.join(DATA_ROOT, case)

    files = os.listdir(case_dir)

    try:
        t1_file    = next(f for f in files if "-t1n" in f.lower())
        t1ce_file  = next(f for f in files if "-t1c" in f.lower())
        t2_file    = next(f for f in files if "-t2w" in f.lower())
        flair_file = next(f for f in files if "-t2f" in f.lower())
        seg_file   = next(f for f in files if "-seg" in f.lower())

    except StopIteration:
        print(f"Skipping {case}")
        continue

    # -------------------------------------------------

    t1 = nib.load(os.path.join(case_dir, t1_file)).get_fdata()
    t1ce = nib.load(os.path.join(case_dir, t1ce_file)).get_fdata()
    t2 = nib.load(os.path.join(case_dir, t2_file)).get_fdata()
    flair = nib.load(os.path.join(case_dir, flair_file)).get_fdata()

    seg = nib.load(os.path.join(case_dir, seg_file)).get_fdata()

    # -------------------------------------------------
    # masks
    # -------------------------------------------------

    brain_mask = (
        (t1 > 0) |
        (t1ce > 0) |
        (t2 > 0) |
        (flair > 0)
    )

    healthy_mask = (seg == 0) & brain_mask

    wt_mask = seg > 0
    tc_mask = np.isin(seg, [1, 3])
    et_mask = seg == 3

    # -------------------------------------------------

    for mod_name, volume in zip(
        modalities,
        [t1, t1ce, t2, flair]
    ):

        h = volume[healthy_mask]
        w = volume[wt_mask]
        t = volume[tc_mask]
        e = volume[et_mask]

        h = h[h > 0]
        w = w[w > 0]
        t = t[t > 0]
        e = e[e > 0]

        # subsample per case

        if len(h) > MAX_SAMPLE_PER_CASE:
            h = np.random.choice(
                h,
                MAX_SAMPLE_PER_CASE,
                replace=False
            )

        if len(w) > MAX_SAMPLE_PER_CASE:
            w = np.random.choice(
                w,
                MAX_SAMPLE_PER_CASE,
                replace=False
            )

        if len(t) > MAX_SAMPLE_PER_CASE:
            t = np.random.choice(
                t,
                MAX_SAMPLE_PER_CASE,
                replace=False
            )

        if len(e) > MAX_SAMPLE_PER_CASE:
            e = np.random.choice(
                e,
                MAX_SAMPLE_PER_CASE,
                replace=False
            )

        healthy[mod_name].append(h)
        wt[mod_name].append(w)
        tc[mod_name].append(t)
        et[mod_name].append(e)

# =====================================================
# CONCAT
# =====================================================

for mod in modalities:

    healthy[mod] = np.concatenate(healthy[mod])
    wt[mod]      = np.concatenate(wt[mod])
    tc[mod]      = np.concatenate(tc[mod])
    et[mod]      = np.concatenate(et[mod])

    print(
        mod,
        len(healthy[mod]),
        len(wt[mod]),
        len(tc[mod]),
        len(et[mod])
    )

# =====================================================
# KDE PLOT
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = {
    "Healthy": "black",
    "WT": "blue",
    "TC": "green",
    "ET": "red"
}

for ax, mod in zip(axes, modalities):

    curves = {
        "Healthy": healthy[mod],
        "WT": wt[mod],
        "TC": tc[mod],
        "ET": et[mod]
    }

    sampled = {}

    for label, values in curves.items():

        if len(values) > KDE_SAMPLE:
            values = np.random.choice(
                values,
                KDE_SAMPLE,
                replace=False
            )

        sampled[label] = values

    all_values = np.concatenate(
        list(sampled.values())
    )

    xmin = np.percentile(all_values, 1)
    xmax = np.percentile(all_values, 99)

    x = np.linspace(
        xmin,
        xmax,
        500
    )

    for label, values in sampled.items():

        if len(values) < 10:
            continue

        kde = gaussian_kde(values)

        ax.plot(
            x,
            kde(x),
            linewidth=2.5,
            color=colors[label],
            label=label
        )

    ax.set_title(mod, fontsize=14)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.show()