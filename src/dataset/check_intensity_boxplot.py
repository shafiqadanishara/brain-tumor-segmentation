import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

DATA_ROOT = r"data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

N_CASES = 200          # None = semua kasus
MAX_SAMPLE_CASE = 3000
BOX_SAMPLE = 5000

# =====================================================
# STORAGE
# =====================================================

modalities = ["T1", "T1ce", "T2", "FLAIR"]

healthy = {m: [] for m in modalities}
wt = {m: [] for m in modalities}
tc = {m: [] for m in modalities}
et = {m: [] for m in modalities}

# =====================================================
# CASE LIST
# =====================================================

cases = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
])

if N_CASES is not None:
    cases = cases[:N_CASES]

print(f"Using {len(cases)} cases")

# =====================================================
# LOAD DATA
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
    # Brain mask
    # -------------------------------------------------

    brain_mask = (
        (t1 > 0) |
        (t1ce > 0) |
        (t2 > 0) |
        (flair > 0)
    )

    # Healthy tissue
    healthy_mask = (seg == 0) & brain_mask

    # BraTS2023 labels
    # WT = seluruh tumor
    wt_mask = seg > 0

    # TC = NCR + ET
    tc_mask = np.isin(seg, [1, 3])

    # ET
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

        if len(h) > MAX_SAMPLE_CASE:
            h = np.random.choice(
                h,
                MAX_SAMPLE_CASE,
                replace=False
            )

        if len(w) > MAX_SAMPLE_CASE:
            w = np.random.choice(
                w,
                MAX_SAMPLE_CASE,
                replace=False
            )

        if len(t) > MAX_SAMPLE_CASE:
            t = np.random.choice(
                t,
                MAX_SAMPLE_CASE,
                replace=False
            )

        if len(e) > MAX_SAMPLE_CASE:
            e = np.random.choice(
                e,
                MAX_SAMPLE_CASE,
                replace=False
            )

        healthy[mod_name].append(h)
        wt[mod_name].append(w)
        tc[mod_name].append(t)
        et[mod_name].append(e)

# =====================================================
# CONCATENATE
# =====================================================

for mod in modalities:

    healthy[mod] = np.concatenate(healthy[mod])
    wt[mod] = np.concatenate(wt[mod])
    tc[mod] = np.concatenate(tc[mod])
    et[mod] = np.concatenate(et[mod])

    print(
        f"{mod}: "
        f"Healthy={len(healthy[mod]):,} "
        f"WT={len(wt[mod]):,} "
        f"TC={len(tc[mod]):,} "
        f"ET={len(et[mod]):,}"
    )

# =====================================================
# BOXPLOT
# =====================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes = axes.flatten()

for ax, mod in zip(axes, modalities):

    h = healthy[mod]
    w = wt[mod]
    t = tc[mod]
    e = et[mod]

    if len(h) > BOX_SAMPLE:
        h = np.random.choice(h, BOX_SAMPLE, replace=False)

    if len(w) > BOX_SAMPLE:
        w = np.random.choice(w, BOX_SAMPLE, replace=False)

    if len(t) > BOX_SAMPLE:
        t = np.random.choice(t, BOX_SAMPLE, replace=False)

    if len(e) > BOX_SAMPLE:
        e = np.random.choice(e, BOX_SAMPLE, replace=False)

    bp = ax.boxplot(
        [h, w, t, e],
        labels=["Healthy", "WT", "TC", "ET"],
        patch_artist=True,
        showfliers=False,
        widths=0.6
    )

    colors = [
        "#BDBDBD",  # Healthy
        "#64B5F6",  # WT
        "#81C784",  # TC
        "#E57373"   # ET
    ]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title(
        mod,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Intensity")

plt.suptitle(
    "Intensity Distribution Across MRI Modalities",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()