# import json
# with open('data/folds/folds.json') as f:
#     folds = json.load(f)
# fold2 = folds[0]
# print('Train cases:', len(fold2['train']))
# print('Val cases  :', len(fold2['val']))

# import json
# import os
# import numpy as np
# import nibabel as nib

# with open('data/folds/folds.json') as f:
#     folds = json.load(f)

# root = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# def get_tumor_volume(case):
#     case_path = os.path.join(root, case)
#     files = os.listdir(case_path)
#     seg_file = [f for f in files if "seg" in f][0]
#     seg = nib.load(os.path.join(case_path, seg_file)).get_fdata()
#     return (seg > 0).sum()  # jumlah voxel tumor

# for fold in folds:
#     fold_idx = fold['fold']
    
#     train_vols = [get_tumor_volume(c) for c in fold['train'][:20]]  # sample 20
#     val_vols   = [get_tumor_volume(c) for c in fold['val'][:20]]

#     print(f"\nFold {fold_idx}")
#     print(f"  Train tumor volume (mean): {np.mean(train_vols):.0f} voxels")
#     print(f"  Val   tumor volume (mean): {np.mean(val_vols):.0f} voxels")

# import json, os
# import numpy as np
# import nibabel as nib

# with open('data/folds/folds.json') as f:
#     folds = json.load(f)

# root = 'data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

# def stats(cases, n=30):
#     vols, ets = [], []
#     for c in cases[:n]:
#         files = os.listdir(os.path.join(root, c))
#         seg_f = [f for f in files if 'seg' in f][0]
#         seg = nib.load(os.path.join(root, c, seg_f)).get_fdata()
#         vols.append((seg > 0).sum())
#         ets.append((seg == 3).sum())  # ET only
#     return np.mean(vols), np.std(vols), np.mean(ets)

# fold1 = folds[1]
# tm, ts, te = stats(fold1['train'])
# vm, vs, ve = stats(fold1['val'])
# print(f'Train | mean={tm:.0f} std={ts:.0f} ET={te:.0f}')
# print(f'Val   | mean={vm:.0f} std={vs:.0f} ET={ve:.0f}')

"""
src/check_distribution.py
=========================
Visualisasi distribusi dataset BraTS per fold:
1. Distribusi ukuran tumor (volume voxel) — WT, TC, ET
2. Histogram intensitas voxel per modalitas (T1, T1ce, T2, FLAIR)
3. Class imbalance ratio (tumor vs background)

Jalankan:
    py -m src.check_distribution
"""

import json
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ---- Config ----
FOLDS_JSON = "data/folds/folds.json"
DATA_ROOT  = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUT_PATH   = "experiments/dual/output/ablation/distribution_analysis.png"
N_SAMPLE   = 50  # kasus per split yang disampling (lebih banyak = lebih akurat)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ---- Load folds ----
with open(FOLDS_JSON) as f:
    folds = json.load(f)


def load_case(case):
    """Load seg + semua modalitas untuk satu case."""
    path  = os.path.join(DATA_ROOT, case)
    files = os.listdir(path)

    def find(kw):
        return [f for f in files if kw in f][0]

    seg   = nib.load(os.path.join(path, find("seg"))).get_fdata()
    t1    = nib.load(os.path.join(path, find("t1n"))).get_fdata()
    t1ce  = nib.load(os.path.join(path, find("t1c"))).get_fdata()
    t2    = nib.load(os.path.join(path, find("t2w"))).get_fdata()
    flair = nib.load(os.path.join(path, find("t2f"))).get_fdata()

    return seg, t1, t1ce, t2, flair


def get_stats(cases, n=N_SAMPLE):
    """Ambil stats untuk n kasus pertama."""
    cases = cases[:n]
    wt_vols, tc_vols, et_vols = [], [], []
    imbalance_ratios = []
    intensities = {"T1": [], "T1ce": [], "T2": [], "FLAIR": []}

    for case in tqdm(cases, desc="Loading", leave=False):
        seg, t1, t1ce, t2, flair = load_case(case)

        # Volume per region
        wt = (seg > 0)
        tc = np.isin(seg, [1, 3])
        et = (seg == 3)

        wt_vols.append(wt.sum())
        tc_vols.append(tc.sum())
        et_vols.append(et.sum())

        # Class imbalance: tumor / total voxels
        total = seg.size
        imbalance_ratios.append(wt.sum() / total * 100)

        # Intensitas dalam ROI tumor saja (lebih informatif)
        mask = wt
        if mask.sum() > 0:
            intensities["T1"].extend(t1[mask].flatten().tolist())
            intensities["T1ce"].extend(t1ce[mask].flatten().tolist())
            intensities["T2"].extend(t2[mask].flatten().tolist())
            intensities["FLAIR"].extend(flair[mask].flatten().tolist())

    return {
        "wt_vols":    np.array(wt_vols),
        "tc_vols":    np.array(tc_vols),
        "et_vols":    np.array(et_vols),
        "imbalance":  np.array(imbalance_ratios),
        "intensities": intensities,
    }


# ---- Collect stats per fold ----
print(f"Sampling {N_SAMPLE} cases per split per fold...")
fold_stats = []
for fold in folds:
    print(f"\n=== Fold {fold['fold']} ===")
    print(f"  Train ({len(fold['train'])} cases, sampling {min(N_SAMPLE, len(fold['train']))})")
    train_stats = get_stats(fold["train"])
    print(f"  Val   ({len(fold['val'])} cases, sampling {min(N_SAMPLE, len(fold['val']))})")
    val_stats   = get_stats(fold["val"])
    fold_stats.append((fold["fold"], train_stats, val_stats))

    # Print summary
    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        print(f"\n  {split_name}:")
        print(f"    WT  volume: mean={stats['wt_vols'].mean():.0f}  std={stats['wt_vols'].std():.0f}")
        print(f"    TC  volume: mean={stats['tc_vols'].mean():.0f}  std={stats['tc_vols'].std():.0f}")
        print(f"    ET  volume: mean={stats['et_vols'].mean():.0f}  std={stats['et_vols'].std():.0f}")
        print(f"    Imbalance : {stats['imbalance'].mean():.2f}% tumor voxels")


# ==============================
# PLOT
# ==============================
COLORS = {"Train": "#4C72B0", "Val": "#DD8452"}
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
REGIONS    = ["WT", "TC", "ET"]

n_folds = len(fold_stats)

fig = plt.figure(figsize=(22, 6 * n_folds))
fig.suptitle("BraTS Dataset Distribution Analysis", fontsize=16, fontweight="bold", y=1.01)

for row_idx, (fold_idx, train_s, val_s) in enumerate(fold_stats):
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.05, right=0.97,
                           top=1 - row_idx/n_folds - 0.02,
                           bottom=1 - (row_idx+1)/n_folds + 0.04,
                           wspace=0.35)

    fold_label = f"Fold {fold_idx}"

    # --- Plot 1: Tumor Volume Distribution (boxplot) ---
    ax1 = fig.add_subplot(gs[0])
    data_train = [train_s["wt_vols"], train_s["tc_vols"], train_s["et_vols"]]
    data_val   = [val_s["wt_vols"],   val_s["tc_vols"],   val_s["et_vols"]]

    positions_train = [1, 4, 7]
    positions_val   = [2, 5, 8]

    bp1 = ax1.boxplot(data_train, positions=positions_train, widths=0.7,
                      patch_artist=True,
                      boxprops=dict(facecolor=COLORS["Train"], alpha=0.7),
                      medianprops=dict(color="black", linewidth=2))
    bp2 = ax1.boxplot(data_val, positions=positions_val, widths=0.7,
                      patch_artist=True,
                      boxprops=dict(facecolor=COLORS["Val"], alpha=0.7),
                      medianprops=dict(color="black", linewidth=2))

    ax1.set_xticks([1.5, 4.5, 7.5])
    ax1.set_xticklabels(REGIONS)
    ax1.set_ylabel("Volume (voxels)")
    ax1.set_title(f"{fold_label} — Tumor Volume Distribution")
    ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Train", "Val"], loc="upper right")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # --- Plot 2: Class Imbalance ---
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(train_s["imbalance"], bins=20, alpha=0.7, color=COLORS["Train"],
             label=f"Train (mean={train_s['imbalance'].mean():.1f}%)")
    ax2.hist(val_s["imbalance"],   bins=20, alpha=0.7, color=COLORS["Val"],
             label=f"Val   (mean={val_s['imbalance'].mean():.1f}%)")
    ax2.set_xlabel("Tumor voxels (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{fold_label} — Class Imbalance (WT)")
    ax2.legend()

    # --- Plot 3: Intensity Histogram (T1ce + FLAIR — yang kamu pakai) ---
    ax3 = fig.add_subplot(gs[2])
    for mod, ls in [("T1ce", "-"), ("FLAIR", "--")]:
        t_vals = np.array(train_s["intensities"][mod])
        v_vals = np.array(val_s["intensities"][mod])

        # clip extreme outliers for visualization
        p1, p99 = np.percentile(t_vals, [1, 99])
        t_clipped = np.clip(t_vals, p1, p99)
        v_clipped = np.clip(v_vals, p1, p99)

        ax3.hist(t_clipped, bins=60, alpha=0.5, color=COLORS["Train"],
                 linestyle=ls, density=True, label=f"Train {mod}")
        ax3.hist(v_clipped, bins=60, alpha=0.5, color=COLORS["Val"],
                 linestyle=ls, density=True, label=f"Val {mod}")

    ax3.set_xlabel("Intensity (raw)")
    ax3.set_ylabel("Density")
    ax3.set_title(f"{fold_label} — Intensity Distribution (tumor ROI)")
    ax3.legend(fontsize=7)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")