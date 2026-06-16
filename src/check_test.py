import json
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

FOLDS_JSON = "data/folds/folds.json"
TEST_JSON  = "data/folds/test.json"

DATA_ROOT = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

OUT_PATH = "experiments/dual/output/ablation/test_distribution_analysis.png"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# =====================================================
# LOAD SPLITS
# =====================================================

with open(FOLDS_JSON, "r") as f:
    folds = json.load(f)

with open(TEST_JSON, "r") as f:
    test_cases = json.load(f)

# =====================================================
# MERGE ALL VALIDATION CASES
# =====================================================

all_val_cases = []

for fold in folds:
    all_val_cases.extend(fold["val"])

print("=" * 60)
print("DATASET DISTRIBUTION ANALYSIS")
print("=" * 60)
print(f"Validation cases : {len(all_val_cases)}")
print(f"Test cases       : {len(test_cases)}")
print("=" * 60)

# =====================================================
# HELPERS
# =====================================================

def load_seg(case):

    case_path = os.path.join(DATA_ROOT, case)

    if not os.path.exists(case_path):
        print(f"[WARNING] Missing case: {case}")
        return None

    files = os.listdir(case_path)

    seg_file = [f for f in files if "seg" in f.lower()][0]

    seg = nib.load(
        os.path.join(case_path, seg_file)
    ).get_fdata()

    return seg


def get_stats(cases, desc="Loading"):

    wt_vols = []
    tc_vols = []
    et_vols = []

    imbalance = []
    et_ratio = []

    for case in tqdm(
        cases,
        desc=desc,
        leave=True
    ):

        try:

            seg = load_seg(case)

            if seg is None:
                continue

            wt = (seg > 0)
            tc = np.isin(seg, [1, 3])
            et = (seg == 3)

            wt_count = wt.sum()
            tc_count = tc.sum()
            et_count = et.sum()

            wt_vols.append(wt_count)
            tc_vols.append(tc_count)
            et_vols.append(et_count)

            imbalance.append(
                wt_count / seg.size * 100
            )

            if wt_count > 0:
                et_ratio.append(
                    et_count / wt_count * 100
                )

        except Exception as e:
            print(f"\n[ERROR] {case}: {e}")

    return {
        "wt": np.array(wt_vols),
        "tc": np.array(tc_vols),
        "et": np.array(et_vols),
        "imbalance": np.array(imbalance),
        "et_ratio": np.array(et_ratio)
    }


# =====================================================
# COMPUTE STATS
# =====================================================

print("\nLoading validation set...")
val_stats = get_stats(
    all_val_cases,
    desc="Validation"
)

print("\nLoading test set...")
test_stats = get_stats(
    test_cases,
    desc="Test"
)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"WT mean volume     : {val_stats['wt'].mean():.0f}")
print(f"TC mean volume     : {val_stats['tc'].mean():.0f}")
print(f"ET mean volume     : {val_stats['et'].mean():.0f}")
print(f"Class imbalance    : {val_stats['imbalance'].mean():.2f}%")
print(f"ET / WT ratio      : {val_stats['et_ratio'].mean():.2f}%")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(f"WT mean volume     : {test_stats['wt'].mean():.0f}")
print(f"TC mean volume     : {test_stats['tc'].mean():.0f}")
print(f"ET mean volume     : {test_stats['et'].mean():.0f}")
print(f"Class imbalance    : {test_stats['imbalance'].mean():.2f}%")
print(f"ET / WT ratio      : {test_stats['et_ratio'].mean():.2f}%")

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "Validation vs Test Distribution Analysis",
    fontsize=16,
    fontweight="bold"
)

# =====================================================
# Tumor Volume Distribution
# =====================================================

axes[0, 0].boxplot(
    [
        val_stats["wt"],
        val_stats["tc"],
        val_stats["et"],
        test_stats["wt"],
        test_stats["tc"],
        test_stats["et"]
    ],
    labels=[
        "Val WT",
        "Val TC",
        "Val ET",
        "Test WT",
        "Test TC",
        "Test ET"
    ]
)

axes[0, 0].set_title("Tumor Volume Distribution")
axes[0, 0].set_ylabel("Voxel Count")

# =====================================================
# Class Imbalance
# =====================================================

axes[0, 1].hist(
    val_stats["imbalance"],
    bins=25,
    alpha=0.6,
    label=f"Val ({val_stats['imbalance'].mean():.2f}%)"
)

axes[0, 1].hist(
    test_stats["imbalance"],
    bins=25,
    alpha=0.6,
    label=f"Test ({test_stats['imbalance'].mean():.2f}%)"
)

axes[0, 1].set_title("Class Imbalance (WT)")
axes[0, 1].set_xlabel("Tumor Voxels (%)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].legend()

# =====================================================
# ET Volume
# =====================================================

axes[1, 0].boxplot(
    [
        val_stats["et"],
        test_stats["et"]
    ],
    labels=[
        "Validation",
        "Test"
    ]
)

axes[1, 0].set_title("ET Volume Distribution")
axes[1, 0].set_ylabel("Voxel Count")

# =====================================================
# ET / WT Ratio
# =====================================================

axes[1, 1].hist(
    val_stats["et_ratio"],
    bins=25,
    alpha=0.6,
    label=f"Val ({val_stats['et_ratio'].mean():.2f}%)"
)

axes[1, 1].hist(
    test_stats["et_ratio"],
    bins=25,
    alpha=0.6,
    label=f"Test ({test_stats['et_ratio'].mean():.2f}%)"
)

axes[1, 1].set_title("ET / WT Ratio")
axes[1, 1].set_xlabel("ET as % of WT")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()

plt.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\n" + "=" * 60)
print("Saved to:")
print(OUT_PATH)
print("=" * 60)

plt.show()