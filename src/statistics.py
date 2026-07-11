import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_excel("src/wilcoxon.xlsx")

folds = [1, 2, 3]
regions = ["WT", "TC", "ET"]

def rank_biserial(x, y):
    diff = y - x
    diff = diff[diff != 0]
    n = len(diff)
    ranks = pd.Series(np.abs(diff)).rank()
    W_plus = ranks[diff > 0].sum()
    W_minus = ranks[diff < 0].sum()
    return (W_plus - W_minus) / (n * (n + 1) / 2)

# ============================================================
# 1. WILCOXON: Ensemble vs T1ce+FLAIR
# ============================================================
print("=" * 70)
print("WILCOXON: Ensemble vs T1ce+FLAIR")
print("=" * 70)
print(f"{'Region':<6} {'N':>5} {'Statistic':>12} {'P-value':>10} {'Effect Size (r)':>16} {'Interpretasi'}")
print("=" * 70)

for region in regions:
    all_base, all_ens = [], []
    for f in folds:
        c1 = f"dsc_{region}_t1ce_flair_{f}"
        c2 = f"dsc_{region}_ensemble_{f}"
        temp = df[[c1, c2]].dropna()
        temp[c1] = pd.to_numeric(temp[c1], errors="coerce")
        temp[c2] = pd.to_numeric(temp[c2], errors="coerce")
        temp = temp.dropna()
        all_base.extend(temp[c1].tolist())
        all_ens.extend(temp[c2].tolist())

    b, e = np.array(all_base), np.array(all_ens)
    mask = b != e
    stat, p = wilcoxon(b[mask], e[mask], zero_method="wilcox")
    r = rank_biserial(b[mask], e[mask])
    interp = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
    print(f"{region:<6} {mask.sum():>5} {stat:>12.3f} {p:>10.6f} {r:>16.4f}  {interp}")

# ============================================================
# 2. WILCOXON: Ensemble vs T2+T1ce
# ============================================================
print("\n" + "=" * 70)
print("WILCOXON: Ensemble vs T2+T1ce")
print("=" * 70)
print(f"{'Region':<6} {'N':>5} {'Statistic':>12} {'P-value':>10} {'Effect Size (r)':>16} {'Interpretasi'}")
print("=" * 70)

for region in regions:
    all_base, all_ens = [], []
    for f in folds:
        c1 = f"dsc_{region}_t2_t1ce_{f}"
        c2 = f"dsc_{region}_ensemble_{f}"
        temp = df[[c1, c2]].dropna()
        temp[c1] = pd.to_numeric(temp[c1], errors="coerce")
        temp[c2] = pd.to_numeric(temp[c2], errors="coerce")
        temp = temp.dropna()
        all_base.extend(temp[c1].tolist())
        all_ens.extend(temp[c2].tolist())

    b, e = np.array(all_base), np.array(all_ens)
    mask = b != e
    stat, p = wilcoxon(b[mask], e[mask], zero_method="wilcox")
    r = rank_biserial(b[mask], e[mask])
    interp = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
    print(f"{region:<6} {mask.sum():>5} {stat:>12.3f} {p:>10.6f} {r:>16.4f}  {interp}")

# ============================================================
# 3. THRESHOLD ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS (DSC < 0.8 = gagal)")
print("=" * 70)

threshold = 0.8

for region in regions:
    all_f, all_t, all_e = [], [], []
    for f in folds:
        cf = f"dsc_{region}_t1ce_flair_{f}"
        ct = f"dsc_{region}_t2_t1ce_{f}"
        ce = f"dsc_{region}_ensemble_{f}"
        temp = df[[cf, ct, ce]].copy()
        for c in [cf, ct, ce]:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")
        temp = temp.dropna()
        all_f.extend(temp[cf].tolist())
        all_t.extend(temp[ct].tolist())
        all_e.extend(temp[ce].tolist())

    f_arr = np.array(all_f)
    t_arr = np.array(all_t)
    e_arr = np.array(all_e)
    N = len(f_arr)

    both_fail     = ((f_arr < threshold) & (t_arr < threshold)).sum()
    only_flair    = ((f_arr < threshold) & (t_arr >= threshold)).sum()
    only_t2       = ((f_arr >= threshold) & (t_arr < threshold)).sum()
    both_success  = ((f_arr >= threshold) & (t_arr >= threshold)).sum()

    # ensemble bisa bantu di kasus only_flair fail
    ens_rescue = ((f_arr < threshold) & (t_arr >= threshold) & (e_arr >= threshold)).sum()

    print(f"\n{region} (N={N}):")
    print(f"  Keduanya gagal              : {both_fail:>4} ({both_fail/N*100:.1f}%)")
    print(f"  Hanya T1ce+FLAIR gagal      : {only_flair:>4} ({only_flair/N*100:.1f}%)  <- T2+T1ce bisa bantu")
    print(f"  Hanya T2+T1ce gagal         : {only_t2:>4} ({only_t2/N*100:.1f}%)")
    print(f"  Keduanya sukses             : {both_success:>4} ({both_success/N*100:.1f}%)")
    print(f"  Ensemble berhasil rescue    : {ens_rescue:>4} dari {only_flair} kasus komplementer")

# import nibabel as nib
# import numpy as np
# import json
# import os
# import pandas as pd

# # Load test cases
# with open("data/folds/test.json") as f:
#     test_cases = json.load(f)

# # Path ke folder dataset BraTS
# data_dir = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"  # ganti ini

# records = []
# for case in test_cases:
#     seg_path = os.path.join(data_dir, case, f"{case}-seg.nii.gz")
#     seg = nib.load(seg_path).get_fdata()
    
#     # Label BraTS: 1=NCR(TC), 2=ED(WT-TC), 3=ET
#     vol_WT = np.sum(seg > 0)
#     vol_TC = np.sum((seg == 1) | (seg == 3))  # NCR + ET = TC
#     vol_ET = np.sum(seg == 3)
    
#     records.append({
#         "case": case,
#         "vol_WT": vol_WT,
#         "vol_TC": vol_TC,
#         "vol_ET": vol_ET,
#     })

# df = pd.DataFrame(records)
# df.to_csv("src/volumes.csv", index=False)
# print(df.describe())