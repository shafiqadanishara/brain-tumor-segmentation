import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_excel("src/wilcoxon_sens_prec.xlsx")

folds = [1, 2, 3]
regions = ["WT", "TC", "ET"]
metrics = {"sens": "Sensitivity", "prec": "Precision"}

def rank_biserial(x, y):
    diff = y - x
    diff = diff[diff != 0]
    n = len(diff)
    ranks = pd.Series(np.abs(diff)).rank()
    W_plus = ranks[diff > 0].sum()
    W_minus = ranks[diff < 0].sum()
    return (W_plus - W_minus) / (n * (n + 1) / 2)

def get_combined(metric, region, model):
    all_vals = []
    for f in folds:
        col = f"{metric}_{region}_{model}_{f}"
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        all_vals.extend(vals)
    return np.array(all_vals)

# ============================================================
# WILCOXON: Ensemble vs T1ce+FLAIR (sens & prec)
# ============================================================
for metric, label in metrics.items():
    print("=" * 70)
    print(f"WILCOXON {label.upper()}: Ensemble vs T1ce+FLAIR")
    print("=" * 70)
    print(f"{'Region':<6} {'N':>5} {'Statistic':>12} {'P-value':>10} {'Effect Size (r)':>16} {'Interpretasi'}")
    print("=" * 70)

    for region in regions:
        b = get_combined(metric, region, "t1ce_flair")
        e = get_combined(metric, region, "ensemble")
        min_len = min(len(b), len(e))
        b, e = b[:min_len], e[:min_len]
        mask = b != e
        stat, p = wilcoxon(b[mask], e[mask], zero_method="wilcox")
        r = rank_biserial(b[mask], e[mask])
        interp = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
        print(f"{region:<6} {mask.sum():>5} {stat:>12.3f} {p:>10.6f} {r:>16.4f}  {interp}")
    print()

# ============================================================
# THRESHOLD ANALYSIS sens & prec
# ============================================================
threshold = 0.8

for metric, label in metrics.items():
    print("=" * 70)
    print(f"THRESHOLD ANALYSIS {label.upper()} (< {threshold} = gagal)")
    print("=" * 70)

    for region in regions:
        f_arr = get_combined(metric, region, "t1ce_flair")
        t_arr = get_combined(metric, region, "t2_t1ce")
        e_arr = get_combined(metric, region, "ensemble")
        min_len = min(len(f_arr), len(t_arr), len(e_arr))
        f_arr, t_arr, e_arr = f_arr[:min_len], t_arr[:min_len], e_arr[:min_len]
        N = min_len

        both_fail    = ((f_arr < threshold) & (t_arr < threshold)).sum()
        only_flair   = ((f_arr < threshold) & (t_arr >= threshold)).sum()
        only_t2      = ((f_arr >= threshold) & (t_arr < threshold)).sum()
        both_success = ((f_arr >= threshold) & (t_arr >= threshold)).sum()
        ens_rescue   = ((f_arr < threshold) & (t_arr >= threshold) & (e_arr >= threshold)).sum()

        print(f"\n{region} (N={N}):")
        print(f"  Keduanya gagal              : {both_fail:>4} ({both_fail/N*100:.1f}%)")
        print(f"  Hanya T1ce+FLAIR gagal      : {only_flair:>4} ({only_flair/N*100:.1f}%)  <- T2+T1ce bisa bantu")
        print(f"  Hanya T2+T1ce gagal         : {only_t2:>4} ({only_t2/N*100:.1f}%)")
        print(f"  Keduanya sukses             : {both_success:>4} ({both_success/N*100:.1f}%)")
        print(f"  Ensemble berhasil rescue    : {ens_rescue:>4} dari {only_flair} kasus komplementer")
    print()