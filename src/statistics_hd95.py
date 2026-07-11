import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df_dsc = pd.read_excel("src/wilcoxon.xlsx")
df_sp = pd.read_excel("src/wilcoxon_sens_prec.xlsx")
df_hd95 = pd.read_excel("src/wilcoxon_hd95.xlsx")

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

def get_pair(df, prefix, region, model1, model2):
    all_m1, all_m2 = [], []
    for f in folds:
        c1 = f"{prefix}_{region}_{model1}_{f}"
        c2 = f"{prefix}_{region}_{model2}_{f}"
        t = df[[c1, c2]].apply(pd.to_numeric, errors="coerce").dropna()
        all_m1.extend(t[c1].tolist())
        all_m2.extend(t[c2].tolist())
    return np.array(all_m1), np.array(all_m2)

def get_single(df, prefix, region, model):
    all_vals = []
    for f in folds:
        col = f"{prefix}_{region}_{model}_{f}"
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        all_vals.extend(vals)
    return np.array(all_vals)

def run_wilcoxon_block(df, prefix, title, model1, model2, higher_better=True):
    print("=" * 75)
    print(f"WILCOXON {title}: {model1.upper()} vs {model2.upper()}")
    print("=" * 75)
    print(f"{'Region':<6} {'N':>5} {'Statistic':>12} {'P-value':>10} {'r':>8}  {'Interp':<8} {'Arah'}")
    print("=" * 75)
    for region in regions:
        b, e = get_pair(df, prefix, region, model1, model2)
        mask = b != e
        stat, p = wilcoxon(b[mask], e[mask], zero_method="wilcox")
        r = rank_biserial(b[mask], e[mask])
        interp = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
        if higher_better:
            arah = "↑ ensemble lebih baik" if r > 0 else "↓ baseline lebih baik"
        else:
            arah = "↓ ensemble lebih baik" if r < 0 else "↑ baseline lebih baik"
        sig = "✓" if p < 0.05 else "✗"
        print(f"{region:<6} {mask.sum():>5} {stat:>12.3f} {p:>10.6f} {r:>8.4f}  {interp:<8} {sig} {arah}")
    print()

# ============================================================
# WILCOXON — DSC
# ============================================================
run_wilcoxon_block(df_dsc, "dsc", "DSC", "t1ce_flair", "ensemble", higher_better=True)
run_wilcoxon_block(df_dsc, "dsc", "DSC", "t2_t1ce", "ensemble", higher_better=True)

# ============================================================
# WILCOXON — HD95
# ============================================================
run_wilcoxon_block(df_hd95, "hd95", "HD95", "t1ce_flair", "ensemble", higher_better=False)
run_wilcoxon_block(df_hd95, "hd95", "HD95", "t2_t1ce", "ensemble", higher_better=False)

# ============================================================
# WILCOXON — SENSITIVITY & PRECISION
# ============================================================
run_wilcoxon_block(df_sp, "sens", "SENSITIVITY", "t1ce_flair", "ensemble", higher_better=True)
run_wilcoxon_block(df_sp, "prec", "PRECISION", "t1ce_flair", "ensemble", higher_better=True)

# ============================================================
# THRESHOLD ANALYSIS — DSC
# ============================================================
threshold = 0.8
print("=" * 75)
print(f"THRESHOLD ANALYSIS DSC (< {threshold} = gagal)")
print("=" * 75)
for region in regions:
    all_f, all_t, all_e = [], [], []
    for f in folds:
        cf = f"dsc_{region}_t1ce_flair_{f}"
        ct = f"dsc_{region}_t2_t1ce_{f}"
        ce = f"dsc_{region}_ensemble_{f}"
        t = df_dsc[[cf, ct, ce]].apply(pd.to_numeric, errors="coerce").dropna()
        all_f.extend(t[cf].tolist())
        all_t.extend(t[ct].tolist())
        all_e.extend(t[ce].tolist())

    f_arr, t_arr, e_arr = np.array(all_f), np.array(all_t), np.array(all_e)
    N = len(f_arr)
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

# ============================================================
# PLOT 1: Box plot DSC
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
colors = ["#4C72B0", "#DD8452", "#55A868"]
models = ["t1ce_flair", "t2_t1ce", "ensemble"]
labels = ["T1ce+FLAIR", "T2+T1ce", "Ensemble"]

for ri, region in enumerate(regions):
    ax = axes[ri]
    data = [get_single(df_dsc, "dsc", region, m) for m in models]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"DSC — {region}", fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("DSC" if ri == 0 else "")
    ax.set_ylim([-0.1, 1.1])
    ax.grid(axis="y", alpha=0.3)

patches = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Distribusi DSC per Region", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("boxplot_dsc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: boxplot_dsc.png")

# ============================================================
# PLOT 2: Scatter ΔDSC vs ΔHD95
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_region = {"WT": "#4C72B0", "TC": "#DD8452", "ET": "#55A868"}

for ri, region in enumerate(regions):
    ax = axes[ri]
    all_ddsc, all_dhd = [], []

    for f in folds:
        c_fd = f"dsc_{region}_t1ce_flair_{f}"
        c_ed = f"dsc_{region}_ensemble_{f}"
        c_fh = f"hd95_{region}_t1ce_flair_{f}"
        c_eh = f"hd95_{region}_ensemble_{f}"

        dsc = df_dsc[[c_fd, c_ed]].apply(pd.to_numeric, errors="coerce").dropna()
        hd = df_hd95[[c_fh, c_eh]].apply(pd.to_numeric, errors="coerce").dropna()

        ddsc = (dsc[c_ed] - dsc[c_fd]).tolist()
        dhd = (hd[c_eh] - hd[c_fh]).tolist()
        min_len = min(len(ddsc), len(dhd))
        all_ddsc.extend(ddsc[:min_len])
        all_dhd.extend(dhd[:min_len])

    all_ddsc = np.array(all_ddsc)
    all_dhd = np.array(all_dhd)

    ax.scatter(all_ddsc, all_dhd, alpha=0.3, s=15, color=colors_region[region])
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)

    N = len(all_ddsc)
    q1 = ((all_ddsc > 0) & (all_dhd < 0)).sum()
    q2 = ((all_ddsc < 0) & (all_dhd < 0)).sum()
    q3 = ((all_ddsc < 0) & (all_dhd > 0)).sum()
    q4 = ((all_ddsc > 0) & (all_dhd > 0)).sum()
    q0 = N - q1 - q2 - q3 - q4

    ax.text(0.98, 0.98, f"DSC↑ HD↓\n{q1/N*100:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(0.02, 0.98, f"DSC↓ HD↓\n{q2/N*100:.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(0.02, 0.02, f"DSC↓ HD↑\n{q3/N*100:.1f}%", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(0.98, 0.02, f"DSC↑ HD↑\n{q4/N*100:.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="orange",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(f"{region}", fontsize=13, fontweight="bold")
    ax.set_xlabel("ΔDSC (Ensemble − T1ce+FLAIR)")
    ax.set_ylabel("ΔHD95 (Ensemble − T1ce+FLAIR)" if ri == 0 else "")
    ax.grid(alpha=0.3)

plt.suptitle("ΔDSC vs ΔHD95 per Kasus (Ensemble − T1ce+FLAIR)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("scatter_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scatter_delta.png")