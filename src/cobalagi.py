import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# ============================
# Load data
# ============================

dsc = pd.read_excel("src/wilcoxon_hd95.xlsx")
volume = pd.read_csv("src/volumes.csv")      # atau read_excel()

folds = [1,2,3]
regions = ["WT","TC","ET"]

print("="*80)
print("CORRELATION BETWEEN ΔHD95 AND TUMOR VOLUME")
print("="*80)
print(f"{'Region':<6}{'Pearson r':>15}{'p':>12}{'Spearman ρ':>15}{'p':>12}")
print("="*80)

for region in regions:

    delta_all = []
    volume_all = []

    for f in folds:

        base = f"hd95_{region}_t1ce_flair_{f}"
        ens  = f"hd95_{region}_ensemble_{f}"

        temp = pd.DataFrame({
            "base": pd.to_numeric(dsc[base], errors="coerce"),
            "ens": pd.to_numeric(dsc[ens], errors="coerce"),
            "vol": volume[f"vol_{region}"]
        })

        temp = temp.dropna()

        delta = temp["ens"] - temp["base"]

        delta_all.extend(delta.tolist())
        volume_all.extend(temp["vol"].tolist())

    delta_all = np.array(delta_all)
    volume_all = np.array(volume_all)

    r_p, p_p = pearsonr(volume_all, delta_all)
    r_s, p_s = spearmanr(volume_all, delta_all)

    print(f"{region:<6}{r_p:>15.4f}{p_p:>12.4g}{r_s:>15.4f}{p_s:>12.4g}")