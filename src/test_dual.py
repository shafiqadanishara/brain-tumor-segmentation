# =========================
# test_dual.py
# Standardized to (C,D,H,W)
# Reads test cases from data/folds/test.json
# Save visuals + postprocess + nii.gz
# =========================

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import os

from src.dataset.dataset3D import BraTSDataset3D
from src.models.unet import UNet3D
from src.utils.metrics import compute_metrics
from src.dataset.postprocess import restore_to_original, save_nifti


MODALITY_CHANNELS = {
    "t2_t1ce":    [2, 1],
    "t1ce_flair": [1, 3],
    "t2_flair":   [2, 3],
}


# --------------------------------------------------
# SLICE HELPERS
# --------------------------------------------------

def get_best_slice(vol):
    """vol: (3, D, H, W) — find slice with most tumor"""
    if vol.sum() == 0:
        return vol.shape[1] // 2
    return int(np.argmax(vol.sum(axis=(0, 2, 3))))


def get_best_slice_single(vol):
    """vol: (D, H, W) — find slice with most tumor"""
    if vol.sum() == 0:
        return vol.shape[0] // 2
    return int(np.argmax(vol.sum(axis=(1, 2))))


# --------------------------------------------------
# OVERLAY BUILDER (seg on MRI background)
# --------------------------------------------------

def build_overlay(mri_slice, seg_slice):
    """
    mri_slice : (H, W) float
    seg_slice : (3, H, W) binary — WT, TC, ET
    Color: WT=green, TC=yellow, ET=red
    Returns: (H, W, 3) float [0,1]
    """
    mri_min, mri_max = mri_slice.min(), mri_slice.max()
    mri_norm = (mri_slice - mri_min) / (mri_max - mri_min + 1e-8)
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    rgb[seg_slice[0] > 0.5] = [0.0, 0.8, 0.0]   # WT green
    rgb[seg_slice[1] > 0.5] = [1.0, 1.0, 0.0]   # TC yellow
    rgb[seg_slice[2] > 0.5] = [1.0, 0.0, 0.0]   # ET red
    return rgb


LEGEND = [
    Patch(facecolor='green',  label='WT'),
    Patch(facecolor='yellow', label='TC'),
    Patch(facecolor='red',    label='ET'),
]


# --------------------------------------------------
# 1. SUMMARY — all sequences + GT + Pred WITH brain background
#    Rows = slices, Cols = [Flair, T1, T1ce, T2, GT, Pred]
# --------------------------------------------------

def save_summary(path, img_4ch, pred, gt, n_slices=4):
    """
    img_4ch : (4, D, H, W) — [t1, t1ce, t2, flair]
    pred    : (3, D, H, W)
    gt      : (3, D, H, W)
    """
    best_z = get_best_slice(pred)
    D      = pred.shape[1]
    half   = n_slices // 2
    start  = max(0, best_z - half)
    end    = min(D, start + n_slices)
    slices = list(range(start, end))

    n_cols = 6
    fig, axes = plt.subplots(
        len(slices), n_cols,
        figsize=(3 * n_cols, 3 * len(slices)),
        facecolor="black"
    )
    if len(slices) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Flair", "T1", "T1ce", "T2", "Ground Truth", "Prediction"]

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    for row, z in enumerate(slices):
        flair = img_4ch[3, z]
        t1    = img_4ch[0, z]
        t1ce  = img_4ch[1, z]
        t2    = img_4ch[2, z]

        for col, seq in enumerate([flair, t1, t1ce, t2]):
            axes[row, col].imshow(norm(seq), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], color="white", fontsize=10)

        gt_rgb = build_overlay(flair, gt[:, z])
        axes[row, 4].imshow(gt_rgb)
        axes[row, 4].axis("off")
        if row == 0:
            axes[row, 4].set_title(col_titles[4], color="white", fontsize=10)

        pred_rgb = build_overlay(flair, pred[:, z])
        axes[row, 5].imshow(pred_rgb)
        axes[row, 5].axis("off")
        if row == 0:
            axes[row, 5].set_title(col_titles[5], color="white", fontsize=10)

    fig.legend(handles=LEGEND, loc='lower center', ncol=3,
               fontsize=11, facecolor='black', labelcolor='white')
    plt.suptitle("Summary", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200, facecolor="black")
    plt.close()


# --------------------------------------------------
# 2. COMPARISON per region — mask only, NO brain
# --------------------------------------------------

def save_comparison(path, pred, gt):
    """pred, gt: (3, D, H, W) — mask only"""
    z = get_best_slice(pred)

    regions = ["WT", "TC", "ET"]
    cmaps   = ["Greens", "Blues", "Reds"]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for row, (name, cmap) in enumerate(zip(regions, cmaps)):
        axes[row, 0].imshow(pred[row, z], cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_title(f"Pred {name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt[row, z], cmap=cmap, vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT {name}")
        axes[row, 1].axis("off")

    plt.suptitle("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------------------------------
# 3. FULL COMPARISON — colored mask only, NO brain
# --------------------------------------------------

def save_full_comparison(path, pred, gt):
    """pred, gt: (3, D, H, W) — colored mask, no brain"""
    z    = get_best_slice(pred)
    H, W = pred.shape[2], pred.shape[3]

    def make_rgb(seg):
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[seg[0, z] > 0.5] = [0.0, 0.8, 0.0]   # WT green
        rgb[seg[1, z] > 0.5] = [1.0, 0.0, 0.0]   # TC red
        rgb[seg[2, z] > 0.5] = [0.0, 0.0, 1.0]   # ET blue
        return rgb

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(make_rgb(pred))
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(make_rgb(gt))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    plt.suptitle("Full Segmentation Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------------------------------
# 4. SINGLE REGION — mask only, NO brain
# --------------------------------------------------

def save_single_region(path, region_map, title, cmap):
    """region_map: (D, H, W) — mask only"""
    z = get_best_slice_single(region_map)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(region_map[z], cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.test_json, "r") as f:
        test_cases = json.load(f)

    print(f"Test cases: {len(test_cases)}")

    full_dataset = BraTSDataset3D(
        "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        augment=False
    )

    all_cases = full_dataset.cases
    test_idx  = [all_cases.index(c) for c in test_cases]
    dataset   = Subset(full_dataset, test_idx)
    loader    = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=2, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fold_name = Path(args.checkpoint).stem
    out_root  = Path(f"experiments/dual/output_test/{args.modality}/{fold_name}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows     = []
    channels = MODALITY_CHANNELS[args.modality]

    with torch.no_grad():

        for img, mask, meta in tqdm(loader):

            img  = img.to(device)
            mask = mask.to(device)
            case = meta["case"][0]

            case_path = os.path.join(
                "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
                case
            )
            files   = os.listdir(case_path)
            t1_file = [f for f in files if "t1n" in f][0]

            original_t1 = np.transpose(
                nib.load(os.path.join(case_path, t1_file)).get_fdata(),
                (2, 0, 1)
            )

            img_4ch = img[0].cpu().numpy()   # (4, D, H, W)

            x      = img[:, channels]
            logits = model(x)
            pred   = (torch.sigmoid(logits) > 0.5).float()

            metrics         = compute_metrics(logits, mask)
            metrics["case"] = case
            rows.append(metrics)

            pred_np     = pred[0].cpu().numpy()
            gt_np       = mask[0].cpu().numpy()
            affine      = meta["affine"][0].numpy()

            restored    = restore_to_original(pred_np, original_t1)
            gt_restored = restore_to_original(gt_np,   original_t1)

            case_dir = out_root / case
            case_dir.mkdir(parents=True, exist_ok=True)

            # ==================================================
            # 128³ VISUALS
            # ==================================================

            # Summary: all sequences + GT + Pred WITH brain
            save_summary(
                case_dir / "summary_128.png",
                img_4ch, pred_np, gt_np, n_slices=4
            )

            # Per region: mask only
            save_comparison(
                case_dir / "comparison_pred_gt_128.png",
                pred_np, gt_np
            )

            # Full GT vs Pred: mask only
            save_full_comparison(
                case_dir / "comparison_full_128.png",
                pred_np, gt_np
            )

            # Single regions: mask only
            save_single_region(case_dir / "seg_wt_128.png", pred_np[0], "Predicted WT", "Greens")
            save_single_region(case_dir / "seg_tc_128.png", pred_np[1], "Predicted TC", "Blues")
            save_single_region(case_dir / "seg_et_128.png", pred_np[2], "Predicted ET", "Reds")

            # ==================================================
            # ORIGINAL MRI SPACE VISUALS
            # ==================================================

            # Summary original — use t1 as background for all 4 cols
            orig_4ch = np.stack([original_t1] * 4, axis=0)
            save_summary(
                case_dir / "summary_original.png",
                orig_4ch, restored, gt_restored, n_slices=4
            )

            save_comparison(
                case_dir / "comparison_pred_gt_original.png",
                restored, gt_restored
            )
            save_full_comparison(
                case_dir / "comparison_full_original.png",
                restored, gt_restored
            )
            save_single_region(case_dir / "seg_wt_original.png", restored[0], "Predicted WT (original)", "Greens")
            save_single_region(case_dir / "seg_tc_original.png", restored[1], "Predicted TC (original)", "Blues")
            save_single_region(case_dir / "seg_et_original.png", restored[2], "Predicted ET (original)", "Reds")

            # ==================================================
            # RAW ARRAYS
            # ==================================================

            np.save(case_dir / "pred_full.npy",          pred_np)
            np.save(case_dir / "pred_wt.npy",            pred_np[0])
            np.save(case_dir / "pred_tc.npy",            pred_np[1])
            np.save(case_dir / "pred_et.npy",            pred_np[2])
            np.save(case_dir / "pred_full_original.npy", restored)

            # ==================================================
            # NIFTI
            # ==================================================

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )
            save_nifti(restored[0], affine, case_dir / "pred_wt_original.nii.gz")
            save_nifti(restored[1], affine, case_dir / "pred_tc_original.nii.gz")
            save_nifti(restored[2], affine, case_dir / "pred_et_original.nii.gz")

    # ==================================================
    # METRICS
    # ==================================================

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)
    df.mean(numeric_only=True).to_csv(out_root / "metrics_mean.csv")

    with open(out_root / "metrics.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"\nResults saved to: {out_root}")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        choices=list(MODALITY_CHANNELS.keys())
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--test_json",
        default="data/folds/test.json"
    )

    args = parser.parse_args()
    main(args)