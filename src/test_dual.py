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
    """
    vol: (3, D, H, W) — find slice with most tumor
    Returns: int slice index along D axis
    """
    tumor_per_slice = vol.sum(axis=(0, 2, 3))  # sum over C, H, W
    best = int(np.argmax(tumor_per_slice))
    # fallback to middle if no tumor found
    if vol.sum() == 0:
        best = vol.shape[1] // 2
    return best


def get_best_slice_single(vol):
    """
    vol: (D, H, W) — find slice with most tumor
    """
    tumor_per_slice = vol.sum(axis=(1, 2))
    best = int(np.argmax(tumor_per_slice))
    if vol.sum() == 0:
        best = vol.shape[0] // 2
    return best


# --------------------------------------------------
# OVERLAY HELPER
# --------------------------------------------------

def make_overlay(mri_slice, pred_slice, gt_slice):
    """
    Overlay segmentation on MRI background.

    mri_slice : (H, W) float — MRI grayscale
    pred_slice: (3, H, W) binary — WT, TC, ET
    gt_slice  : (3, H, W) binary — WT, TC, ET

    Returns: pred_rgb, gt_rgb — (H, W, 3) float [0,1]

    Color scheme (matching reference image):
        WT = green
        TC = yellow (WT + TC overlap)
        ET = red
    """
    # Normalize MRI to [0, 1]
    mri_min, mri_max = mri_slice.min(), mri_slice.max()
    if mri_max > mri_min:
        mri_norm = (mri_slice - mri_min) / (mri_max - mri_min)
    else:
        mri_norm = mri_slice.copy()

    def build_rgb(seg, background):
        H, W = background.shape
        rgb = np.stack([background, background, background], axis=-1)  # grayscale base

        # WT — green
        wt_mask = seg[0] > 0.5
        rgb[wt_mask] = [0.0, 0.8, 0.0]

        # TC — yellow (overrides WT)
        tc_mask = seg[1] > 0.5
        rgb[tc_mask] = [1.0, 1.0, 0.0]

        # ET — red (overrides TC)
        et_mask = seg[2] > 0.5
        rgb[et_mask] = [1.0, 0.0, 0.0]

        return rgb

    pred_rgb = build_rgb(pred_slice, mri_norm)
    gt_rgb   = build_rgb(gt_slice,   mri_norm)

    return pred_rgb, gt_rgb


# --------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------

def save_comparison(path, pred, gt, mri=None):
    """
    Predicted vs Ground Truth side by side for WT, TC, ET.
    If mri provided, overlay on MRI background.

    pred, gt : (3, D, H, W)
    mri      : (D, H, W) optional
    """
    z = get_best_slice(pred)

    regions = ["WT", "TC", "ET"]
    cmaps   = ["Greens", "Blues", "Reds"]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for row, (name, cmap) in enumerate(zip(regions, cmaps)):
        if mri is not None:
            mri_min, mri_max = mri[z].min(), mri[z].max()
            mri_norm = (mri[z] - mri_min) / (mri_max - mri_min + 1e-8)
            axes[row, 0].imshow(mri_norm, cmap="gray")
            axes[row, 1].imshow(mri_norm, cmap="gray")
            axes[row, 0].imshow(pred[row, z], cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            axes[row, 1].imshow(gt[row, z],   cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        else:
            axes[row, 0].imshow(pred[row, z], cmap=cmap, vmin=0, vmax=1)
            axes[row, 1].imshow(gt[row, z],   cmap=cmap, vmin=0, vmax=1)

        axes[row, 0].set_title(f"Pred {name}")
        axes[row, 1].set_title(f"GT {name}")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.suptitle("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_full_comparison(path, pred, gt, mri=None):
    """
    Full segmentation overlay comparison: Pred vs GT side by side.

    pred, gt : (3, D, H, W)
    mri      : (D, H, W) optional — used as background
    """
    z = get_best_slice(pred)

    if mri is not None:
        mri_slice = mri[z]
    else:
        # blank background
        H, W = pred.shape[2], pred.shape[3]
        mri_slice = np.zeros((H, W))

    pred_rgb, gt_rgb = make_overlay(mri_slice, pred[:, z], gt[:, z])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pred_rgb)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green',  label='WT'),
        Patch(facecolor='yellow', label='TC'),
        Patch(facecolor='red',    label='ET'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    plt.suptitle("Full Segmentation Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_full_seg(path, pred, mri=None):
    """
    Overlay WT / TC / ET on MRI background (single panel).

    pred : (3, D, H, W)
    mri  : (D, H, W) optional
    """
    z = get_best_slice(pred)

    if mri is not None:
        mri_slice = mri[z]
    else:
        H, W = pred.shape[2], pred.shape[3]
        mri_slice = np.zeros((H, W))

    # build overlay (only pred, no gt needed)
    mri_min, mri_max = mri_slice.min(), mri_slice.max()
    mri_norm = (mri_slice - mri_min) / (mri_max - mri_min + 1e-8)
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)

    rgb[pred[0, z] > 0.5] = [0.0, 0.8, 0.0]   # WT green
    rgb[pred[1, z] > 0.5] = [1.0, 1.0, 0.0]   # TC yellow
    rgb[pred[2, z] > 0.5] = [1.0, 0.0, 0.0]   # ET red

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green',  label='WT'),
        Patch(facecolor='yellow', label='TC'),
        Patch(facecolor='red',    label='ET'),
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("Full Segmentation")
    ax.axis("off")
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_single_region(path, region_map, title, cmap, mri=None):
    """
    Save one region map, optionally overlaid on MRI.

    region_map : (D, H, W)
    mri        : (D, H, W) optional
    """
    z = get_best_slice_single(region_map)

    fig, ax = plt.subplots(figsize=(5, 5))

    if mri is not None:
        mri_min, mri_max = mri[z].min(), mri[z].max()
        mri_norm = (mri[z] - mri_min) / (mri_max - mri_min + 1e-8)
        ax.imshow(mri_norm, cmap="gray")
        ax.imshow(region_map[z], cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    else:
        ax.imshow(region_map[z], cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------------------------------
# MULTI-SLICE OVERVIEW (reference-style grid)
# --------------------------------------------------

def save_multislice_overview(path, pred, gt, mri, n_slices=4):
    """
    Grid visualization like reference image:
    Rows = slices, Cols = [Flair/MRI, GT overlay, Pred overlay]

    pred, gt : (3, D, H, W)
    mri      : (D, H, W) — used as background (flair or t1)
    """
    best_z = get_best_slice(pred)
    D = pred.shape[1]

    # pick n_slices around best_z
    half = n_slices // 2
    start = max(0, best_z - half)
    end   = min(D, start + n_slices)
    slices = list(range(start, end))

    fig, axes = plt.subplots(len(slices), 3, figsize=(12, 4 * len(slices)))

    if len(slices) == 1:
        axes = axes[np.newaxis, :]

    for row, z in enumerate(slices):
        mri_min, mri_max = mri[z].min(), mri[z].max()
        mri_norm = (mri[z] - mri_min) / (mri_max - mri_min + 1e-8)

        # Col 0 — MRI only
        axes[row, 0].imshow(mri_norm, cmap="gray")
        axes[row, 0].set_title(f"MRI (slice {z})")
        axes[row, 0].axis("off")

        # Col 1 — GT overlay
        _, gt_rgb = make_overlay(mri[z], pred[:, z] * 0, gt[:, z])
        axes[row, 1].imshow(gt_rgb)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        # Col 2 — Pred overlay
        pred_rgb, _ = make_overlay(mri[z], pred[:, z], gt[:, z] * 0)
        axes[row, 2].imshow(pred_rgb)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green',  label='WT'),
        Patch(facecolor='yellow', label='TC'),
        Patch(facecolor='red',    label='ET'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    plt.suptitle("Multi-slice Overview", fontsize=14)
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

            # flair for overlay background (channel 3)
            flair_128 = img[0, 3].cpu().numpy()   # (D,H,W) normalized flair

            x      = img[:, channels]
            logits = model(x)
            pred   = (torch.sigmoid(logits) > 0.5).float()

            metrics         = compute_metrics(logits, mask)
            metrics["case"] = case
            rows.append(metrics)

            pred_np = pred[0].cpu().numpy()   # (3,D,H,W)
            gt_np   = mask[0].cpu().numpy()
            affine  = meta["affine"][0].numpy()

            restored = restore_to_original(pred_np, original_t1)
            gt_restored = restore_to_original(gt_np, original_t1)

            case_dir = out_root / case
            case_dir.mkdir(parents=True, exist_ok=True)

            # ---- 128³ VISUALS (with MRI overlay) ----

            save_comparison(
                case_dir / "comparison_pred_gt_128.png",
                pred_np, gt_np, mri=flair_128
            )
            save_full_seg(
                case_dir / "seg_full_128.png",
                pred_np, mri=flair_128
            )
            save_full_comparison(
                case_dir / "comparison_full_128.png",
                pred_np, gt_np, mri=flair_128
            )
            save_single_region(
                case_dir / "seg_et_128.png",
                pred_np[2], "Predicted ET", "Reds", mri=flair_128
            )
            save_single_region(
                case_dir / "seg_tc_128.png",
                pred_np[1], "Predicted TC", "Blues", mri=flair_128
            )
            save_single_region(
                case_dir / "seg_wt_128.png",
                pred_np[0], "Predicted WT", "Greens", mri=flair_128
            )

            # ---- MULTI-SLICE OVERVIEW (reference-style) ----
            save_multislice_overview(
                case_dir / "multislice_overview.png",
                pred_np, gt_np, mri=flair_128, n_slices=4
            )

            # ---- ORIGINAL SPACE VISUALS ----

            save_comparison(
                case_dir / "comparison_pred_gt_original.png",
                restored, gt_restored, mri=original_t1
            )
            save_full_seg(
                case_dir / "seg_full_original.png",
                restored, mri=original_t1
            )
            save_full_comparison(
                case_dir / "comparison_full_original.png",
                restored, gt_restored, mri=original_t1
            )
            save_single_region(
                case_dir / "seg_et_original.png",
                restored[2], "Predicted ET (original)", "Reds", mri=original_t1
            )
            save_single_region(
                case_dir / "seg_tc_original.png",
                restored[1], "Predicted TC (original)", "Blues", mri=original_t1
            )
            save_single_region(
                case_dir / "seg_wt_original.png",
                restored[0], "Predicted WT (original)", "Greens", mri=original_t1
            )

            # ---- RAW ARRAYS ----

            np.save(case_dir / "pred_full.npy",    pred_np)
            np.save(case_dir / "pred_wt.npy",      pred_np[0])
            np.save(case_dir / "pred_tc.npy",      pred_np[1])
            np.save(case_dir / "pred_et.npy",      pred_np[2])
            np.save(case_dir / "pred_full_original.npy", restored)

            # ---- NIFTI ----

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )
            save_nifti(restored[0], affine, case_dir / "pred_wt_original.nii.gz")
            save_nifti(restored[1], affine, case_dir / "pred_tc_original.nii.gz")
            save_nifti(restored[2], affine, case_dir / "pred_et_original.nii.gz")

    # ---- METRICS ----

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