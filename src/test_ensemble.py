# =========================
# test_ensemble.py
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

from src.models.unet import UNet3D
from src.models.dual_ensemble import DualEnsemble
from src.dataset.dataset3D import BraTSDataset3D
from src.utils.metrics import compute_metrics
from src.dataset.postprocess import restore_to_original, save_nifti


# --------------------------------------------------
# SLICE HELPERS
# --------------------------------------------------

def get_best_slice(vol):
    """vol: (3, D, H, W)"""
    if vol.sum() == 0:
        return vol.shape[1] // 2
    return int(np.argmax(vol.sum(axis=(0, 2, 3))))


def get_best_slice_single(vol):
    """vol: (D, H, W)"""
    if vol.sum() == 0:
        return vol.shape[0] // 2
    return int(np.argmax(vol.sum(axis=(1, 2))))


def get_top_slices(seg, n_slices=4):

    tumor_per_slice = seg.sum(axis=(0, 2, 3))

    if tumor_per_slice.sum() == 0:
        D = seg.shape[1]
        return [D // 2]

    top_idx = np.argsort(tumor_per_slice)[::-1]

    selected = []

    min_gap = 8

    for idx in top_idx:

        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(int(idx))

        if len(selected) == n_slices:
            break

    return sorted(selected)


def norm_slice(x):

    mn, mx = x.min(), x.max()

    return (x - mn) / (mx - mn + 1e-8)


# --------------------------------------------------
# OVERLAY BUILDER
# --------------------------------------------------

def build_overlay(mri_slice, seg_slice):

    mri_norm = norm_slice(mri_slice)

    rgb = np.stack(
        [mri_norm, mri_norm, mri_norm],
        axis=-1
    )

    rgb[seg_slice[0] > 0.5] = [0.0, 0.8, 0.0]   # WT
    rgb[seg_slice[1] > 0.5] = [1.0, 1.0, 0.0]   # TC
    rgb[seg_slice[2] > 0.5] = [1.0, 0.0, 0.0]   # ET

    return rgb


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def save_summary(path, flair_vol, t1_vol, t1ce_vol, t2_vol, pred, gt):

    slice_idxs = get_top_slices(pred, n_slices=4)

    vols = [
        flair_vol,
        t1_vol,
        t1ce_vol,
        t2_vol
    ]

    col_labels = [
        "Flair",
        "T1",
        "T1ce",
        "T2",
        "Ground-truth",
        "Prediction"
    ]

    n_rows = len(slice_idxs)
    n_cols = 6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        facecolor="black"
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, z in enumerate(slice_idxs):

        # MRI modalities
        for col, vol in enumerate(vols):

            slc = vol[z]

            axes[row, col].imshow(
                norm_slice(slc),
                cmap="gray"
            )

            axes[row, col].axis("off")

        # GT
        gt_overlay = build_overlay(
            flair_vol[z],
            gt[:, z]
        )

        axes[row, 4].imshow(gt_overlay)
        axes[row, 4].axis("off")

        # Prediction
        pred_overlay = build_overlay(
            flair_vol[z],
            pred[:, z]
        )

        axes[row, 5].imshow(pred_overlay)
        axes[row, 5].axis("off")

        # Slice label
        axes[row, 0].text(
            -0.22,
            0.5,
            f"Slice {z}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            color="white",
            fontweight="bold"
        )

    # Bottom labels
    for col, label in enumerate(col_labels):

        axes[-1, col].text(
            0.5,
            -0.14,
            label,
            transform=axes[-1, col].transAxes,
            ha="center",
            va="top",
            fontsize=16,
            color="white",
            fontweight="bold"
        )

    plt.subplots_adjust(
        wspace=0.01,
        hspace=0.01,
        left=0.08,
        right=0.99,
        top=0.99,
        bottom=0.10
    )

    plt.savefig(
        path,
        dpi=220,
        facecolor="black",
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

def save_comparison(path, pred, gt):

    z = get_best_slice(pred)

    regions = ["WT", "TC", "ET"]
    cmaps   = ["Greens", "Blues", "Reds"]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for row, (name, cmap) in enumerate(zip(regions, cmaps)):

        axes[row, 0].imshow(
            pred[row, z],
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        axes[row, 0].set_title(f"Pred {name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            gt[row, z],
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        axes[row, 1].set_title(f"GT {name}")
        axes[row, 1].axis("off")

    plt.suptitle("Prediction vs Ground Truth")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------------------------------
# FULL COMPARISON
# --------------------------------------------------

def save_full_comparison(path, pred, gt):

    z = get_best_slice(pred)

    H, W = pred.shape[2], pred.shape[3]

    def make_rgb(seg):

        rgb = np.zeros((H, W, 3), dtype=np.float32)

        rgb[seg[0, z] > 0.5] = [0.0, 0.8, 0.0]
        rgb[seg[1, z] > 0.5] = [1.0, 0.0, 0.0]
        rgb[seg[2, z] > 0.5] = [0.0, 0.0, 1.0]

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
# MAIN
# --------------------------------------------------

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==================================================
    # LOAD TEST CASES
    # ==================================================

    with open(args.test_json, "r") as f:
        test_cases = json.load(f)

    print(f"Test cases: {len(test_cases)}")

    full_dataset = BraTSDataset3D(
        "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        augment=False
    )

    all_cases = full_dataset.cases

    test_idx = [
        all_cases.index(c)
        for c in test_cases
    ]

    dataset = Subset(full_dataset, test_idx)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    # ==================================================
    # LOAD MODELS
    # ==================================================

    model_a = UNet3D(
        in_channels=2,
        out_channels=3
    ).to(device)

    model_b = UNet3D(
        in_channels=2,
        out_channels=3
    ).to(device)

    model_a.load_state_dict(
        torch.load(args.model_a, map_location=device)
    )

    model_b.load_state_dict(
        torch.load(args.model_b, map_location=device)
    )

    model = DualEnsemble(
        model_a,
        model_b,
        out_channels=3
    ).to(device)

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )

    model.eval()

    fold_name = Path(args.checkpoint).stem

    out_root = Path(
        f"experiments/ensemble/output_test/{fold_name}"
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    with torch.no_grad():

        for img, mask, meta in tqdm(loader):

            img  = img.to(device)
            mask = mask.to(device)

            case = meta["case"][0]

            # ==================================================
            # LOAD ORIGINAL MRI
            # ==================================================

            case_path = os.path.join(
                "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
                case
            )

            files = os.listdir(case_path)

            def load_mod(keyword):

                f = [x for x in files if keyword in x][0]

                return np.transpose(
                    nib.load(
                        os.path.join(case_path, f)
                    ).get_fdata(),
                    (2, 0, 1)
                )

            orig_t1    = load_mod("t1n")
            orig_t1ce  = load_mod("t1c")
            orig_t2    = load_mod("t2w")
            orig_flair = load_mod("t2f")

            affine = nib.load(
                os.path.join(
                    case_path,
                    [x for x in files if "t1n" in x][0]
                )
            ).affine

            # ==================================================
            # 128³ MRI
            # ==================================================

            flair_128 = img[0, 3].cpu().numpy()
            t1_128    = img[0, 0].cpu().numpy()
            t1ce_128  = img[0, 1].cpu().numpy()
            t2_128    = img[0, 2].cpu().numpy()

            # ==================================================
            # ENSEMBLE INPUTS
            # ==================================================

            x1 = img[:, [1, 3]]   # T1ce + Flair
            x2 = img[:, [2, 1]]   # T2 + T1ce

            logits = model(x1, x2)

            pred = (
                torch.sigmoid(logits) > 0.5
            ).float()

            metrics = compute_metrics(logits, mask)

            metrics["case"] = case

            rows.append(metrics)

            pred_np = pred[0].cpu().numpy()
            gt_np   = mask[0].cpu().numpy()

            # ==================================================
            # RESTORE ORIGINAL SPACE
            # ==================================================

            restored = restore_to_original(
                pred_np,
                orig_t1
            )

            gt_restored = restore_to_original(
                gt_np,
                orig_t1
            )

            case_dir = out_root / case

            case_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ==================================================
            # 128³ VISUALS
            # ==================================================

            save_summary(
                case_dir / "summary_128.png",
                flair_128,
                t1_128,
                t1ce_128,
                t2_128,
                pred_np,
                gt_np
            )

            save_comparison(
                case_dir / "comparison_pred_gt_128.png",
                pred_np,
                gt_np
            )

            save_full_comparison(
                case_dir / "comparison_full_128.png",
                pred_np,
                gt_np
            )

            # ==================================================
            # ORIGINAL VISUALS
            # ==================================================

            save_summary(
                case_dir / "summary_original.png",
                orig_flair,
                orig_t1,
                orig_t1ce,
                orig_t2,
                restored,
                gt_restored
            )

            save_comparison(
                case_dir / "comparison_pred_gt_original.png",
                restored,
                gt_restored
            )

            save_full_comparison(
                case_dir / "comparison_full_original.png",
                restored,
                gt_restored
            )

            # ==================================================
            # RAW ARRAYS
            # ==================================================

            np.save(
                case_dir / "pred_full.npy",
                pred_np
            )

            np.save(
                case_dir / "pred_full_original.npy",
                restored
            )

            # ==================================================
            # NIFTI
            # ==================================================

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )

    # ==================================================
    # METRICS
    # ==================================================

    df = pd.DataFrame(rows)

    df.to_csv(
        out_root / "metrics.csv",
        index=False
    )

    df.mean(
        numeric_only=True
    ).to_csv(
        out_root / "metrics_mean.csv"
    )

    with open(out_root / "metrics.json", "w") as f:

        json.dump(
            df.to_dict(orient="records"),
            f,
            indent=2
        )

    print(f"\nResults saved to: {out_root}")

    print(
        df.mean(numeric_only=True)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_a",
        required=True
    )

    parser.add_argument(
        "--model_b",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--test_json",
        default="data/folds/test.json"
    )

    args = parser.parse_args()

    main(args)