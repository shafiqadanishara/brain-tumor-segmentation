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

from src.models.unet import UNet3D
from src.models.dual_ensemble import DualEnsemble
from src.dataset.dataset3D import BraTSDataset3D
from src.utils.metrics import compute_metrics
from src.dataset.postprocess import restore_to_original, save_nifti


def save_comparison(path, pred, gt):
    """
    1. Predicted vs Ground Truth side by side for ET, TC, WT
    pred, gt: (3, D, H, W)  — channels: WT=0, TC=1, ET=2
    """
    z       = pred.shape[1] // 2
    regions = ["WT", "TC", "ET"]
    cmaps   = ["Greens", "Blues", "Reds"]
    ch      = [0, 1, 2]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for row, (name, cmap, c) in enumerate(zip(regions, cmaps, ch)):
        axes[row, 0].imshow(pred[c, z, :, :], cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_title(f"Pred {name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt[c, z, :, :], cmap=cmap, vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT {name}")
        axes[row, 1].axis("off")

    plt.suptitle("Prediction vs Ground Truth", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_full_seg(path, pred):
    """
    2. All three regions overlaid on one slice (colour coded)
    pred: (3, D, H, W)
    """
    z    = pred.shape[1] // 2
    H, W = pred.shape[2], pred.shape[3]

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[pred[0, z] > 0.5] = [0.0, 0.8, 0.0]   # WT — green
    rgb[pred[1, z] > 0.5] = [0.0, 0.0, 1.0]   # TC — blue
    rgb[pred[2, z] > 0.5] = [1.0, 0.0, 0.0]   # ET — red

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("Full Segmentation (WT=green, TC=blue, ET=red)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_single_region(path, region_map, title, cmap):
    """
    3/4/5. Single region map
    region_map: (D, H, W)
    """
    z = region_map.shape[0] // 2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(region_map[z], cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load test cases ----
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

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- Load pretrained branches ----
    model_a = UNet3D(in_channels=2, out_channels=3).to(device)
    model_b = UNet3D(in_channels=2, out_channels=3).to(device)

    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    model = DualEnsemble(model_a, model_b, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_root = Path("outputs/ensemble/test")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    with torch.no_grad():
        for img, mask, meta in tqdm(loader):
            img  = img.to(device)
            mask = mask.to(device)

            case = meta["case"][0]

            x1 = img[:, [1, 3]]   # T1ce + Flair
            x2 = img[:, [2, 1]]   # T2   + T1ce

            logits = model(x1, x2)
            pred   = (torch.sigmoid(logits) > 0.5).float()

            metrics         = compute_metrics(logits, mask)
            metrics["case"] = case
            rows.append(metrics)

            pred_np = pred[0].cpu().numpy()    # (3,D,H,W)
            gt_np   = mask[0].cpu().numpy()    # (3,D,H,W)

            affine         = meta["affine"][0].numpy()
            original_shape = meta["original_shape"][0].numpy().astype(int)
            bbox           = meta["bbox"][0].numpy().astype(int)

            # Postprocess — restore to original space
            restored = restore_to_original(pred_np, original_shape, bbox)

            case_dir = out_root / case
            case_dir.mkdir(parents=True, exist_ok=True)

            # ---- 128^3 visuals ----
            # 1. Pred vs GT comparison (128^3)
            save_comparison(case_dir / "comparison_pred_gt_128.png", pred_np, gt_np)

            # 2. Full segmentation overlay (128^3)
            save_full_seg(case_dir / "seg_full_128.png", pred_np)

            # 3. ET only (128^3)
            save_single_region(
                case_dir / "seg_et_128.png",
                pred_np[2], "Predicted ET", "Reds"
            )

            # 4. TC only (128^3)
            save_single_region(
                case_dir / "seg_tc_128.png",
                pred_np[1], "Predicted TC", "Blues"
            )

            # 5. WT only (128^3)
            save_single_region(
                case_dir / "seg_wt_128.png",
                pred_np[0], "Predicted WT", "Greens"
            )

            # ---- Original resolution visuals ----
            # 1. Pred vs GT comparison (original) — GT also restored for fair comparison
            gt_restored = restore_to_original(gt_np, original_shape, bbox)
            save_comparison(case_dir / "comparison_pred_gt_original.png", restored, gt_restored)

            # 2. Full segmentation overlay (original)
            save_full_seg(case_dir / "seg_full_original.png", restored)

            # 3. ET only (original)
            save_single_region(
                case_dir / "seg_et_original.png",
                restored[2], "Predicted ET (original)", "Reds"
            )

            # 4. TC only (original)
            save_single_region(
                case_dir / "seg_tc_original.png",
                restored[1], "Predicted TC (original)", "Blues"
            )

            # 5. WT only (original)
            save_single_region(
                case_dir / "seg_wt_original.png",
                restored[0], "Predicted WT (original)", "Greens"
            )

            # ---- Raw arrays (128^3) ----
            np.save(case_dir / "pred_full.npy", pred_np)
            np.save(case_dir / "pred_wt.npy",   pred_np[0])
            np.save(case_dir / "pred_tc.npy",   pred_np[1])
            np.save(case_dir / "pred_et.npy",   pred_np[2])

            # ---- Restored to original space ----
            np.save(case_dir / "pred_full_original.npy", restored)

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )

            save_nifti(restored[0], affine, case_dir / "pred_wt_original.nii.gz")
            save_nifti(restored[1], affine, case_dir / "pred_tc_original.nii.gz")
            save_nifti(restored[2], affine, case_dir / "pred_et_original.nii.gz")

    # 6. CSV metrics per case
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)
    df.mean(numeric_only=True).to_csv(out_root / "metrics_mean.csv")

    # 7. JSON metrics per case
    metrics_list = df.to_dict(orient="records")
    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"\nResults saved to: {out_root}")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a",    required=True, help="checkpoint for t1ce_flair branch")
    parser.add_argument("--model_b",    required=True, help="checkpoint for t2_t1ce branch")
    parser.add_argument("--checkpoint", required=True, help="checkpoint for ensemble fusion head")
    parser.add_argument(
        "--test_json",
        default="data/folds/test.json",
        help="Path to test.json generated by splitDataset.py"
    )
    args = parser.parse_args()
    main(args)