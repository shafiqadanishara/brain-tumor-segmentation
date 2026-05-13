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

from src.dataset.dataset3D import BraTSDataset3D
from src.models.unet import UNet3D
from src.utils.metrics import compute_metrics
from src.dataset.postprocess import restore_to_original, save_nifti


MODALITY_CHANNELS = {
    "t2_t1ce":    [2, 1],
    "t1ce_flair": [1, 3],
    "t2_flair":   [2, 3],
    # "t1_t1ce":    [0, 1],
    # "t1_flair":   [0, 3],
    # "t1_t2":      [0, 2],
}


def save_comparison(path, pred, gt):
    """
    Predicted vs Ground Truth side by side for ET, TC, WT
    pred, gt: (3, D, H, W)
    """
    z = pred.shape[1] // 2

    regions = ["WT", "TC", "ET"]
    cmaps   = ["Greens", "Blues", "Reds"]
    ch      = [0, 1, 2]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for row, (name, cmap, c) in enumerate(zip(regions, cmaps, ch)):

        axes[row, 0].imshow(pred[c, z], cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_title(f"Pred {name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt[c, z], cmap=cmap, vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT {name}")
        axes[row, 1].axis("off")

    plt.suptitle("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_full_seg(path, pred):
    """
    Overlay WT / TC / ET on one RGB image
    """
    z    = pred.shape[1] // 2
    H, W = pred.shape[2], pred.shape[3]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    rgb[pred[0, z] > 0.5] = [0.0, 0.8, 0.0]   # WT green
    rgb[pred[1, z] > 0.5] = [0.0, 0.0, 1.0]   # TC blue
    rgb[pred[2, z] > 0.5] = [1.0, 0.0, 0.0]   # ET red

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("Full Segmentation")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_single_region(path, region_map, title, cmap):
    """
    Save one region map
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

    # =========================
    # LOAD TEST CASES
    # =========================

    with open(args.test_json, "r") as f:
        test_cases = json.load(f)

    print(f"Test cases: {len(test_cases)}")

    full_dataset = BraTSDataset3D(
        "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        augment=False
    )

    all_cases = full_dataset.cases

    test_idx = [all_cases.index(c) for c in test_cases]

    dataset = Subset(full_dataset, test_idx)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    # =========================
    # MODEL
    # =========================

    model = UNet3D(
        in_channels=2,
        out_channels=3
    ).to(device)

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )

    model.eval()

    # =========================
    # OUTPUT
    # =========================

    fold_name = Path(args.checkpoint).stem

    out_root = Path(
        f"experiments/dual/output_test/{args.modality}/{fold_name}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    channels = MODALITY_CHANNELS[args.modality]

    # =========================
    # TEST LOOP
    # =========================

    with torch.no_grad():

        for img, mask, meta in tqdm(loader):

            img  = img.to(device)
            mask = mask.to(device)

            case = meta["case"][0]

            # modality selection
            x = img[:, channels]

            logits = model(x)

            pred = (torch.sigmoid(logits) > 0.5).float()

            # =========================
            # METRICS
            # =========================

            metrics = compute_metrics(logits, mask)
            metrics["case"] = case

            rows.append(metrics)

            # =========================
            # NUMPY
            # =========================

            pred_np = pred[0].cpu().numpy()   # (3,D,H,W)
            gt_np   = mask[0].cpu().numpy()

            # =========================
            # ORIGINAL MRI METADATA
            # =========================

            affine = meta["affine"][0].numpy()

            original_shape = (
                meta["original_shape"][0]
                .numpy()
                .astype(int)
            )

            bbox = (
                meta["bbox"][0]
                .numpy()
                .astype(int)
            )

            # =========================
            # RESTORE TO ORIGINAL SPACE
            # =========================

            restored = restore_to_original(
                pred_np,
                original_shape,
                bbox
            )

            gt_restored = restore_to_original(
                gt_np,
                original_shape,
                bbox
            )

            # =========================
            # OUTPUT FOLDER
            # =========================

            case_dir = out_root / case
            case_dir.mkdir(parents=True, exist_ok=True)

            # ==================================================
            # 128³ VISUALS
            # ==================================================

            save_comparison(
                case_dir / "comparison_pred_gt_128.png",
                pred_np,
                gt_np
            )

            save_full_seg(
                case_dir / "seg_full_128.png",
                pred_np
            )

            save_single_region(
                case_dir / "seg_et_128.png",
                pred_np[2],
                "Predicted ET",
                "Reds"
            )

            save_single_region(
                case_dir / "seg_tc_128.png",
                pred_np[1],
                "Predicted TC",
                "Blues"
            )

            save_single_region(
                case_dir / "seg_wt_128.png",
                pred_np[0],
                "Predicted WT",
                "Greens"
            )

            # ==================================================
            # ORIGINAL MRI SPACE VISUALS
            # ==================================================

            save_comparison(
                case_dir / "comparison_pred_gt_original.png",
                restored,
                gt_restored
            )

            save_full_seg(
                case_dir / "seg_full_original.png",
                restored
            )

            save_single_region(
                case_dir / "seg_et_original.png",
                restored[2],
                "Predicted ET (original)",
                "Reds"
            )

            save_single_region(
                case_dir / "seg_tc_original.png",
                restored[1],
                "Predicted TC (original)",
                "Blues"
            )

            save_single_region(
                case_dir / "seg_wt_original.png",
                restored[0],
                "Predicted WT (original)",
                "Greens"
            )

            # ==================================================
            # RAW ARRAYS (128³)
            # ==================================================

            np.save(
                case_dir / "pred_full.npy",
                pred_np
            )

            np.save(
                case_dir / "pred_wt.npy",
                pred_np[0]
            )

            np.save(
                case_dir / "pred_tc.npy",
                pred_np[1]
            )

            np.save(
                case_dir / "pred_et.npy",
                pred_np[2]
            )

            # ==================================================
            # ORIGINAL MRI SPACE ARRAYS
            # ==================================================

            np.save(
                case_dir / "pred_full_original.npy",
                restored
            )

            # ==================================================
            # NIFTI OUTPUTS
            # ==================================================

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )

            save_nifti(
                restored[0],
                affine,
                case_dir / "pred_wt_original.nii.gz"
            )

            save_nifti(
                restored[1],
                affine,
                case_dir / "pred_tc_original.nii.gz"
            )

            save_nifti(
                restored[2],
                affine,
                case_dir / "pred_et_original.nii.gz"
            )

    # ==================================================
    # SAVE METRICS
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

    metrics_list = df.to_dict(orient="records")

    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    # ==================================================
    # DONE
    # ==================================================

    print(f"\nResults saved to: {out_root}")

    print(df.mean(numeric_only=True))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        choices=list(MODALITY_CHANNELS.keys())
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--test_json",
        default="data/folds/test.json",
        help="Path to test.json"
    )

    args = parser.parse_args()

    main(args)