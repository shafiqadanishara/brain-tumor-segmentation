# src/test_ensemble.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.unet import UNet3D
from src.models.dual_ensemble import DualEnsemble
from src.dataset.dataset_dual_ensemble import BraTSDualEnsembleDataset
from src.utils.metrics import compute_metrics

TARGET_SIZE = (128, 128, 128)


def save_nifti(arr, path):
    nii = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    nib.save(nii, str(path))


def save_visual(case_dir, x1, x2, gt, pred):
    z = pred.shape[1] // 2

    fig, ax = plt.subplots(2, 4, figsize=(18, 10))

    # Inputs
    ax[0, 0].imshow(x1[0, :, :, z], cmap="gray")
    ax[0, 0].set_title("T1ce")

    ax[0, 1].imshow(x1[1, :, :, z], cmap="gray")
    ax[0, 1].set_title("Flair")

    ax[0, 2].imshow(x2[0, :, :, z], cmap="gray")
    ax[0, 2].set_title("T2")

    ax[0, 3].imshow(gt.max(axis=0)[:, :, z], cmap="viridis")
    ax[0, 3].set_title("GT")

    # Prediction
    ax[1, 0].imshow(pred.max(axis=0)[:, :, z], cmap="viridis")
    ax[1, 0].set_title("Prediction")

    labels = ["WT", "TC", "ET"]
    cmaps = ["Greens", "Yellows", "Reds"]

    for i in range(3):
        ax[1, i + 1].imshow(pred[i, :, :, z], cmap=cmaps[i])
        ax[1, i + 1].set_title(labels[i])

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.savefig(case_dir / "comparison.png", dpi=200)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BraTSDualEnsembleDataset(
        f"data/split/{args.split}",
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Branch models
    model_a = UNet3D(in_channels=2, out_channels=3).to(device)
    model_b = UNet3D(in_channels=2, out_channels=3).to(device)

    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    # Ensemble model
    model = DualEnsemble(model_a, model_b, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_root = Path(f"outputs/ensemble/{args.split}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    with torch.no_grad():
        for idx, (x1, x2, mask) in enumerate(tqdm(loader, desc="Testing Ensemble")):
            x1 = x1.to(device)
            x2 = x2.to(device)
            mask = mask.to(device)

            if x1.shape[2:] != TARGET_SIZE:
                x1 = F.interpolate(x1, TARGET_SIZE, mode="trilinear", align_corners=False)

            if x2.shape[2:] != TARGET_SIZE:
                x2 = F.interpolate(x2, TARGET_SIZE, mode="trilinear", align_corners=False)

            if mask.shape[2:] != TARGET_SIZE:
                mask = F.interpolate(mask, TARGET_SIZE, mode="nearest")

            logits = model(x1, x2)
            pred = (torch.sigmoid(logits) > 0.5).float()

            metrics = compute_metrics(logits, mask)
            metrics["case"] = idx
            rows.append(metrics)

            pred_np = pred[0].cpu().numpy()
            gt_np = mask[0].cpu().numpy()
            x1_np = x1[0].cpu().numpy()
            x2_np = x2[0].cpu().numpy()

            case_dir = out_root / f"case_{idx:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # Save NPY
            np.save(case_dir / "pred_full.npy", pred_np)
            np.save(case_dir / "pred_wt.npy", pred_np[0])
            np.save(case_dir / "pred_tc.npy", pred_np[1])
            np.save(case_dir / "pred_et.npy", pred_np[2])

            # Save NIfTI
            save_nifti(pred_np.transpose(1,2,3,0), case_dir / "pred_full.nii.gz")
            save_nifti(pred_np[0], case_dir / "pred_wt.nii.gz")
            save_nifti(pred_np[1], case_dir / "pred_tc.nii.gz")
            save_nifti(pred_np[2], case_dir / "pred_et.nii.gz")

            save_visual(case_dir, x1_np, x2_np, gt_np, pred_np)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)

    mean_df = df.mean(numeric_only=True).to_frame(name="mean")
    mean_df.to_csv(out_root / "metrics_mean.csv")

    print(mean_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    main(args)