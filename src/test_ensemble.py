import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.unet import UNet3D
from src.models.dual_ensemble import DualEnsemble
from src.dataset.dataset3D import BraTSDataset3D
from src.utils.metrics import compute_metrics
from src.dataset.postprocess import restore_to_original, save_nifti


def save_visual(path, pred):
    z = pred.shape[-1] // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    titles = ["WT", "TC", "ET"]
    cmaps = ["Reds", "Greens", "Blues"]

    for i in range(3):
        ax[i].imshow(pred[i, :, :, z], cmap=cmaps[i])
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BraTSDataset3D(
        f"data/split/{args.split}",
        augment=False
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Branch A: T2 + T1ce
    model_a = UNet3D(in_channels=2, out_channels=3).to(device)
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))

    # Branch B: T1ce + Flair
    model_b = UNet3D(in_channels=2, out_channels=3).to(device)
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    model = DualEnsemble(model_a, model_b, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_root = Path(f"outputs/ensemble/{args.split}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    with torch.no_grad():
        for idx, (img, mask, meta) in enumerate(tqdm(loader)):
            img = img.to(device)
            mask = mask.to(device)

            x1 = img[:, [2, 1]]  # T2 + T1ce
            x2 = img[:, [1, 3]]  # T1ce + Flair

            logits = model(x1, x2)
            pred = (torch.sigmoid(logits) > 0.5).float()

            metrics = compute_metrics(logits, mask)
            metrics["case"] = idx
            rows.append(metrics)

            pred_np = pred[0].cpu().numpy()

            # metadata
            affine = meta["affine"][0].numpy()
            original_shape = meta["original_shape"][0].numpy().astype(int)
            bbox = meta["bbox"][0].numpy().astype(int)
            case = meta["case"][0]

            # restore to original space
            restored = restore_to_original(
                pred_np,
                original_shape,
                bbox
            )

            case_dir = out_root / case
            case_dir.mkdir(parents=True, exist_ok=True)

            np.save(case_dir / "pred_full.npy", pred_np)
            np.save(case_dir / "pred_full_original.npy", restored)

            save_nifti(
                restored.transpose(1, 2, 3, 0),
                affine,
                case_dir / "pred_full_original.nii.gz"
            )

            save_nifti(restored[0], affine, case_dir / "pred_wt_original.nii.gz")
            save_nifti(restored[1], affine, case_dir / "pred_tc_original.nii.gz")
            save_nifti(restored[2], affine, case_dir / "pred_et_original.nii.gz")

            save_visual(case_dir / "comparison.png", restored)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)
    df.mean(numeric_only=True).to_csv(out_root / "metrics_mean.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args)