import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.dataset.dataset3D import BraTSDataset3D
from src.models.unet import UNet3D

MODALITY_CHANNELS = {
    "flair": [0],
    "t1":    [1],
    "t1ce":  [2],
    "t2":    [3],
}
MODALITY_NAMES = ["FLAIR", "T1", "T1ce", "T2"]


def main(args):
    channels  = MODALITY_CHANNELS[args.modality]
    save_path = f"model_{args.modality}.pth"

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"{save_path} not found. Train first:\n"
            f"  python -m src.train --modality {args.modality}"
        )

    dataset = BraTSDataset3D("data/split/val")
    model   = UNet3D(in_channels=len(channels), out_channels=4)
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    model.eval()

    img, mask = dataset[0]

    with torch.no_grad():
        img_input = img[channels, :, :, :].unsqueeze(0)  # [1, C, D, H, W]
        pred = model(img_input)
        pred = torch.argmax(pred, dim=1).squeeze(0)

    img  = img.numpy()    # [4, D, H, W] — all modalities for display
    mask = mask.numpy()   # [D, H, W]
    pred = pred.numpy()   # [D, H, W]

    # Find tumor slices along depth axis
    tumor_slices = np.where((mask != 0).any(axis=(1, 2)))[0]
    if len(tumor_slices) > 0:
        indices = np.linspace(0, len(tumor_slices) - 1, 6, dtype=int)
        slices  = tumor_slices[indices]
        print(f"Showing tumor slices: {slices.tolist()}")
    else:
        D      = img.shape[1]
        slices = np.linspace(D // 4, 3 * D // 4, 6, dtype=int)
        print("No tumor found, showing center slices.")

    # Rows: 4 modalities + ground truth + prediction
    num_cols  = len(slices)
    row_labels = MODALITY_NAMES + ["Ground Truth", "Prediction"]

    fig, axes = plt.subplots(6, num_cols, figsize=(num_cols * 3, 6 * 3))
    fig.suptitle(f"Brain Tumor Segmentation — {args.modality.upper()} model",
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(row_labels):
        for col, d in enumerate(slices):
            ax = axes[row, col]
            ax.axis('off')

            if row < 4:
                ax.imshow(img[row, d, :, :], cmap='gray')
            elif row == 4:
                ax.imshow(mask[d, :, :], cmap='jet', vmin=0, vmax=3)
            else:
                ax.imshow(pred[d, :, :], cmap='jet', vmin=0, vmax=3)

            if col == 0:
                ax.text(-0.15, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        va='center', ha='right', rotation=90)

            if row == 0:
                ax.set_title(f"Slice {d}", fontsize=9)

    legend_elements = [
        Patch(facecolor='blue',   label='Background (0)'),
        Patch(facecolor='red',    label='Necrosis (1)'),
        Patch(facecolor='cyan',   label='Edema (2)'),
        Patch(facecolor='yellow', label='Enhancing Tumor (3)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True,
                        choices=["flair", "t1", "t1ce", "t2"])
    args = parser.parse_args()
    main(args)