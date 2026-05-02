import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.dataset3D import BraTSDataset3D
from src.models.unet import UNet3D

dataset = BraTSDataset3D("data/split/train")

model = UNet3D()

if not os.path.exists("model.pth"):
    raise FileNotFoundError(
        "model.pth not found. Please run training first:\n"
        "  python -m src.train"
    )

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

img, mask = dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0))
    pred = torch.argmax(pred, dim=1).squeeze(0)

img  = img.numpy()   # [4, D, H, W]
mask = mask.numpy()  # [D, H, W]
pred = pred.numpy()  # [D, H, W]

modality_names = ["FLAIR", "T1", "T1ce", "T2"]

# Find slices along depth axis that contain tumor
tumor_slices = np.where((mask != 0).any(axis=(1, 2)))[0]

if len(tumor_slices) > 0:
    indices = np.linspace(0, len(tumor_slices) - 1, 6, dtype=int)
    slices = tumor_slices[indices]
    print(f"Showing tumor slices: {slices.tolist()}")
else:
    D = img.shape[1]
    slices = np.linspace(D // 4, 3 * D // 4, 6, dtype=int)
    print("No tumor found, showing center slices.")

num_rows = 6   # 4 modalities + ground truth + prediction
num_cols = len(slices)
row_labels = modality_names + ["Ground Truth", "Prediction"]

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
fig.suptitle("Brain Tumor Segmentation — All Modalities", fontsize=14)

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

        # Row label on the LEFT side of the first column using text
        if col == 0:
            ax.text(
                -0.15, 0.5, label,
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='center', ha='right',
                rotation=90
            )

        # Slice number on top row
        if row == 0:
            ax.set_title(f"Slice {d}", fontsize=9)

plt.tight_layout()
plt.subplots_adjust(left=0.1)   # make room for row labels
plt.show()