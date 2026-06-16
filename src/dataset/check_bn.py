import os
import glob
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis

from src.dataset.preprocessing import (
    crop_roi_t1,
    resize_3d,
    normalize
)

from src.models.unet import UNet3D


# ==================================================
# CONFIG
# ==================================================

MODEL_PATH = "experiments/dual/output/checkpoints/resume_fold0_t1ce_flair_latest.pth"

DATA_ROOT = (
    "data/raw/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

CASE_DIR = sorted(
    glob.glob(
        os.path.join(
            DATA_ROOT,
            "BraTS-GLI-*"
        )
    )
)[0]


# ==================================================
# LOAD MRI
# ==================================================

def load_vol(path):
    return np.transpose(
        nib.load(path).get_fdata(),
        (2, 0, 1)
    )


files = os.listdir(CASE_DIR)

t1_file    = next(f for f in files if "-t1n" in f.lower())
t1ce_file  = next(f for f in files if "-t1c" in f.lower())
t2_file    = next(f for f in files if "-t2w" in f.lower())
flair_file = next(f for f in files if "-t2f" in f.lower())
seg_file   = next(f for f in files if "-seg" in f.lower())

t1    = load_vol(os.path.join(CASE_DIR, t1_file))
t1ce  = load_vol(os.path.join(CASE_DIR, t1ce_file))
t2    = load_vol(os.path.join(CASE_DIR, t2_file))
flair = load_vol(os.path.join(CASE_DIR, flair_file))
seg   = load_vol(os.path.join(CASE_DIR, seg_file))


# ==================================================
# SAME PREPROCESSING AS TRAINING
# ==================================================

t1, t1ce, t2, flair, seg = crop_roi_t1(
    t1,
    t1ce,
    t2,
    flair,
    seg
)

t1ce = resize_3d(t1ce)
flair = resize_3d(flair)

t1ce = normalize(t1ce)
flair = normalize(flair)

x = torch.tensor(
    np.stack(
        [t1ce, flair],
        axis=0
    )[np.newaxis],
    dtype=torch.float32
)

print("Input shape:", x.shape)


# ==================================================
# MODEL
# ==================================================

model = UNet3D(
    in_channels=2,
    out_channels=3
)

loaded = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# ----------------------------------
# BEST MODEL
# ----------------------------------
if "model_state_dict" not in loaded:

    model.load_state_dict(
        loaded
    )

    print(
        "Loaded BEST MODEL format"
    )

# ----------------------------------
# CHECKPOINT
# ----------------------------------
else:

    model.load_state_dict(
        loaded["model_state_dict"]
    )

    print(
        "Loaded CHECKPOINT format"
    )

    print(
        f"Epoch: {loaded['epoch']}"
    )

model.eval()

print(
    "Model loaded successfully"
)


# ==================================================
# BN RUNNING STATS
# ==================================================

print("\n==============================")
print("BATCHNORM RUNNING STATS")
print("==============================")

for block_name in [
    "enc1",
    "enc2",
    "enc3",
    "enc4",
    "bottleneck"
]:

    bn = getattr(
        model,
        block_name
    ).net[1]

    print(f"\n{block_name}")

    print(
        "running_mean =",
        bn.running_mean.mean().item()
    )

    print(
        "running_var  =",
        bn.running_var.mean().item()
    )


# ==================================================
# HOOKS
# ==================================================

acts = {}


def save(name):

    def hook(module, inp, out):
        acts[name] = (
            out.detach()
               .cpu()
               .numpy()
        )

    return hook


blocks = {
    "enc1": model.enc1,
    "enc2": model.enc2,
    "enc3": model.enc3,
    "enc4": model.enc4,
    "bottleneck": model.bottleneck,
}

watch = {
    "conv1": 0,
    "bn1": 1,
    "relu1": 2,

    "conv2": 3,
    "bn2": 4,
    "relu2": 5,
}

for block_name, block in blocks.items():

    for layer_name, idx in watch.items():

        block.net[idx].register_forward_hook(
            save(
                f"{block_name}_{layer_name}"
            )
        )


# ==================================================
# FORWARD
# ==================================================

with torch.no_grad():

    _ = model(x)


# ==================================================
# STATS
# ==================================================

def get_stats(arr):

    arr = arr.flatten()

    return (
        arr.mean(),
        arr.std(),
        skew(arr),
        kurtosis(arr)
    )


print("\n==============================")
print("FEATURE MAP STATISTICS")
print("==============================")

print(
    f"{'Layer':25s}"
    f"{'Mean':>12s}"
    f"{'Std':>12s}"
    f"{'Skew':>12s}"
    f"{'Kurt':>12s}"
)

for name in sorted(acts.keys()):

    mean, std, sk, kurt = get_stats(
        acts[name]
    )

    print(
        f"{name:25s}"
        f"{mean:12.4f}"
        f"{std:12.4f}"
        f"{sk:12.4f}"
        f"{kurt:12.4f}"
    )

from scipy.stats import norm

# ==================================================
# HISTOGRAM GRID
# ==================================================

plot_names = []

for block in [
    "enc1",
    "enc2",
    "enc3",
    "enc4",
    "bottleneck"
]:

    plot_names.extend([

        f"{block}_conv1",
        f"{block}_bn1",
        f"{block}_relu1",

        f"{block}_conv2",
        f"{block}_bn2",
        f"{block}_relu2",
    ])


rows = 5
cols = 6

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(24, 16)
)

axes = axes.flatten()

for ax, name in zip(
    axes,
    plot_names
):

    data = acts[name].flatten()

    mu = data.mean()
    sigma = data.std()

    sk = skew(data)
    kt = kurtosis(data)

    hist, bins = np.histogram(
        data,
        bins=100,
        density=True
    )

    centers = (
        bins[:-1] +
        bins[1:]
    ) / 2

    ax.plot(
        centers,
        hist,
        linewidth=1.2,
        label="Actual"
    )

    gauss = norm.pdf(
        centers,
        mu,
        sigma
    )

    ax.plot(
        centers,
        gauss,
        linestyle=":",
        label="Gaussian"
    )

    ax.axvline(
        mu,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"{name}\n"
        f"skew={sk:.2f}, kurt={kt:.2f}",
        fontsize=8
    )

    ax.tick_params(
        labelsize=7
    )

# hapus subplot kosong kalau ada
for i in range(
    len(plot_names),
    len(axes)
):
    fig.delaxes(
        axes[i]
    )

plt.tight_layout()

plt.savefig(
    "bn_histogram_grid_relu.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()