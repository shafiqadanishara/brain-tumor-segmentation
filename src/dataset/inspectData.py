import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

from src.dataset.preprocessing import (
    normalize,
    crop_roi_t1,
    resize_3d
)


def load_case(case_path):

    files = os.listdir(case_path)

    t1_file = [f for f in files if "t1n" in f.lower()][0]
    t1c_file = [f for f in files if "t1c" in f.lower()][0]
    t2_file = [f for f in files if "t2w" in f.lower()][0]
    flair_file = [
        f for f in files
        if ("flair" in f.lower() or "t2f" in f.lower())
    ][0]

    seg_file = [f for f in files if "seg" in f.lower()][0]

    t1 = nib.load(
        os.path.join(case_path, t1_file)
    ).get_fdata()

    t1c = nib.load(
        os.path.join(case_path, t1c_file)
    ).get_fdata()

    t2 = nib.load(
        os.path.join(case_path, t2_file)
    ).get_fdata()

    flair = nib.load(
        os.path.join(case_path, flair_file)
    ).get_fdata()

    seg = nib.load(
        os.path.join(case_path, seg_file)
    ).get_fdata()

    return t1, t1c, t2, flair, seg


def visualize(case_path):

    # =====================================
    # LOAD
    # =====================================

    t1_raw, t1c_raw, t2_raw, flair_raw, seg_raw = load_case(
        case_path
    )

    # =====================================
    # TRANSPOSE (H,W,D -> D,H,W)
    # SAME AS TRAINING PIPELINE
    # =====================================

    t1_raw = np.transpose(t1_raw, (2, 0, 1))
    t1c_raw = np.transpose(t1c_raw, (2, 0, 1))
    t2_raw = np.transpose(t2_raw, (2, 0, 1))
    flair_raw = np.transpose(flair_raw, (2, 0, 1))
    seg_raw = np.transpose(seg_raw, (2, 0, 1))

    titles = [
        "T1",
        "T1ce",
        "T2",
        "FLAIR"
    ]

    # =====================================
    # RAW
    # =====================================

    d_raw = t1_raw.shape[0] // 2

    # =====================================
    # ROI CROP
    # =====================================

    (
        t1_crop,
        t1c_crop,
        t2_crop,
        flair_crop,
        seg_crop
    ) = crop_roi_t1(
        t1_raw,
        t1c_raw,
        t2_raw,
        flair_raw,
        seg_raw
    )

    d_crop = t1_crop.shape[0] // 2

    # =====================================
    # RESIZE
    # =====================================

    t1_resize = resize_3d(t1_crop)
    t1c_resize = resize_3d(t1c_crop)
    t2_resize = resize_3d(t2_crop)
    flair_resize = resize_3d(flair_crop)

    seg_resize = resize_3d(
        seg_crop,
        order=0
    )

    d_resize = t1_resize.shape[0] // 2

    # =====================================
    # NORMALIZE
    # =====================================

    t1_norm = normalize(t1_resize)
    t1c_norm = normalize(t1c_resize)
    t2_norm = normalize(t2_resize)
    flair_norm = normalize(flair_resize)

    # =====================================
    # INFO
    # =====================================

    print("\n===== SHAPE =====")
    print("RAW      :", t1_raw.shape)
    print("ROI      :", t1_crop.shape)
    print("RESIZED  :", t1_resize.shape)

    print("\n===== NORMALIZATION =====")

    for name, before, after in [
        ("T1", t1_resize, t1_norm),
        ("T1ce", t1c_resize, t1c_norm),
        ("T2", t2_resize, t2_norm),
        ("FLAIR", flair_resize, flair_norm)
    ]:

        print(f"\n{name}")

        print(
            f"Before : "
            f"min={before.min():.4f} "
            f"max={before.max():.4f} "
            f"mean={before.mean():.4f}"
        )

        print(
            f"After  : "
            f"min={after.min():.4f} "
            f"max={after.max():.4f} "
            f"mean={after.mean():.4f}"
        )

    # =====================================
    # PLOT
    # =====================================

    fig, axes = plt.subplots(
        5,
        4,
        figsize=(16, 15)
    )

    # =====================================
    # ROW 1 RAW
    # =====================================

    raw_imgs = [
        t1_raw,
        t1c_raw,
        t2_raw,
        flair_raw
    ]

    for i, img in enumerate(raw_imgs):

        axes[0, i].imshow(
            img[d_raw],
            cmap="gray"
        )

        axes[0, i].set_title(
            f"RAW {titles[i]}"
        )

        axes[0, i].axis("off")

    # =====================================
    # ROW 2 ROI
    # =====================================

    crop_imgs = [
        t1_crop,
        t1c_crop,
        t2_crop,
        flair_crop
    ]

    for i, img in enumerate(crop_imgs):

        axes[1, i].imshow(
            img[d_crop],
            cmap="gray"
        )

        axes[1, i].set_title(
            f"ROI Crop {titles[i]}"
        )

        axes[1, i].axis("off")

    # =====================================
    # ROW 3 RESIZE
    # =====================================

    resize_imgs = [
        t1_resize,
        t1c_resize,
        t2_resize,
        flair_resize
    ]

    for i, img in enumerate(resize_imgs):

        axes[2, i].imshow(
            img[d_resize],
            cmap="gray"
        )

        axes[2, i].set_title(
            f"Resize {titles[i]}"
        )

        axes[2, i].axis("off")

    # =====================================
    # ROW 4 NORMALIZE
    # =====================================

    norm_imgs = [
        t1_norm,
        t1c_norm,
        t2_norm,
        flair_norm
    ]

    for i, img in enumerate(norm_imgs):

        axes[3, i].imshow(
            img[d_resize],
            cmap="gray"
        )

        axes[3, i].set_title(
            f"Normalize {titles[i]}"
        )

        axes[3, i].axis("off")

    # =====================================
    # ROW 5 MASK
    # =====================================

    axes[4, 0].imshow(
        seg_raw[d_raw],
        cmap="jet"
    )

    axes[4, 0].set_title(
        "RAW Mask"
    )

    axes[4, 0].axis("off")

    axes[4, 1].imshow(
        seg_crop[d_crop],
        cmap="jet"
    )

    axes[4, 1].set_title(
        "ROI Mask"
    )

    axes[4, 1].axis("off")

    axes[4, 2].imshow(
        seg_resize[d_resize],
        cmap="jet"
    )

    axes[4, 2].set_title(
        "Resize Mask"
    )

    axes[4, 2].axis("off")

    axes[4, 3].imshow(
        t1_norm[d_resize],
        cmap="gray"
    )

    axes[4, 3].imshow(
        seg_resize[d_resize],
        cmap="jet",
        alpha=0.4
    )

    axes[4, 3].set_title(
        "Final Overlay"
    )

    axes[4, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    case = (
        # "data/split/train/"
        # "BraTS-GLI-00000-000"
        "data_upenn/UPENN-GBM-00020_11"
    )

    visualize(case)