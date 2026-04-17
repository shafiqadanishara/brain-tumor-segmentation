import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from src.dataset.preprocessing import normalize, crop_roi_t1, crop_depth, resize_3d


def load_case(case_path):
    files = os.listdir(case_path)

    t1_file = [f for f in files if "t1n" in f.lower()][0]
    t1c_file = [f for f in files if "t1c" in f.lower()][0]
    t2_file = [f for f in files if "t2w" in f.lower()][0]
    flair_file = [f for f in files if ("flair" in f.lower() or "t2f" in f.lower())][0]
    seg_file = [f for f in files if "seg" in f.lower()][0]

    t1 = nib.load(os.path.join(case_path, t1_file)).get_fdata()
    t1c = nib.load(os.path.join(case_path, t1c_file)).get_fdata()
    t2 = nib.load(os.path.join(case_path, t2_file)).get_fdata()
    flair = nib.load(os.path.join(case_path, flair_file)).get_fdata()
    seg = nib.load(os.path.join(case_path, seg_file)).get_fdata()

    return t1, t1c, t2, flair, seg


def visualize(case_path):
    t1_raw, t1c_raw, t2_raw, flair_raw, seg_raw = load_case(case_path)

    titles = ["T1", "T1ce", "T2", "FLAIR"]

    # ========= RAW =========
    d_raw = t1_raw.shape[2] // 2

    # ========= ROI + DEPTH =========
    t1, t1c, t2, flair, seg = crop_roi_t1(t1_raw, t1c_raw, t2_raw, flair_raw, seg_raw)
    t1, t1c, t2, flair, seg = crop_depth(t1, t1c, t2, flair, seg)

    d_crop = t1.shape[2] // 2

    # ========= RESIZE =========
    t1_r = resize_3d(t1, order=1)
    t1c_r = resize_3d(t1c, order=1)
    t2_r = resize_3d(t2, order=1)
    flair_r = resize_3d(flair, order=1)
    seg_r = resize_3d(seg, order=0)  # IMPORTANT

    d_resize = t1_r.shape[2] // 2

    # ========= NORMALIZE =========
    t1_n = normalize(t1_r)
    t1c_n = normalize(t1c_r)
    t2_n = normalize(t2_r)
    flair_n = normalize(flair_r)

    # ========= PLOT =========
    plt.figure(figsize=(16, 12))

    # ===== ROW 1: RAW =====
    for i, img in enumerate([t1_raw, t1c_raw, t2_raw, flair_raw]):
        plt.subplot(4, 4, i+1)
        plt.title(f"RAW {titles[i]}")
        plt.imshow(img[:, :, d_raw], cmap='gray')
        plt.axis('off')

    # ===== ROW 2: ROI + DEPTH =====
    for i, img in enumerate([t1, t1c, t2, flair]):
        plt.subplot(4, 4, i+5)
        plt.title(f"ROI Crop {titles[i]}")
        plt.imshow(img[:, :, d_crop], cmap='gray')
        plt.axis('off')

    # ===== ROW 3: RESIZED + NORMALIZED =====
    for i, img in enumerate([t1_n, t1c_n, t2_n, flair_n]):
        plt.subplot(4, 4, i+9)
        plt.title(f"Resized {titles[i]}")
        plt.imshow(img[:, :, d_resize], cmap='gray')
        plt.axis('off')

    # ===== ROW 4 =====
    # RAW mask
    plt.subplot(4, 4, 13)
    plt.title("RAW Mask")
    plt.imshow(seg_raw[:, :, d_raw], cmap='jet')
    plt.axis('off')

    # ROI mask
    plt.subplot(4, 4, 14)
    plt.title("ROI Mask")
    plt.imshow(seg[:, :, d_crop], cmap='jet')
    plt.axis('off')

    # Resized mask
    plt.subplot(4, 4, 15)
    plt.title("Resized Mask")
    plt.imshow(seg_r[:, :, d_resize], cmap='jet')
    plt.axis('off')

    # Overlay final
    plt.subplot(4, 4, 16)
    plt.title("Final Overlay")
    plt.imshow(t1_n[:, :, d_resize], cmap='gray')
    plt.imshow(seg_r[:, :, d_resize], cmap='jet', alpha=0.4)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case = "data/split/train/BraTS-GLI-00000-000"
    visualize(case)