import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates, gaussian_filter, affine_transform

from src.dataset.preprocessing import crop_roi_t1, resize_3d, normalize


# =========================
# LOAD RAW MRI
# =========================
def load_raw_case(case_path):
    files = os.listdir(case_path)

    def find(name):
        return [f for f in files if name in f][0]

    t1 = np.transpose(nib.load(os.path.join(case_path, find("t1n"))).get_fdata(), (2,0,1))
    t1ce = np.transpose(nib.load(os.path.join(case_path, find("t1c"))).get_fdata(), (2,0,1))
    t2 = np.transpose(nib.load(os.path.join(case_path, find("t2w"))).get_fdata(), (2,0,1))
    flair = np.transpose(nib.load(os.path.join(case_path, find("t2f"))).get_fdata(), (2,0,1))

    return t1, t1ce, t2, flair


# =========================
# AUGMENT FUNCTIONS
# =========================
def aug_flip(img):
    return np.flip(img, axis=1)

def aug_rotate(img):
    angle = np.deg2rad(15)
    mat = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    out = np.zeros_like(img)
    center = np.array(img.shape[1:]) / 2
    offset = center - mat @ center

    for c in range(img.shape[0]):
        out[c] = affine_transform(img[c], mat, offset=offset, order=1)

    return out

def aug_elastic(img):
    np.random.seed(42)
    C, D, H, W = img.shape

    dz = gaussian_filter(np.random.randn(D,H,W), 6)*30
    dy = gaussian_filter(np.random.randn(D,H,W), 6)*30
    dx = gaussian_filter(np.random.randn(D,H,W), 6)*30

    z,y,x = np.meshgrid(np.arange(D),np.arange(H),np.arange(W), indexing='ij')
    coords = [z+dz, y+dy, x+dx]

    out = np.zeros_like(img)
    for c in range(C):
        out[c] = map_coordinates(img[c], coords, order=1)

    return out

def aug_gamma(img):
    x = img.copy()
    mn, mx = x.min(), x.max()
    x = (x - mn) / (mx - mn + 1e-8)
    x = x ** 1.5
    return x * (mx - mn) + mn


# =========================
# PREPROCESS VISUALIZATION
# =========================
def visualize_preprocessing(case_path):

    names = ["T1", "T1ce", "T2", "FLAIR"]

    t1, t1ce, t2, flair = load_raw_case(case_path)

    # CROP
    t1_c, t1ce_c, t2_c, flair_c, _ = crop_roi_t1(
        t1, t1ce, t2, flair, np.zeros_like(t1)
    )

    # RESIZE
    t1_r = resize_3d(t1_c)
    t1ce_r = resize_3d(t1ce_c)
    t2_r = resize_3d(t2_c)
    flair_r = resize_3d(flair_c)

    # NORMALIZE
    t1_n = normalize(t1_r)
    t1ce_n = normalize(t1ce_r)
    t2_n = normalize(t2_r)
    flair_n = normalize(flair_r)

    steps = [
        ("RAW", [t1, t1ce, t2, flair]),
        ("CROP", [t1_c, t1ce_c, t2_c, flair_c]),
        ("RESIZE", [t1_r, t1ce_r, t2_r, flair_r]),
        ("NORMALIZE", [t1_n, t1ce_n, t2_n, flair_n]),
    ]

    fig, axes = plt.subplots(len(steps), 4, figsize=(14, 10))

    for row, (name, imgs) in enumerate(steps):
        for col in range(4):

            d = imgs[col].shape[0] // 2

            axes[row, col].imshow(imgs[col][d], cmap='gray')
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(names[col])

            if col == 0:
                axes[row, col].set_ylabel(name)

    plt.suptitle("Preprocessing: RAW → CROP → RESIZE → NORMALIZE")
    plt.tight_layout()
    plt.show()

    return np.stack([t1_n, t1ce_n, t2_n, flair_n], axis=0)


# =========================
# AUGMENT VISUALIZATION (WINDOW PER METHOD)
# =========================
def visualize_augmentations(img_norm):

    names = ["T1", "T1ce", "T2", "FLAIR"]
    d = img_norm.shape[1] // 2

    aug_dict = {
        "Flip": aug_flip(img_norm.copy()),
        "Rotate": aug_rotate(img_norm.copy()),
        "Elastic": aug_elastic(img_norm.copy()),
        "Gamma": aug_gamma(img_norm.copy()),
    }

    for name, aug_img in aug_dict.items():

        fig, axes = plt.subplots(2, 4, figsize=(12,6))

        for i in range(4):
            axes[0, i].imshow(img_norm[i, d], cmap='gray')
            axes[0, i].set_title(names[i])
            axes[0, i].axis('off')

            axes[1, i].imshow(aug_img[i, d], cmap='gray')
            axes[1, i].axis('off')

        axes[0,0].set_ylabel("Normalized")
        axes[1,0].set_ylabel(name)

        plt.suptitle(f"{name} Augmentation (Same Slice)")
        plt.tight_layout()
        plt.show()


# =========================
# MAIN
# =========================
def run_visualization(case_path):

    # 1. Preprocessing
    img_norm = visualize_preprocessing(case_path)

    # 2. Augmentations
    visualize_augmentations(img_norm)


# =========================
# RUN
# =========================
run_visualization("data/split/train/BraTS-GLI-00000-000")