# # import os
# # import nibabel as nib
# # import numpy as np

# # from src.dataset.preprocessing import (
# #     normalize,
# #     crop_roi_t1,
# #     resize_3d
# # )


# # case_path = "data/split/train/BraTS-GLI-00000-000"  # ganti salah satu case

# # files = os.listdir(case_path)

# # t1_file    = [f for f in files if "t1n" in f.lower()][0]
# # t1ce_file  = [f for f in files if "t1c" in f.lower()][0]
# # t2_file    = [f for f in files if "t2w" in f.lower()][0]
# # flair_file = [f for f in files if "t2f" in f.lower()][0]
# # seg_file   = [f for f in files if "seg" in f.lower()][0]


# # # =====================
# # # LOAD
# # # =====================

# # t1 = np.transpose(
# #     nib.load(os.path.join(case_path, t1_file)).get_fdata(),
# #     (2, 0, 1)
# # )

# # t1ce = np.transpose(
# #     nib.load(os.path.join(case_path, t1ce_file)).get_fdata(),
# #     (2, 0, 1)
# # )

# # t2 = np.transpose(
# #     nib.load(os.path.join(case_path, t2_file)).get_fdata(),
# #     (2, 0, 1)
# # )

# # flair = np.transpose(
# #     nib.load(os.path.join(case_path, flair_file)).get_fdata(),
# #     (2, 0, 1)
# # )

# # seg = np.transpose(
# #     nib.load(os.path.join(case_path, seg_file)).get_fdata(),
# #     (2, 0, 1)
# # )

# # # =====================
# # # PREPROCESS
# # # =====================

# # t1, t1ce, t2, flair, seg = crop_roi_t1(
# #     t1, t1ce, t2, flair, seg
# # )

# # t1 = resize_3d(t1)
# # t1ce = resize_3d(t1ce)
# # t2 = resize_3d(t2)
# # flair = resize_3d(flair)

# # # =====================
# # # BEFORE
# # # =====================

# # print("\n===== BEFORE =====")

# # for name, img in zip(
# #     ["T1", "T1CE", "T2", "FLAIR"],
# #     [t1, t1ce, t2, flair]
# # ):
# #     brain = img[img > 0]

# #     print(f"\n{name}")
# #     print("mean =", brain.mean())
# #     print("std  =", brain.std())
# #     print("min  =", brain.min())
# #     print("max  =", brain.max())


# # # =====================
# # # NORMALIZE
# # # =====================

# # t1_n = normalize(t1)
# # t1ce_n = normalize(t1ce)
# # t2_n = normalize(t2)
# # flair_n = normalize(flair)

# # # =====================
# # # AFTER
# # # =====================

# # print("\n===== AFTER =====")

# # for name, img, ref in zip(
# #     ["T1", "T1CE", "T2", "FLAIR"],
# #     [t1_n, t1ce_n, t2_n, flair_n],
# #     [t1, t1ce, t2, flair]
# # ):
# #     brain = img[ref > 0]

# #     print(f"\n{name}")
# #     print("mean =", brain.mean())
# #     print("std  =", brain.std())
# #     print("min  =", brain.min())
# #     print("max  =", brain.max())

# # mask = t1 > 0

# # mean_all = t1.mean()
# # std_all = t1.std()

# # mean_brain = t1[mask].mean()
# # std_brain = t1[mask].std()

# # print("ALL")
# # print(mean_all, std_all)

# # print("BRAIN")
# # print(mean_brain, std_brain)

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.preprocessing import (
    crop_roi_t1,
    resize_3d,
    normalize
)

# =====================================
# CASE
# =====================================

case_path = (
    "data/split/train/"
    "BraTS-GLI-00000-000"
)

# =====================================
# LOAD
# =====================================

files = os.listdir(case_path)

t1_file = [f for f in files if "t1n" in f.lower()][0]
t1c_file = [f for f in files if "t1c" in f.lower()][0]
t2_file = [f for f in files if "t2w" in f.lower()][0]
flair_file = [f for f in files if "t2f" in f.lower()][0]
seg_file = [f for f in files if "seg" in f.lower()][0]

t1 = np.transpose(
    nib.load(
        os.path.join(case_path, t1_file)
    ).get_fdata(),
    (2,0,1)
)

t1c = np.transpose(
    nib.load(
        os.path.join(case_path, t1c_file)
    ).get_fdata(),
    (2,0,1)
)

t2 = np.transpose(
    nib.load(
        os.path.join(case_path, t2_file)
    ).get_fdata(),
    (2,0,1)
)

flair = np.transpose(
    nib.load(
        os.path.join(case_path, flair_file)
    ).get_fdata(),
    (2,0,1)
)

seg = np.transpose(
    nib.load(
        os.path.join(case_path, seg_file)
    ).get_fdata(),
    (2,0,1)
)

# =====================================
# RAW INFO
# =====================================

print("\n========== RAW ==========")
print("Shape :", t1.shape)

brain = t1[t1 > 0]

print("Brain Mean :", brain.mean())
print("Brain Std  :", brain.std())

print("All Mean   :", t1.mean())
print("All Std    :", t1.std())

# =====================================
# CROP
# =====================================

(
    t1_crop,
    t1c_crop,
    t2_crop,
    flair_crop,
    seg_crop
) = crop_roi_t1(
    t1,
    t1c,
    t2,
    flair,
    seg
)

print("\n========== ROI ==========")
print("Shape :", t1_crop.shape)

# =====================================
# RESIZE
# =====================================

t1_resize = resize_3d(t1_crop)

print("\n========== RESIZE ==========")
print("Shape :", t1_resize.shape)

# =====================================
# NORMALIZE
# =====================================

t1_norm = normalize(t1_resize)

print("\n========== NORMALIZE ==========")

print("Before")
print("Mean :", t1_resize.mean())
print("Std  :", t1_resize.std())
print("Min  :", t1_resize.min())
print("Max  :", t1_resize.max())

print()

print("After")
print("Mean :", t1_norm.mean())
print("Std  :", t1_norm.std())
print("Min  :", t1_norm.min())
print("Max  :", t1_norm.max())

# =====================================
# HISTOGRAM
# =====================================

fig, ax = plt.subplots(
    2,
    2,
    figsize=(12,8)
)

ax[0,0].hist(
    t1_resize.flatten(),
    bins=100
)

ax[0,0].set_title(
    "Before Normalize"
)

ax[0,1].hist(
    t1_norm.flatten(),
    bins=100
)

ax[0,1].set_title(
    "After Normalize"
)

# =====================================
# SAME WINDOW VISUALIZATION
# =====================================

mid = t1_resize.shape[0] // 2

ax[1,0].imshow(
    t1_resize[mid],
    cmap="gray",
    vmin=0,
    vmax=2000
)

ax[1,0].set_title(
    "Before Normalize"
)

ax[1,1].imshow(
    t1_norm[mid],
    cmap="gray",
    vmin=0,
    vmax=2000
)

ax[1,1].set_title(
    "After Normalize\n(Fixed Window)"
)

plt.tight_layout()
plt.show()

# =====================================
# PIXEL EXAMPLE
# =====================================

print("\n========== SAMPLE PIXELS ==========")

coords = [
    (64,64,64),
    (64,70,70),
    (64,80,80)
]

for d,h,w in coords:

    print(
        f"({d},{h},{w}) : "
        f"{t1_resize[d,h,w]:.3f}"
        " -> "
        f"{t1_norm[d,h,w]:.3f}"
    )


# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt

# from src.dataset.preprocessing import (
#     crop_roi_t1,
#     resize_3d,
#     normalize
# )

# # =====================================
# # LOAD CASE
# # =====================================

# case_path = (
#     "data/split/train/"
#     "BraTS-GLI-00000-000"
# )

# files = os.listdir(case_path)

# t1_file = [f for f in files if "t1n" in f.lower()][0]
# t1c_file = [f for f in files if "t1c" in f.lower()][0]
# t2_file = [f for f in files if "t2w" in f.lower()][0]
# flair_file = [f for f in files if "t2f" in f.lower()][0]
# seg_file = [f for f in files if "seg" in f.lower()][0]

# t1 = np.transpose(
#     nib.load(
#         os.path.join(case_path, t1_file)
#     ).get_fdata(),
#     (2, 0, 1)
# )

# t1c = np.transpose(
#     nib.load(
#         os.path.join(case_path, t1c_file)
#     ).get_fdata(),
#     (2, 0, 1)
# )

# t2 = np.transpose(
#     nib.load(
#         os.path.join(case_path, t2_file)
#     ).get_fdata(),
#     (2, 0, 1)
# )

# flair = np.transpose(
#     nib.load(
#         os.path.join(case_path, flair_file)
#     ).get_fdata(),
#     (2, 0, 1)
# )

# seg = np.transpose(
#     nib.load(
#         os.path.join(case_path, seg_file)
#     ).get_fdata(),
#     (2, 0, 1)
# )

# # =====================================
# # PREPROCESS
# # =====================================

# (
#     t1,
#     t1c,
#     t2,
#     flair,
#     seg
# ) = crop_roi_t1(
#     t1,
#     t1c,
#     t2,
#     flair,
#     seg
# )

# t1 = resize_3d(t1)

# t1_norm = normalize(t1)

# # =====================================
# # INFO
# # =====================================

# print("\nBEFORE")
# print("mean =", t1.mean())
# print("std  =", t1.std())
# print("min  =", t1.min())
# print("max  =", t1.max())

# print("\nAFTER")
# print("mean =", t1_norm.mean())
# print("std  =", t1_norm.std())
# print("min  =", t1_norm.min())
# print("max  =", t1_norm.max())

# # =====================================
# # VISUALIZATION
# # =====================================

# mid = t1.shape[0] // 2

# fig, ax = plt.subplots(
#     2,
#     3,
#     figsize=(15, 10)
# )

# # -------------------------
# # BEFORE (AUTO WINDOW)
# # -------------------------

# ax[0,0].imshow(
#     t1[mid],
#     cmap="gray"
# )

# ax[0,0].set_title(
#     "Before (Auto Window)"
# )

# # -------------------------
# # AFTER (AUTO WINDOW)
# # -------------------------

# ax[0,1].imshow(
#     t1_norm[mid],
#     cmap="gray"
# )

# ax[0,1].set_title(
#     "After Normalize (Auto Window)"
# )

# # -------------------------
# # AFTER (FIXED Z-SCORE WINDOW)
# # -------------------------

# im = ax[0,2].imshow(
#     t1_norm[mid],
#     cmap="gray",
#     vmin=-3,
#     vmax=3
# )

# ax[0,2].set_title(
#     "After Normalize (vmin=-3 vmax=3)"
# )

# plt.colorbar(
#     im,
#     ax=ax[0,2]
# )

# # -------------------------
# # HIST BEFORE
# # -------------------------

# ax[1,0].hist(
#     t1.flatten(),
#     bins=100
# )

# ax[1,0].set_title(
#     "Histogram Before"
# )

# # -------------------------
# # HIST AFTER
# # -------------------------

# ax[1,1].hist(
#     t1_norm.flatten(),
#     bins=100
# )

# ax[1,1].set_title(
#     "Histogram After"
# )

# # -------------------------
# # HIST AFTER ZOOM
# # -------------------------

# brain = t1_norm[t1 > 0]

# ax[1,2].hist(
#     brain,
#     bins=100
# )

# ax[1,2].set_title(
#     "Brain Voxels Only"
# )

# for a in ax.flatten():
#     a.axis("off") if a in ax[0] else None

# plt.tight_layout()
# plt.show()