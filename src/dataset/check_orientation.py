# import nibabel as nib

# path = "data/split/test/BraTS-GLI-00120-000/BraTS-GLI-00120-000-t2f.nii.gz"

# nii = nib.load(path)

# print("Original orientation:")
# print(nib.aff2axcodes(nii.affine))

# nii_can = nib.as_closest_canonical(nii)

# print("Canonical orientation:")
# print(nib.aff2axcodes(nii_can.affine))

# print("Original shape :", nii.shape)
# print("Canonical shape:", nii_can.shape)

# import nibabel as nib
# import matplotlib.pyplot as plt

# path = r"data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00120-000/BraTS-GLI-00120-000-t2f.nii.gz"

# nii = nib.load(path)

# img_lps = nii.get_fdata()

# img_ras = nib.as_closest_canonical(
#     nii
# ).get_fdata()

# z = img_lps.shape[2] // 2

# fig, ax = plt.subplots(
#     1,
#     2,
#     figsize=(10,5)
# )

# ax[0].imshow(
#     img_lps[:,:,z],
#     cmap="gray"
# )
# ax[0].set_title("Original LPS")

# ax[1].imshow(
#     img_ras[:,:,z],
#     cmap="gray"
# )
# ax[1].set_title("Canonical RAS")

# plt.show()

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

path = "data/split/test/BraTS-GLI-00120-000/BraTS-GLI-00120-000-t2f.nii.gz"

nii = nib.load(path)

img = np.transpose(
    nii.get_fdata(),
    (2,0,1)
)

mid = img.shape[0] // 2

plt.imshow(
    img[mid],
    cmap="gray"
)
plt.show()