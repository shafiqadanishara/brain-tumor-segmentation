import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.preprocessing import (
    normalize, crop_roi_t1, resize_3d
)
from src.dataset.augmentedData import augment


class BraTSDataset3D(Dataset):
    """
    BraTS 3D Dataset supporting both BraTS 2018-2021 and BraTS 2023+ label conventions.

    Label conventions:
        BraTS 2018-2021: NCR=1, ED=2, ET=4
        BraTS 2023+:     NCR=1, ED=2, ET=3

    Output mask channels:
        Channel 0 - WT (Whole Tumor):   all labels > 0
        Channel 1 - TC (Tumor Core):    NCR + ET
        Channel 2 - ET (Enhancing):     ET only
    """

    BRATS_2018 = "2018"
    BRATS_2023 = "2023"

    def __init__(self, root_dir, brats_version=None, augment=False):
        self.root_dir = root_dir
        self.augment = augment

        self.cases = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not self.cases:
            raise RuntimeError(f"No cases found in {root_dir}")

        if brats_version is None:
            self.brats_version = self._detect_version()
        else:
            assert brats_version in (self.BRATS_2018, self.BRATS_2023)
            self.brats_version = brats_version

        print(
            f"[BraTSDataset3D] {len(self.cases)} cases | "
            f"BraTS version: {self.brats_version} | "
            f"Augment: {self.augment}"
        )

    def _detect_version(self):
        case = self.cases[0]
        case_path = os.path.join(self.root_dir, case)
        files = os.listdir(case_path)

        seg_file = [f for f in files if "seg" in f][0]
        seg = nib.load(os.path.join(case_path, seg_file)).get_fdata()

        unique = np.unique(seg).astype(int).tolist()
        print(f"[BraTSDataset3D] Auto-detected labels: {unique}")

        if 4 in unique:
            return self.BRATS_2018
        elif 3 in unique:
            return self.BRATS_2023
        else:
            raise RuntimeError(f"Unexpected labels {unique}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.root_dir, case)
        files = os.listdir(case_path)

        # --- Locate files ---
        t1_file    = self._find_file(files, "t1n", case)
        t1ce_file  = self._find_file(files, "t1c", case)
        t2_file    = self._find_file(files, "t2w", case)
        flair_file = self._find_file(files, "t2f", case)
        seg_file   = self._find_file(files, "seg", case)

        # --- Load NIfTI ---
        t1_nii = nib.load(os.path.join(case_path, t1_file))
        affine = t1_nii.affine
        original_shape = np.array(t1_nii.shape)

        t1    = t1_nii.get_fdata()
        t1ce  = nib.load(os.path.join(case_path, t1ce_file)).get_fdata()
        t2    = nib.load(os.path.join(case_path, t2_file)).get_fdata()
        flair = nib.load(os.path.join(case_path, flair_file)).get_fdata()
        seg   = nib.load(os.path.join(case_path, seg_file)).get_fdata()

        # --- Save bbox BEFORE crop ---
        coords = np.argwhere(t1 > 0)
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1
        bbox = np.array([x0, x1, y0, y1, z0, z1])

        # --- Preprocessing ---
        # ROI crop
        t1, t1ce, t2, flair, seg = crop_roi_t1(t1, t1ce, t2, flair, seg)

        # Resize
        t1    = resize_3d(t1)
        t1ce  = resize_3d(t1ce)
        t2    = resize_3d(t2)
        flair = resize_3d(flair)
        seg   = resize_3d(seg, order=0)

        # Labels -> WT TC ET
        seg = self.convert_to_regions(seg)

        # Normalize
        t1    = normalize(t1)
        t1ce  = normalize(t1ce)
        t2    = normalize(t2)
        flair = normalize(flair)

        # Stack (C,H,W,D) -> (C,D,H,W)
        image = np.stack([t1, t1ce, t2, flair], axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        seg   = np.transpose(seg,   (0, 3, 1, 2))

        # Augmentation
        if self.augment:
            image, seg = augment(image, seg)

        # Tensor
        image = torch.tensor(image, dtype=torch.float32)
        seg   = torch.tensor(seg, dtype=torch.float32)

        # Metadata for postprocess
        meta = {
            "affine": affine.astype(np.float32),
            "original_shape": original_shape.astype(np.int32),
            "bbox": bbox.astype(np.int32),
            "case": case
        }

        return image, seg, meta

    def convert_to_regions(self, mask):
        wt = (mask > 0).astype(np.float32)

        if self.brats_version == self.BRATS_2018:
            tc = np.isin(mask, [1, 4]).astype(np.float32)
            et = (mask == 4).astype(np.float32)
        else:
            tc = np.isin(mask, [1, 3]).astype(np.float32)
            et = (mask == 3).astype(np.float32)

        return np.stack([wt, tc, et], axis=0)

    def _find_file(self, files, keyword, case):
        matches = [f for f in files if keyword in f]
        if not matches:
            raise FileNotFoundError(
                f"No file with keyword '{keyword}' in case '{case}'. "
                f"Available files: {files}"
            )
        return matches[0]

    def get_class_weights(self, num_samples=None):
        samples = range(len(self)) if num_samples is None \
            else range(min(num_samples, len(self)))

        totals = np.zeros(3)
        voxels = 0

        print("Computing class weights...")

        for i in samples:
            _, mask, _ = self[i]
            for c in range(3):
                totals[c] += mask[c].sum().item()
            voxels += mask[0].numel()

        freq = totals / voxels
        weights = 1.0 / (freq + 1e-6)
        weights = weights / weights.sum()

        print(f"WT freq={freq[0]:.4f} weight={weights[0]:.4f}")
        print(f"TC freq={freq[1]:.4f} weight={weights[1]:.4f}")
        print(f"ET freq={freq[2]:.4f} weight={weights[2]:.4f}")

        return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/split/train"

    dataset = BraTSDataset3D(root, augment=False)

    print(f"\nTotal cases : {len(dataset)}")

    image, mask, meta = dataset[0]
    print("\n--- Single sample ---")
    print("Image shape :", image.shape)
    print("Mask shape  :", mask.shape)
    print("Original shape:", meta["original_shape"])
    print("BBox:", meta["bbox"])

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    img_b, mask_b, meta_b = next(iter(loader))

    print("\n--- DataLoader batch ---")
    print("Batch image :", img_b.shape)
    print("Batch mask  :", mask_b.shape)

    print("\nAll checks passed ✓")