# =========================
# src/dataset/dataset3D.py
# Standardized to (C,D,H,W)
# =========================
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.preprocessing import (
    normalize,
    crop_roi_t1,
    resize_3d
)
from src.dataset.augmentedData import augment


class BraTSDataset3D(Dataset):
    BRATS_2018 = "2018"
    BRATS_2023 = "2023"

    def __init__(self, root_dir, brats_version=None, augment=False, elastic=True):
        self.root_dir = root_dir
        self.augment  = augment
        self.elastic  = elastic   # passed through to augment()

        self.cases = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not self.cases:
            raise RuntimeError(f"No cases found in {root_dir}")

        if brats_version is None:
            self.brats_version = self._detect_version()
        else:
            self.brats_version = brats_version

        print(
            f"[BraTSDataset3D] {len(self.cases)} cases | "
            f"BraTS version: {self.brats_version} | "
            f"Augment: {self.augment} | "
            f"Elastic: {self.elastic if self.augment else 'N/A'}"
        )

    def _detect_version(self):
        case = self.cases[0]
        case_path = os.path.join(self.root_dir, case)
        files = os.listdir(case_path)

        seg_file = [f for f in files if "seg" in f][0]
        seg = nib.load(os.path.join(case_path, seg_file)).get_fdata()

        unique = np.unique(seg).astype(int).tolist()

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

        t1_file    = self._find_file(files, "t1n")
        t1ce_file  = self._find_file(files, "t1c")
        t2_file    = self._find_file(files, "t2w")
        flair_file = self._find_file(files, "t2f")
        seg_file   = self._find_file(files, "seg")

        # -----------------
        # Load
        # nib loads as (H,W,D)
        # convert immediately -> (D,H,W)
        # -----------------
        t1_nii = nib.load(os.path.join(case_path, t1_file))
        affine = t1_nii.affine

        original_shape = np.array(t1_nii.shape)[[2, 0, 1]]

        t1 = np.transpose(t1_nii.get_fdata(), (2, 0, 1))
        t1ce = np.transpose(
            nib.load(os.path.join(case_path, t1ce_file)).get_fdata(),
            (2, 0, 1)
        )
        t2 = np.transpose(
            nib.load(os.path.join(case_path, t2_file)).get_fdata(),
            (2, 0, 1)
        )
        flair = np.transpose(
            nib.load(os.path.join(case_path, flair_file)).get_fdata(),
            (2, 0, 1)
        )
        seg = np.transpose(
            nib.load(os.path.join(case_path, seg_file)).get_fdata(),
            (2, 0, 1)
        )

        # -----------------
        # bbox in (D,H,W)
        # -----------------
        coords = np.argwhere(t1 > 0)
        d0, h0, w0 = coords.min(axis=0)
        d1, h1, w1 = coords.max(axis=0) + 1
        bbox = np.array([d0, d1, h0, h1, w0, w1])

        # -----------------
        # preprocess
        # -----------------
        t1, t1ce, t2, flair, seg = crop_roi_t1(
            t1, t1ce, t2, flair, seg
        )

        t1    = resize_3d(t1)
        t1ce  = resize_3d(t1ce)
        t2    = resize_3d(t2)
        flair = resize_3d(flair)
        seg   = resize_3d(seg, order=0)

        seg = self.convert_to_regions(seg)

        t1    = normalize(t1)
        t1ce  = normalize(t1ce)
        t2    = normalize(t2)
        flair = normalize(flair)

        # image -> (C,D,H,W)
        image = np.stack([t1, t1ce, t2, flair], axis=0)

        # seg already (3,D,H,W)
        if self.augment:
            image, seg = augment(image, seg, elastic=self.elastic)

        image = torch.tensor(image, dtype=torch.float32)
        seg   = torch.tensor(seg,   dtype=torch.float32)

        meta = {
            "affine":          affine.astype(np.float32),
            "original_shape":  original_shape.astype(np.int32),
            "bbox":            bbox.astype(np.int32),
            "case":            case
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

    def _find_file(self, files, keyword):
        matches = [f for f in files if keyword in f]
        if not matches:
            raise FileNotFoundError(
                f"No file with keyword '{keyword}' in {files}"
            )
        return matches[0]


if __name__ == "__main__":
    root = "data/split/train"
    ds = BraTSDataset3D(root, augment=False)

    img, mask, meta = ds[0]

    print("Image :", img.shape)   # (4,D,H,W)
    print("Mask  :", mask.shape)  # (3,D,H,W)
    print("Orig  :", meta["original_shape"])
    print("BBox  :", meta["bbox"])

    dl = DataLoader(ds, batch_size=1)
    xb, yb, mb = next(iter(dl))

    print("Batch image:", xb.shape)   # (N,C,D,H,W)
    print("Batch mask :", yb.shape)