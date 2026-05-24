# =========================
# src/dataset/dataset_upenn.py
# UPenn-GBM cross-dataset testing
# Standardized to (C,D,H,W) — same convention as BraTSDataset3D
#
# Actual folder structure:
#   data/UPENN_GBM/
#       images_segm/
#           UPENN-GBM-00001_11_segm.nii.gz
#           UPENN-GBM-00002_11_segm.nii.gz
#           ...
#       images_structural/
#           UPENN-GBM-00001_11/
#               UPENN-GBM-00001_11_T1.nii.gz
#               UPENN-GBM-00001_11_T1GD.nii.gz
#               UPENN-GBM-00001_11_T2.nii.gz
#               UPENN-GBM-00001_11_FLAIR.nii.gz
#           UPENN-GBM-00002_11/
#               ...
#
# Labels confirmed [0, 1, 2, 4] — BraTS 2018 style:
#   1 = NCR  2 = ED  4 = ET
# =========================

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from src.dataset.preprocessing import (
    normalize,
    crop_roi_t1,
    resize_3d
)


class UPennDataset3D(Dataset):

    def __init__(self, root_dir):
        self.root_dir    = root_dir
        self.struct_dir  = os.path.join(root_dir, "images_structural")
        self.segm_dir    = os.path.join(root_dir, "images_segm")

        # Cases = subfolders inside images_structural/
        self.cases = sorted([
            d for d in os.listdir(self.struct_dir)
            if os.path.isdir(os.path.join(self.struct_dir, d))
        ])

        # Only keep cases that also have a matching seg file
        self.cases = [
            c for c in self.cases
            if os.path.exists(
                os.path.join(self.segm_dir, f"{c}_segm.nii.gz")
            )
        ]

        if not self.cases:
            raise RuntimeError(
                f"No valid cases found in {root_dir}. "
                "Check that images_structural/ and images_segm/ exist."
            )

        print(f"[UPennDataset3D] {len(self.cases)} cases | Labels: BraTS-2018 style [0,1,2,4]")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case       = self.cases[idx]
        case_dir   = os.path.join(self.struct_dir, case)
        seg_path   = os.path.join(self.segm_dir, f"{case}_segm.nii.gz")

        # ── Load ──────────────────────────────────────────────────────
        # nibabel loads as (H,W,D) → transpose immediately to (D,H,W)

        t1_nii = nib.load(os.path.join(case_dir, f"{case}_T1.nii.gz"))
        affine  = t1_nii.affine
        original_shape = np.array(t1_nii.shape)[[2, 0, 1]]   # (D,H,W)

        def load_vol(path):
            return np.transpose(nib.load(path).get_fdata(), (2, 0, 1))

        t1    = load_vol(os.path.join(case_dir, f"{case}_T1.nii.gz"))
        t1ce  = load_vol(os.path.join(case_dir, f"{case}_T1GD.nii.gz"))
        t2    = load_vol(os.path.join(case_dir, f"{case}_T2.nii.gz"))
        flair = load_vol(os.path.join(case_dir, f"{case}_FLAIR.nii.gz"))
        seg   = load_vol(seg_path)

        # ── Bounding box (D,H,W) ──────────────────────────────────────
        coords = np.argwhere(t1 > 0)
        d0, h0, w0 = coords.min(axis=0)
        d1, h1, w1 = coords.max(axis=0) + 1
        bbox = np.array([d0, d1, h0, h1, w0, w1])

        # ── Preprocess — identical pipeline to BraTSDataset3D ─────────
        t1, t1ce, t2, flair, seg = crop_roi_t1(t1, t1ce, t2, flair, seg)

        t1    = resize_3d(t1)
        t1ce  = resize_3d(t1ce)
        t2    = resize_3d(t2)
        flair = resize_3d(flair)
        seg   = resize_3d(seg, order=0)

        seg = self.convert_to_regions(seg)   # (3,D,H,W)

        t1    = normalize(t1)
        t1ce  = normalize(t1ce)
        t2    = normalize(t2)
        flair = normalize(flair)

        # ── Stack (C,D,H,W) — channel order matches BraTSDataset3D ────
        # index 0: T1 | index 1: T1ce | index 2: T2 | index 3: FLAIR
        image = np.stack([t1, t1ce, t2, flair], axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        seg   = torch.tensor(seg,   dtype=torch.float32)

        meta = {
            "affine":         affine.astype(np.float32),
            "original_shape": original_shape.astype(np.int32),
            "bbox":           bbox.astype(np.int32),
            "case":           case,
        }

        return image, seg, meta

    def convert_to_regions(self, mask):
        """
        UPenn labels confirmed [0, 1, 2, 4] — BraTS 2018 style:
            1 = NCR (necrotic core)
            2 = ED  (edema)
            4 = ET  (enhancing tumor)
        Regions:
            WT = 1 + 2 + 4  (all tumor)
            TC = 1 + 4       (tumor core)
            ET = 4           (enhancing)
        """
        wt = (mask > 0).astype(np.float32)
        tc = np.isin(mask, [1, 4]).astype(np.float32)
        et = (mask == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)   # (3,D,H,W)


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    root = sys.argv[1] if len(sys.argv) > 1 else "data/UPENN_GBM"
    ds   = UPennDataset3D(root)

    img, mask, meta = ds[0]
    print("Image :", img.shape)    # (4,D,H,W)
    print("Mask  :", mask.shape)   # (3,D,H,W)
    print("Orig  :", meta["original_shape"])
    print("BBox  :", meta["bbox"])
    print("Case  :", meta["case"])

    dl = DataLoader(ds, batch_size=1)
    xb, yb, mb = next(iter(dl))
    print("Batch image:", xb.shape)
    print("Batch mask :", yb.shape)