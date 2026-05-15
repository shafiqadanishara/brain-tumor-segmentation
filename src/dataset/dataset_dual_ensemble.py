# src/dataset/dataset_dual_ensemble.py

import torch
from torch.utils.data import Dataset
from src.dataset.dataset3D import BraTSDataset3D


class BraTSDualEnsembleDataset(Dataset):
    """
    Reuses BraTSDataset3D.

    Returns:
        x1   = [t1ce, flair]  — for branch A (t1ce_flair model)
        x2   = [t2,   t1ce]   — for branch B (t2_t1ce model)
        mask

    Elastic deformation is disabled (elastic=False) because backbones
    are frozen during ensemble training — elastic augmentation would
    produce out-of-distribution feature maps from the frozen encoders.
    """

    def __init__(self, root_dir, augment=False):
        self.base = BraTSDataset3D(
            root_dir,
            augment=augment,
            elastic=False   # disabled for ensemble training
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask, _ = self.base[idx]

        # img channel order from BraTSDataset3D:
        # 0=t1, 1=t1ce, 2=t2, 3=flair

        x1 = img[[1, 3], :, :, :]   # t1ce + flair
        x2 = img[[2, 1], :, :, :]   # t2   + t1ce

        return x1.float(), x2.float(), mask.float()