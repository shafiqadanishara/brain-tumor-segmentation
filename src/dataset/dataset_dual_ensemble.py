# src/dataset/dataset_dual_ensemble.py

import torch
from torch.utils.data import Dataset
from src.dataset.dataset3D import BraTSDataset3D


class BraTSDualEnsembleDataset(Dataset):
    """
    Reuses your existing BraTSDataset3D.

    Returns:
        x1 = [t1ce, flair]
        x2 = [t2, t1ce]
        mask
    """

    def __init__(self, root_dir, augment=False):
        self.base = BraTSDataset3D(root_dir, augment=augment)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]

        # img channel order:
        # 0=t1, 1=t1ce, 2=t2, 3=flair

        x1 = img[[1, 3], :, :, :]   # t1ce + flair
        x2 = img[[2, 1], :, :, :]   # t2 + t1ce

        return x1.float(), x2.float(), mask.float()