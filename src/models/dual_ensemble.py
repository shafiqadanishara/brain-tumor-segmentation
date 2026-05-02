# src/models/dual_ensemble.py

import torch
import torch.nn as nn


class DualEnsemble(nn.Module):
    """
    Two pretrained dual-input U-Nets fused together.

    Branch A: t1ce + flair
    Branch B: t2 + t1ce
    """

    def __init__(self, model_a, model_b, out_channels=3):
        super().__init__()

        self.model_a = model_a
        self.model_b = model_b

        # Freeze pretrained branches
        for p in self.model_a.parameters():
            p.requires_grad = False

        for p in self.model_b.parameters():
            p.requires_grad = False

        # Fusion head
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x1, x2):
        y1 = self.model_a(x1)
        y2 = self.model_b(x2)

        x = torch.cat([y1, y2], dim=1)
        x = self.fuse(x)

        return x