# src/models/dual_ensemble.py

import torch
import torch.nn as nn


class DualEnsemble(nn.Module):
    """
    Two pretrained dual-input U-Nets fused at the pre-output feature level.

    Branch A: t1ce + flair
    Branch B: t2   + t1ce

    Following the reference paper methodology:
    - Both backbones are frozen after loading pretrained weights.
    - Feature maps are extracted BEFORE the final Conv3d (model.out),
      i.e. from dec1 output → [B, 16, D, H, W] per branch.
    - Concatenated features [B, 32, D, H, W] are passed through a
      lightweight fusion head that is trained from scratch.
    """

    def __init__(self, model_a, model_b, out_channels=3):
        super().__init__()

        self.model_a = model_a  # UNet3D pretrained on t1ce + flair
        self.model_b = model_b  # UNet3D pretrained on t2   + t1ce

        # Freeze pretrained branches (also done in train_ensemble.py,
        # but repeated here for safety if DualEnsemble is used standalone)
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False

        # Fusion head — only part that gets trained
        # Input: 16 (branch A) + 16 (branch B) = 32 channels
        self.fuse = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    # ------------------------------------------------------------------
    # Extract dec1 output (pre-softmax feature map) from a UNet3D branch
    # ------------------------------------------------------------------
    def _get_features(self, model, x):
        """
        Re-runs the UNet3D forward pass up to (and including) dec1,
        stopping before model.out.

        Returns: [B, 16, D, H, W]
        """
        # Encoder
        e1 = model.enc1(x)
        e2 = model.enc2(model.pool(e1))
        e3 = model.enc3(model.pool(e2))
        e4 = model.enc4(model.pool(e3))

        # Bottleneck
        b = model.bottleneck(model.pool(e4))

        # Decoder
        d4 = model.dec4(torch.cat([model.up4(b),  e4], dim=1))
        d3 = model.dec3(torch.cat([model.up3(d4), e3], dim=1))
        d2 = model.dec2(torch.cat([model.up2(d3), e2], dim=1))
        d1 = model.dec1(torch.cat([model.up1(d2), e1], dim=1))

        return d1  # [B, 16, D, H, W]

    # ------------------------------------------------------------------

    def forward(self, x1, x2):
        """
        x1: [B, 2, D, H, W]  — t1ce + flair  (for branch A)
        x2: [B, 2, D, H, W]  — t2   + t1ce   (for branch B)
        """
        feat_a = self._get_features(self.model_a, x1)  # [B, 16, D, H, W]
        feat_b = self._get_features(self.model_b, x2)  # [B, 16, D, H, W]

        fused = torch.cat([feat_a, feat_b], dim=1)     # [B, 32, D, H, W]

        return self.fuse(fused)                         # [B, out_channels, D, H, W]


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    from src.models.unet import UNet3D

    model_a = UNet3D(in_channels=2, out_channels=3)
    model_b = UNet3D(in_channels=2, out_channels=3)

    ensemble = DualEnsemble(model_a, model_b, out_channels=3)

    x1 = torch.randn(1, 2, 128, 128, 128)
    x2 = torch.randn(1, 2, 128, 128, 128)

    out = ensemble(x1, x2)
    print("x1 shape :", x1.shape)   # [1, 2, 128, 128, 128]
    print("x2 shape :", x2.shape)   # [1, 2, 128, 128, 128]
    print("out shape:", out.shape)  # [1, 3, 128, 128, 128]

    # Verify only fusion head has trainable params
    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in ensemble.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")