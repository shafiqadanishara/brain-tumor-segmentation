import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):  # in_channels=1 for single modality, out channels=3 for et tc wt
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up4 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(256, 128)   # 128 (up) + 128 (skip) = 256

        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)    # 64 + 64 = 128

        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)     # 32 + 32 = 64

        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)      # 16 + 16 = 32

        self.out = nn.Conv3d(16, out_channels, 1)

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(e4))

        # --- Decoder ---
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =========================
# QUICK TEST
# =========================
if __name__ == "__main__":
    model = UNet3D(in_channels=1)         # single modality
    x = torch.randn(1, 1, 128, 128, 128) # [batch, 1 channel, D, H, W]
    y = model(x)
    print("Input shape: ", x.shape)       # [1, 1, 128, 128, 128]
    print("Output shape:", y.shape)       # [1, 3, 128, 128, 128]