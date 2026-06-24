from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "INSTANCE") -> None:
        super().__init__()
        if norm.upper() == "BATCH":
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = nn.InstanceNorm3d
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "INSTANCE") -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3d(in_ch=out_ch * 2, out_ch=out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, dx, 0, dy, 0, dz))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NoCTFixerUNet(nn.Module):
    """
    Fixer that ignores CT entirely: only uses initial segmentation as input.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8, norm: str = "INSTANCE"):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3d(in_channels, c, norm)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(c * 2, c * 2, norm)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(c * 4, c * 4, norm)

        self.up2 = UpBlock3d(c * 4, c * 2, norm)
        self.up1 = UpBlock3d(c * 2, c, norm)
        self.out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x_main: torch.Tensor, x_err: torch.Tensor | None = None) -> torch.Tensor:
        e1 = self.enc1(x_main)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.up2(e3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


__all__ = ["NoCTFixerUNet"]
