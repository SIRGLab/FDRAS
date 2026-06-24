from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import ViT


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
        # Pad if needed
        if x.shape[-3:] != skip.shape[-3:]:
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, dx, 0, dy, 0, dz))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CrossAttention3D(nn.Module):
    def __init__(
        self,
        q_ch: int,
        kv_ch: int,
        attn_dim: int | None = None,
        spatial_reduction: int = 4,
    ) -> None:
        super().__init__()
        d = attn_dim or max(8, q_ch // 4)
        self.sr = max(1, int(spatial_reduction))
        self.q = nn.Conv3d(q_ch, d, kernel_size=1)
        self.k = nn.Conv3d(kv_ch, d, kernel_size=1)
        self.v = nn.Conv3d(kv_ch, d, kernel_size=1)
        self.proj = nn.Conv3d(d, q_ch, kernel_size=1)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        b, _, dz, dy, dx = x_q.shape
        q = self.q(x_q)
        k = self.k(x_kv)
        v = self.v(x_kv)

        if self.sr > 1:
            q = nn.functional.avg_pool3d(q, kernel_size=self.sr, stride=self.sr, ceil_mode=True)
            k = nn.functional.avg_pool3d(k, kernel_size=self.sr, stride=self.sr, ceil_mode=True)
            v = nn.functional.avg_pool3d(v, kernel_size=self.sr, stride=self.sr, ceil_mode=True)

        b, c, d, h, w = q.shape
        q = q.view(b, c, d * h * w)
        k = k.view(b, c, d * h * w)
        v = v.view(b, c, d * h * w)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q, k) / (c ** 0.5), dim=-1)
        out = torch.einsum("bcn,bnm->bcm", v, attn).view(b, c, d, h, w)
        if self.sr > 1:
            out = nn.functional.interpolate(out, size=(dz, dy, dx), mode="trilinear", align_corners=False)
        out = self.proj(out)
        return x_q + out


class ErrorMapUNet(nn.Module):
    """Small U-Net to predict tree/sdf error maps from CT+mask."""

    def __init__(self, in_channels: int = 2, base_channels: int = 8, norm: str = "INSTANCE") -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3d(in_channels, c, norm)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(c * 2, c * 2, norm)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(c * 4, c * 4, norm)

        self.up2 = UpBlock3d(c * 4, c * 2, norm)
        self.up1 = UpBlock3d(c * 2, c, norm)
        self.out = nn.Conv3d(c, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.up2(e3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


class CrossModalFixerUNet(nn.Module):
    """Small U-Net with cross-attention using error map features."""

    def __init__(self, in_channels: int = 2, err_channels: int = 2, base_channels: int = 8, norm: str = "INSTANCE"):
        super().__init__()
        c = base_channels
        # Main encoder (CT+mask)
        self.enc1 = ConvBlock3d(in_channels, c, norm)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(c * 2, c * 2, norm)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(c * 4, c * 4, norm)

        # Error encoder
        self.e_enc1 = ConvBlock3d(err_channels, c, norm)
        self.e_down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.e_enc2 = ConvBlock3d(c * 2, c * 2, norm)
        self.e_down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.e_enc3 = ConvBlock3d(c * 4, c * 4, norm)

        # Decoder + cross-attn at each scale
        self.up2 = UpBlock3d(c * 4, c * 2, norm)
        self.attn2 = CrossAttention3D(q_ch=c * 2, kv_ch=c * 2)
        self.up1 = UpBlock3d(c * 2, c, norm)
        self.attn1 = CrossAttention3D(q_ch=c, kv_ch=c)

        self.out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x_main: torch.Tensor, x_err: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x_main)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        ee1 = self.e_enc1(x_err)
        ee2 = self.e_enc2(self.e_down1(ee1))
        ee3 = self.e_enc3(self.e_down2(ee2))

        d2 = self.up2(e3, e2)
        d2 = self.attn2(d2, ee2)
        d1 = self.up1(d2, e1)
        d1 = self.attn1(d1, ee1)
        return self.out(d1)


class ViTDiscriminator(nn.Module):
    """ViT-Base discriminator for CT+mask input."""

    def __init__(
        self,
        in_channels: int = 2,
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            num_classes=1,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=None,
        )
        self.vit.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.vit(x)
        return out


__all__ = ["ErrorMapUNet", "CrossModalFixerUNet", "ViTDiscriminator"]
