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


class ErrorMapCrossAttnCT(nn.Module):
    """
    Predict tree/sdf error maps using mask as Q and CT features as K/V.
    Two lightweight cross-attn heads produce two error maps.
    """

    def __init__(self, mask_channels: int = 1, ct_channels: int = 1, base_channels: int = 8, norm: str = "INSTANCE"):
        super().__init__()
        c = base_channels
        self.mask_enc = ConvBlock3d(mask_channels, c, norm)
        self.ct_enc = ConvBlock3d(ct_channels, c, norm)
        self.attn_tree = CrossAttention3D(q_ch=c, kv_ch=c, spatial_reduction=4)
        self.attn_sdf = CrossAttention3D(q_ch=c, kv_ch=c, spatial_reduction=4)
        self.out_tree = nn.Conv3d(c, 1, kernel_size=1)
        self.out_sdf = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, mask: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        q = self.mask_enc(mask)
        kv = self.ct_enc(ct)
        tree_feat = self.attn_tree(q, kv)
        sdf_feat = self.attn_sdf(q, kv)
        tree = self.out_tree(tree_feat)
        sdf = self.out_sdf(sdf_feat)
        return torch.cat([tree, sdf], dim=1)


__all__ = ["ErrorMapCrossAttnCT"]
