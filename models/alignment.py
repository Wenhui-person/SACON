
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from matplotlib import pyplot as plt

from configs.config import AblationConfig


class DeformableRegistration3D(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.enc1 = self._conv_block(in_channels * 2, c)
        self.enc2 = self._conv_block(c, c * 2, stride=2)
        self.enc3 = self._conv_block(c * 2, c * 4, stride=2)

        self.dec2 = nn.ConvTranspose3d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2_conv = self._conv_block(c * 2, c * 2)
        self.dec1 = nn.ConvTranspose3d(c * 2, c, 4, stride=2, padding=1)
        self.dec1_conv = self._conv_block(c, c)

        self.flow = nn.Conv3d(c, 3, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    @staticmethod
    def _conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fixed, moving], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)


        if self.dec2(e3).shape != e2.shape:
            dec2_e3 = F.interpolate(self.dec2(e3), size=e2.shape[2:], mode='trilinear', align_corners=True)

        d2 = F.leaky_relu(dec2_e3, 0.2) + e2
        d2 = self.dec2_conv(d2)

        if self.dec1(d2).shape != e1.shape:
            dec1_d2 = F.interpolate(self.dec1(d2), size=e1.shape[2:], mode='trilinear', align_corners=True)

        d1 = F.leaky_relu(dec1_d2, 0.2) + e1
        d1 = self.dec1_conv(d1)

        return self.flow(d1)

    @staticmethod
    def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        device = x.device

        gz, gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, D, device=device),
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([gx, gy, gz], dim=0).unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = (base_grid + flow).permute(0, 2, 3, 4, 1)

        warped = F.grid_sample(x, grid, align_corners=True,
                             mode='bilinear', padding_mode='border')

        return warped



class SymmetryAlignmentModule(nn.Module):

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config

        if config.registration_type == 'deformable':
            self.reg_shared = DeformableRegistration3D(in_channels=2, base_channels=16)
            self.max_disp = 0.04

    def forward(
        self,
        xl_t1: torch.Tensor, xl_t2: torch.Tensor,
        xr_t1: torch.Tensor, xr_t2: torch.Tensor
    ):
        if self.config.registration_type == 'none':
            return xl_t1, xl_t2, xr_t1, xr_t2, None

        xr_t1_flip = torch.flip(xr_t1, dims=[-1])
        xr_t2_flip = torch.flip(xr_t2, dims=[-1])

        if self.config.registration_type == 'flip':
            return xl_t1,xl_t2,xr_t1_flip, xr_t2_flip,None

        fixed = torch.cat([xl_t1, xl_t2], dim=1)
        moving = torch.cat([xr_t1_flip, xr_t2_flip], dim=1)

        flow = self.reg_shared(fixed, moving)
        flow = torch.tanh(flow) * self.max_disp

        xr_t1_aligned = DeformableRegistration3D.warp(xr_t1_flip, flow)
        xr_t2_aligned = DeformableRegistration3D.warp(xr_t2_flip, flow)

        return xl_t1, xl_t2, xr_t1_aligned, xr_t2_aligned, flow