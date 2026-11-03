
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from configs.config import AblationConfig


class StructureSignalDecoupling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.structure_branch = nn.Sequential(
            nn.Conv3d(channels, channels, 5, padding=2),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 5, padding=2),
            nn.InstanceNorm3d(channels),
        )

        self.signal_branch = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
        )

        self.fusion = nn.Conv3d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structure = self.structure_branch(x)
        signal = self.signal_branch(x)
        combined = torch.cat([structure, signal], dim=1)
        output = F.leaky_relu(self.fusion(combined) + x, 0.2)
        return output, structure, signal


class DifferenceGuidedCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels * 2, channels, 1)
        self.v_proj = nn.Conv3d(channels * 2, channels, 1)

        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)

    def forward(
        self,
        diff_feat: torch.Tensor,
        left_feat: torch.Tensor,
        right_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C, D, H, W = diff_feat.shape
        L = D * H * W

        q = self.q_proj(diff_feat).view(B, self.num_heads, self.head_dim, L)
        q = q.transpose(-2, -1)

        kv = torch.cat([left_feat, right_feat], dim=1)
        k = self.k_proj(kv).view(B, self.num_heads, self.head_dim, L)
        v = self.v_proj(kv).view(B, self.num_heads, self.head_dim, L).transpose(-2, -1)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        out = self.norm(self.out_proj(out))
        return out


class ContralateralInteractionModule(nn.Module):
    def __init__(self, channels: int, config: AblationConfig):
        super().__init__()
        self.config = config

        if config.use_structure_signal_decoupling:
            self.decouple_left = StructureSignalDecoupling(channels)
            self.decouple_right = StructureSignalDecoupling(channels)

        if config.use_diff_attention:
            self.diff_attn = DifferenceGuidedCrossAttention(channels, num_heads=8)

        self.norm = nn.InstanceNorm3d(channels)

    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor
    ):
        if not self.config.use_contralateral:
            return None, None, None, None, (None, None), (None, None)

        if self.config.use_structure_signal_decoupling:
            f_left, left_str, left_sig = self.decouple_left(f_left)
            f_right, right_str, right_sig = self.decouple_right(f_right)
        else:
            f_left, left_str, left_sig = f_left,None,None
            f_right, right_str, right_sig = f_right,None,None

        diff_feat = self.norm(f_left - f_right)
        diff_abs = torch.abs(f_left - f_right)
        common_feat = self.norm(f_left + f_right)

        if self.config.use_diff_attention:
            diff_aware = self.diff_attn(diff_feat, f_left, f_right)
        else:
            diff_aware = diff_feat


        return diff_feat, diff_aware, common_feat, diff_abs,(left_str, right_str), (left_sig, right_sig)