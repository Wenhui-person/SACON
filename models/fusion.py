
import torch
import torch.nn as nn


class CrossModalGatedFusion(nn.Module):
    def __init__(self, channels: int, use_gating: bool = True):
        super().__init__()
        self.use_gating = use_gating

        if use_gating:
            self.gate_net = nn.Sequential(
                nn.Conv3d(channels * 2, channels // 2, 1),
                nn.InstanceNorm3d(channels // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(channels // 2, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, f_t1: torch.Tensor, f_t2: torch.Tensor, return_gate: bool = False):
        if not self.use_gating:
            fused = (f_t1 + f_t2) / 2.0
            gate = torch.full_like(f_t1[:, :1], 0.5)
            return (fused, gate) if return_gate else fused

        alpha = self.gate_net(torch.cat([f_t1, f_t2], dim=1))
        fused = alpha * f_t1 + (1.0 - alpha) * f_t2
        return (fused, alpha) if return_gate else fused
