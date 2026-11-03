
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x) + self.shortcut(x), 0.2)


class WindowedAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        ws = self.window_size

        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_perm = F.pad(x_perm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        _, Dp, Hp, Wp, _ = x_perm.shape

        x_win = x_perm.view(B, Dp // ws, ws, Hp // ws, ws, Wp // ws, ws, C)
        x_win = x_win.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x_win = x_win.view(-1, ws ** 3, C)

        shortcut = x_win
        x_win = self.norm(x_win)

        qkv = self.qkv(x_win).reshape(-1, ws ** 3, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x_win = (attn @ v).transpose(1, 2).reshape(-1, ws ** 3, C)
        x_win = self.proj(x_win) + shortcut

        x_win = x_win.view(B, Dp // ws, Hp // ws, Wp // ws, ws, ws, ws, C)
        x_win = x_win.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x_win = x_win.view(B, Dp, Hp, Wp, C)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_win = x_win[:, :D, :H, :W, :]

        return x_win.permute(0, 4, 1, 2, 3).contiguous()


class ModalityEncoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.stem = Conv3DBlock(in_channels, c)

        self.layer1_conv = Conv3DBlock(c, c * 2, stride=2)
        self.layer1_attn = WindowedAttention3D(c * 2, num_heads=4, window_size=4)

        self.layer2_conv = Conv3DBlock(c * 2, c * 4, stride=2)
        self.layer2_attn = WindowedAttention3D(c * 4, num_heads=8, window_size=4)

        self.layer3_conv = Conv3DBlock(c * 4, c * 8, stride=2)
        self.layer3_attn = WindowedAttention3D(c * 8, num_heads=8, window_size=4)

        self.out_channels = c * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1_attn(self.layer1_conv(x))
        x = self.layer2_attn(self.layer2_conv(x))
        x = self.layer3_attn(self.layer3_conv(x))
        return x