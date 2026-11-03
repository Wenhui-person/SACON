
import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, in_channels: int, num_stages: int = 5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.binary_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.stage_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_stages-1)
        )

    def forward(self, features: torch.Tensor):
        pooled = self.global_pool(features).flatten(1)
        return self.binary_head(pooled), self.stage_head(pooled)

class BinaryHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(hidden_channels, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        feat = self.proj(x)
        logits = self.classifier(feat)
        return logits

class StageHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, num_stages: int = 4, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(hidden_channels, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_stages - 1)
        )


    def forward(self, x: torch.Tensor):
        feat = self.proj(x)
        logits = self.classifier(feat)
        return logits
