"""ResNet-18 baseline detector (early prototype from ``LoadDataset.ipynb``).

This is the simpler ``nn.MultiheadAttention``-based prototype kept for reference
and comparison; the DNSAMNet model in :mod:`src.models.dnsamnet` is the main
model.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultiHeadSelfAttention(nn.Module):
    """Thin wrapper around ``nn.MultiheadAttention`` with a residual LayerNorm."""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: [N, T, D]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class TemporalAggregation(nn.Module):
    """Baseline temporal aggregation over the T axis."""

    def __init__(self, T, D, num_heads=8, num_layers=2):
        super().__init__()
        self.attn_blocks = nn.Sequential(
            *[MultiHeadSelfAttention(D, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):  # x: [B, T, H, W]
        B, T, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, T, 1)  # [B*H*W, T, D=1]
        out = self.attn_blocks(x)                           # [B*H*W, T, D]
        last = out[:, -1, :]                                # [B*H*W, D]
        return last.view(B, H, W, 1)                        # [B, H, W, 1]


class SpatialFocusing(nn.Module):
    """Baseline spatial focusing that produces a per-pixel attention weight."""

    def __init__(self, H, W, D=1, num_heads=8, num_layers=2):
        super().__init__()
        self.D = D
        self.attn_blocks = nn.Sequential(
            *[MultiHeadSelfAttention(D, num_heads) for _ in range(num_layers)]
        )
        self.conv1x1 = nn.Conv1d(D, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [B, T, H, W]
        B, T, H, W = x.shape
        pooled = (x.mean(dim=1) + x.max(dim=1)[0]) / 2       # [B, H, W]
        pooled = pooled.view(B, H * W, 1).expand(-1, -1, self.D)  # [B, H*W, D]
        x = self.attn_blocks(pooled)                          # [B, H*W, D]
        x = x.permute(0, 2, 1)                                # [B, D, H*W]
        weights = self.softmax(self.conv1x1(x))               # [B, 1, H*W]
        return weights.view(B, 1, H, W).permute(0, 2, 3, 1)   # [B, H, W, 1]


class ResNet18_4ch(nn.Module):
    """ResNet-18 adapted for 4-channel latent-difference inputs."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):  # x: [B, H, W, C=4]
        x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        return self.model(x)


class AttentionClassifier(nn.Module):
    """Baseline: temporal x spatial attention feeding a ResNet-18 classifier."""

    def __init__(self, temporal_module, spatial_module, classifier):
        super().__init__()
        self.temporal = temporal_module
        self.spatial = spatial_module
        self.classifier = classifier

    def forward(self, x):  # x: [B, T=20, C=4, H=32, W=32]
        B = x.size(0)
        T, C, H, W = x.size(1), x.size(2), x.size(3), x.size(4)
        # Temporal attention
        xt = x.permute(0, 2, 3, 4, 1).reshape(B * H * W, T, C)  # [B*H*W, T, C]
        T_out = self.temporal.attn_blocks(xt)                   # [B*H*W, T, C]
        T_last = T_out[:, -1, :].view(B, H, W, C)               # [B, H, W, C]

        # Spatial attention
        F_prime = self.spatial(x)                          # [B, H, W, 1] (broadcast)

        T_final = T_last * F_prime                         # [B, H, W, C]
        return self.classifier(T_final)                    # [B, num_classes]
