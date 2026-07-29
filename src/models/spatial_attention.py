"""Spatial Focusing module (canonical DNSAMNet cell-38 variant).

Average- and max-pooling over the temporal axis are concatenated, passed
through a small convolutional stack and self-attention over the flattened
spatial grid, then softmax-normalised into a ``(B, 1, H, W)`` spatial attention
map.

Bug fixes relative to the notebook:
  * ``MHSABlock(embed_dim, num_head)`` referenced an undefined ``num_head``;
    it now correctly uses the ``num_heads`` constructor argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from .attention import MHSABlock


class SpatialFocusing(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM, num_heads=config.NUM_HEADS,
                 feature_dim=config.FEATURE_DIM):
        super().__init__()
        self.avg_pool = lambda x: torch.mean(x, dim=1, keepdim=True)     # (B, 1, D, H, W)
        self.max_pool = lambda x: torch.max(x, dim=1, keepdim=True)[0]   # (B, 1, D, H, W)

        pooled_ch = 2 * feature_dim  # avg + max concatenated along the channel axis
        self.conv1x1 = nn.Conv2d(pooled_ch, 64, kernel_size=1)
        self.d_conv3x3 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3x3 = nn.Conv2d(32, pooled_ch, kernel_size=3, padding=1)
        # NOTE: fixed ``num_head`` -> ``num_heads`` (undefined-name bug in the notebook).
        self.layers = nn.ModuleList([MHSABlock(embed_dim, num_heads) for _ in range(2)])
        self.proj = nn.Linear(pooled_ch, embed_dim)
        self.proj_out = nn.Linear(embed_dim, 1)

        self.pooled_ch = pooled_ch

    def forward(self, x):  # x: (B, T, D, H, W)
        B, T, D, H, W = x.shape
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)  # (B, 2, D, H, W)
        pooled = pooled.reshape(B, self.pooled_ch, H, W)                 # (B, 2D, H, W)

        pooled = self.conv1x1(pooled)
        pooled = self.d_conv3x3(pooled)
        pooled = self.conv3x3(pooled)

        pooled = pooled.permute(0, 2, 3, 1)              # (B, H, W, 2D)
        pooled = pooled.reshape(B, H * W, self.pooled_ch)

        pooled = self.proj(pooled)                       # (B, H*W, embed_dim)
        for layer in self.layers:
            pooled = layer(pooled)                       # (B, H*W, embed_dim)

        pooled = self.proj_out(pooled)                   # (B, H*W, 1)
        pooled = pooled.permute(0, 2, 1)                 # (B, 1, H*W)
        pooled = pooled.reshape(B, 1, H, W)              # (B, 1, H, W)

        attention = F.softmax(pooled.view(B, 1, -1), dim=-1).view(B, 1, H, W)
        return attention
