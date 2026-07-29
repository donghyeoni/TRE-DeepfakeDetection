"""DNSAMNet: the full TRE detector.

Temporal aggregation and spatial focusing are combined multiplicatively and fed
to a small MLP classifier that outputs a real/fake probability.

    TSC(x) = Classifier( TemporalAggregation(x) * SpatialFocusing(x) )
"""

import torch
import torch.nn as nn

from .. import config
from .spatial_attention import SpatialFocusing
from .temporal_attention import TemporalAggregation


class Classifier(nn.Module):
    """Flatten the fused feature map and predict a single probability."""

    def __init__(self, in_features=config.FEATURE_DIM * config.FEATURE_H * config.FEATURE_W):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )

    def forward(self, x):  # x: (B, D, H, W)
        x = x.flatten(1)   # (B, D*H*W)
        x = self.fc(x)     # (B, 1)
        return torch.sigmoid(x)


class TSC(nn.Module):
    """Temporal x Spatial -> Classifier -- the full DNSAMNet detector."""

    def __init__(self, embed_dim=config.EMBED_DIM, num_head=config.NUM_HEADS,
                 feature_dim=config.FEATURE_DIM):
        super().__init__()
        self.temporal = TemporalAggregation(embed_dim, num_head, feature_dim)
        self.spatial = SpatialFocusing(embed_dim, num_head, feature_dim)
        self.classifier = Classifier(
            in_features=feature_dim * config.FEATURE_H * config.FEATURE_W
        )

    def forward(self, x):  # x: (B, T, D, H, W)
        temporal = self.temporal(x)   # (B, D, H, W)
        spatial = self.spatial(x)     # (B, 1, H, W)
        result = temporal * spatial   # (B, D, H, W)
        return self.classifier(result)


# Backwards-compatible alias for the full model.
DNSAMNet = TSC
