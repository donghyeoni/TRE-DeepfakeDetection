"""Temporal Aggregation module (canonical DNSAMNet cell-38 variant).

Attention is applied along the temporal axis (T=20) independently at every
spatial location; the last token is projected back to the latent-channel
dimension, yielding a ``(B, D, H, W)`` temporal summary.
"""

import torch
import torch.nn as nn

from .. import config
from .attention import MHSABlock


class TemporalAggregation(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM, num_head=config.NUM_HEADS,
                 feature_dim=config.FEATURE_DIM):
        super().__init__()
        self.layers = nn.ModuleList([MHSABlock(embed_dim, num_head) for _ in range(2)])
        # Residual feed-forward blocks interleaved with the attention layers.
        self.ffn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(0.1),
                )
                for _ in range(2)
            ]
        )
        self.embed_dim = embed_dim
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):  # x: (B, T=20, D=8, H, W)
        B, T, D, H, W = x.shape
        out = torch.zeros((B, D, H, W)).to(x.device)
        for i in range(H):
            for j in range(W):
                seq = x[:, :, :, i, j]        # (B, T, D)
                seq = self.proj(seq)          # (B, T, embed_dim)
                for layer, ffn in zip(self.layers, self.ffn_blocks):
                    seq = layer(seq)                                      # (B, T, embed_dim)
                    seq_ffn = ffn(seq.transpose(1, 2)).transpose(1, 2)    # (B, T, embed_dim)
                    seq = seq + seq_ffn                                   # residual connection
                last_token = seq[:, -1, :]           # (B, embed_dim)
                last_token = self.proj_out(last_token)  # (B, D)
                out[:, :, i, j] = last_token
        return out
