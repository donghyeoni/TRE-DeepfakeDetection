"""Hand-written multi-head self-attention (MHSA) and a transformer block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


class MHSA(nn.Module):
    """Multi-head self-attention over a sequence of tokens ``(B, N, E)``."""

    def __init__(self, embed_dim=config.EMBED_DIM, num_head=config.NUM_HEADS):
        super().__init__()
        assert embed_dim % num_head == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, N, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_head, self.head_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out(out)

        return out, attn


class MHSABlock(nn.Module):
    """Pre-norm-style transformer block: MHSA + FFN with residual connections."""

    def __init__(self, embed_dim=config.EMBED_DIM, num_head=config.NUM_HEADS):
        super().__init__()
        self.mhsa = MHSA(embed_dim, num_head)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.mhsa(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
