"""MouseEncoderFinal — final mouse embedder (additive; does not touch
MouseEncoderP2).

Architecture (hybrid):
  * sequence branch : (B, 8, T) per-event channels [speed, accel, jerk, dx, dy,
                      is_idle, is_click, is_scroll] -> 1D CNN -> adaptive mean
                      pool -> seq embedding
  * stats branch    : (B, n_stats) handcrafted behavioral stats -> MLP
  * fusion          : concat -> Linear -> LayerNorm -> 64-D

A contrastive projection head and an auxiliary stats head are used ONLY in
training. Exported embedding is the 64-D fusion output.

Output contract: forward(seq, stats) -> (B, 64) float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(cout), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MouseEncoderFinal(nn.Module):
    def __init__(
        self,
        seq_channels: int = 8,
        n_stats: int = 8,
        seq_hidden: int = 64,
        stats_hidden: int = 64,
        embedding_dim: int = 64,
        l2_normalize: bool = False,
        n_aux: int = 6,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize
        self.seq_net = nn.Sequential(
            _ConvBlock(seq_channels, 32, 5),
            _ConvBlock(32, seq_hidden, 3),
            _ConvBlock(seq_hidden, seq_hidden, 3),
        )
        self.stats_net = nn.Sequential(
            nn.Linear(n_stats, stats_hidden), nn.LayerNorm(stats_hidden), nn.ReLU(),
            nn.Linear(stats_hidden, stats_hidden), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(seq_hidden + stats_hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.aux_head = nn.Linear(embedding_dim, n_aux)

    def embed(self, seq: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        s = self.seq_net(seq)                  # (B, seq_hidden, T)
        s = s.mean(dim=2)                      # adaptive mean pool -> (B, seq_hidden)
        st = self.stats_net(stats)             # (B, stats_hidden)
        z = self.fusion(torch.cat([s, st], dim=1))
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, seq: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.embed(seq, stats)

    def forward_train(self, seq: torch.Tensor, stats: torch.Tensor):
        z = self.embed(seq, stats)
        return z, self.proj_head(z), self.aux_head(z)
