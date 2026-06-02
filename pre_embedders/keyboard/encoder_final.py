"""KeyboardEncoderFinal — final keyboard embedder (additive; does not touch the
existing KeystrokeEncoder).

Architecture: BiGRU over (B, W, 3) keystroke windows -> last-step concat ->
projection to 64-D with LayerNorm. A small contrastive projection head and an
auxiliary behavioral-feature head are used ONLY during training; the exported
embedding is the 64-D backbone output (SimCLR convention: representation before
the projection head).

Output contract: forward(x) -> (B, 64) float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyboardEncoderFinal(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = True,
        l2_normalize: bool = False,
        n_aux: int = 5,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize
        dirs = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size * dirs, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        # training-only heads
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.aux_head = nn.Linear(embedding_dim, n_aux)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                       # (B, W, H*dirs)
        h = out[:, -1, :]                          # last step
        z = self.backbone(h)                       # (B, 64)
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def forward_train(self, x: torch.Tensor):
        z = self.embed(x)
        return z, self.proj_head(z), self.aux_head(z)
