"""
TFN Layer — 4-way outer product with 1-extension + elementwise tanh.

Reference: Zadeh et al. (2017) "Tensor Fusion Network for Multimodal
Sentiment Analysis" — extended here to 4 modalities.
"""

from typing import List
import torch
import torch.nn as nn


class TFNLayer(nn.Module):
    """
    Appends 1 to each modality embedding, computes 4-way outer product,
    applies elementwise tanh. Produces all interaction orders:
    unimodal / bimodal / trimodal / quadmodal.
    """

    def __init__(self, d_dims: List[int]):
        """
        Args:
            d_dims: [d_mouse, d_keyboard, d_notif, d_gnn] — sizes BEFORE 1-extension
        """
        super().__init__()
        assert len(d_dims) == 4
        self.d_dims   = d_dims
        self.ext_dims = [d + 1 for d in d_dims]

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: [h_mouse, h_kb, h_notif, h_gnn], each (B, d_m)
        Returns:
            (B, d0+1, d1+1, d2+1, d3+1) after tanh
        """
        assert len(embeddings) == 4
        B = embeddings[0].shape[0]

        extended = []
        for emb in embeddings:
            ones = torch.ones(B, 1, device=emb.device, dtype=emb.dtype)
            extended.append(torch.cat([emb, ones], dim=1))

        t = torch.einsum(
            'bi,bj,bk,bl->bijkl',
            extended[0], extended[1], extended[2], extended[3]
        )
        return torch.tanh(t)

    def get_slice(self, tensor: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """
        Extract and flatten the slice for modality_idx.
        Drops the 1-extension index on the modality's own axis;
        keeps full extended range on all other axes.

        Returns: (B, d_m × ∏_{j≠m}(d_j+1))
        """
        d      = self.d_dims[modality_idx]
        slices = [slice(None)] * 5
        slices[modality_idx + 1] = slice(0, d)
        return tensor[tuple(slices)].flatten(start_dim=1)

    def flat_size(self, modality_idx: int) -> int:
        """Flat input size for model slot i."""
        d      = self.d_dims[modality_idx]
        others = [self.ext_dims[j] for j in range(4) if j != modality_idx]
        return d * others[0] * others[1] * others[2]