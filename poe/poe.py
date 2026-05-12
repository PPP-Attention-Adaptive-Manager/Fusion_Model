"""
PoE Fusion — Product of Experts in logit space.

Two modes (hot-swappable):
    'vanilla'  — equal-weight logit sum  (Hinton 2002)
    'weighted' — weight each expert by (1 - H_norm) before summing
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoEFusion(nn.Module):
    VANILLA  = "vanilla"
    WEIGHTED = "weighted"

    def __init__(self, mode: str = "vanilla"):
        super().__init__()
        assert mode in (self.VANILLA, self.WEIGHTED), \
            f"mode must be 'vanilla' or 'weighted', got '{mode}'"
        self.mode = mode

    def forward(self, per_model_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            per_model_outputs: list of 4 × (B, 12)
                dims 5–9  : state logits
                dim  10   : H_norm (used only in weighted mode)
        Returns:
            p_PoE: (B, 5)
        """
        logits = [out[:, 5:10] for out in per_model_outputs]  # 4 × (B, 5)

        if self.mode == self.VANILLA:
            fused = sum(logits)

        else:  # WEIGHTED
            H     = [out[:, 10] for out in per_model_outputs]
            w     = [1.0 - h for h in H]
            fused = sum(wi.unsqueeze(-1) * li for wi, li in zip(w, logits))

        return F.softmax(fused, dim=-1)   # (B, 5)