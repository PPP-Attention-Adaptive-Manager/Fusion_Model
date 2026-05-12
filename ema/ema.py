"""
EMA smoother + uncertainty metric helpers.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_uncertainty(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalised predictive entropy and margin confidence from logits.

    Args:
        logits: (B, 5)
    Returns:
        H_norm : (B,)  ∈ [0,1]  — 0 = confident, 1 = maximally uncertain
        M      : (B,)           — p_top1 − p_top2, small = ambiguous
    """
    probs  = F.softmax(logits, dim=-1)
    H      = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    H_norm = H / math.log(5)

    top2   = torch.topk(probs, k=2, dim=-1).values
    M      = top2[:, 0] - top2[:, 1]

    return H_norm, M


class EMAsmoother:
    """
    Exponential Moving Average over probability vectors.
    Stateful, parameter-free. Call reset() at every LOSO subject boundary.

        p_final(t) = α · p_PoE(t) + (1 − α) · p_final(t−1)

    First window uses uniform prior [0.2 × 5].
    """
    N_STATES = 5

    def __init__(self, alpha: float = 0.7):
        assert 0.0 < alpha <= 1.0
        self.alpha   = alpha
        self._p_prev: Optional[torch.Tensor] = None

    def reset(self):
        self._p_prev = None

    def step(self, p_poe: torch.Tensor) -> torch.Tensor:
        """
        Args:  p_poe: (B, 5)
        Returns: p_final: (B, 5)
        """
        if self._p_prev is None:
            self._p_prev = torch.full_like(p_poe, 1.0 / self.N_STATES)
        p_final      = self.alpha * p_poe + (1.0 - self.alpha) * self._p_prev
        self._p_prev = p_final.detach()
        return p_final