"""
LowRankTuckerFusion — tractable 4-modality fusion via Tucker decomposition.

Replaces the full outer product TFN which is computationally impossible
at AAM's embedding dimensions:

    128 × 64 × 16 × 64 = 8,388,608 elements per sample  ← full TFN, intractable
    R^4 = 8^4          = 4,096     elements per sample  ← Tucker R=8, tractable

Reference:
    Zadeh et al. (2017) TFN — original full outer product
    Liu et al.  (2018) LMF — low-rank multimodal fusion, direct predecessor

Interface contract (identical to TFNLayer — plug/unplug compatible):
    forward(embeddings)      → (B, R, R, R, R) Tucker tensor
    get_slice(tensor, i)     → (B, R^3)         flat slice for model slot i
    flat_size(i)             → R^3              input_flat_dim for BaseModalityModel

Key invariants (from spec):
    - Projection matrices W_m/k/n/g are ALWAYS trainable — never frozen.
      Their gradients encode which directions in embedding space carry
      cross-modal interaction signal. Freezing them breaks Tucker.
    - tanh applied AFTER outer product, not before.
      tanh(z_i) ⊗ tanh(z_j) ≠ tanh(z_i ⊗ z_j) — applying before breaks interactions.
    - bias=False in all projection matrices.
      Bias before outer product introduces spurious additive terms that
      contaminate the interaction tensor with non-interaction signal.

Ablation:
    Rank R is the primary hyperparameter.
    To ablate: change rank= at construction in fusion_model.py.
    Recommended sweep: R ∈ {4, 8, 16}. Default R=8.
"""

from typing import List
import torch
import torch.nn as nn


class LowRankTuckerFusion(nn.Module):
    """
    Low-Rank Tucker Fusion for 4 modalities.

    Args:
        d_dims : [d_mouse, d_keyboard, d_notif, d_switching] — input embedding dims
        rank   : R — rank of the Tucker decomposition (default 8)
    """

    def __init__(self, d_dims: List[int], rank: int = 8) -> None:
        super().__init__()
        assert len(d_dims) == 4, "exactly 4 modality dims required"

        self.d_dims = d_dims
        self.rank   = rank

        # Per-modality projection matrices — TRAINABLE, bias=False (see invariants)
        self.projections = nn.ModuleList([
            nn.Linear(d, rank, bias=False)
            for d in d_dims
        ])

    # ------------------------------------------------------------------
    # Interface contract — must match TFNLayer exactly
    # ------------------------------------------------------------------

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings : [h_mouse, h_kb, h_notif, h_switching], each (B, d_m)
        Returns:
            (B, R, R, R, R) Tucker tensor after tanh
        """
        assert len(embeddings) == 4

        # Step 1 — project each modality to rank-R space
        # z_i : (B, R)
        z = [self.projections[i](embeddings[i]) for i in range(4)]

        # Step 2 — outer product in rank-R space via successive einsum
        # (B,R) ⊗ (B,R) → (B,R,R) → (B,R,R,R) → (B,R,R,R,R)
        Z = torch.einsum('bi,bj->bij',     z[0], z[1])   # (B, R, R)
        Z = torch.einsum('bij,bk->bijk',   Z,    z[2])   # (B, R, R, R)
        Z = torch.einsum('bijk,bl->bijkl', Z,    z[3])   # (B, R, R, R, R)

        # Step 3 — elementwise tanh AFTER outer product (see invariants)
        return torch.tanh(Z)                              # (B, R, R, R, R)

    def get_slice(self, tensor: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """
        Extract and flatten the slice for modality_idx.

        In the rank-R Tucker tensor, each axis corresponds to one modality.
        Slicing axis (modality_idx + 1) gives that modality's interaction
        context with all other modalities, collapsed to (B, R^3).

        This is structurally identical to TFNLayer.get_slice() — the only
        difference is that here every axis has size R instead of (d_m + 1).

        No 1-extension exists in Tucker (projections replace it), so we
        take the full R range on the modality's own axis.

        Args:
            tensor       : (B, R, R, R, R) from forward()
            modality_idx : 0=mouse, 1=keyboard, 2=notif, 3=switching
        Returns:
            (B, R^3)
        """
        # Select one "slice" along modality_idx's axis by summing over it
        # (equivalent to marginalising — keeps cross-modal context, removes
        # modality's own axis to avoid redundancy with the raw embedding
        # which the model also receives via the projector).
        #
        # Implementation: permute modality axis to last position,
        # then mean-pool over it → (B, R, R, R) → flatten → (B, R^3)
        #
        # Axis mapping: tensor axes are [B, mod0, mod1, mod2, mod3]
        #               modality_idx 0 → axis 1, idx 1 → axis 2, etc.
        axis = modality_idx + 1

        # Permute: move the modality's own axis to the end
        other_axes = [0] + [a for a in range(1, 5) if a != axis] + [axis]
        t = tensor.permute(*other_axes)          # (B, R, R, R, R)

        # Mean-pool over the modality's own axis (last dim after permute)
        t = t.mean(dim=-1)                       # (B, R, R, R)

        return t.flatten(start_dim=1)            # (B, R^3)

    def flat_size(self, modality_idx: int = 0) -> int:
        """
        Flat input size for any model slot.

        In Tucker all modality axes have the same size R, so flat_size
        is identical for all 4 modalities: R^3.

        modality_idx argument kept for interface compatibility with TFNLayer.
        """
        return self.rank ** 3