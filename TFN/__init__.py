"""
TFN — active fusion layer.

Swap point: change the import below to switch between implementations.
Both satisfy the same interface (forward / get_slice / flat_size).

    ActiveTFN = LowRankTuckerFusion   ← default, tractable at AAM dims
    ActiveTFN = TFNLayer              ← full outer product, only feasible
                                         at small d_dims (ablation / debug)
"""

from .low_rank_tucker import LowRankTuckerFusion as ActiveTFN   # default
# from .tfn           import TFNLayer            as ActiveTFN   # full OProduct

__all__ = ["ActiveTFN"]