"""
SwitchingBufferedEncoder — buffered TCN encoder for GNN/switching modality.

Input format (from teammate's behavioral graph module, every 120s):
{
    "embedding":     np.array shape (64,) dtype float32,
    "graph_metrics": {
        "num_nodes":      int,
        "num_edges":      int,
        "density":        float [0.0, 1.0],
        "switch_rate":    float [0.0, 1.0],
        "fragmentation":  float [0.0, 1.0],
        "focus_ratio":    float [0.0, 1.0],
        "multitask_score":float [0.0, 1.0],
    },
    "state":    str   — IGNORED (heuristic only)
    "metadata": dict  — IGNORED
}

Parsed feature vector:
    [embedding(64), num_nodes, num_edges, density, switch_rate,
     fragmentation, focus_ratio, multitask_score]
d_in = 71

num_nodes and num_edges are raw integers — normalized by clipping to [0,1]
using reasonable upper bounds (50 nodes, 200 edges) to keep all features
in a comparable range before entering the TCN.

Update frequency: every 120s → very high staleness between updates.
tau_decay = 60.0s: after 60s freshness ≈ 0.37, after 120s ≈ 0.14.
The contribution of switching to PoE naturally fades between updates
and spikes back to 1.0 the moment a new graph arrives.

TCN config: shallow (RF=7, n_channels=32) — sparse signal.
Output: (1, 32)
"""

import numpy as np
from typing import Optional

from ..buffered_encoder import BufferedEncoder
from ..configs          import shallow


D_IN          = 71      # 64-dim embedding + 7 graph metrics
TAU_DECAY     = 60.0    # seconds — freshness decay constant
WINDOW        = 10      # buffered timesteps

# Normalization upper bounds for raw integer metrics
_MAX_NODES    = 50.0
_MAX_EDGES    = 200.0


class SwitchingBufferedEncoder(BufferedEncoder):

    def __init__(self) -> None:
        super().__init__(
            d_in        = D_IN,
            cfg         = shallow,
            tau_decay   = TAU_DECAY,
            window_size = WINDOW,
        )

    def _parse_input(self, raw) -> Optional[np.ndarray]:
        """
        Args:
            raw : dict from behavioral graph module  OR  None if no update this tick
        Returns:
            np.ndarray of shape (71,)  OR  None
        """
        if raw is None:
            return None

        embedding = np.asarray(raw["embedding"], dtype=np.float32)   # (64,)

        assert embedding.shape == (64,), \
            f"Switching embedding expected shape (64,), got {embedding.shape}"

        m = raw["graph_metrics"]

        metrics = np.array([
            float(m["num_nodes"])       / _MAX_NODES,    # normalize int → [0,1]
            float(m["num_edges"])       / _MAX_EDGES,    # normalize int → [0,1]
            float(m["density"]),                         # already [0,1]
            float(m["switch_rate"]),                     # already [0,1]
            float(m["fragmentation"]),                   # already [0,1]
            float(m["focus_ratio"]),                     # already [0,1]
            float(m["multitask_score"]),                 # already [0,1]
        ], dtype=np.float32)                             # (7,)

        return np.concatenate([embedding, metrics])      # (71,)