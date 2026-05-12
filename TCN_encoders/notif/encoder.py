"""
NotifBufferedEncoder — buffered TCN encoder for notification modality.

Input format (from teammate's notification module, every 5s):
{
    "embedding":        np.array shape (16,) dtype float32,
    "npi":              float  [0.0, 1.0],
    "burstiness":       float  [0.0, 1.0],
    "disruption_score": float  [0.0, 1.0],
    "state":            str    — IGNORED (heuristic only)
    "metadata":         dict   — IGNORED
}

Parsed feature vector: [embedding(16), npi, burstiness, disruption_score]
d_in = 19

Update frequency: every 5s → staleness grows for 4 ticks between updates.
tau_decay = 10.0s: after 10s without update freshness ≈ 0.37
            after 20s freshness ≈ 0.14 — notif contribution nearly muted.

TCN config: shallow (RF=7, n_channels=32) — sparse signal, no deep history needed.
Output: (1, 32)
"""

import numpy as np
from typing import Optional

from ..buffered_encoder import BufferedEncoder
from ..configs          import shallow


D_IN      = 19      # 16-dim embedding + npi + burstiness + disruption_score
TAU_DECAY = 10.0    # seconds — freshness decay constant
WINDOW    = 10      # number of timesteps buffered (10s at 1Hz)


class NotifBufferedEncoder(BufferedEncoder):

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
            raw : dict from notification module  OR  None if no update this tick
        Returns:
            np.ndarray of shape (19,)  OR  None
        """
        if raw is None:
            return None

        embedding = np.asarray(raw["embedding"], dtype=np.float32)   # (16,)

        assert embedding.shape == (16,), \
            f"Notif embedding expected shape (16,), got {embedding.shape}"

        scalars = np.array([
            float(raw["npi"]),
            float(raw["burstiness"]),
            float(raw["disruption_score"]),
        ], dtype=np.float32)                                           # (3,)

        return np.concatenate([embedding, scalars])                    # (19,)