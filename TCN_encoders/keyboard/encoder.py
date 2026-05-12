"""
KeyboardBufferedEncoder — buffered TCN encoder for keyboard modality.

STATUS: v2 — real windowed implementation.

Input format (from teammate's KeystrokeEncoder / KeystrokeEncoder BiLSTM):
    numpy array of shape (64,) dtype float32
    — one embedding per completed window of W=20 keystrokes
    — emitted every S=10 keystrokes ≈ every 2–3 seconds at average typing speed

Emission is event-driven (per S keystrokes), not wall-clock timed.
At each 1Hz fusion tick the buffer serves the last known embedding
if no new one has arrived — staleness tracked via tau_decay.

Cold start behavior:
    First embedding arrives only after W=20 keystrokes (~4–5s).
    Until then, buffer is empty → left-padded with zeros → TCN sees
    zero sequence → output is near-zero embedding. Expected and handled.

d_in       = 64     (embed_dim from KeystrokeEncoder output contract)
TCN config : narrow (RF=31, kernel=3, layers=4) — IKI burst patterns at 100–500ms
tau_decay  = 15.0s  — ~2–3s between emissions, mild decay between updates
window_size = 10    — 10 buffered embeddings = ~20–30s of typing context

Output: (1, 64)  — narrow config n_channels=64
"""

import numpy as np
from typing import Optional

from ..buffered_encoder import BufferedEncoder
from ..configs          import narrow


D_IN        = 64     # embed_dim from KeystrokeEncoder output contract
TAU_DECAY   = 15.0   # seconds — freshness decay constant
WINDOW      = 10     # buffered embeddings


class KeyboardBufferedEncoder(BufferedEncoder):

    def __init__(self) -> None:
        super().__init__(
            d_in        = D_IN,
            cfg         = narrow,
            tau_decay   = TAU_DECAY,
            window_size = WINDOW,
        )

    def _parse_input(self, raw) -> Optional[np.ndarray]:
        """
        Args:
            raw : np.ndarray of shape (64,) from KeystrokeEncoder
                  OR torch.Tensor of shape (1, 64) or (64,)
                  OR None if no new window completed this tick
        Returns:
            np.ndarray of shape (64,)  OR  None
        """
        if raw is None:
            return None

        # handle both numpy and tensor inputs gracefully
        if hasattr(raw, 'detach'):
            raw = raw.detach().cpu().numpy()

        vec = np.asarray(raw, dtype=np.float32).reshape(D_IN,)

        assert vec.shape == (D_IN,), \
            f"KeyboardBufferedEncoder expected shape ({D_IN},), got {vec.shape}"

        return vec