"""
BufferedEncoder — base class for all modality encoders.

Responsibilities:
    - Maintain a fixed-length deque of raw feature vectors
    - Pad (left) or truncate (right) to fixed T at every tick
    - Track staleness: seconds elapsed since last real update
    - Compute freshness weight for PoE staleness-aware weighting
    - Run FixedTCNEncoder on the assembled sequence
    - Return (embedding, freshness) at every tick

Subclasses must implement:
    _parse_input(raw) → Optional[np.ndarray]
        Convert raw incoming data (dict, tensor, list, None) into a 1D
        numpy array of shape (d_in,). Return None if nothing arrived.

Mouse is a special case — it overrides step() entirely (passthrough).
Keyboard v1 is a stub — it overrides step() with a static embedding.

Usage:
    class NotifBufferedEncoder(BufferedEncoder):
        def _parse_input(self, raw):
            if raw is None: return None
            return np.array([*raw["embedding"], raw["npi"], ...])

    enc = NotifBufferedEncoder(d_in=19, cfg=shallow, tau_decay=10.0, window_size=10)
    emb, freshness = enc.step(notif_dict_or_None)
    # emb: (1, 32)   freshness: float in (0, 1]
"""

import math
import numpy as np
import torch
from collections import deque
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .fixed_tcn import FixedTCNEncoder
from .configs   import TCNEncoderConfig


class BufferedEncoder(ABC):
    """
    Args:
        d_in        : feature dimension per timestep (modality-specific)
        cfg         : TCNEncoderConfig — passed to FixedTCNEncoder
        tau_decay   : freshness decay constant in seconds
                      freshness = exp(-staleness / tau_decay)
                      after tau_decay seconds without update → freshness ≈ 0.37
        window_size : number of timesteps T the TCN receives
    """

    def __init__(
        self,
        d_in       : int,
        cfg        : TCNEncoderConfig,
        tau_decay  : float,
        window_size: int = 10,
    ) -> None:
        self.d_in        = d_in
        self.tau_decay   = tau_decay
        self.window_size = window_size
        self.staleness   = 0.0                        # seconds since last real update
        self._buffer     = deque(maxlen=window_size)  # holds np.ndarray of shape (d_in,)
        self.encoder     = FixedTCNEncoder(d_in=d_in, cfg=cfg)

    # ------------------------------------------------------------------
    # Abstract — subclass implements this only
    # ------------------------------------------------------------------

    @abstractmethod
    def _parse_input(self, raw) -> Optional[np.ndarray]:
        """
        Convert raw incoming data into a feature vector.

        Args:
            raw : whatever the pre_embedder sends — dict, None, tensor, list
        Returns:
            np.ndarray of shape (d_in,)  if new data arrived
            None                         if nothing arrived this tick
        """
        ...

    # ------------------------------------------------------------------
    # Core tick — called once per second by the orchestrator
    # ------------------------------------------------------------------

    def step(self, raw_input) -> Tuple[torch.Tensor, float]:
        """
        Main entry point. Call once per 1s tick.

        Args:
            raw_input : raw data from pre_embedder (modality-specific format)
        Returns:
            embedding  : (1, n_channels) float32 tensor
            freshness  : float in (0, 1] — 1.0 = just updated, decays over time
        """
        vec = self._parse_input(raw_input)

        if vec is not None:
            assert vec.shape == (self.d_in,), \
                f"_parse_input must return shape ({self.d_in},), got {vec.shape}"
            self._buffer.append(vec)
            self.staleness = 0.0
        else:
            self.staleness += 1.0   # one more tick with no new data

        seq = self._pad_or_truncate()          # (1, window_size, d_in)
        emb = self.encoder(seq)                # (1, n_channels)
        return emb, self.freshness()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_or_truncate(self) -> torch.Tensor:
        """
        Assemble buffer into a (1, window_size, d_in) tensor.

        If buffer has fewer than window_size entries → left-pad with zeros.
        If buffer has exactly window_size entries   → use as-is.
        (deque maxlen handles the truncation case automatically.)
        """
        n = len(self._buffer)

        if n == 0:
            seq = np.zeros((self.window_size, self.d_in), dtype=np.float32)
        elif n < self.window_size:
            pad = np.zeros((self.window_size - n, self.d_in), dtype=np.float32)
            seq = np.vstack([pad, np.stack(list(self._buffer))])
        else:
            seq = np.stack(list(self._buffer))   # exactly window_size rows

        return torch.from_numpy(seq).unsqueeze(0)  # (1, window_size, d_in)

    def freshness(self) -> float:
        """
        Exponential decay of confidence based on staleness.
            freshness = exp(-staleness / tau_decay)
        Returns 1.0 when staleness=0, decays toward 0 as staleness grows.
        """
        return math.exp(-self.staleness / self.tau_decay)

    def reset(self) -> None:
        """Call at every LOSO subject boundary."""
        self._buffer.clear()
        self.staleness = 0.0