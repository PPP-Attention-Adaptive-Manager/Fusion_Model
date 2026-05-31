"""Switching encoders for the GNN/behavior graph modality.

Default path:
    GraphSAGE-GAE embedding (64D) -> SwitchingIdentityEncoder -> (1,64)

Ablation path:
    GraphSAGE-GAE embedding (64D) -> random-frozen TCN -> (1,32)

The GNN encoder already summarizes a 120-second graph window, so the default
fusion path preserves the learned 64D embedding. The temporal logic here only
handles buffering, cold start, and freshness/staleness.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import numpy as np
import torch

from ..buffered_encoder import BufferedEncoder
from ..configs import shallow


IDENTITY_MODE = "identity"
RANDOM_FROZEN_TCN_MODE = "random_frozen_tcn"
VALID_SWITCHING_ENCODER_MODES = {IDENTITY_MODE, RANDOM_FROZEN_TCN_MODE}
SWITCHING_ENCODER_MODE = os.getenv("SWITCHING_ENCODER_MODE", IDENTITY_MODE).strip().lower()

EMBEDDING_DIM = 64
RANDOM_FROZEN_D_IN = 71
TAU_DECAY = 60.0
WINDOW = 10

_MAX_NODES = 50.0
_MAX_EDGES = 200.0


def resolve_switching_encoder_mode(mode: Optional[str] = None) -> str:
    selected = (mode or os.getenv("SWITCHING_ENCODER_MODE", SWITCHING_ENCODER_MODE)).strip().lower()
    if selected not in VALID_SWITCHING_ENCODER_MODES:
        raise ValueError(
            "switching_encoder_mode must be one of "
            f"{sorted(VALID_SWITCHING_ENCODER_MODES)}, got {selected!r}"
        )
    return selected


class SwitchingIdentityEncoder:
    """Identity-only encoder for the exported GraphSAGE-GAE embedding.

    Input:
        dict with "embedding": np.ndarray shape (64,), dtype float32
        or None when no new 120-second graph window is available.

    Output:
        torch.Tensor shape (1,64), unchanged from the latest valid embedding.
    """

    output_dim = EMBEDDING_DIM

    def __init__(self, tau_decay: float = TAU_DECAY) -> None:
        self.tau_decay = tau_decay
        self.staleness = 0.0
        self._last_embedding: Optional[torch.Tensor] = None
        self.last_metadata: Dict[str, object] = {}
        self.last_cold_start = False

    def step(self, raw_input) -> tuple[torch.Tensor, float]:
        if raw_input is None:
            if self._last_embedding is None:
                self.staleness = 0.0
                return torch.zeros(1, EMBEDDING_DIM, dtype=torch.float32), 0.0
            self.staleness += 1.0
            return self._last_embedding.clone(), self.freshness()

        embedding = self._parse_embedding(raw_input)
        self._last_embedding = embedding
        self.staleness = 0.0
        self.last_metadata = dict(raw_input.get("metadata") or {})
        self.last_cold_start = bool(
            raw_input.get("cold_start", self.last_metadata.get("cold_start", False))
        )
        return embedding.clone(), 1.0

    def _parse_embedding(self, raw_input) -> torch.Tensor:
        if "embedding" not in raw_input:
            raise KeyError("Switching input is missing 'embedding'.")
        embedding = np.asarray(raw_input["embedding"], dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Switching embedding expected shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Switching embedding contains NaN or Inf values.")
        return torch.from_numpy(embedding.copy()).to(dtype=torch.float32).unsqueeze(0)

    def freshness(self) -> float:
        if self._last_embedding is None:
            return 0.0
        return math.exp(-self.staleness / self.tau_decay)

    def debug_state(self) -> Dict[str, object]:
        return {
            "metadata": dict(self.last_metadata),
            "cold_start": self.last_cold_start,
            "staleness": self.staleness,
            "freshness": self.freshness(),
            "mode": IDENTITY_MODE,
        }

    def reset(self) -> None:
        self.staleness = 0.0
        self._last_embedding = None
        self.last_metadata = {}
        self.last_cold_start = False


class SwitchingRandomFrozenTCNEncoder(BufferedEncoder):
    """Legacy random-frozen TCN switching path kept for ablations.

    Input is the same 64D GraphSAGE embedding. Optional graph_metrics are
    appended for the old 71D TCN input. Missing graph metrics default to zeros.
    Output shape is (1,32).
    """

    output_dim = 32

    def __init__(self) -> None:
        super().__init__(
            d_in=RANDOM_FROZEN_D_IN,
            cfg=shallow,
            tau_decay=TAU_DECAY,
            window_size=WINDOW,
        )
        self.last_metadata: Dict[str, object] = {}
        self.last_cold_start = False

    def _parse_input(self, raw) -> Optional[np.ndarray]:
        if raw is None:
            return None

        self.last_metadata = dict(raw.get("metadata") or {})
        self.last_cold_start = bool(
            raw.get("cold_start", self.last_metadata.get("cold_start", False))
        )

        embedding = np.asarray(raw["embedding"], dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Switching embedding expected shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Switching embedding contains NaN or Inf values.")

        metrics = self._parse_graph_metrics(raw.get("graph_metrics") or {})
        return np.concatenate([embedding, metrics])

    def _parse_graph_metrics(self, metrics: Dict[str, object]) -> np.ndarray:
        return np.array(
            [
                float(metrics.get("num_nodes", 0.0)) / _MAX_NODES,
                float(metrics.get("num_edges", 0.0)) / _MAX_EDGES,
                float(metrics.get("density", 0.0)),
                float(metrics.get("switch_rate", 0.0)),
                float(metrics.get("fragmentation", 0.0)),
                float(metrics.get("focus_ratio", 0.0)),
                float(metrics.get("multitask_score", 0.0)),
            ],
            dtype=np.float32,
        )

    def debug_state(self) -> Dict[str, object]:
        return {
            "metadata": dict(self.last_metadata),
            "cold_start": self.last_cold_start,
            "staleness": self.staleness,
            "freshness": self.freshness(),
            "mode": RANDOM_FROZEN_TCN_MODE,
        }

    def reset(self) -> None:
        super().reset()
        self.last_metadata = {}
        self.last_cold_start = False


def build_switching_encoder(mode: Optional[str] = None):
    selected = resolve_switching_encoder_mode(mode)
    if selected == IDENTITY_MODE:
        return SwitchingIdentityEncoder()
    return SwitchingRandomFrozenTCNEncoder()


class SwitchingBufferedEncoder:
    """Compatibility factory for the active switching encoder mode."""

    def __new__(cls, mode: Optional[str] = None):
        return build_switching_encoder(mode)

