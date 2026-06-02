"""Switching graph encoder integration for fusion.

This module wraps the encoder-only export package:

    pre_embedders/switching/exports/switching_encoder/

It loads the exported GraphSAGE encoder once, calls its public get_output API
for each 120-second graph window, validates the embedding contract, and exposes
a fusion-ready payload. The GAE decoder is never loaded or called here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import numpy as np


EMBEDDING_DIM = 64
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "exports" / "switching_encoder"
_EXPORT_MODULE_NAME = "_fusion_switching_encoder_export"


def _load_export_module(export_dir: Path) -> ModuleType:
    output_path = export_dir / "output.py"
    if not output_path.is_file():
        raise FileNotFoundError(f"Switching encoder output.py not found at {output_path}")

    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, output_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import switching encoder export from {output_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_MODULE_NAME] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "load_model") or not hasattr(module, "get_output"):
        raise AttributeError("Switching export must expose load_model() and get_output().")
    return module


def validate_switching_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the exported switching encoder result."""
    if not isinstance(result, dict):
        raise TypeError(f"Switching encoder result must be a dict, got {type(result)!r}")
    if "embedding" not in result:
        raise KeyError("Switching encoder result is missing 'embedding'.")

    embedding = result["embedding"]
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"Switching embedding must be np.ndarray, got {type(embedding)!r}")
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Switching embedding expected shape ({EMBEDDING_DIM},), got {embedding.shape}")
    if embedding.dtype != np.float32:
        raise TypeError(f"Switching embedding expected dtype np.float32, got {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise ValueError("Switching embedding contains NaN or Inf values.")

    metadata = dict(result.get("metadata") or {})
    cold_start = bool(metadata.get("cold_start", False))
    if not cold_start:
        norm = float(np.linalg.norm(embedding))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"Switching embedding must be L2-normalized, got norm={norm:.6f}")
        if metadata.get("decoder_used_at_inference") is not False:
            raise ValueError("Switching export metadata must report decoder_used_at_inference=False.")

    metadata.setdefault("module", "switching")
    metadata.setdefault("embedding_dim", EMBEDDING_DIM)
    metadata.setdefault("cold_start", cold_start)
    return {"embedding": embedding, "metadata": metadata}


def attach_switching_debug(
    fusion_output: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    freshness: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach switching metadata to a fusion output dict for debug logging."""
    debug = dict(fusion_output.get("debug") or {})
    switching_debug = {"metadata": dict(metadata)}
    if freshness is not None:
        switching_debug["freshness"] = float(freshness)
    switching_debug["cold_start"] = bool(metadata.get("cold_start", False))
    debug["switching"] = switching_debug
    fusion_output["debug"] = debug
    return fusion_output


class SwitchingGraphEncoder:
    """Clean runtime submodule for the exported switching GraphSAGE encoder."""

    def __init__(
        self,
        export_dir: str | Path = DEFAULT_EXPORT_DIR,
        device: str = "auto",
    ) -> None:
        self.export_dir = Path(export_dir)
        self.device = device
        self._export_api = _load_export_module(self.export_dir)
        self.session = self._export_api.load_model(export_dir=self.export_dir, device=device)
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_metadata: Dict[str, Any] = {}

    def get_output(self, graph_json_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        """Run encoder-only inference for one graph window."""
        result = self._export_api.get_output(self.session, graph_json_or_path)
        validated = validate_switching_output(result)
        self.last_result = validated
        self.last_metadata = dict(validated["metadata"])
        return validated

    def get_fusion_input(self, graph_json_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        """Return the payload expected by TCN_encoders.switching.SwitchingBufferedEncoder."""
        result = self.get_output(graph_json_or_path)
        return {
            "embedding": result["embedding"],
            "metadata": result["metadata"],
            "cold_start": bool(result["metadata"].get("cold_start", False)),
        }

    def attach_debug(
        self,
        fusion_output: Dict[str, Any],
        *,
        freshness: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Attach the last switching metadata to the fusion output debug block."""
        return attach_switching_debug(fusion_output, self.last_metadata, freshness=freshness)


def load_model(export_dir: str | Path = DEFAULT_EXPORT_DIR, device: str = "auto") -> SwitchingGraphEncoder:
    """Fusion-facing loader. Loads the switching encoder once at startup."""
    return SwitchingGraphEncoder(export_dir=export_dir, device=device)


def get_output(
    session: SwitchingGraphEncoder,
    graph_json_or_path: Dict[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Fusion-facing output API mirroring the exported package."""
    return session.get_output(graph_json_or_path)

