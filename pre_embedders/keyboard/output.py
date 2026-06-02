"""Keyboard pre-embedder load / inference API for fusion use.

This module is additive — it does NOT change the encoder architecture, the
training objective, or any existing public API. It only provides a stable way to
reload exported weights and run inference.

Public API
----------
    load_model(export_dir="pre_embedders/keyboard/exports/keyboard_encoder",
               device="auto") -> dict
    get_output(session, events_or_window) -> dict

Output contract (unchanged): 64-D float32 embedding. No L2 normalization is
applied — the KeystrokeEncoder emits a raw (unbounded) embedding by design, so
the fusion-facing contract here is identical to the encoder's native output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .encoder import KeystrokeEncoder
from .preprocess import normalize_sequence

MODULE_NAME = "keyboard"
EMBEDDING_DIM = 64
DEFAULT_EXPORT_DIR = "pre_embedders/keyboard/exports/keyboard_encoder"


def _select_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _torch_load(path: Path, *, map_location: torch.device) -> Dict[str, Any]:
    # Local, trusted checkpoint that intentionally contains config dicts as well
    # as tensors, so we cannot use weights_only=True.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        return torch.load(path, map_location=map_location)


def load_model(export_dir: Union[str, Path] = DEFAULT_EXPORT_DIR, device: str = "auto") -> Dict[str, Any]:
    """Load the exported keyboard encoder and rebuild the exact architecture.

    Returns a session dict with the eval-mode model, its config, and device.
    """
    export_path = Path(export_dir)
    selected_device = _select_device(device)

    checkpoint = _torch_load(export_path / "encoder.pt", map_location=selected_device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid keyboard checkpoint at {export_path / 'encoder.pt'}")

    config = dict(checkpoint.get("model_config") or {})
    # Fall back to model_config.json if the checkpoint omitted the config.
    cfg_json = export_path / "model_config.json"
    if not config and cfg_json.is_file():
        config = json.loads(cfg_json.read_text(encoding="utf-8"))

    variant = checkpoint.get("variant", "bilstm" if config.get("bidirectional") else "lstm")

    model = KeystrokeEncoder(
        input_size=int(config.get("input_size", 3)),
        hidden_size=int(config.get("hidden_size", EMBEDDING_DIM)),
        num_layers=int(config.get("num_layers", 2)),
        bidirectional=bool(config.get("bidirectional", variant == "bilstm")),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return {
        "model": model,
        "model_config": config,
        "variant": variant,
        "device": selected_device,
        "export_dir": export_path,
        "embedding_dim": int(config.get("embedding_dim", EMBEDDING_DIM)),
    }


def _cold_start(variant: str, reason: str) -> Dict[str, Any]:
    return {
        "embedding": np.zeros(EMBEDDING_DIM, dtype=np.float32),
        "metadata": {
            "module": MODULE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "variant": variant,
            "cold_start": True,
            "reason": reason,
        },
    }


def _to_window_array(events_or_window: Any) -> Optional[np.ndarray]:
    """Coerce supported inputs into a normalized (W, 3) float32 array.

    * torch.Tensor / np.ndarray of shape (W, 3) -> taken as already-normalized
      feature windows (run as-is; we do not re-normalize numeric feature arrays).
    * list of event dicts {code, hold, ikl} -> normalize_sequence(...) -> (W, 3).
    * anything empty / None / malformed -> None (caller emits cold start).
    """
    if events_or_window is None:
        return None

    if isinstance(events_or_window, torch.Tensor):
        arr = events_or_window.detach().cpu().numpy()
    elif isinstance(events_or_window, np.ndarray):
        arr = events_or_window
    elif isinstance(events_or_window, list):
        if len(events_or_window) == 0:
            return None
        if isinstance(events_or_window[0], dict):
            arr = normalize_sequence(events_or_window)
        else:
            arr = np.asarray(events_or_window, dtype=np.float32)
    else:
        return None

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 3:
        return None
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr).astype(np.float32)
    return arr.astype(np.float32)


def get_output(session: Dict[str, Any], events_or_window: Any) -> Dict[str, Any]:
    """Run the keyboard encoder on one window / event list.

    Returns {"embedding": (64,) float32, "metadata": {...}}.
    On missing/invalid input returns a zero embedding with cold_start=True.
    """
    variant = session.get("variant", "lstm")
    window = _to_window_array(events_or_window)
    if window is None:
        return _cold_start(variant, reason="no_valid_input")

    model: KeystrokeEncoder = session["model"]
    device: torch.device = session["device"]

    x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, W, 3)
    with torch.no_grad():
        emb = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (64,)

    if not np.isfinite(emb).all():
        return _cold_start(variant, reason="non_finite_output")

    return {
        "embedding": emb,
        "metadata": {
            "module": MODULE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "variant": variant,
            "cold_start": False,
        },
    }
