"""Mouse pre-embedder load / inference API for fusion use.

Additive only — does NOT change ``MouseEncoderP2`` or any existing API. Provides a
stable way to reload exported weights and run inference.

Public API
----------
    load_model(export_dir="pre_embedders/mouse/exports/mouse_encoder",
               device="auto") -> dict
    get_output(session, payload) -> dict

Output contract (unchanged): 64-D float32 embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .mouse_encoder import MouseEncoderP2

MODULE_NAME = "mouse"
EMBEDDING_DIM = 64
DEFAULT_EXPORT_DIR = "pre_embedders/mouse/exports/mouse_encoder"


def _select_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _torch_load(path: Path, *, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(export_dir: Union[str, Path] = DEFAULT_EXPORT_DIR, device: str = "auto") -> Dict[str, Any]:
    """Load the exported mouse encoder and rebuild the exact architecture."""
    export_path = Path(export_dir)
    selected_device = _select_device(device)

    checkpoint = _torch_load(export_path / "encoder.pt", map_location=selected_device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid mouse checkpoint at {export_path / 'encoder.pt'}")

    config = dict(checkpoint.get("model_config") or {})
    cfg_json = export_path / "model_config.json"
    if not config and cfg_json.is_file():
        config = json.loads(cfg_json.read_text(encoding="utf-8"))

    model = MouseEncoderP2(
        stats_dim=int(config.get("stats_dim", 22)),
        tcn_out_dim=int(config.get("tcn_out_dim", 64)),
        click_dim=int(config.get("click_dim", 32)),
        mlp_hidden=int(config.get("mlp_hidden", 64)),
        fusion_dim=int(config.get("fusion_dim", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return {
        "model": model,
        "model_config": config,
        "device": selected_device,
        "export_dir": export_path,
        "embedding_dim": int(config.get("embedding_dim", EMBEDDING_DIM)),
    }


def _cold_start(reason: str) -> Dict[str, Any]:
    return {
        "embedding": np.zeros(EMBEDDING_DIM, dtype=np.float32),
        "metadata": {
            "module": MODULE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "cold_start": True,
            "reason": reason,
        },
    }


def _to_tensor(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        t = value.detach().to(torch.float32)
    else:
        try:
            t = torch.as_tensor(np.asarray(value, dtype=np.float32))
        except (TypeError, ValueError):
            return None
    return t.to(device)


def get_output(session: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run MouseEncoderP2 on one payload.

    payload keys:
      seq           : (1,8,T) or (8,T)        — required
      stats         : (1,22)  or (22,)        — required
      pre_click_seq : (n_clicks,1,20)         — optional

    Returns {"embedding": (64,) float32, "metadata": {...}}. Missing / invalid
    payload -> zeros(64) with cold_start=True.
    """
    if not isinstance(payload, dict):
        return _cold_start("no_payload")

    device: torch.device = session["device"]
    model: MouseEncoderP2 = session["model"]

    seq = _to_tensor(payload.get("seq"), device)
    stats = _to_tensor(payload.get("stats"), device)
    if seq is None or stats is None:
        return _cold_start("missing_seq_or_stats")

    # Normalize shapes: seq -> (B,8,T), stats -> (B,22).
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)
    if stats.dim() == 1:
        stats = stats.unsqueeze(0)
    if seq.dim() != 3 or stats.dim() != 2:
        return _cold_start("bad_shape")

    pre_click = _to_tensor(payload.get("pre_click_seq"), device)
    if pre_click is not None and (pre_click.dim() != 3 or pre_click.shape[0] == 0):
        pre_click = None  # encoder treats this as no clicks

    try:
        with torch.no_grad():
            emb = model(seq, stats, pre_click).squeeze(0).detach().cpu().numpy().astype(np.float32)
    except (RuntimeError, ValueError) as exc:
        return _cold_start(f"forward_error:{type(exc).__name__}")

    if emb.shape != (EMBEDDING_DIM,) or not np.isfinite(emb).all():
        return _cold_start("bad_output")

    return {
        "embedding": emb,
        "metadata": {
            "module": MODULE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "cold_start": False,
        },
    }
