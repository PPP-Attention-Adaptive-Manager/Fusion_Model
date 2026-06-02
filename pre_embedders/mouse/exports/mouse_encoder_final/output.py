"""Load / inference API for the FINAL mouse embedder.

    session = load_model(export_dir, device="auto")
    result  = get_output(session, payload)

payload = {"seq": (8,T) or (1,8,T) float, "stats": raw (8,) or (1,8) behavioral
features in MOUSE_FEATURES order}. The saved stats_scaler is applied internally.

result = {"embedding": (64,) float32, "metadata": {...}}
Cold-start (zeros) ONLY when seq/stats are missing/invalid. A low-activity
(idle) window still produces a valid non-zero embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_embedders.mouse.encoder_final import MouseEncoderFinal  # noqa: E402

MODULE = "mouse"
EMB_DIM = 64


def _device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


class _Scaler:
    def __init__(self, d):
        self.median = np.asarray(d["median"], np.float64)
        self.scale = np.asarray(d["scale"], np.float64)
        self.clip = float(d.get("clip", 5.0))

    def transform(self, X):
        z = (np.asarray(X, np.float64) - self.median) / self.scale
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def load_model(export_dir: str | Path = None, device: str = "auto") -> Dict[str, Any]:
    export_dir = Path(export_dir) if export_dir else Path(__file__).resolve().parent
    dev = _device(device)
    try:
        ckpt = torch.load(export_dir / "encoder.pt", map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(export_dir / "encoder.pt", map_location=dev)
    cfg = ckpt.get("model_config") or json.loads((export_dir / "model_config.json").read_text())
    model = MouseEncoderFinal(
        seq_channels=int(cfg.get("seq_channels", 8)),
        n_stats=int(cfg.get("n_stats", 8)),
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        l2_normalize=bool(cfg.get("l2_normalize", False)),
        n_aux=int(cfg.get("n_aux", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()
    return {"model": model, "config": cfg, "device": dev,
            "stats_scaler": _Scaler(ckpt["stats_scaler"]),
            "seq_len": int(cfg.get("seq_len", 256))}


def _cold(reason: str) -> Dict[str, Any]:
    return {"embedding": np.zeros(EMB_DIM, np.float32),
            "metadata": {"module": MODULE, "embedding_dim": EMB_DIM, "trained": True,
                         "cold_start": True, "reason": reason, "model_version": "final_v1"}}


def _as_arr(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    try:
        return np.asarray(v, dtype=np.float32)
    except (TypeError, ValueError):
        return None


def get_output(session: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _cold("no_payload")
    seq = _as_arr(payload.get("seq"))
    stats = _as_arr(payload.get("stats"))
    if seq is None or stats is None:
        return _cold("missing_seq_or_stats")
    if seq.ndim == 2:
        seq = seq[None, ...]
    if stats.ndim == 1:
        stats = stats[None, ...]
    if seq.ndim != 3 or seq.shape[1] != int(session["config"].get("seq_channels", 8)):
        return _cold("bad_seq_shape")
    stats_n = session["stats_scaler"].transform(stats)
    dev = session["device"]
    with torch.no_grad():
        emb = session["model"](
            torch.from_numpy(seq).to(dev), torch.from_numpy(stats_n).to(dev)
        ).squeeze(0).cpu().numpy().astype(np.float32)
    if emb.shape != (EMB_DIM,) or not np.isfinite(emb).all():
        return _cold("bad_output")
    return {"embedding": emb,
            "metadata": {"module": MODULE, "embedding_dim": EMB_DIM, "trained": True,
                         "cold_start": False, "model_version": "final_v1"}}
