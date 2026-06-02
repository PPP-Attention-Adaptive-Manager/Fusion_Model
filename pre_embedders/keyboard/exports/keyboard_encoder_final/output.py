"""Load / inference API for the FINAL keyboard embedder.

    session = load_model(export_dir, device="auto")
    result  = get_output(session, payload)   # payload: (W,3) array | [event dicts] | None

result = {"embedding": (64,) float32, "metadata": {...}}
Cold-start (zeros) ONLY when input is truly missing/invalid; a real (even short)
window produces a valid non-zero embedding.
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

from pre_embedders.keyboard.encoder_final import KeyboardEncoderFinal  # noqa: E402

MODULE = "keyboard"
EMB_DIM = 64


def _device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def load_model(export_dir: str | Path = None, device: str = "auto") -> Dict[str, Any]:
    export_dir = Path(export_dir) if export_dir else Path(__file__).resolve().parent
    dev = _device(device)
    try:
        ckpt = torch.load(export_dir / "encoder.pt", map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(export_dir / "encoder.pt", map_location=dev)
    cfg = ckpt.get("model_config") or json.loads((export_dir / "model_config.json").read_text())
    model = KeyboardEncoderFinal(
        input_size=int(cfg.get("input_size", 3)),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        bidirectional=bool(cfg.get("bidirectional", True)),
        l2_normalize=bool(cfg.get("l2_normalize", False)),
        n_aux=int(cfg.get("n_aux", 5)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()
    return {"model": model, "config": cfg, "device": dev,
            "window_size": int(cfg.get("window_size", 20))}


def _cold(reason: str) -> Dict[str, Any]:
    return {"embedding": np.zeros(EMB_DIM, np.float32),
            "metadata": {"module": MODULE, "embedding_dim": EMB_DIM, "trained": True,
                         "cold_start": True, "reason": reason, "model_version": "final_v1"}}


def _to_window(payload: Any) -> Optional[np.ndarray]:
    from pre_embedders.keyboard.preprocess import normalize_sequence
    if payload is None:
        return None
    if isinstance(payload, torch.Tensor):
        arr = payload.detach().cpu().numpy()
    elif isinstance(payload, np.ndarray):
        arr = payload
    elif isinstance(payload, list):
        if not payload:
            return None
        if isinstance(payload[0], dict):
            arr = normalize_sequence(payload)
        else:
            arr = np.asarray(payload, dtype=np.float32)
    else:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 3:
        return None
    return np.nan_to_num(arr).astype(np.float32)


def get_output(session: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    w = _to_window(payload)
    if w is None:
        return _cold("no_valid_input")
    x = torch.from_numpy(w).unsqueeze(0).to(session["device"])
    with torch.no_grad():
        emb = session["model"](x).squeeze(0).cpu().numpy().astype(np.float32)
    if not np.isfinite(emb).all():
        return _cold("non_finite")
    return {"embedding": emb,
            "metadata": {"module": MODULE, "embedding_dim": EMB_DIM, "trained": True,
                         "cold_start": False, "model_version": "final_v1"}}
