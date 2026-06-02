"""Discover and verify mouse + keyboard embedder weight exports.

For each embedder it checks:
  * encoder.pt exists
  * load_model() works
  * dummy inference works
  * output shape == (64,), float32
  * output has no NaN / Inf and is not all-zero
  * output is deterministic after a fresh reload

Writes a JSON summary to outputs/embedders/embedder_weight_check.json and exits
non-zero if any embedder fails.

Usage:
    python scripts/embedders/check_embedder_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

OUT_DIR = PROJECT_ROOT / "outputs" / "embedders"
KEYBOARD_EXPORT = PROJECT_ROOT / "pre_embedders" / "keyboard" / "exports" / "keyboard_encoder"
MOUSE_EXPORT = PROJECT_ROOT / "pre_embedders" / "mouse" / "exports" / "mouse_encoder"


def _base_result() -> Dict[str, Any]:
    return {
        "encoder_pt_exists": False,
        "load_model_ok": False,
        "inference_ok": False,
        "shape_ok": False,
        "dtype_ok": False,
        "no_nan": False,
        "no_inf": False,
        "not_all_zero": False,
        "deterministic_after_reload": False,
        "passed": False,
        "error": None,
    }


def _finalize(r: Dict[str, Any]) -> Dict[str, Any]:
    r["passed"] = all(
        r[k] for k in [
            "encoder_pt_exists", "load_model_ok", "inference_ok", "shape_ok",
            "dtype_ok", "no_nan", "no_inf", "not_all_zero", "deterministic_after_reload",
        ]
    )
    return r


def check_keyboard() -> Dict[str, Any]:
    r = _base_result()
    r["export_dir"] = str(KEYBOARD_EXPORT)
    try:
        from pre_embedders.keyboard.output import get_output, load_model
        r["encoder_pt_exists"] = (KEYBOARD_EXPORT / "encoder.pt").is_file()
        if not r["encoder_pt_exists"]:
            r["error"] = "encoder.pt missing"
            return _finalize(r)

        session = load_model(export_dir=KEYBOARD_EXPORT, device="cpu")
        r["load_model_ok"] = True
        r["variant"] = session.get("variant")

        window = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
        out = get_output(session, window)
        emb = out["embedding"]
        r["inference_ok"] = True
        r["shape_ok"] = emb.shape == (64,)
        r["dtype_ok"] = emb.dtype == np.float32
        r["no_nan"] = not bool(np.isnan(emb).any())
        r["no_inf"] = bool(np.isfinite(emb).all())
        r["not_all_zero"] = int(np.count_nonzero(emb)) > 0
        r["l2_norm"] = float(np.linalg.norm(emb))

        emb2 = get_output(load_model(export_dir=KEYBOARD_EXPORT, device="cpu"), window)["embedding"]
        r["deterministic_after_reload"] = bool(np.allclose(emb, emb2, atol=1e-5))
        r["max_reload_diff"] = float(np.abs(emb - emb2).max())
    except Exception as exc:  # noqa: BLE001 - report any failure
        r["error"] = f"{type(exc).__name__}: {exc}"
    return _finalize(r)


def check_mouse() -> Dict[str, Any]:
    r = _base_result()
    r["export_dir"] = str(MOUSE_EXPORT)
    try:
        from pre_embedders.mouse.output import get_output, load_model
        r["encoder_pt_exists"] = (MOUSE_EXPORT / "encoder.pt").is_file()
        if not r["encoder_pt_exists"]:
            r["error"] = "encoder.pt missing"
            return _finalize(r)

        session = load_model(export_dir=MOUSE_EXPORT, device="cpu")
        r["load_model_ok"] = True

        rng = np.random.default_rng(0)
        payload = {
            "seq": rng.standard_normal((1, 8, 512)).astype(np.float32),
            "stats": rng.standard_normal((1, 22)).astype(np.float32),
            "pre_click_seq": rng.standard_normal((3, 1, 20)).astype(np.float32),
        }
        out = get_output(session, payload)
        emb = out["embedding"]
        r["inference_ok"] = True
        r["shape_ok"] = emb.shape == (64,)
        r["dtype_ok"] = emb.dtype == np.float32
        r["no_nan"] = not bool(np.isnan(emb).any())
        r["no_inf"] = bool(np.isfinite(emb).all())
        r["not_all_zero"] = int(np.count_nonzero(emb)) > 0
        r["l2_norm"] = float(np.linalg.norm(emb))

        emb2 = get_output(load_model(export_dir=MOUSE_EXPORT, device="cpu"), payload)["embedding"]
        r["deterministic_after_reload"] = bool(np.allclose(emb, emb2, atol=1e-5))
        r["max_reload_diff"] = float(np.abs(emb - emb2).max())
    except Exception as exc:  # noqa: BLE001
        r["error"] = f"{type(exc).__name__}: {exc}"
    return _finalize(r)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"keyboard": check_keyboard(), "mouse": check_mouse()}
    summary = {
        "all_passed": all(v["passed"] for v in results.values()),
        "embedders": results,
    }
    out_path = OUT_DIR / "embedder_weight_check.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {out_path}")
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
