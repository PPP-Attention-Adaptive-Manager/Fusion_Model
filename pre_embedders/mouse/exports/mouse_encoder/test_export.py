"""Smoke test for the mouse encoder export package.

Verifies: load -> dummy payload (seq (1,8,512), stats (1,22), pre_click (3,1,20))
-> get_output -> shape/dtype/finite/non-zero, then reloads and checks determinism.

Run:
    python pre_embedders/mouse/exports/mouse_encoder/test_export.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_embedders.mouse.output import get_output, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the mouse encoder export.")
    parser.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)
    payload = {
        "seq": rng.standard_normal((1, 8, 512)).astype(np.float32),
        "stats": rng.standard_normal((1, 22)).astype(np.float32),
        "pre_click_seq": rng.standard_normal((3, 1, 20)).astype(np.float32),
    }

    session = load_model(export_dir=args.export_dir, device=args.device)
    result = get_output(session, payload)
    emb = result["embedding"]
    meta = result["metadata"]

    assert isinstance(emb, np.ndarray), type(emb)
    assert emb.shape == (64,), emb.shape
    assert emb.dtype == np.float32, emb.dtype
    assert not np.isnan(emb).any(), "NaN in embedding"
    assert np.isfinite(emb).all(), "Inf in embedding"
    assert np.count_nonzero(emb) > 0, "embedding is all-zero"
    assert meta["cold_start"] is False, meta

    # Determinism after reload.
    session2 = load_model(export_dir=args.export_dir, device=args.device)
    emb2 = get_output(session2, payload)["embedding"]
    assert np.allclose(emb, emb2, atol=1e-5), float(np.abs(emb - emb2).max())

    # Cold-start path.
    cold = get_output(session, None)
    assert cold["metadata"]["cold_start"] is True
    assert np.count_nonzero(cold["embedding"]) == 0

    print("Mouse export smoke test OK")
    print(f"shape={emb.shape}  dtype={emb.dtype}")
    print(f"l2_norm={float(np.linalg.norm(emb)):.6f}  max_reload_diff={float(np.abs(emb - emb2).max()):.2e}")


if __name__ == "__main__":
    main()
