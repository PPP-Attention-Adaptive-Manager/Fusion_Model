"""Smoke test for the FINAL keyboard embedder export."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import output  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    window = rng.standard_normal((20, 3)).astype(np.float32)
    session = output.load_model(device="cpu")
    res = output.get_output(session, window)
    emb, meta = res["embedding"], res["metadata"]

    assert emb.shape == (64,), emb.shape
    assert emb.dtype == np.float32, emb.dtype
    assert not np.isnan(emb).any() and np.isfinite(emb).all()
    assert np.count_nonzero(emb) > 0, "all-zero embedding"
    assert meta["cold_start"] is False and meta["trained"] is True

    emb2 = output.get_output(output.load_model(device="cpu"), window)["embedding"]
    assert np.allclose(emb, emb2, atol=1e-5), float(np.abs(emb - emb2).max())

    cold = output.get_output(session, None)
    assert cold["metadata"]["cold_start"] is True and np.count_nonzero(cold["embedding"]) == 0

    print("Keyboard FINAL export smoke test OK")
    print(f"shape={emb.shape} dtype={emb.dtype} l2={float(np.linalg.norm(emb)):.4f} "
          f"reload_diff={float(np.abs(emb - emb2).max()):.2e}")


if __name__ == "__main__":
    main()
