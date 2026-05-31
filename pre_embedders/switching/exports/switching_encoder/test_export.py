"""Smoke test for the encoder-only switching export package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from output import get_output, load_model


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _find_graph(data_dir: Path, window_size: str) -> Path:
    target_dir = f"data_graph_{window_size}"
    candidates = sorted(path for path in data_dir.glob(f"**/{target_dir}/graph_*.json") if path.is_file())
    if not candidates:
        candidates = sorted(path for path in data_dir.rglob("graph_*.json") if path.is_file())
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("nodes"), list) and payload["nodes"]:
            return path
    raise FileNotFoundError(f"No non-empty graph JSON found under {data_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the switching encoder export.")
    parser.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--window-size", type=str, default="120s")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph or _find_graph(args.data_dir, args.window_size)
    session = load_model(export_dir=args.export_dir, device=args.device)
    result = get_output(session, graph_path)
    embedding = result["embedding"]
    metadata = result["metadata"]

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,), embedding.shape
    assert embedding.dtype == np.float32, embedding.dtype
    assert not np.isnan(embedding).any()
    assert np.isfinite(embedding).all()
    if not metadata.get("cold_start", False):
        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) < 1e-4, norm

    print("Export smoke test OK")
    print(f"graph={graph_path}")
    print(f"cold_start={metadata.get('cold_start', False)}")
    print(f"embedding_shape={embedding.shape}")
    print(f"embedding_dtype={embedding.dtype}")
    print(f"embedding_l2_norm={float(np.linalg.norm(embedding)):.6f}")


if __name__ == "__main__":
    main()
