"""Runtime smoke test for the switching GraphSAGE -> fusion path.

Examples:
    python run_switching_fusion_smoke.py
    python run_switching_fusion_smoke.py --graph data/session_x/data_graph/data_graph_120s/graph_001.json
    python run_switching_fusion_smoke.py --require-graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from fusion_model import InferrerFusion
from pre_embedders.switching import load_model as load_switching_model
from TCN_encoders.switching.encoder import SwitchingBufferedEncoder


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _find_graph(data_dir: Path) -> Optional[Path]:
    patterns = [
        "**/data_graph_120s/graph_*.json",
        "**/switching/data_graph_120s/graph_*.json",
        "**/graph_*.json",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(data_dir.glob(pattern)):
            if path in seen or not path.is_file() or path.stat().st_size == 0:
                continue
            seen.add(path)
            return path
    return None


def _cold_start_graph() -> dict[str, Any]:
    return {
        "graph_id": "cold_start_smoke",
        "session_id": "runtime_smoke",
        "window": {"window_id": "window_000"},
        "nodes": [],
        "edges": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraphSAGE switching encoder and InferrerFusion once."
    )
    parser.add_argument("--graph", type=Path, default=None, help="Path to a graph JSON file.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("pre_embedders/switching/exports/switching_encoder"),
        help="Switching encoder export directory.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument(
        "--require-graph",
        action="store_true",
        help="Fail instead of running cold start when no graph is found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph or _find_graph(args.data_dir)
    if graph_path is None and args.require_graph:
        raise FileNotFoundError(f"No graph_*.json found under {args.data_dir}")

    graph_input: str | Path | dict[str, Any]
    if graph_path is None:
        graph_input = _cold_start_graph()
        graph_source = "cold_start_payload"
    else:
        graph_input = graph_path
        graph_source = str(graph_path)

    switching_session = load_switching_model(export_dir=args.export_dir, device=args.device)
    switching_encoder = SwitchingBufferedEncoder()

    switching_payload = switching_session.get_fusion_input(graph_input)
    h_switching, switching_freshness = switching_encoder.step(switching_payload)

    fusion = InferrerFusion()
    fusion.eval()

    embeddings = [
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(1, 32, dtype=torch.float32),
        h_switching,
    ]

    with torch.no_grad():
        output = fusion(embeddings)

    output = switching_session.attach_debug(output, freshness=switching_freshness)

    switching_embedding = switching_payload["embedding"]
    metadata = switching_payload["metadata"]
    summary = {
        "graph_source": graph_source,
        "switching_embedding_shape": tuple(switching_embedding.shape),
        "switching_embedding_dtype": str(switching_embedding.dtype),
        "switching_embedding_l2_norm": float(np.linalg.norm(switching_embedding)),
        "switching_tensor_shape": tuple(h_switching.shape),
        "switching_freshness": float(switching_freshness),
        "switching_metadata": metadata,
        "fusion_global_shape": tuple(output["global"].shape),
        "fusion_per_model_shapes": [tuple(t.shape) for t in output["per_model"]],
        "fusion_state_logits": output["global"][0, 5:10],
        "fusion_state_probs": torch.softmax(output["global"][0, 5:10], dim=-1),
        "debug": output.get("debug", {}),
        "note": "Mouse/keyboard/notif embeddings are zero dummies; predictive models are smoke-test baselines except existing notif.",
    }

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
