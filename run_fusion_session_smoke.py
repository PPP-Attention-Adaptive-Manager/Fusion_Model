"""Run fusion over all 120s switching graph windows in a session.

This is a runtime smoke runner, not a training/evaluation script. It uses real
switching GraphSAGE embeddings and zero dummy embeddings for mouse, keyboard,
and notif until those modalities are wired to session data.

Examples:
    python run_fusion_session_smoke.py
    python run_fusion_session_smoke.py --session-dir data/session_20260510_191823_9a2425
    python run_fusion_session_smoke.py --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

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


def _session_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("session_"):
            return part
    return path.stem


def _find_session_dirs(data_dir: Path) -> list[Path]:
    sessions = sorted(path for path in data_dir.glob("session_*") if path.is_dir())
    if sessions:
        return sessions
    return [data_dir] if data_dir.exists() else []


def _find_graphs(root: Path) -> list[Path]:
    patterns = [
        "**/data_graph_120s/graph_*.json",
        "**/switching/data_graph_120s/graph_*.json",
    ]
    graphs: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in seen and path.is_file() and path.stat().st_size > 0:
                seen.add(path)
                graphs.append(path)
    return sorted(graphs)


def _iter_graphs(
    *,
    data_dir: Path,
    session_dir: Optional[Path],
    graph_dir: Optional[Path],
    limit: Optional[int],
) -> tuple[str, list[Path]]:
    if graph_dir is not None:
        graphs = sorted(path for path in graph_dir.glob("graph_*.json") if path.is_file())
        selected_session = _session_id_from_path(graph_dir)
    elif session_dir is not None:
        graphs = _find_graphs(session_dir)
        selected_session = _session_id_from_path(session_dir)
    else:
        sessions = _find_session_dirs(data_dir)
        if not sessions:
            return "no_session", []
        selected = sessions[0]
        graphs = _find_graphs(selected)
        selected_session = _session_id_from_path(selected)

    if limit is not None:
        graphs = graphs[:limit]
    return selected_session, graphs


def _default_output_path(output_dir: Path, session_id: str) -> Path:
    return output_dir / session_id / "fusion_outputs.jsonl"


def _build_embeddings(h_switching: torch.Tensor) -> list[torch.Tensor]:
    return [
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(1, 32, dtype=torch.float32),
        h_switching,
    ]


def _record_for_window(
    *,
    graph_path: Path,
    index: int,
    switching_payload: dict[str, Any],
    h_switching: torch.Tensor,
    switching_freshness: float,
    fusion_output: dict[str, Any],
) -> dict[str, Any]:
    metadata = switching_payload["metadata"]
    embedding = switching_payload["embedding"]
    return {
        "window_index": index,
        "graph_path": str(graph_path),
        "graph_id": metadata.get("graph_id"),
        "session_id": metadata.get("session_id"),
        "window_id": metadata.get("window_id"),
        "switching": {
            "embedding_shape": tuple(embedding.shape),
            "embedding_dtype": str(embedding.dtype),
            "embedding_l2_norm": float(np.linalg.norm(embedding)),
            "tensor_shape": tuple(h_switching.shape),
            "freshness": float(switching_freshness),
            "metadata": metadata,
            "debug_state": fusion_output.get("debug", {}).get("switching", {}),
        },
        "fusion": {
            "global": fusion_output["global"][0],
            "per_model": [tensor[0] for tensor in fusion_output["per_model"]],
            "global_shape": tuple(fusion_output["global"].shape),
            "per_model_shapes": [tuple(t.shape) for t in fusion_output["per_model"]],
        },
        "note": "Mouse/keyboard/notif embeddings are zero dummies; predictive models are smoke-test baselines except existing notif.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fusion over switching graph windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion_smoke"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("pre_embedders/switching/exports/switching_encoder"),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id, graphs = _iter_graphs(
        data_dir=args.data_dir,
        session_dir=args.session_dir,
        graph_dir=args.graph_dir,
        limit=args.limit,
    )
    if not graphs:
        raise FileNotFoundError(
            "No graph_*.json files found. Expected data/**/data_graph_120s/graph_*.json."
        )

    output_path = args.output or _default_output_path(args.output_dir, session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    switching_session = load_switching_model(export_dir=args.export_dir, device=args.device)
    switching_encoder = SwitchingBufferedEncoder()

    fusion = InferrerFusion()
    fusion.eval()

    cold_starts = 0
    norms: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, graph_path in enumerate(graphs):
            switching_payload = switching_session.get_fusion_input(graph_path)
            h_switching, switching_freshness = switching_encoder.step(switching_payload)

            with torch.no_grad():
                fusion_output = fusion(_build_embeddings(h_switching))
            fusion_output = switching_session.attach_debug(
                fusion_output,
                freshness=switching_freshness,
            )

            metadata = switching_payload["metadata"]
            cold_starts += int(bool(metadata.get("cold_start", False)))
            norms.append(float(np.linalg.norm(switching_payload["embedding"])))

            record = _record_for_window(
                graph_path=graph_path,
                index=index,
                switching_payload=switching_payload,
                h_switching=h_switching,
                switching_freshness=switching_freshness,
                fusion_output=fusion_output,
            )
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    summary = {
        "session_id": session_id,
        "num_windows": len(graphs),
        "cold_starts": cold_starts,
        "output_path": str(output_path),
        "first_graph": str(graphs[0]),
        "last_graph": str(graphs[-1]),
        "embedding_norm_min": min(norms),
        "embedding_norm_max": max(norms),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

