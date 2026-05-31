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
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from fusion_model import InferrerFusion
from pre_embedders.switching import load_model as load_switching_model
from TCN_encoders.notif.encoder import NotifBufferedEncoder
from TCN_encoders.switching.encoder import SwitchingBufferedEncoder


NOTIF_SCALER_PARAMS = {
    "min_": [0.0, 0.0, 0.0, 0.0, 0.0],
    "max_": [3.0, 2.509182763787976e-07, 1.0, 0.021615064589633373, 1.0],
}


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


def _session_root_from_path(path: Path) -> Optional[Path]:
    resolved = path.resolve()
    for candidate in [resolved, *resolved.parents]:
        if candidate.name.startswith("session_") and candidate.is_dir():
            return candidate
    return None


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
) -> tuple[str, Optional[Path], list[Path]]:
    if graph_dir is not None:
        graphs = sorted(path for path in graph_dir.glob("graph_*.json") if path.is_file())
        selected_session = _session_id_from_path(graph_dir)
        selected_root = _session_root_from_path(graph_dir)
    elif session_dir is not None:
        graphs = _find_graphs(session_dir)
        selected_session = _session_id_from_path(session_dir)
        selected_root = session_dir
    else:
        sessions = _find_session_dirs(data_dir)
        if not sessions:
            return "no_session", None, []
        selected = sessions[0]
        graphs = _find_graphs(selected)
        selected_session = _session_id_from_path(selected)
        selected_root = selected

    if limit is not None:
        graphs = graphs[:limit]
    return selected_session, selected_root, graphs


def _default_output_path(output_dir: Path, session_id: str) -> Path:
    return output_dir / session_id / "fusion_outputs.jsonl"


def _build_embeddings(h_notif: torch.Tensor, h_switching: torch.Tensor) -> list[torch.Tensor]:
    return [
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(1, 64, dtype=torch.float32),
        h_notif,
        h_switching,
    ]


def _find_notif_features_csv(session_root: Optional[Path]) -> Optional[Path]:
    if session_root is None:
        return None
    candidates = [
        session_root / "features" / "system" / "features_system_120s.csv",
        session_root / "features_system_120s.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(session_root.glob("**/features_system_120s.csv"))
    return matches[0] if matches else None


def _load_csv_by_window(path: Optional[Path]) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["window_id"]: row
            for row in csv.DictReader(handle)
            if row.get("window_id")
        }


def _window_id_from_graph(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    window = payload.get("window") if isinstance(payload, dict) else None
    if isinstance(window, dict) and window.get("window_id") is not None:
        return str(window["window_id"])
    return None


def _float_from_row(row: Optional[dict[str, str]], key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _normalize_notif_features(features: np.ndarray) -> np.ndarray:
    min_ = np.asarray(NOTIF_SCALER_PARAMS["min_"], dtype=np.float32)
    max_ = np.asarray(NOTIF_SCALER_PARAMS["max_"], dtype=np.float32)
    denom = np.where(np.abs(max_ - min_) < 1e-8, 1.0, max_ - min_)
    normalized = (features - min_) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _load_notif_onnx_session(model_path: Path):
    if not model_path.is_file():
        return None, "missing_model"
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime_missing"
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=session_options), "onnx"


def _notif_embedding(onnx_session, features: np.ndarray) -> tuple[np.ndarray, str]:
    if onnx_session is None:
        return np.zeros(16, dtype=np.float32), "zero_fallback"
    normalized = _normalize_notif_features(features).reshape(1, 5)
    embedding = onnx_session.run(None, {"features": normalized})[0]
    return np.asarray(embedding, dtype=np.float32).reshape(16), "onnx"


def _notif_payload_for_window(
    *,
    window_id: Optional[str],
    row: Optional[dict[str, str]],
    onnx_session,
    onnx_status: str,
    features_path: Optional[Path],
) -> dict[str, Any]:
    arrival_rate = _float_from_row(row, "notification_rate")
    interruption_density = _float_from_row(row, "interruption_density")
    response_latency_mean = _float_from_row(row, "response_latency_mean")

    latency_norm = float(np.clip(response_latency_mean / 5000.0, 0.0, 1.0))
    burstiness = float(np.clip(interruption_density, 0.0, 1.0))
    source_entropy = 0.0
    disruption_score = float(np.clip(interruption_density + latency_norm, 0.0, 1.0))
    time_since_last = 0.0 if arrival_rate > 0 else 1.0

    raw_features = np.asarray(
        [
            arrival_rate,
            burstiness,
            source_entropy,
            disruption_score,
            time_since_last,
        ],
        dtype=np.float32,
    )
    npi = float(
        np.clip(
            raw_features[0] * 0.30
            + raw_features[1] * 0.20
            + raw_features[2] * 0.15
            + raw_features[3] * 0.25
            + raw_features[4] * 0.10,
            0.0,
            1.0,
        )
    )
    embedding, embedding_source = _notif_embedding(onnx_session, raw_features)
    return {
        "embedding": embedding,
        "npi": npi,
        "burstiness": burstiness,
        "disruption_score": disruption_score,
        "metadata": {
            "module": "notifications",
            "window_id": window_id,
            "features_source": str(features_path) if features_path else None,
            "features_available": row is not None,
            "embedding_source": embedding_source,
            "onnx_status": onnx_status,
            "raw_features": raw_features,
        },
    }


def _record_for_window(
    *,
    graph_path: Path,
    index: int,
    effective_window_id: Optional[str],
    switching_payload: dict[str, Any],
    notif_payload: dict[str, Any],
    h_notif: torch.Tensor,
    notif_freshness: float,
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
        "window_id": effective_window_id,
        "switching": {
            "embedding_shape": tuple(embedding.shape),
            "embedding_dtype": str(embedding.dtype),
            "embedding_l2_norm": float(np.linalg.norm(embedding)),
            "tensor_shape": tuple(h_switching.shape),
            "freshness": float(switching_freshness),
            "metadata": metadata,
            "debug_state": fusion_output.get("debug", {}).get("switching", {}),
        },
        "notif": {
            "embedding_shape": tuple(notif_payload["embedding"].shape),
            "embedding_dtype": str(notif_payload["embedding"].dtype),
            "tensor_shape": tuple(h_notif.shape),
            "freshness": float(notif_freshness),
            "npi": float(notif_payload["npi"]),
            "burstiness": float(notif_payload["burstiness"]),
            "disruption_score": float(notif_payload["disruption_score"]),
            "metadata": notif_payload.get("metadata", {}),
        },
        "fusion": {
            "global": fusion_output["global"][0],
            "per_model": [tensor[0] for tensor in fusion_output["per_model"]],
            "global_shape": tuple(fusion_output["global"].shape),
            "per_model_shapes": [tuple(t.shape) for t in fusion_output["per_model"]],
        },
        "note": "Mouse/keyboard embeddings are zero dummies; notif uses session system features with ONNX embedding when available.",
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
    parser.add_argument(
        "--notif-onnx",
        type=Path,
        default=Path("pre_embedders/notif/models/notif_mlp.onnx"),
        help="Notification ONNX model path. Falls back to zero embedding if onnxruntime is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id, session_root, graphs = _iter_graphs(
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
    notif_encoder = NotifBufferedEncoder()
    notif_features_path = _find_notif_features_csv(session_root)
    notif_rows = _load_csv_by_window(notif_features_path)
    notif_onnx_session, notif_onnx_status = _load_notif_onnx_session(args.notif_onnx)

    fusion = InferrerFusion()
    fusion.eval()

    cold_starts = 0
    notif_windows_matched = 0
    norms: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, graph_path in enumerate(graphs):
            graph_window_id = _window_id_from_graph(graph_path)
            switching_payload = switching_session.get_fusion_input(graph_path)
            h_switching, switching_freshness = switching_encoder.step(switching_payload)
            window_id = switching_payload["metadata"].get("window_id") or graph_window_id
            notif_row = notif_rows.get(str(window_id)) if window_id is not None else None
            notif_windows_matched += int(notif_row is not None)
            notif_payload = _notif_payload_for_window(
                window_id=window_id,
                row=notif_row,
                onnx_session=notif_onnx_session,
                onnx_status=notif_onnx_status,
                features_path=notif_features_path,
            )
            h_notif, notif_freshness = notif_encoder.step(notif_payload)

            with torch.no_grad():
                fusion_output = fusion(_build_embeddings(h_notif, h_switching))
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
                effective_window_id=window_id,
                switching_payload=switching_payload,
                notif_payload=notif_payload,
                h_notif=h_notif,
                notif_freshness=notif_freshness,
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
        "notif_features_path": str(notif_features_path) if notif_features_path else None,
        "notif_windows_matched": notif_windows_matched,
        "notif_onnx_status": notif_onnx_status,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
