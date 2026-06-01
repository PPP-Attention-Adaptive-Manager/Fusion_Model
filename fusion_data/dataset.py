"""Dataset builder for fusion-model training and smoke tests.

This module keeps the switching/GNN contract unchanged:

    graph JSON -> exported GraphSAGE encoder -> L2-normalized 64D embedding
    -> SwitchingBufferedEncoder identity path -> Tensor(64)

Mouse and keyboard support real 64D embedding columns when present. Until those
embedders are available in the dataset, they fall back to zero vectors with a
warning and metadata flag.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from pre_embedders.switching import load_model as load_switching_model
from TCN_encoders.notif.encoder import NotifBufferedEncoder
from TCN_encoders.switching.encoder import SwitchingBufferedEncoder


FACTOR_KEYS = [
    "mental_demand",
    "temporal_demand",
    "effort",
    "frustration",
    "arousal",
]

STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]

NOTIF_SCALER_PARAMS = {
    "min_": [0.0, 0.0, 0.0, 0.0, 0.0],
    "max_": [3.0, 2.509182763787976e-07, 1.0, 0.021615064589633373, 1.0],
}


@dataclass(frozen=True)
class LabelThresholds:
    high: float = 0.66
    low: float = 0.33
    flow_performance_min: float = 0.66
    flow_frustration_max: float = 0.33
    flow_effort_min: float = 0.35
    flow_effort_max: float = 0.75
    fragmentation_high: float = 0.60


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_label_value(value: Any, default: float = 0.5) -> float:
    """Normalize common 0-100 labels to [0, 1] and clip."""
    x = _safe_float(value, default)
    if abs(x) > 1.0:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def normalize_factor_labels(row: Dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [normalize_label_value(row.get(key), default=0.5) for key in FACTOR_KEYS],
        dtype=torch.float32,
    )


def _first_present(row: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return normalize_label_value(row[key], default=default)
    return default


def derive_state_label(
    row: Dict[str, Any],
    thresholds: LabelThresholds | None = None,
) -> int:
    """Derive a coarse cognitive state from NASA-TLX/stress/arousal values.

    Returns:
        0 Flow, 1 Neutral, 2 Bored, 3 Distracted, 4 Overloaded.
    """

    t = thresholds or LabelThresholds()
    mental = normalize_label_value(row.get("mental_demand"), default=0.5)
    temporal = normalize_label_value(row.get("temporal_demand"), default=0.5)
    effort = normalize_label_value(row.get("effort"), default=0.5)
    frustration = normalize_label_value(row.get("frustration"), default=0.5)
    arousal = normalize_label_value(row.get("arousal"), default=0.5)
    performance = _first_present(row, ["performance", "performance_score"], default=0.5)
    fragmentation = _first_present(
        row,
        [
            "fragmentation_proxy",
            "fragmentation",
            "interruption_density",
            "switch_rate",
            "multitask_score",
        ],
        default=0.0,
    )

    if mental >= t.high and frustration >= t.high:
        return 4
    if (
        performance >= t.flow_performance_min
        and frustration <= t.flow_frustration_max
        and t.flow_effort_min <= effort <= t.flow_effort_max
    ):
        return 0
    if mental <= t.low and arousal <= t.low:
        return 2
    if temporal >= t.high or fragmentation >= t.fragmentation_high:
        return 3
    return 1


def _read_csv_rows(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_by_window(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    rows = _read_csv_rows(path)
    return {row["window_id"]: row for row in rows if row.get("window_id")}


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _find_session_dirs(data_dir: Path) -> List[Path]:
    sessions = sorted(path for path in data_dir.glob("session_*") if path.is_dir())
    return sessions if sessions else ([data_dir] if data_dir.exists() else [])


def _session_id(session_dir: Path) -> str:
    return session_dir.name if session_dir.name.startswith("session_") else session_dir.stem


def _find_graphs(session_dir: Path) -> List[Path]:
    patterns = [
        "data_graph/data_graph_120s/graph_*.json",
        "**/data_graph_120s/graph_*.json",
        "**/switching/data_graph_120s/graph_*.json",
    ]
    seen: set[Path] = set()
    graphs: List[Path] = []
    for pattern in patterns:
        for path in sorted(session_dir.glob(pattern)):
            if path not in seen and path.is_file() and path.stat().st_size > 0:
                seen.add(path)
                graphs.append(path)
    return sorted(graphs)


def _load_graph_window(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    window = payload.get("window", {}) if isinstance(payload, dict) else {}
    return window if isinstance(window, dict) else {}


def _find_label_path(session_dir: Path) -> Optional[Path]:
    return _find_first_existing(
        [
            session_dir / "labels.csv",
            session_dir / "labels" / "nasa_tlx.csv",
            session_dir / "raw" / "labels.csv",
        ]
    )


def _label_for_window(labels: List[Dict[str, str]], window_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not labels:
        return None
    if window_id is not None:
        for row in labels:
            if row.get("window_id") == window_id:
                return row
    return labels[0]


def _parse_vector_cell(value: Any, dim: int) -> Optional[np.ndarray]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        arr = np.asarray(parsed, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (dim,) and np.isfinite(arr).all():
        return arr.astype(np.float32)
    return None


def _extract_embedding_from_row(row: Optional[Dict[str, str]], dim: int, prefixes: Sequence[str]) -> Optional[np.ndarray]:
    if not row:
        return None

    for key in ["embedding", "embeddings", "vector", "features"]:
        vec = _parse_vector_cell(row.get(key), dim)
        if vec is not None:
            return vec

    for prefix in prefixes:
        indexed: List[tuple[int, str]] = []
        for key in row:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :].strip("_")
            if suffix.isdigit():
                indexed.append((int(suffix), key))
        if len(indexed) >= dim:
            values = [_safe_float(row[key]) for _, key in sorted(indexed)[:dim]]
            arr = np.asarray(values, dtype=np.float32)
            if arr.shape == (dim,) and np.isfinite(arr).all():
                return arr
    return None


def _normalize_notif_features(features: np.ndarray) -> np.ndarray:
    min_ = np.asarray(NOTIF_SCALER_PARAMS["min_"], dtype=np.float32)
    max_ = np.asarray(NOTIF_SCALER_PARAMS["max_"], dtype=np.float32)
    denom = np.where(np.abs(max_ - min_) < 1e-8, 1.0, max_ - min_)
    normalized = (features - min_) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def load_notif_onnx_session(model_path: Path):
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


def build_notif_payload(
    row: Optional[Dict[str, str]],
    *,
    window_id: Optional[str],
    onnx_session,
    onnx_status: str,
    features_path: Optional[Path],
) -> Dict[str, Any]:
    arrival_rate = _safe_float(row.get("notification_rate") if row else 0.0)
    interruption_density = _safe_float(row.get("interruption_density") if row else 0.0)
    response_latency_mean = _safe_float(row.get("response_latency_mean") if row else 0.0)

    latency_norm = float(np.clip(response_latency_mean / 5000.0, 0.0, 1.0))
    burstiness = float(np.clip(interruption_density, 0.0, 1.0))
    source_entropy = 0.0
    disruption_score = float(np.clip(interruption_density + latency_norm, 0.0, 1.0))
    time_since_last = 0.0 if arrival_rate > 0 else 1.0
    raw_features = np.asarray(
        [arrival_rate, burstiness, source_entropy, disruption_score, time_since_last],
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
    embedding, source = _notif_embedding(onnx_session, raw_features)
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
            "embedding_source": source,
            "onnx_status": onnx_status,
            "raw_features": raw_features,
        },
    }


class FusionWindowDataset(Dataset):
    """Build one sample per 120-second graph window."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        session_ids: Optional[Sequence[str]] = None,
        limit_sessions: Optional[int] = None,
        limit_samples: Optional[int] = None,
        switching_export_dir: str | Path = "pre_embedders/switching/exports/switching_encoder",
        switching_device: str = "auto",
        notif_onnx_path: str | Path = "pre_embedders/notif/models/notif_mlp.onnx",
        thresholds: LabelThresholds | None = None,
        allow_zero_fallback: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.thresholds = thresholds or LabelThresholds()
        self.allow_zero_fallback = allow_zero_fallback
        self._limit_samples = limit_samples
        self.fallback_counts = {"mouse": 0, "keyboard": 0, "notif": 0}
        self.skipped_sessions: List[str] = []

        selected = _find_session_dirs(self.data_dir)
        if session_ids is not None:
            wanted = set(session_ids)
            selected = [path for path in selected if _session_id(path) in wanted]
        if limit_sessions is not None:
            selected = selected[:limit_sessions]

        self.switching_session = load_switching_model(
            export_dir=switching_export_dir,
            device=switching_device,
        )
        self.notif_onnx_session, self.notif_onnx_status = load_notif_onnx_session(Path(notif_onnx_path))
        self.samples: List[Dict[str, Any]] = []
        for session_dir in selected:
            self._append_session(session_dir)
            if self._limit_samples is not None and len(self.samples) >= self._limit_samples:
                break

        if self.fallback_counts["mouse"] or self.fallback_counts["keyboard"]:
            warnings.warn(
                "Mouse/keyboard 64D embeddings were not found for some windows; "
                f"zero fallback counts={self.fallback_counts}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _append_session(self, session_dir: Path) -> None:
        session_id = _session_id(session_dir)
        graphs = _find_graphs(session_dir)
        label_path = _find_label_path(session_dir)
        labels = _read_csv_rows(label_path)
        if not graphs or not labels:
            self.skipped_sessions.append(session_id)
            return

        mouse_path = _find_first_existing(
            [
                session_dir / "embeddings" / "mouse_120s.csv",
                session_dir / "features" / "mouse" / "embeddings_mouse_120s.csv",
                session_dir / "features" / "mouse" / "features_mouse_120s.csv",
            ]
        )
        keyboard_path = _find_first_existing(
            [
                session_dir / "embeddings" / "keyboard_120s.csv",
                session_dir / "features" / "keyboard" / "embeddings_keyboard_120s.csv",
                session_dir / "features" / "keyboard" / "features_keyboard_120s.csv",
            ]
        )
        notif_path = _find_first_existing(
            [
                session_dir / "features" / "system" / "features_system_120s.csv",
                session_dir / "features_system_120s.csv",
            ]
        )
        behavior_path = _find_first_existing(
            [
                session_dir / "features" / "behavior" / "features_behavior_120s.csv",
                session_dir / "features_behavior_120s.csv",
            ]
        )

        mouse_rows = _csv_by_window(mouse_path)
        keyboard_rows = _csv_by_window(keyboard_path)
        notif_rows = _csv_by_window(notif_path)
        behavior_rows = _csv_by_window(behavior_path)
        switching_encoder = SwitchingBufferedEncoder(mode="identity")
        notif_encoder = NotifBufferedEncoder()

        for graph_path in graphs:
            if self._limit_samples is not None and len(self.samples) >= self._limit_samples:
                break
            window = _load_graph_window(graph_path)
            window_id = str(window.get("window_id")) if window.get("window_id") is not None else None
            label_row = _label_for_window(labels, window_id)
            if label_row is None:
                continue

            mouse_vec = _extract_embedding_from_row(
                mouse_rows.get(str(window_id)),
                64,
                prefixes=["mouse_emb_", "mouse_embedding_", "emb_", "embedding_"],
            )
            keyboard_vec = _extract_embedding_from_row(
                keyboard_rows.get(str(window_id)),
                64,
                prefixes=["keyboard_emb_", "keyboard_embedding_", "key_emb_", "emb_", "embedding_"],
            )
            if mouse_vec is None:
                if not self.allow_zero_fallback:
                    continue
                mouse_vec = np.zeros(64, dtype=np.float32)
                self.fallback_counts["mouse"] += 1
            if keyboard_vec is None:
                if not self.allow_zero_fallback:
                    continue
                keyboard_vec = np.zeros(64, dtype=np.float32)
                self.fallback_counts["keyboard"] += 1

            switching_payload = self.switching_session.get_fusion_input(graph_path)
            h_switching, switching_freshness = switching_encoder.step(switching_payload)
            effective_window_id = (
                switching_payload["metadata"].get("window_id")
                or window_id
            )

            notif_row = notif_rows.get(str(effective_window_id)) if effective_window_id is not None else None
            if notif_row is None:
                self.fallback_counts["notif"] += 1
            notif_payload = build_notif_payload(
                notif_row,
                window_id=str(effective_window_id) if effective_window_id is not None else None,
                onnx_session=self.notif_onnx_session,
                onnx_status=self.notif_onnx_status,
                features_path=notif_path,
            )
            h_notif, notif_freshness = notif_encoder.step(notif_payload)

            behavior_row = behavior_rows.get(str(effective_window_id)) if effective_window_id is not None else None
            label_context = dict(label_row)
            if behavior_row:
                for key in ["fragmentation", "fragmentation_proxy", "switch_rate", "multitask_score"]:
                    if key in behavior_row:
                        label_context.setdefault(key, behavior_row[key])
            if notif_row:
                label_context.setdefault("interruption_density", notif_row.get("interruption_density", "0"))

            metadata = {
                "session_id": session_id,
                "window_id": str(effective_window_id) if effective_window_id is not None else None,
                "window_start": _safe_float(window.get("window_start")),
                "window_end": _safe_float(window.get("window_end")),
                "graph_path": str(graph_path),
                "graph_id": switching_payload["metadata"].get("graph_id"),
                "label_source": str(label_path) if label_path else None,
                "mouse_source": str(mouse_path) if mouse_path else None,
                "keyboard_source": str(keyboard_path) if keyboard_path else None,
                "notif_source": str(notif_path) if notif_path else None,
                "notif_embedding_source": notif_payload["metadata"].get("embedding_source"),
                "notif_onnx_status": self.notif_onnx_status,
                "switching_cold_start": bool(switching_payload["metadata"].get("cold_start", False)),
                "switching_freshness": float(switching_freshness),
                "notif_freshness": float(notif_freshness),
                "mouse_zero_fallback": bool(np.count_nonzero(mouse_vec) == 0),
                "keyboard_zero_fallback": bool(np.count_nonzero(keyboard_vec) == 0),
            }

            self.samples.append(
                {
                    "mouse": torch.from_numpy(mouse_vec.astype(np.float32)),
                    "keyboard": torch.from_numpy(keyboard_vec.astype(np.float32)),
                    "notif": h_notif.detach().cpu().to(torch.float32).squeeze(0),
                    "switching": h_switching.detach().cpu().to(torch.float32).squeeze(0),
                    "factors": normalize_factor_labels(label_row),
                    "state_label": torch.tensor(
                        derive_state_label(label_context, self.thresholds),
                        dtype=torch.long,
                    ),
                    "metadata": metadata,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def summary(self) -> Dict[str, Any]:
        sessions = sorted({sample["metadata"]["session_id"] for sample in self.samples})
        return {
            "num_samples": len(self.samples),
            "num_sessions": len(sessions),
            "sessions": sessions,
            "fallback_counts": dict(self.fallback_counts),
            "skipped_sessions": list(self.skipped_sessions),
            "notif_onnx_status": self.notif_onnx_status,
            "thresholds": asdict(self.thresholds),
        }


def fusion_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mouse": torch.stack([item["mouse"] for item in batch], dim=0),
        "keyboard": torch.stack([item["keyboard"] for item in batch], dim=0),
        "notif": torch.stack([item["notif"] for item in batch], dim=0),
        "switching": torch.stack([item["switching"] for item in batch], dim=0),
        "factors": torch.stack([item["factors"] for item in batch], dim=0),
        "state_label": torch.stack([item["state_label"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


class SessionBatchSampler(Sampler[List[int]]):
    """Batch sampler that never mixes sessions inside one batch."""

    def __init__(
        self,
        dataset: FusionWindowDataset,
        batch_size: int,
        *,
        shuffle_sessions: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_sessions = shuffle_sessions
        self.seed = seed

        self._by_session: Dict[str, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            session_id = str(sample["metadata"]["session_id"])
            self._by_session.setdefault(session_id, []).append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        sessions = sorted(self._by_session)
        if self.shuffle_sessions:
            rng = np.random.default_rng(self.seed)
            sessions = list(rng.permutation(sessions))
        for session_id in sessions:
            indices = self._by_session[session_id]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self._by_session.values()
        )
