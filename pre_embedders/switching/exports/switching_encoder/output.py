"""Encoder-only switching graph output API for fusion.

This module intentionally loads only GraphSAGE encoder weights. The GAE decoder
was used during training for reconstruction and is not used here.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import to_undirected


MODULE_NAME = "switching"
ENCODER_NAME = "GraphSAGE_GAE"
NUMERIC_EPS = 1e-12


@dataclass
class FeatureSpec:
    node_numeric_keys: Tuple[str, ...]
    node_categorical_keys: Tuple[str, ...]
    edge_numeric_keys: Tuple[str, ...] = ()
    edge_categorical_keys: Tuple[str, ...] = ()
    node_numeric_mean: Dict[str, float] | None = None
    node_numeric_std: Dict[str, float] | None = None
    edge_numeric_mean: Dict[str, float] | None = None
    edge_numeric_std: Dict[str, float] | None = None
    node_categorical_hash_size: int = 16
    edge_categorical_hash_size: int = 8
    normalize_numeric: bool = True

    @property
    def node_dim(self) -> int:
        return len(self.node_numeric_keys) + len(self.node_categorical_keys) * self.node_categorical_hash_size

    @property
    def edge_dim(self) -> int:
        return 1 + len(self.edge_numeric_keys) + len(self.edge_categorical_keys) * self.edge_categorical_hash_size

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FeatureSpec":
        return cls(
            node_numeric_keys=tuple(payload.get("node_numeric_keys", ())),
            node_categorical_keys=tuple(payload.get("node_categorical_keys", ())),
            edge_numeric_keys=tuple(payload.get("edge_numeric_keys", ())),
            edge_categorical_keys=tuple(payload.get("edge_categorical_keys", ())),
            node_numeric_mean={str(k): float(v) for k, v in payload.get("node_numeric_mean", {}).items()},
            node_numeric_std={str(k): float(v) for k, v in payload.get("node_numeric_std", {}).items()},
            edge_numeric_mean={str(k): float(v) for k, v in payload.get("edge_numeric_mean", {}).items()},
            edge_numeric_std={str(k): float(v) for k, v in payload.get("edge_numeric_std", {}).items()},
            node_categorical_hash_size=int(payload.get("node_categorical_hash_size", 16)),
            edge_categorical_hash_size=int(payload.get("edge_categorical_hash_size", 8)),
            normalize_numeric=bool(payload.get("normalize_numeric", True)),
        )


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _float_or_none(value: Any) -> Optional[float]:
    if _is_missing_value(value):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if np.isfinite(numeric_value) else None


def _coerce_finite_float(value: Any, *, field: str) -> float:
    numeric_value = _float_or_none(value)
    if numeric_value is None:
        raise ValueError(f"Non-numeric or non-finite value for {field}: {value!r}")
    return numeric_value


def _stable_hash_bucket(key: str, value: Any, hash_size: int) -> int:
    encoded = f"{key}={value}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % hash_size


def _encode_numeric(
    features: Dict[str, Any],
    key: str,
    *,
    means: Dict[str, float],
    stds: Dict[str, float],
    normalize: bool,
) -> float:
    observed_value = _float_or_none(features.get(key))
    if observed_value is None:
        value = means.get(key, 0.0) if normalize else 0.0
    else:
        value = observed_value
    if not np.isfinite(value):
        raise ValueError(f"Non-finite feature value for {key}: {value!r}")
    if not normalize:
        return float(value)
    return float((value - means.get(key, 0.0)) / stds.get(key, 1.0))


def _encode_categorical(features: Dict[str, Any], key: str, *, hash_size: int) -> List[float]:
    row = [0.0] * hash_size
    value = features.get(key)
    if _is_missing_value(value):
        return row
    row[_stable_hash_bucket(key, value, hash_size)] = 1.0
    return row


def _edge_feature_map(edge: Dict[str, Any]) -> Dict[str, Any]:
    features = edge.get("features", {})
    if features is None:
        return {}
    if not isinstance(features, dict):
        raise ValueError("Edge features must be a JSON object.")
    return features


def _graph_id_from_payload(payload: Dict[str, Any], graph_path: Optional[Path]) -> Optional[str]:
    graph_id = payload.get("graph_id")
    if graph_id:
        return str(graph_id)
    return graph_path.stem if graph_path is not None else None


def _session_id_from_path(graph_path: Optional[Path]) -> Optional[str]:
    if graph_path is None:
        return None
    for part in graph_path.parts:
        if part.startswith("session_"):
            return part
    return None


def _window_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    window = payload.get("window", {})
    if isinstance(window, dict) and window.get("window_id") is not None:
        return str(window.get("window_id"))
    return None


def graph_payload_to_data(
    payload: Dict[str, Any],
    *,
    feature_spec: FeatureSpec,
    make_undirected: bool = False,
) -> Data:
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Graph has no nodes.")
    if not isinstance(edges, list):
        raise ValueError("Graph edges must be a list.")

    node_ids: List[str] = []
    x_rows: List[List[float]] = []
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node:
            raise ValueError("Every node must be an object with an id.")
        node_id = str(node["id"])
        if node_id in node_ids:
            raise ValueError(f"Duplicate node id: {node_id}")
        features = node.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"Node features must be a JSON object for {node_id}.")
        row = [
            _encode_numeric(
                features,
                key,
                means=feature_spec.node_numeric_mean or {},
                stds=feature_spec.node_numeric_std or {},
                normalize=feature_spec.normalize_numeric,
            )
            for key in feature_spec.node_numeric_keys
        ]
        for key in feature_spec.node_categorical_keys:
            row.extend(_encode_categorical(features, key, hash_size=feature_spec.node_categorical_hash_size))
        node_ids.append(node_id)
        x_rows.append(row)

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_pairs: List[Tuple[int, int]] = []
    edge_rows: List[List[float]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Every edge must be a JSON object.")
        if "source" not in edge or "target" not in edge:
            raise ValueError("Every edge must include source and target.")
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in node_to_idx or target not in node_to_idx:
            raise ValueError(f"Edge references unknown node: {source!r} -> {target!r}")
        edge_features = _edge_feature_map(edge)
        row = [_coerce_finite_float(edge.get("weight", 1.0), field="edge.weight")]
        row.extend(
            _encode_numeric(
                edge_features,
                key,
                means=feature_spec.edge_numeric_mean or {},
                stds=feature_spec.edge_numeric_std or {},
                normalize=feature_spec.normalize_numeric,
            )
            for key in feature_spec.edge_numeric_keys
        )
        for key in feature_spec.edge_categorical_keys:
            row.extend(_encode_categorical(edge_features, key, hash_size=feature_spec.edge_categorical_hash_size))
        edge_pairs.append((node_to_idx[source], node_to_idx[target]))
        edge_rows.append(row)

    x = torch.tensor(x_rows, dtype=torch.float32)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, feature_spec.edge_dim), dtype=torch.float32)
    if make_undirected and edge_index.numel() > 0:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr, reduce="mean")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))


def aggregate_edge_attr_to_nodes(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    *,
    num_nodes: int,
    edge_attr_dim: int,
) -> torch.Tensor:
    summary_dim = edge_attr_dim * 2 + 2
    if edge_attr_dim <= 0:
        return torch.empty((num_nodes, 0), dtype=torch.float32, device=edge_index.device)
    if edge_attr is None or edge_attr.numel() == 0 or edge_index.numel() == 0:
        return torch.zeros((num_nodes, summary_dim), dtype=torch.float32, device=edge_index.device)

    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)
    if edge_attr.size(1) != edge_attr_dim:
        raise ValueError(f"Expected edge_attr_dim={edge_attr_dim}, got {edge_attr.size(1)}.")

    source = edge_index[0]
    target = edge_index[1]
    edge_attr = edge_attr.to(dtype=torch.float32)
    ones = torch.ones((edge_attr.size(0), 1), dtype=torch.float32, device=edge_attr.device)

    incoming_sum = torch.zeros((num_nodes, edge_attr_dim), dtype=torch.float32, device=edge_attr.device)
    outgoing_sum = torch.zeros((num_nodes, edge_attr_dim), dtype=torch.float32, device=edge_attr.device)
    incoming_count = torch.zeros((num_nodes, 1), dtype=torch.float32, device=edge_attr.device)
    outgoing_count = torch.zeros((num_nodes, 1), dtype=torch.float32, device=edge_attr.device)

    incoming_sum.index_add_(0, target, edge_attr)
    outgoing_sum.index_add_(0, source, edge_attr)
    incoming_count.index_add_(0, target, ones)
    outgoing_count.index_add_(0, source, ones)

    incoming_mean = incoming_sum / incoming_count.clamp_min(1.0)
    outgoing_mean = outgoing_sum / outgoing_count.clamp_min(1.0)
    return torch.cat(
        [incoming_mean, outgoing_mean, torch.log1p(incoming_count), torch.log1p(outgoing_count)],
        dim=-1,
    )


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.0,
        normalize: bool = False,
        edge_attr_dim: int = 0,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim
        self.use_edge_attr = use_edge_attr
        self.edge_summary_dim = edge_attr_dim * 2 + 2 if use_edge_attr and edge_attr_dim > 0 else 0
        self.conv1 = SAGEConv(input_dim + self.edge_summary_dim, hidden_dim, normalize=normalize)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim, normalize=normalize)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_edge_attr and self.edge_attr_dim > 0:
            edge_summary = aggregate_edge_attr_to_nodes(
                edge_index,
                edge_attr,
                num_nodes=x.size(0),
                edge_attr_dim=self.edge_attr_dim,
            )
            x = torch.cat([x, edge_summary.to(dtype=x.dtype)], dim=-1)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return self.conv2(x, edge_index)


def _select_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def _torch_load(path: Path, *, map_location: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_model(export_dir: str | Path = "gnn_project/exports/switching_encoder", device: str = "auto") -> Dict[str, Any]:
    export_path = Path(export_dir)
    selected_device = _select_device(device)
    model_config = _load_json(export_path / "model_config.json")
    feature_spec_payload = _load_json(export_path / "feature_spec.json")
    contract = _load_json(export_path / "embedding_contract.json")
    feature_spec = FeatureSpec.from_dict(feature_spec_payload)

    encoder = GraphSAGEEncoder(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        embedding_dim=int(model_config.get("embedding_dim", 64)),
        dropout=float(model_config.get("dropout", 0.0)),
        normalize=bool(model_config.get("normalize", False)),
        edge_attr_dim=int(model_config.get("edge_attr_dim", 0)),
        use_edge_attr=bool(model_config.get("use_edge_attr", False)),
    )
    state_dict = _torch_load(export_path / "encoder.pt", map_location=selected_device)
    if isinstance(state_dict, dict) and "encoder_state_dict" in state_dict:
        state_dict = state_dict["encoder_state_dict"]
    encoder.load_state_dict(state_dict)
    encoder.to(selected_device)
    encoder.eval()
    return {
        "encoder": encoder,
        "feature_spec": feature_spec,
        "model_config": model_config,
        "contract": contract,
        "device": selected_device,
        "export_dir": export_path,
    }


def _cold_start(
    *,
    reason: str,
    embedding_dim: int,
    graph_id: Optional[str] = None,
    session_id: Optional[str] = None,
    window_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "embedding": np.zeros(embedding_dim, dtype=np.float32),
        "metadata": {
            "module": MODULE_NAME,
            "encoder": ENCODER_NAME,
            "embedding_dim": embedding_dim,
            "window_size_s": 120,
            "normalized": "l2",
            "graph_id": graph_id,
            "session_id": session_id,
            "window_id": window_id,
            "cold_start": True,
            "reason": reason,
        },
    }


def _resolve_payload(graph_json_or_path: Dict[str, Any] | str | Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    if isinstance(graph_json_or_path, dict):
        return graph_json_or_path, None
    graph_path = Path(graph_json_or_path)
    return _load_json(graph_path), graph_path


@torch.no_grad()
def get_output(session: Dict[str, Any], graph_json_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    model_config = session["model_config"]
    embedding_dim = int(model_config.get("embedding_dim", 64))
    try:
        payload, graph_path = _resolve_payload(graph_json_or_path)
        graph_id = _graph_id_from_payload(payload, graph_path)
        session_id = str(payload.get("session_id")) if payload.get("session_id") is not None else _session_id_from_path(graph_path)
        window_id = _window_id_from_payload(payload)
        if not isinstance(payload.get("nodes", []), list) or len(payload.get("nodes", [])) == 0:
            return _cold_start(
                reason="empty_or_invalid_graph",
                embedding_dim=embedding_dim,
                graph_id=graph_id,
                session_id=session_id,
                window_id=window_id,
            )

        data = graph_payload_to_data(
            payload,
            feature_spec=session["feature_spec"],
            make_undirected=bool(model_config.get("make_undirected", False)),
        )
        device = session["device"]
        data = data.to(device)
        z_node = session["encoder"](data.x, data.edge_index, getattr(data, "edge_attr", None))
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        z_graph = global_mean_pool(z_node, batch).squeeze(0)
        embedding = z_graph.detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= NUMERIC_EPS or not np.isfinite(embedding).all():
            return _cold_start(
                reason="empty_or_invalid_graph",
                embedding_dim=embedding_dim,
                graph_id=graph_id,
                session_id=session_id,
                window_id=window_id,
            )
        embedding = (embedding / norm).astype(np.float32)
        return {
            "embedding": embedding,
            "metadata": {
                "module": MODULE_NAME,
                "encoder": ENCODER_NAME,
                "embedding_dim": embedding_dim,
                "window_size_s": int(model_config.get("window_size_s", 120)),
                "normalized": "l2",
                "graph_id": graph_id,
                "session_id": session_id,
                "window_id": window_id,
                "cold_start": False,
                "decoder_used_at_inference": False,
            },
        }
    except Exception:
        return _cold_start(reason="empty_or_invalid_graph", embedding_dim=embedding_dim)
