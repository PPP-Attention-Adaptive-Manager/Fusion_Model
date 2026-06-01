"""Runtime smoke test for the full fusion dataset and forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fusion_dataset import FusionWindowDataset, fusion_collate_fn
from fusion_model import InferrerFusion


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise AssertionError(f"{name} expected shape {expected}, got {tuple(tensor.shape)}")


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test FusionWindowDataset and InferrerFusion.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit-sessions", type=int, default=10)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_train/dataset_smoke_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FusionWindowDataset(
        data_dir=args.data_dir,
        limit_sessions=args.limit_sessions,
        limit_samples=args.limit_samples,
        switching_device=args.device,
    )
    if len(dataset) == 0:
        raise RuntimeError("FusionWindowDataset produced 0 samples.")

    model = InferrerFusion()
    model.eval()

    checked = 0
    cold_starts = 0
    switching_norms: list[float] = []
    state_labels: dict[str, int] = {}
    first_records: list[dict[str, Any]] = []

    for sample in dataset:
        batch = fusion_collate_fn([sample])
        _assert_shape("mouse", batch["mouse"], (1, 64))
        _assert_shape("keyboard", batch["keyboard"], (1, 64))
        _assert_shape("notif", batch["notif"], (1, 32))
        _assert_shape("switching", batch["switching"], (1, 64))
        for key in ["mouse", "keyboard", "notif", "switching", "factors"]:
            _assert_finite(key, batch[key])

        with torch.no_grad():
            output = model([batch["mouse"], batch["keyboard"], batch["notif"], batch["switching"]])
        _assert_shape("fusion.global", output["global"], (1, 11))
        if len(output["per_model"]) != 4:
            raise AssertionError(f"per_model expected 4 tensors, got {len(output['per_model'])}")
        for idx, tensor in enumerate(output["per_model"]):
            _assert_shape(f"per_model[{idx}]", tensor, (1, 12))
            _assert_finite(f"per_model[{idx}]", tensor)
        _assert_finite("fusion.global", output["global"])

        metadata = sample["metadata"]
        cold_starts += int(bool(metadata.get("switching_cold_start")))
        switching_norms.append(float(torch.linalg.vector_norm(sample["switching"]).item()))
        label_name = str(int(sample["state_label"].item()))
        state_labels[label_name] = state_labels.get(label_name, 0) + 1
        if len(first_records) < 5:
            first_records.append(
                {
                    "session_id": metadata.get("session_id"),
                    "window_id": metadata.get("window_id"),
                    "graph_id": metadata.get("graph_id"),
                    "switching_shape": list(sample["switching"].shape),
                    "switching_norm": switching_norms[-1],
                    "notif_shape": list(sample["notif"].shape),
                    "state_label": int(sample["state_label"].item()),
                    "global_shape": list(output["global"].shape),
                    "per_model_shapes": [list(t.shape) for t in output["per_model"]],
                }
            )
        checked += 1

    summary = {
        "status": "ok",
        "checked_samples": checked,
        "dataset": dataset.summary(),
        "cold_starts": cold_starts,
        "switching_norm_min": min(switching_norms),
        "switching_norm_max": max(switching_norms),
        "state_label_counts": state_labels,
        "first_records": first_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
