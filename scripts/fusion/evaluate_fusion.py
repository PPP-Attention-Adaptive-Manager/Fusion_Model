"""Evaluate a trained fusion checkpoint on a session-level split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from fusion_data.dataset import (
    FusionWindowDataset,
    SessionBatchSampler,
    fusion_collate_fn,
)
from fusion_model import InferrerFusion
from fusion_data.splits import load_splits


def _scores_to_probs(scores: torch.Tensor) -> torch.Tensor:
    row_sums = scores.sum(dim=-1)
    if torch.all(scores >= 0) and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        return scores
    return torch.softmax(scores, dim=-1)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def _macro_f1(cm: np.ndarray) -> float:
    scores = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _mcc(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    trace = np.trace(cm)
    numerator = trace * total - np.dot(true_counts, pred_counts)
    denom_a = total**2 - np.dot(pred_counts, pred_counts)
    denom_b = total**2 - np.dot(true_counts, true_counts)
    denom = np.sqrt(max(denom_a * denom_b, 0.0))
    return float(numerator / denom) if denom > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate InferrerFusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/fusion_train/checkpoints/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--splits", type=Path, default=Path("outputs/fusion_train/splits.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_train/eval_metrics.json"))
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    splits = load_splits(args.splits)
    session_ids = splits[args.split]

    dataset = FusionWindowDataset(
        args.data_dir,
        session_ids=session_ids,
        limit_samples=args.limit_samples,
        switching_device="cpu",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset for split {args.split!r} is empty.")
    sampler = SessionBatchSampler(dataset, batch_size=args.batch_size, shuffle_sessions=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=fusion_collate_fn)

    model = InferrerFusion().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    factor_preds = []
    factor_targets = []
    state_preds = []
    state_targets = []
    entropies = []
    margins = []

    with torch.no_grad():
        for batch in loader:
            model.reset_subject()
            for key in ["mouse", "keyboard", "notif", "switching", "factors", "state_label"]:
                batch[key] = batch[key].to(device)
            output = model([batch["mouse"], batch["keyboard"], batch["notif"], batch["switching"]])
            factors = output["global"][:, :5]
            scores = output["global"][:, 5:10]
            probs = _scores_to_probs(scores)
            top2 = torch.topk(probs, k=2, dim=-1).values

            factor_preds.append(factors.cpu().numpy())
            factor_targets.append(batch["factors"].cpu().numpy())
            state_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
            state_targets.append(batch["state_label"].cpu().numpy())
            entropies.append((-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).cpu().numpy())
            margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())

    y_factor = np.vstack(factor_targets)
    y_factor_pred = np.vstack(factor_preds)
    y_state = np.concatenate(state_targets)
    y_state_pred = np.concatenate(state_preds)
    cm = _confusion_matrix(y_state, y_state_pred, n_classes=5)

    factor_error = y_factor_pred - y_factor
    metrics: Dict[str, Any] = {
        "split": args.split,
        "num_samples": int(len(dataset)),
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        "state_accuracy": float(np.mean(y_state_pred == y_state)),
        "macro_f1": _macro_f1(cm),
        "mcc": _mcc(cm),
        "confusion_matrix": cm.tolist(),
        "entropy_mean": float(np.mean(np.concatenate(entropies))),
        "margin_mean": float(np.mean(np.concatenate(margins))),
        "checkpoint": str(args.checkpoint),
        "dataset": dataset.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
