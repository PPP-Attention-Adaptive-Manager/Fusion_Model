"""Evaluate LOSO checkpoints for the switching predictive model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from predictive_models.switching.v1_gru_switching import SwitchingGRU
from scripts.switching.train_switching_predictive import (
    INPUT_FLAT_DIM,
    STATE_NAMES,
    load_training_arrays,
    prepare_targets,
    users_from_metadata,
)


def _finite_metric(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict(
    checkpoint_path: Path,
    X: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    users: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path, device)
    test_user = str(checkpoint["test_user"])
    idx = np.where(users == test_user)[0]
    if idx.size == 0:
        raise ValueError(f"No samples found for test_user={test_user!r}")

    mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    X_test = ((X[idx] - mean) / std).astype(np.float32)
    y_factors = factors[idx]
    y_states = states[idx]

    model = SwitchingGRU(
        input_flat_dim=int(checkpoint.get("input_flat_dim", INPUT_FLAT_DIM)),
        d_proj=int(checkpoint.get("d_proj", 256)),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    factor_preds: List[np.ndarray] = []
    state_preds: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], batch_size):
            batch = torch.from_numpy(X_test[start : start + batch_size]).to(device)
            model.reset_microstate()
            output = model(batch)
            factor_preds.append(output[:, :5].detach().cpu().numpy())
            logits = output[:, 5:10].detach().cpu().numpy()
            logits_all.append(logits)
            state_preds.append(np.argmax(logits, axis=1))

    y_factor_pred = np.vstack(factor_preds)
    y_state_pred = np.concatenate(state_preds)
    logits_np = np.vstack(logits_all)
    factor_error = y_factor_pred - y_factors
    cm = confusion_matrix(y_states, y_state_pred, labels=list(range(5)))

    return {
        "test_user": test_user,
        "num_samples": int(idx.size),
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        "state_accuracy": _finite_metric(accuracy_score(y_states, y_state_pred)),
        "macro_f1": _finite_metric(f1_score(y_states, y_state_pred, labels=list(range(5)), average="macro", zero_division=0)),
        "mcc": _finite_metric(matthews_corrcoef(y_states, y_state_pred)),
        "cohen_kappa": _finite_metric(cohen_kappa_score(y_states, y_state_pred, labels=list(range(5)))),
        "per_class_f1": f1_score(y_states, y_state_pred, labels=list(range(5)), average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist(),
        "y_true": y_states.tolist(),
        "y_pred": y_state_pred.tolist(),
        "factor_targets": y_factors.tolist(),
        "factor_preds": y_factor_pred.tolist(),
        "logits": logits_np.tolist(),
        "checkpoint": str(checkpoint_path),
    }


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "test_user",
        "num_samples",
        "factor_mae",
        "factor_rmse",
        "state_accuracy",
        "macro_f1",
        "mcc",
        "cohen_kappa",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Switching predictive LOSO confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(STATE_NAMES)
    for row in range(5):
        for col in range(5):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate switching predictive LOSO checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_for_training"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/switching_predictive"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    X, labels, metadata, resolved_dir = load_training_arrays(args.data_dir)
    factors, states = prepare_targets(labels)
    users = users_from_metadata(metadata)

    checkpoint_paths = sorted(args.checkpoint_dir.glob("fold_*/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold_*/best.pt checkpoints found under {args.checkpoint_dir}")

    fold_results = [
        _predict(path, X, factors, states, users, device=device, batch_size=args.batch_size)
        for path in checkpoint_paths
    ]

    all_true = np.asarray([label for row in fold_results for label in row["y_true"]], dtype=np.int64)
    all_pred = np.asarray([label for row in fold_results for label in row["y_pred"]], dtype=np.int64)
    all_factor_targets = np.asarray([row for fold in fold_results for row in fold["factor_targets"]], dtype=np.float32)
    all_factor_preds = np.asarray([row for fold in fold_results for row in fold["factor_preds"]], dtype=np.float32)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(5)))
    factor_error = all_factor_preds - all_factor_targets

    summary = {
        "data_dir": str(resolved_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "num_folds": len(fold_results),
        "num_samples": int(all_true.size),
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        "state_accuracy": _finite_metric(accuracy_score(all_true, all_pred)),
        "macro_f1": _finite_metric(f1_score(all_true, all_pred, labels=list(range(5)), average="macro", zero_division=0)),
        "mcc": _finite_metric(matthews_corrcoef(all_true, all_pred)),
        "cohen_kappa": _finite_metric(cohen_kappa_score(all_true, all_pred, labels=list(range(5)))),
        "per_class_f1": f1_score(all_true, all_pred, labels=list(range(5)), average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist(),
        "folds": fold_results,
    }

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "loso_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(args.checkpoint_dir / "loso_summary.csv", fold_results)
    _plot_confusion_matrix(args.checkpoint_dir / "confusion_matrix.png", cm)
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
