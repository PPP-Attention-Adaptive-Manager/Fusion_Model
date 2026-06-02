"""Evaluate the dual-task switching predictive model.

Loads the per-fold checkpoints produced by ``train_switching_predictive.py``
(with ``--task dual_task_regression`` or ``dual_task_binary``), re-creates the
exact train/val/test splits from ``splits.json``, rebuilds the load targets with
train-only scaling, runs inference on the held-out windows, and reports
regression + (optionally) binary metrics together with simple baselines.

Outputs (under --checkpoint-dir):
    eval_metrics.json      aggregate + per-fold metrics
    per_user_metrics.csv   per-user MAE / RMSE / Pearson / counts
    predictions.csv        per-window target/prediction/error
    baselines.json         baseline comparison (mean / user-mean / session / prev-window)

Usage:
    python scripts/switching/evaluate_dual_task_switching.py \\
        --data-dir data_training \\
        --checkpoint-dir outputs/switching_dual_task_loso \\
        --dual-task-labels data_training/dual_task_window_labels.csv \\
        --split-mode loso --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from predictive_models.switching.v1_gru_switching import SwitchingGRU
from predictive_models.switching.v1_mlp_switching import SwitchingMLP
from scripts.switching import dual_task_common as dtc


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra.astype(float), rb.astype(float))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if y.size == 0:
        return {k: float("nan") for k in ["mae", "rmse", "pearson_r", "spearman_rho", "r2", "n"]}
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err ** 2))),
        "pearson_r": _pearson(y, p),
        "spearman_rho": _spearman(y, p),
        "r2": _r2(y, p),
        "n": int(y.size),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """accuracy / macro-F1 / MCC / Cohen kappa / confusion matrix for 2 classes."""
    if y_true.size == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "mcc": float("nan"),
                "cohen_kappa": float("nan"), "confusion_matrix": [[0, 0], [0, 0]], "n": 0}
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, q in zip(yt, yp):
        cm[t, q] += 1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(yt.size, 1)

    def f1(tp_, fp_, fn_):
        denom = 2 * tp_ + fp_ + fn_
        return (2 * tp_ / denom) if denom > 0 else 0.0

    f1_pos = f1(tp, fp, fn)
    f1_neg = f1(tn, fn, fp)
    macro_f1 = 0.5 * (f1_pos + f1_neg)

    # MCC
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    # Cohen kappa
    n = yt.size
    po = acc
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n) if n > 0 else 0.0
    kappa = ((po - pe) / (1 - pe)) if (1 - pe) > 1e-12 else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# Model loading / inference
# --------------------------------------------------------------------------- #
def load_model(ckpt: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_name = ckpt.get("model_name", "gru")
    cls = SwitchingGRU if model_name == "gru" else SwitchingMLP
    model = cls(
        input_flat_dim=ckpt.get("input_flat_dim", dtc.INPUT_FLAT_DIM),
        d_proj=ckpt.get("d_proj", 256),
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_states=5,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([], dtype=np.float64)
    model.reset_microstate()
    with torch.no_grad():
        out = model(torch.from_numpy(X.astype(np.float32)).to(device))
        return out[:, 0].cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def compute_baselines(
    labels: dtc.DualTaskLabels,
    target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Regression baselines, all fit on train only and scored on test."""
    out: Dict[str, Dict[str, float]] = {}
    y_test = target[test_idx]
    if y_test.size == 0 or train_idx.size == 0:
        return out

    # 1. Global train mean predictor.
    global_mean = float(np.mean(target[train_idx]))
    out["global_mean"] = regression_metrics(y_test, np.full_like(y_test, global_mean))

    # 2. Train-user mean predictor (fallback to global mean for unseen users).
    user_means: Dict[str, float] = {}
    for u in set(labels.users[train_idx].tolist()):
        m = (labels.users[train_idx] == u)
        user_means[u] = float(np.mean(target[train_idx][m]))
    pred_user = np.asarray(
        [user_means.get(str(labels.users[i]), global_mean) for i in test_idx], dtype=np.float64
    )
    out["train_user_mean"] = regression_metrics(y_test, pred_user)

    # 3. Session baseline predictor (train session mean; fallback global).
    sess_means: Dict[str, float] = {}
    for s in set(labels.sessions[train_idx].tolist()):
        m = (labels.sessions[train_idx] == s)
        sess_means[s] = float(np.mean(target[train_idx][m]))
    pred_sess = np.asarray(
        [sess_means.get(str(labels.sessions[i]), global_mean) for i in test_idx], dtype=np.float64
    )
    out["session_mean"] = regression_metrics(y_test, pred_sess)

    # 4. Previous-window predictor (within same session, ordered by window_idx).
    #    For each test window, predict the target of the previous available
    #    window in that session; fallback to global mean if none.
    avail_target = target.copy()
    prev_pred = []
    for i in test_idx:
        sess = labels.sessions[i]
        wi = labels.window_idx[i]
        cand = np.where(
            (labels.sessions == sess)
            & labels.available
            & (labels.window_idx < wi)
        )[0]
        if cand.size > 0:
            j = cand[np.argmax(labels.window_idx[cand])]
            prev_pred.append(avail_target[j])
        else:
            prev_pred.append(global_mean)
    out["previous_window"] = regression_metrics(y_test, np.asarray(prev_pred, dtype=np.float64))
    return out


def compute_binary_baselines(
    y_train_bin: np.ndarray,
    y_test_bin: np.ndarray,
    *,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Majority-class and stratified-random binary baselines (fit on train)."""
    out: Dict[str, Dict[str, Any]] = {}
    if y_test_bin.size == 0 or y_train_bin.size == 0:
        return out
    # Majority class from train.
    majority = int(round(float(np.mean(y_train_bin)) >= 0.5))
    out["majority_class"] = binary_metrics(y_test_bin, np.full_like(y_test_bin, majority))
    # Stratified random using train positive rate.
    p_pos = float(np.mean(y_train_bin))
    rng = np.random.default_rng(seed)
    strat = (rng.random(y_test_bin.size) < p_pos).astype(int)
    out["stratified_random"] = binary_metrics(y_test_bin, strat)
    return out


def aggregate_baselines(fold_baselines: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Average each baseline's metrics across folds (weighted by n)."""
    keys = set()
    for fb in fold_baselines:
        keys.update(fb.keys())
    agg: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        metrics = [fb[key] for fb in fold_baselines if key in fb]
        agg[key] = _weighted_avg_metrics(metrics)
    return agg


def _weighted_avg_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    total_n = sum(m.get("n", 0) for m in metrics)
    out: Dict[str, float] = {"n": int(total_n)}
    for field in ["mae", "rmse", "pearson_r", "spearman_rho", "r2"]:
        vals = [(m[field], m.get("n", 0)) for m in metrics
                if field in m and not (isinstance(m[field], float) and math.isnan(m[field]))]
        wsum = sum(n for _, n in vals)
        out[field] = float(sum(v * n for v, n in vals) / wsum) if wsum > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-task switching model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_training"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--dual-task-labels", type=Path, default=Path("data_training/dual_task_window_labels.csv")
    )
    parser.add_argument("--split-mode", choices=["loso", "session_within_user"], default="loso")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    X, metadata, _ = dtc.load_switching_slice(args.data_dir)
    labels = dtc.load_dual_task_labels(args.dual_task_labels, metadata)

    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    target_mode = train_config.get("dual_task_target", "relative")
    task = train_config.get("task", "dual_task_regression")
    threshold = float(train_config.get("binary_threshold", 0.5))
    is_binary = task == "dual_task_binary"

    splits = json.loads((args.checkpoint_dir / "splits.json").read_text(encoding="utf-8"))

    all_pred_rows: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    fold_baselines: List[Dict[str, Dict[str, float]]] = []
    binary_baseline_accum: Dict[str, List[Dict[str, Any]]] = {}
    y_all: List[float] = []
    p_all: List[float] = []

    for fold in splits:
        fold_id = str(fold["fold_id"])
        ckpt_path = args.checkpoint_dir / f"fold_{fold_id}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[eval] fold {fold_id}: no checkpoint, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        test_idx = np.asarray(fold["test_idx"], dtype=np.int64)
        if test_idx.size == 0:
            print(f"[eval] fold {fold_id}: empty test set, skipping")
            continue

        # Rebuild target with the SAME train-only scaling used in training.
        target, _, _ = dtc.build_load_targets(labels, target_mode=target_mode, train_idx=train_idx)

        mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
        std = np.asarray(ckpt["x_std"], dtype=np.float32)
        X_test = ((X[test_idx] - mean) / std).astype(np.float32)

        model = load_model(ckpt, device)
        pred_reg = predict(model, X_test, device)
        y_reg = target[test_idx]

        # Regression metrics on the continuous load (always computed).
        m_reg = regression_metrics(y_reg, pred_reg)
        fold_record: Dict[str, Any] = {"fold_id": fold_id, "regression": m_reg}

        if is_binary:
            y_bin = (y_reg > threshold).astype(int)
            p_bin = (pred_reg > threshold).astype(int)
            fold_record["binary"] = binary_metrics(y_bin, p_bin)
            y_train_bin = (target[train_idx] > threshold).astype(int)
            for name, m in compute_binary_baselines(y_train_bin, y_bin).items():
                binary_baseline_accum.setdefault(name, []).append(m)

        fold_metrics.append(fold_record)
        fold_baselines.append(compute_baselines(labels, target, train_idx, test_idx))

        y_all.extend(y_reg.tolist())
        p_all.extend(pred_reg.tolist())

        for k, i in enumerate(test_idx):
            all_pred_rows.append(
                {
                    "row_idx": int(i),
                    "user_id": str(labels.users[i]),
                    "session_id": str(labels.sessions[i]),
                    "window_idx": int(labels.window_idx[i]),
                    "window_start": float(labels.window_start[i]),
                    "window_end": float(labels.window_end[i]),
                    "target": float(y_reg[k]),
                    "prediction": float(pred_reg[k]),
                    "error": float(pred_reg[k] - y_reg[k]),
                    "split": args.split_mode,
                    "fold": fold_id,
                    "dual_task_count": float(labels.dual_task_count[i]),
                    "reaction_time_mean": float(labels.reaction_time_mean[i]),
                    "miss_rate": float(labels.miss_rate[i]),
                    "error_rate": float(labels.error_rate[i]),
                }
            )

    y_arr = np.asarray(y_all, dtype=np.float64)
    p_arr = np.asarray(p_all, dtype=np.float64)
    overall_reg = regression_metrics(y_arr, p_arr)

    # --- per-user metrics ---
    per_user_rows: List[Dict[str, Any]] = []
    users_in_pred = sorted({r["user_id"] for r in all_pred_rows})
    for u in users_in_pred:
        yi = np.asarray([r["target"] for r in all_pred_rows if r["user_id"] == u])
        pi = np.asarray([r["prediction"] for r in all_pred_rows if r["user_id"] == u])
        mu = regression_metrics(yi, pi)
        total_u = int(np.sum(labels.users == u))
        avail_u = int(np.sum((labels.users == u) & labels.available))
        per_user_rows.append(
            {
                "user_id": u,
                "n_eval_windows": int(yi.size),
                "n_available_windows": avail_u,
                "n_missing_windows": total_u - avail_u,
                "mae": mu["mae"],
                "rmse": mu["rmse"],
                "pearson_r": mu["pearson_r"],
            }
        )

    baselines_agg: Dict[str, Any] = {"regression": aggregate_baselines(fold_baselines)}
    if is_binary and binary_baseline_accum:
        # Average accuracy/macro_f1/mcc/kappa across folds (weighted by n).
        bin_agg: Dict[str, Dict[str, float]] = {}
        for name, metrics in binary_baseline_accum.items():
            total = sum(m.get("n", 0) for m in metrics)
            agg = {"n": int(total)}
            for f in ["accuracy", "macro_f1", "mcc", "cohen_kappa"]:
                vals = [(m[f], m.get("n", 0)) for m in metrics
                        if f in m and not (isinstance(m[f], float) and math.isnan(m[f]))]
                wsum = sum(nn for _, nn in vals)
                agg[f] = float(sum(v * nn for v, nn in vals) / wsum) if wsum > 0 else float("nan")
            bin_agg[name] = agg
        baselines_agg["binary"] = bin_agg

    # Does the model beat the simple global-mean baseline?
    beats = None
    reg_baselines = baselines_agg.get("regression", {})
    if "global_mean" in reg_baselines and not math.isnan(overall_reg["mae"]):
        beats = bool(overall_reg["mae"] < reg_baselines["global_mean"].get("mae", float("inf")))

    eval_metrics: Dict[str, Any] = {
        "split_mode": args.split_mode,
        "task": task,
        "dual_task_target": target_mode,
        "num_folds_evaluated": len(fold_metrics),
        "overall_regression": overall_reg,
        "per_fold": fold_metrics,
        "n_eval_windows": int(y_arr.size),
        "model_beats_global_mean_mae": beats,
    }
    if is_binary:
        y_bin_all = (y_arr > threshold).astype(int)
        p_bin_all = (p_arr > threshold).astype(int)
        eval_metrics["overall_binary"] = binary_metrics(y_bin_all, p_bin_all)

    out_dir = args.checkpoint_dir
    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    (out_dir / "baselines.json").write_text(json.dumps(baselines_agg, indent=2), encoding="utf-8")
    write_csv(out_dir / "per_user_metrics.csv", per_user_rows)
    write_csv(out_dir / "predictions.csv", all_pred_rows)

    print(json.dumps(eval_metrics, indent=2))
    print("\n--- baselines (aggregated across folds) ---")
    print(json.dumps(baselines_agg, indent=2))
    if beats is not None:
        verdict = "BEATS" if beats else "DOES NOT BEAT"
        print(f"\n[verdict] switching model {verdict} the global-mean baseline (MAE).")


if __name__ == "__main__":
    main()
