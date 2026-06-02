"""Shared utilities for dual-task window-level supervision of the switching model.

This module is imported by both ``train_switching_predictive.py`` and
``evaluate_dual_task_switching.py`` so that label loading, load-score
computation, splits, weighting and leakage rules stay identical.

Key design rules (see DUAL_TASK_SWITCHING_SUPERVISION_REPORT.md):

* Labels are aligned 1:1 with metadata / tucker_slices row order.
* Only rows with ``dual_task_available == True`` are usable; we never fabricate
  labels for empty windows.
* Robust RT scaling is fit on the TRAIN split only (no test leakage).
* The per-user RT baseline used by the *relative* target may use a user's own
  available windows even for a held-out LOSO user.  This deliberately mimics a
  short personalization / calibration step and is documented as such.
* In this dataset every covered window contains exactly one probe, so
  ``miss_rate`` is binary and ``error_rate`` is always 0.  Miss-only windows
  have no valid reaction time; their RT-norm is imputed to 1.0 (a failed probe
  is treated as a maximum-latency / maximum-load proxy).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Load-score weights (shared by absolute and relative targets).
W_RT = 0.60
W_MISS = 0.25
W_ERROR = 0.15

SWITCHING_MODALITY_INDEX = 3
INPUT_FLAT_DIM = 512


# --------------------------------------------------------------------------- #
# Label container
# --------------------------------------------------------------------------- #
class DualTaskLabels:
    """Per-window dual-task arrays, aligned to metadata row order (length N)."""

    def __init__(
        self,
        *,
        available: np.ndarray,
        dual_task_count: np.ndarray,
        reaction_time_mean: np.ndarray,
        miss_rate: np.ndarray,
        error_rate: np.ndarray,
        users: np.ndarray,
        sessions: np.ndarray,
        window_idx: np.ndarray,
        window_start: np.ndarray,
        window_end: np.ndarray,
    ) -> None:
        self.available = available
        self.dual_task_count = dual_task_count
        self.reaction_time_mean = reaction_time_mean
        self.miss_rate = miss_rate
        self.error_rate = error_rate
        self.users = users
        self.sessions = sessions
        self.window_idx = window_idx
        self.window_start = window_start
        self.window_end = window_end

    def __len__(self) -> int:
        return int(self.available.shape[0])


def load_dual_task_labels(csv_path: str | Path, metadata: List[Dict[str, Any]]) -> DualTaskLabels:
    """Load the dual-task label CSV and verify 1:1 alignment with metadata.

    Alignment is verified by ``row_idx`` *and* by (user_id, session_id,
    window_idx) matching the metadata row.  A mismatch raises immediately —
    silent misalignment would corrupt supervision.
    """
    csv_path = Path(csv_path)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    n = len(metadata)
    if len(rows) != n:
        raise ValueError(
            f"dual-task label rows ({len(rows)}) != metadata rows ({n}); "
            "rebuild labels with build_dual_task_window_labels.py"
        )

    def _f(v: Any) -> float:
        if v is None or str(v).strip().lower() in {"", "nan"}:
            return float("nan")
        return float(v)

    available = np.zeros(n, dtype=bool)
    count = np.zeros(n, dtype=np.float64)
    rt_mean = np.full(n, np.nan, dtype=np.float64)
    miss_rate = np.full(n, np.nan, dtype=np.float64)
    error_rate = np.full(n, np.nan, dtype=np.float64)
    users = np.empty(n, dtype=object)
    sessions = np.empty(n, dtype=object)
    window_idx = np.zeros(n, dtype=np.int64)
    window_start = np.full(n, np.nan, dtype=np.float64)
    window_end = np.full(n, np.nan, dtype=np.float64)

    for i, (row, meta) in enumerate(zip(rows, metadata)):
        if int(row["row_idx"]) != i:
            raise ValueError(f"row_idx mismatch at line {i}: {row['row_idx']}")
        if str(row["session_id"]) != str(meta.get("session_id")) or str(row["user_id"]) != str(
            meta.get("user_id")
        ):
            raise ValueError(
                f"user/session mismatch at row {i}: "
                f"label=({row['user_id']},{row['session_id']}) "
                f"meta=({meta.get('user_id')},{meta.get('session_id')})"
            )
        available[i] = str(row["dual_task_available"]).strip().lower() == "true"
        count[i] = _f(row.get("dual_task_count"))
        rt_mean[i] = _f(row.get("reaction_time_mean"))
        miss_rate[i] = _f(row.get("miss_rate"))
        error_rate[i] = _f(row.get("error_rate"))
        users[i] = str(row["user_id"])
        sessions[i] = str(row["session_id"])
        window_idx[i] = int(float(row.get("window_idx", i)))
        window_start[i] = _f(row.get("window_start"))
        window_end[i] = _f(row.get("window_end"))

    return DualTaskLabels(
        available=available,
        dual_task_count=count,
        reaction_time_mean=rt_mean,
        miss_rate=miss_rate,
        error_rate=error_rate,
        users=users,
        sessions=sessions,
        window_idx=window_idx,
        window_start=window_start,
        window_end=window_end,
    )


# --------------------------------------------------------------------------- #
# Robust scaler (median / IQR -> sigmoid), fit on train only
# --------------------------------------------------------------------------- #
class RobustUnitScaler:
    """Fit robust center/scale on train values; transform to [0,1] via sigmoid."""

    def __init__(self) -> None:
        self.median = 0.0
        self.scale = 1.0

    def fit(self, values: np.ndarray) -> "RobustUnitScaler":
        v = values[~np.isnan(values)]
        if v.size == 0:
            self.median, self.scale = 0.0, 1.0
            return self
        self.median = float(np.median(v))
        q75, q25 = np.percentile(v, [75, 25])
        iqr = float(q75 - q25)
        if iqr > 1e-8:
            self.scale = iqr
        else:
            std = float(np.std(v))
            self.scale = std if std > 1e-8 else 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.median) / (self.scale + 1e-8)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> Dict[str, float]:
        return {"median": self.median, "scale": self.scale}


# --------------------------------------------------------------------------- #
# Load-score computation (no test leakage in the fitted scaler)
# --------------------------------------------------------------------------- #
def compute_user_baselines(labels: DualTaskLabels) -> Dict[str, float]:
    """Median reaction_time_mean per user over that user's available windows.

    Used as the personalization baseline for the *relative* target.  Computed
    from each user's own data (including a held-out LOSO user's own windows),
    mimicking a calibration step — documented as such in the report.
    """
    baselines: Dict[str, float] = {}
    for user in sorted(set(labels.users.tolist())):
        mask = (labels.users == user) & labels.available & ~np.isnan(labels.reaction_time_mean)
        vals = labels.reaction_time_mean[mask]
        if vals.size > 0:
            baselines[user] = float(np.median(vals))
    # Global fallback for users with no valid RT.
    glob_mask = labels.available & ~np.isnan(labels.reaction_time_mean)
    global_baseline = float(np.median(labels.reaction_time_mean[glob_mask])) if glob_mask.any() else 0.0
    baselines["__global__"] = global_baseline
    return baselines


def build_load_targets(
    labels: DualTaskLabels,
    *,
    target_mode: str,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, RobustUnitScaler, Dict[str, float]]:
    """Compute the dual-task load target for all rows.

    The RT robust scaler is fit on the TRAIN split's available rows only.  The
    relative target additionally uses per-user RT baselines (personalization).

    Returns
    -------
    target : (N,) float array; NaN where dual_task_available is False.
    scaler : the fitted RobustUnitScaler (RT-space or relative-RT-space).
    baselines : per-user RT baselines (empty dict for the absolute target).
    """
    if target_mode not in ("absolute", "relative"):
        raise ValueError(f"Unknown dual-task target: {target_mode}")

    n = len(labels)
    rt = labels.reaction_time_mean.copy()
    miss = np.nan_to_num(labels.miss_rate, nan=0.0)
    err = np.nan_to_num(labels.error_rate, nan=0.0)

    train_avail = np.zeros(n, dtype=bool)
    train_avail[train_idx] = True
    train_avail &= labels.available

    baselines: Dict[str, float] = {}

    if target_mode == "absolute":
        scaler = RobustUnitScaler().fit(rt[train_avail & ~np.isnan(rt)])
        rt_norm = scaler.transform(rt)
    else:  # relative
        baselines = compute_user_baselines(labels)
        rel = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if not labels.available[i] or np.isnan(rt[i]):
                continue
            base = baselines.get(str(labels.users[i]), baselines["__global__"])
            rel[i] = (rt[i] - base) / (base + 1e-8)
        scaler = RobustUnitScaler().fit(rel[train_avail & ~np.isnan(rel)])
        rt_norm = scaler.transform(rel)

    # Impute RT-norm for available miss-only windows (no valid RT): treat a
    # failed probe as a maximum-latency / maximum-load proxy.
    miss_only = labels.available & np.isnan(rt)
    rt_norm = np.where(miss_only, 1.0, rt_norm)

    target = W_RT * rt_norm + W_MISS * miss + W_ERROR * err
    target = np.clip(target, 0.0, 1.0)
    target[~labels.available] = np.nan
    return target.astype(np.float64), scaler, baselines


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def loso_folds(
    users: np.ndarray,
    available: np.ndarray,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_test_labels: int = 1,
) -> List[Dict[str, Any]]:
    """Leave-one-subject-out folds, only over users that have >= min_test_labels
    available dual-task windows (others can never be evaluated as test)."""
    unique_users = sorted(set(users.tolist()))
    avail_counts = {u: int((available & (users == u)).sum()) for u in unique_users}
    testable = [u for u in unique_users if avail_counts[u] >= min_test_labels]

    folds: List[Dict[str, Any]] = []
    for fold_idx, test_user in enumerate(testable):
        train_val_users = [u for u in unique_users if u != test_user]
        rng = np.random.default_rng(seed + fold_idx)
        shuffled = list(rng.permutation(train_val_users))
        if len(shuffled) <= 1:
            val_users, train_users = list(shuffled), list(shuffled)
        else:
            n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_ratio))))
            val_users = sorted(shuffled[:n_val])
            train_users = sorted(shuffled[n_val:])
        folds.append(
            {
                "fold_id": sanitize_id(test_user),
                "split_mode": "loso",
                "test_user": test_user,
                "train_users": train_users,
                "val_users": val_users,
                "train_idx": np.where(np.isin(users, train_users) & available)[0],
                "val_idx": np.where(np.isin(users, val_users) & available)[0],
                "test_idx": np.where((users == test_user) & available)[0],
            }
        )
    return folds


def session_within_user_folds(
    users: np.ndarray,
    sessions: np.ndarray,
    available: np.ndarray,
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Single within-user/session split.

    Sessions are partitioned per user into train/val/test so that no session
    appears in more than one split (users may appear in all splits, sessions
    may not).  Returns a single fold (fold_id='within_user').
    """
    rng = np.random.default_rng(seed)
    train_sessions: List[str] = []
    val_sessions: List[str] = []
    test_sessions: List[str] = []

    for user in sorted(set(users.tolist())):
        # Sessions of this user that contain at least one available window.
        u_mask = users == user
        u_sessions = sorted(set(sessions[u_mask & available].tolist()))
        if not u_sessions:
            continue
        shuffled = list(rng.permutation(u_sessions))
        k = len(shuffled)
        if k == 1:
            train_sessions.extend(shuffled)
        elif k == 2:
            train_sessions.append(shuffled[0])
            test_sessions.append(shuffled[1])
        else:
            n_test = max(1, int(round(k * 0.3)))
            n_val = max(1, int(round(k * 0.15)))
            n_val = min(n_val, k - n_test - 1) if k - n_test - 1 > 0 else 0
            test_sessions.extend(shuffled[:n_test])
            val_sessions.extend(shuffled[n_test : n_test + n_val])
            train_sessions.extend(shuffled[n_test + n_val :])

    def _idx(sess_list: List[str]) -> np.ndarray:
        if not sess_list:
            return np.array([], dtype=np.int64)
        return np.where(np.isin(sessions, sess_list) & available)[0]

    fold = {
        "fold_id": "within_user",
        "split_mode": "session_within_user",
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "test_sessions": sorted(test_sessions),
        "train_idx": _idx(train_sessions),
        "val_idx": _idx(val_sessions if val_sessions else train_sessions),
        "test_idx": _idx(test_sessions),
    }
    return [fold]


def build_folds(
    labels: DualTaskLabels,
    *,
    split_mode: str,
    seed: int = 42,
    min_test_labels: int = 1,
) -> List[Dict[str, Any]]:
    if split_mode == "loso":
        return loso_folds(
            labels.users, labels.available, seed=seed, min_test_labels=min_test_labels
        )
    if split_mode == "session_within_user":
        return session_within_user_folds(
            labels.users, labels.sessions, labels.available, seed=seed
        )
    raise ValueError(f"Unsupported split_mode: {split_mode}")


# --------------------------------------------------------------------------- #
# User-balanced sample weights
# --------------------------------------------------------------------------- #
def user_balanced_weights(users_subset: np.ndarray) -> np.ndarray:
    """weight_i = 1 / (#windows for user_i in this subset); normalized to mean 1."""
    counts: Dict[str, int] = {}
    for u in users_subset:
        counts[u] = counts.get(u, 0) + 1
    w = np.asarray([1.0 / counts[u] for u in users_subset], dtype=np.float64)
    mean = w.mean() if w.size else 1.0
    if mean > 0:
        w = w / mean
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature normalization (train-only) — same convention as the NASA pipeline
# --------------------------------------------------------------------------- #
def fit_feature_norm(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if train_idx.size == 0:
        mean = np.zeros(X.shape[1], dtype=np.float32)
        std = np.ones(X.shape[1], dtype=np.float32)
        return mean, std
    mean = X[train_idx].mean(axis=0).astype(np.float32)
    std = X[train_idx].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def load_switching_slice(data_dir: str | Path) -> Tuple[np.ndarray, List[Dict[str, Any]], Path]:
    """Load tucker_slices switching slice (N,512) and metadata, with safety checks."""
    root = Path(data_dir)
    if not root.exists():
        alt = Path("data_training")
        if str(root) in ("data_for_training",) and alt.exists():
            root = alt
        else:
            raise FileNotFoundError(f"Training data folder not found: {data_dir}")
    slices = np.load(root / "tucker_slices.npy").astype(np.float32)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    nasa = np.load(root / "nasa_tlx_labels.npy")
    if slices.ndim != 3 or slices.shape[1:] != (4, INPUT_FLAT_DIM):
        raise ValueError(f"Expected tucker_slices (N,4,512), got {slices.shape}")
    if len(metadata) != slices.shape[0]:
        raise ValueError(
            f"metadata length ({len(metadata)}) != tucker_slices N ({slices.shape[0]})"
        )
    if nasa.shape[0] != slices.shape[0]:
        raise ValueError(
            f"nasa_tlx_labels N ({nasa.shape[0]}) != tucker_slices N ({slices.shape[0]})"
        )
    return slices[:, SWITCHING_MODALITY_INDEX, :], metadata, root
