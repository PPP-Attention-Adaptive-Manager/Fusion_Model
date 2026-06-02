"""Build window-level dual-task labels for the switching predictive model.

Motivation
----------
NASA-TLX is collected once per session and copied to every 120s window in that
session. That is weak, temporally-constant supervision for a window-level model.

Dual-task probes happen *during* the session and can be aligned to individual
120s windows, giving genuine window-level cognitive-load supervision:

    reaction_time_ms increases  -> higher load
    miss_rate        increases  -> higher load
    error_rate       increases  -> higher load

This script aligns dual-task events to the windows described in
``data_training/metadata.json`` and writes per-window statistics.

Outputs
-------
    data_training/dual_task_window_labels.csv   (human-readable, all columns)
    data_training/dual_task_window_labels.npy   (numeric columns, metadata order)

Important
---------
* Rows are aligned 1:1 with ``metadata.json`` (and therefore with
  ``tucker_slices.npy`` row order).
* Windows with no dual-task file or no event in range get NaN label fields and
  ``dual_task_available = false``.  We never fabricate labels.
* The ``dual_task_load_*`` columns written here are PRELIMINARY (global robust
  scaling) and only meant for visualization.  Training recomputes load scores
  using train-split-only statistics to avoid leakage.

Usage
-----
    python scripts/switching/build_dual_task_window_labels.py \\
        --data-training-dir data_training \\
        --raw-data-dir data \\
        --output data_training/dual_task_window_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Search order for a session's dual_task.csv.  The first non-empty file wins.
# "" means directly under the session folder; the rest are subdirectories.
DUAL_TASK_SUBDIRS = ["", "raw", "events", "features", "labels"]

# Numeric columns saved to the .npy file (aligned to metadata row order).
NUMERIC_COLUMNS = [
    "dual_task_available",
    "dual_task_count",
    "reaction_time_mean",
    "reaction_time_median",
    "reaction_time_std",
    "reaction_time_min",
    "reaction_time_max",
    "miss_count",
    "error_count",
    "success_count",
    "miss_rate",
    "error_rate",
    "success_rate",
    "dual_task_load_absolute_prelim",
    "dual_task_load_relative_prelim",
]


def _as_bool(value: Any) -> bool:
    """Parse the loose True/False/1/0 booleans found in the raw CSVs."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def find_dual_task_file(session_dir: Path) -> Optional[Path]:
    """Locate the most likely dual_task.csv for a session.

    Preference order is given by DUAL_TASK_SUBDIRS; we additionally fall back to
    a recursive glob.  A file is only selected if it contains at least a header
    plus one data row, otherwise we keep looking (but remember the first
    header-only file so an empty-but-present file is still reported).
    """
    header_only_fallback: Optional[Path] = None

    candidates: List[Path] = []
    for sub in DUAL_TASK_SUBDIRS:
        candidates.append(session_dir / sub / "dual_task.csv" if sub else session_dir / "dual_task.csv")
    # Recursive fallback for anything we have not enumerated explicitly.
    candidates.extend(sorted(session_dir.glob("**/dual_task.csv")))

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    continue
                first_row = next(reader, None)
                if first_row is not None:
                    return path  # has data -> best candidate
                if header_only_fallback is None:
                    header_only_fallback = path
        except OSError:
            continue

    return header_only_fallback


def read_dual_task_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ts = _as_float(row.get("timestamp"))
            if ts is None:
                continue
            events.append(
                {
                    "timestamp": ts,
                    "reaction_time_ms": _as_float(row.get("reaction_time_ms")),
                    "success": _as_bool(row.get("success")),
                    "miss": _as_bool(row.get("miss")),
                    "error": _as_bool(row.get("error")),
                    "app_name": row.get("app_name"),
                }
            )
    return events


def _nan() -> float:
    return float("nan")


def empty_window_stats() -> Dict[str, float]:
    """Stats for a window with no usable dual-task events."""
    return {
        "dual_task_count": 0,
        "reaction_time_mean": _nan(),
        "reaction_time_median": _nan(),
        "reaction_time_std": _nan(),
        "reaction_time_min": _nan(),
        "reaction_time_max": _nan(),
        "miss_count": 0,
        "error_count": 0,
        "success_count": 0,
        "miss_rate": _nan(),
        "error_rate": _nan(),
        "success_rate": _nan(),
    }


def compute_window_stats(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute per-window dual-task statistics.

    Reaction-time stats only use events with a strictly positive reaction time
    (missed probes record rt=0, which is not a real latency); but those misses
    still count towards miss/error/success rates.
    """
    n = len(events)
    if n == 0:
        return empty_window_stats()

    miss_count = sum(1 for e in events if e["miss"])
    error_count = sum(1 for e in events if e["error"])
    success_count = sum(1 for e in events if e["success"])

    # Valid reaction times: positive and not from a missed probe.
    rts = [
        e["reaction_time_ms"]
        for e in events
        if e["reaction_time_ms"] is not None and e["reaction_time_ms"] > 0.0 and not e["miss"]
    ]
    rt_arr = np.asarray(rts, dtype=np.float64)

    if rt_arr.size > 0:
        rt_mean = float(rt_arr.mean())
        rt_median = float(np.median(rt_arr))
        rt_std = float(rt_arr.std(ddof=0))
        rt_min = float(rt_arr.min())
        rt_max = float(rt_arr.max())
    else:
        rt_mean = rt_median = rt_std = rt_min = rt_max = _nan()

    miss_rate = float(np.clip(miss_count / n, 0.0, 1.0))
    error_rate = float(np.clip(error_count / n, 0.0, 1.0))
    success_rate = float(np.clip(success_count / n, 0.0, 1.0))

    return {
        "dual_task_count": n,
        "reaction_time_mean": rt_mean,
        "reaction_time_median": rt_median,
        "reaction_time_std": rt_std,
        "reaction_time_min": rt_min,
        "reaction_time_max": rt_max,
        "miss_count": miss_count,
        "error_count": error_count,
        "success_count": success_count,
        "miss_rate": miss_rate,
        "error_rate": error_rate,
        "success_rate": success_rate,
    }


def robust_scale_to_unit(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] via robust (median / IQR) scaling, then sigmoid.

    NaNs are preserved.  Used only for the *preliminary* visualization scores in
    the CSV; training recomputes scaling from the train split.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return out
    v = values[mask]
    median = np.median(v)
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    scale = iqr if iqr > 1e-8 else (np.std(v) if np.std(v) > 1e-8 else 1.0)
    z = (v - median) / (scale + 1e-8)
    out[mask] = 1.0 / (1.0 + np.exp(-z))
    return out


def compute_preliminary_loads(rows: List[Dict[str, Any]]) -> None:
    """Attach preliminary (global) absolute & relative load scores in-place.

    These are for visualization only.  Relative RT uses each user's own median
    reaction time as a personalization baseline (mimicking calibration).
    """
    rt_mean = np.asarray([r["reaction_time_mean"] for r in rows], dtype=np.float64)
    miss_rate = np.asarray([r["miss_rate"] for r in rows], dtype=np.float64)
    error_rate = np.asarray([r["error_rate"] for r in rows], dtype=np.float64)

    # Absolute: global robust-scaled RT.
    rt_norm_abs = robust_scale_to_unit(rt_mean)

    # Relative: per-user baseline RT.
    users = [r["user_id"] for r in rows]
    rel_rt = np.full(rt_mean.shape, np.nan, dtype=np.float64)
    for user in set(users):
        idx = np.asarray([i for i, u in enumerate(users) if u == user])
        u_rt = rt_mean[idx]
        valid = u_rt[~np.isnan(u_rt)]
        if valid.size == 0:
            continue
        baseline = float(np.median(valid))
        rel_rt[idx] = (u_rt - baseline) / (baseline + 1e-8)
    rt_norm_rel = robust_scale_to_unit(rel_rt)

    miss = np.nan_to_num(miss_rate, nan=0.0)
    err = np.nan_to_num(error_rate, nan=0.0)

    load_abs = 0.60 * np.nan_to_num(rt_norm_abs, nan=0.0) + 0.25 * miss + 0.15 * err
    load_rel = 0.60 * np.nan_to_num(rt_norm_rel, nan=0.0) + 0.25 * miss + 0.15 * err
    load_abs = np.clip(load_abs, 0.0, 1.0)
    load_rel = np.clip(load_rel, 0.0, 1.0)

    avail = np.asarray([r["dual_task_available"] for r in rows], dtype=bool)
    for i, r in enumerate(rows):
        r["dual_task_load_absolute_prelim"] = float(load_abs[i]) if avail[i] else _nan()
        r["dual_task_load_relative_prelim"] = float(load_rel[i]) if avail[i] else _nan()


CSV_COLUMNS = [
    "row_idx",
    "user_id",
    "session_id",
    "window_idx",
    "window_start",
    "window_end",
    "dual_task_file",
    "dual_task_available",
    "dual_task_count",
    "reaction_time_mean",
    "reaction_time_median",
    "reaction_time_std",
    "reaction_time_min",
    "reaction_time_max",
    "miss_count",
    "error_count",
    "success_count",
    "miss_rate",
    "error_rate",
    "success_rate",
    "dual_task_load_absolute_prelim",
    "dual_task_load_relative_prelim",
]


def build_rows(metadata: List[Dict[str, Any]], raw_data_dir: Path) -> List[Dict[str, Any]]:
    # Cache resolved dual-task file + events per session.
    file_cache: Dict[str, Optional[Path]] = {}
    event_cache: Dict[str, List[Dict[str, Any]]] = {}

    rows: List[Dict[str, Any]] = []
    for row_idx, meta in enumerate(metadata):
        session_id = str(meta.get("session_id", ""))
        user_id = str(meta.get("user_id", "unknown_user"))
        window_idx = meta.get("window_idx")
        window_start = _as_float(meta.get("window_start"))
        window_end = _as_float(meta.get("window_end"))

        if session_id not in file_cache:
            session_dir = raw_data_dir / session_id
            dt_file = find_dual_task_file(session_dir) if session_dir.exists() else None
            file_cache[session_id] = dt_file
            if dt_file is not None:
                events = read_dual_task_events(dt_file)
                event_cache[session_id] = events
                print(
                    f"[dual-task] {session_id}: selected {dt_file} "
                    f"({len(events)} events)"
                )
            else:
                event_cache[session_id] = []
                print(f"[dual-task] {session_id}: NO dual_task.csv found")

        dt_file = file_cache[session_id]
        events = event_cache[session_id]

        row: Dict[str, Any] = {
            "row_idx": row_idx,
            "user_id": user_id,
            "session_id": session_id,
            "window_idx": window_idx,
            "window_start": window_start,
            "window_end": window_end,
            "dual_task_file": str(dt_file) if dt_file is not None else "",
        }

        if dt_file is None or window_start is None or window_end is None:
            row["dual_task_available"] = False
            row.update(empty_window_stats())
        else:
            in_window = [e for e in events if window_start <= e["timestamp"] < window_end]
            stats = compute_window_stats(in_window)
            row.update(stats)
            # available iff at least one event fell in the window
            row["dual_task_available"] = len(in_window) > 0

        rows.append(row)

    compute_preliminary_loads(rows)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in CSV_COLUMNS}
            if isinstance(out["dual_task_available"], bool):
                out["dual_task_available"] = "true" if out["dual_task_available"] else "false"
            writer.writerow(out)


def write_npy(path: Path, rows: List[Dict[str, Any]]) -> None:
    arr = np.full((len(rows), len(NUMERIC_COLUMNS)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, col in enumerate(NUMERIC_COLUMNS):
            val = r.get(col)
            if col == "dual_task_available":
                arr[i, j] = 1.0 if r.get("dual_task_available") else 0.0
            elif val is None:
                arr[i, j] = np.nan
            else:
                arr[i, j] = float(val)
    np.save(path, arr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build window-level dual-task labels.")
    parser.add_argument("--data-training-dir", type=Path, default=Path("data_training"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_training/dual_task_window_labels.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.data_training_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"[dual-task] loaded {len(metadata)} metadata rows from {metadata_path}")

    rows = build_rows(metadata, args.raw_data_dir)

    csv_path = args.output
    npy_path = csv_path.with_suffix(".npy")
    write_csv(csv_path, rows)
    write_npy(npy_path, rows)

    # Coverage summary.
    n = len(rows)
    avail = sum(1 for r in rows if r["dual_task_available"])
    per_user: Dict[str, List[int]] = {}
    for r in rows:
        bucket = per_user.setdefault(r["user_id"], [0, 0])
        bucket[1] += 1
        if r["dual_task_available"]:
            bucket[0] += 1

    print("\n========== DUAL-TASK LABEL COVERAGE ==========")
    print(f"total windows                 : {n}")
    print(f"windows with dual-task labels : {avail} ({100.0 * avail / max(n, 1):.1f}%)")
    print(f"windows missing labels        : {n - avail}")
    print("per-user coverage (available / total):")
    for user in sorted(per_user):
        a, t = per_user[user]
        print(f"  {user:<14s} {a:>4d} / {t:<4d}")
    print(f"\nwrote: {csv_path}")
    print(f"wrote: {npy_path}")


if __name__ == "__main__":
    main()
