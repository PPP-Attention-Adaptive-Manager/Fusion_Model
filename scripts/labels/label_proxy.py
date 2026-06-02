"""Configurable LABEL PROXY layer (STEP 5.6).

Centralizes supervision-target construction so training scripts never hardcode a
single label assumption. Supports four proxies:

  nasa_tlx           : session-level NASA-TLX factors (constant per session)
  dual_task_rt       : window-level dual-task reaction-time load (sparse, ~26%)
  nasa_time_weighted : NASA factors, but sample-weighted by within-session progress
  hybrid_rt_nasa     : RT load where available, else time-weighted NASA load

NONE of these are ground truth — they are proxy labels with documented modeling
assumptions (see LABEL_PROXY_SYSTEM_REPORT.md). Rows are never silently dropped;
unusable rows are masked and counted with reasons.

Public API:
  build_label_proxy(data_dir, proxy, config=None) -> dict
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NASA_COLS = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort",
             "frustration", "stress_self_report", "valence", "arousal"]
NASA_IDX = {c: i for i, c in enumerate(NASA_COLS)}
# alias: the task uses "arousal_proxy"; map it to the recorded arousal column.
FACTOR_ALIAS = {"arousal_proxy": "arousal"}
DEFAULT_NASA_FACTORS = ["mental_demand", "temporal_demand", "effort", "frustration", "arousal"]
PROXIES = ["nasa_tlx", "dual_task_rt", "nasa_time_weighted", "hybrid_rt_nasa"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _resolve_factor(name: str) -> int:
    name = FACTOR_ALIAS.get(name, name)
    if name not in NASA_IDX:
        raise KeyError(f"unknown NASA factor '{name}' (known: {NASA_COLS})")
    return NASA_IDX[name]


def _load(data_dir: Path):
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    nasa = np.load(data_dir / "nasa_tlx_labels.npy").astype(np.float64)
    n = len(metadata)
    if nasa.shape[0] != n:
        raise ValueError(f"nasa rows {nasa.shape[0]} != metadata {n}")

    dt_path = data_dir / "dual_task_window_labels.csv"
    dt: Dict[int, Dict[str, Any]] = {}
    if dt_path.exists():
        for r in csv.DictReader(dt_path.open(encoding="utf-8")):
            try:
                dt[int(r["row_idx"])] = r
            except (KeyError, ValueError):
                pass
    return metadata, nasa, dt


def _f(v) -> Optional[float]:
    if v is None or str(v).strip().lower() in ("", "nan"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _dt_reaction(row: Dict[str, Any]) -> Optional[float]:
    for k in ("reaction_time_mean", "reaction_time_ms", "reaction_time"):
        if k in row:
            v = _f(row.get(k))
            if v is not None and v > 0:
                return v
    return None


def _dt_field(row: Dict[str, Any], *keys, default=0.0) -> float:
    for k in keys:
        if k in row:
            v = _f(row.get(k))
            if v is not None:
                return v
    return default


def _dt_available(row: Optional[Dict[str, Any]]) -> bool:
    if row is None:
        return False
    return str(row.get("dual_task_available", "")).strip().lower() == "true"


# --------------------------------------------------------------------------- #
# helpers: robust normalization, progress, weight functions, states
# --------------------------------------------------------------------------- #
def robust_unit(values: np.ndarray, usable: np.ndarray, clip: float = 3.0):
    """median/IQR z, clipped to +/-clip, mapped to [0,1]. Returns (out, stats)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    v = values[usable]
    v = v[~np.isnan(v)]
    if v.size == 0:
        return out, {"median": 0.0, "iqr": 1.0}
    median = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > 1e-8 else (float(np.std(v)) if np.std(v) > 1e-8 else 1.0)
    z = (values - median) / scale
    z = np.clip(z, -clip, clip)
    out = (z + clip) / (2 * clip)
    return out, {"median": median, "iqr": iqr, "scale": scale, "clip": clip}


def session_progress(metadata) -> np.ndarray:
    """(window_center - session_start) / (session_end - session_start), clamped [0,1]."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for m in metadata:
        s = str(m.get("session_id"))
        ws = _f(m.get("window_start")); we = _f(m.get("window_end"))
        if ws is None or we is None:
            continue
        lo, hi = bounds.get(s, (ws, we))
        bounds[s] = (min(lo, ws), max(hi, we))
    prog = np.zeros(len(metadata), dtype=np.float64)
    for i, m in enumerate(metadata):
        s = str(m.get("session_id"))
        ws = _f(m.get("window_start")); we = _f(m.get("window_end"))
        if s not in bounds or ws is None or we is None:
            prog[i] = 0.5
            continue
        lo, hi = bounds[s]
        span = hi - lo
        center = 0.5 * (ws + we)
        prog[i] = float(np.clip((center - lo) / span, 0.0, 1.0)) if span > 1e-9 else 1.0
    return prog


def time_weights(progress: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    fn = cfg.get("weight_function", "linear")
    mn = float(cfg.get("min_weight", 0.2)); mx = float(cfg.get("max_weight", 1.0))
    if fn == "linear":
        w = progress
    elif fn == "quadratic":
        w = progress ** 2
    elif fn == "sigmoid":
        k = float(cfg.get("sigmoid_k", 8.0)); w = 1.0 / (1.0 + np.exp(-k * (progress - 0.5)))
    elif fn == "exponential":
        k = float(cfg.get("exponential_k", 4.0)); w = np.exp(k * (progress - 1.0))
    else:
        raise ValueError(f"unknown weight_function '{fn}'")
    wmin, wmax = float(w.min()), float(w.max())
    if wmax - wmin > 1e-9:
        w = (w - wmin) / (wmax - wmin)
    else:
        w = np.ones_like(w)
    return (mn + (mx - mn) * w).astype(np.float64)


def derive_states(nasa: np.ndarray, state_mode: str) -> Optional[np.ndarray]:
    if state_mode in ("none", None):
        return None
    from scripts.switching.train_switching_predictive import (derive_state_label,
                                                               map_5class_to_3class)
    s5 = np.asarray([derive_state_label(row) for row in nasa], dtype=np.int64)
    if state_mode == "5class":
        return s5
    if state_mode == "3class":
        return map_5class_to_3class(s5)
    raise ValueError(f"unsupported state_mode '{state_mode}' for NASA")


# --------------------------------------------------------------------------- #
# proxies
# --------------------------------------------------------------------------- #
def _nasa_factor_targets(nasa: np.ndarray, factor_names: List[str], normalize: bool) -> Tuple[np.ndarray, List[str]]:
    idx = [_resolve_factor(f) for f in factor_names]
    t = nasa[:, idx].astype(np.float64)
    if normalize:
        t = np.clip(t / 100.0, 0.0, 1.0)
    return t.astype(np.float32), list(factor_names)


def _nasa_load(nasa: np.ndarray, factor_weights: Dict[str, float]) -> np.ndarray:
    total = sum(factor_weights.values()) or 1.0
    load = np.zeros(nasa.shape[0], dtype=np.float64)
    for name, w in factor_weights.items():
        load += (w / total) * np.clip(nasa[:, _resolve_factor(name)] / 100.0, 0.0, 1.0)
    return np.clip(load, 0.0, 1.0)


def _rt_load(metadata, dt, cfg, usable_mask):
    """Robust RT load + miss/error penalty over rows where dual-task is available."""
    n = len(metadata)
    rt = np.full(n, np.nan)
    miss = np.zeros(n); err = np.zeros(n)
    for i in range(n):
        row = dt.get(i)
        if not _dt_available(row):
            continue
        r = _dt_reaction(row)
        if r is not None:
            rt[i] = r
        miss[i] = _dt_field(row, "miss_rate", default=0.0)
        err[i] = _dt_field(row, "error_rate", default=0.0)
    rt_norm, stats = robust_unit(rt, usable_mask)
    rt_w = float(cfg.get("rt_weight", 0.8)); miss_w = float(cfg.get("miss_weight", 0.2))
    err_w = float(cfg.get("error_weight", 0.0))
    # miss-only windows (available but no valid RT) -> rt_norm imputed to 1.0 (failed probe)
    miss_only = usable_mask & np.isnan(rt)
    rt_norm = np.where(miss_only, 1.0, rt_norm)
    load = rt_w * np.nan_to_num(rt_norm) + miss_w * miss + err_w * err
    return np.clip(load, 0.0, 1.0), stats, rt, miss, err


def _coverage(metadata, mask) -> Dict[str, Any]:
    by_user: Dict[str, List[int]] = {}
    by_sess: Dict[str, List[int]] = {}
    for i, m in enumerate(metadata):
        by_user.setdefault(str(m.get("user_id")), [0, 0]); by_user[str(m.get("user_id"))][1] += 1
        if mask[i]:
            by_user[str(m.get("user_id"))][0] += 1
        by_sess.setdefault(str(m.get("session_id")), [0, 0]); by_sess[str(m.get("session_id"))][1] += 1
        if mask[i]:
            by_sess[str(m.get("session_id"))][0] += 1
    return {
        "available": int(mask.sum()), "missing": int((~mask).sum()),
        "coverage_pct": round(100.0 * float(mask.mean()), 2),
        "by_user": {u: {"available": v[0], "total": v[1]} for u, v in sorted(by_user.items())},
        "by_session": {s: {"available": v[0], "total": v[1]} for s, v in sorted(by_sess.items())},
    }


def build_label_proxy(data_dir: str | Path, proxy: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if proxy not in PROXIES:
        raise ValueError(f"unknown proxy '{proxy}' (supported: {PROXIES})")
    data_dir = Path(data_dir)
    cfg = dict(config or {})
    metadata, nasa, dt = _load(data_dir)
    n = len(metadata)
    nasa_valid = np.isfinite(nasa).all(axis=1)  # all rows valid in this dataset

    diagnostics: Dict[str, Any] = {"n_rows": n, "proxy": proxy}
    progress = session_progress(metadata)

    if proxy == "nasa_tlx":
        cfg.setdefault("factor_names", DEFAULT_NASA_FACTORS)
        cfg.setdefault("normalize", True)
        cfg.setdefault("state_mode", "5class")
        targets, names = _nasa_factor_targets(nasa, cfg["factor_names"], cfg["normalize"])
        mask = nasa_valid.copy()
        weights = np.where(mask, 1.0, 0.0).astype(np.float32)
        states = derive_states(nasa, cfg["state_mode"])
        diagnostics["coverage"] = _coverage(metadata, mask)

    elif proxy == "dual_task_rt":
        cfg.setdefault("rt_weight", 0.8); cfg.setdefault("miss_weight", 0.2)
        cfg.setdefault("error_weight", 0.0); cfg.setdefault("normalization", "robust_iqr")
        cfg.setdefault("state_mode", "none"); cfg.setdefault("binary_threshold", 0.5)
        mask = np.asarray([_dt_available(dt.get(i)) for i in range(n)], dtype=bool)
        load, rt_stats, rt, miss, err = _rt_load(metadata, dt, cfg, mask)
        targets = np.where(mask, load, np.nan).reshape(-1, 1).astype(np.float32)
        # keep finite for masked-out rows (filled 0) but mask marks validity
        targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)
        names = ["dual_task_load"]
        weights = np.where(mask, 1.0, 0.0).astype(np.float32)
        if cfg["state_mode"] == "binary":
            states = np.where(load > cfg["binary_threshold"], 1, 0).astype(np.int64)
        elif cfg["state_mode"] == "3class":
            states = np.digitize(load, [0.33, 0.66]).astype(np.int64)
        else:
            states = None
        diagnostics["rt_stats"] = rt_stats
        diagnostics["coverage"] = _coverage(metadata, mask)

    elif proxy == "nasa_time_weighted":
        cfg.setdefault("weight_function", "sigmoid"); cfg.setdefault("min_weight", 0.2)
        cfg.setdefault("max_weight", 1.0); cfg.setdefault("sigmoid_k", 8.0)
        cfg.setdefault("exponential_k", 4.0); cfg.setdefault("state_mode", "5class")
        cfg.setdefault("factor_names", DEFAULT_NASA_FACTORS); cfg.setdefault("normalize", True)
        targets, names = _nasa_factor_targets(nasa, cfg["factor_names"], cfg["normalize"])
        mask = nasa_valid.copy()
        weights = time_weights(progress, cfg).astype(np.float32)
        weights = np.where(mask, weights, 0.0).astype(np.float32)
        states = derive_states(nasa, cfg["state_mode"])
        diagnostics["coverage"] = _coverage(metadata, mask)
        diagnostics["progress_weight"] = {"function": cfg["weight_function"],
                                          "min": float(weights[mask].min()), "max": float(weights[mask].max())}

    else:  # hybrid_rt_nasa
        cfg.setdefault("mode", "single_load"); cfg.setdefault("alpha_rt", 0.7)
        cfg.setdefault("rt_weight", 0.8); cfg.setdefault("miss_weight", 0.2)
        cfg.setdefault("error_weight", 0.0)
        cfg.setdefault("nasa_factor_weights", {"mental_demand": 0.25, "temporal_demand": 0.25,
                                               "effort": 0.20, "frustration": 0.20, "arousal_proxy": 0.10})
        cfg.setdefault("nasa_time_weight", {"weight_function": "sigmoid", "min_weight": 0.2,
                                            "max_weight": 1.0, "sigmoid_k": 8.0})
        cfg.setdefault("rt_sample_weight", 1.0); cfg.setdefault("state_mode", "none")
        cfg.setdefault("binary_threshold", 0.5)
        if cfg["mode"] != "single_load":
            # multi_task documented as TODO; fall back to single_load
            diagnostics["note"] = "multi_task is a documented TODO; using single_load"
            cfg["mode"] = "single_load"
        rt_mask = np.asarray([_dt_available(dt.get(i)) for i in range(n)], dtype=bool)
        rt_load, rt_stats, rt, miss, err = _rt_load(metadata, dt, cfg, rt_mask)
        nasa_load = _nasa_load(nasa, cfg["nasa_factor_weights"])
        nasa_w = time_weights(progress, cfg["nasa_time_weight"])
        alpha = float(cfg["alpha_rt"])
        target = np.where(rt_mask, alpha * rt_load + (1 - alpha) * nasa_load, nasa_load)
        weights = np.where(rt_mask, float(cfg["rt_sample_weight"]), nasa_w).astype(np.float32)
        targets = np.clip(target, 0.0, 1.0).reshape(-1, 1).astype(np.float32)
        names = ["hybrid_load"]
        mask = nasa_valid.copy()  # every NASA-valid row is usable (RT optional)
        if cfg["state_mode"] == "binary":
            states = np.where(targets[:, 0] > cfg["binary_threshold"], 1, 0).astype(np.int64)
        elif cfg["state_mode"] == "3class":
            states = np.digitize(targets[:, 0], [0.33, 0.66]).astype(np.int64)
        else:
            states = None
        both = rt_mask & nasa_valid
        corr = float(np.corrcoef(rt_load[both], nasa_load[both])[0, 1]) if both.sum() > 2 else float("nan")
        diagnostics["rt_stats"] = rt_stats
        diagnostics["rows_with_rt"] = int(rt_mask.sum())
        diagnostics["rows_without_rt"] = int((~rt_mask).sum())
        diagnostics["rt_vs_nasa_load_corr"] = corr
        diagnostics["coverage"] = _coverage(metadata, rt_mask)

    # safety: targets finite for masked rows
    if np.isnan(targets[mask]).any():
        raise RuntimeError("NaN in targets for masked rows")
    weights = np.nan_to_num(weights, nan=0.0).astype(np.float32)
    diagnostics["target_dim"] = int(targets.shape[1])
    diagnostics["target_mean"] = [round(float(np.mean(targets[mask, j])), 4) for j in range(targets.shape[1])] if mask.any() else []
    diagnostics["masked_excluded"] = int((~mask).sum())

    return {
        "targets": targets.astype(np.float32),
        "state_labels": (states.astype(np.int64) if states is not None else None),
        "sample_weights": weights.astype(np.float32),
        "mask": mask.astype(bool),
        "progress": progress.astype(np.float32),
        "target_names": names,
        "proxy_name": proxy,
        "config": cfg,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }
