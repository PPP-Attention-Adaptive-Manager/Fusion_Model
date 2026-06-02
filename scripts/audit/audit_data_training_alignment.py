"""Audit alignment between the record-tool output (``data/``) and the processed
fusion training dataset (``data_training/``).

This script is READ-ONLY. It never writes to ``data/`` or ``data_training/`` and
never trains anything. It only inspects, cross-references, and emits audit CSVs
plus feeds a written report.

Authoritative source of truth : ``data/``  (record-tool output)
Derived / possibly-incomplete  : ``data_training/`` (tucker_slices, metadata, ...)

Usage
-----
    python scripts/audit/audit_data_training_alignment.py \\
        --data-training-dir data_training \\
        --raw-data-dir data \\
        --output-dir outputs/audit

Outputs (all under --output-dir):
    raw_session_inventory.csv
    dual_task_file_candidates.csv
    processed_metadata_inventory.csv
    processed_user_summary.csv
    processed_session_summary.csv
    session_coverage_diff.csv
    session_alignment_summary.csv
    timestamp_alignment_report.csv
    graph_metadata_alignment.csv
    dual_task_matching_debug.csv
    dual_task_matching_mode_summary.csv
    zero_coverage_user_report.csv
    row_alignment_check.csv
    probe_count_reality_check.csv

Notes / assumptions
-------------------
* Raw sessions carry only ``device_id``; ``user_id`` is a processed-only mapping
  that lives in ``metadata.json``. We therefore capture ``device_id`` for raw
  sessions and additionally attach the metadata user_id where the session is
  processed (clearly labelled as processed-derived).
* Graph numbering convention observed in this repo: ``graph_001.json`` carries
  ``window.window_id == 'w000000'`` -> window_idx 0 (i.e. graph_N -> idx N-1).
  We test both ``graph_{idx+1}`` and ``graph_{idx}`` to be safe.
* Unix-second timestamps in this dataset are ~1.7e9. We use magnitude bands to
  guess the unit (seconds / milliseconds / relative).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Priority order for dual_task.csv discovery (root first, then known subdirs).
DUAL_TASK_PRIORITY = ["", "raw", "events", "features"]
GRAPH_120S_DIRNAME = "data_graph_120s"
ZERO_COVERAGE_USERS = ["Graja", "ladabos", "mohanned"]

# Unix-second magnitude band (roughly years 2001..2286).
UNIX_SECONDS_LO = 1.0e9
UNIX_SECONDS_HI = 1.0e10
UNIX_MS_LO = 1.0e12
UNIX_MS_HI = 1.0e13


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still emit a header-less empty file so downstream tooling sees it exists
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as h:
                csv.DictWriter(h, fieldnames=fieldnames).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        # Union of keys across all rows, preserving first-seen order, so rows
        # with heterogeneous schemas (e.g. special-case rows) never drop columns.
        fn: List[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fn.append(k)
    else:
        fn = fieldnames
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _to_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def guess_ts_unit(ts_min: Optional[float], ts_max: Optional[float]) -> str:
    if ts_min is None or ts_max is None:
        return "unknown"
    m = ts_max
    if UNIX_SECONDS_LO <= m < UNIX_SECONDS_HI:
        return "seconds"
    if UNIX_MS_LO <= m < UNIX_MS_HI:
        return "milliseconds"
    if 0 <= m < 1.0e6:
        return "relative"
    return "unknown"


# --------------------------------------------------------------------------- #
# Dual-task file discovery + reading
# --------------------------------------------------------------------------- #
def _event_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as h:
            r = csv.reader(h)
            next(r, None)  # header
            return sum(1 for _ in r)
    except OSError:
        return -1


def find_dual_task_candidates(session_dir: Path) -> List[Dict[str, Any]]:
    """Return all dual_task.csv candidates with metadata, in discovery order."""
    candidates: List[Path] = []
    seen: set[Path] = set()
    for sub in DUAL_TASK_PRIORITY:
        p = (session_dir / sub / "dual_task.csv") if sub else (session_dir / "dual_task.csv")
        if p.exists() and p not in seen:
            candidates.append(p)
            seen.add(p)
    for p in sorted(session_dir.glob("**/dual_task.csv")):
        if p not in seen:
            candidates.append(p)
            seen.add(p)

    out: List[Dict[str, Any]] = []
    for p in candidates:
        try:
            rel = str(p.relative_to(session_dir))
        except ValueError:
            rel = str(p)
        # priority rank for selection
        parent = p.parent.name if p.parent != session_dir else ""
        rank = DUAL_TASK_PRIORITY.index(parent) if parent in DUAL_TASK_PRIORITY else len(DUAL_TASK_PRIORITY)
        if p.parent == session_dir:
            rank = 0
        n = _event_count(p)
        out.append(
            {
                "path": str(p),
                "relative_path": rel,
                "priority_rank": rank,
                "event_count": n,
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
        )
    return out


def select_dual_task_file(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the most likely dual_task.csv: prefer priority dir with events,
    else largest non-empty, else first present."""
    if not candidates:
        return None
    with_events = [c for c in candidates if c["event_count"] and c["event_count"] > 0]
    if with_events:
        # lowest priority_rank, tie-break by most events
        return sorted(with_events, key=lambda c: (c["priority_rank"], -c["event_count"]))[0]
    # no events anywhere -> prefer priority rank, tie-break largest file
    return sorted(candidates, key=lambda c: (c["priority_rank"], -c["size_bytes"]))[0]


def read_dual_task_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if path is None or not path.exists():
        return events
    with path.open("r", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            ts = _to_float(row.get("timestamp"))
            if ts is None:
                continue
            events.append(
                {
                    "timestamp": ts,
                    "reaction_time_ms": _to_float(row.get("reaction_time_ms")),
                    "success": _as_bool(row.get("success")),
                    "miss": _as_bool(row.get("miss")),
                    "error": _as_bool(row.get("error")),
                }
            )
    return events


# --------------------------------------------------------------------------- #
# Graph discovery + reading
# --------------------------------------------------------------------------- #
def find_graph_120s_dir(session_dir: Path) -> Optional[Path]:
    direct = session_dir / "data_graph" / GRAPH_120S_DIRNAME
    if direct.is_dir():
        return direct
    hits = [p for p in session_dir.glob(f"**/{GRAPH_120S_DIRNAME}") if p.is_dir()]
    return hits[0] if hits else None


def list_graph_files(graph_dir: Optional[Path]) -> List[Path]:
    if graph_dir is None:
        return []
    return sorted(graph_dir.glob("graph_*.json"))


def extract_graph_window(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Pull (window_start, window_end) from a graph JSON, tolerating layouts."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None

    def grab(obj: Any) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(obj, dict):
            return None, None
        ws = _to_float(obj.get("window_start"))
        we = _to_float(obj.get("window_end"))
        return ws, we

    for container in (d.get("window"), d.get("graph"), d):
        ws, we = grab(container)
        if ws is not None or we is not None:
            return ws, we
    return None, None


def session_graph_window_bounds(graph_files: List[Path]) -> Tuple[Optional[float], Optional[float], int]:
    starts: List[float] = []
    ends: List[float] = []
    for p in graph_files:
        ws, we = extract_graph_window(p)
        if ws is not None:
            starts.append(ws)
        if we is not None:
            ends.append(we)
    return (min(starts) if starts else None, max(ends) if ends else None, len(graph_files))


# --------------------------------------------------------------------------- #
# PART 1 — raw inventory
# --------------------------------------------------------------------------- #
def build_raw_inventory(
    raw_dir: Path, session_to_user: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sessions = sorted([p for p in raw_dir.glob("session_*") if p.is_dir()], key=lambda p: p.name)
    inventory: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    raw_index: Dict[str, Dict[str, Any]] = {}

    def has(name: str, sdir: Path) -> bool:
        return (sdir / name).exists() or (sdir / "raw" / name).exists() or bool(list(sdir.glob(f"**/{name}")))

    def device_id_for(sdir: Path) -> str:
        for cand in ["raw/labels.csv", "raw/behavior.csv", "labels/labels.csv", "raw/nasa_tlx.csv"]:
            p = sdir / cand
            if p.exists():
                row = next(csv.DictReader(p.open("r", encoding="utf-8", newline="")), None)
                if row and row.get("device_id"):
                    return str(row["device_id"])
        return ""

    for sdir in sessions:
        sid = sdir.name
        cands = find_dual_task_candidates(sdir)
        for c in cands:
            candidate_rows.append({"session_id": sid, **c})
        selected = select_dual_task_file(cands)
        events = read_dual_task_events(Path(selected["path"])) if selected else []
        dt_ts = [e["timestamp"] for e in events]
        dt_min = min(dt_ts) if dt_ts else None
        dt_max = max(dt_ts) if dt_ts else None

        graph_dir = find_graph_120s_dir(sdir)
        graph_files = list_graph_files(graph_dir)
        g_start, g_end, g_count = session_graph_window_bounds(graph_files)

        total_csv = len(list(sdir.glob("**/*.csv")))
        total_json = len(list(sdir.glob("**/*.json")))

        rec = {
            "session_id": sid,
            "session_path": str(sdir),
            "user_id_processed": session_to_user.get(sid, ""),
            "device_id_raw": device_id_for(sdir),
            "has_behavior_csv": has("behavior.csv", sdir),
            "has_keyboard_csv": has("keyboard.csv", sdir),
            "has_mouse_csv": has("mouse.csv", sdir),
            "has_labels_csv": has("labels.csv", sdir),
            "has_nasa_tlx": has("nasa_tlx.csv", sdir) or has("labels.csv", sdir),
            "has_dual_task_csv": selected is not None,
            "selected_dual_task_path": selected["path"] if selected else "",
            "dual_task_candidate_count": len(cands),
            "dual_task_event_count": len(events),
            "dual_task_ts_min": dt_min,
            "dual_task_ts_max": dt_max,
            "dual_task_ts_unit_guess": guess_ts_unit(dt_min, dt_max),
            "graph_dir": str(graph_dir) if graph_dir else "",
            "graph_120s_count": g_count,
            "graph_window_start_min": g_start,
            "graph_window_end_max": g_end,
            "total_csv_files": total_csv,
            "total_graph_json_files": len(list((graph_dir.glob("graph_*.json"))) if graph_dir else []),
            "total_json_files": total_json,
        }
        inventory.append(rec)
        raw_index[sid] = {
            "record": rec,
            "events": events,
            "graph_files": graph_files,
            "graph_dir": graph_dir,
        }
    return inventory, candidate_rows, raw_index


# --------------------------------------------------------------------------- #
# PART 2 — processed inventory
# --------------------------------------------------------------------------- #
def load_processed(dt_dir: Path) -> Dict[str, Any]:
    tucker = np.load(dt_dir / "tucker_slices.npy")
    nasa = np.load(dt_dir / "nasa_tlx_labels.npy")
    metadata = json.loads((dt_dir / "metadata.json").read_text(encoding="utf-8"))

    dtwl_path = dt_dir / "dual_task_window_labels.csv"
    dtwl: List[Dict[str, Any]] = []
    if dtwl_path.exists():
        dtwl = list(csv.DictReader(dtwl_path.open("r", encoding="utf-8", newline="")))
    return {"tucker": tucker, "nasa": nasa, "metadata": metadata, "dtwl": dtwl, "dtwl_path": dtwl_path}


def processed_inventory_rows(proc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    metadata = proc["metadata"]
    tucker = proc["tucker"]
    nasa = proc["nasa"]
    dtwl = proc["dtwl"]
    n = len(metadata)

    missing = {k: 0 for k in ["user_id", "session_id", "window_idx", "window_start", "window_end"]}
    starts, ends = [], []
    per_user: Dict[str, int] = {}
    per_session: Dict[str, int] = {}
    for r in metadata:
        for k in missing:
            if r.get(k) is None:
                missing[k] += 1
        u = r.get("user_id", "")
        s = r.get("session_id", "")
        per_user[u] = per_user.get(u, 0) + 1
        per_session[s] = per_session.get(s, 0) + 1
        ws = _to_float(r.get("window_start"))
        we = _to_float(r.get("window_end"))
        if ws is not None:
            starts.append(ws)
        if we is not None:
            ends.append(we)

    inv = [{
        "n_metadata_rows": n,
        "tucker_shape": str(tuple(tucker.shape)),
        "tucker_rows": int(tucker.shape[0]),
        "nasa_shape": str(tuple(nasa.shape)),
        "nasa_rows": int(nasa.shape[0]),
        "dual_task_window_labels_rows": len(dtwl),
        "metadata_eq_tucker": n == tucker.shape[0],
        "metadata_eq_nasa": n == nasa.shape[0],
        "metadata_eq_dtwl": (n == len(dtwl)) if dtwl else None,
        "n_unique_users": len(per_user),
        "n_unique_sessions": len(per_session),
        "window_start_min": min(starts) if starts else None,
        "window_start_max": max(starts) if starts else None,
        "window_end_min": min(ends) if ends else None,
        "window_end_max": max(ends) if ends else None,
        "missing_user_id": missing["user_id"],
        "missing_session_id": missing["session_id"],
        "missing_window_idx": missing["window_idx"],
        "missing_window_start": missing["window_start"],
        "missing_window_end": missing["window_end"],
    }]

    user_rows = [{"user_id": u, "windows": c} for u, c in sorted(per_user.items(), key=lambda kv: -kv[1])]
    sess_rows = [{"session_id": s, "windows": c} for s, c in sorted(per_session.items())]
    extras = {"per_user": per_user, "per_session": per_session}
    return inv, user_rows, sess_rows, extras


# --------------------------------------------------------------------------- #
# PART 6 helpers — dual-task matching modes
# --------------------------------------------------------------------------- #
def count_modes(
    ts_list: List[float],
    window_start: float,
    window_end: float,
    session_min_ws: float,
    dt_ts_min: Optional[float],
) -> Dict[str, int]:
    arr = np.asarray(ts_list, dtype=np.float64) if ts_list else np.empty(0)

    def between(lo: float, hi: float) -> int:
        if arr.size == 0:
            return 0
        return int(np.sum((arr >= lo) & (arr < hi)))

    direct = between(window_start, window_end)
    ms = between(window_start * 1000.0, window_end * 1000.0)
    rel = between(window_start - session_min_ws, window_end - session_min_ws)
    if dt_ts_min is not None:
        shift_to = dt_ts_min - session_min_ws
        shifted_to = between(window_start + shift_to, window_end + shift_to)
        shift_from = session_min_ws - dt_ts_min
        shifted_from = between(window_start - shift_from, window_end - shift_from)
    else:
        shifted_to = shifted_from = 0
    return {
        "direct": direct,
        "ms": ms,
        "relative": rel,
        "shifted_to_dual_task_min": shifted_to,
        "shifted_from_dual_task_min": shifted_from,
    }


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit data/ -> data_training/ alignment (read-only).")
    ap.add_argument("--data-training-dir", type=Path, default=Path("data_training"))
    ap.add_argument("--raw-data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    ap.add_argument("--timestamp-tolerance", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tol = args.timestamp_tolerance

    proc = load_processed(args.data_training_dir)
    metadata = proc["metadata"]
    session_to_user = {r.get("session_id", ""): r.get("user_id", "") for r in metadata}

    # ---- PART 1 ----
    raw_inv, dt_cands, raw_index = build_raw_inventory(args.raw_data_dir, session_to_user)
    write_csv(out / "raw_session_inventory.csv", raw_inv)
    write_csv(out / "dual_task_file_candidates.csv", dt_cands,
              fieldnames=["session_id", "path", "relative_path", "priority_rank", "event_count", "size_bytes"])

    # ---- PART 2 ----
    proc_inv, user_rows, sess_rows, extras = processed_inventory_rows(proc)
    write_csv(out / "processed_metadata_inventory.csv", proc_inv)
    write_csv(out / "processed_user_summary.csv", user_rows)
    write_csv(out / "processed_session_summary.csv", sess_rows)
    per_session = extras["per_session"]

    # Precompute processed dual-task availability per session from dtwl.
    dtwl = proc["dtwl"]
    proc_dt_avail_by_session: Dict[str, int] = {}
    proc_dt_rows_by_session: Dict[str, int] = {}
    for r in dtwl:
        s = r.get("session_id", "")
        proc_dt_rows_by_session[s] = proc_dt_rows_by_session.get(s, 0) + 1
        if str(r.get("dual_task_available", "")).strip().lower() == "true":
            proc_dt_avail_by_session[s] = proc_dt_avail_by_session.get(s, 0) + 1

    raw_session_ids = {r["session_id"] for r in raw_inv}
    proc_session_ids = set(per_session.keys())

    # ---- PART 3 ----
    all_sessions = sorted(raw_session_ids | proc_session_ids)
    coverage_rows: List[Dict[str, Any]] = []
    for sid in all_sessions:
        raw_present = sid in raw_session_ids
        proc_present = sid in proc_session_ids
        rawrec = raw_index.get(sid, {}).get("record", {}) if raw_present else {}
        raw_graph = rawrec.get("graph_120s_count", 0)
        proc_win = per_session.get(sid, 0)
        raw_dt_found = bool(rawrec.get("has_dual_task_csv", False))
        raw_dt_events = int(rawrec.get("dual_task_event_count", 0))
        proc_dt_avail = proc_dt_avail_by_session.get(sid, 0)
        proc_dt_total = proc_dt_rows_by_session.get(sid, 0)
        cov = (proc_dt_avail / proc_win) if proc_win else 0.0

        if not proc_present:
            status = "missing_from_processed"
        elif not raw_present:
            status = "processed_session_missing_in_raw"
        elif not raw_dt_found:
            status = "no_raw_dual_task_file"
        elif raw_dt_events == 0:
            status = "raw_dual_task_empty"
        elif raw_graph != proc_win and proc_win > 0:
            status = "graph_count_mismatch"
        elif proc_dt_avail == 0:
            status = "zero_dual_task_coverage"
        elif cov < 0.5:
            status = "low_dual_task_coverage"
        else:
            status = "ok"

        coverage_rows.append({
            "session_id": sid,
            "raw_present": raw_present,
            "processed_present": proc_present,
            "raw_graph_count": raw_graph,
            "processed_window_count": proc_win,
            "raw_dual_task_file_found": raw_dt_found,
            "raw_dual_task_event_count": raw_dt_events,
            "processed_dual_task_available_count": proc_dt_avail,
            "processed_dual_task_total_rows": proc_dt_total,
            "coverage_ratio": round(cov, 4),
            "status": status,
        })
    write_csv(out / "session_coverage_diff.csv", coverage_rows)

    in_raw_not_proc = sorted(raw_session_ids - proc_session_ids)
    in_proc_not_raw = sorted(proc_session_ids - raw_session_ids)
    # user-level (processed-derived; raw has no user_id)
    proc_users = set(session_to_user.values())
    align_summary = [{
        "raw_sessions_count": len(raw_session_ids),
        "processed_sessions_count": len(proc_session_ids),
        "sessions_in_raw_not_processed": len(in_raw_not_proc),
        "sessions_in_processed_not_raw": len(in_proc_not_raw),
        "sessions_in_raw_not_processed_list": ";".join(in_raw_not_proc),
        "sessions_in_processed_not_raw_list": ";".join(in_proc_not_raw),
        "processed_user_count": len(proc_users),
        "note": "raw sessions expose device_id only; user_id is processed-derived",
    }]
    write_csv(out / "session_alignment_summary.csv", align_summary)

    # ---- PART 4 — timestamp scale check ----
    ts_rows: List[Dict[str, Any]] = []
    # session -> processed window bounds
    sess_meta_bounds: Dict[str, Tuple[float, float, float]] = {}
    sess_meta_windows: Dict[str, List[Tuple[int, float, float]]] = {}
    for i, r in enumerate(metadata):
        s = r.get("session_id", "")
        ws = _to_float(r.get("window_start"))
        we = _to_float(r.get("window_end"))
        if ws is None or we is None:
            continue
        sess_meta_windows.setdefault(s, []).append((i, ws, we))
    for s, wins in sess_meta_windows.items():
        starts = [w[1] for w in wins]
        ends = [w[2] for w in wins]
        sess_meta_bounds[s] = (min(starts), max(ends), min(starts))

    for sid in sorted(proc_session_ids):
        wins = sess_meta_windows.get(sid, [])
        rawrec = raw_index.get(sid, {})
        events = rawrec.get("events", []) if rawrec else []
        ts = [e["timestamp"] for e in events]
        if not wins:
            ts_rows.append({"session_id": sid, "suspected_issue": "no_processed_windows",
                            "best_overlap_mode": "", "best_overlap_count": 0})
            continue
        if not ts:
            ts_rows.append({"session_id": sid, "suspected_issue": "no_dual_task",
                            "metadata_start_min": min(w[1] for w in wins),
                            "metadata_end_max": max(w[2] for w in wins),
                            "best_overlap_mode": "", "best_overlap_count": 0})
            continue

        m_start = min(w[1] for w in wins)
        m_end = max(w[2] for w in wins)
        session_min_ws = m_start
        dt_min, dt_max = min(ts), max(ts)

        totals = {"direct": 0, "ms": 0, "relative": 0, "shifted_to_dual_task_min": 0, "shifted_from_dual_task_min": 0}
        for (_, ws, we) in wins:
            c = count_modes(ts, ws, we, session_min_ws, dt_min)
            for k in totals:
                totals[k] += c[k]
        best_mode = max(totals, key=lambda k: totals[k])
        best_count = totals[best_mode]

        if best_count == 0:
            suspected = "no_overlap"
        elif best_mode == "direct":
            suspected = "none"
        elif best_mode == "ms":
            suspected = "ms_vs_seconds"
        elif best_mode == "relative":
            suspected = "relative_time"
        else:
            suspected = "shifted_origin"

        ts_rows.append({
            "session_id": sid,
            "metadata_start_min": m_start,
            "metadata_end_max": m_end,
            "dual_task_ts_min": dt_min,
            "dual_task_ts_max": dt_max,
            "direct_overlap_event_count": totals["direct"],
            "ms_overlap_event_count": totals["ms"],
            "relative_overlap_event_count": totals["relative"],
            "shifted_overlap_event_count": max(totals["shifted_to_dual_task_min"], totals["shifted_from_dual_task_min"]),
            "best_overlap_mode": best_mode,
            "best_overlap_count": best_count,
            "suspected_issue": suspected,
        })
    write_csv(out / "timestamp_alignment_report.csv", ts_rows)

    # ---- PART 5 — graph vs metadata alignment ----
    graph_rows: List[Dict[str, Any]] = []
    graph_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[float], str]]] = {}

    def graph_lookup(sid: str) -> Dict[str, Tuple[Optional[float], Optional[float], str]]:
        if sid in graph_cache:
            return graph_cache[sid]
        m: Dict[str, Tuple[Optional[float], Optional[float], str]] = {}
        files = raw_index.get(sid, {}).get("graph_files", []) if raw_index.get(sid) else []
        for p in files:
            ws, we = extract_graph_window(p)
            m[p.stem] = (ws, we, str(p))  # stem e.g. 'graph_001'
        graph_cache[sid] = m
        return m

    g_found = g_aligned = g_missing = g_bigmiss = 0
    for i, r in enumerate(metadata):
        sid = r.get("session_id", "")
        widx = r.get("window_idx")
        m_ws = _to_float(r.get("window_start"))
        m_we = _to_float(r.get("window_end"))
        gmap = graph_lookup(sid)
        # try graph_{idx+1} then graph_{idx}
        gpath = ""
        gws = gwe = None
        if isinstance(widx, int):
            for cand in (f"graph_{widx + 1:03d}", f"graph_{widx:03d}"):
                if cand in gmap:
                    gws, gwe, gpath = gmap[cand]
                    break
        found = gpath != ""
        if found:
            g_found += 1
        else:
            g_missing += 1
        sdiff = abs(m_ws - gws) if (m_ws is not None and gws is not None) else None
        ediff = abs(m_we - gwe) if (m_we is not None and gwe is not None) else None
        aligned = bool(sdiff is not None and ediff is not None and sdiff <= tol and ediff <= tol)
        if aligned:
            g_aligned += 1
        if (sdiff is not None and sdiff > tol) or (ediff is not None and ediff > tol):
            g_bigmiss += 1
        graph_rows.append({
            "row_idx": i,
            "user_id": r.get("user_id", ""),
            "session_id": sid,
            "window_idx": widx,
            "graph_found": found,
            "graph_path": gpath,
            "metadata_window_start": m_ws,
            "metadata_window_end": m_we,
            "graph_window_start": gws,
            "graph_window_end": gwe,
            "abs_start_diff": sdiff,
            "abs_end_diff": ediff,
            "aligned": aligned,
        })
    write_csv(out / "graph_metadata_alignment.csv", graph_rows)
    graph_summary = {
        "total_rows": len(metadata),
        "graph_found_count": g_found,
        "graph_found_ratio": round(g_found / max(len(metadata), 1), 4),
        "timestamp_aligned_count": g_aligned,
        "timestamp_aligned_ratio": round(g_aligned / max(len(metadata), 1), 4),
        "missing_graph_count": g_missing,
        "large_timestamp_mismatch_count": g_bigmiss,
    }

    # ---- PART 6 — dual-task window match debug ----
    dtwl_by_rowidx: Dict[int, Dict[str, Any]] = {}
    for r in dtwl:
        try:
            dtwl_by_rowidx[int(r["row_idx"])] = r
        except (KeyError, ValueError):
            pass

    debug_rows: List[Dict[str, Any]] = []
    mode_totals = {"current_available": 0, "direct": 0, "ms": 0, "relative": 0,
                   "shifted_to_dual_task_min": 0, "shifted_from_dual_task_min": 0}
    for i, r in enumerate(metadata):
        sid = r.get("session_id", "")
        widx = r.get("window_idx")
        m_ws = _to_float(r.get("window_start"))
        m_we = _to_float(r.get("window_end"))
        events = raw_index.get(sid, {}).get("events", []) if raw_index.get(sid) else []
        ts = [e["timestamp"] for e in events]
        bounds = sess_meta_bounds.get(sid)
        session_min_ws = bounds[2] if bounds else (m_ws if m_ws is not None else 0.0)
        dt_min = min(ts) if ts else None
        cur_avail = 0
        if i in dtwl_by_rowidx:
            cur_avail = 1 if str(dtwl_by_rowidx[i].get("dual_task_available", "")).strip().lower() == "true" else 0
        if m_ws is None or m_we is None:
            counts = {"direct": 0, "ms": 0, "relative": 0, "shifted_to_dual_task_min": 0, "shifted_from_dual_task_min": 0}
        else:
            counts = count_modes(ts, m_ws, m_we, session_min_ws, dt_min)
        best_mode = max(counts, key=lambda k: counts[k]) if any(counts.values()) else "none"
        best_count = counts.get(best_mode, 0) if best_mode != "none" else 0

        mode_totals["current_available"] += cur_avail
        for k in ["direct", "ms", "relative", "shifted_to_dual_task_min", "shifted_from_dual_task_min"]:
            mode_totals[k] += counts[k]

        debug_rows.append({
            "row_idx": i,
            "user_id": r.get("user_id", ""),
            "session_id": sid,
            "window_idx": widx,
            "window_start": m_ws,
            "window_end": m_we,
            "current_available": cur_avail,
            "direct_count": counts["direct"],
            "ms_count": counts["ms"],
            "relative_count": counts["relative"],
            "shifted_to_dual_task_min_count": counts["shifted_to_dual_task_min"],
            "shifted_from_dual_task_min_count": counts["shifted_from_dual_task_min"],
            "best_mode": best_mode,
            "best_count": best_count,
        })
    write_csv(out / "dual_task_matching_debug.csv", debug_rows)
    write_csv(out / "dual_task_matching_mode_summary.csv", [{
        "total_current_available": mode_totals["current_available"],
        "total_direct_matched": mode_totals["direct"],
        "total_ms_matched": mode_totals["ms"],
        "total_relative_matched": mode_totals["relative"],
        "total_shifted_to_matched": mode_totals["shifted_to_dual_task_min"],
        "total_shifted_from_matched": mode_totals["shifted_from_dual_task_min"],
    }])

    # ---- PART 7 — zero coverage users ----
    zero_rows: List[Dict[str, Any]] = []
    for user in ZERO_COVERAGE_USERS:
        user_sessions_proc = sorted({s for s, u in session_to_user.items() if u == user})
        # raw sessions for this user can only be inferred via processed mapping
        raw_found = [s for s in user_sessions_proc if s in raw_session_ids]
        proc_windows = sum(per_session.get(s, 0) for s in user_sessions_proc)
        dt_files = 0
        dt_events = 0
        dt_ranges = []
        meta_ranges = []
        graph_counts = 0
        mode_acc = {"direct": 0, "ms": 0, "relative": 0, "shifted_to_dual_task_min": 0, "shifted_from_dual_task_min": 0}
        for s in user_sessions_proc:
            rec = raw_index.get(s, {})
            r0 = rec.get("record", {}) if rec else {}
            if r0.get("has_dual_task_csv"):
                dt_files += 1
            dt_events += int(r0.get("dual_task_event_count", 0))
            if r0.get("dual_task_ts_min") is not None:
                dt_ranges.append(f"{r0['dual_task_ts_min']:.0f}-{r0['dual_task_ts_max']:.0f}")
            graph_counts += int(r0.get("graph_120s_count", 0))
            bounds = sess_meta_bounds.get(s)
            if bounds:
                meta_ranges.append(f"{bounds[0]:.0f}-{bounds[1]:.0f}")
            events = rec.get("events", []) if rec else []
            ts = [e["timestamp"] for e in events]
            for (_, ws, we) in sess_meta_windows.get(s, []):
                c = count_modes(ts, ws, we, bounds[2] if bounds else ws, min(ts) if ts else None)
                for k in mode_acc:
                    mode_acc[k] += c[k]

        # likely reason
        if not user_sessions_proc:
            reason = "user not present in processed metadata"
        elif dt_files == 0:
            reason = "no dual_task.csv found in user's sessions"
        elif dt_events == 0:
            reason = "dual_task.csv present but empty"
        elif max(mode_acc.values()) == 0:
            reason = "windows do not overlap dual_task events under any timestamp mode (real gap)"
        elif mode_acc["direct"] == 0 and max(mode_acc.values()) > 0:
            reason = "timestamp mismatch (overlap only under non-direct mode)"
        else:
            reason = "unknown / events exist and overlap (coverage may be partial)"

        zero_rows.append({
            "user_id": user,
            "raw_sessions_found": len(raw_found),
            "processed_sessions_found": len(user_sessions_proc),
            "processed_windows": proc_windows,
            "raw_dual_task_files_found": dt_files,
            "raw_dual_task_event_count": dt_events,
            "raw_dual_task_ts_ranges": ";".join(dt_ranges),
            "metadata_window_ts_ranges": ";".join(meta_ranges),
            "graph_120s_counts": graph_counts,
            "match_direct": mode_acc["direct"],
            "match_ms": mode_acc["ms"],
            "match_relative": mode_acc["relative"],
            "match_shifted_to": mode_acc["shifted_to_dual_task_min"],
            "match_shifted_from": mode_acc["shifted_from_dual_task_min"],
            "likely_reason": reason,
        })
    write_csv(out / "zero_coverage_user_report.csv", zero_rows)

    # ---- PART 8 — row alignment checks ----
    tucker = proc["tucker"]
    nasa = proc["nasa"]
    n_meta = len(metadata)
    row_rows: List[Dict[str, Any]] = []
    mismatches = 0
    for i, r in enumerate(metadata):
        dt = dtwl_by_rowidx.get(i)
        dt_sid = dt.get("session_id") if dt else ""
        dt_widx = dt.get("window_idx") if dt else ""
        dt_uid = dt.get("user_id") if dt else ""
        matches = bool(dt and str(dt_sid) == str(r.get("session_id")) and
                       str(dt_uid) == str(r.get("user_id")) and
                       str(dt_widx) == str(r.get("window_idx")))
        if dt and not matches:
            mismatches += 1
        row_rows.append({
            "row_idx": i,
            "metadata_session_id": r.get("session_id", ""),
            "metadata_window_idx": r.get("window_idx"),
            "metadata_user_id": r.get("user_id", ""),
            "dtwl_session_id": dt_sid,
            "dtwl_window_idx": dt_widx,
            "dtwl_user_id": dt_uid,
            "row_matches_dual_task_labels": matches if dt else None,
            "tucker_row_exists": i < tucker.shape[0],
            "nasa_label_row_exists": i < nasa.shape[0],
        })
    write_csv(out / "row_alignment_check.csv", row_rows)

    # ---- PART 9 — probe count reality check ----
    probe_rows: List[Dict[str, Any]] = []
    for mode in ["direct", "ms", "relative", "shifted_to_dual_task_min", "shifted_from_dual_task_min"]:
        key = {"direct": "direct_count", "ms": "ms_count", "relative": "relative_count",
               "shifted_to_dual_task_min": "shifted_to_dual_task_min_count",
               "shifted_from_dual_task_min": "shifted_from_dual_task_min_count"}[mode]
        counts = [d[key] for d in debug_rows]
        nonzero = [c for c in counts if c > 0]
        probe_rows.append({
            "mode": mode,
            "windows_with_0_probes": sum(1 for c in counts if c == 0),
            "windows_with_1_probe": sum(1 for c in counts if c == 1),
            "windows_with_2plus_probes": sum(1 for c in counts if c >= 2),
            "max_probes_per_window": max(counts) if counts else 0,
            "mean_probes_per_labelled_window": round(float(np.mean(nonzero)), 4) if nonzero else 0.0,
            "labelled_windows": len(nonzero),
        })
    write_csv(out / "probe_count_reality_check.csv", probe_rows)

    # ---- console summary ----
    summary = {
        "raw_sessions": len(raw_session_ids),
        "processed_sessions": len(proc_session_ids),
        "sessions_in_raw_not_processed": len(in_raw_not_proc),
        "metadata_rows": n_meta,
        "metadata_eq_tucker": proc_inv[0]["metadata_eq_tucker"],
        "metadata_eq_nasa": proc_inv[0]["metadata_eq_nasa"],
        "metadata_eq_dtwl": proc_inv[0]["metadata_eq_dtwl"],
        "row_alignment_mismatches": mismatches,
        "graph_alignment": graph_summary,
        "dual_task_mode_totals": mode_totals,
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nAudit CSVs written to: {out}")


if __name__ == "__main__":
    main()
