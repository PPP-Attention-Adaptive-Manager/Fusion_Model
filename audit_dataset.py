"""
audit_dataset.py  (v2 — recursive session finder)
==================================================
Reads every session across all users without skipping anything.
Handles two folder layouts:
  Layout A: user/session_*/           (most users)
  Layout B: user/data/session_*/      (oussama, gassarra)

Two graph formats:
  Format A (JSON):  data_graph/data_graph_120s/graph_*.json    (per-window)
  Format B (CSV):   graph/windows/120s/{nodes,edges}.csv       (aggregate)

Usage:
    cd ~/fusion_model
    python audit_dataset.py \
        --data_root "~/aam_data_6_1/Data/After update" \
        --output_dir ~/tucker_outputs/audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NASA_TLX_COLS = [
    "mental_demand", "physical_demand", "temporal_demand",
    "performance", "effort", "frustration",
    "stress_self_report", "valence", "arousal",
]


# ─────────────────────────────────────────────────────────────────────────────
# Session discovery — recursive
# ─────────────────────────────────────────────────────────────────────────────

def find_sessions(user_dir: Path) -> list[Path]:
    """
    Recursively find all session directories under a user folder.
    Handles Layout A (user/session_*/) and Layout B (user/data/session_*/).
    A session dir is any folder whose name starts with 'session_'.
    Excludes session dirs that are nested inside another session dir.
    """
    results = []
    for p in sorted(user_dir.rglob("session_*")):
        if not p.is_dir():
            continue
        # skip if any ancestor (relative to user_dir) is itself a session dir
        rel_parts = p.relative_to(user_dir).parts
        if any(part.startswith("session_") for part in rel_parts[:-1]):
            continue
        results.append(p)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-session field extractors
# ─────────────────────────────────────────────────────────────────────────────

def try_load_nasa_tlx(session_path: Path) -> dict:
    result = {c: None for c in NASA_TLX_COLS}
    result["label_file"] = None

    candidates = [
        session_path / "raw"    / "labels.csv",
        session_path / "labels" / "nasa_tlx.csv",
        session_path / "raw"    / "nasa_tlx.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if df.empty:
                result["label_file"] = str(p) + " (empty)"
                continue
            row = df.iloc[0]
            found_any = False
            for c in NASA_TLX_COLS:
                if c in row and pd.notna(row[c]):
                    result[c] = float(row[c])
                    found_any = True
            if found_any:
                result["label_file"] = str(p)
                return result
            result["label_file"] = str(p) + " (columns missing)"
        except Exception as e:
            result["label_file"] = str(p) + f" (error: {e})"
    return result


def try_load_dual_task(session_path: Path) -> dict:
    result = {
        "dual_task_file":        None,
        "dual_task_n_probes":    None,
        "dual_task_mean_rt_ms":  None,
        "dual_task_success_rate":None,
    }
    candidates = [
        session_path / "raw"    / "dual_task.csv",
        session_path / "labels" / "dual_task.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if df.empty:
                result["dual_task_file"] = str(p) + " (empty)"
                continue
            result["dual_task_file"]    = str(p)
            result["dual_task_n_probes"]= len(df)
            if "reaction_time_ms" in df.columns:
                valid = df["reaction_time_ms"].dropna()
                result["dual_task_mean_rt_ms"] = float(valid.mean()) if len(valid) > 0 else None
            if "success" in df.columns:
                result["dual_task_success_rate"] = float(df["success"].mean())
            return result
        except Exception as e:
            result["dual_task_file"] = str(p) + f" (error: {e})"
    return result


def try_load_graph_info(session_path: Path) -> dict:
    result = {
        "graphs_120s_count": None,
        "graphs_30s_count":  None,
        "graphs_5s_count":   None,
        "graph_format":      None,
    }

    # Format A: data_graph/data_graph_{size}/graph_*.json  (per-window JSON)
    json_found = False
    for size in ["120s", "30s", "5s"]:
        d = session_path / "data_graph" / f"data_graph_{size}"
        if d.exists():
            count = len(list(d.glob("graph_*.json")))
            result[f"graphs_{size}_count"] = count
            if count > 0:
                json_found = True
    if json_found:
        result["graph_format"] = "json"
        return result

    # Format B: graph/windows/{size}/{nodes,edges}.csv  (aggregate CSV)
    csv_found = False
    for size in ["120s", "30s", "5s"]:
        d = session_path / "graph" / "windows" / size
        nodes_f = d / "nodes.csv"
        edges_f = d / "edges.csv"
        if nodes_f.exists() and edges_f.exists():
            try:
                n_nodes = len(pd.read_csv(nodes_f))
                n_edges = len(pd.read_csv(edges_f))
                if n_nodes > 0:
                    # CSV format = one aggregate graph per session
                    # we report it as "1 window" for the purpose of the audit
                    result[f"graphs_{size}_count"] = 1
                    csv_found = True
            except Exception:
                pass
    if csv_found:
        result["graph_format"] = "csv_aggregate"

    return result


def try_load_raw_csv_info(session_path: Path) -> dict:
    result = {}
    raw_dir = session_path / "raw"
    for fname in ["mouse.csv", "keyboard.csv", "notification.csv"]:
        key = fname.replace(".csv", "_rows")
        fpath = raw_dir / fname
        if not fpath.exists():
            result[key] = None
            continue
        try:
            df = pd.read_csv(fpath)
            result[key] = len(df) if not df.empty else 0
        except Exception:
            result[key] = -1
    return result


def try_load_session_duration(session_path: Path) -> dict:
    result = {"session_duration_minutes": None}
    mouse_csv = session_path / "raw" / "mouse.csv"
    if not mouse_csv.exists():
        return result
    try:
        df = pd.read_csv(mouse_csv, usecols=["timestamp"])
        if not df.empty and len(df) > 1:
            duration_s = float(df["timestamp"].max() - df["timestamp"].min())
            result["session_duration_minutes"] = round(duration_s / 60.0, 1)
    except Exception:
        pass
    return result


def audit_session(session_path: Path, user_id: str) -> dict:
    row = {
        "user_id":      user_id,
        "session_id":   session_path.name,
        "session_path": str(session_path),
    }
    row.update(try_load_nasa_tlx(session_path))
    row.update(try_load_dual_task(session_path))
    row.update(try_load_graph_info(session_path))
    row.update(try_load_raw_csv_info(session_path))
    row.update(try_load_session_duration(session_path))

    has_label  = all(row.get(c) is not None for c in NASA_TLX_COLS)
    has_graphs = (row.get("graphs_120s_count") or 0) > 0
    has_mouse  = (row.get("mouse_rows") or 0) > 10
    has_kb     = (row.get("keyboard_rows") or 0) > 10

    row["has_nasa_tlx"]    = has_label
    row["has_graphs_120s"] = has_graphs
    row["has_mouse"]       = has_mouse
    row["has_keyboard"]    = has_kb
    row["has_dual_task"]   = (row.get("dual_task_n_probes") or 0) > 0
    row["usable"]          = has_label and has_graphs
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str,
        default=str(Path.home() / "aam_data_6_1" / "Data" / "After update"))
    parser.add_argument("--output_dir", type=str,
        default=str(Path.home() / "tucker_outputs" / "audit"))
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"Found {len(user_dirs)} user(s): {[d.name for d in user_dirs]}\n")

    all_rows = []

    for user_dir in user_dirs:
        user_id  = user_dir.name
        sessions = find_sessions(user_dir)
        print(f"User: {user_id}  ({len(sessions)} sessions)")

        for session_dir in sessions:
            row = audit_session(session_dir, user_id)
            all_rows.append(row)

            status   = "✓ usable" if row["usable"] else "✗ skip"
            reasons  = []
            if not row["has_nasa_tlx"]:    reasons.append("no NASA-TLX")
            if not row["has_graphs_120s"]: reasons.append("no graphs")
            if not row["has_mouse"]:       reasons.append("no mouse")
            reason_str = f"  [{', '.join(reasons)}]" if reasons else ""
            duration   = f"  {row['session_duration_minutes']}min" if row["session_duration_minutes"] else ""
            graphs     = f"  graphs_120s={row['graphs_120s_count']}({row['graph_format']})" \
                         if row["graphs_120s_count"] else ""
            dt         = f"  dual_task={row['dual_task_n_probes']}p" if row["has_dual_task"] else ""
            print(f"  {status}  {session_dir.name}{duration}{graphs}{dt}{reason_str}")

    df = pd.DataFrame(all_rows)
    audit_csv = output_dir / "dataset_audit.csv"
    df.to_csv(audit_csv, index=False)

    total     = len(df)
    usable    = int(df["usable"].sum())
    has_label = int(df["has_nasa_tlx"].sum())
    has_graph = int(df["has_graphs_120s"].sum())
    has_dt    = int(df["has_dual_task"].sum())
    has_mouse = int(df["has_mouse"].sum())
    has_kb    = int(df["has_keyboard"].sum())

    json_sessions = int((df["graph_format"] == "json").sum())
    csv_sessions  = int((df["graph_format"] == "csv_aggregate").sum())

    user_summary = df.groupby("user_id").agg(
        sessions_total   =("session_id",        "count"),
        sessions_usable  =("usable",            "sum"),
        has_label        =("has_nasa_tlx",       "sum"),
        has_graphs       =("has_graphs_120s",    "sum"),
        has_dual_task    =("has_dual_task",      "sum"),
        total_graphs_120s=("graphs_120s_count",  "sum"),
    ).reset_index()

    lines = [
        "=" * 65,
        "DATASET AUDIT SUMMARY  (v2 — recursive session finder)",
        "=" * 65,
        f"Total sessions:               {total}",
        f"Usable (label + graphs):      {usable}",
        f"Has NASA-TLX:                 {has_label}",
        f"Has 120s graphs:              {has_graph}",
        f"  — JSON format (per-window): {json_sessions}",
        f"  — CSV format (aggregate):   {csv_sessions}",
        f"Has dual task probes:         {has_dt}",
        f"Has mouse data:               {has_mouse}",
        f"Has keyboard data:            {has_kb}",
        "",
        "Per-user breakdown:",
        user_summary.to_string(index=False),
        "",
        "NASA-TLX present but no graphs (recoverable if graphs regenerated):",
    ]
    for _, r in df[df["has_nasa_tlx"] & ~df["has_graphs_120s"]].iterrows():
        lines.append(f"  {r['user_id']} / {r['session_id']}")

    lines += ["", "Unlabeled sessions with graphs (can't use for supervised training):"]
    for _, r in df[~df["has_nasa_tlx"] & df["has_graphs_120s"]].iterrows():
        lines.append(f"  {r['user_id']} / {r['session_id']}")

    lines += ["", "Sessions with dual task data:"]
    for _, r in df[df["has_dual_task"]].iterrows():
        rt = f"  mean_rt={r['dual_task_mean_rt_ms']:.0f}ms" if r["dual_task_mean_rt_ms"] else ""
        lines.append(
            f"  {r['user_id']} / {r['session_id']}  "
            f"({int(r['dual_task_n_probes'])} probes{rt})"
        )

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = output_dir / "dataset_audit_summary.txt"
    summary_path.write_text(summary)

    print(f"\nFull audit CSV: {audit_csv}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    main()