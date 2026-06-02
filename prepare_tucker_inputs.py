"""
prepare_tucker_inputs.py  (v2 — recursive session finder + CSV graph support)
==============================================================================
Reads every session across all users, handles both folder layouts and both
graph formats. Outputs Tucker slices as sequences (one per session) for GRU
training.

Folder layouts handled:
  Layout A: user/session_*/           (most users)
  Layout B: user/data/session_*/      (oussama, gassarra)

Graph formats handled:
  Format A (JSON):  data_graph/data_graph_120s/graph_*.json  per-window JSON
  Format B (CSV):   graph/windows/120s/{nodes,edges}.csv     aggregate CSV
                    → converted to JSON on the fly, one embedding per session
                    → repeated across all windows (zero-order hold)

Output files (in OUTPUT_DIR):
  sequences.npy        — list of dicts, one per session:
                           X:          (T, 4, 512) float32  Tucker slices
                           y:          (9,)        float32  NASA-TLX
                           user_id:    str
                           session_id: str
                           T:          int
                           window_starts: list[float]

  dataset_info.json    — summary stats + per-session metadata

Usage:
    cd ~/fusion_model
    python prepare_tucker_inputs.py \
        --data_root "~/aam_data_6_1/Data/After update" \
        --output_dir ~/tucker_outputs

Requirements:
    pip install torch pandas numpy onnxruntime torch_geometric
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

NASA_TLX_COLS = [
    "mental_demand", "physical_demand", "temporal_demand",
    "performance", "effort", "frustration",
    "stress_self_report", "valence", "arousal",
]
GRAPH_WINDOW_DIR = "data_graph/data_graph_120s"


# ─────────────────────────────────────────────────────────────────────────────
# Session discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_sessions(user_dir: Path) -> list[Path]:
    """
    Recursively find all session directories under a user folder.
    Handles Layout A and Layout B (extra data/ level).
    """
    results = []
    for p in sorted(user_dir.rglob("session_*")):
        if not p.is_dir():
            continue
        rel_parts = p.relative_to(user_dir).parts
        if any(part.startswith("session_") for part in rel_parts[:-1]):
            continue
        results.append(p)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_tlx(session_path: Path) -> np.ndarray | None:
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
                continue
            row = df.iloc[0]
            vals = []
            for c in NASA_TLX_COLS:
                if c not in row or pd.isna(row[c]):
                    break
                vals.append(float(row[c]))
            if len(vals) == len(NASA_TLX_COLS):
                return np.array(vals, dtype=np.float32)
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Graph loading — two formats
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_jsons(session_path: Path) -> list[dict]:
    """Format A: individual JSON files per 120s window."""
    graph_dir = session_path / GRAPH_WINDOW_DIR
    if not graph_dir.exists():
        return []
    paths = sorted(graph_dir.glob("graph_*.json"))
    graphs = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                graphs.append(json.load(f))
        except Exception:
            continue
    return graphs


def load_csv_graph_as_json(session_path: Path) -> dict | None:
    """
    Format B: aggregate CSV graph → convert to JSON format the switching
    encoder expects. Returns one graph dict representing the whole session.
    """
    nodes_f = session_path / "graph" / "windows" / "120s" / "nodes.csv"
    edges_f = session_path / "graph" / "windows" / "120s" / "edges.csv"

    if not nodes_f.exists() or not edges_f.exists():
        return None
    try:
        nodes_df = pd.read_csv(nodes_f)
        edges_df = pd.read_csv(edges_f)
        if nodes_df.empty:
            return None

        # Build nodes list — use all numeric/boolean columns as features
        non_feature_cols = {"node_id", "id", "session_id", "window_id",
                            "window_start", "window_end", "label",
                            "node_kind", "node_type", "app_name",
                            "window_title", "title", "url", "domain",
                            "path", "path_depth"}
        id_col = "node_id" if "node_id" in nodes_df.columns else \
                 "id"      if "id"      in nodes_df.columns else None
        if id_col is None:
            return None

        feature_cols = [c for c in nodes_df.columns
                       if c not in non_feature_cols
                       and nodes_df[c].dtype in (float, int, "float64", "int64", "bool")]

        nodes = []
        for _, r in nodes_df.iterrows():
            features = {}
            for c in feature_cols:
                v = r[c]
                if pd.notna(v):
                    features[c] = float(v)
            nodes.append({
                "id":       str(r[id_col]),
                "features": features,
            })

        # Build edges list
        edges = []
        if not edges_df.empty:
            src_col = "source" if "source" in edges_df.columns else \
                      "src"    if "src"    in edges_df.columns else None
            tgt_col = "target" if "target" in edges_df.columns else \
                      "tgt"    if "tgt"    in edges_df.columns else None
            w_col   = "weight" if "weight" in edges_df.columns else None

            if src_col and tgt_col:
                for _, r in edges_df.iterrows():
                    edge = {
                        "source": str(r[src_col]),
                        "target": str(r[tgt_col]),
                        "weight": float(r[w_col]) if w_col and pd.notna(r[w_col]) else 1.0,
                    }
                    edges.append(edge)

        return {
            "graph_id":  "csv_aggregate",
            "session_id": session_path.name,
            "window":    {"window_id": "w000000", "window_start": 0, "window_end": 120},
            "nodes":     nodes,
            "edges":     edges,
        }
    except Exception as e:
        log.debug("CSV graph load error: %s", e)
        return None


def get_window_timestamps(graphs: list[dict]) -> list[tuple[float, float]]:
    windows = []
    for g in graphs:
        w     = g.get("window", {})
        start = float(w.get("window_start", 0))
        end   = float(w.get("window_end", start + 120))
        windows.append((start, end))
    return windows


def estimate_windows_from_duration(session_path: Path) -> list[tuple[float, float]]:
    """Fallback: estimate 120s windows from mouse.csv timestamps."""
    mouse_csv = session_path / "raw" / "mouse.csv"
    if not mouse_csv.exists():
        return []
    try:
        df = pd.read_csv(mouse_csv, usecols=["timestamp"])
        if df.empty or len(df) < 2:
            return []
        t_start = float(df["timestamp"].min())
        t_end   = float(df["timestamp"].max())
        windows = []
        t = t_start
        while t + 120.0 <= t_end:
            windows.append((t, t + 120.0))
            t += 120.0
        return windows
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Modality embedding extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_mouse_embeddings(session_path, windows):
    try:
        from pre_embedders.mouse.mouse_encoder import Phase2Pipeline, load_csv
    except ImportError as e:
        log.debug("Mouse import failed: %s", e)
        return [None] * len(windows)

    mouse_csv = session_path / "raw" / "mouse.csv"
    if not mouse_csv.exists():
        return [None] * len(windows)
    try:
        events = load_csv(str(mouse_csv))
    except Exception:
        return [None] * len(windows)

    pipeline = Phase2Pipeline(window_sec=120.0, stride_sec=120.0, device="cpu")
    results  = []
    for (start, end) in windows:
        win_events = [e for e in events if start <= e.timestamp < end]
        if len(win_events) < 10:
            results.append(None)
            continue
        try:
            outputs = pipeline.process_session(win_events, user_id="tmp",
                                               fit_normalizer=True)
            results.append(outputs[-1].detach() if outputs else None)
        except Exception:
            results.append(None)
    return results


def extract_keyboard_embeddings(session_path, windows):
    try:
        from pre_embedders.keyboard import (
            KeystrokeEncoder, parse_csv_events, KeystrokeWindowDataset,
        )
    except ImportError as e:
        log.debug("Keyboard import failed: %s", e)
        return [None] * len(windows)

    kb_csv = session_path / "raw" / "keyboard.csv"
    if not kb_csv.exists():
        return [None] * len(windows)
    try:
        all_rows = pd.read_csv(kb_csv).to_dict("records")
    except Exception:
        return [None] * len(windows)

    encoder = KeystrokeEncoder(hidden_size=64, num_layers=2).eval()
    results = []
    for (start, end) in windows:
        win_rows = [r for r in all_rows if start <= float(r["timestamp"]) < end]
        if len(win_rows) < 10:
            results.append(None)
            continue
        try:
            events  = parse_csv_events(win_rows)
            if len(events) < 10:
                results.append(None)
                continue
            dataset = KeystrokeWindowDataset(events, window_size=20, stride=10)
            if len(dataset) == 0:
                results.append(None)
                continue
            with torch.no_grad():
                emb = encoder(dataset[-1].unsqueeze(0))
            results.append(emb)
        except Exception:
            results.append(None)
    return results


def extract_notif_embeddings(session_path, windows):
    try:
        import onnxruntime  # noqa — availability check
        from pre_embedders.notif.model    import load_model, get_embedding, normalize_features
        from pre_embedders.notif.features import (
            compute_arrival_rate, compute_burstiness, compute_source_entropy,
            compute_disruption_score, compute_npi, WINDOW_SECONDS,
        )
        from TCN_encoders.notif.encoder import NotifBufferedEncoder
    except ImportError as e:
        log.debug("Notif import failed: %s", e)
        return [None] * len(windows)

    notif_csv = session_path / "raw" / "notification.csv"
    try:
        all_rows = pd.read_csv(notif_csv).to_dict("records") \
                   if notif_csv.exists() else []
    except Exception:
        all_rows = []

    SCALER = {
        "min_": [0.0, 0.0, 0.0, 0.0, 0.0],
        "max_": [3.0, 2.509182763787976e-7, 1.0, 0.021615064589633373, 1.0],
    }

    import os
    orig_dir = os.getcwd()
    try:
        os.chdir(Path(__file__).resolve().parent / "pre_embedders" / "notif")
        session = load_model()
    except Exception:
        session = None
    finally:
        os.chdir(orig_dir)

    enc     = NotifBufferedEncoder()
    results = []

    for (start, end) in windows:
        win_rows = []
        for r in all_rows:
            try:
                if start <= float(r["timestamp"]) < end:
                    win_rows.append({
                        "timestamp_arrival": float(r["timestamp"]),
                        "timestamp_action":  None,
                        "app_name":          str(r.get("app_source", "unknown")),
                        "interaction_type":  str(r.get("interaction_type", "added")),
                        "response_time":     (
                            float(r["response_latency_ms"]) / 1000.0
                            if pd.notna(r.get("response_latency_ms")) else None
                        ),
                    })
            except Exception:
                continue

        if session is None or not win_rows:
            emb, _ = enc.step(None)
            results.append(emb)
            continue
        try:
            arr_rate   = compute_arrival_rate(win_rows)
            burstiness = compute_burstiness(win_rows)
            src_ent    = compute_source_entropy(win_rows)
            disrupt    = compute_disruption_score(win_rows)
            added_ts   = [r["timestamp_arrival"] for r in win_rows
                          if r["interaction_type"] == "added"]
            tsl = np.float32((end - max(added_ts)) / WINDOW_SECONDS) \
                  if added_ts else np.float32(1.0)
            raw = np.array([arr_rate, burstiness, src_ent, disrupt, tsl],
                           dtype=np.float32)
            npi    = compute_npi(*[float(x) for x in raw])
            norm   = normalize_features(raw, SCALER)
            emb_v  = get_embedding(session, norm)
            notif_dict = {
                "embedding":        emb_v,
                "npi":              float(npi),
                "burstiness":       float(burstiness),
                "disruption_score": float(disrupt),
            }
            emb, _ = enc.step(notif_dict)
            results.append(emb)
        except Exception:
            emb, _ = enc.step(None)
            results.append(emb)
    return results


def extract_switching_embeddings(session_path, graphs, n_windows):
    """
    For Format A (JSON graphs): one embedding per graph.
    For Format B (CSV aggregate): one embedding, repeated across all windows.
    """
    try:
        from pre_embedders.switching.encoder import SwitchingGraphEncoder
        from TCN_encoders.switching.encoder  import SwitchingIdentityEncoder
    except ImportError as e:
        log.debug("Switching import failed: %s", e)
        return [None] * n_windows

    export_dir = (
        Path(__file__).resolve().parent
        / "pre_embedders" / "switching" / "exports" / "switching_encoder"
    )

    enc = SwitchingIdentityEncoder()

    if not (export_dir / "encoder.pt").exists():
        return [enc.step(None)[0]] * n_windows

    try:
        gnn = SwitchingGraphEncoder(export_dir=export_dir, device="cpu")
    except Exception:
        return [enc.step(None)[0]] * n_windows

    results = []

    if graphs:
        # Format A — one embedding per graph window
        for g in graphs:
            try:
                payload = gnn.get_fusion_input(g)
                emb, _  = enc.step(payload)
                results.append(emb)
            except Exception:
                results.append(enc.step(None)[0])
    else:
        # Format B — one aggregate embedding, repeated
        csv_graph = load_csv_graph_as_json(session_path)
        if csv_graph is not None:
            try:
                payload  = gnn.get_fusion_input(csv_graph)
                base_emb, _ = enc.step(payload)
            except Exception:
                base_emb = enc.step(None)[0]
        else:
            base_emb = enc.step(None)[0]
        results = [base_emb.clone() for _ in range(n_windows)]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tucker fusion
# ─────────────────────────────────────────────────────────────────────────────

def run_tucker(m_emb, k_emb, n_emb, s_emb, tfn) -> np.ndarray:
    D_DIMS = [64, 64, 32, 64]
    embeddings = []
    for emb, d in zip([m_emb, k_emb, n_emb, s_emb], D_DIMS):
        embeddings.append(emb.float() if emb is not None else torch.zeros(1, d))
    with torch.no_grad():
        tensor = tfn(embeddings)
        slices = [tfn.get_slice(tensor, i).squeeze(0).numpy() for i in range(4)]
    return np.stack(slices, axis=0)   # (4, 512)


# ─────────────────────────────────────────────────────────────────────────────
# Session processing
# ─────────────────────────────────────────────────────────────────────────────

def process_session(session_path, user_id, tfn, stats):
    session_id = session_path.name

    nasa_tlx = load_nasa_tlx(session_path)
    if nasa_tlx is None:
        stats["skipped_no_label"] += 1
        return None

    # Try Format A first, then Format B, then duration estimate
    graphs   = load_graph_jsons(session_path)
    csv_mode = False

    if graphs:
        windows = get_window_timestamps(graphs)
    else:
        # Check Format B
        csv_graph = load_csv_graph_as_json(session_path)
        if csv_graph is not None:
            csv_mode = True
            # Estimate windows from mouse timestamps for Format B
            windows = estimate_windows_from_duration(session_path)
            if not windows:
                stats["skipped_no_graphs"] += 1
                return None
            log.info("    %d windows (CSV graph mode)", len(windows))
        else:
            stats["skipped_no_graphs"] += 1
            return None

    n_wins = len(windows)
    if n_wins == 0:
        stats["skipped_no_graphs"] += 1
        return None

    log.info("    %s  %d windows  %s",
             session_id, n_wins,
             "csv-graph" if csv_mode else "json-graph")

    m_embs  = extract_mouse_embeddings(session_path, windows)
    k_embs  = extract_keyboard_embeddings(session_path, windows)
    n_embs  = extract_notif_embeddings(session_path, windows)
    s_embs  = extract_switching_embeddings(
        session_path,
        [] if csv_mode else graphs,
        n_wins,
    )

    tucker_rows = []
    for w_idx in range(n_wins):
        slices = run_tucker(
            m_embs[w_idx]  if w_idx < len(m_embs)  else None,
            k_embs[w_idx]  if w_idx < len(k_embs)  else None,
            n_embs[w_idx]  if w_idx < len(n_embs)  else None,
            s_embs[w_idx]  if w_idx < len(s_embs)  else None,
            tfn,
        )
        tucker_rows.append(slices)

    stats["sessions_processed"] += 1
    stats["windows_total"]      += n_wins

    return {
        "X":            np.stack(tucker_rows, axis=0).astype(np.float32),  # (T, 4, 512)
        "y":            nasa_tlx,                                           # (9,)
        "user_id":      user_id,
        "session_id":   session_id,
        "T":            n_wins,
        "window_starts":[float(s) for s, _ in windows],
        "graph_format": "csv_aggregate" if csv_mode else "json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str,
        default=str(Path.home() / "aam_data_6_1" / "Data" / "After update"))
    parser.add_argument("--output_dir", type=str,
        default=str(Path.home() / "tucker_outputs"))
    parser.add_argument("--users", nargs="*", default=None)
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_dir = Path(__file__).resolve().parent
    if str(fusion_dir) not in sys.path:
        sys.path.insert(0, str(fusion_dir))

    from TFN import ActiveTFN
    tfn = ActiveTFN(d_dims=[64, 64, 32, 64], rank=8).eval()
    for p in tfn.parameters():
        p.requires_grad_(False)
    torch.manual_seed(42)

    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if args.users:
        user_dirs = [d for d in user_dirs if d.name in args.users]
    log.info("Found %d user(s): %s", len(user_dirs), [d.name for d in user_dirs])

    stats = {
        "sessions_processed": 0,
        "skipped_no_label":   0,
        "skipped_no_graphs":  0,
        "windows_total":      0,
    }

    sequences = []
    for user_dir in user_dirs:
        user_id  = user_dir.name
        sessions = find_sessions(user_dir)
        log.info("User: %s  (%d sessions)", user_id, len(sessions))

        for session_dir in sessions:
            result = process_session(session_dir, user_id, tfn, stats)
            if result is not None:
                sequences.append(result)

    if not sequences:
        log.error("No sessions processed. Check --data_root.")
        sys.exit(1)

    # Save as sequence list (variable T per session)
    seq_path = output_dir / "sequences.npy"
    np.save(seq_path, sequences, allow_pickle=True)

    # Also save flat versions for teammates who want window-level arrays
    all_X    = np.concatenate([s["X"] for s in sequences], axis=0)  # (N, 4, 512)
    all_y    = np.concatenate([np.tile(s["y"], (s["T"], 1))
                               for s in sequences], axis=0)           # (N, 9)
    meta     = []
    for s in sequences:
        for w_idx, ws in enumerate(s["window_starts"]):
            meta.append({
                "user_id":    s["user_id"],
                "session_id": s["session_id"],
                "window_idx": w_idx,
                "window_start": ws,
                "graph_format": s["graph_format"],
            })

    np.save(output_dir / "tucker_slices.npy", all_X)
    np.save(output_dir / "nasa_tlx_labels.npy", all_y)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # Dataset info
    info = {
        "n_sessions":         stats["sessions_processed"],
        "n_windows_total":    stats["windows_total"],
        "skipped_no_label":   stats["skipped_no_label"],
        "skipped_no_graphs":  stats["skipped_no_graphs"],
        "users":              sorted(set(s["user_id"] for s in sequences)),
        "sessions_per_user":  {
            uid: sum(1 for s in sequences if s["user_id"] == uid)
            for uid in sorted(set(s["user_id"] for s in sequences))
        },
        "windows_per_session": {
            s["session_id"]: s["T"] for s in sequences
        },
        "graph_formats": {
            "json": sum(1 for s in sequences if s["graph_format"] == "json"),
            "csv_aggregate": sum(1 for s in sequences if s["graph_format"] == "csv_aggregate"),
        }
    }
    with (output_dir / "dataset_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    log.info("=" * 65)
    log.info("Sessions processed:  %d", stats["sessions_processed"])
    log.info("Skipped no label:    %d", stats["skipped_no_label"])
    log.info("Skipped no graphs:   %d", stats["skipped_no_graphs"])
    log.info("Windows total:       %d", stats["windows_total"])
    log.info("sequences.npy:       %d sequences  →  %s", len(sequences), seq_path)
    log.info("tucker_slices.npy:   %s", all_X.shape)
    log.info("nasa_tlx_labels.npy: %s", all_y.shape)
    log.info("=" * 65)

    # Sanity
    assert all_X.shape == (stats["windows_total"], 4, 512)
    assert all_y.shape == (stats["windows_total"], 9)
    assert len(meta)   == stats["windows_total"]
    log.info("Sanity check passed.")


if __name__ == "__main__":
    main()