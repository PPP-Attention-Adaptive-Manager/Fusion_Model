"""STEP 3 — Validate the full 4-modality Tucker rebuild (data_training_full_rebuilt/).

Checks row alignment, Tucker shape, finiteness, per-modality slice health,
pre-Tucker modality embedding health, fallback report, graph/metadata timestamp
alignment, device/user coverage, and old-vs-rebuilt comparison. Emits a PASS/FAIL
gate -> tucker_ready_for_predictive_training.

Usage:
  python scripts/validate/validate_tucker_full_rebuild.py \
    --data-training-dir data_training_full_rebuilt --raw-data-dir data \
    --output-dir outputs/tucker_full_validation
"""

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

import numpy as np

MODALITIES = ["mouse", "keyboard", "notif", "switching"]
VAR_MIN = 1e-6


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _duplicate_ratio(E):
    if E.shape[0] < 2:
        return 0.0
    nz = E[np.abs(E).sum(1) > 0]
    if nz.shape[0] < 2:
        return 0.0
    n = nz / (np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12)
    if n.shape[0] > 6000:
        n = n[np.random.default_rng(0).choice(n.shape[0], 6000, replace=False)]
    sim = n @ n.T; np.fill_diagonal(sim, -1.0)
    return float(np.mean(sim.max(1) > 0.9999))


def _effective_rank(E):
    nz = E[np.abs(E).sum(1) > 0]
    if nz.shape[0] < 2:
        return 0.0
    s = np.linalg.svd(nz - nz.mean(0), compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _health(E):
    nz = np.abs(E).sum(1) > 0
    valid = E[nz]; norms = np.linalg.norm(E, axis=1)
    return {
        "shape": list(E.shape), "all_zero_rows": int((~nz).sum()),
        "nan_count": int(np.isnan(E).sum()), "inf_count": int(np.isinf(E).sum()),
        "mean_abs_value": float(np.abs(E).mean()),
        "variance_mean": float(valid.var()) if valid.size else 0.0,
        "per_dim_var_min": float(valid.var(0).min()) if valid.size else 0.0,
        "per_dim_var_max": float(valid.var(0).max()) if valid.size else 0.0,
        "duplicate_ratio": _duplicate_ratio(E), "effective_rank": _effective_rank(E),
        "l2_norm_mean": float(norms[nz].mean()) if nz.any() else 0.0,
        "l2_norm_std": float(norms[nz].std()) if nz.any() else 0.0,
        "l2_norm_min": float(norms[nz].min()) if nz.any() else 0.0,
        "l2_norm_max": float(norms[nz].max()) if nz.any() else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="STEP 3 — validate full Tucker rebuild.")
    ap.add_argument("--data-training-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--raw-data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/tucker_full_validation")
    ap.add_argument("--timestamp-tolerance", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    d = args.data_training_dir
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    fail_reasons: List[str] = []

    tucker = np.load(d / "tucker_slices.npy")
    nasa = np.load(d / "nasa_tlx_labels.npy")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    n = len(meta)

    # 1. row alignment
    align = {"metadata_rows": n, "tucker_rows": int(tucker.shape[0]), "nasa_rows": int(nasa.shape[0]),
             "metadata_eq_tucker": n == tucker.shape[0], "metadata_eq_nasa": n == nasa.shape[0]}
    dt_csv = d / "dual_task_window_labels.csv"
    if dt_csv.exists():
        dt_rows = sum(1 for _ in csv.DictReader(dt_csv.open(encoding="utf-8")))
        align["dual_task_rows"] = dt_rows
        align["metadata_eq_dual_task"] = dt_rows == n
        if dt_rows != n:
            fail_reasons.append("dual_task row count mismatch")
    if not (align["metadata_eq_tucker"] and align["metadata_eq_nasa"]):
        fail_reasons.append("row alignment mismatch")

    # 2/3 shape + finite
    shape_ok = tucker.ndim == 3 and tucker.shape[1] == 4 and tucker.shape[2] == 512
    if not shape_ok:
        fail_reasons.append(f"tucker shape {tucker.shape} != (N,4,512)")
    if np.isnan(tucker).any() or np.isinf(tucker).any():
        fail_reasons.append("tucker has NaN/Inf")

    # 4. per-modality slice health
    slice_rows = []
    slice_json = {}
    for m, name in enumerate(MODALITIES):
        h = _health(tucker[:, m, :])
        slice_json[name] = h
        slice_rows.append({"modality": name, "index": m, **{k: v for k, v in h.items() if k != "shape"}})
        if h["variance_mean"] <= VAR_MIN:
            fail_reasons.append(f"{name} slice variance_mean {h['variance_mean']:.2e} <= {VAR_MIN}")
        if h["all_zero_rows"] != 0:
            fail_reasons.append(f"{name} slice has {h['all_zero_rows']} all-zero rows")
    (out / "tucker_slice_validation.json").write_text(json.dumps(slice_json, indent=2), encoding="utf-8")
    _write_csv(out / "tucker_slice_validation.csv", slice_rows)

    # 5. pre-Tucker modality embeddings
    emb_json = {}
    dims = {"mouse": 64, "keyboard": 64, "notif": 32, "switching": 64}
    for name in MODALITIES:
        p = d / "modality_embeddings" / f"{name}_embeddings.npy"
        if not p.exists():
            emb_json[name] = {"error": "missing"}; fail_reasons.append(f"{name} embeddings missing"); continue
        E = np.load(p)
        h = _health(E)
        emb_json[name] = h
        if h["shape"][1] != dims[name]:
            fail_reasons.append(f"{name} embedding dim {h['shape'][1]} != {dims[name]}")
        if h["variance_mean"] <= VAR_MIN:
            fail_reasons.append(f"{name} embedding variance_mean <= {VAR_MIN}")
        if h["nan_count"] or h["inf_count"]:
            fail_reasons.append(f"{name} embedding NaN/Inf")
    (out / "modality_embedding_validation.json").write_text(json.dumps(emb_json, indent=2), encoding="utf-8")

    # 6. fallback report (from modality_embeddings_metadata.csv)
    fb_rows = []
    me_csv = d / "modality_embeddings_metadata.csv"
    fb_summary = {}
    if me_csv.exists():
        me = list(csv.DictReader(me_csv.open(encoding="utf-8")))
        for name in MODALITIES:
            flags = [str(r.get(f"fallback_{name}", "")).lower() == "true" for r in me]
            reasons = {}
            sess_fb: Dict[str, int] = {}
            for r, f in zip(me, flags):
                if f:
                    rr = r.get(f"fallback_reason_{name}", "")
                    reasons[rr] = reasons.get(rr, 0) + 1
                    sess_fb[r["session_id"]] = sess_fb.get(r["session_id"], 0) + 1
            top_sess = sorted(sess_fb.items(), key=lambda kv: -kv[1])[:5]
            fb_summary[name] = {"fallback_rows": int(sum(flags)),
                                "fallback_ratio": float(np.mean(flags)) if flags else 0.0}
            fb_rows.append({"modality": name, "fallback_rows": int(sum(flags)),
                            "fallback_ratio": round(float(np.mean(flags)) if flags else 0.0, 4),
                            "top_reasons": json.dumps(reasons),
                            "top_sessions": json.dumps(top_sess)})
    _write_csv(out / "fallback_report.csv", fb_rows)

    # 7. graph/metadata timestamp alignment
    found = aligned = 0
    for r in meta:
        gp = r.get("graph_path")
        if not gp:
            continue
        try:
            w = json.loads(Path(gp).read_text(encoding="utf-8")).get("window", {})
            gs, ge = float(w.get("window_start")), float(w.get("window_end"))
            found += 1
            if abs(gs - r["window_start"]) <= args.timestamp_tolerance and abs(ge - r["window_end"]) <= args.timestamp_tolerance:
                aligned += 1
        except Exception:  # noqa: BLE001
            continue
    graph_align = {"graph_found": found, "graph_aligned": aligned, "total": n,
                   "aligned_ratio": round(aligned / max(n, 1), 4)}
    if found != n or aligned != n:
        fail_reasons.append(f"graph alignment {aligned}/{n}")

    # 8. device/user coverage
    user_null = sum(1 for r in meta if not r.get("user_id"))
    by_user: Dict[str, Dict[str, set]] = {}
    for r in meta:
        u = r.get("user_id") or "<null>"
        by_user.setdefault(u, {"windows": 0, "sessions": set()})
        by_user[u]["windows"] += 1
        by_user[u]["sessions"].add(r["session_id"])
    cov_rows = [{"user_id": u, "windows": v["windows"], "sessions": len(v["sessions"])}
                for u, v in sorted(by_user.items(), key=lambda kv: -kv[1]["windows"])]
    _write_csv(out / "user_session_coverage.csv", cov_rows)
    if user_null > 0:
        fail_reasons.append(f"{user_null} windows have null user_id")

    # 9. old vs rebuilt comparison
    cmp_rows = []
    for label, path in [("data_training", PROJECT_ROOT / "data_training"),
                        ("data_training_rebuilt", PROJECT_ROOT / "data_training_rebuilt"),
                        ("data_training_full_rebuilt", d)]:
        cmp_rows.append(_summarize_dataset(label, path))
    _write_csv(out / "old_vs_full_rebuilt_comparison.csv", cmp_rows)

    status = "PASS" if not fail_reasons else "FAIL"
    summary = {
        "status": status, "tucker_ready_for_predictive_training": status == "PASS",
        "n_windows": n, "n_sessions": len(set(r["session_id"] for r in meta)),
        "row_alignment": align, "tucker_shape_ok": shape_ok,
        "tucker_finite": not (bool(np.isnan(tucker).any()) or bool(np.isinf(tucker).any())),
        "slice_variance": {k: slice_json[k]["variance_mean"] for k in MODALITIES},
        "slice_all_zero_rows": {k: slice_json[k]["all_zero_rows"] for k in MODALITIES},
        "fallback": fb_summary, "graph_alignment": graph_align,
        "user_null_windows": user_null, "fail_reasons": fail_reasons,
    }
    (out / "tucker_full_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status == "FAIL":
        print("\n[GATE] FULL TUCKER REBUILD = FAIL — predictive training NOT allowed.")
    else:
        print("\n[GATE] FULL TUCKER REBUILD = PASS — tucker_ready_for_predictive_training=true.")


def _summarize_dataset(label, path):
    row = {"dataset": label, "exists": path.exists()}
    if not path.exists():
        return row
    try:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        t = np.load(path / "tucker_slices.npy")
        row["windows"] = len(meta)
        row["sessions"] = len(set(m.get("session_id") for m in meta))
        row["users"] = len(set(str(m.get("user_id")) for m in meta if m.get("user_id")))
        row["tucker_all_zero"] = bool(np.abs(t).sum() == 0)
        for m, name in enumerate(MODALITIES):
            if t.ndim == 3 and t.shape[1] == 4:
                row[f"var_{name}"] = round(float(t[:, m, :].var()), 5)
        dt = path / "dual_task_window_labels.csv"
        if dt.exists():
            row["dual_task_labelled"] = sum(1 for r in csv.DictReader(dt.open(encoding="utf-8"))
                                            if str(r.get("dual_task_available", "")).lower() == "true")
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)
    return row


if __name__ == "__main__":
    main()
