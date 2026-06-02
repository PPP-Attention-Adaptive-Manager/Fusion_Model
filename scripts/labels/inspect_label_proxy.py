"""Inspect a label proxy: build it, save arrays/CSV/config/diagnostics + plots.

Examples:
  python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt \
    --proxy nasa_tlx --output-dir outputs/label_proxies/nasa_tlx
  python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt \
    --proxy nasa_time_weighted --weight-function sigmoid \
    --output-dir outputs/label_proxies/nasa_time_weighted_sigmoid
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.labels.label_proxy import build_label_proxy


def parse_args():
    ap = argparse.ArgumentParser(description="Inspect a label proxy.")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--proxy", required=True, choices=["nasa_tlx", "dual_task_rt",
                                                       "nasa_time_weighted", "hybrid_rt_nasa"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weight-function", type=str, default=None)
    ap.add_argument("--state-mode", type=str, default=None)
    ap.add_argument("--alpha-rt", type=float, default=None)
    ap.add_argument("--config-json", type=str, default=None, help="extra config as JSON string")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = {}
    if args.config_json:
        cfg.update(json.loads(args.config_json))
    if args.weight_function:
        cfg["weight_function"] = args.weight_function
        cfg.setdefault("nasa_time_weight", {})
        if isinstance(cfg["nasa_time_weight"], dict):
            cfg["nasa_time_weight"]["weight_function"] = args.weight_function
    if args.state_mode:
        cfg["state_mode"] = args.state_mode
    if args.alpha_rt is not None:
        cfg["alpha_rt"] = args.alpha_rt

    res = build_label_proxy(args.data_dir, args.proxy, cfg)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    np.save(out / "targets.npy", res["targets"])
    np.save(out / "sample_weights.npy", res["sample_weights"])
    np.save(out / "mask.npy", res["mask"])
    if res["state_labels"] is not None:
        np.save(out / "state_labels.npy", res["state_labels"])
    (out / "label_proxy_config.json").write_text(json.dumps(res["config"], indent=2, default=str), encoding="utf-8")
    (out / "label_proxy_diagnostics.json").write_text(json.dumps(res["diagnostics"], indent=2, default=str), encoding="utf-8")

    _write_rows_csv(out / "label_proxy_rows.csv", res)
    _plots(out, res)

    print(f"[label-proxy] {args.proxy}: target_dim={res['targets'].shape[1]} "
          f"usable={int(res['mask'].sum())}/{len(res['metadata'])} "
          f"({res['diagnostics'].get('coverage', {}).get('coverage_pct', 100.0)}%)")
    print(json.dumps(res["diagnostics"].get("coverage", res["diagnostics"]), indent=2)[:600])
    print(f"wrote: {out}")


def _write_rows_csv(path, res):
    md = res["metadata"]; t = res["targets"]; w = res["sample_weights"]; mask = res["mask"]
    prog = res["progress"]; states = res["state_labels"]; names = res["target_names"]
    rows = []
    for i, m in enumerate(md):
        row = {"row_idx": i, "session_id": m.get("session_id"), "user_id": m.get("user_id"),
               "device_id": m.get("device_id"), "window_id": m.get("window_id"),
               "window_start": m.get("window_start"), "window_end": m.get("window_end"),
               "mask": bool(mask[i]), "sample_weight": round(float(w[i]), 5),
               "progress": round(float(prog[i]), 4),
               "fallback_dual_task_available": bool(m.get("fallback_switching", False)) if False else None}
        for j, nm in enumerate(names):
            row[f"target_{nm}"] = round(float(t[i, j]), 5)
        if states is not None:
            row["state_label"] = int(states[i])
        row["exclusion_reason"] = "" if mask[i] else "no_valid_label_for_proxy"
        rows.append(row)
    fields = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)


def _plots(out, res):
    t = res["targets"]; w = res["sample_weights"]; mask = res["mask"]; prog = res["progress"]
    md = res["metadata"]; names = res["target_names"]; proxy = res["proxy_name"]
    tv = t[mask]

    # target distribution
    plt.figure(figsize=(8, 4))
    for j, nm in enumerate(names):
        plt.hist(tv[:, j], bins=30, alpha=0.6, label=nm)
    plt.title(f"{proxy} target distribution (usable rows)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "target_distribution.png", dpi=90); plt.close()

    # sample weight distribution
    plt.figure(figsize=(7, 4)); plt.hist(w[mask], bins=30, color="slateblue")
    plt.title(f"{proxy} sample_weight distribution"); plt.tight_layout()
    plt.savefig(out / "sample_weight_distribution.png", dpi=90); plt.close()

    # target by user / session (mean of first target)
    def _by(keyname, fname):
        agg = {}
        for i, m in enumerate(md):
            if not mask[i]:
                continue
            agg.setdefault(str(m.get(keyname)), []).append(float(t[i, 0]))
        keys = sorted(agg); vals = [np.mean(agg[k]) for k in keys]
        plt.figure(figsize=(max(8, len(keys) * 0.4), 4)); plt.bar(range(len(keys)), vals, color="teal")
        plt.xticks(range(len(keys)), keys, rotation=90, fontsize=6)
        plt.title(f"{proxy} mean {names[0]} by {keyname}"); plt.tight_layout()
        plt.savefig(out / fname, dpi=90); plt.close()
    _by("user_id", "target_by_user.png")
    _by("session_id", "target_by_session.png")

    # progress->weight curve (time-weighted proxies)
    if proxy in ("nasa_time_weighted", "hybrid_rt_nasa"):
        order = np.argsort(prog)
        plt.figure(figsize=(7, 4)); plt.scatter(prog[order], w[order], s=8, alpha=0.5)
        plt.xlabel("session progress"); plt.ylabel("sample_weight")
        plt.title(f"{proxy} session-progress weight curve"); plt.tight_layout()
        plt.savefig(out / "session_progress_weight_curve.png", dpi=90); plt.close()

    # coverage by user (dual-task involved)
    cov = res["diagnostics"].get("coverage", {}).get("by_user")
    if cov:
        users = sorted(cov); avail = [cov[u]["available"] for u in users]; tot = [cov[u]["total"] for u in users]
        x = np.arange(len(users)); plt.figure(figsize=(max(8, len(users) * 0.5), 4))
        plt.bar(x, tot, label="total", color="lightgray"); plt.bar(x, avail, label="usable", color="darkorange")
        plt.xticks(x, users, rotation=90, fontsize=6); plt.legend(); plt.title(f"{proxy} coverage by user")
        plt.tight_layout(); plt.savefig(out / "coverage_by_user.png", dpi=90); plt.close()

    # rt vs nasa load (hybrid, where both exist)
    if proxy == "hybrid_rt_nasa":
        from scripts.labels.label_proxy import _nasa_load, _rt_load, _dt_available, _load
        data_dir = res_data_dir(res)
        meta2, nasa2, dt2 = _load(data_dir)
        rt_mask = np.asarray([_dt_available(dt2.get(i)) for i in range(len(meta2))], dtype=bool)
        rt_load, _stats, _rt, _miss, _err = _rt_load(meta2, dt2, res["config"], rt_mask)
        nasa_load = _nasa_load(nasa2, res["config"].get("nasa_factor_weights", {}))
        both = rt_mask
        if both.sum() > 1:
            plt.figure(figsize=(5, 5)); plt.scatter(nasa_load[both], rt_load[both], s=14, alpha=0.6)
            plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("NASA load"); plt.ylabel("RT load")
            corr = res["diagnostics"].get("rt_vs_nasa_load_corr")
            plt.title(f"hybrid: RT vs NASA load (r={corr:.2f})" if corr == corr else "hybrid: RT vs NASA load")
            plt.tight_layout(); plt.savefig(out / "rt_vs_nasa_load.png", dpi=90); plt.close()

    # state distribution
    if res["state_labels"] is not None:
        s = res["state_labels"][mask]
        plt.figure(figsize=(6, 4)); vals, counts = np.unique(s, return_counts=True)
        plt.bar(vals, counts, color="indianred"); plt.title(f"{proxy} state distribution")
        plt.xlabel("state label"); plt.tight_layout(); plt.savefig(out / "state_distribution.png", dpi=90); plt.close()


def res_data_dir(res):
    # metadata rows carry source_session_path -> parent of session dir is data/, but the dataset dir
    # is the one we loaded; fall back to default.
    return PROJECT_ROOT / "data_training_full_rebuilt"


if __name__ == "__main__":
    main()
