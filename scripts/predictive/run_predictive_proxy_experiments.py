"""Run STEP 6 per-modality training across multiple label proxies.

Launches train_predictive_experts.py once per proxy (default split
session_within_user) and writes a combined comparison summary. LOSO is NOT run
unless --split-mode loso is explicitly passed.

Usage:
  python scripts/predictive/run_predictive_proxy_experiments.py \
    --data-dir data_training_full_rebuilt --output-dir outputs/predictive_training \
    --proxies nasa_tlx nasa_time_weighted dual_task_rt hybrid_rt_nasa \
    --split-mode session_within_user --modalities mouse keyboard notif switching --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser(description="Run STEP 6 across label proxies.")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/predictive_training")
    ap.add_argument("--proxies", nargs="+",
                    default=["nasa_tlx", "nasa_time_weighted", "dual_task_rt", "hybrid_rt_nasa"])
    ap.add_argument("--split-mode", choices=["session_within_user", "loso"], default="session_within_user")
    ap.add_argument("--modalities", nargs="+", default=["mouse", "keyboard", "notif", "switching"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    trainer = PROJECT_ROOT / "scripts/predictive/train_predictive_experts.py"
    rows = []
    for proxy in args.proxies:
        cmd = [sys.executable, str(trainer), "--data-dir", str(args.data_dir),
               "--output-dir", str(args.output_dir), "--modalities", *args.modalities,
               "--label-proxy", proxy, "--split-mode", args.split_mode,
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--device", args.device, "--seed", str(args.seed)]
        print(f"\n=== proxy {proxy} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        rs = args.output_dir / proxy / args.split_mode / "run_summary.json"
        if rs.exists():
            data = json.loads(rs.read_text(encoding="utf-8"))
            for m in data["modalities"]:
                rows.append({"label_proxy": proxy, "split_mode": args.split_mode,
                             "modality": m["modality"], "status": m["status"],
                             "target_dim": m["target_dim"], "test_mae": m["test_mae"],
                             "baseline_train_mean_mae": m["baseline_train_mean_mae"],
                             "beats_train_mean": m["beats_train_mean"], "test_r2": m["test_r2"],
                             "test_pearson": m["test_pearson"],
                             "classification_macro_f1": m["classification_macro_f1"],
                             "beats_majority": m["beats_majority"],
                             "n_train": m["n_train"], "n_test": m["n_test"]})
    out = args.output_dir / f"proxy_comparison_{args.split_mode}.csv"
    if rows:
        import csv
        with out.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\n[runner] comparison -> {out}")
        # print quick table
        print(f"{'proxy':20s} {'modality':9s} {'MAE':>7s} {'base':>7s} {'beats':>6s} {'R2':>7s} {'macroF1':>8s}")
        for r in rows:
            mf = "" if r["classification_macro_f1"] is None else f"{r['classification_macro_f1']:.3f}"
            print(f"{r['label_proxy']:20s} {r['modality']:9s} {r['test_mae']:7.4f} "
                  f"{r['baseline_train_mean_mae']:7.4f} {str(r['beats_train_mean']):>6s} "
                  f"{(r['test_r2'] if r['test_r2'] is not None else float('nan')):7.3f} {mf:>8s}")


if __name__ == "__main__":
    main()
