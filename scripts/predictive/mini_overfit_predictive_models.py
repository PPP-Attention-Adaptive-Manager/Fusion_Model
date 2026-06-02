"""STEP 5 — technical mini-overfit for the four predictive models.

Goal is NOT accuracy. It verifies each modality model can technically LEARN from
the rebuilt Tucker dataset: loader works, slice indexing/shape correct, forward
+ loss + non-zero grads + optimizer updates work, training loss drops, checkpoint
saves/reloads identically, output shape == (B,12), no NaN/Inf.

It does NOT run LOSO, final training, evaluation, or fusion. No model/embedder/
Tucker/GNN/notif files are modified. Inputs are standardized by a train-only
StandardScaler (Tucker variance ~1e-6) instead of editing the models.

Reused output contract (BaseModalityModel): (B,12) = [0:5] factors,
[5:10] state logits (5 classes), [10] H_norm, [11] M.

Usage:
  python scripts/predictive/mini_overfit_predictive_models.py \
    --data-dir data_training_full_rebuilt --output-dir outputs/predictive_mini_overfit \
    --modalities mouse keyboard notif switching --num-samples 32 --epochs 300 \
    --device cuda --seed 42
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
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictive_models import MODALITY_MODELS
from scripts.switching.train_switching_predictive import prepare_targets

MODALITY_INDEX = {"mouse": 0, "keyboard": 1, "notif": 2, "switching": 3}
STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]
FACTOR_NAMES = ["mental", "temporal", "effort", "frustration", "arousal"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="STEP 5 mini-overfit (technical learnability).")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/predictive_mini_overfit")
    ap.add_argument("--modalities", nargs="+", default=["mouse", "keyboard", "notif", "switching"])
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def select_subset(metadata, states, dual_avail, num_samples, seed) -> List[int]:
    """Deterministic, state- and user-diverse subset."""
    rng = np.random.default_rng(seed)
    by_key: Dict[tuple, List[int]] = {}
    for i, m in enumerate(metadata):
        by_key.setdefault((int(states[i]), str(m.get("user_id"))), []).append(i)
    # round-robin across (state,user) groups for diversity
    groups = sorted(by_key.keys())
    order = list(rng.permutation(len(groups)))
    chosen: List[int] = []
    pos = {g: 0 for g in range(len(groups))}
    while len(chosen) < min(num_samples, len(metadata)):
        progressed = False
        for gi in order:
            g = groups[gi]
            if pos[gi] < len(by_key[g]):
                chosen.append(by_key[g][pos[gi]]); pos[gi] += 1; progressed = True
                if len(chosen) >= num_samples:
                    break
        if not progressed:
            break
    return sorted(chosen)


def standardize(X_sub: np.ndarray):
    mean = X_sub.mean(axis=0)
    std = X_sub.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return ((X_sub - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_model(modality: str, device):
    cls = MODALITY_MODELS[MODALITY_INDEX[modality]]
    return cls(input_flat_dim=512).to(device)


def grad_norm(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().norm().item()) ** 2
    return tot ** 0.5


def forward(model, x):
    if hasattr(model, "reset_microstate"):
        model.reset_microstate()
    return model(x)


def compute_loss(out, factors, states):
    factor_loss = F.smooth_l1_loss(out[:, 0:5], factors)
    state_loss = F.cross_entropy(out[:, 5:10], states)
    return factor_loss + state_loss, factor_loss, state_loss


def train_modality(modality, X, factors, states, args, device, out_dir):
    md = out_dir / modality
    md.mkdir(parents=True, exist_ok=True)
    Xt = torch.from_numpy(X).to(device)
    ft = torch.from_numpy(factors).to(device)
    st = torch.from_numpy(states).to(device)

    torch.manual_seed(args.seed)
    model = build_model(modality, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: List[Dict[str, Any]] = []
    init_loss = None
    nan_inf = False
    for ep in range(1, args.epochs + 1):
        model.train()
        out = forward(model, Xt)
        loss, fl, sl = compute_loss(out, ft, st)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            o = out.detach()
            if torch.isnan(o).any() or torch.isinf(o).any() or torch.isnan(loss).any():
                nan_inf = True
        if init_loss is None:
            init_loss = float(loss.item())
        history.append({"epoch": ep, "total_loss": float(loss.item()),
                        "state_loss": float(sl.item()), "factor_loss": float(fl.item()),
                        "gradient_norm": gn, "output_mean": float(o.mean().item()),
                        "output_std": float(o.std().item()),
                        "has_nan": bool(torch.isnan(o).any().item()),
                        "has_inf": bool(torch.isinf(o).any().item())})

    # save loss history
    _write_csv(md / "loss_history.csv", history)

    # final eval outputs
    model.eval()
    with torch.no_grad():
        final_out = forward(model, Xt).detach()
    final_loss = history[-1]["total_loss"]
    final_fl = history[-1]["factor_loss"]; final_sl = history[-1]["state_loss"]
    init_fl = history[0]["factor_loss"]; init_sl = history[0]["state_loss"]
    grad_mean = float(np.mean([h["gradient_norm"] for h in history]))
    grad_any_nonzero = any(h["gradient_norm"] > 1e-12 for h in history)

    # checkpoint save + reload check
    ckpt = md / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "modality": modality,
                "input_flat_dim": 512, "contract": "(B,12) base"}, ckpt)
    model2 = build_model(modality, device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"]
    model2.load_state_dict(sd); model2.eval()
    with torch.no_grad():
        out2 = forward(model2, Xt).detach()
    max_abs_diff = float((final_out - out2).abs().max().item())
    reload_ok = max_abs_diff < 1e-5
    (md / "reload_check.json").write_text(json.dumps(
        {"max_abs_diff": max_abs_diff, "reload_ok": reload_ok, "tolerance": 1e-5}, indent=2), encoding="utf-8")

    # ---- PASS criteria ----
    reduction = (final_loss / init_loss) if init_loss > 0 else 1.0
    shape_ok = tuple(final_out.shape) == (X.shape[0], 12)
    if init_loss < 0.05:  # already very low -> require >=50% drop
        loss_ok = final_loss <= 0.5 * init_loss
    else:
        loss_ok = final_loss < 0.30 * init_loss
    fails = []
    if nan_inf:
        fails.append("NaN/Inf encountered")
    if not shape_ok:
        fails.append(f"output shape {tuple(final_out.shape)} != (B,12)")
    if not grad_any_nonzero:
        fails.append("gradients all zero")
    if not loss_ok:
        fails.append(f"loss not reduced enough (final/init={reduction:.3f})")
    if not reload_ok:
        fails.append(f"reload mismatch (max_abs_diff={max_abs_diff:.2e})")
    status = "PASS" if not fails else "FAIL"

    # ---- plots ----
    _plots(md, modality, history, final_out.cpu().numpy(), factors, states)

    return {
        "modality": modality, "status": status,
        "initial_loss": round(init_loss, 6), "final_loss": round(final_loss, 6),
        "loss_reduction_ratio": round(reduction, 4),
        "state_loss_initial": round(init_sl, 6), "state_loss_final": round(final_sl, 6),
        "factor_loss_initial": round(init_fl, 6), "factor_loss_final": round(final_fl, 6),
        "grad_norm_mean": round(grad_mean, 6), "reload_ok": reload_ok,
        "output_shape_ok": shape_ok, "nan_or_inf": nan_inf,
        "fail_reason": "; ".join(fails), "checkpoint_path": str(ckpt),
    }


def _plots(md, modality, history, final_out, factors, states):
    ep = [h["epoch"] for h in history]
    # loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["total_loss"] for h in history], label="total")
    plt.plot(ep, [h["state_loss"] for h in history], label="state")
    plt.plot(ep, [h["factor_loss"] for h in history], label="factor")
    plt.title(f"{modality} mini-overfit loss"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(md / "loss_curve.png", dpi=90); plt.close()
    # gradient norm
    plt.figure(figsize=(7, 4)); plt.plot(ep, [h["gradient_norm"] for h in history], color="purple")
    plt.title(f"{modality} gradient norm"); plt.xlabel("epoch"); plt.tight_layout()
    plt.savefig(md / "gradient_norm.png", dpi=90); plt.close()
    # factor pred vs true
    pred_f = final_out[:, 0:5]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for j in range(5):
        axes[j].scatter(factors[:, j], pred_f[:, j], s=14, alpha=0.7)
        lo, hi = 0, 1; axes[j].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[j].set_title(FACTOR_NAMES[j]); axes[j].set_xlabel("true"); axes[j].set_ylabel("pred")
    fig.suptitle(f"{modality} factor pred vs true (mini-overfit)"); plt.tight_layout()
    plt.savefig(md / "factor_pred_vs_true.png", dpi=90); plt.close()
    # output distribution
    plt.figure(figsize=(8, 4)); plt.hist(final_out.flatten(), bins=40, color="teal")
    plt.title(f"{modality} output distribution (12D)"); plt.tight_layout()
    plt.savefig(md / "output_distribution.png", dpi=90); plt.close()
    # confusion matrix
    pred_state = final_out[:, 5:10].argmax(axis=1)
    k = 5
    cm = np.zeros((k, k), int)
    for t, p in zip(states, pred_state):
        cm[int(t), int(p)] += 1
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(k):
        for j in range(k):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(k), STATE_NAMES, rotation=45, ha="right"); plt.yticks(range(k), STATE_NAMES)
    plt.xlabel("pred"); plt.ylabel("true"); plt.title(f"{modality} state confusion (overfit)")
    plt.colorbar(); plt.tight_layout(); plt.savefig(md / "state_confusion_matrix.png", dpi=90); plt.close()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    tucker = np.load(args.data_dir / "tucker_slices.npy").astype(np.float32)
    nasa = np.load(args.data_dir / "nasa_tlx_labels.npy").astype(np.float32)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    assert tucker.shape[0] == nasa.shape[0] == len(metadata), "row alignment mismatch"

    factors_all, states_all = prepare_targets(nasa, state_mode="5class")

    # dual-task availability (optional)
    dual_avail = {}
    dt = args.data_dir / "dual_task_window_labels.csv"
    if dt.exists():
        for r in csv.DictReader(dt.open(encoding="utf-8")):
            dual_avail[int(r["row_idx"])] = (str(r.get("dual_task_available", "")).lower() == "true")

    subset = select_subset(metadata, states_all, dual_avail, args.num_samples, args.seed)
    print(f"[mini-overfit] subset = {len(subset)} samples; states={np.bincount(states_all[subset], minlength=5).tolist()}")

    # save subset csv
    sub_rows = []
    for i in subset:
        m = metadata[i]
        sub_rows.append({"row_idx": i, "session_id": m.get("session_id"), "user_id": m.get("user_id"),
                         "device_id": m.get("device_id"), "window_id": m.get("window_id"),
                         "window_start": m.get("window_start"), "window_end": m.get("window_end"),
                         "state_label": int(states_all[i]),
                         **{f"factor_{FACTOR_NAMES[j]}": round(float(factors_all[i, j]), 4) for j in range(5)},
                         "dual_task_available": dual_avail.get(i, False)})
    _write_csv(out / "mini_overfit_subset.csv", sub_rows)

    factors_sub = factors_all[subset].astype(np.float32)
    states_sub = states_all[subset].astype(np.int64)

    results = []
    for modality in args.modalities:
        m_idx = MODALITY_INDEX[modality]
        X_sub = tucker[subset, m_idx, :]
        Xz, mean, std = standardize(X_sub)
        print(f"[mini-overfit] {modality}: slice idx {m_idx}, X {Xz.shape} "
              f"(raw var {X_sub.var():.2e} -> std var {Xz.var():.2f})")
        res = train_modality(modality, Xz, factors_sub, states_sub, args, device, out)
        # store scaler with checkpoint
        ck = torch.load(res["checkpoint_path"], map_location="cpu", weights_only=False)
        ck["scaler"] = {"mean": mean.tolist(), "std": std.tolist()}
        torch.save(ck, res["checkpoint_path"])
        results.append(res)
        print(f"  -> {res['status']} (init {res['initial_loss']:.4f} -> final {res['final_loss']:.4f}, "
              f"ratio {res['loss_reduction_ratio']:.3f}, reload_ok {res['reload_ok']})")

    all_pass = all(r["status"] == "PASS" for r in results)
    summary = {"predictive_final_training_allowed": all_pass,
               "num_samples": len(subset), "epochs": args.epochs,
               "modalities": {r["modality"]: r for r in results}}
    (out / "mini_overfit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out / "mini_overfit_summary.csv", results)
    print(json.dumps(summary, indent=2))
    print(f"\n[GATE] predictive_final_training_allowed = {all_pass}")


if __name__ == "__main__":
    main()
