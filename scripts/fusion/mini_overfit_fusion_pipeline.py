"""STEP 5.5 — end-to-end mini-overfit for the full predictive + PoE fusion chain.

Validates TECHNICAL correctness of: tucker_slices[:,m,:] -> 4 expert predictive
models -> (B,12) each -> PoE fusion -> final (B,12). NOT accuracy, NOT LOSO, NOT
final/production training. data_training_full_rebuilt/, embedders, Tucker,
GNN/switching, notif, and production fusion_model.py / poe.py are NOT modified.

Fusion module (documented wrapper, NOT a production replacement):
  The project PoE (poe.poe.PoEFusion, vanilla) is parameter-free, so a pure
  fusion_only mode would have nothing to train. We therefore use the REAL
  PoEFusion as the fusion BASE (factors = mean of expert factor dims; state =
  log(PoE(experts)); uncertainty via ema.compute_uncertainty — identical to
  InferrerFusion's global output), PLUS a thin zero-initialized residual head so
  the fusion is trainable. At init the residual is 0 -> output == real PoE output.
  EMA (a temporal smoother) is intentionally excluded from batch mini-overfit.

Modes (both run by default):
  fusion_only : experts frozen, train only the fusion residual head.
  end_to_end  : experts + fusion trained; loss += 0.25 * expert_aux_loss.

Usage:
  python scripts/fusion/mini_overfit_fusion_pipeline.py --data-dir data_training_full_rebuilt \
    --output-dir outputs/fusion_mini_overfit --num-samples 32 --epochs 300 \
    --device cuda --seed 42 --init fresh
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
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictive_models import MODALITY_MODELS
from poe.poe import PoEFusion
from ema.ema import compute_uncertainty
from scripts.switching.train_switching_predictive import prepare_targets

MODALITIES = ["mouse", "keyboard", "notif", "switching"]
STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]
FACTOR_NAMES = ["mental", "temporal", "effort", "frustration", "arousal"]
STEP5_DIR = PROJECT_ROOT / "outputs/predictive_mini_overfit"


# --------------------------------------------------------------------------- #
# Fusion wrapper
# --------------------------------------------------------------------------- #
class FullPredictiveFusionMiniOverfitModel(nn.Module):
    """4 experts + real PoE base + thin trainable residual head -> (B,12)."""

    def __init__(self, poe_mode: str = "vanilla"):
        super().__init__()
        self.experts = nn.ModuleList([MODALITY_MODELS[i](input_flat_dim=512) for i in range(4)])
        self.poe = PoEFusion(mode=poe_mode)          # real project PoE (param-free)
        self.residual = nn.Sequential(               # thin trainable fusion head
            nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 12))
        nn.init.zeros_(self.residual[-1].weight)     # at init: residual == 0 -> output == real PoE
        nn.init.zeros_(self.residual[-1].bias)

    def reset_microstate(self):
        for e in self.experts:
            if hasattr(e, "reset_microstate"):
                e.reset_microstate()

    def forward(self, x: torch.Tensor):
        # x: (B, 4, 512)
        self.reset_microstate()
        per = [self.experts[i](x[:, i, :]) for i in range(4)]   # 4 x (B,12)
        p_poe = self.poe(per)                                   # (B,5) probs
        state_logits = torch.log(p_poe + 1e-8)                  # (B,5)
        factors = torch.stack([o[:, 0:5] for o in per], 0).mean(0)  # (B,5)
        H, M = compute_uncertainty(state_logits)
        base = torch.cat([factors, state_logits, H.unsqueeze(-1), M.unsqueeze(-1)], dim=-1)
        resid = self.residual(torch.cat(per, dim=1))            # (B,12)
        return base + resid, per


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="STEP 5.5 fusion-pipeline mini-overfit.")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/fusion_mini_overfit")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", choices=["fresh", "step5"], default="fresh")
    ap.add_argument("--mode", choices=["fusion_only", "end_to_end", "both"], default="both")
    ap.add_argument("--poe-mode", choices=["vanilla", "weighted"], default="vanilla")
    return ap.parse_args()


def select_subset(metadata, states, num_samples, seed):
    rng = np.random.default_rng(seed)
    by: Dict[tuple, List[int]] = {}
    for i, m in enumerate(metadata):
        by.setdefault((int(states[i]), str(m.get("user_id"))), []).append(i)
    groups = sorted(by.keys()); order = list(rng.permutation(len(groups)))
    chosen: List[int] = []; pos = {g: 0 for g in range(len(groups))}
    while len(chosen) < min(num_samples, len(metadata)):
        progressed = False
        for gi in order:
            g = groups[gi]
            if pos[gi] < len(by[g]):
                chosen.append(by[g][pos[gi]]); pos[gi] += 1; progressed = True
                if len(chosen) >= num_samples:
                    break
        if not progressed:
            break
    return sorted(chosen)


def standardize_all(tucker_sub):
    """Per-modality train-only standardization. Returns (Xz (n,4,512), scalers)."""
    n = tucker_sub.shape[0]
    Xz = np.zeros_like(tucker_sub)
    scalers = {}
    for m in range(4):
        x = tucker_sub[:, m, :]
        mean = x.mean(0); std = x.std(0); std = np.where(std < 1e-12, 1.0, std)
        Xz[:, m, :] = (x - mean) / std
        scalers[MODALITIES[m]] = {"mean": mean.astype(np.float32).tolist(),
                                  "std": std.astype(np.float32).tolist()}
    return Xz.astype(np.float32), scalers


def load_step5_experts(model, device):
    for i, name in enumerate(MODALITIES):
        ck = STEP5_DIR / name / "best.pt"
        if ck.exists():
            sd = torch.load(ck, map_location=device, weights_only=False)["model_state_dict"]
            model.experts[i].load_state_dict(sd)
            print(f"  [init step5] loaded {name} expert from {ck}")
        else:
            print(f"  [init step5] WARNING no checkpoint for {name}; left fresh")


def grad_norm(params) -> float:
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().norm().item()) ** 2
    return tot ** 0.5


def loss_fn(out, factors, states):
    fl = F.smooth_l1_loss(out[:, 0:5], factors)
    sl = F.cross_entropy(out[:, 5:10], states)
    return fl + sl, fl, sl


def train_mode(mode, Xz, factors, states, scalers, args, device, out_dir):
    md = out_dir / mode; md.mkdir(parents=True, exist_ok=True)
    Xt = torch.from_numpy(Xz).to(device)
    ft = torch.from_numpy(factors).to(device)
    st = torch.from_numpy(states).to(device)

    torch.manual_seed(args.seed)
    model = FullPredictiveFusionMiniOverfitModel(poe_mode=args.poe_mode).to(device)
    if args.init == "step5":
        load_step5_experts(model, device)

    expert_params = [p for e in model.experts for p in e.parameters()]
    fusion_params = list(model.residual.parameters())
    if mode == "fusion_only":
        for p in expert_params:
            p.requires_grad_(False)
        trainable = fusion_params
    else:
        trainable = expert_params + fusion_params
    opt = torch.optim.AdamW([p for p in trainable if p.requires_grad], lr=args.lr)

    history = []
    init_loss = None
    nan_inf = False
    for ep in range(1, args.epochs + 1):
        model.train()
        y, per = model(Xt)
        final_loss, fl, sl = loss_fn(y, ft, st)
        expert_aux = torch.stack([loss_fn(o, ft, st)[0] for o in per]).mean()
        loss = final_loss + (0.25 * expert_aux if mode == "end_to_end" else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn_exp = grad_norm(expert_params)
        gn_fus = grad_norm(fusion_params)
        torch.nn.utils.clip_grad_norm_([p for p in trainable if p.requires_grad], 10.0)
        opt.step()
        with torch.no_grad():
            if torch.isnan(y).any() or torch.isinf(y).any() or torch.isnan(loss).any():
                nan_inf = True
        if init_loss is None:
            init_loss = float(loss.item())
        history.append({"epoch": ep, "total_loss": float(loss.item()),
                        "final_loss": float(final_loss.item()), "expert_aux_loss": float(expert_aux.item()),
                        "state_loss": float(sl.item()), "factor_loss": float(fl.item()),
                        "grad_norm_total": gn_exp + gn_fus, "grad_norm_experts": gn_exp,
                        "grad_norm_fusion": gn_fus, "output_mean": float(y.mean().item()),
                        "output_std": float(y.std().item()),
                        "has_nan": bool(torch.isnan(y).any().item()),
                        "has_inf": bool(torch.isinf(y).any().item())})

    _write_csv(md / "loss_history.csv", history)

    model.eval()
    with torch.no_grad():
        y_final, per_final = model(Xt)
        y_final = y_final.detach()
    final_loss_val = history[-1]["total_loss"]
    reduction = final_loss_val / init_loss if init_loss > 0 else 1.0

    # shapes
    expert_shapes_ok = all(tuple(o.shape) == (Xz.shape[0], 12) for o in per_final)
    final_shape_ok = tuple(y_final.shape) == (Xz.shape[0], 12)

    # grad non-zero for trainable set
    gn_key = "grad_norm_fusion" if mode == "fusion_only" else "grad_norm_total"
    grad_nonzero = any(h[gn_key] > 1e-12 for h in history)

    # checkpoint
    ckpt = md / "best.pt"
    torch.save({"experts_state_dicts": [e.state_dict() for e in model.experts],
                "fusion_state_dict": model.residual.state_dict(),
                "poe_mode": args.poe_mode, "scalers": scalers, "mode": mode, "init": args.init,
                "modality_order": MODALITIES, "output_contract": "(B,12) base",
                "target_config": {"factors": FACTOR_NAMES, "state_classes": STATE_NAMES}}, ckpt)
    # reload
    model2 = FullPredictiveFusionMiniOverfitModel(poe_mode=args.poe_mode).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    for i in range(4):
        model2.experts[i].load_state_dict(ck["experts_state_dicts"][i])
    model2.residual.load_state_dict(ck["fusion_state_dict"]); model2.eval()
    with torch.no_grad():
        y2, _ = model2(Xt)
    max_abs_diff = float((y_final - y2.detach()).abs().max().item())
    reload_ok = max_abs_diff < 1e-5
    (md / "reload_check.json").write_text(json.dumps(
        {"max_abs_diff": max_abs_diff, "reload_ok": reload_ok, "tolerance": 1e-5}, indent=2), encoding="utf-8")

    # PASS
    loss_ok = final_loss_val < 0.30 * init_loss if init_loss >= 0.05 else final_loss_val <= 0.5 * init_loss
    fails = []
    if nan_inf:
        fails.append("NaN/Inf encountered")
    if not expert_shapes_ok:
        fails.append("expert output shape != (B,12)")
    if not final_shape_ok:
        fails.append(f"final shape {tuple(y_final.shape)} != (B,12)")
    if not grad_nonzero:
        fails.append("trainable gradients all zero")
    if not loss_ok:
        fails.append(f"loss not reduced enough (final/init={reduction:.3f})")
    if not reload_ok:
        fails.append(f"reload mismatch ({max_abs_diff:.2e})")
    status = "PASS" if not fails else "FAIL"

    _plots(md, mode, history, y_final.cpu().numpy(), per_final, factors, states)

    return {"mode": mode, "status": status, "initial_loss": round(init_loss, 6),
            "final_loss": round(final_loss_val, 6), "loss_reduction_ratio": round(reduction, 4),
            "reload_ok": reload_ok, "expert_output_shapes_ok": expert_shapes_ok,
            "final_output_shape_ok": final_shape_ok, "nan_or_inf": nan_inf,
            "grad_norm_experts_mean": round(float(np.mean([h["grad_norm_experts"] for h in history])), 6),
            "grad_norm_fusion_mean": round(float(np.mean([h["grad_norm_fusion"] for h in history])), 6),
            "fail_reason": "; ".join(fails), "checkpoint_path": str(ckpt)}


def _plots(md, mode, history, y_final, per_final, factors, states):
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    for k in ["total_loss", "final_loss", "expert_aux_loss"]:
        plt.plot(ep, [h[k] for h in history], label=k)
    plt.yscale("log"); plt.title(f"{mode} losses"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(md / "loss_curve.png", dpi=90); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["final_loss"] for h in history], label="final")
    plt.plot(ep, [h["expert_aux_loss"] for h in history], label="expert_aux")
    plt.yscale("log"); plt.title(f"{mode} expert vs final loss"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(md / "expert_vs_final_loss.png", dpi=90); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["grad_norm_experts"] for h in history], label="experts")
    plt.plot(ep, [h["grad_norm_fusion"] for h in history], label="fusion")
    plt.title(f"{mode} gradient norms"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(md / "gradient_norms.png", dpi=90); plt.close()

    pf = y_final[:, 0:5]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for j in range(5):
        axes[j].scatter(factors[:, j], pf[:, j], s=14, alpha=0.7); axes[j].plot([0, 1], [0, 1], "k--", lw=1)
        axes[j].set_title(FACTOR_NAMES[j]); axes[j].set_xlabel("true"); axes[j].set_ylabel("pred")
    fig.suptitle(f"{mode} final factor pred vs true"); plt.tight_layout()
    plt.savefig(md / "factor_pred_vs_true_final.png", dpi=90); plt.close()

    plt.figure(figsize=(8, 4)); plt.hist(y_final.flatten(), bins=40, color="teal")
    plt.title(f"{mode} final output distribution"); plt.tight_layout()
    plt.savefig(md / "output_distribution_final.png", dpi=90); plt.close()

    pred = y_final[:, 5:10].argmax(1); cm = np.zeros((5, 5), int)
    for t, p in zip(states, pred):
        cm[int(t), int(p)] += 1
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(5), STATE_NAMES, rotation=45, ha="right"); plt.yticks(range(5), STATE_NAMES)
    plt.title(f"{mode} state confusion (final)"); plt.colorbar(); plt.tight_layout()
    plt.savefig(md / "state_confusion_matrix_final.png", dpi=90); plt.close()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    tucker = np.load(args.data_dir / "tucker_slices.npy").astype(np.float32)
    nasa = np.load(args.data_dir / "nasa_tlx_labels.npy").astype(np.float32)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    factors_all, states_all = prepare_targets(nasa, state_mode="5class")

    subset = select_subset(metadata, states_all, args.num_samples, args.seed)
    print(f"[fusion-mini] subset={len(subset)} init={args.init} states={np.bincount(states_all[subset], minlength=5).tolist()}")

    Xz, scalers = standardize_all(tucker[subset])
    factors = factors_all[subset].astype(np.float32)
    states = states_all[subset].astype(np.int64)

    # interface sanity: real PoE accepts the experts' (B,12) outputs
    with torch.no_grad():
        probe = FullPredictiveFusionMiniOverfitModel(poe_mode=args.poe_mode).to(device)
        yf, per = probe(torch.from_numpy(Xz).to(device))
        poe_out = probe.poe(per)
        print(f"[fusion-mini] real PoEFusion interface OK: experts {len(per)}x{tuple(per[0].shape)} "
              f"-> PoE {tuple(poe_out.shape)} -> final {tuple(yf.shape)}")

    modes = ["fusion_only", "end_to_end"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        print(f"[fusion-mini] === mode={mode} ===")
        r = train_mode(mode, Xz, factors, states, scalers, args, device, out)
        results.append(r)
        print(f"  -> {r['status']} init {r['initial_loss']:.4f} -> final {r['final_loss']:.4f} "
              f"(ratio {r['loss_reduction_ratio']:.3f}) reload_ok={r['reload_ok']}")

    # modes comparison plot
    if len(results) >= 1:
        plt.figure(figsize=(7, 4))
        for r in results:
            h = list(csv.DictReader((out / r["mode"] / "loss_history.csv").open(encoding="utf-8")))
            plt.plot([int(x["epoch"]) for x in h], [float(x["total_loss"]) for x in h], label=r["mode"])
        plt.yscale("log"); plt.title("fusion modes: total loss"); plt.xlabel("epoch"); plt.legend()
        plt.tight_layout(); plt.savefig(out / "fusion_modes_comparison.png", dpi=90); plt.close()

    all_pass = all(r["status"] == "PASS" for r in results)
    summary = {"fusion_mini_overfit_pass": all_pass, "step6_allowed": all_pass,
               "init": args.init, "num_samples": len(subset), "epochs": args.epochs,
               "fusion_module": "real PoEFusion base + thin trainable residual head (mini-overfit wrapper); "
                                "production poe.py/fusion_model.py unchanged; EMA excluded",
               "modes": {r["mode"]: r for r in results}}
    (out / "fusion_mini_overfit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out / "fusion_mini_overfit_summary.csv", results)
    print(json.dumps(summary, indent=2))
    print(f"\n[GATE] fusion_mini_overfit_pass = {all_pass} | step6_allowed = {all_pass}")


if __name__ == "__main__":
    main()
