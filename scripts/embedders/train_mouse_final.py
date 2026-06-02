"""Train the FINAL mouse embedder (MouseEncoderFinal).

Objective: SimCLR NT-Xent on two augmented views of each mouse window (real
positive pairs) + auxiliary regression of behavioral stats (speed_mean,
click_rate, idle_ratio, n_events, distance_total, scroll_event_count).

    loss = ntxent + aux_weight * MSE(aux_pred, standardized_stats)

Windows are EVENT-COUNT based (default 256 events / stride 128) -> guarantees
enough evidence per window, removing the previous cold-start zeros. Idle is a
real behavior (is_idle channel + idle_ratio feature), not a cold start.

Split is BY SESSION (80/20). Saves encoder.pt, model_config.json,
train_config.json, loss_history.csv, stats_scaler + aux_scaler.

Usage:
  python scripts/embedders/train_mouse_final.py --data-dir data --epochs 100 \
    --batch-size 128 --device cuda \
    --output-dir pre_embedders/mouse/exports/mouse_encoder_final
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from pre_embedders.mouse.encoder_final import MouseEncoderFinal
from scripts.embedders import embedder_common as ec


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train final mouse embedder (SimCLR + aux).")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "pre_embedders/mouse/exports/mouse_encoder_final")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--aux-weight", type=float, default=0.2)
    ap.add_argument("--w-events", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--min-events", type=int, default=32)
    ap.add_argument("--l2-normalize", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def run_epoch(model, SEQ, STATS, aux_t, idx, *, device, optimizer, ntxent, aux_weight, bs, rng, train):
    model.train(train)
    order = idx.copy()
    if train:
        np.random.default_rng().shuffle(order)
    tot = {"loss": 0.0, "con": 0.0, "aux": 0.0, "acc": 0.0, "n": 0.0}
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        if len(b) < 2:
            continue
        seq = torch.from_numpy(SEQ[b]).to(device)
        stt = torch.from_numpy(STATS[b]).to(device)
        ab = torch.from_numpy(aux_t[b]).to(device)
        s1, s2 = ec.augment_mouse(seq, rng), ec.augment_mouse(seq, rng)
        with torch.set_grad_enabled(train):
            _, p1, a1 = model.forward_train(s1, stt)
            _, p2, a2 = model.forward_train(s2, stt)
            con = ntxent(p1, p2)
            aux = 0.5 * (F.mse_loss(a1, ab) + F.mse_loss(a2, ab))
            loss = con + aux_weight * aux
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = len(b)
        tot["loss"] += float(loss.item()) * n
        tot["con"] += float(con.item()) * n
        tot["aux"] += float(aux.item()) * n
        tot["acc"] += ec.contrastive_accuracy(p1.detach(), p2.detach()) * n
        tot["n"] += n
    d = max(tot["n"], 1.0)
    return {k: tot[k] / d for k in ["loss", "con", "aux", "acc"]}


def main() -> None:
    args = parse_args()
    ec.set_seed(args.seed)
    device = ec.select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inv_dir = PROJECT_ROOT / "outputs/embedders/final_training"
    inv_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    SEQ, Ffeat, sessions, _ = ec.build_mouse_windows(
        args.data_dir, args.w_events, args.stride, args.min_events, log_rows=log_rows)
    ec.write_csv(inv_dir / "data_inventory_mouse.csv", log_rows)
    sessions = np.asarray(sessions, dtype=object)
    print(f"[mouse] windows={len(SEQ)} sessions={len(set(sessions.tolist()))}")
    if len(SEQ) < 50:
        raise SystemExit(f"Too few mouse windows ({len(SEQ)}).")

    tr, va = ec.session_split(sessions, 0.2, args.seed)

    # standardize the stats-branch INPUT (full 8 features) on train
    stats_scaler = ec.RobustScaler().fit(Ffeat[tr])
    STATS = stats_scaler.transform(Ffeat)

    aux_idx = [ec.MOUSE_FEATURES.index(f) for f in ec.MOUSE_AUX]
    aux_scaler = ec.RobustScaler().fit(Ffeat[tr][:, aux_idx])
    aux_t = aux_scaler.transform(Ffeat[:, aux_idx])

    model = MouseEncoderFinal(
        seq_channels=8, n_stats=len(ec.MOUSE_FEATURES), embedding_dim=64,
        l2_normalize=bool(args.l2_normalize), n_aux=len(ec.MOUSE_AUX),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ntxent = ec.NTXentLoss(args.temperature)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    best_val = float("inf"); best_ep = -1; patience = args.patience
    history = []
    ckpt_path = args.output_dir / "encoder.pt"
    for ep in range(1, args.epochs + 1):
        trm = run_epoch(model, SEQ, STATS, aux_t, tr, device=device, optimizer=opt, ntxent=ntxent,
                        aux_weight=args.aux_weight, bs=args.batch_size, rng=rng, train=True)
        vam = run_epoch(model, SEQ, STATS, aux_t, va, device=device, optimizer=None, ntxent=ntxent,
                        aux_weight=args.aux_weight, bs=args.batch_size, rng=rng, train=False)
        sched.step()
        history.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in trm.items()},
                        **{f"val_{k}": v for k, v in vam.items()}})
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d} train loss {trm['loss']:.4f} (con {trm['con']:.3f} aux {trm['aux']:.3f} "
                  f"acc {trm['acc']:.2f}) | val loss {vam['loss']:.4f} acc {vam['acc']:.2f}")
        if vam["loss"] < best_val - 1e-5:
            best_val = vam["loss"]; best_ep = ep; patience = args.patience
            _save(ckpt_path, model, args, stats_scaler, aux_scaler, best_ep, best_val)
        else:
            patience -= 1
            if patience <= 0:
                print(f"  early stop @ {ep}")
                break

    ec.write_csv(args.output_dir / "loss_history.csv", history)
    (args.output_dir / "train_config.json").write_text(json.dumps({
        "modality": "mouse", "epochs_run": len(history), "best_epoch": best_ep,
        "best_val_loss": best_val, "n_windows": int(len(SEQ)),
        "n_sessions": len(set(sessions.tolist())), "train_windows": int(len(tr)),
        "val_windows": int(len(va)), "device": str(device), "args": vars(args) | {
            "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }, indent=2, default=str), encoding="utf-8")
    print(f"[mouse] done. best val {best_val:.4f} @ ep{best_ep} -> {ckpt_path}")


def _save(path, model, args, stats_scaler, aux_scaler, ep, val):
    cfg = {
        "encoder_class": "MouseEncoderFinal",
        "architecture": "seq-branch 1D-CNN(8->32->64->64)+meanpool | stats-branch MLP(8->64) | fusion->64D LN; "
                        "SimCLR proj head + aux head (train only)",
        "input_schema": "seq (B,8,%d) channels[speed,accel,jerk,dx,dy,is_idle,is_click,is_scroll]; "
                        "stats (B,8) standardized %s; event-count window=%d stride=%d"
                        % (args.w_events, ec.MOUSE_FEATURES, args.w_events, args.stride),
        "training_objective": "NT-Xent (SimCLR, real augmented positive pairs) + %.2f * MSE behavioral-stats aux"
                              % args.aux_weight,
        "seq_channels": 8, "seq_len": args.w_events, "n_stats": len(ec.MOUSE_FEATURES),
        "stats_features": ec.MOUSE_FEATURES, "aux_features": ec.MOUSE_AUX, "n_aux": len(ec.MOUSE_AUX),
        "embedding_dim": 64, "l2_normalize": bool(args.l2_normalize),
        "w_events": args.w_events, "stride": args.stride, "min_events": args.min_events,
        "trained": True, "pretrained": False, "model_version": "final_v1",
    }
    torch.save({
        "model_state_dict": model.state_dict(), "model_config": cfg,
        "stats_scaler": stats_scaler.to_dict(), "aux_scaler": aux_scaler.to_dict(),
        "best_epoch": ep, "best_val_loss": val, "exported_at": datetime.now(timezone.utc).isoformat(),
    }, path)
    (path.parent / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
