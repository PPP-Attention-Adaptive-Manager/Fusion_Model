"""STEP 6 — real per-modality predictive training via the label proxy system.

Trains each modality model independently on its Tucker slice, with HELD-OUT
splits (session_within_user or LOSO), using configurable supervision from the
label proxy layer. Reports test metrics + baselines. NOT fusion, NOT mini-overfit.

This is technical PASS vs scientific GOOD: technical = trains/evaluates cleanly;
GOOD = beats the train_mean / majority baselines on held-out data. We never claim
GOOD from training-loss alone, and never claim final cognitive accuracy from
session_within_user alone.

Usage:
  python scripts/predictive/train_predictive_experts.py \
    --data-dir data_training_full_rebuilt --output-dir outputs/predictive_training \
    --modalities mouse keyboard notif switching --label-proxy nasa_time_weighted \
    --split-mode session_within_user --epochs 200 --batch-size 32 --device cuda --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

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
from scripts.predictive.label_proxy_integration import load_targets_for_training

MODALITY_INDEX = {"mouse": 0, "keyboard": 1, "notif": 2, "switching": 3}
STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]


# --------------------------------------------------------------------------- #
# splits (deterministic, by session)
# --------------------------------------------------------------------------- #
def session_within_user_split(metadata, mask, seed, ratios=(0.70, 0.15, 0.15)):
    """Partition each user's sessions into train/val/test; no session shared."""
    rng = np.random.default_rng(seed)
    user_sessions: Dict[str, List[str]] = {}
    for i, m in enumerate(metadata):
        if not mask[i]:
            continue
        u = str(m.get("user_id")); s = str(m.get("session_id"))
        user_sessions.setdefault(u, [])
        if s not in user_sessions[u]:
            user_sessions[u].append(s)
    assign: Dict[str, str] = {}
    for u, sessions in user_sessions.items():
        sess = sorted(sessions)
        perm = list(rng.permutation(sess))
        k = len(perm)
        if k == 1:
            assign[perm[0]] = "train"
        elif k == 2:
            assign[perm[0]] = "train"; assign[perm[1]] = "test"
        else:
            n_tr = max(1, int(round(k * ratios[0])))
            n_va = max(1, int(round(k * ratios[1])))
            n_va = min(n_va, k - n_tr - 1) if k - n_tr - 1 > 0 else 0
            for j, s in enumerate(perm):
                assign[s] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
    split = np.array(["none"] * len(metadata), dtype=object)
    for i, m in enumerate(metadata):
        if mask[i]:
            split[i] = assign.get(str(m.get("session_id")), "train")
    return split


def loso_folds(metadata, mask, seed, min_rows=10):
    users = sorted({str(m.get("user_id")) for i, m in enumerate(metadata) if mask[i]})
    folds = []
    for test_user in users:
        test_idx = np.array([i for i, m in enumerate(metadata)
                             if mask[i] and str(m.get("user_id")) == test_user])
        rest = [u for u in users if u != test_user]
        if len(test_idx) < min_rows or not rest:
            folds.append({"test_user": test_user, "skip": True,
                          "reason": f"test rows {len(test_idx)} < {min_rows} or no train users"})
            continue
        rng = np.random.default_rng(seed + hash(test_user) % 1000)
        val_user = sorted(rest)[rng.integers(0, len(rest))]
        train_idx = np.array([i for i, m in enumerate(metadata)
                              if mask[i] and str(m.get("user_id")) in rest and str(m.get("user_id")) != val_user])
        val_idx = np.array([i for i, m in enumerate(metadata)
                            if mask[i] and str(m.get("user_id")) == val_user])
        if len(train_idx) < min_rows or len(val_idx) < 1:
            folds.append({"test_user": test_user, "skip": True,
                          "reason": f"train {len(train_idx)} / val {len(val_idx)} too small"})
            continue
        folds.append({"test_user": test_user, "skip": False,
                      "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    return folds


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if a.size < 3:
        return float("nan")
    return _pearson(np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float))


def regression_metrics(y, p, w=None, names=None):
    out: Dict[str, Any] = {}
    err = p - y
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = (1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    yf, pf = y.reshape(-1), p.reshape(-1)
    out["pearson_r"] = _pearson(yf, pf); out["spearman_rho"] = _spearman(yf, pf)
    if w is not None and w.sum() > 0:
        out["weighted_mae"] = float(np.sum(w.reshape(-1, 1) * np.abs(err)) / (w.sum() * y.shape[1]))
    if names and y.shape[1] == len(names):
        out["per_target"] = {names[j]: {"mae": float(np.mean(np.abs(err[:, j]))),
                                        "rmse": float(math.sqrt(np.mean(err[:, j] ** 2)))}
                             for j in range(len(names))}
    out["n"] = int(y.shape[0])
    return out


def classification_metrics(yt, yp, k=5):
    yt = yt.astype(int); yp = yp.astype(int)
    cm = np.zeros((k, k), int)
    for t, q in zip(yt, yp):
        cm[t, q] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    f1s, precs, recs = [], {}, {}
    for c in range(k):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        present = cm[c, :].sum() > 0
        if present:
            f1s.append(f1)
        precs[STATE_NAMES[c]] = round(prec, 4); recs[STATE_NAMES[c]] = round(rec, 4)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    n = cm.sum()
    # MCC (multiclass)
    t_sum = cm.sum(1).astype(float); p_sum = cm.sum(0).astype(float); c = float(np.trace(cm)); s = float(n)
    num = c * s - float(t_sum @ p_sum)
    den = math.sqrt(max((s * s - float(p_sum @ p_sum)) * (s * s - float(t_sum @ t_sum)), 0.0))
    mcc = num / den if den > 1e-12 else 0.0
    pe = float(t_sum @ p_sum) / (s * s) if s > 0 else 0.0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "mcc": float(mcc), "cohen_kappa": float(kappa),
            "confusion_matrix": cm.tolist(), "precision": precs, "recall": recs, "n": int(n)}


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def _simple_mlp(in_dim, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))


def compute_baselines(Xz, y, states, idx_tr, idx_te, metadata, target_dim, device, seed):
    out: Dict[str, Any] = {}
    ytr, yte = y[idx_tr], y[idx_te]
    # regression baselines
    gm = ytr.mean(axis=0)
    out["train_mean"] = regression_metrics(yte, np.tile(gm, (len(idx_te), 1)))
    # train_user_mean
    um = {}
    for k, i in enumerate(idx_tr):
        u = str(metadata[i].get("user_id")); um.setdefault(u, []).append(ytr[k])
    um = {u: np.mean(v, axis=0) for u, v in um.items()}
    pred_u = np.array([um.get(str(metadata[i].get("user_id")), gm) for i in idx_te])
    out["train_user_mean"] = regression_metrics(yte, pred_u)
    # previous_window (chronological within session, using true previous target)
    order = sorted(range(len(metadata)), key=lambda i: (str(metadata[i].get("session_id")),
                                                        float(metadata[i].get("window_start") or 0)))
    prev_target = {}
    last_by_sess: Dict[str, np.ndarray] = {}
    for i in order:
        s = str(metadata[i].get("session_id"))
        prev_target[i] = last_by_sess.get(s)
        last_by_sess[s] = y[i]
    pred_prev = np.array([prev_target[i] if prev_target[i] is not None else gm for i in idx_te])
    out["previous_window"] = regression_metrics(yte, pred_prev)
    # simple_mlp regression
    out["simple_mlp"] = _train_eval_simple(Xz, y, idx_tr, idx_te, target_dim, device, seed, cls=False)

    cls_out = {}
    if states is not None:
        st_tr, st_te = states[idx_tr], states[idx_te]
        gmaj = int(np.bincount(states, minlength=5).argmax())
        cls_out["global_majority"] = classification_metrics(st_te, np.full_like(st_te, gmaj))
        tmaj = int(np.bincount(st_tr, minlength=5).argmax())
        cls_out["train_majority"] = classification_metrics(st_te, np.full_like(st_te, tmaj))
        rng = np.random.default_rng(seed)
        probs = np.bincount(st_tr, minlength=5).astype(float); probs = probs / probs.sum()
        strat = rng.choice(5, size=len(st_te), p=probs)
        cls_out["stratified_random"] = classification_metrics(st_te, strat)
        cls_out["simple_mlp"] = _train_eval_simple(Xz, states, idx_tr, idx_te, 5, device, seed, cls=True)
    return out, cls_out


def _train_eval_simple(Xz, y, idx_tr, idx_te, out_dim, device, seed, cls):
    torch.manual_seed(seed)
    m = _simple_mlp(Xz.shape[1], out_dim).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(Xz[idx_tr]).to(device)
    if cls:
        ytr = torch.from_numpy(y[idx_tr].astype(np.int64)).to(device)
    else:
        ytr = torch.from_numpy(y[idx_tr].astype(np.float32)).to(device)
        if ytr.ndim == 1:
            ytr = ytr.unsqueeze(1)
    for _ in range(150):
        m.train(); opt.zero_grad()
        o = m(Xtr)
        loss = F.cross_entropy(o, ytr) if cls else F.smooth_l1_loss(o, ytr)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = m(torch.from_numpy(Xz[idx_te]).to(device)).cpu().numpy()
    if cls:
        return classification_metrics(y[idx_te], o.argmax(1))
    return regression_metrics(y[idx_te].reshape(len(idx_te), -1), o.reshape(len(idx_te), -1))


# --------------------------------------------------------------------------- #
# core train/eval for one modality
# --------------------------------------------------------------------------- #
def standardize(X, idx_tr):
    mean = X[idx_tr].mean(0); std = X[idx_tr].std(0); std = np.where(std < 1e-12, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def loss_fn(out, targets, states, w, target_dim, lambda_state, loss_kind):
    reg = F.smooth_l1_loss if loss_kind == "smoothl1" else F.mse_loss
    if target_dim == 5:
        per = reg(out[:, 0:5], targets, reduction="none").mean(1)
    else:
        per = reg(out[:, 0], targets[:, 0], reduction="none")
    total = (w * per).sum() / (w.sum() + 1e-8)
    state_loss = torch.tensor(0.0, device=out.device)
    if states is not None:
        sl = F.cross_entropy(out[:, 5:10], states, reduction="none")
        state_loss = (w * sl).sum() / (w.sum() + 1e-8)
        total = total + lambda_state * state_loss
    return total, state_loss


def train_one(modality, Xz, targets, states, weights, idx_tr, idx_va, idx_te,
              metadata, target_names, args, device, out_dir, proxy_cfg, diagnostics):
    md = out_dir / modality; (md / "plots").mkdir(parents=True, exist_ok=True)
    target_dim = targets.shape[1]
    cls_on = states is not None

    def to(idx, arr):
        return torch.from_numpy(arr[idx]).to(device)

    Xtr, Xva = to(idx_tr, Xz), to(idx_va, Xz) if len(idx_va) else None
    ttr = torch.from_numpy(targets[idx_tr]).to(device)
    tva = torch.from_numpy(targets[idx_va]).to(device) if len(idx_va) else None
    wtr = torch.from_numpy(weights[idx_tr]).to(device)
    wva = torch.from_numpy(weights[idx_va]).to(device) if len(idx_va) else None
    str_tr = torch.from_numpy(states[idx_tr]).to(device) if cls_on else None
    str_va = torch.from_numpy(states[idx_va]).to(device) if (cls_on and len(idx_va)) else None

    torch.manual_seed(args.seed)
    model = MODALITY_MODELS[MODALITY_INDEX[modality]](input_flat_dim=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf"); best_state = None; best_ep = -1; patience = args.patience
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "reset_microstate"):
            model.reset_microstate()
        out = model(Xtr)
        loss, sloss = loss_fn(out, ttr, str_tr, wtr, target_dim, args.lambda_state, args.loss)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        # val
        if Xva is not None and len(idx_va):
            model.eval()
            with torch.no_grad():
                if hasattr(model, "reset_microstate"):
                    model.reset_microstate()
                vo = model(Xva)
                vloss, _ = loss_fn(vo, tva, str_va, wva, target_dim, args.lambda_state, args.loss)
            vloss_f = float(vloss.item())
        else:
            vloss_f = float(loss.item())
        history.append({"epoch": ep, "train_loss": float(loss.item()), "val_loss": vloss_f,
                        "state_loss": float(sloss.item())})
        if vloss_f < best_val - 1e-6:
            best_val = vloss_f; best_ep = ep; patience = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # eval on test
    model.eval()
    with torch.no_grad():
        if hasattr(model, "reset_microstate"):
            model.reset_microstate()
        te_out = model(to(idx_te, Xz)).cpu().numpy()
    yte = targets[idx_te]; wte = weights[idx_te]
    nan_inf = bool(np.isnan(te_out).any() or np.isinf(te_out).any())

    if target_dim == 5:
        reg = regression_metrics(yte, te_out[:, 0:5], wte, target_names)
    else:
        reg = regression_metrics(yte.reshape(-1, 1), te_out[:, 0:1], wte, target_names)
    metrics = {"regression": reg}
    if cls_on:
        metrics["classification"] = classification_metrics(states[idx_te], te_out[:, 5:10].argmax(1))

    # baselines
    base_reg, base_cls = compute_baselines(Xz, targets if target_dim > 1 else targets,
                                           states, idx_tr, idx_te, metadata, target_dim, device, args.seed)
    beats = {"train_mean": reg["mae"] < base_reg["train_mean"]["mae"]}
    if cls_on and "train_majority" in base_cls:
        beats["majority_macro_f1"] = metrics["classification"]["macro_f1"] > base_cls["train_majority"]["macro_f1"]

    # checkpoint + reload
    ckpt = md / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "modality": modality,
                "scaler": SCALER_STORE.get(modality, {}),
                "input_flat_dim": 512, "target_dim": target_dim,
                "target_names": target_names, "label_proxy": args.label_proxy}, ckpt)
    (md / "scaler.json").write_text(json.dumps(SCALER_STORE.get(modality, {}), indent=2), encoding="utf-8")
    m2 = MODALITY_MODELS[MODALITY_INDEX[modality]](input_flat_dim=512).to(device)
    m2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"]); m2.eval()
    with torch.no_grad():
        if hasattr(m2, "reset_microstate"):
            m2.reset_microstate()
        te2 = m2(to(idx_te, Xz)).cpu().numpy()
    reload_ok = bool(np.abs(te2 - te_out).max() < 1e-4)

    # save artifacts
    _write_csv(md / "loss_history.csv", history)
    (md / "metrics.json").write_text(json.dumps({
        "modality": modality, "target_dim": target_dim, "target_names": target_names,
        "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
        "nan_or_inf": nan_inf, "reload_ok": reload_ok, "best_epoch": best_ep,
        "metrics": metrics, "beats_baseline": beats}, indent=2), encoding="utf-8")
    (md / "baseline_metrics.json").write_text(json.dumps({"regression": base_reg, "classification": base_cls},
                                                         indent=2), encoding="utf-8")
    (md / "train_config.json").write_text(json.dumps(vars(args) | {"data_dir": str(args.data_dir),
                                                                    "output_dir": str(args.output_dir)},
                                                     indent=2, default=str), encoding="utf-8")
    (md / "label_proxy_config.json").write_text(json.dumps(proxy_cfg, indent=2, default=str), encoding="utf-8")
    _save_predictions(md / "predictions.csv", metadata, idx_te, yte, te_out, target_dim, target_names, states)
    _plots(md / "plots", modality, history, yte, te_out, target_dim, target_names, states,
           metadata, idx_te, base_reg, reg)

    status = "PASS" if (not nan_inf and reload_ok) else "FAIL"
    return {"modality": modality, "status": status, "nan_or_inf": nan_inf, "reload_ok": reload_ok,
            "target_dim": target_dim, "test_mae": reg["mae"], "test_r2": reg.get("r2"),
            "test_pearson": reg.get("pearson_r"), "baseline_train_mean_mae": base_reg["train_mean"]["mae"],
            "beats_train_mean": beats["train_mean"],
            "classification_macro_f1": metrics.get("classification", {}).get("macro_f1"),
            "beats_majority": beats.get("majority_macro_f1"),
            "n_train": len(idx_tr), "n_test": len(idx_te)}


SCALER_STORE: Dict[str, Dict[str, Any]] = {}


def _save_predictions(path, metadata, idx_te, yte, te_out, target_dim, names, states):
    rows = []
    for k, i in enumerate(idx_te):
        m = metadata[i]
        row = {"row_idx": int(i), "session_id": m.get("session_id"), "user_id": m.get("user_id"),
               "window_id": m.get("window_id")}
        if target_dim == 5:
            for j, nm in enumerate(names):
                row[f"true_{nm}"] = round(float(yte[k, j]), 5); row[f"pred_{nm}"] = round(float(te_out[k, j]), 5)
        else:
            row[f"true_{names[0]}"] = round(float(yte[k, 0]), 5); row[f"pred_{names[0]}"] = round(float(te_out[k, 0]), 5)
        if states is not None:
            row["true_state"] = int(states[i]); row["pred_state"] = int(te_out[k, 5:10].argmax())
        rows.append(row)
    _write_csv(path, rows)


def _plots(pd_dir, modality, history, yte, te_out, target_dim, names, states, metadata, idx_te, base_reg, reg):
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4)); plt.plot(ep, [h["train_loss"] for h in history], label="train")
    plt.plot(ep, [h["val_loss"] for h in history], label="val"); plt.legend(); plt.title(f"{modality} loss")
    plt.tight_layout(); plt.savefig(pd_dir / "loss_curve.png", dpi=85); plt.close()

    if target_dim == 5:
        fig, ax = plt.subplots(1, 5, figsize=(18, 3.4))
        for j in range(5):
            ax[j].scatter(yte[:, j], te_out[:, j], s=12, alpha=0.6); ax[j].plot([0, 1], [0, 1], "k--", lw=1)
            ax[j].set_title(names[j]); ax[j].set_xlabel("true"); ax[j].set_ylabel("pred")
        plt.tight_layout(); plt.savefig(pd_dir / "pred_vs_true_factors.png", dpi=85); plt.close()
        mae = [np.mean(np.abs(te_out[:, j] - yte[:, j])) for j in range(5)]
        plt.figure(figsize=(7, 4)); plt.bar(names, mae, color="teal"); plt.title(f"{modality} per-target MAE")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(pd_dir / "per_target_mae.png", dpi=85); plt.close()
        resid = (te_out[:, 0:5] - yte).reshape(-1)
    else:
        plt.figure(figsize=(5, 5)); plt.scatter(yte[:, 0], te_out[:, 0], s=14, alpha=0.6)
        plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("true"); plt.ylabel("pred")
        plt.title(f"{modality} {names[0]}"); plt.tight_layout(); plt.savefig(pd_dir / "pred_vs_true_scalar.png", dpi=85); plt.close()
        resid = (te_out[:, 0] - yte[:, 0])

    plt.figure(figsize=(7, 4)); plt.hist(resid, bins=30, color="indianred"); plt.title(f"{modality} residuals")
    plt.tight_layout(); plt.savefig(pd_dir / "residual_distribution.png", dpi=85); plt.close()

    # baseline comparison (MAE)
    labels = ["model"] + list(base_reg.keys()); vals = [reg["mae"]] + [base_reg[k]["mae"] for k in base_reg]
    plt.figure(figsize=(8, 4)); plt.bar(labels, vals, color="slateblue"); plt.title(f"{modality} test MAE vs baselines")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(pd_dir / "baseline_comparison.png", dpi=85); plt.close()

    if states is not None:
        cm = np.zeros((5, 5), int)
        for k, i in enumerate(idx_te):
            cm[int(states[i]), int(te_out[k, 5:10].argmax())] += 1
        plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
        for a in range(5):
            for b in range(5):
                plt.text(b, a, cm[a, b], ha="center", va="center", fontsize=7)
        plt.xticks(range(5), STATE_NAMES, rotation=45, ha="right"); plt.yticks(range(5), STATE_NAMES)
        plt.title(f"{modality} confusion"); plt.tight_layout(); plt.savefig(pd_dir / "confusion_matrix.png", dpi=85); plt.close()

    # prediction by user / session (mean |error|)
    for keyname, fname in [("user_id", "prediction_by_user.png"), ("session_id", "prediction_by_session.png")]:
        agg = {}
        for k, i in enumerate(idx_te):
            e = float(np.mean(np.abs(resid))) if False else float(np.abs(te_out[k, 0] - yte[k, 0]))
            agg.setdefault(str(metadata[i].get(keyname)), []).append(e)
        keys = sorted(agg); vals = [np.mean(agg[x]) for x in keys]
        plt.figure(figsize=(max(7, len(keys) * 0.4), 4)); plt.bar(range(len(keys)), vals, color="darkorange")
        plt.xticks(range(len(keys)), keys, rotation=90, fontsize=6); plt.title(f"{modality} test |err| by {keyname}")
        plt.tight_layout(); plt.savefig(pd_dir / fname, dpi=85); plt.close()


def _write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8"); return
    with Path(path).open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="STEP 6 per-modality predictive training.")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_training_full_rebuilt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/predictive_training")
    ap.add_argument("--modalities", nargs="+", default=["mouse", "keyboard", "notif", "switching"])
    ap.add_argument("--label-proxy", required=True,
                    choices=["nasa_tlx", "nasa_time_weighted", "dual_task_rt", "hybrid_rt_nasa"])
    ap.add_argument("--split-mode", choices=["session_within_user", "loso"], default="session_within_user")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)  # full-batch on this small set
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-state", type=float, default=0.5)
    ap.add_argument("--loss", choices=["smoothl1", "mse"], default="smoothl1")
    ap.add_argument("--state-mode", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-loso-rows", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    proxy_cfg = {}
    if args.state_mode:
        proxy_cfg["state_mode"] = args.state_mode
    bundle = load_targets_for_training(args.data_dir, args.label_proxy, proxy_cfg or None)
    metadata = bundle["metadata"]; mask = bundle["mask"]
    targets = bundle["targets"]; states = bundle["state_labels"]; weights = bundle["sample_weights"]
    target_names = bundle["target_names"]; diagnostics = bundle["diagnostics"]
    tucker = np.load(args.data_dir / "tucker_slices.npy").astype(np.float32)

    run_dir = args.output_dir / args.label_proxy / args.split_mode
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "session_within_user":
        split = session_within_user_split(metadata, mask, args.seed)
        # save splits.csv
        srows = [{"row_idx": i, "session_id": metadata[i].get("session_id"),
                  "user_id": metadata[i].get("user_id"), "split": split[i]}
                 for i in range(len(metadata)) if mask[i]]
        _write_csv(run_dir / "splits.csv", srows)
        idx_tr = np.array([i for i in range(len(metadata)) if mask[i] and split[i] == "train"])
        idx_va = np.array([i for i in range(len(metadata)) if mask[i] and split[i] == "val"])
        idx_te = np.array([i for i in range(len(metadata)) if mask[i] and split[i] == "test"])
        print(f"[step6] proxy={args.label_proxy} split=session_within_user "
              f"train/val/test = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)} "
              f"(usable {int(mask.sum())}, target_dim {targets.shape[1]}, states={'yes' if states is not None else 'no'})")
        if len(idx_tr) < 5 or len(idx_te) < 1:
            raise SystemExit(f"too few rows for split (train {len(idx_tr)}, test {len(idx_te)})")

        summary = []
        for modality in args.modalities:
            X = tucker[:, MODALITY_INDEX[modality], :]
            Xz, mean, std = standardize(X, idx_tr)
            SCALER_STORE[modality] = {"mean": mean.tolist(), "std": std.tolist()}
            res = train_one(modality, Xz, targets, states, weights, idx_tr, idx_va, idx_te,
                            metadata, target_names, args, device, run_dir, bundle["config"], diagnostics)
            summary.append(res)
            print(f"  {modality:9s} {res['status']} test_MAE={res['test_mae']:.4f} "
                  f"(train_mean {res['baseline_train_mean_mae']:.4f}, beats={res['beats_train_mean']}) "
                  f"R2={res['test_r2']:.3f} macroF1={res['classification_macro_f1']}")
        _write_csv(run_dir / "modalities_summary.csv", summary)
        (run_dir / "run_summary.json").write_text(json.dumps({
            "label_proxy": args.label_proxy, "split_mode": args.split_mode,
            "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
            "n_sessions": len({metadata[i]["session_id"] for i in range(len(metadata)) if mask[i]}),
            "n_users": len({str(metadata[i].get("user_id")) for i in range(len(metadata)) if mask[i]}),
            "target_names": target_names, "coverage": diagnostics.get("coverage", {}).get("coverage_pct"),
            "modalities": summary}, indent=2, default=str), encoding="utf-8")
        print(f"[step6] -> {run_dir}")
    else:
        # LOSO: implemented but only run when explicitly requested
        folds = loso_folds(metadata, mask, args.seed, args.min_loso_rows)
        fold_summary = []
        for fold in folds:
            tu = fold["test_user"]
            if fold.get("skip"):
                print(f"  [loso] skip {tu}: {fold['reason']}"); fold_summary.append({"test_user": tu, "skipped": True, "reason": fold["reason"]}); continue
            fdir = run_dir / f"fold_{tu}"
            for modality in args.modalities:
                X = tucker[:, MODALITY_INDEX[modality], :]
                Xz, mean, std = standardize(X, fold["train_idx"])
                SCALER_STORE[modality] = {"mean": mean.tolist(), "std": std.tolist()}
                res = train_one(modality, Xz, targets, states, weights, fold["train_idx"], fold["val_idx"],
                                fold["test_idx"], metadata, target_names, args, device, fdir, bundle["config"], diagnostics)
                fold_summary.append({"test_user": tu, **res})
        _write_csv(run_dir / "loso_summary.csv", fold_summary)
        print(f"[step6] LOSO -> {run_dir}")


if __name__ == "__main__":
    main()
