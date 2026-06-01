"""Tiny overfit test for the switching predictive model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from predictive_models.switching.v1_gru_switching import SwitchingGRU
from scripts.switching.train_switching_predictive import (
    INPUT_FLAT_DIM,
    compute_loss,
    load_training_arrays,
    prepare_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit 8 switching Tucker slices.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_for_training"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    X, labels, _, resolved_dir = load_training_arrays(args.data_dir)
    factors, states = prepare_targets(labels)
    if X.shape[0] < args.samples:
        raise RuntimeError(f"Need at least {args.samples} samples, got {X.shape[0]}.")

    X_tiny = X[: args.samples]
    mean = X_tiny.mean(axis=0).astype(np.float32)
    std = X_tiny.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    X_tiny = ((X_tiny - mean) / std).astype(np.float32)

    x = torch.from_numpy(X_tiny).to(device)
    y_factors = torch.from_numpy(factors[: args.samples].astype(np.float32)).to(device)
    y_states = torch.from_numpy(states[: args.samples].astype(np.int64)).to(device)

    model = SwitchingGRU(input_flat_dim=INPUT_FLAT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    initial_loss = None
    final_loss = None
    for _ in range(args.epochs):
        model.train()
        model.reset_microstate()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        losses = compute_loss(output, y_factors, y_states)
        loss = losses["loss"]
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    assert initial_loss is not None and final_loss is not None
    passed = final_loss < 0.2 * initial_loss
    result = {
        "data_dir": str(resolved_dir),
        "samples": args.samples,
        "epochs": args.epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "ratio": final_loss / max(initial_loss, 1e-12),
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
