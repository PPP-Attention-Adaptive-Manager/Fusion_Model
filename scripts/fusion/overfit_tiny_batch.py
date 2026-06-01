"""Tiny overfit test for the fusion training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fusion_data.dataset import FusionWindowDataset, fusion_collate_fn
from fusion_model import InferrerFusion
from scripts.fusion.train_fusion import compute_fusion_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit 8 fusion samples to catch training bugs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = FusionWindowDataset(args.data_dir, limit_samples=args.samples, switching_device="cpu")
    if len(dataset) < args.samples:
        raise RuntimeError(f"Need at least {args.samples} samples, got {len(dataset)}.")

    batch = fusion_collate_fn([dataset[i] for i in range(args.samples)])
    for key in ["mouse", "keyboard", "notif", "switching", "factors", "state_label"]:
        batch[key] = batch[key].to(device)

    model = InferrerFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    initial_loss = None
    final_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.reset_subject()
        optimizer.zero_grad(set_to_none=True)
        output = model([batch["mouse"], batch["keyboard"], batch["notif"], batch["switching"]])
        losses = compute_fusion_loss(output, batch["factors"], batch["state_label"])
        loss = losses["loss"]
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    assert initial_loss is not None and final_loss is not None
    passed = final_loss < 0.2 * initial_loss
    result = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "ratio": final_loss / max(initial_loss, 1e-12),
        "passed": passed,
        "samples": args.samples,
        "epochs": args.epochs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
