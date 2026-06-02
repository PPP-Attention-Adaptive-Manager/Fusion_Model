"""Integration hook so STEP 6 training never hardcodes a label assumption.

STEP 6 trainers should obtain targets ONLY through this function, passing the
chosen proxy + config from the command line / experiment file.

Example (inside a future STEP 6 trainer):

    from scripts.predictive.label_proxy_integration import load_targets_for_training

    bundle = load_targets_for_training(
        data_dir="data_training_full_rebuilt",
        label_proxy="nasa_time_weighted",
        label_proxy_config={"weight_function": "sigmoid", "state_mode": "5class"},
    )
    X = tucker_slices[:, modality_idx, :]          # standardize (train-only) before the model
    mask    = bundle["mask"]                        # train only on usable rows
    targets = bundle["targets"]                     # (N, target_dim) float32
    states  = bundle["state_labels"]                # (N,) or None
    weights = bundle["sample_weights"]              # (N,) float32 -> per-sample loss weight

    # per-sample weighted loss, restricted to mask:
    #   loss = (weights[mask] * per_sample_loss(pred[mask], targets[mask])).mean()
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.labels.label_proxy import build_label_proxy


def load_targets_for_training(
    data_dir: str,
    label_proxy: str,
    label_proxy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build targets for STEP 6 via the configurable label proxy.

    Returns a bundle with everything a trainer needs; NEVER drops rows silently
    (use ``mask`` to select usable rows).
    """
    res = build_label_proxy(data_dir, label_proxy, label_proxy_config)
    return {
        "X_mask": res["mask"],            # (N,) bool — alias requested by the spec
        "mask": res["mask"],
        "targets": res["targets"],        # (N, target_dim) float32
        "state_labels": res["state_labels"],  # (N,) int64 or None
        "sample_weights": res["sample_weights"],  # (N,) float32
        "target_names": res["target_names"],
        "proxy_name": res["proxy_name"],
        "config": res["config"],
        "metadata": res["metadata"],
        "diagnostics": res["diagnostics"],
    }


if __name__ == "__main__":
    import json
    b = load_targets_for_training("data_training_full_rebuilt", "nasa_time_weighted",
                                  {"weight_function": "sigmoid"})
    print(json.dumps({"proxy": b["proxy_name"], "target_names": b["target_names"],
                      "usable": int(b["mask"].sum()), "target_shape": list(b["targets"].shape),
                      "has_states": b["state_labels"] is not None}, indent=2))
