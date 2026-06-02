"""Tests for the label proxy system (STEP 5.6).

Run with pytest:  python -m pytest tests/test_label_proxy.py
or directly:      python tests/test_label_proxy.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.labels.label_proxy import build_label_proxy

DATA_DIR = PROJECT_ROOT / "data_training_full_rebuilt"


def _n():
    import json
    return len(json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8")))


def test_nasa_tlx_builds():
    r = build_label_proxy(DATA_DIR, "nasa_tlx")
    assert r["targets"].shape[0] == _n()
    assert r["targets"].dtype == np.float32
    assert r["mask"].all()  # every row has valid NASA
    assert r["state_labels"] is not None


def test_dual_task_rt_coverage_positive():
    r = build_label_proxy(DATA_DIR, "dual_task_rt")
    assert r["targets"].shape == (_n(), 1)
    assert int(r["mask"].sum()) > 0
    assert int(r["mask"].sum()) < _n()  # sparse, not all rows
    # masked rows must have finite target
    assert np.isfinite(r["targets"][r["mask"]]).all()


def test_nasa_time_weighted_increases_with_progress():
    r = build_label_proxy(DATA_DIR, "nasa_time_weighted", {"weight_function": "linear"})
    prog = r["progress"]; w = r["sample_weights"]; m = r["mask"]
    # Spearman-ish: correlation between progress and weight should be strongly positive
    p = prog[m]; ww = w[m]
    corr = float(np.corrcoef(p, ww)[0, 1])
    assert corr > 0.8, f"progress->weight corr too low: {corr}"


def test_hybrid_builds():
    r = build_label_proxy(DATA_DIR, "hybrid_rt_nasa")
    assert r["targets"].shape == (_n(), 1)
    assert r["mask"].all()  # NASA-valid everywhere -> all usable
    assert "rt_vs_nasa_load_corr" in r["diagnostics"]


def test_row_counts_equal_metadata():
    n = _n()
    for p in ["nasa_tlx", "dual_task_rt", "nasa_time_weighted", "hybrid_rt_nasa"]:
        r = build_label_proxy(DATA_DIR, p)
        assert r["targets"].shape[0] == n
        assert r["sample_weights"].shape[0] == n
        assert r["mask"].shape[0] == n


def test_no_nan_in_masked_targets():
    for p in ["nasa_tlx", "dual_task_rt", "nasa_time_weighted", "hybrid_rt_nasa"]:
        r = build_label_proxy(DATA_DIR, p)
        assert not np.isnan(r["targets"][r["mask"]]).any()


def test_sample_weights_finite_nonneg():
    for p in ["nasa_tlx", "dual_task_rt", "nasa_time_weighted", "hybrid_rt_nasa"]:
        r = build_label_proxy(DATA_DIR, p)
        w = r["sample_weights"]
        assert np.isfinite(w).all()
        assert (w >= 0).all()


def test_deterministic():
    for p in ["nasa_tlx", "dual_task_rt", "nasa_time_weighted", "hybrid_rt_nasa"]:
        a = build_label_proxy(DATA_DIR, p)
        b = build_label_proxy(DATA_DIR, p)
        assert np.array_equal(a["targets"], b["targets"])
        assert np.array_equal(a["sample_weights"], b["sample_weights"])
        assert np.array_equal(a["mask"], b["mask"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
