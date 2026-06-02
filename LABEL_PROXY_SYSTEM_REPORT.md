# Label Proxy System Report (STEP 5.6)

> Configurable supervision layer built **before** STEP 6 so training never hardcodes one label
> assumption. **No models were trained**; embedders/Tucker/GNN/notif/fusion and
> `data_training_full_rebuilt/` are untouched. **All targets here are PROXY labels with documented
> modeling assumptions — none is ground truth.**

## 1. Why a label proxy layer is needed
This corpus has two very different supervision sources, each flawed:
- **NASA-TLX** — 100 % coverage but **session-level** (one questionnaire copied to every window).
- **Dual-task reaction time** — genuinely **window-level** but **sparse** (~26 % of windows).

Hardcoding either into trainers would bias every downstream result and make experiments
non-comparable. The proxy layer makes the supervision assumption an explicit, swappable config.

## 2. Supported proxies
`scripts/labels/label_proxy.py` → `build_label_proxy(data_dir, proxy, config)` returning
`targets, state_labels, sample_weights, mask, progress, target_names, proxy_name, config, metadata,
diagnostics`. Rows are never dropped — unusable rows are **masked** and counted.

| proxy | level | target | coverage on this dataset |
|---|---|---|---|
| `nasa_tlx` | session | 5 NASA factors → [0,1] | **806/806 (100 %)** |
| `dual_task_rt` | window | 1 scalar RT-load | **211/806 (26.2 %)** |
| `nasa_time_weighted` | session + time weight | 5 NASA factors, weighted | 806/806 (weights 0.2–1.0) |
| `hybrid_rt_nasa` | window+session | 1 scalar load | 806 usable (211 with RT) |

## 3. Formula for each proxy
**nasa_tlx** — `targets = NASA[[mental,temporal,effort,frustration,arousal]]/100` (clip [0,1]);
`sample_weight = 1`. State (optional) via the project `derive_state_label` (5class) → 3class map.

**dual_task_rt** — over rows where `dual_task_available`:
`rt_norm = robust_unit(reaction_time_mean)` (median/IQR z, clip ±3, → [0,1]); miss-only windows (no valid
RT) impute `rt_norm = 1.0`.
`target = 0.8·rt_norm + 0.2·miss_rate + 0.0·error_rate` (clip [0,1]). `mask = available`,
`sample_weight = 1`. State (optional): binary `>0.5` or 3-class digitize.

**nasa_time_weighted** — NASA factor targets (as above); `sample_weight = f(progress)` rescaled to
`[min_weight, max_weight]`, where `progress = (window_center − session_start)/(session_end − session_start)`
and `f ∈ {linear, quadratic, sigmoid(k), exponential(k)}`. Rationale: TLX is answered at session end, so
later windows may better reflect the reported state — **this is a hypothesis, not a fact.**

**hybrid_rt_nasa** (`single_load`) — `nasa_load = Σ wᵢ·NASAᵢ/100` (default weights mental .25, temporal
.25, effort .20, frustration .20, arousal .10); `rt_load` as in dual_task_rt.
If RT present: `target = α·rt_load + (1−α)·nasa_load` (α=0.7), `weight = 1`.
Else: `target = nasa_load`, `weight = time_weight(progress)`. `mask` = all NASA-valid rows.
`multi_task` mode is a **documented TODO** (falls back to `single_load`).

## 4. Config options
Each proxy accepts a config dict (CLI flags `--weight-function`, `--state-mode`, `--alpha-rt`, or
`--config-json`). Defaults match the spec: `rt_weight=0.8, miss_weight=0.2, error_weight=0.0`,
`alpha_rt=0.7`, sigmoid `k=8`, `min_weight=0.2, max_weight=1.0`, NASA factor weights as above. Full
resolved config is saved to `label_proxy_config.json`.

## 5. Diagnostics generated
Per proxy: `target_dim`, `target_mean`, masked/excluded counts, coverage (overall + by user + by session),
RT robust stats, and (hybrid) `rows_with_rt`, `rows_without_rt`, `rt_vs_nasa_load_corr`. Plus per-row CSV
(`label_proxy_rows.csv`) and arrays (`targets.npy`, `sample_weights.npy`, `mask.npy`, `state_labels.npy`).

## 6. How to inspect proxies
```powershell
python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt --proxy nasa_tlx --output-dir outputs/label_proxies/nasa_tlx
python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt --proxy dual_task_rt --output-dir outputs/label_proxies/dual_task_rt
python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt --proxy nasa_time_weighted --weight-function sigmoid --output-dir outputs/label_proxies/nasa_time_weighted_sigmoid
python scripts/labels/inspect_label_proxy.py --data-dir data_training_full_rebuilt --proxy hybrid_rt_nasa --output-dir outputs/label_proxies/hybrid_rt_nasa
```
Plots per dir: `target_distribution`, `sample_weight_distribution`, `target_by_user`, `target_by_session`,
`coverage_by_user`, `session_progress_weight_curve` (time-weighted), `rt_vs_nasa_load` (hybrid),
`state_distribution` (if states). Notebook: `notebooks/label_proxy_dashboard.ipynb`.

## 7. How STEP 6 should consume proxies
STEP 6 trainers must obtain targets **only** via
`scripts/predictive/label_proxy_integration.load_targets_for_training(data_dir, label_proxy, config)`:
```python
b = load_targets_for_training("data_training_full_rebuilt", "nasa_time_weighted", {"weight_function":"sigmoid"})
mask, targets, states, weights = b["mask"], b["targets"], b["state_labels"], b["sample_weights"]
# train only on mask; per-sample weighted loss:
#   loss = (weights[mask] * per_sample_loss(pred[mask], targets[mask])).mean()
```
The proxy + config become CLI/experiment parameters → no label assumption is baked into trainers.

## 8. Current coverage on `data_training_full_rebuilt`
- `nasa_tlx` / `nasa_time_weighted`: **806/806 (100 %)** usable.
- `dual_task_rt`: **211/806 (26.2 %)** usable (window-level RT exists for 26 % of windows).
- `hybrid_rt_nasa`: **806 usable**, of which **211 have RT**, **595 NASA-only**.
- Tests: `tests/test_label_proxy.py` → **8/8 pass** (build, coverage, monotonic time-weight, row counts,
  no-NaN-in-mask, finite/non-negative weights, determinism).

## 9. Caveats (do not hide)
- **NASA is session-level** → constant within a session; window-level structure is absent for it.
- **Dual-task is sparse** (~26 %) and, in this corpus, has **1 probe/window**, binary `miss_rate`, and
  `error_rate ≈ 0` — a thin signal.
- **Time weighting is a hypothesis** (later windows ≈ reported state); it is unproven on this data.
- **Hybrid is a modeling assumption.** Importantly, where both exist, **RT-load and NASA-load are weakly
  *negatively* correlated (r ≈ −0.14)** — the two supervision signals largely *disagree*. The hybrid
  target therefore blends two sources that do not agree; interpret hybrid results with caution.
- None of these proxies is validated against an external cognitive-load ground truth.

## 10. Recommended first STEP 6 experiments
Run these as **parallel, comparable** experiments (same splits, same models), each via the proxy layer:
1. `nasa_tlx` — baseline (100 % coverage, session-level).
2. `nasa_time_weighted` (sigmoid) — tests the end-of-session weighting hypothesis vs the baseline.
3. `dual_task_rt` — pure window-level regression on the 26 % labelled windows (sparse but genuine).
4. `hybrid_rt_nasa` (single_load, α=0.7) — combined, **read against the r≈−0.14 caveat above**.

Compare with `session_within_user` first, then LOSO (user_id coverage is complete). The proxy that both
generalizes across held-out users **and** is internally consistent should drive the production choice.
