# `data_training/` Rebuild Report

**Type:** dataset rebuild + diagnosis. No models trained; no GNN / `fusion_model.py` / predictive /
TFN architecture files modified; `data/` and the original `data_training/` left untouched; timestamp
logic unchanged (direct Unix seconds). The rebuild is written to a **new** folder `data_training_rebuilt/`.

**Scripts**
- Builder trace: [outputs/audit/data_training_builder_trace.md](outputs/audit/data_training_builder_trace.md)
- Rebuild: [scripts/build/rebuild_data_training.py](scripts/build/rebuild_data_training.py)
- Validation audit (reused): [scripts/audit/audit_data_training_alignment.py](scripts/audit/audit_data_training_alignment.py)

---

## 1. Why the rebuild was needed

Two independent defects in v1 `data_training/`:

1. **Incomplete sessions.** v1 contained **25 of 73** sessions (371 windows). An eligibility scan shows
   **54 sessions are eligible** (have 120 s graphs *and* labels) — so **29 eligible sessions were dropped**.
2. **All-zero feature matrix (the severe one).** v1 `tucker_slices.npy` is **entirely zeros**
   (`min = max = sum = 0`, all 371×4×512 entries). There are **no trained mouse/keyboard pre-embedder
   weights** in the repo, so both modalities zero-fall-back; the Tucker fusion is an **outer product**, so
   a zero in any modality zeroes the whole tensor — including the switching slice. Every switching /
   dual-task model trained earlier was trained on **identically-zero inputs**, which fully explains the
   near-chance results.

## 2. Existing builder trace (summary)

No builder script is checked into the repo — v1 was produced by an uncommitted "AAM Tucker Outputs v1"
process. The only reusable in-repo pipeline is `fusion_data/dataset.py::FusionWindowDataset` (real
GraphSAGE switching, ONNX notif, zero-fallback mouse/keyboard) + `TFN.ActiveTFN` (rank-8 Tucker). Full
detail in [data_training_builder_trace.md](outputs/audit/data_training_builder_trace.md).

## 3. Why old `data_training/` had only 25 sessions

`FusionWindowDataset` skips a session unless it has **both** 120 s graphs and a label file — matching the
documented drops ("no NASA-TLX (13)", "no graphs (7)"). **But 54 sessions satisfy that rule, not 25.** The
extra 29 drops are not explained by any in-repo code; the uncommitted v1 builder most likely applied a
stricter label rule (e.g. required a root-level `labels.csv`, ignoring `raw/labels.csv`) or a hard-coded
list. **Conclusion: the v1 session filtering was at least partly accidental / over-strict.**

## 4. Eligibility criteria (Part 2)

Implemented in `scan_eligibility()` →
[session_eligibility_rebuild.csv](outputs/audit/session_eligibility_rebuild.csv):

- `eligible_nasa` = has ≥1 `data_graph_120s/graph_*.json` **AND** a usable label file
  (`labels.csv` / `labels/nasa_tlx.csv` / `raw/labels.csv`). Dual-task **not** required.
- `eligible_dual_task` = `eligible_nasa` **AND** `dual_task.csv` has ≥1 event.

## 5. Session eligibility summary

| | sessions |
|---|---|
| raw `session_*` dirs | 73 |
| **eligible_nasa (included in rebuild)** | **54** |
| eligible_dual_task | 42 |
| excluded | 19 |

**Every exclusion has a recorded reason** (all are missing 120 s graphs):

| exclusion_reason | count |
|---|---|
| `no_graph_120s;no_labels` | 13 |
| `no_graph_120s` | 6 |

No session was silently dropped; no random filtering was used; sessions and windows are ordered
deterministically (sorted session dirs, sorted graphs).

## 6. Old vs rebuilt comparison (Part 8 — [old_vs_rebuilt_data_training.csv](outputs/audit/old_vs_rebuilt_data_training.csv))

| metric | old | rebuilt | added |
|---|---|---|---|
| windows (N) | 371 | **806** | **+435** |
| sessions | 25 | **54** | **+29** |
| graph windows (eligible) | 371 | 806 | +435 |
| dual-task labelled windows | 97 | **211** | **+114** |
| distinct users/devices | 10 (users) | 19 (devices) | — |

## 7. Recovered sessions / windows / probes

- **+29 sessions**, **+435 graph windows**, **+114 dual-task-labelled windows** recovered.
- Rebuilt dual-task coverage ([rebuilt_dual_task_coverage.csv](outputs/audit/rebuilt_dual_task_coverage.csv)):
  **211 / 806 windows = 26.2 %** (similar *ratio* to v1's 26.1 %, because probes are intrinsically
  sparse — but **2.2× more labelled windows in absolute terms**). Probe-count distribution is still
  `{1: 211}` — exactly one probe per covered window — consistent with the audit's finding that this is
  real (probe cadence ~200–285 s > 120 s window), not a matching artifact.

## 8. `device_id` / `user_id` mapping limitations (Part 4)

Raw sessions expose only `device_id`; `user_id` lived only in v1 metadata. Per the rules, the rebuild
**does not fabricate `user_id`** — it stores `device_id` and sets `user_id = null` unless a
`--device-user-map device_id,user_id` CSV is supplied. None was provided here, so all 806 rows have
`user_id = null` and `device_id` populated. Devices needing a mapping are listed in
[missing_device_user_map.csv](outputs/audit/missing_device_user_map.csv) (19 devices, with session counts,
graph-window counts, and whether they have dual-task events). **User-level analysis (LOSO, per-user
balancing) requires this map before re-training.**

## 9. Label extraction summary (Part 5 — [label_extraction_report.csv](outputs/audit/label_extraction_report.csv))

For each included session the 9 NASA-TLX columns
(`mental_demand … arousal`, 0–100) are read from the label file and repeated across that session's
windows (expected for session-level NASA labels). All 54 included sessions had the core demand columns
present (`usable = True`). Sessions without a usable label file are not eligible and were excluded with a
reason (§5), so `nasa_tlx_labels.npy` contains no fabricated label rows.

## 10. Tucker slice validation (Part 6 — [tucker_rebuild_validation.csv](outputs/audit/tucker_rebuild_validation.csv))

`tucker_slices.npy` shape **(806, 4, 512)**, no NaN, no Inf.

| modality (idx) | variance | rows_nonzero | note |
|---|---|---|---|
| mouse (0) | 0.0 | 0 / 806 | zero-fallback (no embedder weights) → Tucker collapse |
| keyboard (1) | 0.0 | 0 / 806 | same |
| notif (2) | 0.0 | 0 / 806 | real ONNX embedding, but slot zeroed by Tucker outer-product with zero mouse/kb |
| **switching (3)** | **0.0148** | **789 / 806** | **real GraphSAGE 64-D via seeded 64→512 projection + tanh** |

- **Switching slice variance > 0 ✓** (requirement met). The 17 zero rows are cold-start windows (the
  switching encoder emits a zero embedding on cold start); these are genuine, not errors.
- Slots 0–2 remain zero **by design / structural necessity** — they cannot be non-zero until trained
  mouse/keyboard embedders exist (notif alone is annihilated by the outer-product). This is documented in
  `build_manifest.json`.
- **Raw switching embeddings** saved separately as `switching_embeddings.npy` **(806, 64)**, L2 norm
  ≈ 0.98 (≈1.0 except cold-start zeros) — a clean, Tucker-independent switching feature.

## 11. Dual-task coverage after rebuild

211 labelled windows (26.2 %), one probe each. Because `user_id` is null without a device map, per-user
coverage is not yet attributable; per-session coverage is in
[rebuilt_dual_task_coverage.csv](outputs/audit/rebuilt_dual_task_coverage.csv). Dual-task labels were
produced by the **existing** builder (`build_dual_task_window_labels.py`) with **unchanged direct
Unix-seconds matching**.

## 12. Validation audit on rebuilt data (Part 9 — `outputs/audit_rebuilt/`)

Re-running the alignment audit on `data_training_rebuilt/`:

- `metadata == tucker == nasa == dual_task_labels == 806`; **0 row mismatches**.
- graph found 806/806 (100 %); timestamp aligned 806/806 (100 %); 0 large mismatches.
- dual-task matching: direct = 211 = current; ms/relative = 0 → **no timestamp issue**.
- raw-not-processed sessions: **19** (exactly the ineligible no-graph sessions).

## 13. Remaining limitations

1. **mouse/keyboard slots are zero** — no trained embedders exist. Until they do, only the switching slice
   (and `switching_embeddings.npy`) carries signal; full 4-modality fusion cannot be reproduced.
2. **`user_id` is null** for all rows pending a `device_id → user_id` map → LOSO / per-user balancing not
   yet possible on the rebuilt set.
3. **Dual-task remains sparse** (one probe/window, `error_rate` ≈ 0) — a recording-side limitation, not a
   processing one.
4. The **switching slice uses a seeded projection**, not the trained Tucker interaction context (which is
   unavailable while other modalities are zero); absolute slice values are init-dependent but
   deterministic (seed 1234, recorded in `build_manifest.json`).

## 14. Exact commands

**Rebuild:**
```bash
python scripts/build/rebuild_data_training.py \
  --raw-data-dir data \
  --output-dir data_training_rebuilt \
  --window-size 120s \
  --modality switching \
  --include-all-eligible \
  --device cpu
# optional, once available:  --device-user-map device_user_map.csv
```

**Validate the rebuild:**
```bash
python scripts/audit/audit_data_training_alignment.py \
  --data-training-dir data_training_rebuilt \
  --raw-data-dir data \
  --output-dir outputs/audit_rebuilt
```

**Re-run dual-task switching training on the rebuilt data** (uses the switching slice = real GraphSAGE
signal). Train LOSO only after a device→user map is supplied; otherwise use the within-session split:
```bash
# 1. dual-task labels already built into data_training_rebuilt/ by the rebuild script.
# 2. within-user/session split (works without user_id):
python scripts/switching/train_switching_predictive.py \
  --data-dir data_training_rebuilt \
  --task dual_task_regression \
  --dual-task-labels data_training_rebuilt/dual_task_window_labels.csv \
  --dual-task-target relative \
  --split-mode session_within_user \
  --sample-weighting user_balanced \
  --device cpu \
  --output-dir outputs/switching_dual_task_rebuilt_within

python scripts/switching/evaluate_dual_task_switching.py \
  --data-dir data_training_rebuilt \
  --checkpoint-dir outputs/switching_dual_task_rebuilt_within \
  --dual-task-labels data_training_rebuilt/dual_task_window_labels.csv \
  --split-mode session_within_user --device cpu

# 3. LOSO — only after data_training_rebuilt/metadata.json has real user_id values:
#    rebuild with --device-user-map, then train with --split-mode loso.
```

---

## Required explicit statements

- **Was old `data_training` incomplete?** **Yes** — 25 of 73 sessions, and its `tucker_slices.npy` was
  entirely zero (mouse/keyboard never embedded; Tucker outer-product annihilated all slices).
- **Does the rebuilt dataset include all eligible sessions?** **Yes** — all **54** sessions with 120 s
  graphs and labels (806 windows). The remaining 19 are excluded only for missing 120 s graphs, each with
  a recorded reason.
- **Were timestamps unchanged?** **Yes** — direct Unix-seconds matching, untouched; alignment audit on the
  rebuild is 100 %.
- **Were any sessions excluded, and why?** 19, all for `no_graph_120s` (13 of them also lack labels). No
  silent drops, no random filtering.
- **Does `user_id` mapping remain incomplete?** **Yes** — `user_id` is null pending a `device_id → user_id`
  map (`missing_device_user_map.csv` lists the 19 devices). No `user_id` was fabricated.
- **Should training be re-run on `data_training_rebuilt`?** **Yes** — it is the first dataset with a
  non-zero, real switching signal and 2.2× more dual-task labels. Use the within-session split immediately;
  use LOSO after supplying the device→user map. Note that mouse/keyboard/notif fusion slots remain zero
  until embedder weights exist, so this rebuild specifically unblocks **switching** predictive training.
