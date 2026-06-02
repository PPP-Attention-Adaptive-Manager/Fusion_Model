# Data-Training Alignment Audit: `data/` → `data_training/`

**Type:** read-only audit. No models trained, no `data/` or `data_training/` files modified.
**Script:** [scripts/audit/audit_data_training_alignment.py](scripts/audit/audit_data_training_alignment.py)
**Evidence:** 14 CSVs in [outputs/audit/](outputs/audit/) (regenerate with the command at the end).
**Source of truth:** `data/` (record-tool output). **Derived/suspect:** `data_training/`.

---

## 1. Executive summary

The processed dataset `data_training/` is **internally consistent and correctly aligned**, but it is a
**partial subset** of what the record tool produced. The two findings, side by side:

| Question | Verdict |
|---|---|
| Are `tucker_slices`, `nasa_tlx_labels`, `dual_task_window_labels` row-aligned with `metadata.json`? | ✅ **Yes — perfectly** (371 = 371 = 371 = 371; 0 row mismatches) |
| Are metadata windows aligned with the graph JSON windows? | ✅ **Yes — 100%** (371/371 found, 371/371 timestamp-aligned ≤ 1 s; in fact exact) |
| Are timestamp units consistent (s / ms / relative)? | ✅ **Yes — Unix seconds throughout; no scale/offset issue** |
| Is the low dual-task coverage (97/371) a matching artifact? | ❌ **No — it is real** (direct match = 97 = current; no other mode finds more) |
| Does every covered window truly have exactly one probe? | ✅ **Yes — real** (max = 1 probe/window under *every* timestamp mode) |
| **Is `data_training/` missing sessions from `data/`?** | ⚠️ **YES — 48 of 73 raw sessions were dropped** |

**Bottom line:** Nothing is *misaligned*. The problem is *incompleteness* — preprocessing kept only
**25 of 73** recorded sessions, discarding ~**435 graph windows** and ~**185 dual-task events** that exist
in `data/`. The weak dual-task supervision seen earlier is therefore driven by (a) genuinely sparse
probes in the kept sessions and (b) a large amount of valid recorded data never entering training.

---

## 2. Raw data inventory summary  ([raw_session_inventory.csv](outputs/audit/raw_session_inventory.csv))

- **73** raw `session_*` directories.
- **54** contain a `data_graph/data_graph_120s/` folder (graph windows).
- **56** have at least one dual-task probe event; the rest have an empty `dual_task.csv`.
- Every session exposes only `device_id` in its CSVs — **`user_id` does not exist in raw data**; it is a
  processed-only mapping (see §11 caveat).
- Dual-task discovery ([dual_task_file_candidates.csv](outputs/audit/dual_task_file_candidates.csv)):
  `raw/dual_task.csv` exists in all 73 sessions, `labels/dual_task.csv` in 59. No root-level file. The
  selector prefers the first candidate that has events (ranked root → raw → events → features), so
  `raw/dual_task.csv` is consistently chosen. Raw vs labels copies have identical event *counts*.
- Dual-task timestamps: all in the **Unix-seconds** band (~1.7e9–1.8e9). No millisecond or relative files.

## 3. Processed data inventory summary  ([processed_metadata_inventory.csv](outputs/audit/processed_metadata_inventory.csv))

- `metadata.json`: **371** rows; `tucker_slices.npy`: **(371, 4, 512)**; `nasa_tlx_labels.npy`: **(371, 9)**;
  `dual_task_window_labels.csv`: **371** rows.
- `len(metadata) == tucker.shape[0] == nasa.shape[0] == len(dtwl)` → **all True**.
- **10** unique users, **25** unique sessions. No missing `user_id`/`session_id`/`window_idx`/
  `window_start`/`window_end` fields.
- Windows per user ([processed_user_summary.csv](outputs/audit/processed_user_summary.csv)): dem **161**,
  layss 64, hffz 41, amrdroid/ignorant/ladabos/mohanned 16, Makki/ghoss 15, Graja 11 — heavily skewed.

## 4. Raw vs processed session coverage  ([session_coverage_diff.csv](outputs/audit/session_coverage_diff.csv), [session_alignment_summary.csv](outputs/audit/session_alignment_summary.csv))

| metric | value |
|---|---|
| raw sessions | 73 |
| processed sessions | 25 |
| **sessions in raw but NOT processed** | **48** |
| sessions in processed but not raw | 0 |

Status breakdown (all 73 sessions):

| status | count |
|---|---|
| `missing_from_processed` | **48** |
| `low_dual_task_coverage` | 16 |
| `raw_dual_task_empty` | 9 |

All 25 processed sessions are present in raw (no orphan processed sessions). What the **48 dropped**
sessions contained:

- **29** had 120s graph folders → **~435 graph windows** discarded.
- **40** had dual-task events → **~185 dual-task probe events** discarded.

This is the single most important result: a large fraction of valid recorded behaviour — including most of
the dual-task probes in the corpus — never entered `data_training/`.

## 5. Row alignment checks  ([row_alignment_check.csv](outputs/audit/row_alignment_check.csv))

- For all 371 rows, `tucker_row_exists` and `nasa_label_row_exists` are **True**.
- `dual_task_window_labels.csv` matches metadata on `(session_id, window_idx, user_id)` for **every** row;
  **0 mismatches**.
- **Conclusion:** `tucker_slices.npy` and `nasa_tlx_labels.npy` rows **are** aligned with `metadata.json`,
  and the dual-task label file is correctly keyed to the same rows. The processed dataset's internal
  bookkeeping is sound.

## 6. Graph ↔ metadata alignment  ([graph_metadata_alignment.csv](outputs/audit/graph_metadata_alignment.csv))

Convention confirmed: `graph_001.json` carries `window.window_id = w000000` → `window_idx 0`
(`graph_{idx+1}`). Using `window.window_start/window_end`:

| metric | value |
|---|---|
| total rows | 371 |
| graph_found | 371 (100%) |
| timestamp_aligned (≤ 1.0 s) | 371 (100%) |
| missing graph | 0 |
| large timestamp mismatch | 0 |

For every processed session, `raw_graph_count == processed_window_count` (0 mismatches). So **within the
kept sessions, all graph windows are represented and exactly aligned**. (Windows are only "missing" at the
*session* level — the 48 dropped sessions — not within kept sessions.)

## 7. Dual-task timestamp alignment  ([timestamp_alignment_report.csv](outputs/audit/timestamp_alignment_report.csv))

Across the 25 processed sessions:

| suspected_issue | count |
|---|---|
| `none` (direct overlap works) | 16 |
| `no_dual_task` (empty file) | 9 |
| `ms_vs_seconds` / `relative_time` / `shifted_origin` / `no_overlap` | **0** |

There is **no timestamp-unit or origin problem**. Where dual-task events exist, they overlap the metadata
windows directly in Unix seconds.

## 8. Dual-task matching mode comparison  ([dual_task_matching_mode_summary.csv](outputs/audit/dual_task_matching_mode_summary.csv), [dual_task_matching_debug.csv](outputs/audit/dual_task_matching_debug.csv))

Total probe events matched into windows, by interpretation:

| mode | total matched |
|---|---|
| current labels (`dual_task_available`) | 97 |
| **direct** (`ws ≤ t < we`) | **97** |
| milliseconds | 0 |
| relative-to-first-window | 0 |
| shifted_to_dual_task_min | 97 |
| shifted_from_dual_task_min | 97 |

The direct interpretation reproduces the existing 97 labels exactly. Alternative interpretations find
**no additional** events (the "shifted" modes equal direct only because the computed shift is ≈ 0, i.e. the
clocks already coincide). **Low coverage is not a matching bug** — it is the true number of probes that
fall inside kept windows.

## 9. Zero-coverage user investigation  ([zero_coverage_user_report.csv](outputs/audit/zero_coverage_user_report.csv))

| user | processed sessions | windows | raw dual-task files | raw events | direct/ms/rel matches | likely reason |
|---|---|---|---|---|---|---|
| Graja | 1 | 11 | 1 | **0** | 0 / 0 / 0 | dual_task.csv present but **empty** |
| ladabos | 1 | 16 | 1 | **0** | 0 / 0 / 0 | dual_task.csv present but **empty** |
| mohanned | 1 | 16 | 1 | **0** | 0 / 0 / 0 | dual_task.csv present but **empty** |

Each of these users has exactly one processed session, and that session's `dual_task.csv` contains **zero
probe events** in `data/` itself. Their zero coverage is **real and originates in the raw recording**, not
in preprocessing or timestamp handling. (Caveat: because `user_id` is processed-only, we can only see the
*one* session per user that was kept; if these users recorded other sessions, those are among the 48
dropped and cannot be attributed to a user from raw data alone — see §11.)

## 10. Probe-count reality check  ([probe_count_reality_check.csv](outputs/audit/probe_count_reality_check.csv))

| mode | 0 probes | 1 probe | 2+ probes | max | mean/labelled |
|---|---|---|---|---|---|
| direct | 274 | 97 | **0** | 1 | 1.0 |
| ms | 371 | 0 | 0 | 0 | 0 |
| relative | 371 | 0 | 0 | 0 | 0 |
| shifted_to | 274 | 97 | 0 | 1 | 1.0 |
| shifted_from | 274 | 97 | 0 | 1 | 1.0 |

"Exactly one probe per covered window" is **real**: no timestamp interpretation ever places 2+ probes in a
single 120 s window. The probe scheduler simply fires roughly once per ~200–285 s (sparser than the 120 s
window), so a window holds 0 or 1 probe. This also explains the binary `miss_rate` and all-zero
`error_rate` seen downstream.

## 11. Root-cause diagnosis

1. **Primary defect — session-level dropping during preprocessing.** `data_training/` was built from only
   **25 of 73** recorded sessions. The dropped 48 include 29 with graphs (~435 windows) and 40 with
   dual-task events (~185 probes). This is why the corpus looks small and dual-task-poor. **This is a
   completeness defect in the `data/` → `data_training/` transform, not a misalignment.**
2. **Secondary, and genuine — sparse probes.** Within kept sessions, probes are intrinsically sparse
   (≤ 1 per 120 s window) and 9 kept sessions recorded no probes at all. So even a complete rebuild would
   yield modest per-window dual-task density.
3. **Not a problem:** row alignment, graph alignment, timestamp units, and the matching logic are all
   correct. The earlier coverage numbers were accurate for the data that was included.

**Caveat / uncertainty (stated explicitly):** raw sessions carry only `device_id`; `user_id` lives only in
`metadata.json`. We therefore **cannot attribute the 48 dropped sessions to specific users** from raw data
alone, nor confirm whether the drop was intentional (e.g. a quality filter) or accidental. A
`device_id → user_id` map is needed to close this. The audit also cannot tell *why* sessions were
dropped — only that they were.

## 12. Required fixes before re-training the switching predictive model

1. **Recover the dropped sessions.** Investigate the `data/` → `data_training/` builder and determine why
   48/73 sessions were excluded (missing graphs? a label/QC filter? a hard-coded session list?). Decide
   per-session whether each drop is intended.
2. **Rebuild `tucker_slices` / `metadata` / labels from the full eligible set** of sessions that have
   120s graphs (54 sessions, not 25), then re-run the dual-task label builder. Expected gain: on the order
   of the ~185 additional probe events currently discarded.
3. **Establish a `device_id → user_id` mapping** and store it with the raw data so dropped sessions and
   zero-coverage users can be attributed and audited per user.
4. **Accept that dual-task supervision will remain sparse** (≤ 1 probe/window, no error signal). For denser
   window-level load labels, the probe cadence must be increased at recording time; otherwise keep
   switching as a **weak expert** in fusion (consistent with the earlier supervision experiment).
5. **Do not "fix" timestamps** — they are correct. Any rebuild should keep the direct Unix-seconds
   `ws ≤ t < we` matching, which this audit validated.

### Direct answers to the required questions

- **Is `data_training` missing sessions from `data/`?** **Yes — 48 of 73.**
- **Are `tucker_slices.npy` rows aligned with `metadata.json`?** **Yes (371 = 371, exact).**
- **Are `nasa_tlx_labels.npy` rows aligned with `metadata.json`?** **Yes (371 = 371, exact).**
- **Are graph windows aligned with metadata?** **Yes — 100% found and timestamp-exact for kept sessions.**
- **Are `dual_task.csv` files missing?** **No** — present in every session (raw/), but 9 kept sessions'
  files are empty; many *populated* files belong to dropped sessions.
- **Are dual-task timestamps mismatched?** **No — consistent Unix seconds, direct overlap.**
- **Is low dual-task coverage real?** **Yes** — confirmed under all timestamp interpretations; *but* it is
  made worse by the 48 dropped sessions, which removed most of the corpus's probes.
- **Why do Graja / ladabos / mohanned have zero coverage?** Their single kept session each has an **empty
  `dual_task.csv`** in `data/` (0 recorded probes). Real, not a processing bug.
- **Does every covered window truly have exactly one probe?** **Yes** (max = 1 under every mode).
- **What to fix before re-training?** Recover/rebuild from the full eligible session set; add a
  device→user map; keep timestamp matching as-is; treat switching as a weak expert until probe density
  improves.

---

## Reproduce

```bash
python scripts/audit/audit_data_training_alignment.py \
  --data-training-dir data_training \
  --raw-data-dir data \
  --output-dir outputs/audit
```

Optional visualization: [notebooks/data_training_alignment_audit.ipynb](notebooks/data_training_alignment_audit.ipynb).
