# Project Structure

Date: 2026-06-01

The project root is now kept for core packages and high-level entry files.

```text
Fusion_Model/
  fusion_model.py              # Core orchestrator: TFN -> predictive models -> PoE -> EMA
  requirements.txt
  README.md
  test_fusion.py

  TFN/                         # Fusion layers
  poe/                         # Product of Experts
  ema/                         # EMA smoothing and uncertainty helpers
  TCN_encoders/                # Buffered modality encoders
  pre_embedders/               # Modality pre-embedders and exported GNN encoder
  predictive_models/           # Per-modality predictive models
  fusion_data/                 # Dataset builder and session split helpers
  scripts/
    fusion/                    # Full fusion smoke/train/eval scripts
    switching/                 # Switching predictive train/eval scripts
  tests/                       # Unit/smoke tests
  notebooks/                   # Inspection and performance notebooks
  docs/                        # Architecture and training reports

  data/                        # Local data, ignored by git
  data_training/               # Local switching training arrays, ignored by git
  outputs/                     # Generated checkpoints/metrics, ignored by git
```

## Common Commands

Fusion smoke:

```powershell
python scripts/fusion/run_fusion_dataset_smoke.py --data-dir data --limit-sessions 10
```

Fusion training:

```powershell
python scripts/fusion/train_fusion.py --data-dir data --epochs 50 --batch-size 8 --device cuda
```

Switching LOSO training:

```powershell
python scripts/switching/train_switching_predictive.py --data-dir data_for_training --epochs 100 --batch-size 32 --device cpu --output-dir outputs/switching_predictive
```

Switching final single model:

```powershell
python scripts/switching/train_switching_final.py --data-dir data_for_training --epochs 100 --batch-size 32 --device cpu --output-dir outputs/switching_predictive/final
```

Switching evaluation:

```powershell
python scripts/switching/evaluate_switching_predictive.py --data-dir data_for_training --checkpoint-dir outputs/switching_predictive --device cpu
```

Tests:

```powershell
python test_fusion.py
python -m unittest tests.test_fusion_shapes
python -m unittest tests.test_switching_encoder_integration
```
