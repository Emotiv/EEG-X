## EEG-X: Device-Agnostic and Noise-Robust Foundation Model for EEG


#### EEG-X Paper: [comming soon](https:)


This is a PyTorch implementation of **EEG-X: Device-Agnostic and Noise-Robust Foundation Model for EEG**
<p align="center">
    <img src="Figs/EEG-X.png">
</p> 


## Setup

Use Python 3.12 or 3.13 and the repository environment:

```bash
uv sync --locked --dev
uv run --locked --no-active python -m pytest -q
uv run --locked --no-active python -m pretrain --help
```

## Code layout

The executable EEG-X code lives in one package:

```text
pretrain/
  __main__.py       command entrypoint
  models/           EEG-X encoder, Predictor, reconstruction and Soft Hydra
  datasets/         historical array loader and raw STEW reader
  pipeline/         preparation, study planning, training, evaluation and tables
  optimizers/       local StableAdamW
  compute.py        precision and deterministic-kernel policy
  normalization.py  shared FP32 RMSNorm
  metadata.py       fingerprints and atomic metadata
  rng.py            reproducible random states
```

`pretrain/pipeline/` constructs models through `pretrain/models/factory.py`.
Dataset reader code lives in `pretrain/datasets/`; input data paths remain
external configuration. Native adapters and audited release preparation stay
in `pretrain/pipeline/`.
The historical baseline models and training loops remain in Git history.
`Main.py paper ...` is a compatibility command for this same pipeline.

## Selected defaults

| Setting | Default |
|---|---|
| Primary encoder | 8 layers, width 64, 8 heads |
| Scaling | Depths 4/8; widths 64/128/512; primary model first |
| Tokenizer | Raw plus STFT, non-overlapping patches of 32 samples |
| Spatial encoding | Fixed normalized EEG-X2 XYZ sin/cos, base 5 |
| Reconstruction | Masked raw MSE plus soft DiCT; G=8, K=32, lengths 9/15, dilations 1/2/4/8/16 |
| Masks | Reproducible token masks, 75% hidden |
| Repeats | 5 pooled subject splits, 70/10/20 train/validation/test |

These are the user-selected defaults. RMSNorm, Q/K normalization, FlashAttention,
and the trainable Predictor are retained; teacher and EMA are absent. Non-rASR
processing applies to both input and target. The [task log](docs/task.md) records
the deliberate manuscript differences and training settings.

Omitting model size selects the primary encoder; omitting the objective selects
soft DiCT plus raw MSE. Explicit `objective: null` selects a raw-only ablation.
Train from scratch in a fresh study and carry the existing shared GPU budget
forward. Saved checkpoints support reproducible resume of those new runs.

## Reproducible paper rerun

The [rerun specification](docs/spec.md) records the selected configuration,
non-rASR data requirements, subject splits, validation selection and reporting.
The [task log](docs/task.md) distinguishes author requirements and selected local choices.
The [readiness review](docs/training-review.md) records measured readiness.

Import an audited processed release without applying preprocessing twice:

```bash
uv run --locked --no-active python -m pretrain.pipeline.processed \
  --contract /path/to/audited-release.json --output .runs/revised-data/task
```

The contract identifies the artifact and hash, explicit NCT or NTC layout,
physical channel order, signal/label/participant fields for each old partition,
verified label mapping, and attributable subject and preprocessing evidence.
See [the contract guide](docs/reproduction.md). Native adapters remain available
through `python -m pretrain.pipeline.prepare --help`.

Plan into a fresh study directory and carry the existing shared GPU ledger:

```bash
uv run --locked --no-active python -m pretrain plan \
  --datasets /path/to/verified-datasets.json --study .runs/revised-study \
  --budget .runs/paper-gpu1-v1 --gpu 1
uv run --locked --no-active python -m pretrain run --study .runs/revised-study
uv run --locked --no-active python -m pretrain report --study .runs/revised-study
```

Planning pools old subject partitions and freezes separate splits for the seed
bank. Unknown non-rASR history blocks a task. Other eligible tasks can proceed.
The controller uses GNU screen, waits for the approved GPU, leaves unrelated
work running, and charges all reserved time to the existing ledger.

Checkpoint selection and early stopping use validation only. Diagnostic test
curves use a frozen recipe and stay separate from final evaluation. Resume
checks source, environment, data, split, model, optimizer and compute contracts.
Actual attention dispatch is saved; CPU fallback does not prove FlashAttention
performance. Compilation is opt-in and remains part of run provenance.

Reporting rechecks saved selections, predictions, exact test membership and
metrics. `results.json` retains individual seeds, evidence hashes and blocked
cells. `paper_tables.csv` and `paper_tables.tex` contain only complete seed
groups. Use those measured tables to update the paper. Synthetic checks,
engineering smoke tests and historical scores never fill missing cells.
