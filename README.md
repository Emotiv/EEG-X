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
  pipeline/         preparation, training and evaluation
  optimizers/       local StableAdamW
  compute.py        precision and deterministic-kernel policy
  normalization.py  shared FP32 RMSNorm
  metadata.py       fingerprints and atomic metadata
  rng.py            reproducible random states
  tracking.py       W&B logging for one seed
  gpu.py            GPU availability checks
```

`pretrain/pipeline/` constructs models through `pretrain/models/factory.py`.
Dataset reader code lives in `pretrain/datasets/`; input data paths remain
external configuration. Native adapters and audited release preparation stay
in `pretrain/pipeline/`.
The historical baseline models and training loops remain in Git history.
`Main.py --config ...` is the main entrypoint.

## Selected defaults

| Setting | Default |
|---|---|
| Primary encoder | 8 layers, width 64, 8 heads |
| Tokenizer | Raw plus STFT, non-overlapping patches of 32 samples |
| Spatial encoding | Fixed normalized EEG-X2 XYZ sin/cos, base 5 |
| Reconstruction | Masked raw MSE plus soft DiCT; G=8, K=32, lengths 9/15, dilations 1/2/4/8/16 |
| Masks | Reproducible token masks, 75% hidden |
| Repeats | 5 pooled subject splits, 70/10/20 train/validation/test |

These are the user-selected defaults. RMSNorm, Q/K normalization, FlashAttention,
and the trainable Predictor are retained; teacher and EMA are absent. Non-rASR
processing applies to both input and target. The [task log](docs/task.md) records
the deliberate manuscript differences and training settings.

Omitting encoder depth and width selects the primary encoder; omitting the objective selects
soft DiCT plus raw MSE. Explicit `objective: null` selects a raw-only ablation.
Train new models from scratch and carry the existing shared GPU budget
forward. Saved checkpoints support reproducible resume of those new runs.

## Train EEG-X from YAML

Edit `configs/eegx.yaml` to set the prepared dataset manifest and output directory,
then run one model directly:

```bash
uv run --locked --no-active python Main.py --config configs/eegx.yaml
```

The YAML holds the stage, encoder depth and width, seeds, batch sizes, epochs,
optimizer, learning-rate schedule, objective, device and W&B settings. Paths resolve beside the YAML file. The script
creates a subject split for each selected seed and runs the seeds sequentially
on the local machine. Each seed uses a fresh process on the selected GPU.
Input errors stop the command directly. Only the configured model runs.

```yaml
seeds: [241, 242, 243, 244, 245]
output: ../.runs/eegx-d8w64
```

The output contains `seed-241/`, `seed-242/` and the remaining seed directories.
Each contains `config.yaml`, `split/manifest.json`, W&B files and `training/`
logs/checkpoints. Validation chooses the best checkpoint independently per seed.
The loop stops on failure and leaves completed outputs intact. It uses no SLURM.

For one run, replace `seeds` with `seed: 241`; its files go directly into `output`.
To resume that seed, set `resume` to its previous `training/last.pth` and choose a
fresh output. Keep the other settings and input data unchanged. Batch checkpoint
or resume paths must contain `{seed}` so each run loads its matching checkpoint.

`device: cuda:1` selects physical GPU 1 and preserves the existing shared budget.
For a CPU check, use `device: cpu` and `precision: fp32`. Inputs use the prepared
manifest described below, with physical channel IDs and verified source history.

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

Each seed runs the stage selected in YAML:

| Stage | Configuration | Result |
|---|---|---|
| `pretrain` | `checkpoint: null` | Train an EEG-X encoder from scratch |
| `finetune` | `checkpoint` points to pretraining `best.pth` | Tune encoder and classifier; evaluate the validation-selected model |
| `frozen` | `checkpoint` points to pretraining `best.pth` | Freeze encoder and fit a classifier on training data |

Use a fresh output for each invocation. Preserve the existing budget ledger.
Checkpoint selection and early stopping use validation. Test probes are diagnostic.
Source, environment, data, split and model contracts protect reproducible resume.

## Weights & Biases

The example YAML enables W&B. Authenticate once with `wandb login` or your
existing W&B environment, then run the same command. Set `wandb.enabled: false`
to keep only local logs, or `wandb.mode: offline` to write W&B files locally.

```yaml
wandb:
  enabled: true
  project: eeg-x
  entity: null
  group: null
  mode: online
```

Each seed logs one W&B run. `model_name`, `task_name`, `seed`, `stage` and
`recipe_id` are saved automatically. A null group derives a group shared by
seeds using the same model recipe, dataset and stage. Set a group label if needed.

In W&B, group by `model_name` and `task_name`. Filter to the same `stage` and
`recipe_id`, with `run_status: complete`. Compare the individual seeds or display
mean and standard deviation. For paper results, include all distinct completed
seeds and use final `test/*` values from validation-selected checkpoints.
`diagnostic/test/*` and `selected/diagnostic/test/*` remain diagnostic.

Pretraining logs loss, raw MSE, Soft DiCT loss, learning rate, gradient norm and
validation MSE. Probes and evaluation log accuracy, balanced accuracy, AUROC,
AUPRC and micro/macro F1. Multiclass evaluation also keeps weighted F1. Final
evaluation logs a confusion matrix and preserves predictions locally.

Online resume reuses the saved W&B run ID in `wandb.json`. Keep project and group
unchanged. W&B performs grouping and aggregation; this repository has no study
plan, controller, task registry or seed-summary job.
