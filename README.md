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
  models/           EEG-X encoder, Predictor, reconstruction and Soft DiCT
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
and the trainable Predictor are retained; teacher and EMA are absent. The same processed X arrays supply input and target; no local rASR is added. The [task log](docs/task.md) records
the deliberate manuscript differences and training settings.

Omitting encoder depth and width selects the primary encoder; omitting the objective selects
soft DiCT plus raw MSE. Explicit `objective: null` selects a raw-only ablation.
Train new models from scratch and carry the existing shared GPU budget
forward. Saved checkpoints support reproducible resume of those new runs.

## Train EEG-X from YAML

Edit `configs/eegx.yaml` to set the prepared HDF5 dataset and output directory,
then run one model directly:

```bash
uv run --locked --no-active python Main.py --config configs/eegx.yaml
```

The YAML holds the stage, encoder depth and width, seeds, batch sizes, epochs,
optimizer, learning-rate schedule, objective, device and W&B settings. Paths resolve beside the YAML file. The script
selects the saved subject split for each seed and runs the seeds sequentially
on the local machine. Each seed uses a fresh process on the selected GPU.
Input errors stop the command directly. Only the configured model runs.

```yaml
seeds: [241, 242, 243, 244, 245]
output: ../.runs/eegx-d8w64
```

The output contains `seed-241/`, `seed-242/` and the remaining seed directories.
Each contains `config.yaml`, `split.json`, W&B files and `training/`
logs/checkpoints. Validation chooses the best checkpoint independently per seed.
The loop stops on failure and leaves completed outputs intact. It uses no SLURM.

To run the complete paper configuration, use `bash run_paper.sh`. The current
paper expansion uses two fixed queues, with four dataset pipelines per GPU.
GPU one owns seeds 241 through 243; GPU zero owns seeds 244 and 245 and the one
shared TUAB source. Each queue finishes source training before its dependent
stages. Interrupted tasks resume from verified checkpoints into fresh attempt directories.
Completed tasks are checked and skipped, and an output lock prevents duplicate
dispatch. Live process reparenting is unsupported. The active queue pointer
selects these fixed assignments; there is no model sweep or seed-mean controller.

Both devices spend the original shared GPU-hour cap. The auxiliary GPU-zero
ledger is a usage meter, not another independent training allowance. Run its
configurations through the shared launcher, which transfers incremental usage to
the original ledger and requests safe stops at the aggregate cap. Four concurrent
runs on one device charge its occupied time once; two occupied devices charge
twice. The frozen scientific training source and its checkpoint contracts stay
unchanged across checkpoint resume. Queue processes have separate sessions so a
terminal closing cannot interrupt training before checkpoint cleanup.

For one run, replace `seeds` with `seed: 241`; its files go directly into `output`.
To resume that seed, set `resume` to its previous `training/last.pth` and choose a
fresh output. Keep the other settings and input data unchanged. Resume paths for multiple seeds use `{seed}`. A shared TUAB source checkpoint
uses the same path for every target seed.

## Bounded performance checks

The separate configurations in `configs/benchmarks/` prepare a pretraining
microbatch of 512 and a global batch of 4096. These are engineering probes;
the active paper configurations remain separate. Each optimizer update uses
eight microbatches. Changing the global batch creates a different training
recipe and is not an exact resume of the current paper runs.

```bash
python scripts/profile_training.py --config configs/benchmarks/tuab.yaml \
  --output .runs/benchmark-tuab --updates 2 --warmup 1 --profile
```

The driver uses the real training loss, stops after bounded updates, records
memory and timings, and creates no W&B run. It requires the original shared
budget. By default, it refuses an active paper queue or device meter. The explicit
`--reserve-live-slot` option reserves a bounded measurement window through the
same accounting lock. Running paper tasks keep computing, but epoch accounting
and new admissions can wait until the reservation ends. The driver releases its
lease before unlocking and exports traces afterward. It never resets the budget.
Run comparisons without profiling on an otherwise idle GPU before claiming an
isolated speedup.

The bounded dictionary implementation preserves FP32 features, fixed filters
and temporal counts. Forward activations are pooled and recomputed for backward
instead of retaining every full temporal probability tensor. The default
`dictionary_backend: torch` uses deterministic matrix operations and a native
gradient fold. The optional `dictionary_backend: triton` fuses dictionary work
on CUDA and supports first-order gradients only. Its group and kernel counts
must be powers of two, with at most 256 filters in total. Use
`configs/benchmarks/tuab-triton.yaml` to benchmark it. Runtime contracts record
the installed Triton version. Full-model `torch.compile` performance remains
unvalidated. Unrelated threads must not change process-wide CUDA precision while
the model is running. See the [measured audit](docs/performance-audit.md).

An alternative engineering configuration, `configs/benchmarks/tuab-hydra-target.yaml`,
uses `objective.name: raw_hydra_patch_target_v1`. It keeps waveform reconstruction
and adds a decoder head that predicts detached HYDRA features for each channel
and non-overlapping target region. No gradient passes through HYDRA. Its explicit
`patch_size`, `kernel_size`, `groups`, `k`, `weight` and `seed` define a different
objective from SoftDiCT. Target regions must contain complete model patches and
divide the input duration. The variant uses the Torch backend and keeps fixed
filters in its checkpoint. A target region contributes when any constituent
model patch is masked. The head pools all decoder tokens in that region, so its
auxiliary gradient can also reach visible tokens within a contributing region.
Fully visible regions are excluded. Validation still selects by masked waveform MSE.
This is an optional experiment, not a replacement for active paper settings.

`device: cuda:1` selects physical GPU 1 and preserves the existing shared budget.
For a CPU check, use `device: cpu` and `precision: fp32`. Inputs use the prepared
HDF5 dataset described below, with physical channel IDs, stored labels and subject IDs. Upstream non-rASR proof is not required.

## Reproducible paper rerun

The [rerun specification](docs/spec.md) records the selected configuration,
processed input contracts, subject splits, validation selection and reporting.
The [task log](docs/task.md) distinguishes author requirements and selected local choices.
The [readiness review](docs/training-review.md) records measured readiness.

Import an audited processed release without applying preprocessing twice:

```bash
uv run --locked --no-active python -m pretrain.pipeline.processed \
  --contract /path/to/audited-release.json --output .runs/revised-data/task.h5
```

The contract identifies the artifact and hash, explicit NCT or NTC layout,
physical channel order, signal/label/participant fields for each old partition,
stored label mapping, participant IDs and declared processing history.
See [the contract guide](docs/reproduction.md). Native adapters remain available
through `python -m pretrain.pipeline.prepare --help`.

```bash
python -m pretrain.pipeline.prepare --task stew \
  --raw /data/data_downstream_task/STEW/Raw --output .runs/data/stew.h5
```

Unverified original crop mappings are reported as unresolved before writing a
new dataset. See [the current reference status](docs/task.md).

BCICIV-2a, Kalunga and PhysioNet preparation also require `--reference-proof`
with their verified original-array lineage JSON. The output embeds that proof;
training reads the H5 without a proof sidecar. Existing H5 metadata exposes it as
`reference_provenance` for reproducible regeneration.

Native DREAMER labels use the selected arousal rating threshold; TUEV uses its
explicit native REC binary map. Archived label artifacts are optional comparison
audits, not prerequisites for raw preparation.

Each dataset is one HDF5 file with `X[N,C,T]`, `y`, records, subjects,
physical channel names/IDs/XYZ, window IDs, source hashes and all five saved
subject splits. TUAB also contains its shared source train/validation split.
The writer preserves original overlap and sample multiplicity, groups duplicate-linked
subjects, and publishes atomically after reference-count and class-coverage checks. No prepared NPY recordings or dataset manifest sidecars are produced.

CUDA runs require an idle device by default. Set `shared_gpu: true` explicitly
when sharing a device with existing workloads. This keeps those workloads running
and records per-process peak PyTorch allocation and wall time in `gpu.json` and
W&B summary. Native W&B System metrics also include device-wide resource use.

YAML defaults to `purpose: paper`, which requires a complete verified sample
catalog. Use `purpose: engineering` explicitly for smoke checks on pilot datasets;
these runs remain labelled as engineering checks.

Training selects the saved H5 split. Its Dataset reads windows lazily; each
DataLoader worker opens its own H5 handle. `split.json` in a run directory is
provenance only, not another dataset. Window IDs and row ordering are stable
when source files are enumerated differently or the H5 is moved to another host.

STEW and PhysioNet use complete native subject scopes by default. An explicit
`--subjects` list prepares a clearly marked pilot cohort. PhysioNet keeps the
six imagery runs and their task-rest annotations; it reads each EDF's native
sampling rate before preprocessing. The existing raw preparation
command now uses EEG-X2 QuantPlus preprocessing: native reference, no extra notch,
high-pass 0.5 Hz, low-pass 45 Hz, resample 128 Hz, recording-level StandardScaler,
clipping to ±15 and float32 output. Prepare fresh outputs with this recipe;
existing prepared values retain their original preprocessing metadata.

Each seed runs the stage selected in YAML:

| Stage | Configuration | Result |
|---|---|---|
| `pretrain` | `checkpoint: null` | Train from scratch; log linear-probe validation and diagnostic test curves in this run |
| `finetune` | `checkpoint` points to pretraining `best.pth` | Tune encoder and classifier; evaluate the validation-selected model |
| `linear_probe` | `checkpoint` points to the shared TUAB `best.pth` | Cross-domain linear probe: freeze encoder and fit a classifier on target training data |

The agreed protocol has no standalone in-domain linear-probe runs. Pretraining
probes fit only on training subjects. Fine-tuning logs validation curves each epoch,
selects the best validation balanced accuracy, then evaluates test once. Linear probes
also select their head and apply early stopping using validation balanced accuracy.

| Ready configuration | Purpose | Seeds |
|---|---|---|
| `configs/stew.yaml` | STEW pretraining with probe curves | 241, 242, 243, 244, 245 |
| `configs/stew-finetune.yaml` | Fine-tune the matching STEW checkpoints | 241, 242, 243, 244, 245 |
| `configs/tuab.yaml` | TUAB pretraining with probe curves | 241, 242, 243, 244, 245 |
| `configs/tuab-source.yaml` | One shared TUAB source with validation selection | 241 |
| `configs/crowdsourced-linear-probe.yaml` | TUAB-to-Crowdsourced linear probes | 241, 242, 243, 244, 245 |
| `configs/crowdsourced-finetune.yaml` | TUAB-to-Crowdsourced fine-tuning | 241, 242, 243, 244, 245 |

Run each with `python Main.py --config CONFIG`. Cross-domain evaluation requires
the shared TUAB checkpoint first. Change the target dataset and output for
BCICIV-2a or Kalunga. Saved subject splits prevent overlap within each seed;
repeating seeds alone does not prevent leakage.

Use a fresh output for each invocation. Preserve the existing budget ledger.
Checkpoint selection and early stopping use validation. Test probes are diagnostic.
Source, environment, data, split and model contracts protect reproducible resume.

New checkpoints save the pretraining task, source release and subject split.
In-domain fine-tuning requires the same dataset and split. Cross-domain linear
probing and fine-tuning require the shared TUAB source with the matching model.
The source uses separate subject-wise training and validation sets, selects best
validation reconstruction loss, and is independent of target seeds.
Evaluation rejects mismatched sources before test access or W&B result logging.

## Weights & Biases

The example YAML enables W&B. Authenticate once with `wandb login` or your
existing W&B environment, then run the same command. Set `wandb.enabled: false`
to keep only local logs, or `wandb.mode: offline` to write W&B files locally.

```yaml
wandb:
  enabled: true
  project: eegx
  domain: in_domain
  entity: null
  group: null
  mode: online
```

Each seed logs one W&B run. `domain`, `model_name`, `task_name`, `seed`, `stage`, `evaluation_mode` and
`recipe_id` are saved automatically. A null group leaves runs ungrouped.
Choose `task_name` in W&B's Group by menu when needed. Filter by stage and keep
the same model and recipe to compare an experiment. An explicit group label remains optional.
Set `wandb.domain: cross_domain` for transfer experiments, including the shared
TUAB source run. In a shared workspace, choose `domain`, then `stage`, then
`task_name` to keep seed means separate across domains and methods.
Set `wandb.name: test_stew_seed_{seed}` for test runs; `{seed}` expands for each seed.

In W&B, group by `model_name`, `task_name` and `stage`. Filter to the same
`recipe_id`, with `run_status: complete`. Compare the individual seeds or display
mean and standard deviation. For paper results, include all distinct completed
seeds and use final `test/*` values from validation-selected checkpoints.
Final test scalars are logged once to history and summary after checkpoint selection.
Native automated workspaces discover these metrics without manual panels. Training and validation keep epoch curves.
`linear_probe/test/*` and `selected/linear_probe/test/*` remain diagnostic.

Each W&B run enables native system monitoring and stdout/stderr capture. Open
the run's System tab for resource metrics and Logs for console output. GPU metrics
depend on the available NVIDIA devices. The SDK samples system metrics independently
of training epochs, so very short runs may not contain a full sampling interval.

| Run | `stage` / Job Type | `evaluation_mode` | Metrics |
|---|---|---|---|
| Pretraining with embedded probes | `pretrain` | `pretrain` | `linear_probe/val/*`, `linear_probe/test/*` |
| In-domain fine-tuning | `finetune` | `finetune` | `val/*` curves; final `test/*` |
| Cross-domain linear probe | `linear_probe` | `linear_probe` | `val/*` curves; final `test/*` |

Group seed averages by model, task and stage, keeping one recipe. Embedded probes
remain part of their pretraining run; they do not create additional W&B runs.

Pretraining logs loss, raw MSE, Soft DiCT loss, learning rate, gradient norm and
validation MSE, plus the selected epoch and validation loss. W&B config and summary
record total and trainable model parameter counts. Fine-tuning logs its learning
rate range across optimizer groups and validation classification loss.
Probes and evaluation log classification loss, accuracy, balanced accuracy, AUROC,
AUPRC, micro/macro/weighted F1, and macro precision/recall. Binary tasks also
report class-1 F1. W&B names balanced accuracy `bal_acc`. Final
evaluation logs a confusion-matrix image and preserves predictions locally. Validation
confusion matrices are logged during fine-tuning and linear probing. Embedded
pretraining probes retain separate validation and diagnostic test matrix keys.

Each completed pretraining, linear-probe and fine-tuning run uploads its selected checkpoint as
a W&B model artifact with the `best` alias and validation selection metadata.
The artifact reference is saved in `selected/model_artifact`. The shared TUAB source also selects its checkpoint on validation loss.

Online resume reuses the saved W&B run ID in `wandb.json`. Keep project and group
unchanged. W&B performs grouping and aggregation; this repository has no study
plan, controller, task registry or seed-summary job.

## Selected STEW release

`configs/stew.yaml` reads `.runs/data/stew.h5`. Prepare it from native STEW recordings:

```bash
python -m pretrain.pipeline.prepare --task stew --raw /data/EEG-X/RAW/STEW/Raw --output .runs/data/stew.h5
uv run --locked --no-active python Main.py --config configs/stew.yaml
```

The native reader pairs low/high recordings by participant and applies the
shared signal recipe before writing the H5. Historical processed STEW imports
remain explicit releases with their own recorded preprocessing history.


W&B classifier panels follow the Neuralbench DREAMER valence layout in both
evaluation domains. Learning-rate names match the actual optimizer:
`lr-StableAdamW` for pretraining and `lr-AdamW` for current fine-tuning/probing.
The native workspace utility (`python -m pretrain.workspace --help`) applies a
shared layout to explicitly selected views. Its bar panels use bare summary keys;
final test metrics have one history event and matching summary values. See [the panel audit](docs/wandb-panel-audit.md).
