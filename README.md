# EEG-X

PyTorch implementation of EEG-X: Device-Agnostic and Noise-Robust Foundation Model for EEG.
See [the model](docs/model.md) for architecture, loss and training settings.

## Setup

Use Python 3.12 or 3.13. Install the pinned environment:

```sh
uv sync --locked --dev
uv run --locked --no-active python Main.py --help
```

## Data

Place the latest prepared HDF5 bundle in `data/`, beside its tracked
`manifest.json`, `protocol.json`, `study.json` and `split-selection-evidence.json`.
The HDF5 files are distributed separately and excluded from Git.
Each file contains signals, labels, channel identities, preprocessing,
subject provenance and the saved train/validation/test indices.

| Dataset files |
|---|
| `dreamer-arousal.h5`, `stew.h5`, `tuab.h5`, `tuev-binary.h5` |
| `crowdsourced.h5`, `bciciv-2a.h5`, `kalunga.h5`, `physionet-mi.h5` |

Both evaluation domains use the same target file and saved subject split.
Verify the bundle against the tracked manifest before training:

```sh
uv run --locked --no-active python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

root = Path("data")
for entry in json.loads((root / "manifest.json").read_text())["datasets"].values():
    path = root / entry["file"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"Checksum mismatch: {path}")
    print(f"Verified: {path.name}")
PYTHON
```

To rebuild from the earlier prepared corpus, use the saved protocol:

```sh
uv run --locked --no-active python -m pretrain.pipeline.release \
  --source /path/to/original-prepared-data \
  --output .runs/rebuilt-data --protocol data/protocol.json
```

The output directory must be new. Copy its HDF5 files into `data/` to use the
configs below. The builder preserves signal values and audits all window hashes.
For native recording preparation, see `python -m pretrain.pipeline.prepare --help`.

## Reproduce training

In `configs/`, set the local device, output, budget and W&B account.
Paths resolve beside each YAML. Keep dataset identities, saved splits and training
settings unchanged. Use the existing budget ledger for an ongoing study.
Authenticate with `wandb login`, or set `wandb.enabled: false` for local logs.

For a new study, initialize the budget at the path configured in YAML:

```sh
uv run --locked --no-active python - <<'PYTHON'
from pretrain.pipeline.budget import Budget
Budget(".runs/eegx-fixed-v1/usage/gpu1").initialize()
PYTHON
```

For an existing study, reuse its saved ledger. Run the complete study sequentially
in dependency order:

```sh
bash run_paper.sh
```

The launcher reads YAMLs directly and stops on the first failed stage.
It trains each target before its in-domain fine-tuning, then trains the shared
TUAB source before cross-domain evaluation. Logs remain in each run output.

For a single stage:

```sh
uv run --locked --no-active python Main.py --config configs/stew.yaml
```
In-domain fine-tuning uses each seed's selected pretraining checkpoint.
Cross-domain stages share the selected TUAB source checkpoint.

| Study rule | Value |
|---|---|
| Training seeds | 241, 242, 243; same subject split for every seed |
| Paired study | 73 runs, as defined in `data/study.json` |
| TUAB and TUEV | Official test; official training subjects split 80/20 for train/validation |
| STEW / DREAMER arousal | Saved split 241 / 242; historical test-informed selection, exploratory results |
| Other targets | Saved split 241, fixed without a target score search |

Pretraining selects minimum validation raw MSE. Downstream stages select maximum
validation balanced accuracy, then evaluate test once. Embedded test probes are diagnostic.

Each output saves resolved config, split identity, checkpoints, history and predictions.
To resume, set a scalar `seed`, set `resume` to its `training/last.pth`, and use a fresh
output with the same source, environment, data and training settings.

## Checks

```sh
uv run --locked --no-active python -m pytest -q
```
