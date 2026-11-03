# Reproduce the fixed-split EEG-X study

This release uses the original hybrid raw/STFT EEG-X model. It does not use the EEG-X2 research models. The protocol, split-selection evidence and release manifest are versioned beside this document.

| Item | Value |
|---|---|
| Encoder | Depth 8, width 64, heads 8 |
| Training seeds | 241, 242, 243 |
| Pretraining | Maximum 100 epochs; microbatch and global batch 256 |
| Probe and fine-tuning | Maximum 20 epochs; batch 64; patience 5 |
| Tracked experiments | 73 |

All cross-domain targets also have in-domain runs. Both domains use exactly the same HDF5 file and train, validation and test indices for a target. In-domain pretraining uses that target training set; cross-domain pretraining uses TUAB training subjects. All training seeds use the same train, validation and test memberships. They change initialization, shuffling, dropout and masks. Checkpoints use validation metrics only.

| Dataset | Split |
|---|---|
| TUAB, TUEV binary | Official eval is test; official training subjects are split 80/20 into train/validation |
| STEW | Saved split 241, selected from historical test balanced accuracy |
| DREAMER arousal | Saved split 242, selected from historical test balanced accuracy |
| Crowdsourced, BCICIV-2a, Kalunga, PhysioNetMI | Saved split 241; no reference target scores were available |

STEW and DREAMER use test-informed split selection. Interpret those results as exploratory. The other secondary splits were fixed without a score search.

The upload bundle is `.runs/data/eegx-fixed-v1/`. It contains one self-contained HDF5 per dataset, `protocol.json` and `manifest.json`. Each HDF5 contains EEG, labels, subject and recording provenance, channels, preprocessing, stable window identities and split index arrays. No old data directory or raw recording is needed to train from the bundle. Verify each downloaded file against its manifest SHA-256.

To rebuild the fixed-split bundle from the earlier prepared release:

```sh
python -m pretrain.pipeline.release \
  --source /path/to/original-prepared-data \
  --output /path/to/eegx-fixed-v1 \
  --protocol reproduction/eegx-fixed-v1/protocol.json
```

Signal values, class mappings, window counts, overlap and preprocessing remain unchanged. The builder audits every window against its saved signal hash before publishing the new directory.

Install the pinned environment with `uv sync --locked`. Dataset YAMLs live in `configs/`; update only local dataset, output, device, budget and W&B account paths. `seed` controls training randomness; `split_seed` chooses the saved bank. Run `python Main.py --config <yaml>`. Each in-domain fine-tune must use the matching training seed's selected encoder. Cross-domain probes and fine-tuning reuse the one selected TUAB source checkpoint.

The queue freezes committed source, generated YAMLs and their hashes. Its output folder contains the source revision, software contract, task dependencies and placements, local checkpoints, predictions, metrics and W&B run IDs. Copy that folder with the data bundle to reproduce or inspect an experiment. The `source/` snapshot runs independently of later working-tree edits.

W&B project `quoc-emotiv/eegx` keeps one run per tracked experiment. Group completed runs by domain, stage, task, model size and recipe to view seed means. Diagnostic probe curves stay in their pretraining runs.

The paired study adds missing tasks without changing the original queue. Use `scripts/rerun/prepare.py --previous <original-manifest>` to prepare only the missing tasks. The study matrix is `study.json`; the original data protocol and release remain unchanged. Each queue retains its frozen source and selected checkpoints.

The supplement waits for successful completion of the original queue before GPU admission. This prevents independent launchers from reserving the same free memory. It creates no extra W&B runs while waiting.
