# Revised EEG-X reproduction

Use `python Main.py --config configs/eegx.yaml` with the local uv environment.
The model reuses the EEG-X context encoder and active Predictor. Its hybrid
tokenizer, objective and decoder follow the accepted specification. These are
revised experiments; their outputs must be measured before paper tables change.

## Audited processed-release contract

`python -m pretrain.pipeline.processed --contract CONTRACT --output DIRECTORY`
accepts an audited JSON contract. Relative input paths resolve beside that file.
Signals are copied unchanged. The importer does not filter or normalize them.

```json
{
  "task": "dreamer_arousal",
  "artifact": "release.npy",
  "artifact_sha256": "THE_VERIFIED_ARTIFACT_DIGEST",
  "layout": "NCT",
  "channels": ["C1", "Pz"],
  "partitions": [
    {"signals": "X_train", "labels": "Y_train", "subjects": "id_train"},
    {"signals": "X_val", "labels": "Y_val", "subjects": "id_val"},
    {"signals": "X_test", "labels": "Y_test", "subjects": "id_test"}
  ],
  "label_map": {"0": 0, "1": 1},
  "label_evidence": {"status": "verified", "authority": "SOURCE_LABEL_EVIDENCE"},
  "subject_evidence": {"status": "verified", "authority": "SOURCE_PARTICIPANT_EVIDENCE"},
  "preprocessing_evidence": {"status": "verified", "authority": "SOURCE_PROCESSING_EVIDENCE"},
  "preprocessing": {},
  "source_evidence": "non-rasr-history.json"
}
```

This is a shape example, not evidence for a real dataset. Replace the channel
list and label map with the task's verified values. `preprocessing` must contain
the full effective profile returned by `pretrain.pipeline.preprocessing.profile()`.
Matching that dictionary requires source evidence; copying settings into an
unverified release does not establish its history. Each source row must already
be a complete task window. No row concatenation or synthetic padding is allowed.

The non-rASR history file identifies every source artifact:

```json
{
  "status": "verified",
  "authority": "ATTRIBUTABLE_SOURCE_OR_PROCESSING_RECORD",
  "upstream_rasr": false,
  "source_sha256": ["THE_VERIFIED_ARTIFACT_DIGEST"]
}
```

Keep the cited evidence with the dataset. A local `rasr=False` flag cannot
replace upstream history. Native preparation can accept the same evidence
through `python -m pretrain.pipeline.prepare --source-evidence HISTORY`.

## Run and resume

Edit the YAML manifest and output paths, then run
`python Main.py --config configs/eegx.yaml`. Paths resolve beside the YAML file.
Use a seeds list to run independent seeds sequentially on the local machine.
Each seed gets a fresh process, subject split, output directory and W&B run.
The `stage` selects pretraining, fine-tuning or frozen evaluation. Downstream
stages require a pretraining checkpoint. Missing evidence exits directly.

The output saves resolved YAML, split, training history, checkpoints and evaluation
predictions for each seed. To resume one seed, replace the list with a scalar seed,
set `resume` to its previous `training/last.pth` and choose a fresh output.
Batch checkpoint/resume paths use `{seed}` to identify the matching file.
Reuse the same data and training settings. CUDA runs
carry the existing shared GPU ledger; CPU checks use FP32.

## W&B results

Enable W&B in YAML to log model name, task name, seed, stage, recipe and metrics.
Online resume reuses the saved run ID. Local checkpoints remain authoritative.
Group by model and task in W&B, filtering to one stage and recipe. Include only
distinct completed seeds when computing paper averages. Final test metrics follow
validation-selected checkpoints. Diagnostic curves cannot guide selection.

Results remain per-seed records. W&B handles aggregation; the repository does
not build a study plan or generate seed-summary tables.
