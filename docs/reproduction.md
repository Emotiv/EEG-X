# Revised EEG-X reproduction

Use `python Main.py --config configs/eegx.yaml` with the local uv environment.
The model reuses the EEG-X context encoder and active Predictor. Its hybrid
tokenizer, objective and decoder follow the accepted specification. These are
revised experiments; their outputs must be measured before paper tables change.

## Processed-release contract

`python -m pretrain.pipeline.processed --contract CONTRACT --output DATASET.h5`
accepts a JSON import contract. Relative input paths resolve beside that file.
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
  "preprocessing": {
    "name": "processed_release_as_is_v1",
    "sample_rate": 128,
    "output_dtype": "float32",
    "local_operations": ["cast_float32"],
    "upstream_history": "not_verified"
  }
}
```

This is a shape example. Use the artifact's stored labels and participant IDs,
and specify its physical channel order explicitly. Each source row must be a
complete task window. The as-provided profile preserves signal values and casts
to float32. It does not claim that filtering, normalization or artifact removal
were re-run or verified. The user waived proof of upstream non-rASR history.
Optional source-history notes can be retained without becoming a training gate.

## Run and resume

Edit the YAML dataset and output paths, then run
`python Main.py --config configs/eegx.yaml`. Paths resolve beside the YAML file.
Use a seeds list to run independent seeds sequentially on the local machine.
Each seed gets a fresh process, reads its saved H5 split, and writes its own
output directory and W&B run.
The `stage` selects `pretrain`, `finetune` or `linear_probe`. Downstream
stages require a matching pretraining checkpoint. Invalid shapes, labels or participant IDs exit directly.

The agreed in-domain workflow is pretraining with embedded linear-probe validation
and diagnostic test curves, followed by fine-tuning. Fine-tuning logs validation
each epoch, selects the best validation checkpoint, then computes test scores once.
Do not add standalone in-domain probe runs. Separate `linear_probe` runs evaluate
cross-domain transfer from the TUAB encoder.

Pretraining and downstream stages reuse the seed bank. Each seed's pretraining
and in-domain fine-tuning use the same saved subject split. Subject separation,
rather than the number of seeds, prevents train/test overlap within a run.

The output saves resolved YAML, split, training history, checkpoints and evaluation
predictions for each seed. To resume one seed, replace the list with a scalar seed,
set `resume` to its previous `training/last.pth` and choose a fresh output.
Batch checkpoint/resume paths use `{seed}` to identify the matching file.
Reuse the same data and training settings. CUDA runs
carry the existing shared GPU ledger; CPU checks use FP32.

## W&B results

Enable W&B in YAML to log model name, task name, seed, stage, evaluation mode, recipe and metrics.
Online resume reuses the saved run ID. Local checkpoints remain authoritative.
Group by model, task and stage in W&B, filtering to one recipe. Include only
distinct completed seeds when computing paper averages. Final test metrics follow
validation-selected checkpoints. Diagnostic curves cannot guide selection.

Results remain per-seed records. W&B handles aggregation; the repository does
not build a study plan or generate seed-summary tables.

Prepared output is one HDF5 file. It contains all signal windows, metadata and saved seed indices; the import preserves the declared release values. Training uses the YAML `dataset` path and reads a saved split bank.
