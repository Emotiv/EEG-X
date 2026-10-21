# Revised EEG-X reproduction

Use `python -m pretrain` with the local uv environment. The compatibility
command `Main.py paper` enters the same pipeline.
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

## Plan, run and resume

The datasets JSON maps each task to `{"manifest": "/absolute/manifest.json"}`
or `{"blocked": "the missing evidence"}`. Plan into a fresh study directory.
Pass `--budget` pointing to the existing shared ledger. Its usage is preserved.
The production device and scientific matrix are fixed by the specification.

The controller checks software and dataset fingerprints before execution.
Each experiment uses a separate screen session and fresh output directory.
Rerun the controller after a safe budget stop to resume from an optimizer
boundary. Code, environment, data or contract changes require a distinct study;
carry the same budget ledger forward. Failed jobs stay visible for investigation.

## Paper values

Run `python -m pretrain report --study DIRECTORY`. The report recomputes metrics from
saved predictions and checks test membership and selected artifacts. Individual
seeds remain in `results.json`. Only complete seed groups enter
`paper_tables.csv` and `paper_tables.tex`. Diagnostic curves are separate and
cannot select checkpoints or configurations. Preserve revised and historical
table labels until all required comparisons have been rerun.
