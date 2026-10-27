# EEG-X reproduction context

The repository trains EEG-X from prepared HDF5 datasets. Main.py accepts one YAML configuration. Public stages are pretrain, linear_probe and finetune.

A saved subject-wise seed bank defines each train, validation and test partition. Reuse it across model comparisons. Duplicate-linked subjects remain in the same partition. Native preprocessing and sample-catalog rules are defined in docs/task.md and docs/dataset-preparation.md.

The primary model uses the hybrid raw/STFT tokenizer, spatial electrode identities, context encoder, active Predictor and raw-MSE plus full-dictionary Soft DiCT objective. Pretraining selects minimum validation raw MSE. Classification selects maximum validation balanced accuracy. Test scores never guide selection.

In-domain linear probes run inside pretraining. Their validation and diagnostic test trends use linear_probe/val and linear_probe/test. Cross-domain targets use independent linear probes and fine-tuning from one selected TUAB source checkpoint.

W&B logs one run per experiment and seed. Final classifier test scores have one history event and matching summary values, so automated workspaces discover them. Shared workspaces use native chart discovery; stage-specific saved views may use explicit panels. Group distinct completed seeds by domain, stage, task and model while keeping the same recipe.

GPU admission uses measured peak memory, process ownership and per-device headroom. Each generated task can use either reviewed GPU configuration. Persist the chosen placement before spawning and retain it on restart. Never stop unrelated workloads. Fresh study ledgers can have an explicit null time limit; historical ledgers remain immutable.

Source, configuration, software, data release, split, random state and selected-checkpoint provenance remain part of reproducible outputs. Local raw data, generated HDF5 files, checkpoints and private manuscripts are excluded from publication.
