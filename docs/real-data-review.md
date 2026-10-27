# Real-data tracking review

Review runs test the recorded workflow and W&B display using real EEG subsets.
They do not supply paper scores. No synthetic values fill missing metrics. Completed W&B summaries, matrices and
selected model artifacts agree with saved local evidence.

| Scope | Review setting |
|---|---:|
| Prepared datasets available | 7 |
| In-domain datasets | 4 |
| Cross-domain target datasets | 3 |
| Seeds per downstream experiment | 5 |
| Shared TUAB source runs | 1 |
| Completed, verified review runs | 71 |
| Measured metric values compared with W&B | 970 |
| Native confusion-matrix images | 180 |
| Image counts checked against saved predictions | 130 |
| Review pretraining epoch limit | 20 |
| Paper pretraining epoch limit | 100 |
| Probe and fine-tuning epoch limit | 20 |
| Downstream early-stopping patience | 5 |
| CPU tests passed | 147 |
| CUDA tests skipped | 3 |

The full paper scope also includes PhysioNetMI. Its canonical electrode coordinates
are available, but its processed channel order and participant mapping remain unresolved.

| Evidence | Required review result |
|---|---|
| Subject splits | Saved subject-separated training, validation and test assignments; source TUAB has training and validation only |
| In-domain probes | Validation and diagnostic test curves inside pretraining; repeat after loading the selected checkpoint |
| Cross-domain evaluation | Both methods start independently from the same TUAB encoder and reuse each target seed's split |
| Selection | Validation selects each checkpoint; test scores do not guide selection |
| Classification metrics | Recompute from saved predictions and compare with W&B summaries |
| Model artifacts | Compare W&B artifact digests with the selected local checkpoints |
| Confusion matrices | Display images inside validation and test sections; preserve count evidence and remove raw table panels |
| Tracking | Test names, domain, stage and seed; native system monitoring and console capture |
| Chart layout | Neuralbench-style train, val, test and model sections; confusion matrices stay inside val/test; embedded probe validation and diagnostic test curves remain separate |
| System and logs | Native per-run System and Logs tabs |
| Grouping | No preset groups; choose task in the relevant domain and stage view |

Subset scores measure this review corpus only. Completed full-data experiments
are still required before replacing paper results.

The latest user update selects linear-probe heads and fine-tuned models by
validation balanced accuracy. Completed review runs retain their original
checkpoint decisions; additional test scores use their saved predictions.
