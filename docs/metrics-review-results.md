# Complete metric review test

Fresh engineering runs verify metric delivery using the existing data and saved subject splits.

| Run | Epochs | Validation SoftDiCT | Test balanced accuracy |
|---|---:|---:|---:|
| [STEW in-domain pretraining](https://forge.coreweave.com/wandb/quoc-emotiv/eegx/runs/329unh5m) | 2 | 1.496958 | - |
| [STEW in-domain fine-tuning](https://forge.coreweave.com/wandb/quoc-emotiv/eegx/runs/2akph62c) | 2 | - | 0.677572 |
| [TUAB cross-domain source](https://forge.coreweave.com/wandb/quoc-emotiv/eegx/runs/e64fgsyu) | 1 | 16.324858 | - |
| [Crowdsourced linear probe](https://forge.coreweave.com/wandb/quoc-emotiv/eegx/runs/4vrjqfib) | 2 | - | 0.590141 |
| [Crowdsourced fine-tuning](https://forge.coreweave.com/wandb/quoc-emotiv/eegx/runs/6lc0vks4) | 2 | - | 0.547183 |

These are short dashboard-review runs. Pretraining probe scores are diagnostic and do not select the encoder checkpoint.

| Delivery check | Verified result |
|---|---|
| Current and reference projects | Only 5 new finished test runs in eegx; 23 reference runs retained |
| Seed and pretraining batch | Seed 241; microbatch 512, global batch 4096 |
| Validation probe trends | All 11 metrics, each with 2 epoch points, plus a confusion matrix |
| Diagnostic test probe trends | All 11 metrics, each with 2 epoch points, plus a confusion matrix |
| Probe cadence | Test every 1 epoch; production every 10 epochs |
| Final classifier reporting | All 11 scalar scores logged in 1 final history event and summary, plus a confusion matrix |
| Selected checkpoints | Model artifacts uploaded for all 5 runs |
| System and Logs | 49 native system keys and nonempty console output for every run |
| Individual run browser checks | All 5 pass: pretraining has no generic final-test block; classifier pages retain final scores |
| Tests and review | 26 focused checks passed; Standards 0 open findings; Spec 0 open findings |

Validation raw MSE selects pretraining checkpoints. Validation balanced accuracy selects classifier checkpoints. The full paper queue remains stopped for review.
