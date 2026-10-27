# Final-test panel rendering fix

Automatic workspace generation displayed the confusion image while omitting the scalar scores stored in run summaries. Explicit summary bars now expose those selected-checkpoint scores. W&B distinguishes [logged history and summary values](https://docs.wandb.ai/models/support/difference_log_summary).

| Check | Before | Verified after |
|---|---:|---:|
| Visible test panels in project overview | 1 | 12 |
| Test scalar bars | 0 | 11 |
| Confusion-matrix panels | 1 | 1 |
| Browser-verified overview and classifier views | - | 4 |
| Pagination pages checked per view | - | 2 |
| Focused regression tests | - | 10 passed |

Authenticated browser checks cover the project overview, in-domain fine-tuning, cross-domain fine-tuning and cross-domain linear probing. Each displays populated score bars and its confusion matrix.

The binding audit now detects missing final-score charts even when the existing image binding is valid. The repair preserves table grouping, filters, selections and workspace modes. Existing trained results and checkpoints remain intact.
