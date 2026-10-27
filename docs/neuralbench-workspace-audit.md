# Neuralbench workspace audit

| Section | Neuralbench project shown by user | Actual DREAMER run data | EEG-X layout |
|---|---:|---:|---|
| Charts | 3 | Optimizer rate, epoch, best epoch | 3 root charts |
| test | 47 | 8 metric/media keys | Classification metrics, final-checkpoint bars and confusion matrix |
| train | 2 | 2 loss keys | Automatic overview; 2 classifier loss charts plus pretraining diagnostics |
| trainer | 1 | 1 step key | 1 optimizer-step chart |
| val | 15 | 8 metric/media keys | Classification curves and selection; reconstruction validation for pretraining |
| System | 22 | Native monitoring | Native automatic monitoring retained |

The reference project also contains retrieval and regression experiments. Its automatic section counts do not represent the metrics produced by a single DREAMER classifier. EEG-X keeps the applicable classification metrics and the additional agreed scores.

| Pretraining probe trend | Data binding | Horizontal axis |
|---|---|---|
| Validation balanced accuracy | `probe/validation/bal_acc` | Epoch, evaluated every 10 epochs |
| Diagnostic test balanced accuracy | `diagnostic/test/bal_acc` | Epoch, evaluated every 10 epochs |

The overview and in-domain pretraining view retain these separate probe charts. Reconstruction raw MSE still selects the pretraining checkpoint.

| Audit result | State |
|---|---|
| Personal overview and saved stage views | 6 visible views updated and read back from W&B |
| Validation SoftDiCT logging | Implemented and reviewed; not installed in the active frozen queue |
| Missing historical validation SoftDiCT | Cannot be recovered from charts or summaries |
| Focused regression checks | 23 passed |
| Standards and Spec review | 0 open findings |
| Active training jobs | Preserved without restart or source edits |

The new validation path preserves the original masked-MSE reduction and random state. A rollout decision is required for the running campaign because earlier validation auxiliary values were never computed.
