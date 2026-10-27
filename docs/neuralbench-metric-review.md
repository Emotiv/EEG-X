# Neuralbench metric coverage review

| Area | Verified EEG-X coverage |
|---|---|
| Neuralbench classifier core | Loss, accuracy, balanced accuracy, AUROC, AUPRC, F1 micro/macro and confusion images |
| Additional requested classifier scores | Weighted F1, macro precision/recall, positive-class binary F1 |
| Pretraining reconstruction | Train and validation raw MSE and SoftDiCT; validation raw MSE selects the checkpoint |
| Embedded probe history | All 11 classification metrics per validation/test role; 2 epoch points each in the short test |
| Probe presentation | 11 line charts and 1 confusion panel per role; 24 probe panels total |
| Final classifier test | All 11 scores in 1 history event and Summary after checkpoint selection |
| Native tracking | Selected artifacts, parameters, optimizer rate, epoch, best epoch, step, System and Logs |
| Reviews | Standards 0 open findings; Spec 0 open findings |
| Runtime verification | 5 fresh engineering runs completed; 26 focused checks passed |

Metric coverage matches Neuralbench’s classification outputs and includes the agreed extras. Pretraining probe axes use pretraining epochs; diagnostic test scores do not influence encoder selection.

| Definition detail | EEG-X | Neuralbench default |
|---|---|---|
| Binary AUPRC | Positive-class probability, as recorded in the task specification | Macro average precision across classes |
| Multiclass AUPRC | Macro one-vs-rest average precision | Macro average precision |

Coverage does not imply identical binary AUPRC values. Retrieval and regression fields belong to other Neuralbench tasks and are outside this classification study.
