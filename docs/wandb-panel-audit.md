# W&B classifier panel audit

The reference is Neuralbench’s current DREAMER valence classifier. Fine-tuning uses a common layout across domains; pretraining adds diagnostic probe views.

| Finding | Correction |
|---|---|
| Empty final-test and model bars | Native bar bindings now use the real summary keys, without a duplicated summary prefix |
| Missing optimizer and progress panels | Added actual-optimizer learning-rate keys, epoch, selected epoch and persistent optimizer-step progress |
| Stale cross-domain metric names | Replaced old balanced-accuracy and training-loss bindings with current names |
| Old date filters and deselected runs | Removed stale searches and selections; retained manual grouping choices |
| Default workspace still showed demos | Updated the actual personal project workspace as well as the saved stage views |

The training recipe and checkpoint selection are unchanged. Pretraining is labelled StableAdamW; the existing fine-tuning and standalone probe implementation is labelled AdamW.

| Verification | Result |
|---|---|
| Updated workspaces | 6 |
| Configured panels checked | 252 |
| Fine-tuning layouts | Identical in-domain and cross-domain; 34 panels each |
| Unresolved metric bindings | 0 against the compatible logged schema |
| Regression checks | 185 passed; 3 CUDA-only checks skipped |
| Repaired recorded optimizer updates | Pretraining 306; fine-tuning 1,224 |
| Existing runs | 2 completed runs retained; no new training |
| Scientific results | Existing final-test scalars, checkpoint artifacts and logs retained |
| Final test display | Summary-only bars; no test-scalar epoch curves |

Recorded learning rates and optimizer-step counts supplied the logging repair. Training was not repeated and test evaluation was not repeated.

Cross-domain templates have no current matching experiment after the requested run cleanup. Their panel bindings are checked against the same classifier schema, without inventing cross-domain results.


| Selected-checkpoint display | Selection and location |
|---|---|
| Pretraining, either domain | Minimum validation raw MSE; selected/val/raw_mse bar in val; best_epoch in Charts |
| Fine-tuning, either domain | Maximum validation balanced accuracy; selected/val bars in val; selected-checkpoint test bars in test |
| Cross-domain linear probing | Same selected/val and test layout; encoder stays frozen |
| In-domain embedded probing | Selected probe bars in linear_probe/val and linear_probe/test; test remains diagnostic |

Validation curves still show each training epoch. The selected bars identify the checkpoint used for final evaluation.
