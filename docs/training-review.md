# EEG-X paper-run readiness

The selected defaults are implemented. Upstream non-rASR proof is no longer required under the current user policy.
Engineering checks do not establish paper scores. The [task log](task.md)
records selected manuscript differences and local changes.

| Check | Result |
|---|---|
| Local integration | EEG-X reconstruction and evaluation paths connected |
| Reproducibility | Interrupted CPU training resumes with identical weights, progress and random state |
| Configuration records | Resolved optimizer, scheduler, clipping, classifier settings, and Git provenance are saved |
| Real EEG check | Selected defaults passed pretraining, fine-tuning, frozen evaluation, and encoder export/reload on a small DREAMER subset |
| Model checks | Token masking, spectral isolation, gradient flow and checkpoint round trips passed |
| Interrupted diagnostics | Resume completes pending curves and preserves prior completed curves |
| Selection | Validation selects checkpoints; final test predictions follow saved selection locks |
| Results evidence | Evaluation saves selected artifacts, predictions and per-seed metrics; W&B handles aggregation |
| Review | User-selected defaults implemented; manuscript differences remain explicitly documented |
| Spatial port | Fixed XYZ buffers and contracts exactly match retained EEG-X2 scaling across selected widths |
| Source policy | Selected processed signals and stored labels/IDs are accepted; upstream non-rASR proof is waived |
| STEW input check | Selected processed X arrays imported without filtering/normalization; all seed splits retain both classes; primary raw and Soft DiCT CPU gradients are finite |

The available signals can test execution. Record the selected input files and complete measured seed runs before generating paper results. The [manuscript review](paper-review.md) records deliberate differences.

| Study or check | Measured state |
|---|---:|
| Real-EEG CPU check: subjects | 10 |
| Real-EEG CPU check: windows | 20 |
| Real-EEG CPU check: pretraining epochs | 1 |
| Earlier CPU test suite: passed | 131 |
| Selected STEW release: subjects | 48 |
| Selected STEW release: windows | 56,928 |
| Selected STEW release: channels | 14 |
| Selected STEW release: excluded windows | 0 |
| CUDA tests skipped in CPU suite | 3 |
| Primary-model paper scope, independent runs | 55 |
| Paper runs not yet completed | 55 |
| Completed paper seed groups | 0 |
| Shared GPU time already charged, hours | 2.5405 |
| Shared GPU cap, hours | 72 |
| Approved physical GPU | 1 |

The [performance check](pipeline-performance.md) compares original and current EEG-X
with a common raw MSE diagnostic. It excludes dictionary losses and uses a shared GPU.
Full selected-recipe performance remains unmeasured; the existing budget ledger remains intact.

Only completed measured seed groups can supply revised paper values. Group their final metrics in W&B and retain the per-seed predictions locally.
