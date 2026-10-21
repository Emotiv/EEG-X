# EEG-X paper-run readiness

This change documents the rerun contract. Model and training changes must be built on the current EEG-X implementation and reviewed before experiments begin.

| Area | Current main behavior or available evidence | Required work |
|---|---|---|
| Tokenizer | STFT magnitude input | Add aligned raw waveform features and freeze the fusion and mask boundaries |
| Encoder | Configurable depth, width, and heads | Set the agreed primary configuration, then add scaling configurations |
| Decoder | Predictor feeds reconstruction through a transposed-convolution head | Resolve how the agreed Transformer decoder fits the active reconstruction path |
| Normalization and attention | LayerNorm and manual attention | Apply agreed RMSNorm changes and verify supported FlashAttention |
| Reconstruction loss | Straight-through DiCT feature MSE only; signed root can block gradients at zero | Add agreed raw MSE and Soft Hydra; verify gradients and numerical stability |
| Teacher | Teacher targets are computed but do not enter the current loss | Resolve unused teacher work without removing the active Predictor path by accident |
| Data | Processed release exists; local label and ownership audits are available | Verify non-rASR history, participant IDs, labels, units, channel order, and duplicates |
| Splits and selection | Existing experimental partitions cannot supply the new split bank | Pool subjects, persist each seed's splits, and enforce validation-only selection |
| Training and evaluation | Current experiment code needs rerun-specific checks | Add early stopping, complete fine-tuning and frozen evaluation, resume, and saved results |

Current main is the implementation base. Configuration changes alone do not resolve its tokenizer, decoder, and loss differences from the agreed table.

| Dataset | Evidence or remaining issue |
|---|---|
| DREAMER arousal | Local arousal rule verified; select the correct task artifact and verify participant mapping |
| STEW | Processed data exists; native folder access and original rating verification remain blocked |
| TUAB | Native normal/abnormal labels and participant ownership verified; new subject splits pending |
| TUEV binary | Native binary event map verified; processed correspondence and duplicate policy need review |
| Crowdsourced | Author recording ownership and markers verified; preserve owners across acquisition blocks |
| BCICIV-2a and Kalunga | Classes and channel mappings checked locally; verify processed correspondence before frozen evaluation |

These checks are preparation evidence, not regenerated paper results. No training implementation or new experiment results are included in this change.

| Execution constraint | Value |
|---|---|
| Primary model | Encoder depth 8, embedding width 64, attention heads 8 |
| Shared GPU cap | 72 GPU-hours |
| Previously charged local usage | Approximately 2.52 GPU-hours |
| Device | Physical GPU 1 |
| Storage | Work volume for environments, data, caches, and outputs |

Carry existing GPU usage forward. Review the implementation, fix findings, and verify the real-data pipeline before full training.
