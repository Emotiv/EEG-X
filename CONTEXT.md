# EEG-X paper reproduction

**Paper rerun**: Regenerate the EEG-X evaluation results using the author-confirmed requirements in docs/spec.md. Adapt the current main implementation with minimal changes. The specification is the target; docs/training-review.md records the remaining work.

**Hybrid tokenizer**: Aligned raw waveform patches and STFT magnitude features representing the same electrode and time interval.

**Channel identity**: A stable physical electrode identity and explicit channel order. A channel count alone does not establish the montage.

**Non-rASR source**: Input with verified evidence that upstream rASR was not applied. Preparation must preserve that evidence and must not apply rASR. Unknown source history blocks reuse.

**Seed split**: A persisted subject-wise train/validation/test assignment after pooling the old partitions. Reuse each seed's assignment across model configurations. Subjects and unresolved duplicate groups must not cross partitions.

**Reconstruction objective**: Masked raw waveform MSE plus Soft Hydra feature MSE. The target is detached, feature filters are fixed and seeded, and the full numerical configuration must be saved.

**Selection lock**: Checkpoint, early-stopping, threshold, and hyperparameter decisions use validation data. Diagnostic test probes stay separate from final results and cannot guide selection.

**Reproducibility record**: Source revision, locked environment, resolved configuration, preprocessing, physical channel identities, source evidence, split assignments, random states, checkpoints, predictions, exclusions, and measured metrics.

**GPU budget**: One persistent ledger covers all local experiments and paper runs. Carry existing usage forward. The approved device, cap, and storage policy are defined in the specification.

**Implementation status**: This change contains documentation only. Model changes, data preparation, training, and evaluation still require implementation and review against the agreed contract.
