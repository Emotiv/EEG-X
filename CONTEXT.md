# EEG-X paper reproduction

**Paper rerun**: Regenerate EEG-X evaluation results using the user-selected proposal as the default. docs/task.md separates manuscript descriptions, selected defaults, and fallbacks for author discussion. docs/spec.md is the implemented local protocol; docs/training-review.md records checks and remaining data requirements. The user's selection authorizes implementation without another confirmation step.

**Current paper source**: The author-supplied manuscript, EEG-X: Toward Device-Agnostic and Noise-Robust Foundation Models for EEG, was reviewed locally and is not included in this code release. docs/paper-review.md supersedes the earlier public arXiv comparison. It uses an asymmetric MAE and does not specify SSP or EMA-teacher latent objectives. Its scaling table is present. Record the deliberate selected differences and draft inconsistencies; do not claim identical manuscript methodology.

**Hybrid tokenizer**: Aligned raw waveform patches and STFT magnitude features representing the same electrode and time interval. Each spectral window stays inside its raw patch.

**Channel identity**: A stable physical electrode identity and explicit channel order. A channel count alone does not establish the montage.

**Non-rASR source**: The user selected no rASR in either inputs or targets. Reconstruct the same processed signal supplied as input. This differs from the manuscript's rASR-cleaned target setup. Preserve attributable upstream processing evidence and audit available signals. Do not describe this policy as a manuscript requirement or bypass source verification.

**Seed split**: A persisted subject-wise train/validation/test assignment after pooling the old partitions. Reuse each seed's assignment across model configurations. Subjects and unresolved duplicate groups must not cross partitions.

**Reconstruction objective**: Default raw masked MSE plus Soft DiCT feature MSE. Preserve the paper dictionary: eight groups, thirty-two kernels, lengths nine and fifteen, and dilations one through sixteen in powers of two. Use softmax assignments and temporal sums without extra raw/difference branches, amplitude weights, square-root features, or temporal means. Filters are fixed, centered, unit-L2, locally seeded, and saved. The target is detached. Explicit objective=None remains a raw-only ablation.

**Selection lock**: Checkpoint, early-stopping, threshold, and hyperparameter decisions use validation data. Diagnostic test probes stay separate from final results and cannot guide selection.

**Reproducibility record**: Source revision, locked environment, resolved configuration, preprocessing, physical channel identities, source evidence, split assignments, random states, checkpoints, predictions, exclusions, and measured metrics.

**Fresh training**: The user will train new models from scratch. Do not add support for loading or migrating historical checkpoints. Checkpoint contracts protect integrity and reproducible resume within new runs; downstream evaluation uses the newly trained encoders.

**GPU budget**: One persistent ledger covers all local experiments and paper runs. Carry existing usage forward. The approved device, cap, and storage policy are defined in the specification.

**GPU performance check**: The user selected physical GPU zero for the current bounded training checks and approved sharing it with existing workloads. Shared checks have an explicit allocator limit, retain the existing budget, and record contention. Their timings cannot establish a code speedup. Paper studies keep their planned device.

**Implementation status**: The local revised pipeline reuses the EEG-X context encoder and active Predictor. Model definitions live in pretrain/models; the single training and evaluation pipeline lives in pretrain/pipeline. StableAdamW lives in pretrain/optimizers. CPU checks establish engineering behavior only. Paper scores require aligned and verified source/target processing, independent seed splits, locked selections, and completed measured runs.

**Ported optimizations**: Shared RMSNorm, compute policy, optimizer, batching, and checkpoint behavior belong to EEG-X. The separate raw EEGX2 backbones and corpus runners are retained in Git history. The channel vocabulary keeps its original physical identities and fingerprint.

**Unified entrypoint**: python -m pretrain starts the EEG-X study pipeline. Main.py paper is a compatibility command for that same path. The former root Models and experiments directories and unused baseline implementations remain in Git history.

**Dataset readers**: Historical array loaders and the raw STEW reader live in pretrain/datasets. Native source adapters, audited release preparation, and training data iteration remain in pretrain/pipeline. Moving source code does not change external data paths or archived experiment records.

**Model defaults**: Primary depth 8 and width 64 runs first. Scaling follows manuscript depths 4/8 and widths 64/128/512. Keep RMSNorm, Q/K normalization, FlashAttention, deterministic MAE-style masking, the hybrid tokenizer, and the active trainable Predictor without teacher/EMA. Training remains the selected pretraining/fine-tuning recipe. Receipts record resolved settings and Git provenance. Final evaluation cannot overwrite the matching training model and optimizer snapshot.

**EEG-X2 spatial default**: Fixed normalized XYZ sin/cos features use base five, the frozen vocabulary center/radius, feature normalization, and width padding. The port matches retained EEG-X2 scaling buffers and contracts exactly across selected widths. Use fresh training studies while preserving the shared budget ledger.

**Repeated-run interpretation**: Pool old subject partitions and save independent subject holdouts per seed. Each seed has fresh initialization and training; reuse its subjects across model comparisons. These are repeated holdouts, not disjoint cross-validation folds. Report all seeds and their mean/sample deviation. Current reconstruction uses no teacher or EMA, but retains a trainable cross-attention Predictor.
