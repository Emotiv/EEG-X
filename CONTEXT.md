# EEG-X paper reproduction

**Paper rerun**: Regenerate EEG-X evaluation results using the user-selected proposal as the default. docs/task.md separates manuscript descriptions, selected defaults, and fallbacks for author discussion. docs/spec.md is the implemented local protocol; docs/training-review.md records checks and remaining data requirements. The user's selection authorizes implementation without another confirmation step.

**Current paper source**: The author-supplied manuscript, EEG-X: Toward Device-Agnostic and Noise-Robust Foundation Models for EEG, was reviewed locally and is not included in this code release. docs/paper-review.md supersedes the earlier public arXiv comparison. It uses an asymmetric MAE and does not specify SSP or EMA-teacher latent objectives. Its scaling table is present. Record the deliberate selected differences and draft inconsistencies; do not claim identical manuscript methodology.

**Hybrid tokenizer**: Aligned raw waveform patches and STFT magnitude features representing the same electrode and time interval. Each spectral window stays inside its raw patch.

**Channel identity**: A stable physical electrode identity and explicit channel order. A channel count alone does not establish the montage.

**Processed source policy**: The user explicitly waived proof of upstream non-rASR history. Accept the selected processed data and reconstruct the same X arrays supplied as input; add no local rASR or extra filtering/normalization. Record unknown upstream processing as unknown, rather than claiming it was verified. Preserve artifact hashes, stored labels, subject IDs and explicit physical channel order. STEW uses /data/EEG-X/Processed/STEW/STEW.npy with its saved binary labels and subject IDs; original ratings need not be re-derived for this selected release.

**Seed split**: A persisted subject-wise train/validation/test assignment after pooling the old partitions. Reuse each seed's assignment across model configurations. Subjects and unresolved duplicate groups must not cross partitions.

**Reconstruction objective**: Default raw masked MSE plus Soft DiCT feature MSE. Preserve the paper dictionary: eight groups, thirty-two kernels, lengths nine and fifteen, and dilations one through sixteen in powers of two. Use softmax assignments and temporal sums without extra raw/difference branches, amplitude weights, square-root features, or temporal means. Filters are fixed, centered, unit-L2, locally seeded, and saved. The target is detached. Explicit objective=None remains a raw-only ablation.

**Selection lock**: Checkpoint, early-stopping, threshold, and hyperparameter decisions use validation data. Diagnostic test probes stay separate from final results and cannot guide selection.

**Reproducibility record**: Source revision, locked environment, resolved configuration, preprocessing, physical channel identities, source evidence, split assignments, random states, checkpoints, predictions, exclusions, and measured metrics.

**Fresh training**: The user will train new models from scratch. Do not add support for loading or migrating historical checkpoints. Checkpoint contracts protect integrity and reproducible resume within new runs; downstream evaluation uses the newly trained encoders.

**Checkpoint source**: New checkpoints bind the pretraining task, source release and subject split. Fine-tuning must use the same dataset and split. Frozen transfer to Crowdsourced, BCICIV-2a or Kalunga must use TUAB with the matching seed and model. Enforce this in both the YAML command and downstream evaluation; reject mismatched or missing source provenance before test access and W&B result logging.

**GPU budget**: One persistent ledger covers all local experiments and paper runs. Carry existing usage forward. The approved device, cap, and storage policy are defined in the specification.

**GPU performance check**: The user selected physical GPU zero for the current bounded training checks and approved sharing it with existing workloads. Shared checks have an explicit allocator limit, retain the existing budget, and record contention. Their timings cannot establish a code speedup. Paper studies keep their planned device.

**Implementation status**: The local revised pipeline reuses the EEG-X context encoder and active Predictor. Model definitions live in pretrain/models; the single training and evaluation pipeline lives in pretrain/pipeline. StableAdamW lives in pretrain/optimizers. CPU checks establish engineering behavior only. Paper scores require recorded inputs and targets, independent seed splits, locked selections, and completed measured runs.

**Ported optimizations**: Shared RMSNorm, compute policy, optimizer, batching, and checkpoint behavior belong to EEG-X. The separate raw EEGX2 backbones and corpus runners are retained in Git history. The channel vocabulary keeps its original physical identities and fingerprint.

**Direct training**: The current workflow is one YAML configuration and one local command: python Main.py --config configs/eegx.yaml. The configuration selects the prepared manifest, output, model, seeds, batches, optimizer, schedule and device. A seeds list runs sequentially in fresh local processes on the selected GPU. Each seed saves its subject split, checkpoints and W&B run under its own output directory. A scalar seed still selects one run. Input errors exit immediately. There is no SLURM, study planner or automatic model-size sweep.

**Unified entrypoint**: Main.py accepts one YAML configuration. The stage selects pretraining, fine-tuning or frozen evaluation. Study planning, controllers, task registries and local seed-aggregation code have been removed. The former root Models and experiments directories and unused baseline implementations remain in Git history.

**Dataset readers**: Historical array loaders and the raw STEW reader live in pretrain/datasets. Native source adapters, audited release preparation, and training data iteration remain in pretrain/pipeline. Moving source code does not change external data paths or archived experiment records.

**Model defaults**: Train EEG-X with depth 8, width 64 and eight attention heads. YAML selects one explicit model configuration. Keep RMSNorm, Q/K normalization, FlashAttention, deterministic MAE-style masking, the hybrid tokenizer, and the active trainable Predictor without teacher/EMA. Training remains the selected pretraining/fine-tuning recipe. Receipts record resolved settings and Git provenance. Final evaluation cannot overwrite the matching training model and optimizer snapshot.

**Dataset publication**: Publish a prepared manifest only after label/subject checks and release metadata are finalized. Upstream rASR history is optional. The as-provided processed-release profile records float32 casting as its only local operation; do not claim the raw preprocessing profile was applied to these values. No persistent task state blocks a later retry.

**EEG-X2 spatial default**: Fixed normalized XYZ sin/cos features use base five, the frozen vocabulary center/radius, feature normalization, and width padding. The port matches retained EEG-X2 scaling buffers and contracts exactly across selected widths. Use fresh training outputs while preserving the shared budget ledger.

**Repeated-run interpretation**: Pool old subject partitions and save independent subject holdouts per seed. Each seed has fresh initialization and training; reuse its subjects across model comparisons. These are repeated holdouts, not disjoint cross-validation folds. Report all seeds and their mean/sample deviation. Current reconstruction uses no teacher or EMA, but retains a trainable cross-attention Predictor.

**W&B tracking**: One run records one model, task, stage and seed. Model/task metadata and a seed-independent recipe identity are derived automatically. Online resume reuses the saved run ID. W&B groups runs and computes seed averages; the repo has no seed-summary controller. Group by model and task, filter to one stage and recipe, and include distinct completed seeds. Final test metrics remain separate from diagnostic test probes.
