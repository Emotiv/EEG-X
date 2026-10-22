# EEG-X reproduction: selected defaults

The user selected the defaults below. The goal remains reproducible
EEG-X evaluation with measured replacement results. The fallback column is retained
for author discussion; it does not require another approval before engineering work.
The paper column describes the supplied manuscript, independently of author code
and our implementation choices. Unspecified details are marked explicitly.

The current execution request is direct EEG-X training from one YAML file.
Each invocation selects one model and a seed list. Seeds run sequentially on the
local machine, with independent splits, outputs and W&B runs. Automatic model-size sweeps and their grid have been
removed. Study planning and orchestration have been removed; W&B groups the independent seed runs.

| Point | Paper | Selected default | Fallback |
|---|---|---|---|
| Model | C.1: encoder 8, decoder 4, heads 8. Table 8: widths 64/128/512, depths 4/8. | Train width 64/depth 8 only; no automatic model-size sweep | Corrected author table |
| Normalization | Normalization type and Q/K normalization not mentioned. | RMSNorm + QK normalization + FlashAttention | Author-code LayerNorm; keep FlashAttention |
| Masking | Section 2: MAE encoder sees visible tokens; decoder reconstructs masked regions. Sampler and ratio not mentioned; SSP not mentioned. | Reproducible random-token masking, 75% | Your preferred sampler/ratio |
| DiCT | Section 2.3: argmax winner counts. B.3: G=8, K=32; kernels 9/15; dilations 1/2/4/8/16. Soft routing not mentioned. | Soft routing; preserve paper dictionary | Your corrected soft DiCT |
| Positions | Equation 3: separable XY sin/cos, frequency base 1000. | EEG-X2 normalized XYZ, base 5 | Paper XY encoding |
| Targets | Section 2.2: rASR-cleaned targets. C.4 still describes ICA. | Use selected processed X arrays unchanged for input/target; no added rASR; upstream proof waived | Restore rASR-cleaned targets |
| Training | C.2: Adam, LR 1e-3, cosine, batch 256, 300/10 epochs. Separate fine-tuning optimizer not mentioned. | StableAdamW/AdamW + warmup/cosine; 100/20 epochs | Paper training recipe |
| Splits | C.4: official TUH splits; other datasets use 20% subject-wise test data. Seeds and validation ratio not mentioned. | Five pooled subject splits, 70/10/20 | Your preferred split ratio |
| Teacher | Section 2: MAE encoder/decoder. EMA teacher and latent prediction losses not mentioned for EEG-X. | No EMA teacher; keep trainable Predictor | Plain MAE without Predictor |
| Tokenizer/loss | Section 2.1/C.1: STFT projection, patch 128, overlap 32. Equation 7: DiCT feature MSE; extra raw MSE not mentioned. | Raw+STFT; patch 32, no overlap; raw + soft DiCT MSE | Paper tokenizer; soft DiCT only |

Epoch pairs mean pretraining/fine-tuning. Deliberate differences from the manuscript
are recorded in the [paper review](paper-review.md).

These defaults implement the user's selection. They must not be represented as
identical to every manuscript setting or as completed paper results. Soft routing
and repeated subject splits also have prior author support.

Exact current training settings:

| Stage | Optimizer and schedule |
|---|---|
| Pretraining | StableAdamW; peak LR 2.4e-4; betas (0.9, 0.95); epsilon 1e-9; weight decay 0.01; warmup 10 epochs; cosine minimum LR 1e-5; clip norm 5 |
| Fine-tuning | AdamW; peak LR 1e-3; betas (0.9, 0.999); epsilon 1e-8; matrix weight decay 0.01; layer LR decay 0.75; warmup 5 epochs; cosine minimum LR 1e-6; clip norm 1 |

These are saved settings, not manuscript-specified settings for both stages.

Five-seed protocol:

| Seed | Subject assignment | Execution |
|---|---|---|
| 241 | Save split A | Fresh model; train, select on validation, evaluate on test |
| 242 | Save split B | Fresh model; train, select on validation, evaluate on test |
| 243 | Save split C | Fresh model; train, select on validation, evaluate on test |
| 244 | Save split D | Fresh model; train, select on validation, evaluate on test |
| 245 | Save split E | Fresh model; train, select on validation, evaluate on test |

Each seed controls subject assignment and separate derived random streams for
initialization, data order, and masks. Fixed Hydra filters also have a saved per-run
seed. Subjects cannot cross roles inside a run,
but may change roles between runs. These are repeated subject holdouts, not disjoint
cross-validation folds. Report every result and the mean and sample standard
deviation; do not select the best seed. This measures combined split and training
variability. Test probes remain diagnostic. Cross-domain evaluation uses the TUAB
checkpoint with the matching seed and size.

EEG-X2 history check:

| Evidence | Fixed spatial encoding |
|---|---|
| Retained scaling implementation, commit `dfb4aac` | Frozen vocabulary XYZ; subtract vocabulary center and divide by maximum radius; base-5 frequencies from 1 to 0.2; K=floor(width/6); gain 1/sqrt(3K); zero-pad unused dimensions |
| Current EEG-X implementation | Exact fixed-buffer and encoding-contract match to retained EEG-X2 scaling at widths 64/128/512 |

The full coordinate and feature contract is reused, not only the frequency base.
Its learned-lookup comparison arm is separate; the selected encoding is fixed.

Work sequence:

- [x] Inspect the latest manuscript and scaling table.
- [x] Check retained EEG-X2 spatial encoding and current teacher behavior.
- [x] Save resolved settings and verify interrupted-run checkpoint integrity.
- [x] Apply the user-selected defaults and update the detailed specification.
- [x] Wire the versioned defaults through YAML training, checkpoints, and exports.
- [x] Check real EEG pretraining, fine-tuning, frozen evaluation, and encoder export/reload.
- [ ] Record selected artifacts, labels, subjects, and duplicate exclusions; upstream non-rASR proof is not required.
- [ ] Freeze source revision, environment, split bank, and experiment settings.
- [ ] Run the primary model and regenerate prediction-backed paper tables.

The [paper review](paper-review.md) records manuscript evidence. Train from scratch
in fresh output directories; historical checkpoint compatibility is outside scope. Save new
checkpoints for reproducible resume and keep the existing shared budget ledger.
Keep all work local until publication is explicitly requested.

Current STEW source: use the user-selected processed STEW.npy file, its saved binary labels and stored participant IDs. Preserve X values without filtering or normalization, apart from float32 conversion. Record that the original rating threshold and upstream artifact-removal history were not re-derived.
