# EEG-X reproduction: selected defaults

The latest author-confirmed protocol below takes precedence over earlier local
choices. The user resumed implementation to populate W&B with bounded, real-data
review runs before the full paper experiments. These review runs use reduced data
and a shorter pretraining horizon; they do not supply paper scores.

The public stage name for linear probing is `linear_probe` in YAML and W&B.
The same name appears in W&B Job Type and tracking records.

## Latest raw-data preparation decision

The confirmed next revision restores the original sample-selection rules while
replacing signal preprocessing with the EEG-X2 QuantPlus recipe. The saved
processed NPY datasets define the target sample counts and crops. Archived scripts
supply reconstruction evidence; they are not authoritative when they disagree with
the saved artifacts. Keep the agreed classification tasks, including binary
DREAMER arousal and binary TUEV.

| Part | Confirmed requirement |
|---|---|
| Sampling | Restore original crop lengths, strides, overlap and sample multiplicity. Overlapping windows are valid separate samples. |
| Preprocessing | Preserve native reference; high-pass 0.5 Hz and low-pass 45 Hz; resample from the actual native rate to 128 Hz; per-record/channel StandardScaler; clip at ±15; store float32. Add neither notch nor average reference. |
| Repeated signals | Retain the reference multiplicity. Keep duplicate-linked subjects together in a partition, so subject ratios may be approximate. Do not silently remove samples to deduplicate the release. |
| Target splits | Preserve the five subject-wise banks, seeds 241–245, with target ratio 70/10/20. Group linked subjects before assigning roles. |
| TUAB source split | Preserve the separate subject-wise 80/20 source train/validation bank. |
| Crowdsourced exception | Correct actual sampling-rate timing and use true two-second windows. Keep windows within continuous acquisition blocks and one label interval. Restore overlap, but report resulting count differences from the old concatenation/majority-label catalog. |
| PhysioNet timing correction | Preserve the proven event catalog, including later short rest annotations, and omit the initial event. Historical fixed 320-native-sample crops become true 2-second model windows at actual native rates; preserve exact reference class totals. |
| Missing raw mapping | Leave the affected dataset unresolved and report the mismatch. Do not guess crop offsets, insert windows or silently substitute another signal release. Other verified datasets may proceed. |
| Storage | One HDF5 per dataset, containing signals, labels, record/subject/channel metadata, source hashes, stable sample identities and saved split indices. Preserve original-row provenance where established. |

These choices were confirmed in the requirements interview. Correct physical timing
and the explicit Crowdsourced boundary exception take priority over matching an
old count. Other count differences require investigation and an explicit report.

| Saved reference | Target sample count | Reconstruction status |
|---|---:|---|
| DREAMER arousal | 170,246 | Original stride reproduces the count exactly; convert ratings using the agreed binary task rule. |
| STEW | 56,928 | Original stride reproduces the count exactly. |
| TUAB | 409,455 | Original window catalog established; retain reference multiplicity. |
| TUEV binary | 98,684 | Binary annotation-crop replay reproduces the count and class counts exactly. |
| Crowdsourced | 13,363 | Original selection replay matches; the confirmed timing/boundary exception requires a new measured count. |
| BCICIV-2a | 5,184 | Complete original waveform replay verifies the 2-second crop: native MAT trial index minus 1, then +500 samples at 250 Hz. Preserve feet/left/right/tongue label IDs. |
| Kalunga / SSVEP | 960 | Complete original waveform replay verifies the pinned MOABB 30-run selection. Retain corrected physical frequency labels and record the legacy 17/21 Hz name mismatch. |
| PhysioNetMI | 19,022 | Complete original waveform replay verifies all 19,022 rows. Omit the initial event, retain later annotations, and use the proven 320-native-sample eligibility with the explicit timing correction. |

The reference counts combine the old training, validation and test arrays. They do
not preserve the old partitions; the new subject banks determine membership.

Implementation now covers every verified reference catalog and groups duplicate-linked
subjects. All datasets are regenerated and audited. Every established original-array row mapping, raw bound, grouped split and reference count passed validation.
Established original-array row lineage and artifact hashes are embedded in the
mapped datasets and validated before publication. The earlier non-overlap release is archived; its
audit applies only to its recorded protocol. The
[dataset preparation plan](dataset-preparation.md) describes the implemented rules; the [signal review](notch-reference-review.md) supports the retained
preprocessing choices. Update code, regenerate affected datasets and audit counts,
class mappings and grouped subject splits after implementation begins.

## Latest author-confirmed protocol

| Setting | In-domain | Cross-domain |
|---|---|---|
| Datasets | DREAMER arousal, STEW, TUAB, TUEV_binary | Crowdsourced, BCICIV-2a, SSVEP/Kalunga, PhysioNetMI |
| Subject splits | Five independent train/validation/test splits per dataset | Five independent target train/validation/test splits per dataset |
| Target split ratio | 70% / 10% / 20%, subject-wise | 70% / 10% / 20%, subject-wise |
| Pretraining source | Each dataset's training subjects | TUAB training subjects, with a separate TUAB validation set |
| Pretraining runs | Five fresh models per dataset | One TUAB model shared by every target dataset and target seed |
| Pretraining horizon | 100 epochs | 100 epochs |
| Pretraining checkpoint | Lowest validation reconstruction loss | Lowest TUAB validation reconstruction loss |
| Training-time linear probes | Every 10 pretraining epochs, with validation and diagnostic test curves in the same run | No target probe schedule during source pretraining was requested |
| Selected-checkpoint probe | Run again after loading the validation-selected checkpoint | Fit separately on every target split using the shared TUAB checkpoint |
| Linear-probe horizon | At most 20 epochs per fit | At most 20 epochs per fit |
| Fine-tuning horizon | At most 20 epochs per run | At most 20 epochs per run |
| Probe/fine-tuning early stopping | Patience 5 validation checks without improvement | Patience 5 validation checks without improvement |

The author confirmed the horizons, subject-wise target ratios, validation for TUAB
pretraining, and a shared source model. The proposed TUAB source ratio is recorded
below as a remaining detail rather than an explicit author confirmation.

| Latest user-selected execution setting | Choice |
|---|---|
| Pretraining batches | Microbatch 512 and logical batch 4096, capped by available training windows |
| GPU backends | Enable fused Soft DiCT and RMSNorm, with native RMSNorm fallback where needed |
| Determinism and allocation | Keep deterministic algorithms enabled; disable the uninitialized-memory diagnostic fill |
| Downstream execution | Use fused RMSNorm for linear probing and fine-tuning, including native-backend checkpoints with the same software and environment contract |
| Quality interpretation | Larger logical batches change optimizer update frequency. Numerical checks do not establish the final paper scores |

These are user-selected execution settings. The existing subject splits and validation-based selection remain the scientific protocol.

For each in-domain split, pretrain only on training subjects. Fit each linear probe
with the encoder frozen. Save the best reconstruction checkpoint using validation
loss, probe that checkpoint again, and start fine-tuning from the same checkpoint.
Test curves cannot select checkpoints or training settings. The latest user clarification selects each
linear head and fine-tuned model by validation balanced accuracy. Pretraining
continues to select the encoder by validation reconstruction loss.

For cross-domain experiments, pool the selected TUAB release before assigning its
source training and validation subjects. Pretrain once and reuse the selected
encoder. For each target split, start linear probing and fine-tuning independently
from that shared encoder. Fit probes and fine-tuning only on target training
subjects. Reuse the same target split across both methods. Select downstream
checkpoints on validation, then evaluate the selected checkpoint on test once.

| Seed role | Selected values | Meaning |
|---|---|---|
| In-domain split/training seeds | 241, 242, 243, 244, 245 | Fresh split and fresh pretrained model for each dataset and seed |
| Cross-domain target seeds | 241, 242, 243, 244, 245 | Target splits and downstream randomness; the source encoder stays unchanged |
| Cross-domain TUAB source seed | 241, existing local default | Reproducibility seed for the shared source run; no matching target seed is required |

Subject separation prevents overlap within a split. These are repeated subject
holdouts, not disjoint cross-validation folds. Report all completed target seeds
and their mean and sample standard deviation.

| Remaining detail or implementation gap | Status |
|---|---|
| TUAB source train/validation ratio | 80% / 20%, subject-wise, was proposed; the author confirmed a validation set without restating the exact ratio |
| Probe/fine-tuning stopping metric | Latest user decision: maximize validation balanced accuracy (`val/bal_acc`) for both linear probes and fine-tuning; minimum improvement zero; retain the first checkpoint on ties |
| Epoch-based linear probes | Implemented: frozen encoder, train-only feature scaler, linear head with AdamW, validation selection and patience |
| PhysioNetMI | Complete native HDF5 release verified: 109 subjects, 654 imagery EDFs, 64 physical channels and saved subject-wise banks; participant identities come from native filenames |
| Shared source selection | Implemented TUAB train/validation split and best validation-loss checkpoint; the superseded full-source/final-epoch mode is removed |

These details do not change the confirmed experiment structure. Full-data runs
and complete downstream source evidence are still required for paper results.

| Planned runs for the primary model | Count | Epoch limit per run |
|---|---:|---:|
| In-domain pretraining | 20 | 100 |
| In-domain fine-tuning | 20 | 20 |
| Shared TUAB source pretraining | 1 | 100 |
| Cross-domain linear probes | 20 | 20 |
| Cross-domain fine-tuning | 20 | 20 |
| Total independently tracked runs | 81 | Not applicable |

In-domain probe fits stay inside pretraining runs and do not add independently
tracked runs. The counts cover the primary model without ablations or baselines.

## W&B native grouping

Keep all experiments in the same W&B project. Each seed logs its own run and
metrics. W&B performs seed aggregation; the repository needs no aggregation script
or extra summary run.

| Item | Requirement |
|---|---|
| Project | `eegx` |
| Default display | Individual runs; no preset run groups or composite group names |
| Manual grouping | Choose `task_name` in W&B when needed; use stage/domain filters to select the intended experiment |
| Domains | `in_domain`, `cross_domain` |
| Domain logging | Set `wandb.domain` per configuration; save it in run config, tracking records and model artifact metadata |
| Stages | `pretrain`, `linear_probe`, `finetune` |
| Embedded probe sections | Log canonical `linear_probe/val/*` and `linear_probe/test/*` history keys inside the pretraining run. Show all applicable classification metric trends against pretraining epoch, plus confusion images; test curves remain diagnostic |
| Repeated runs | 5 distinct completed seeds per target experiment; keep test and incomplete runs out of paper averages |
| Curves | Use W&B grouping with mean aggregation; optional standard-deviation shading |
| Final scores | Evaluate the validation-selected checkpoint once per run; log scalar test metrics once to W&B history and summary so Automated workspaces create charts without manual setup. Final scores remain one value per run; seed grouping uses the summaries; diagnostic probes keep separate epoch curves |
| Test metrics | Accuracy, balanced accuracy, F1 micro/macro/weighted, macro precision and recall, AUROC, AUPRC and classification loss; binary tasks also report positive-class F1 |
| Selection record | Log the metric and direction in W&B and checkpoint records; downstream selection maximizes `val/bal_acc` |
| Classifier workspace | Match Neuralbench DREAMER valence in both fine-tuning domains; use actual optimizer names, epoch, best_epoch and trainer/global_step. Native bars use bare summary metric keys. Pretraining adds diagnostic probe trends |
| System and console logs | Enable native W&B system monitoring for every tracked run; show resource metrics in System and stdout/stderr in Logs |
| Model and optimization details | Log total/trainable parameters, fine-tuning learning rates, classification loss and selected checkpoint epoch |
| Confusion matrices | Log validation and final test matrices as images, matching Neuralbench; no raw confusion-matrix tables; embedded probes keep validation and diagnostic test outputs separate |
| Selected model artifacts | Upload the validation-selected pretraining and fine-tuning checkpoints with the best alias and their selection metadata |
| Shared TUAB source | 1 source run reused by downstream experiments; no source seed average |

Grouping is chosen in W&B rather than assigned automatically by the script.
Keep the same model, domain, stage and recipe when averaging the selected seeds.

## Retained local model and optimizer choices

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
| Targets | Section 2.2: rASR-cleaned targets. C.4 still describes ICA. | Use the native EEG-X2-preprocessed HDF5 X arrays for input/target; no added rASR; historical imports keep their recorded recipe | Restore rASR-cleaned targets |
| Training | C.2: Adam, LR 1e-3, cosine, batch 256, 300/10 epochs. Separate fine-tuning optimizer not mentioned. | StableAdamW/AdamW + warmup/cosine; 100/20 epochs | Paper training recipe |
| Splits | C.4: official TUH splits; other datasets use 20% subject-wise test data. Seeds and validation ratio not mentioned. | Subject-wise target holdouts as confirmed above; shared TUAB source has a separate train/validation assignment | Your preferred source split ratio |
| Teacher | Section 2: MAE encoder/decoder. EMA teacher and latent prediction losses not mentioned for EEG-X. | No EMA teacher; keep trainable Predictor | Plain MAE without Predictor |
| Tokenizer/loss | Section 2.1/C.1: STFT projection, patch 128, overlap 32. Equation 7: DiCT feature MSE; extra raw MSE not mentioned. | Raw+STFT; patch 32, no overlap; raw + soft DiCT MSE | Paper tokenizer; soft DiCT only |

Epoch pairs mean pretraining/fine-tuning. Deliberate differences from the manuscript
are recorded in the [paper review](paper-review.md).

These defaults record the user's selection. They must not be represented as
identical to every manuscript setting or as completed paper results. Soft routing
and repeated subject splits also have prior author support.

Exact current training settings:

| Stage | Optimizer and schedule |
|---|---|
| Pretraining | StableAdamW; peak LR 2.4e-4; betas (0.9, 0.95); epsilon 1e-9; weight decay 0.01; warmup 10 epochs; cosine minimum LR 1e-5; clip norm 5 |
| Fine-tuning | AdamW; peak LR 1e-3; betas (0.9, 0.999); epsilon 1e-8; matrix weight decay 0.01; layer LR decay 0.75; warmup 5 epochs; cosine minimum LR 1e-6; clip norm 1 |

These are saved settings, not manuscript-specified settings for both stages.

In-domain seed bank:

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
variability. Test probes remain diagnostic. Cross-domain target seeds reuse the
shared TUAB checkpoint with the matching architecture, independently of its source seed.

Evaluation summary:

| Stage | Agreed behavior |
|---|---|
| In-domain pretraining | Fresh models for the target seed bank; linear-probe validation and diagnostic test trend lines inside each pretraining run |
| Cross-domain pretraining | Shared TUAB model with validation-based checkpoint selection |
| In-domain evaluation | Fine-tuning validation trend lines; select the best validation checkpoint, then compute the final test score once |
| Standalone linear probes | Cross-domain experiments only; use the shared TUAB encoder |
| Cross-domain fine-tuning | Fine-tune the shared encoder separately on every target split |
| Leakage control | Persist subject-separated splits and train only on training subjects; additional seeds alone do not establish separation |

In-domain linear probes do not create separate runs. Test trend lines monitor
representation quality and cannot select checkpoints or training settings.

EEG-X2 history check:

| Evidence | Fixed spatial encoding |
|---|---|
| Retained scaling implementation, commit `dfb4aac` | Frozen vocabulary XYZ; subtract vocabulary center and divide by maximum radius; base-5 frequencies from 1 to 0.2; K=floor(width/6); gain 1/sqrt(3K); zero-pad unused dimensions |
| Current EEG-X implementation | Exact fixed-buffer and encoding-contract match to retained EEG-X2 scaling at widths 64/128/512 |

The full coordinate and feature contract is reused, not only the frequency base.
Its learned-lookup comparison arm is separate; the selected encoding is fixed.

Work sequence:

| Fresh rerun execution | Selected behavior |
|---|---|
| Reference project | Rename old `eegx` to `eegx-bs256`; preserve its 23 runs and local results |
| New project | Create private `eegx` with fresh run identities |
| Study | Restart all 81 tasks using the existing HDF5 files and saved subject splits |
| Batches | Pretraining microbatch 512, global batch 4096 |
| Concurrency | Capacity checks passed with 7 simultaneous TUAB processes on each physical GPU |
| Time policy | User removed the previous 72 GPU-hour cap for the fresh rerun; retain uncapped usage meters |
| Ownership | Fresh outputs, per-output locks, dependency completion checks, frozen reviewed source and safe interruption |

Do not overwrite or reset the old attempts and accounting. Review the generated
queue before starting training. Downstream runs use only the selected checkpoints
from this fresh study.

- [x] Inspect the latest manuscript and scaling table.
- [x] Check retained EEG-X2 spatial encoding and current teacher behavior.
- [x] Save resolved settings and verify interrupted-run checkpoint integrity.
- [x] Apply the user-selected defaults and update the detailed specification.
- [x] Wire the versioned defaults through YAML training, checkpoints, and exports.
- [x] Check real EEG pretraining, fine-tuning, linear-probe evaluation, and encoder export/reload.
- [x] Record selected artifacts, labels, subjects, and duplicate exclusions; upstream non-rASR proof is not required.
- [x] Freeze source revision, environment, split bank, and experiment settings.
- [ ] Run the primary model and regenerate prediction-backed paper tables.

The [paper review](paper-review.md) records manuscript evidence. Train from scratch
in fresh output directories; historical checkpoint compatibility is outside scope. Save new
checkpoints for reproducible resume and preserve historical usage ledgers separately.
Keep all work local until publication is explicitly requested.

Current native release: prepare raw sources with the documented EEG-X2 recipe
and write one HDF5 file per dataset. Historical processed releases remain explicit
imports with their own recorded preprocessing; they are not relabelled as the new
native release. STEW and PhysioNet default to complete raw subject scopes.
