# EEG-X fixed-split rerun

This protocol supersedes the archived multi-split protocol. The agreed model is the original EEG-X recipe from `quoc-emotiv/eegx-bs256`, not an EEG-X2 or normalized SoftDiCT research variant. The machine-readable authority is `reproduction/eegx-fixed-v1/protocol.json`.

| Setting | Agreed value |
|---|---|
| Model | Hybrid raw/STFT EEG-X; encoder depth 8, width 64, heads 8; active Predictor; decoder depth 4; all RMSNorm |
| Objective | Masked raw MSE plus original full-dictionary SoftDiCT temporal-count MSE, weight 1, temperature 1 |
| Training seeds | 241, 242, 243; control model initialization, masks, dropout and training order |
| Split seeds | Independent of training seeds; one frozen train/validation/test membership per dataset |
| Pretraining | Maximum 100 epochs, StableAdamW; warmup 10; validation raw-MSE selection; existing patience 10 after warmup |
| Pretraining batches | Microbatch = logical batch = 256 |
| Probe and fine-tuning | Maximum 20 epochs, AdamW; validation balanced-accuracy selection; patience 5 |
| Fine-tuning and probe batches | Microbatch = logical batch = 64; embedded probe head batch 64 |
| In-domain probe | Every 10 pretraining epochs, plus selected-checkpoint probe; validation and diagnostic test trends embedded in pretraining |
| Shared cross-domain source | One TUAB model, training seed 241; never use official test subjects |
| Tracking | New W&B project `quoc-emotiv/eegx`; grouping and seed mean remain native W&B features |

The architecture, loss, preprocessing, optimizer and classifier recipe remain the reference choices. Checkpoint selection uses validation only. These are repeated training seeds, not repeated data splits.

| Dataset | Fixed partition rule |
|---|---|
| TUAB | Preserve official eval as test; group official training subjects into fixed 80/20 train/validation using split seed 241 |
| TUEV binary | Preserve official eval as test; group official training subjects into fixed 80/20 train/validation using split seed 241 |
| STEW | Reuse saved split 241, selected by completed fine-tune test balanced accuracy from `eegx-bs256`, run `2wpra2j5` |
| DREAMER arousal | Reuse saved split 242, selected by completed fine-tune test balanced accuracy from `eegx-bs256-debug`, run `o74fswps` |
| Crowdsourced, BCICIV-2a, Kalunga, PhysioNetMI | Reuse saved split 241; both reference projects lack completed target fine-tuning scores |

STEW and DREAMER splits are test-informed exploratory choices. Their new results do not constitute an independent held-out validation of the protocol. Keep their selection evidence with the release.

| Tracked stage | Runs |
|---|---:|
| In-domain pretraining | 12 |
| In-domain fine-tuning | 12 |
| Shared TUAB source pretraining | 1 |
| Cross-domain linear probes | 12 |
| Cross-domain fine-tuning | 12 |
| Total | 49 |

In-domain probe fits stay inside pretraining runs. Cross-domain probes and fine-tuning start independently from the same selected TUAB checkpoint.

Preserve the old HDF5 release. Publish independent, self-contained files in a new version directory. Retain original sample counts, signal preprocessing, labels, stable window IDs, raw provenance and duplicate multiplicity. Copy only the selected bank into each new file; add a separate train/validation-only TUAB source bank. Official test subjects must be absent from that source bank.

Each release includes a portable manifest, full file hashes, parent release identities, subject assignments, indices, selection evidence and preprocessing metadata. Each experiment saves resolved YAML, split identity, training seed, code/environment hashes, checkpoints, predictions and metrics. A copied release must load without the old folder or raw files.

Implementation order: update code and data; run meaningful CPU and GPU checks; review Standards and Spec; fix findings; commit; freeze the committed source and generated queue; validate dependencies; launch the real experiments. Preserve unrelated GPU jobs and all historical ledgers. Fresh per-device meters have no time cap. Admission uses measured peak memory and headroom.
