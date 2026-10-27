# Dataset preparation plan

Native preparation applies EEG-X2 signal preprocessing to verified original sample catalogs. Overlapping windows and repeated annotations remain separate samples. All original sample catalogs are now proven; explicit timing and label corrections remain recorded.

| Dataset | Window | Step or crop rule | Reference status |
|---|---:|---|---|
| DREAMER arousal | 2 s | Step 64 samples; separate videos; rating >3 is high | Original count and binary class totals established |
| STEW | 2 s | Step 32 samples; both native recording conditions | Original count and class totals established |
| TUAB | 10 s | Step 1,280 samples over complete EDF recordings | Original catalog established; retain duplicate copies |
| TUEV binary | 2 s | One crop per eligible REC row; onset −0.5 to +1.5 s, or onset to +2 s near recording start; retain repeated/overlapping crops | Original binary crop multiplicities established |
| Crowdsourced | 2 s | Step 32 samples; correct native rate; continuous acquisition and label intervals | Explicit timing/boundary exception |
| BCICIV-2a | 2 s | Trial +2 to +4 s; native one-based trial origin | Complete original waveform and label replay |
| Kalunga / SSVEP | 2 s | Marker +2 to +4 s; pinned 30-run selection | Complete original waveform replay; corrected frequency labels |
| PhysioNetMI | 2 s | Omit initial events; retain later annotations with legacy 320-native-sample eligibility | Complete original waveform/label replay; true duration correction at 128 Hz |

All step sizes refer to the prepared sampling rate. The [confirmed requirements](task.md#latest-raw-data-preparation-decision) record reference counts and permitted exceptions.

| Stage | Rule |
|---|---|
| Reference | Preserve native reference; no additional average reference |
| Filter | Native-rate high-pass 0.5 Hz and low-pass 45 Hz; no additional notch |
| Resample | 128 Hz, using the actual native rate |
| Standardize | Per-channel StandardScaler over the recording before extracting windows |
| Save | Clip ±15 and store float32 windows as X[N,C,T] |
| Repeated data | Retain reference multiplicity; connect subjects sharing whole-record or window signal hashes |
| Target splits | 5 banks, seeds 241–245; target 70/10/20 subject counts with linked groups kept intact |
| TUAB source split | Separate grouped 80/20 train/validation bank, without test rows |
| Output | 1 HDF5 per verified dataset, with signals, labels, records, subjects, channels/XYZ, source hashes, stable occurrence IDs and saved indices |

Recording boundaries are videos, EDF runs or uninterrupted acquisition blocks. Discard only incomplete crops or explicitly documented reference exceptions. Do not add windows to force a reference count, ICA/rASR or model-side signal normalization.

Established original-array field/row mappings, source aliases and reference hashes are embedded in each mapped HDF5 file. Native crop provenance and legacy eligibility remain distinct from the corrected model-window duration.
