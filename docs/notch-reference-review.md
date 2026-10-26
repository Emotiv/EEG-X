# Notch filtering and EEG reference

Use the agreed QuantPlus recipe as the reproduction baseline. Extra notch filtering and average referencing require separate justification.

| Evidence | Finding | Meaning for the agreed baseline |
|---|---|---|
| Audited QuantPlus recipe | **128 Hz**, band-pass **0.5–45 Hz**, per-recording StandardScaler, clipping at **±15**; no extra notch or average-reference operation. | Preserve the native reference. Recipe fidelity does not prove that line noise is absent. |
| [MNE filter definition](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter) | **45 Hz** is the upper pass-band edge. Automatic transition width is **11.25 Hz**, reaching the stopband at **56.25 Hz** for the acquisition rates checked. | A residual **50 Hz** component is possible; the low-pass setting alone does not establish that a notch is redundant. |
| [MNE reference guidance](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html) | EEG is already referenced. Average referencing changes the signal; MNE's strict requirement concerns inverse source modelling. | There is no universal requirement for our classifier. |
| [Reference extractor](https://github.com/facebookresearch/neuroai/blob/main/neuralset-repo/neuralset/extractors/neuro.py), [TUH source loader](https://github.com/facebookresearch/neuroai/blob/main/neuralfetch-repo/neuralfetch/studies/tuh_eeg.py) | Checked source loaders add no average subtraction; the extractor supports optional bipolar reference. | Preserve references already encoded in source signals. |

An extra notch deliberately changes the recipe. Justify its frequency from signal evidence before comparing models.

| Measured filter response | Result |
|---|---:|
| Design | MNE **1.13.2**, zero-phase FIR, automatic transitions, **0.5–45 Hz** |
| Acquisition rates checked | **128, 160, 250, 256, 500 Hz** |
| Gain at **50 Hz** | **-4.50 to -4.44 dB** |
| Gain at **60 Hz** | **-61.2 to -54.6 dB** |
| Bounded raw-signal screen | **12 recordings**, **2 per dataset**, across **6 datasets**; **60 s** segments, before standardization/clipping; no added notch or average reference |
| Median-channel band-energy ratio after filtering/resampling: **49.5–50.5 Hz** relative to **0.5–45 Hz** | **0.0001% to 0.4227%** across screened recordings |
| Same ratio for **59.5–60.5 Hz** | At most **0.00000159%** across screened recordings |
| Not screened | DREAMER and STEW; screened PhysioNet membership was provisional training membership |

Filter gains describe the operator; the raw screen is a separate, limited check and does not establish corpus-wide cleanliness. Band energy includes any signal in that band and is not a pure mains-noise estimate.

| Acquisition evidence | Consequence |
|---|---|
| BCICIV-2a used a left-mastoid reference and an acquisition **50 Hz** notch. [Original protocol](https://bbci.de/competition/iv/desc_2a.pdf). | Do not assume every dataset requires an added **60 Hz** notch or the same new reference. |
| Kalunga recorded **8 posterior channels** with a right-mastoid reference. [Author report](https://arxiv.org/pdf/1501.03227). | Averaging only this posterior subset creates a different reference. Whether that helps decoding is untested here. |
| TUH's native averaged-reference and linked-ear recordings have different signal statistics. [Original TUH study](https://isip.piconepress.com/conferences/ieee_spmb/2016/papers/l01_05.pdf). | Preserve and record the source reference; do not silently replace all recordings with a new average. |

Recommendation: retain the agreed no-extra-notch and native-reference baseline. The measurable residual near mains warrants further release QC, but this limited screen does not prove that a blanket notch is necessary or that average referencing improves decoding.
