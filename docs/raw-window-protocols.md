# Raw EEG window timing

A native trial, a benchmark epoch and a model window are different intervals. Public sources support the following timing rules.

| Dataset | Source-backed timing | Model-window decision |
|---|---|---|
| BCICIV-2a / BNCI | Trial starts at **0 s**, cue appears at **2 s**, imagery ends at **6 s**. The BNCI MAT trial marker is the trial start; native GDF class markers are cue onsets. [Official protocol](https://bbci.de/competition/iv/desc_2a.pdf), [MAT marker source](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/bnci/base.py#L410-L451). | The verified original saved catalog uses **[2, 4) s from trial start**, equivalently **[0, 2) s from cue onset**, giving **2 s** inputs. The full benchmark imagery interval remains longer. [Benchmark interval](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/bnci/bnci_2014.py#L477-L501). |
| SSVEP / Kalunga | MOABB uses **[2, 4] s relative to the class marker**. The author report supports excluding the first **2 s** after the cue because of response delay. [Benchmark interval](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/ssvep_exo.py#L155-L175), [author report](https://arxiv.org/pdf/1501.03227). | Use the benchmark crop **[2, 4) s**, giving **2 s**. The author report describes **6 s** trials, whereas some dataset prose says **5 s**; neither full-trial duration changes this benchmark crop. |
| PhysioNetMI | Official annotations identify rest and task states but prescribe no unique analysis crop. MOABB uses **[0, 3] s from annotation onset**. [Official description](https://archive.physionet.org/pn4/eegmmidb/), [benchmark interval](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/physionet_mi.py#L271-L359). | The agreed **2 s** model width is a study choice. Selected **[0, 2) s** preserves the benchmark's onset. This is the user-approved adaptation, **not an official two-second prescription**. |

The crop origin must match the event marker. Trial-start and cue-start offsets must not be added together.

| PhysioNetMI scope or caveat | Established meaning |
|---|---|
| Imagery runs **4, 8, 12** | **T1:** left fist; **T2:** right fist; **T0:** rest. |
| Imagery runs **6, 10, 14** | **T1:** both fists; **T2:** both feet; **T0:** rest. |
| Baseline runs **1, 2** | Separate eyes-open/closed recordings; the benchmark imagery loader does not use them as task-rest trials. |
| Confirmed study scope | **109 subjects; 5 classes**, including rest. [Official description](https://archive.physionet.org/pn4/eegmmidb/), [event mapping](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/physionet_mi.py#L287-L359). |
| Observed annotation examples | Subjects **S001** (runs **4, 6**) and **S029** (run **4**) contain labelled intervals of **4.1 / 4.2 s**. These are annotation durations, not the benchmark or model crop. [Example recording](https://physionet.org/content/eegmmidb/1.0.0/S001/#files-panel). |
| Acquisition-rate caveat | Current MOABB source identifies subjects **88, 92, 100** at **128 Hz**, with the others normally at **160 Hz**. Preserve the agreed population and use the actual acquisition rate before conversion to **128 Hz**. [Maintainer correction](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/physionet_mi.py#L20-L62). |
| Fixed model lengths | At **128 Hz**, the agreed **2 s / 4 s** windows contain exactly **256 / 512 samples**. Intervals above use half-open bounds; retain complete windows only. |

The confirmed reference revision preserves original overlap and event multiplicity. PhysioNet omits initial events and retains later annotations with the verified legacy eligibility, including short rest annotations. Its model windows use the agreed true duration; only Crowdsourced requires each window to remain within one label interval.
