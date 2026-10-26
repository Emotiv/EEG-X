# PhysioNet channel coordinates and subject mapping

Electrode XYZ coordinates are available. The saved signal array still lacks verified channel-order metadata and participant IDs. XYZ identifies electrodes; it does not identify participants.

| Evidence | Result | Status |
| --- | ---: | --- |
| Official EEG electrode count | 64 | Documented by [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) |
| Canonical electrodes resolved in the frozen vocabulary | 64 / 64 | Complete |
| Local EDF recordings checked | 169 | Identical electrode order in every header |
| Participants represented in the local EDF cache | 29 | Raw recordings retain participant identities |
| Maximum coordinate difference from the MNE template, metres | 0.000000003714 | Float precision only |
| Saved processed windows | 19,022 | Signal and label arrays only |
| Processed channel order verified | 0 | Upstream export metadata missing |
| Processed window-to-participant mappings recovered | 0 | No participant fields or matching index found |

The canonical order starts with FC5, FC3 and FC1 and ends with Oz, O2 and Iz. Local EDF headers agree with the [MOABB dataset source](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/physionet_mi.py). This agreement establishes the raw order; it does not prove the order of the saved processed array.

The exported coordinates use the existing frozen EEG-X vocabulary. They match the [MNE electrode template](https://github.com/mne-tools/mne-python/blob/maint/1.10/mne/channels/data/montages/standard_1005.elc) after converting millimetres to metres. These are standard template locations, not measured positions for each participant. The [historical MOABB loader](https://github.com/NeuroTechX/moabb/blob/v1.1.1/moabb/datasets/physionet_mi.py) assigns this template to the dataset.

The [original conversion notebook](https://github.com/Emotiv/EEG-X/blob/a5e771b/datasets/data_visualization/MAOBB.ipynb) copies signal arrays from an upstream processed export and converts label names to integers. It saves no participant IDs or channel names. That upstream export is absent from the inspected storage. The surviving raw cache does not, by itself, map processed windows back to participants.

The coordinate table is ready as a candidate mapping. Use it for the existing processed release only after its channel order is verified. Subject-wise evaluation also requires the original window-to-participant mapping or an exactly matched reconstruction of the original export. Do not infer participant IDs from electrode coordinates or window counts.
