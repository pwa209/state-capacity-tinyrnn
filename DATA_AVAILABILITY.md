# Data Availability

This repository contains code only. Raw public datasets are intentionally not committed because they are large and should be obtained from the original providers.

## Raw Data Sources

| Dataset | Provider | Identifier / version | Use in this project |
|---|---|---|---|
| COG-BCI | Zenodo | Record 6874129, https://zenodo.org/records/6874129 | Repeated-session behavioural/EEG validation source |
| TU Berlin simultaneous EEG-NIRS N-back | TU Berlin | https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ | N-back load-pressure, EEG and fNIRS validation source |
| CMx7-MM multimodal dataset | OpenNeuro | ds007554 v1.0.0, DOI 10.18112/openneuro.ds007554.v1.0.0 | Multimodal behaviour, EEG, fNIRS and ECG source |
| Healthy Brain Network EEG Release 4 | OpenNeuro | ds005508 v1.0.1, DOI 10.18112/openneuro.ds005508.v1.0.1 | Large-scale EEG behavioural scalability source |

## Derived Data Package

The derived-data/source-data package should be deposited separately on Zenodo or OSF. It should include:

- Harmonised processed event tables.
- State and capacity coordinate tables.
- Artificial-agent tables and intervention-gate outputs.
- Recurrent-dynamics and projection tables.
- Neurophysiology feature tables.
- Statistical result tables.
- Figure source-data files and generated figures.
- Exclusion, reconstruction, manifest and claim-audit records.

The current local derived-data ZIP prepared for Zenodo is:

`state_capacity_tinyrnn_zenodo_derived_data_package.zip`

After upload, insert the Zenodo/OSF DOI into the manuscript Data availability section.
