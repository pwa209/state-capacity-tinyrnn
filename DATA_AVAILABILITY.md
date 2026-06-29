# Data Availability

This repository contains code only. Raw public datasets are intentionally not committed because they are large, externally maintained, and should be obtained from the original providers under their own terms of use.

## Raw Data Sources

Raw data must be downloaded from the original repositories/accessions below. The Zenodo derived-data archive for this study does not redistribute raw EEG, fNIRS, ECG or behavioural files.

| Dataset | Original repository | Accession / version | Raw-data DOI or URL | Use in this project |
|---|---|---|---|---|
| COG-BCI EEG/ECG workload dataset | Zenodo | Record 6874129 | https://zenodo.org/records/6874129; DOI 10.5281/zenodo.6874129 | Repeated-session behavioural/EEG validation source |
| TU Berlin simultaneous EEG-NIRS N-back | TU Berlin Machine Learning Group dataset repository | Official EEG/NIRS/behaviour archives | https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ | N-back load-pressure, EEG and fNIRS validation source |
| CMx7-MM multimodal dataset | OpenNeuro | ds007554 v1.0.0 | https://openneuro.org/datasets/ds007554/versions/1.0.0; DOI 10.18112/openneuro.ds007554.v1.0.0 | Multimodal behaviour, EEG, fNIRS and ECG source |
| Healthy Brain Network EEG Release 4 | OpenNeuro | ds005508 v1.0.1 | https://openneuro.org/datasets/ds005508/versions/1.0.1; DOI 10.18112/openneuro.ds005508.v1.0.1 | Large-scale EEG behavioural scalability source |

## Derived Data Package

The derived-data/source-data package is deposited on Zenodo:

- Record: https://zenodo.org/records/21028379
- DOI: 10.5281/zenodo.21028379

The archive contains the de-identified and reproducibility-ready outputs needed to audit and redraw the study results, including:

- Harmonised processed event tables.
- State and capacity coordinate tables.
- Artificial-agent tables and intervention-gate outputs.
- Recurrent-dynamics and projection tables.
- Neurophysiology feature tables.
- Statistical result tables.
- Figure source-data files and generated figures.
- Exclusion, reconstruction, manifest and claim-audit records.

Raw public datasets are not redistributed in this archive and should be obtained from the original sources listed above.
