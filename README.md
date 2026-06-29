# State-capacity TinyRNN

This repository contains the reproducibility code for the study **Machine-defined state and capacity profiles in human cognition**. The project implements a staged TinyRNN analysis pipeline for separating state-like operating-regime profiles from capacity-like recurrent-resource profiles in public behavioural and neurophysiological datasets.

## What This Repository Contains

- Artificial-agent perturbation and intervention-gate analyses.
- Human recurrent-model fitting and state/capacity coordinate estimation.
- Recurrent-dynamics, projection, robustness and external-validation analyses.
- EEG/fNIRS/ECG feature-extraction and exploratory neurophysiology analyses.
- Statistical tables, figure-generation scripts and manuscript-support utilities.
- Release packaging scripts for GitHub code and Zenodo derived-data deposits.

## What Is Not Stored Here

Raw public datasets are intentionally not committed. They are large, externally maintained and should be downloaded from their original repositories/accessions listed in `DATA_AVAILABILITY.md`: COG-BCI Zenodo record 6874129, TU Berlin simultaneous EEG-NIRS, OpenNeuro ds007554 v1.0.0 and OpenNeuro ds005508 v1.0.1.

Derived tables and source-data files are deposited on Zenodo at https://zenodo.org/records/21028379, DOI 10.5281/zenodo.21028379.

## Repository Layout

- `scripts/`: staged analysis scripts.
- `src/`: reusable project modules.
- `config/`, `configs/`: feature taxonomy and run configuration.
- `docs/`: protocol and step-pack documentation.
- `tests/`: reproducibility guard tests.
- `environment.yml`, `pyproject.toml`: environment definitions.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate state-capacity-tinyrnn
```

Run selected workflow targets:

```bash
make audit_existing
```

Some full analyses require raw public datasets to be downloaded first and can take substantial time and disk space. Use the staged scripts in `scripts/` and the protocol notes in `docs/step_pack.md` to reproduce each stage.

## Release and Citation

This repository is prepared for GitHub release `v1.0.0` and Zenodo software archiving. Before citing the software, create a GitHub release and archive it through Zenodo so that the final software DOI, release tag and commit hash can be inserted into the manuscript.

See also:

- `DATA_AVAILABILITY.md`
- `RELEASE_CHECKLIST.md`
- `CITATION.cff`
