# GitHub upload package

This folder is the code-only repository package for the state-capacity TinyRNN study.

## Suggested public repository

- GitHub user: `pwa209`
- Suggested repository name: `state-capacity-tinyrnn`
- Suggested visibility: public
- Suggested license: MIT

## What is included

- `scripts/`: staged analysis scripts, including the NHB revision and package builders.
- `src/`: reusable project modules.
- `config/` and `configs/`: analysis configuration files.
- `docs/`: step protocol and manuscript/protocol support documents.
- `tests/`: reproducibility and guard tests.
- `environment.yml`, `pyproject.toml`, `Makefile`: environment and execution entry points.

## What is intentionally excluded

- `data/raw/`: raw public datasets must be downloaded from the original providers.
- Large generated outputs, model checkpoints and release ZIPs.
- Local virtual environments and package caches.

## Local upload commands

Create an empty public GitHub repository named `state-capacity-tinyrnn`, then run the following commands from this folder:

```bash
git init
git add .
git commit -m "Release state-capacity TinyRNN reproducibility code"
git branch -M main
git remote add origin https://github.com/pwa209/state-capacity-tinyrnn.git
git push -u origin main
git tag v1.0.0
git push origin v1.0.0
```

After pushing, create a GitHub release from tag `v1.0.0`. If Zenodo is connected to the repository, Zenodo will issue a DOI for that software release.
