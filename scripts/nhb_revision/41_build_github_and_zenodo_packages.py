from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "release_packages"
GITHUB_DIR = OUT / "github_code_package"
ZENODO_DIR = OUT / "zenodo_derived_data_package"


CODE_DIRS = [
    "config",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests",
]

CODE_FILES = [
    "README.md",
    "environment.yml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
]

DATA_DIRS = [
    "data/processed",
    "outputs/audit",
    "outputs/figures",
    "outputs/manifests",
    "outputs/manuscript_text",
    "outputs/nhb_revision",
    "outputs/source_data",
    "outputs/tables",
]

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".git",
    ".venv",
    "raw",
    "archive_previous_run",
    "model_checkpoints",
    "manuscript_placeholder_audit",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".part",
    ".log",
}

EXCLUDE_RELATIVE_PATHS = {
    "outputs/nhb_revision/placeholder_sensitivity/input_placeholder_audit_report.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    if path.as_posix() in EXCLUDE_RELATIVE_PATHS:
        return True
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_relative(relative: str, destination_root: Path) -> None:
    src = ROOT / relative
    dst = destination_root / relative
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for file in src.rglob("*"):
        if not file.is_file() or should_skip(file.relative_to(ROOT)):
            continue
        target = destination_root / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_manifest(package_dir: Path, manifest_name: str = "MANIFEST_SHA256.csv") -> list[dict[str, object]]:
    rows = []
    for file in sorted(package_dir.rglob("*")):
        if not file.is_file() or file.name == manifest_name:
            continue
        rel = file.relative_to(package_dir).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "size_bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
        )
    manifest = package_dir / manifest_name
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def zip_dir(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(package_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(package_dir).as_posix())


def write_github_metadata(package_dir: Path) -> None:
    write_text(
        package_dir / ".gitignore",
        """
        # Local environments and caches
        .venv/
        __pycache__/
        *.py[cod]
        .pytest_cache/
        .mypy_cache/
        .ruff_cache/
        node_modules/

        # Raw and generated large data should not be committed
        data/raw/
        data/interim/
        outputs/archive_previous_run/
        outputs/model_checkpoints/
        outputs/release_packages/
        *.part
        *.tmp
        """,
    )
    write_text(
        package_dir / "LICENSE",
        """
        MIT License

        Copyright (c) 2026 State-capacity TinyRNN contributors

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """,
    )
    write_text(
        package_dir / "README_GITHUB_UPLOAD.md",
        """
        # GitHub upload package

        This folder is the code-only repository package for the state-capacity TinyRNN study.

        ## What is included

        - `scripts/`: staged analysis scripts, including the NHB revision and package builders.
        - `src/`: reusable project modules.
        - `config` and `configs`: analysis configuration files.
        - `docs/`: step protocol and manuscript/protocol support documents.
        - `tests/`: reproducibility and guard tests.
        - `environment.yml`, `pyproject.toml`, `Makefile`: environment and execution entry points.

        ## What is intentionally excluded

        - `data/raw/`: raw public datasets must be downloaded from the original providers.
        - Large generated outputs, model checkpoints and release ZIPs.
        - Local virtual environments and package caches.

        ## Suggested GitHub commands

        ```powershell
        cd <this-folder>
        git init
        git add .
        git commit -m "Release state-capacity TinyRNN reproducibility code"
        git branch -M main
        git remote add origin https://github.com/pwa209/state-capacity-tinyrnn.git
        git push -u origin main
        git tag v1.0.0
        git push origin v1.0.0
        ```

        After pushing, create a GitHub release from tag `v1.0.0`. If Zenodo is connected to
        the GitHub repository, Zenodo will issue a DOI for that software release.
        """,
    )
    write_text(
        package_dir / "CITATION.cff",
        """
        cff-version: 1.2.0
        title: "Machine-defined state and capacity profiles in human cognition"
        message: "If you use this code, please cite the manuscript and archived software release."
        type: software
        authors:
          - name: "State-capacity TinyRNN contributors"
        version: "1.0.0"
        date-released: "2026-06-29"
        license: "MIT"
        """,
    )


def write_zenodo_metadata(package_dir: Path) -> None:
    write_text(
        package_dir / "README_ZENODO_DERIVED_DATA.md",
        """
        # Derived data and source-data package

        This package contains derived, de-identified outputs for the state-capacity TinyRNN study.
        It is deposited on Zenodo at https://zenodo.org/records/21028379, DOI 10.5281/zenodo.21028379.

        ## Included

        - `data/processed/`: harmonised event-level tables derived from public raw datasets.
        - `outputs/tables/`: row-level model, statistical and validation result tables.
        - `outputs/source_data/`: source data used for figure generation.
        - `outputs/figures/`: generated figures.
        - `outputs/nhb_revision/`: manuscript-revision outputs, figure data, audit tables and placeholder sensitivity analyses.
        - `outputs/audit/` and `outputs/manifests/`: inclusion/exclusion, download and file provenance records.
        - `MANIFEST_SHA256.csv`: checksums and sizes for every file in this package.

        ## Excluded

        Raw public datasets are not redistributed here. They should be obtained from the original providers:

        - COG-BCI: Zenodo record 6874129, https://zenodo.org/records/6874129
        - TU Berlin simultaneous EEG-NIRS N-back dataset: https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/
        - CMx7-MM multimodal dataset: OpenNeuro ds007554, version 1.0.0, DOI 10.18112/openneuro.ds007554.v1.0.0
        - Healthy Brain Network EEG Release 4: OpenNeuro ds005508, version 1.0.1, DOI 10.18112/openneuro.ds005508.v1.0.1

        ## Zenodo metadata

        Title: Derived data and source tables for machine-defined state and capacity profiles in human cognition

        Description: Derived behavioural, recurrent-model, neurophysiological, statistical, figure-source and claim-audit outputs from a reproducible TinyRNN state-capacity analysis pipeline. Raw datasets are not redistributed and should be downloaded from the original public providers listed above.

        Resource type: Dataset

        Keywords: recurrent neural network; cognitive state; cognitive capacity; EEG; fNIRS; OpenNeuro; source data; reproducibility
        """,
    )
    write_text(
        package_dir / "DATASET_PROVENANCE.md",
        """
        # Dataset provenance

        This package contains derived outputs only. The raw data sources used by the analysis are:

        | Dataset | Source | Version / identifier | Notes |
        |---|---|---|---|
        | COG-BCI | Zenodo | record 6874129 | Repeated-session EEG/behaviour validation source. |
        | TU Berlin simultaneous EEG-NIRS | Official TU Berlin dataset site | https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ | N-back workload and EEG/fNIRS validation source. |
        | CMx7-MM | OpenNeuro | ds007554 v1.0.0, DOI 10.18112/openneuro.ds007554.v1.0.0 | Multimodal EEG/fNIRS/ECG/behaviour source. |
        | HBN EEG Release 4 | OpenNeuro | ds005508 v1.0.1, DOI 10.18112/openneuro.ds005508.v1.0.1 | Large-scale EEG behavioural scalability source. |
        """,
    )


def package_code() -> dict[str, object]:
    reset_dir(GITHUB_DIR)
    for rel in CODE_DIRS:
        copy_tree_relative(rel, GITHUB_DIR)
    for rel in CODE_FILES:
        copy_tree_relative(rel, GITHUB_DIR)
    write_github_metadata(GITHUB_DIR)
    build_manifest(GITHUB_DIR)
    zip_path = OUT / "state_capacity_tinyrnn_github_code_package.zip"
    zip_dir(GITHUB_DIR, zip_path)
    n_files = sum(1 for file in GITHUB_DIR.rglob("*") if file.is_file())
    return {
        "folder": str(GITHUB_DIR),
        "zip": str(zip_path),
        "n_files": n_files,
        "zip_size_bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
    }


def package_zenodo_data() -> dict[str, object]:
    reset_dir(ZENODO_DIR)
    for rel in DATA_DIRS:
        copy_tree_relative(rel, ZENODO_DIR)
    write_zenodo_metadata(ZENODO_DIR)
    build_manifest(ZENODO_DIR)
    zip_path = OUT / "state_capacity_tinyrnn_zenodo_derived_data_package.zip"
    zip_dir(ZENODO_DIR, zip_path)
    n_files = sum(1 for file in ZENODO_DIR.rglob("*") if file.is_file())
    return {
        "folder": str(ZENODO_DIR),
        "zip": str(zip_path),
        "n_files": n_files,
        "zip_size_bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    code = package_code()
    data = package_zenodo_data()
    summary = {
        "created_utc": started,
        "github_code_package": code,
        "zenodo_derived_data_package": data,
        "raw_data_excluded": True,
        "raw_data_exclusion_reason": "Raw public datasets are large and should be cited/downloaded from original providers rather than redistributed.",
        "next_manual_steps": [
            "Confirm the GitHub repository is current: https://github.com/pwa209/state-capacity-tinyrnn.",
            "Create a GitHub release, suggested tag v1.0.0.",
            "Archive the GitHub release with Zenodo to obtain a software DOI.",
            "Keep the Zenodo derived-data record current: https://zenodo.org/records/21028379, DOI 10.5281/zenodo.21028379.",
            "Replace manuscript Code availability prose with the final software DOI, release tag and commit hash.",
        ],
    }
    summary_path = OUT / "release_package_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
