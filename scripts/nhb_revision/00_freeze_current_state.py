from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
NHB = OUTPUTS / "nhb_revision"
NHB_TABLES = NHB / "tables"
NHB_FIGURES = NHB / "figures"
NHB_AUDIT = NHB / "audit"
NHB_LOGS = NHB / "logs"
NHB_MODELS = NHB / "models"
NHB_MANUSCRIPT = NHB / "manuscript_source"
SOURCE_DIRS = {
    "table": OUTPUTS / "tables",
    "audit": OUTPUTS / "audit",
    "figure": OUTPUTS / "figures",
    "source_data": OUTPUTS / "source_data",
}
TAXONOMY = ROOT / "config" / "nhb_feature_taxonomy.yaml"


CLAIM_COLUMNS = [
    "analysis_id",
    "claim_id",
    "claim_text",
    "construct",
    "source_tables",
    "figure_panel",
    "supporting_effects",
    "negative_effects",
    "controls_passed",
    "controls_failed",
    "claim_strength",
    "allowed_manuscript_language",
    "forbidden_language",
]
REGISTRY_COLUMNS = [
    "analysis_id",
    "script_name",
    "started_at_utc",
    "completed_at_utc",
    "status",
    "outputs",
    "notes",
]
EXCLUSION_COLUMNS = [
    "analysis_id",
    "script_name",
    "dataset",
    "task",
    "model_family",
    "reason",
    "severity",
]
FIGURE_MAP_COLUMNS = [
    "figure_id",
    "panel",
    "source_table",
    "source_script",
    "output_path",
    "legend_path",
]


def ensure_dirs() -> None:
    for path in [NHB_TABLES, NHB_FIGURES, NHB_AUDIT, NHB_LOGS, NHB_MODELS, NHB_MANUSCRIPT]:
        path.mkdir(parents=True, exist_ok=True)
    (NHB_FIGURES / "extended_data").mkdir(parents=True, exist_ok=True)
    (NHB_TABLES / "extended_data_source").mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unavailable"


def package_versions() -> str:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "torch",
        "mne",
        "mne-nirs",
        "neurokit2",
        "pypdf",
        "python-docx",
    ]
    rows = []
    for name in names:
        try:
            rows.append(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            rows.append(f"{name}=missing")
    return ";".join(rows)


def write_header_if_missing(path: Path, columns: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow(columns)


def append_tsv(path: Path, columns: list[str], row: dict[str, object]) -> None:
    write_header_if_missing(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)


def collect_files() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, folder in SOURCE_DIRS.items():
        if not folder.exists():
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = path.relative_to(ROOT).as_posix()
            rows.append(
                {
                    "path": rel,
                    "file_size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "category": category,
                }
            )
    if TAXONOMY.exists():
        rows.append(
            {
                "path": TAXONOMY.relative_to(ROOT).as_posix(),
                "file_size_bytes": TAXONOMY.stat().st_size,
                "sha256": sha256(TAXONOMY),
                "modified_time": datetime.fromtimestamp(TAXONOMY.stat().st_mtime).isoformat(),
                "category": "frozen_feature_taxonomy",
            }
        )
    return rows


def main() -> None:
    ensure_dirs()
    started = datetime.now(timezone.utc).isoformat()
    rows = collect_files()
    manifest = NHB_AUDIT / "current_state_hash_manifest.tsv"
    columns = ["path", "file_size_bytes", "sha256", "modified_time", "category"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    run_manifest = NHB_AUDIT / "nhb_revision_manifest.tsv"
    manifest_cols = [
        "created_at_utc",
        "analysis_id",
        "script_name",
        "output_path",
        "output_type",
        "rows",
        "python_version",
        "platform",
        "package_versions",
        "git_commit",
    ]
    with run_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_cols, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_id": "nhb_00_freeze_current_state",
                "script_name": "scripts/nhb_revision/00_freeze_current_state.py",
                "output_path": manifest.relative_to(ROOT).as_posix(),
                "output_type": "hash_manifest",
                "rows": len(rows),
                "python_version": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "package_versions": package_versions(),
                "git_commit": git_commit(),
            }
        )

    write_header_if_missing(NHB_AUDIT / "nhb_claim_audit.tsv", CLAIM_COLUMNS)
    write_header_if_missing(NHB_AUDIT / "nhb_exclusion_log.tsv", EXCLUSION_COLUMNS)
    write_header_if_missing(NHB_AUDIT / "nhb_figure_source_map.tsv", FIGURE_MAP_COLUMNS)
    append_tsv(
        NHB_AUDIT / "nhb_analysis_registry.tsv",
        REGISTRY_COLUMNS,
        {
            "analysis_id": "nhb_00_freeze_current_state",
            "script_name": "scripts/nhb_revision/00_freeze_current_state.py",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "outputs": manifest.relative_to(ROOT).as_posix(),
            "notes": f"Frozen {len(rows)} current-output files and taxonomy without modifying legacy outputs.",
        },
    )
    print(f"Wrote {manifest} with {len(rows)} rows")


if __name__ == "__main__":
    main()
