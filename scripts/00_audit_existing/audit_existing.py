from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import add_common_args, assert_full_run_allowed


PREVIOUS_STEP_PREFIXES = tuple(f"{i:02d}_" for i in range(14))
INVENTORY_SUFFIXES = {
    ".py": "script",
    ".md": "markdown",
    ".csv": "table",
    ".tsv": "table",
    ".json": "metadata",
    ".yaml": "config",
    ".yml": "config",
    ".parquet": "processed_data",
    ".png": "figure",
    ".pdf": "figure",
    ".svg": "figure",
    ".pt": "model_checkpoint",
    ".pth": "model_checkpoint",
    ".ckpt": "model_checkpoint",
}


def previous_root() -> Path:
    return PROJECT_ROOT.parent


def output_type(path: Path) -> str:
    lower_name = path.name.lower()
    if "checkpoint" in lower_name or path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "model_checkpoint"
    if path.suffix.lower() in INVENTORY_SUFFIXES:
        return INVENTORY_SUFFIXES[path.suffix.lower()]
    return "other"


def run_classification(path: Path) -> str:
    lowered = str(path).lower()
    if "pilot" in lowered:
        return "pilot_or_debug_output"
    if path.suffix.lower() == ".py":
        text = path.read_text(encoding="utf-8", errors="ignore")
        forbidden_markers = ["--sample", "--max-subjects", "nrows", ".head(", "break  #"]
        if any(marker in text for marker in forbidden_markers):
            return "script_needs_full_run_review"
        return "script_inventory_only"
    return "previous_full_candidate_not_accepted"


def locate_previous_step_dirs() -> list[Path]:
    root = previous_root()
    dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(PREVIOUS_STEP_PREFIXES) and path.name != PROJECT_ROOT.name
    ]
    return sorted(dirs, key=lambda p: p.name)


def inventory_previous_project(step_dirs: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step_dir in step_dirs:
        for path in step_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(previous_root())
            rows.append(
                {
                    "previous_step": step_dir.name,
                    "relative_path": str(rel).replace("\\", "/"),
                    "file_name": path.name,
                    "suffix": path.suffix.lower(),
                    "output_type": output_type(path),
                    "size_bytes": path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "run_classification": run_classification(path),
                    "accepted_as_final": False,
                    "revised_pipeline_status": "must_regenerate",
                }
            )
    return rows


def archive_previous_outputs(step_dirs: list[Path]) -> list[dict[str, str]]:
    archive_root = PROJECT_ROOT / "outputs" / "archive_previous_run"
    archive_rows: list[dict[str, str]] = []
    archive_root.mkdir(parents=True, exist_ok=True)
    for step_dir in step_dirs:
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        destination = archive_root / step_dir.name / "outputs"
        shutil.copytree(outputs_dir, destination, dirs_exist_ok=True)
        archive_rows.append(
            {
                "previous_step": step_dir.name,
                "source": str(outputs_dir),
                "archive_destination": str(destination),
            }
        )
    return archive_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_reproducibility_status(rows: list[dict[str, object]], archived: list[dict[str, str]]) -> None:
    audit_dir = PROJECT_ROOT / "outputs" / "audit"
    type_counts = summarize_counts(rows, "output_type")
    class_counts = summarize_counts(rows, "run_classification")
    lines = [
        "# Previous Results Reproducibility Status",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Decision",
        "",
        "No previous empirical result is accepted as final in the revised pipeline.",
        "Every claim must be regenerated by scripts under `state_capacity_tinyrnn/` and recorded in `outputs/audit/claim_audit.tsv`.",
        "",
        "## Previous Inventory Counts by Type",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in type_counts.items())
    lines.extend(["", "## Previous Run Classification", ""])
    lines.extend(f"- {name}: {count}" for name, count in class_counts.items())
    lines.extend(["", "## Archived Previous Outputs", ""])
    if archived:
        lines.extend(
            f"- `{row['previous_step']}`: `{row['archive_destination']}`" for row in archived
        )
    else:
        lines.append("- No previous `outputs/` folders found.")
    lines.extend(
        [
            "",
            "## Rationale",
            "",
            "The earlier implementation remains valuable as a map and sanity check, but the revised protocol adds stricter dataset eligibility, machine perturbation gates, fingerprint convergence, external validation, source-data rules and claim auditing.",
        ]
    )
    (audit_dir / "previous_results_reproducibility_status.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_migration_plan(step_dirs: list[Path]) -> None:
    audit_dir = PROJECT_ROOT / "outputs" / "audit"
    mappings = [
        ("00_project_setup", "setup, configs, Makefile, tests"),
        ("01_data_access_inventory", "scripts/01_download and scripts/02_inventory"),
        ("02_behavioral_preprocessing", "scripts/03_preprocess_behavior"),
        ("03_simulation_recovery", "scripts/05_artificial_agents and scripts/06_ann_intervention_gate"),
        ("04_tiny_rnn_model", "src/state_capacity/models and scripts/07_train_tinyrnn"),
        ("05_state_capacity_estimation", "scripts/08_estimate_coordinates"),
        ("06_discovery_ds007554", "scripts/10_ds007554_discovery"),
        ("07_external_sleepybrain", "diagnostic only unless direct download is proven"),
        ("08_external_maus_workload", "diagnostic only; manual DataPort archive excluded from main paper"),
        ("09_dynamics_analysis", "scripts/09_dynamics"),
        ("10_baselines_robustness", "scripts/15_baselines_robustness"),
        ("11_statistics_figures", "scripts/16_statistics_figures"),
        ("12_manuscript_outputs", "scripts/17_manuscript_package"),
        ("13_neurophys_extension", "scripts/11_ds007554_neurophys"),
    ]
    present = {path.name for path in step_dirs}
    lines = [
        "# Migration Plan",
        "",
        "The previous implementation is not copied forward as final analysis code. It is migrated only as design reference.",
        "",
        "| previous folder | revised destination | status |",
        "| --- | --- | --- |",
    ]
    for old, new in mappings:
        status = "present in previous project" if old in present else "not found in previous project"
        lines.append(f"| `{old}` | `{new}` | {status}; must regenerate |")
    lines.extend(
        [
            "",
            "## New Required Gates Added by Revised Protocol",
            "",
            "- Full-run guard in every script.",
            "- Dataset eligibility audit excluding login/manual-only data from the main paper.",
            "- Theory-to-method crosswalk before manuscript Introduction or Discussion.",
            "- Artificial-agent perturbation library.",
            "- ANN intervention gate before human projection.",
            "- Parameter/fingerprint coordinate convergence before primary human claims.",
            "- Master effects table and claim audit before manuscript package.",
            "- Final decision report before any submission framing.",
        ]
    )
    (audit_dir / "migration_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_check() -> None:
    step_dirs = locate_previous_step_dirs()
    print(f"SMOKE: found {len(step_dirs)} previous step folders.")
    if not (PROJECT_ROOT / "configs" / "run_mode.yaml").exists():
        raise RuntimeError("Missing configs/run_mode.yaml")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    assert_full_run_allowed(allow_smoke=True)
    if args.smoke:
        smoke_check()
        return

    step_dirs = locate_previous_step_dirs()
    if not step_dirs:
        raise RuntimeError(f"No previous step folders found under {previous_root()}")

    rows = inventory_previous_project(step_dirs)
    archived = archive_previous_outputs(step_dirs)

    write_csv(PROJECT_ROOT / "outputs" / "audit" / "existing_project_inventory.csv", rows)
    write_csv(PROJECT_ROOT / "outputs" / "audit" / "archived_previous_outputs.csv", archived)
    write_reproducibility_status(rows, archived)
    write_migration_plan(step_dirs)

    print(f"STEP00_AUDIT: inventoried {len(rows)} previous files from {len(step_dirs)} step folders.")
    print(f"STEP00_AUDIT: archived {len(archived)} previous outputs folders.")


if __name__ == "__main__":
    main()

