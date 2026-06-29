from __future__ import annotations

import csv
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


RAW = PROJECT_ROOT / "data" / "raw"
MANIFESTS = PROJECT_ROOT / "outputs" / "manifests"
TABLES = PROJECT_ROOT / "outputs" / "tables"
LOGS = PROJECT_ROOT / "outputs" / "logs"


DATASET_ROOTS = {
    "ds007554": RAW / "openneuro" / "ds007554",
    "hbn_release_4": RAW / "openneuro" / "ds005508",
    "cog_bci": RAW / "cog_bci",
    "tu_berlin_eeg_nirs": RAW / "tu_berlin_eeg_nirs",
}

EXPECTED_SUBJECTS = {
    "ds007554": None,
    "hbn_release_4": 324,
    "cog_bci": 29,
    "tu_berlin_eeg_nirs": 26,
}


def classify_file(path_text: str) -> dict[str, bool]:
    lower = path_text.lower()
    return {
        "is_event": lower.endswith("_events.tsv") or "events" in lower,
        "is_eeg": "/eeg/" in lower or "\\eeg\\" in lower or lower.endswith((".eeg", ".vhdr", ".vmrk", ".edf", ".bdf")),
        "is_fnirs": any(token in lower for token in ["/nirs/", "\\nirs\\", ".snirf", "nirs", "fnirs", "hbo", "hbr"]),
        "is_ecg": "ecg" in lower,
        "is_questionnaire": any(token in lower for token in ["questionnaire", "survey", "nasa", "tlx", "psqi", "subjective", "scale"]),
    }


def parse_subject(path_text: str) -> str | None:
    match = re.search(r"(sub-[A-Za-z0-9]+)", path_text)
    if match:
        return match.group(1)
    match = re.search(r"(VP[0-9]{3})", path_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"sub-([0-9]{2})\.zip", path_text, flags=re.IGNORECASE)
    if match:
        return f"sub-{match.group(1)}"
    return None


def parse_session(path_text: str) -> str | None:
    match = re.search(r"(ses-[A-Za-z0-9]+)", path_text)
    if match:
        return match.group(1)
    match = re.search(r"session[_ -]?([0-9]+)", path_text, flags=re.IGNORECASE)
    if match:
        return f"session-{match.group(1)}"
    return None


def parse_task(path_text: str) -> str | None:
    match = re.search(r"task-([A-Za-z0-9]+)", path_text)
    if match:
        return match.group(1).lower()
    for token in ["nback", "pvt", "flanker", "matb", "dsr", "wg", "symbol"]:
        if token in path_text.lower():
            return token
    return None


def inventory_extracted(dataset_id: str, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return rows
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        flags = classify_file(rel)
        rows.append(
            {
                "dataset_id": dataset_id,
                "container": "",
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "subject": parse_subject(rel) or "",
                "session": parse_session(rel) or "",
                "task": parse_task(rel) or "",
                **flags,
                "readable": True,
                "error": "",
            }
        )
    return rows


def inventory_zip(dataset_id: str, zip_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rel_zip = zip_path.relative_to(PROJECT_ROOT).as_posix()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = info.filename
                full = f"{rel_zip}!{inner}"
                flags = classify_file(full)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "container": rel_zip,
                        "relative_path": full,
                        "size_bytes": info.file_size,
                        "last_modified": "",
                        "subject": parse_subject(full) or "",
                        "session": parse_session(full) or "",
                        "task": parse_task(full) or "",
                        **flags,
                        "readable": bad is None,
                        "error": "" if bad is None else f"first_bad_member={bad}",
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "dataset_id": dataset_id,
                "container": rel_zip,
                "relative_path": rel_zip,
                "size_bytes": zip_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(zip_path.stat().st_mtime).isoformat(timespec="seconds"),
                "subject": parse_subject(rel_zip) or "",
                "session": "",
                "task": "",
                "is_event": False,
                "is_eeg": False,
                "is_fnirs": False,
                "is_ecg": False,
                "is_questionnaire": False,
                "readable": False,
                "error": str(exc),
            }
        )
    return rows


def inventory_dataset(dataset_id: str, root: Path) -> list[dict[str, object]]:
    rows = inventory_extracted(dataset_id, root)
    if root.exists():
        for zip_path in root.rglob("*.zip"):
            rows.extend(inventory_zip(dataset_id, zip_path))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        dataset_id = str(row["dataset_id"])
        if dataset_id not in grouped:
            grouped[dataset_id] = {
                "dataset_id": dataset_id,
                "n_files": 0,
                "n_subjects": 0,
                "n_sessions": 0,
                "n_tasks": 0,
                "n_event_files": 0,
                "n_eeg_files": 0,
                "n_fnirs_files": 0,
                "n_ecg_files": 0,
                "n_questionnaire_files": 0,
                "expected_subjects": EXPECTED_SUBJECTS.get(dataset_id) or "",
                "subject_fraction": "",
                "passes_95pct_subject_rule": "",
            }
        item = grouped[dataset_id]
        item["n_files"] = int(item["n_files"]) + 1
        if row.get("subject"):
            sets[dataset_id]["subjects"].add(str(row["subject"]))
        if row.get("session"):
            sets[dataset_id]["sessions"].add(str(row["session"]))
        if row.get("task"):
            sets[dataset_id]["tasks"].add(str(row["task"]))
        for key, count_name in [
            ("is_event", "n_event_files"),
            ("is_eeg", "n_eeg_files"),
            ("is_fnirs", "n_fnirs_files"),
            ("is_ecg", "n_ecg_files"),
            ("is_questionnaire", "n_questionnaire_files"),
        ]:
            if row.get(key):
                item[count_name] = int(item[count_name]) + 1
    for dataset_id, item in grouped.items():
        n_subjects = len(sets[dataset_id]["subjects"])
        item["n_subjects"] = n_subjects
        item["n_sessions"] = len(sets[dataset_id]["sessions"])
        item["n_tasks"] = len(sets[dataset_id]["tasks"])
        expected = EXPECTED_SUBJECTS.get(dataset_id)
        if expected:
            frac = n_subjects / expected
            item["subject_fraction"] = f"{frac:.3f}"
            item["passes_95pct_subject_rule"] = str(frac >= 0.95).lower()
    return list(grouped.values())


def missingness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for dataset_id in DATASET_ROOTS:
        subset = [row for row in rows if row["dataset_id"] == dataset_id]
        expected = EXPECTED_SUBJECTS.get(dataset_id)
        subjects = {str(row["subject"]) for row in subset if row.get("subject")}
        unreadable = [row for row in subset if not row.get("readable")]
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "expected_subjects": expected or "",
                "observed_subjects": len(subjects),
                "missing_subjects_vs_expected": "" if expected is None else max(expected - len(subjects), 0),
                "unreadable_files": len(unreadable),
                "has_events": str(any(row.get("is_event") for row in subset)).lower(),
                "has_eeg": str(any(row.get("is_eeg") for row in subset)).lower(),
                "has_fnirs": str(any(row.get("is_fnirs") for row in subset)).lower(),
                "has_ecg": str(any(row.get("is_ecg") for row in subset)).lower(),
            }
        )
    return summary_rows


def write_failures(rows: list[dict[str, object]]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failures = [row for row in rows if not row.get("readable") or row.get("error")]
    with (LOGS / "raw_inventory_failures.log").open("w", encoding="utf-8") as handle:
        if not failures:
            handle.write("No unreadable or corrupt files detected by raw inventory.\n")
        for row in failures:
            handle.write(f"{row['dataset_id']}\t{row['relative_path']}\t{row.get('error','')}\n")


def main() -> None:
    assert_full_run_allowed()
    rows: list[dict[str, object]] = []
    for dataset_id, root in DATASET_ROOTS.items():
        rows.extend(inventory_dataset(dataset_id, root))
    fieldnames = [
        "dataset_id",
        "container",
        "relative_path",
        "size_bytes",
        "last_modified",
        "subject",
        "session",
        "task",
        "is_event",
        "is_eeg",
        "is_fnirs",
        "is_ecg",
        "is_questionnaire",
        "readable",
        "error",
    ]
    write_csv(MANIFESTS / "raw_file_inventory.csv", rows, fieldnames=fieldnames)
    write_csv(TABLES / "dataset_subject_session_counts.csv", summarize(rows))
    write_csv(TABLES / "event_missingness_report.csv", missingness(rows))
    write_failures(rows)
    print(f"RAW_INVENTORY: wrote {len(rows)} file/member rows")


if __name__ == "__main__":
    main()
