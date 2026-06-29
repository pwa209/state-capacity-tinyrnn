from __future__ import annotations

import argparse
import gzip
import json
import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    import mne

    HAVE_MNE = True
    MNE_ERROR = ""
except Exception as exc:  # pragma: no cover - optional reader dependency
    HAVE_MNE = False
    MNE_ERROR = f"{type(exc).__name__}: {exc}"


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
LOGS = ROOT / "outputs" / "logs"

COMMON_COLUMNS = [
    "dataset",
    "subject",
    "session",
    "task",
    "block",
    "trial_index",
    "timestamp",
    "condition",
    "load_level",
    "stimulus_id",
    "target",
    "response",
    "correct",
    "rt",
    "previous_response",
    "previous_correct",
    "time_on_task",
    "available_response_set",
]

PROVENANCE_COLUMNS = [
    "duration",
    "source_file",
    "source_row",
    "source_modality",
    "event_type",
    "event_included",
    "exclusion_reason",
]

OUTPUT_COLUMNS = COMMON_COLUMNS + PROVENANCE_COLUMNS


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def clean_missing(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, float) and np.isnan(value):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"n/a", "na", "nan", "none", "null"}:
            return pd.NA
        return stripped
    return value


def is_missing(value: Any) -> bool:
    return value is pd.NA or value is None or (isinstance(value, float) and np.isnan(value))


def as_float(value: Any) -> float:
    value = clean_missing(value)
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return pd.NA
        if value.size == 1:
            return scalar(value.reshape(-1)[0])
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return clean_missing(value)


def matlab_field(struct: Any, name: str, default: Any = None) -> Any:
    if hasattr(struct, name):
        return getattr(struct, name)
    if isinstance(struct, np.ndarray) and struct.dtype.names and name in struct.dtype.names:
        return struct[name]
    return default


def first_mat_struct(mat: dict[str, Any]) -> Any:
    for key, value in mat.items():
        if not key.startswith("__"):
            return value
    raise ValueError("No MATLAB payload variables found")


def parse_bids_entities(path: Path | str) -> dict[str, str]:
    text = str(path).replace("\\", "/")
    entities: dict[str, str] = {}
    for key in ["sub", "ses", "task", "run"]:
        match = re.search(rf"{key}-([A-Za-z0-9]+)", text)
        if match:
            entities[key] = match.group(1)
    return entities


def relative(path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def task_from_cog_member(member: str) -> str:
    return Path(member).stem.lower().replace("-", "_").replace(" ", "_")


def infer_load_from_task_condition(task: Any, condition: Any) -> Any:
    text = f"{task} {condition}".lower()
    match = re.search(r"([0-9]+)\s*[-_ ]?back", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([0-9]+)back\b", text)
    if match:
        return int(match.group(1))
    if "easy" in text:
        return "easy"
    if "med" in text or "medium" in text:
        return "medium"
    if "diff" in text or "hard" in text:
        return "hard"
    return pd.NA


def parse_correct(value: Any) -> Any:
    value = clean_missing(value)
    if is_missing(value):
        return pd.NA
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"correct", "true", "hit", "1", "success", "smiley_face"}:
            return True
        if lowered in {"incorrect", "false", "miss", "0", "error", "wrong", "sad_face", "non_target"}:
            return False
        return pd.NA
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if np.isnan(numeric):
        return pd.NA
    if numeric == 1:
        return True
    if numeric == 0:
        return False
    return pd.NA


def empty_row(dataset: str) -> dict[str, Any]:
    row = {column: pd.NA for column in OUTPUT_COLUMNS}
    row["dataset"] = dataset
    row["event_included"] = True
    return row


def finalize_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[OUTPUT_COLUMNS].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["load_level"] = pd.to_numeric(df["load_level"], errors="coerce")
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df["event_included"] = df["event_included"].fillna(True).astype(bool)
    df["correct"] = df["correct"].astype("boolean")

    string_columns = [
        "dataset",
        "subject",
        "session",
        "task",
        "block",
        "condition",
        "stimulus_id",
        "target",
        "response",
        "previous_response",
        "available_response_set",
        "source_file",
        "source_modality",
        "event_type",
        "exclusion_reason",
    ]
    for column in string_columns:
        df[column] = df[column].astype("string")

    sort_columns = ["dataset", "subject", "session", "task", "block", "timestamp", "trial_index"]
    df = df.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)

    group_columns = ["dataset", "subject", "session", "task"]
    df["trial_index"] = df.groupby(group_columns, dropna=False).cumcount() + 1
    df["previous_response"] = df.groupby(group_columns, dropna=False)["response"].shift(1)
    df["previous_correct"] = df.groupby(group_columns, dropna=False)["correct"].shift(1)

    missing_timestamp = df["timestamp"].isna()
    if missing_timestamp.any():
        fallback_time = df.groupby(group_columns, dropna=False).cumcount().astype(float)
        df.loc[missing_timestamp, "timestamp"] = fallback_time.loc[missing_timestamp]

    df["time_on_task"] = df.groupby(group_columns, dropna=False)["timestamp"].transform(
        lambda series: series - series.min()
    )
    return df[OUTPUT_COLUMNS]


def file_exclusion(
    dataset: str,
    source_file: Path | str,
    reason: str,
    detail: str,
    subject: Any = pd.NA,
    session: Any = pd.NA,
    task: Any = pd.NA,
    n_events: int = 0,
    n_files: int = 0,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subject": subject,
        "session": session,
        "task": task,
        "exclusion_reason": reason,
        "detail": detail,
        "source_file": relative(source_file) if isinstance(source_file, Path) else str(source_file),
        "n_events": n_events,
        "n_files": n_files,
    }


def parse_ds007554() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    pushbutton_audit: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    root = RAW / "openneuro" / "ds007554"
    files = sorted(root.glob("sub-*/ses-*/*/*_events.tsv"))

    for path in files:
        entities = parse_bids_entities(path)
        modality = path.parent.name
        try:
            table = pd.read_csv(path, sep="\t")
        except Exception as exc:
            exclusions.append(file_exclusion("ds007554", path, "unreadable_events_tsv", str(exc)))
            continue

        response_map: dict[int, dict[str, Any]] = {}
        response_audit = None
        if modality == "eeg":
            response_map, response_audit = reconstruct_ds007554_pushbutton(root, path, table)
        if response_audit:
            pushbutton_audit.append(response_audit)

        for source_row, record in table.iterrows():
            onset = as_float(record.get("onset"))
            duration = as_float(record.get("duration"))
            condition = clean_missing(record.get("trial_type", record.get("value", pd.NA)))
            key = (
                entities.get("sub"),
                entities.get("ses"),
                entities.get("task"),
                onset,
                duration,
                str(condition),
            )
            if key in seen:
                exclusions.append(
                    file_exclusion(
                        "ds007554",
                        path,
                        "duplicate_multimodal_event",
                        "Same BIDS event is present in EEG and fNIRS event files",
                        subject=entities.get("sub"),
                        session=entities.get("ses"),
                        task=entities.get("task"),
                        n_events=1,
                    )
                )
                continue
            seen.add(key)

            row = empty_row("ds007554")
            row.update(
                {
                    "subject": entities.get("sub", pd.NA),
                    "session": entities.get("ses", pd.NA),
                    "task": entities.get("task", pd.NA),
                    "block": entities.get("run", pd.NA),
                    "timestamp": onset,
                    "duration": duration,
                    "condition": condition,
                    "load_level": infer_load_from_task_condition(entities.get("task", ""), condition),
                    "stimulus_id": condition,
                    "target": response_map.get(source_row, {}).get("target", pd.NA),
                    "response": response_map.get(source_row, {}).get("response", pd.NA),
                    "correct": response_map.get(source_row, {}).get("correct", pd.NA),
                    "rt": response_map.get(source_row, {}).get("rt", np.nan),
                    "available_response_set": response_map.get(source_row, {}).get("available_response_set", pd.NA),
                    "source_file": relative(path),
                    "source_row": int(source_row) + 1,
                    "source_modality": modality,
                    "event_type": condition,
                    "event_included": not np.isnan(onset),
                    "exclusion_reason": pd.NA if not np.isnan(onset) else "missing_onset",
                }
            )
            rows.append(row)

    if pushbutton_audit:
        pd.DataFrame(pushbutton_audit).to_csv(TABLES / "ds007554_pushbutton_reconstruction_audit.csv", index=False)
    return finalize_events(pd.DataFrame(rows)), exclusions


def read_ds007554_pushbutton_onsets(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta_path = path.with_suffix("").with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    fs = float(meta.get("SamplingFrequency", np.nan))
    values: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                values.append(float(line.strip().split()[0]))
            except Exception:
                continue
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(fs) or fs <= 0:
        return np.asarray([], dtype=float), {"sampling_frequency": fs, "n_samples": int(len(x)), "threshold": np.nan}
    baseline = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - baseline)))
    q95 = float(np.nanquantile(x, 0.95))
    q99 = float(np.nanquantile(x, 0.99))
    threshold = max(baseline + 10.0 * max(mad, 1e-6), baseline + 0.5, (q95 + q99) / 2.0)
    above = x > threshold
    starts = np.where(above & ~np.r_[False, above[:-1]])[0]
    if len(starts) > 1:
        min_gap = int(fs * 0.20)
        keep = [int(starts[0])]
        for start in starts[1:]:
            if int(start) - keep[-1] >= min_gap:
                keep.append(int(start))
        starts = np.asarray(keep, dtype=int)
    return starts / fs, {
        "sampling_frequency": fs,
        "n_samples": int(len(x)),
        "threshold": threshold,
        "baseline": baseline,
        "n_button_presses": int(len(starts)),
    }


def reconstruct_ds007554_pushbutton(root: Path, event_path: Path, table: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    entities = parse_bids_entities(event_path)
    task = entities.get("task", "")
    if task not in {"nback", "nbackarithmetic"}:
        return {}, None
    push_path = (
        root
        / f"sub-{entities.get('sub')}"
        / f"ses-{entities.get('ses')}"
        / "beh"
        / f"sub-{entities.get('sub')}_ses-{entities.get('ses')}_task-{task}_recording-pushbutton_physio.tsv.gz"
    )
    if not push_path.exists():
        return {}, {
            "subject": entities.get("sub"),
            "session": entities.get("ses"),
            "task": task,
            "source_file": relative(event_path),
            "pushbutton_file": relative(push_path),
            "status": "missing_pushbutton_file",
            "n_scored_trials": 0,
        }

    press_times, signal_info = read_ds007554_pushbutton_onsets(push_path)
    response_map: dict[int, dict[str, Any]] = {}
    trial_records: list[tuple[int, float, str]] = []
    for source_row, record in table.iterrows():
        condition = clean_missing(record.get("trial_type", record.get("value", pd.NA)))
        if str(condition) not in {"target", "trigger"}:
            continue
        onset = as_float(record.get("onset"))
        if np.isnan(onset):
            continue
        trial_records.append((int(source_row), onset, str(condition)))
        response_map[int(source_row)] = {
            "target": "button_press" if str(condition) == "target" else "withhold",
            "response": pd.NA,
            "correct": pd.NA,
            "rt": np.nan,
            "available_response_set": "pushbutton_go_no_go",
        }

    if not trial_records:
        return {}, None

    max_rt = 1.25 if task == "nback" else 2.20
    min_rt = 0.12
    onsets = np.asarray([record[1] for record in trial_records], dtype=float)
    assigned: set[int] = set()
    for press_time in press_times:
        deltas = press_time - onsets
        candidates = np.where((deltas >= min_rt) & (deltas <= max_rt))[0]
        if len(candidates) == 0:
            continue
        candidate = int(candidates[np.argmin(deltas[candidates])])
        source_row = trial_records[candidate][0]
        if source_row in assigned:
            continue
        assigned.add(source_row)
        response_map[source_row]["response"] = "button_press"
        response_map[source_row]["rt"] = float(deltas[candidate])

    n_targets = 0
    n_triggers = 0
    n_hits = 0
    n_false_alarms = 0
    for source_row, _onset, condition in trial_records:
        responded = not is_missing(response_map[source_row]["response"])
        is_target = condition == "target"
        response_map[source_row]["correct"] = bool((is_target and responded) or ((not is_target) and (not responded)))
        n_targets += int(is_target)
        n_triggers += int(not is_target)
        n_hits += int(is_target and responded)
        n_false_alarms += int((not is_target) and responded)

    audit = {
        "subject": entities.get("sub"),
        "session": entities.get("ses"),
        "task": task,
        "source_file": relative(event_path),
        "pushbutton_file": relative(push_path),
        "status": "scored_from_pushbutton",
        "n_scored_trials": int(len(trial_records)),
        "n_targets": int(n_targets),
        "n_triggers": int(n_triggers),
        "n_button_presses": int(signal_info.get("n_button_presses", len(press_times))),
        "n_hits": int(n_hits),
        "n_misses": int(n_targets - n_hits),
        "n_false_alarms": int(n_false_alarms),
        "n_correct_rejections": int(n_triggers - n_false_alarms),
        "mean_accuracy": float(np.mean([response_map[row]["correct"] for row, *_ in trial_records])),
        "median_rt": float(np.nanmedian([response_map[row]["rt"] for row, *_ in trial_records])),
        "sampling_frequency": signal_info.get("sampling_frequency", np.nan),
        "threshold": signal_info.get("threshold", np.nan),
    }
    return response_map, audit


def parse_hbn() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    root = RAW / "openneuro" / "ds005508"
    files = sorted(root.glob("sub-*/eeg/*_events.tsv"))

    for path in files:
        entities = parse_bids_entities(path)
        try:
            table = pd.read_csv(path, sep="\t", low_memory=False)
        except Exception as exc:
            exclusions.append(file_exclusion("hbn_release_4", path, "unreadable_events_tsv", str(exc)))
            continue

        for source_row, record in table.iterrows():
            onset = as_float(record.get("onset"))
            duration = as_float(record.get("duration"))
            value = clean_missing(record.get("value", record.get("trial_type", pd.NA)))
            event_code = clean_missing(record.get("event_code", pd.NA))
            feedback = clean_missing(record.get("feedback", pd.NA))
            user_answer = clean_missing(record.get("user_answer", pd.NA))
            correct_answer = clean_missing(record.get("correct_answer", pd.NA))
            value_text = "" if is_missing(value) else str(value).lower()
            has_explicit_answer = not is_missing(user_answer) and not is_missing(correct_answer)
            explicit_correct = bool(str(user_answer) == str(correct_answer)) if has_explicit_answer else pd.NA
            feedback_correct = parse_correct(feedback)
            correctness = explicit_correct if has_explicit_answer else feedback_correct
            response = pd.NA
            if "response" in value_text or "button" in value_text:
                response = value
            elif has_explicit_answer:
                response = user_answer
            target = pd.NA
            if "target" in value_text:
                target = value
            elif has_explicit_answer:
                target = correct_answer

            row = empty_row("hbn_release_4")
            row.update(
                {
                    "subject": entities.get("sub", pd.NA),
                    "session": entities.get("ses", pd.NA),
                    "task": entities.get("task", pd.NA),
                    "block": entities.get("run", pd.NA),
                    "timestamp": onset,
                    "duration": duration,
                    "condition": value,
                    "load_level": infer_load_from_task_condition(entities.get("task", ""), value),
                    "stimulus_id": event_code if not is_missing(event_code) else value,
                    "target": target,
                    "response": response,
                    "correct": correctness,
                    "available_response_set": "hbn_task_response",
                    "source_file": relative(path),
                    "source_row": int(source_row) + 1,
                    "source_modality": "eeg",
                    "event_type": event_code if not is_missing(event_code) else value,
                    "event_included": not np.isnan(onset),
                    "exclusion_reason": pd.NA if not np.isnan(onset) else "missing_onset",
                }
            )
            rows.append(row)

    return finalize_events(pd.DataFrame(rows)), exclusions


def parse_cog_bci() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    root = RAW / "cog_bci"
    archives = sorted(root.glob("sub-*.zip"))

    for archive in archives:
        subject = archive.stem
        try:
            with zipfile.ZipFile(archive) as zf:
                members = sorted(
                    member
                    for member in zf.namelist()
                    if "/behavioral/" in member and member.lower().endswith(".mat")
                )
                for member in members:
                    session_match = re.search(r"/(ses-[^/]+)/", member)
                    session = session_match.group(1) if session_match else pd.NA
                    task = task_from_cog_member(member)
                    if task != "pvt":
                        exclusions.append(
                            file_exclusion(
                                "cog_bci",
                                f"{archive.name}:{member}",
                                "matlab_mcos_table_unsupported_by_scipy",
                                "COG-BCI behavioral table is stored as a MATLAB object; PVT arrays are parsed directly",
                                subject=subject,
                                session=session,
                                task=task,
                                n_files=1,
                            )
                        )
                        continue

                    try:
                        mat = loadmat(io.BytesIO(zf.read(member)), squeeze_me=True, struct_as_record=False)
                        pvt = mat["PVT"]
                    except Exception as exc:
                        exclusions.append(
                            file_exclusion(
                                "cog_bci",
                                f"{archive.name}:{member}",
                                "unreadable_pvt_mat",
                                str(exc),
                                subject=subject,
                                session=session,
                                task=task,
                            )
                        )
                        continue

                    reaction_times = np.ravel(np.asarray(matlab_field(pvt, "reaction_times", []), dtype=float))
                    error_trial = np.ravel(
                        np.asarray(matlab_field(pvt, "error_trial", np.zeros_like(reaction_times)), dtype=float)
                    )
                    isi_time = np.ravel(np.asarray(matlab_field(pvt, "isi_time", np.ones_like(reaction_times)), dtype=float))
                    n_trials = int(scalar(matlab_field(pvt, "ntrials", len(reaction_times))))
                    usable = min(len(reaction_times), len(error_trial), len(isi_time), n_trials)
                    if usable == 0:
                        exclusions.append(
                            file_exclusion(
                                "cog_bci",
                                f"{archive.name}:{member}",
                                "empty_pvt_mat",
                                "PVT arrays contained no trials",
                                subject=subject,
                                session=session,
                                task=task,
                            )
                        )
                        continue

                    timestamps = np.cumsum(isi_time[:usable])
                    for idx in range(usable):
                        error = bool(error_trial[idx])
                        row = empty_row("cog_bci")
                        row.update(
                            {
                                "subject": subject,
                                "session": session,
                                "task": "pvt",
                                "block": Path(member).stem,
                                "timestamp": float(timestamps[idx]),
                                "condition": "psychomotor_vigilance",
                                "load_level": 0,
                                "stimulus_id": "pvt_prompt",
                                "target": "respond_to_prompt",
                                "response": "button_press" if not error else pd.NA,
                                "correct": not error,
                                "rt": float(reaction_times[idx]),
                                "available_response_set": "single_button",
                                "source_file": f"{archive.name}:{member}",
                                "source_row": idx + 1,
                                "source_modality": "behavior",
                                "event_type": "pvt_trial",
                            }
                        )
                        rows.append(row)
                    if usable < n_trials:
                        exclusions.append(
                            file_exclusion(
                                "cog_bci",
                                f"{archive.name}:{member}",
                                "truncated_pvt_arrays",
                                "PVT arrays had fewer entries than ntrials",
                                subject=subject,
                                session=session,
                                task=task,
                                n_events=n_trials - usable,
                            )
                        )
                eeg_set_members = sorted(
                    member
                    for member in zf.namelist()
                    if "/eeg/" in member
                    and member.lower().endswith(".set")
                    and Path(member).stem.lower() in {"zeroback", "oneback", "twoback", "flanker"}
                )
                for member in eeg_set_members:
                    session_match = re.search(r"/(ses-[^/]+)/", member)
                    session = session_match.group(1) if session_match else pd.NA
                    new_rows, new_exclusions = parse_cog_eeglab_marker_trials(zf, archive.name, member, subject, session)
                    rows.extend(new_rows)
                    exclusions.extend(new_exclusions)
        except Exception as exc:
            exclusions.append(file_exclusion("cog_bci", archive, "unreadable_subject_zip", str(exc), subject=subject))

    return finalize_events(pd.DataFrame(rows)), exclusions


def parse_cog_eeglab_marker_trials(
    zf: zipfile.ZipFile, archive_name: str, member: str, subject: str, session: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    stem = Path(member).stem.lower()
    if not HAVE_MNE:
        return rows, [
            file_exclusion(
                "cog_bci",
                f"{archive_name}:{member}",
                "mne_unavailable_for_eeglab_marker_parsing",
                MNE_ERROR,
                subject=subject,
                session=session,
                task=stem,
            )
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        set_path = tmp / member
        set_path.parent.mkdir(parents=True, exist_ok=True)
        set_path.write_bytes(zf.read(member))
        fdt_path = set_path.with_suffix(".fdt")
        fdt_path.write_bytes(b"")
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
        except Exception as exc:
            exclusions.append(
                file_exclusion(
                    "cog_bci",
                    f"{archive_name}:{member}",
                    "unreadable_eeglab_set_markers",
                    str(exc),
                    subject=subject,
                    session=session,
                    task=stem,
                )
            )
            return rows, exclusions

    annotations = sorted(
        [
            (float(onset), str(description))
            for onset, description in zip(raw.annotations.onset, raw.annotations.description)
            if str(description).lower() != "boundary"
        ],
        key=lambda item: item[0],
    )

    if stem in {"zeroback", "oneback", "twoback"}:
        load = {"zeroback": 0, "oneback": 1, "twoback": 2}[stem]
        task = "nback"
        prefix = {"zeroback": "60", "oneback": "61", "twoback": "62"}[stem]
        onset_codes = {f"{prefix}21": "normal", f"{prefix}22": "hit", f"{prefix}23": "conflict"}
        correct_codes = {f"{prefix}32"}
        error_codes = {f"{prefix}31", f"{prefix}33"}
        trial_counter = 0
        for idx, (onset, code) in enumerate(annotations):
            if code not in onset_codes:
                continue
            following = annotations[idx + 1 :]
            next_trial_onset = next((t for t, c in following if c in onset_codes or c.endswith("12")), onset + 4.0)
            response = next((item for item in following if onset <= item[0] < next_trial_onset and item[1] in correct_codes | error_codes), None)
            trial_counter += 1
            is_target = onset_codes[code] in {"hit", "conflict"}
            is_correct = bool(response is not None and response[1] in correct_codes) if is_target else bool(
                response is None or response[1] in correct_codes
            )
            row = empty_row("cog_bci")
            row.update(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "block": stem,
                    "timestamp": onset,
                    "condition": onset_codes[code],
                    "load_level": load,
                    "stimulus_id": code,
                    "target": "button_press" if is_target else "withhold",
                    "response": "button_press" if response is not None else pd.NA,
                    "correct": is_correct,
                    "rt": float(response[0] - onset) if response is not None else np.nan,
                    "available_response_set": "nback_response_markers",
                    "source_file": f"{archive_name}:{member}",
                    "source_row": trial_counter,
                    "source_modality": "eeg_markers",
                    "event_type": f"eeg_marker_{code}",
                }
            )
            rows.append(row)
    elif stem == "flanker":
        onset_codes = {"241": "congruent", "242": "incongruent"}
        correct_codes = {"2511", "2512"}
        error_codes = {"2521", "2522"}
        trial_counter = 0
        for idx, (onset, code) in enumerate(annotations):
            if code not in onset_codes:
                continue
            following = annotations[idx + 1 :]
            next_trial_onset = next((t for t, c in following if c in onset_codes or c == "210"), onset + 4.0)
            response = next((item for item in following if onset <= item[0] < next_trial_onset and item[1] in correct_codes | error_codes), None)
            trial_counter += 1
            row = empty_row("cog_bci")
            row.update(
                {
                    "subject": subject,
                    "session": session,
                    "task": "flanker",
                    "block": "flanker",
                    "timestamp": onset,
                    "condition": onset_codes[code],
                    "load_level": 1 if onset_codes[code] == "incongruent" else 0,
                    "stimulus_id": code,
                    "target": "left_or_right_response",
                    "response": "button_press" if response is not None else pd.NA,
                    "correct": bool(response is not None and response[1] in correct_codes),
                    "rt": float(response[0] - onset) if response is not None else np.nan,
                    "available_response_set": "flanker_response_markers",
                    "source_file": f"{archive_name}:{member}",
                    "source_row": trial_counter,
                    "source_modality": "eeg_markers",
                    "event_type": f"eeg_marker_{code}",
                }
            )
            rows.append(row)

    if not rows:
        exclusions.append(
            file_exclusion(
                "cog_bci",
                f"{archive_name}:{member}",
                "no_supported_marker_trials_found",
                "EEGLAB annotations did not contain expected COG-BCI marker codes",
                subject=subject,
                session=session,
                task=stem,
            )
        )
    return rows, exclusions


def parse_tu_berlin() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    archive = RAW / "tu_berlin_eeg_nirs" / "behavior.zip"

    try:
        zf = zipfile.ZipFile(archive)
    except Exception as exc:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), [
            file_exclusion("tu_berlin_eeg_nirs", archive, "unreadable_behavior_zip", str(exc))
        ]

    with zf:
        members = sorted(member for member in zf.namelist() if member.lower().endswith(".mat"))
        for member in members:
            task = "nback" if member.lower().startswith("n-back/") else "gonogo"
            subject_match = re.search(r"(VP[0-9]{3})", member, flags=re.IGNORECASE)
            subject = subject_match.group(1).upper() if subject_match else pd.NA
            try:
                mat = loadmat(io.BytesIO(zf.read(member)), squeeze_me=True, struct_as_record=False)
                summary = first_mat_struct(mat)
            except Exception as exc:
                exclusions.append(
                    file_exclusion(
                        "tu_berlin_eeg_nirs",
                        f"{archive.name}:{member}",
                        "unreadable_summary_mat",
                        str(exc),
                        subject=subject,
                        task=task,
                    )
                )
                continue

            session = scalar(matlab_field(summary, "session", pd.NA))
            response = np.asarray(matlab_field(summary, "response", []))
            result = np.asarray(matlab_field(summary, "result", []))
            reaction_time = np.asarray(matlab_field(summary, "reaction_time", np.full_like(response, np.nan, dtype=float)))
            flag = np.asarray(matlab_field(summary, "flag", np.full_like(response, np.nan, dtype=float)))

            if response.size == 0 or result.size == 0:
                exclusions.append(
                    file_exclusion(
                        "tu_berlin_eeg_nirs",
                        f"{archive.name}:{member}",
                        "empty_summary_mat",
                        "Behavior summary contained no trial arrays",
                        subject=subject,
                        session=session,
                        task=task,
                    )
                )
                continue

            response = np.atleast_2d(response)
            result = np.atleast_2d(result)
            reaction_time = np.atleast_2d(reaction_time)
            flag = np.atleast_2d(flag)
            n_back = np.ravel(np.asarray(matlab_field(summary, "nback", [])))

            trial_counter = 0
            for block_index in range(response.shape[0]):
                block_load: Any = pd.NA
                if task == "nback" and block_index < len(n_back):
                    candidate_load = as_float(n_back[block_index])
                    if not np.isnan(candidate_load):
                        block_load = int(candidate_load)
                for within_block in range(response.shape[1]):
                    trial_counter += 1
                    res = as_float(result[block_index, within_block])
                    resp = scalar(response[block_index, within_block])
                    flg = scalar(flag[block_index, within_block])
                    rt = as_float(reaction_time[block_index, within_block])

                    included = not (np.isnan(res) or res < 0)
                    if task == "nback":
                        condition = f"{block_load}-back" if not is_missing(block_load) else "n-back"
                        target = flg
                        available = "nback_response_keys"
                    else:
                        condition = "go" if as_float(flg) == 1 else "nogo"
                        target = condition
                        available = "go_response_or_withhold"

                    row = empty_row("tu_berlin_eeg_nirs")
                    row.update(
                        {
                            "subject": subject,
                            "session": session,
                            "task": task,
                            "block": block_index + 1,
                            "timestamp": trial_counter,
                            "condition": condition,
                            "load_level": block_load,
                            "stimulus_id": flg,
                            "target": target,
                            "response": pd.NA if as_float(resp) == -1 else resp,
                            "correct": bool(res == 1) if included else pd.NA,
                            "rt": rt,
                            "available_response_set": available,
                            "source_file": f"{archive.name}:{member}",
                            "source_row": trial_counter,
                            "source_modality": "behavior",
                            "event_type": f"{task}_trial",
                            "event_included": included,
                            "exclusion_reason": pd.NA if included else "practice_or_not_scored",
                        }
                    )
                    rows.append(row)

    return finalize_events(pd.DataFrame(rows)), exclusions


def build_exclusion_table(dataset_frames: dict[str, pd.DataFrame], file_exclusions: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, df in dataset_frames.items():
        if df.empty:
            continue
        excluded = df[~df["event_included"]].copy()
        if not excluded.empty:
            grouped = (
                excluded.groupby(["dataset", "task", "exclusion_reason"], dropna=False)
                .size()
                .reset_index(name="n_events")
            )
            for record in grouped.to_dict("records"):
                record.update({"n_files": 0, "detail": "event-level exclusion"})
                rows.append(record)

    if file_exclusions:
        file_df = pd.DataFrame(file_exclusions)
        grouped_files = (
            file_df.groupby(["dataset", "task", "exclusion_reason", "detail"], dropna=False)
            .agg(n_events=("n_events", "sum"), n_files=("source_file", "nunique"))
            .reset_index()
        )
        rows.extend(grouped_files.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=["dataset", "task", "exclusion_reason", "detail", "n_events", "n_files"])
    return pd.DataFrame(rows).sort_values(["dataset", "task", "exclusion_reason"]).reset_index(drop=True)


def missing_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.isna().mean())


def build_count_tables(dataset_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    subject_task: list[dict[str, Any]] = []

    for dataset, df in dataset_frames.items():
        if df.empty:
            counts.append(
                {
                    "dataset": dataset,
                    "n_events": 0,
                    "n_included_events": 0,
                    "n_excluded_events": 0,
                    "n_subjects": 0,
                    "n_sessions": 0,
                    "n_tasks": 0,
                    "n_blocks": 0,
                    "n_source_files": 0,
                    "n_events_with_correct": 0,
                    "n_events_with_rt": 0,
                }
            )
            missingness.append(
                {
                    "dataset": dataset,
                    "n_events": 0,
                    "n_included_events": 0,
                    "missing_timestamp_fraction": np.nan,
                    "missing_condition_fraction": np.nan,
                    "missing_response_fraction": np.nan,
                    "missing_correct_fraction": np.nan,
                    "missing_rt_fraction": np.nan,
                    "has_response": False,
                    "has_correct": False,
                    "has_rt": False,
                    "model_event_table_ready": False,
                }
            )
            continue

        included = df[df["event_included"]]
        counts.append(
            {
                "dataset": dataset,
                "n_events": int(len(df)),
                "n_included_events": int(len(included)),
                "n_excluded_events": int((~df["event_included"]).sum()),
                "n_subjects": int(df["subject"].nunique(dropna=True)),
                "n_sessions": int(df["session"].nunique(dropna=True)),
                "n_tasks": int(df["task"].nunique(dropna=True)),
                "n_blocks": int(df["block"].nunique(dropna=True)),
                "n_source_files": int(df["source_file"].nunique(dropna=True)),
                "n_events_with_correct": int(df["correct"].notna().sum()),
                "n_events_with_rt": int(df["rt"].notna().sum()),
            }
        )
        missingness.append(
            {
                "dataset": dataset,
                "n_events": int(len(df)),
                "n_included_events": int(len(included)),
                "missing_timestamp_fraction": missing_fraction(df["timestamp"]),
                "missing_condition_fraction": missing_fraction(df["condition"]),
                "missing_response_fraction": missing_fraction(df["response"]),
                "missing_correct_fraction": missing_fraction(df["correct"]),
                "missing_rt_fraction": missing_fraction(df["rt"]),
                "has_response": bool(df["response"].notna().any()),
                "has_correct": bool(df["correct"].notna().any()),
                "has_rt": bool(df["rt"].notna().any()),
                "model_event_table_ready": bool(len(df) > 0),
            }
        )

        grouped = (
            df.groupby(["dataset", "subject", "session", "task"], dropna=False)
            .agg(
                n_events=("dataset", "size"),
                n_included_events=("event_included", "sum"),
                n_events_with_correct=("correct", lambda series: int(series.notna().sum())),
                n_events_with_rt=("rt", lambda series: int(series.notna().sum())),
            )
            .reset_index()
        )
        subject_task.extend(grouped.to_dict("records"))

    return (
        pd.DataFrame(counts).sort_values("dataset").reset_index(drop=True),
        pd.DataFrame(missingness).sort_values("dataset").reset_index(drop=True),
        pd.DataFrame(subject_task).sort_values(["dataset", "subject", "session", "task"]).reset_index(drop=True),
    )


def write_outputs(dataset_frames: dict[str, pd.DataFrame], exclusions: list[dict[str, Any]]) -> None:
    output_names = {
        "ds007554": "ds007554_model_events.parquet",
        "cog_bci": "cog_bci_model_events.parquet",
        "tu_berlin_eeg_nirs": "tu_berlin_model_events.parquet",
        "hbn_release_4": "hbn_model_events.parquet",
    }

    for dataset, df in dataset_frames.items():
        df.to_parquet(PROCESSED / output_names[dataset], index=False)

    all_events = pd.concat(dataset_frames.values(), ignore_index=True)
    all_events = finalize_events(all_events)
    all_events.to_parquet(PROCESSED / "all_model_events.parquet", index=False)

    counts, missingness, subject_task = build_count_tables(dataset_frames)
    exclusion_table = build_exclusion_table(dataset_frames, exclusions)

    counts.to_csv(TABLES / "event_counts_by_dataset.csv", index=False)
    missingness.to_csv(TABLES / "event_missingness_report.csv", index=False)
    subject_task.to_csv(TABLES / "event_counts_by_subject_task.csv", index=False)
    exclusion_table.to_csv(TABLES / "event_exclusion_counts.csv", index=False)

    log_lines = [
        "Step 03 unified behavioral event schema completed.",
        f"Total events: {len(all_events)}",
        f"Included events: {int(all_events['event_included'].sum())}",
        f"Excluded events: {int((~all_events['event_included']).sum())}",
        "",
        counts.to_csv(index=False),
        "",
        "File-level exclusions:",
        pd.DataFrame(exclusions).to_csv(index=False) if exclusions else "none\n",
    ]
    (LOGS / "preprocess_behavior.log").write_text("\n".join(log_lines), encoding="utf-8")


def main() -> int:
    argparse.ArgumentParser(description="Build full unified behavioral event schema.").parse_args()
    ensure_dirs()

    parsers = {
        "ds007554": parse_ds007554,
        "hbn_release_4": parse_hbn,
        "cog_bci": parse_cog_bci,
        "tu_berlin_eeg_nirs": parse_tu_berlin,
    }

    dataset_frames: dict[str, pd.DataFrame] = {}
    all_exclusions: list[dict[str, Any]] = []

    for dataset, parser_fn in parsers.items():
        print(f"STEP03_PARSE dataset={dataset}", flush=True)
        frame, exclusions = parser_fn()
        dataset_frames[dataset] = frame
        all_exclusions.extend(exclusions)
        included = int(frame["event_included"].sum()) if not frame.empty else 0
        print(f"STEP03_DATASET dataset={dataset} events={len(frame)} included={included}", flush=True)

    write_outputs(dataset_frames, all_exclusions)
    print("STEP03_COMPLETE output=data/processed/all_model_events.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
