from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
AUDIT = OUTPUTS / "audit"
FIGURES = OUTPUTS / "figures"
NHB = OUTPUTS / "nhb_revision"
NHB_TABLES = NHB / "tables"
NHB_FIGURES = NHB / "figures"
NHB_AUDIT = NHB / "audit"
NHB_LOGS = NHB / "logs"
NHB_MODELS = NHB / "models"
NHB_MANUSCRIPT = NHB / "manuscript_source"


RESULT_COLUMNS = [
    "analysis_id",
    "script_name",
    "dataset",
    "task",
    "subject_id",
    "participant_id",
    "session",
    "split",
    "model_family",
    "hidden_size",
    "state_definition",
    "capacity_definition",
    "outcome",
    "predictor",
    "n_rows",
    "n_subjects",
    "estimate",
    "std_error",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "effect_direction",
    "control_status",
    "claim_strength",
    "interpretation",
    "source_table",
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

FIGURE_MAP_COLUMNS = [
    "figure_id",
    "panel",
    "source_table",
    "source_script",
    "output_path",
    "legend_path",
]


def ensure_nhb_dirs() -> None:
    for path in [NHB_TABLES, NHB_FIGURES, NHB_AUDIT, NHB_LOGS, NHB_MODELS, NHB_MANUSCRIPT]:
        path.mkdir(parents=True, exist_ok=True)
    (NHB_FIGURES / "extended_data").mkdir(parents=True, exist_ok=True)
    (NHB_TABLES / "extended_data_source").mkdir(parents=True, exist_ok=True)


def write_header_if_missing(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow(columns)


def append_tsv(path: Path, columns: list[str], row: dict[str, object]) -> None:
    write_header_if_missing(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)


def append_registry(
    analysis_id: str,
    script_name: str,
    started_at_utc: str,
    outputs: Iterable[Path],
    status: str = "complete",
    notes: str = "",
) -> None:
    rel_outputs = ";".join(path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix() for path in outputs)
    append_tsv(
        NHB_AUDIT / "nhb_analysis_registry.tsv",
        REGISTRY_COLUMNS,
        {
            "analysis_id": analysis_id,
            "script_name": script_name,
            "started_at_utc": started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "outputs": rel_outputs,
            "notes": notes,
        },
    )


def append_manifest(analysis_id: str, script_name: str, outputs: Iterable[Path]) -> None:
    cols = ["created_at_utc", "analysis_id", "script_name", "output_path", "output_type", "rows"]
    path = NHB_AUDIT / "nhb_revision_manifest.tsv"
    write_header_if_missing(path, cols)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        for out in outputs:
            rows = ""
            if out.exists() and out.suffix.lower() in {".csv", ".tsv"}:
                try:
                    rows = len(pd.read_csv(out, sep="\t" if out.suffix.lower() == ".tsv" else ","))
                except Exception:
                    rows = ""
            writer.writerow(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "analysis_id": analysis_id,
                    "script_name": script_name,
                    "output_path": out.relative_to(ROOT).as_posix() if out.is_absolute() else out.as_posix(),
                    "output_type": out.suffix.lower().lstrip("."),
                    "rows": rows,
                }
            )


def append_exclusion(
    analysis_id: str,
    script_name: str,
    dataset: str,
    task: str,
    model_family: str,
    reason: str,
    severity: str = "warning",
) -> None:
    append_tsv(
        NHB_AUDIT / "nhb_exclusion_log.tsv",
        EXCLUSION_COLUMNS,
        {
            "analysis_id": analysis_id,
            "script_name": script_name,
            "dataset": dataset,
            "task": task,
            "model_family": model_family,
            "reason": reason,
            "severity": severity,
        },
    )


def bh_q(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    m = len(valid)
    q = valid.to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def claim_strength(p_value: float | None, q_value: float | None, control_status: str, model_internal: bool = False) -> str:
    if control_status == "failed_gate":
        return "failed_gate"
    if control_status == "control_passed":
        return "control_passed"
    if control_status == "control_failed":
        return "control_failed"
    if q_value is not None and np.isfinite(q_value) and q_value < 0.05:
        return "moderate" if model_internal else "strong"
    if p_value is not None and np.isfinite(p_value) and p_value < 0.05:
        return "exploratory"
    return "negative"


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * 0.0
    return (values - values.mean()) / sd


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score

        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return np.nan
