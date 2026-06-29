from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB_TABLES, TABLES, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_05_state_variance_decomposition"
SCRIPT_NAME = "scripts/nhb_revision/05_state_variance_decomposition.py"


STATE_COLUMNS = [
    "state_lapse_axis_z",
    "state_drift_axis_z",
    "state_variability_axis_z",
    "state_reliability_axis_z",
    "state_multidimensional_summary_z",
]
CAPACITY_COLUMNS = [
    "capacity_hidden_size_axis_z_z",
    "capacity_selection_confidence_z",
    "capacity_complexity_preference_axis_z",
    "capacity_high_capacity_nll_advantage_z",
    "capacity_load_robustness_axis_z",
    "capacity_cross_task_consistency_axis_z",
    "capacity_multidimensional_summary_z",
]


def variance_components(df: pd.DataFrame, value_col: str, subject_col: str = "participant_id") -> dict[str, float]:
    data = df[[subject_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        return {"within_person_variance": np.nan, "between_person_variance": np.nan, "icc": np.nan}
    subject_means = data.groupby(subject_col)[value_col].mean()
    between = float(subject_means.var(ddof=0))
    within_values = []
    for _, group in data.groupby(subject_col):
        if len(group) > 1:
            within_values.append(float(group[value_col].var(ddof=0)))
    within = float(np.nanmean(within_values)) if within_values else 0.0
    denom = within + between
    icc = float(between / denom) if denom > 0 else np.nan
    return {"within_person_variance": within, "between_person_variance": between, "icc": icc}


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    state = pd.read_csv(TABLES / "session_state_multiaxis_coordinates.csv")
    state["participant_id"] = state["dataset"].astype(str) + ":" + state["subject"].astype(str)
    capacity = pd.read_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv")
    projection = pd.read_csv(TABLES / "human_state_capacity_multiaxis_projection.csv")
    rows = []
    for col in STATE_COLUMNS:
        vc = variance_components(state, col)
        rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "script_name": SCRIPT_NAME,
                "dataset": "all",
                "task": "all",
                "subject_id": "",
                "participant_id": "",
                "session": "",
                "split": "session_task",
                "model_family": "not_model_specific",
                "hidden_size": "",
                "state_definition": "session_task_state_component",
                "capacity_definition": "",
                "outcome": col,
                "predictor": "subject_random_intercept_variance",
                "n_rows": int(state[col].notna().sum()),
                "n_subjects": int(state["participant_id"].nunique()),
                "estimate": vc["icc"],
                "std_error": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "q_value": "",
                "effect_direction": "lower_icc_indicates_more_within_person_state_variation",
                "control_status": "variance_decomposition",
                "claim_strength": "moderate" if vc["icc"] < 0.5 else "exploratory",
                "interpretation": "State should behave more like a session/task reliability profile than a stable trait.",
                "source_table": "state_capacity_variance_decomposition.csv",
                "construct": "state",
                "component": col,
                **vc,
                "n_sessions": int(state["session"].nunique()),
                "n_session_tasks": int(len(state)),
            }
        )
    repeated = projection.copy()
    for col in CAPACITY_COLUMNS:
        if col not in repeated:
            continue
        vc = variance_components(repeated, col)
        rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "script_name": SCRIPT_NAME,
                "dataset": "all",
                "task": "all",
                "subject_id": "",
                "participant_id": "",
                "session": "",
                "split": "participant_capacity_repeated_over_session_tasks",
                "model_family": "not_model_specific",
                "hidden_size": "",
                "state_definition": "",
                "capacity_definition": "participant_level_capacity_component",
                "outcome": col,
                "predictor": "subject_random_intercept_variance",
                "n_rows": int(repeated[col].notna().sum()),
                "n_subjects": int(repeated["participant_id"].nunique()),
                "estimate": vc["icc"],
                "std_error": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "q_value": "",
                "effect_direction": "higher_icc_expected_for_capacity_than_state",
                "control_status": "variance_decomposition",
                "claim_strength": "moderate" if vc["icc"] > 0.5 else "exploratory",
                "interpretation": "Capacity is participant-level and should show greater between-person stability than state.",
                "source_table": "state_capacity_variance_decomposition.csv",
                "construct": "capacity",
                "component": col,
                **vc,
                "n_sessions": int(repeated["session"].nunique()),
                "n_session_tasks": int(len(repeated)),
            }
        )
    # Also include the one-row participant-only capacity variance to make the design boundary explicit.
    for col in CAPACITY_COLUMNS:
        if col not in capacity:
            continue
        vc = {"within_person_variance": 0.0, "between_person_variance": float(pd.to_numeric(capacity[col], errors="coerce").var(ddof=0)), "icc": 1.0}
        rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "script_name": SCRIPT_NAME,
                "dataset": "all",
                "task": "participant_level",
                "subject_id": "",
                "participant_id": "",
                "session": "",
                "split": "participant_only",
                "model_family": "not_model_specific",
                "hidden_size": "",
                "state_definition": "",
                "capacity_definition": "participant_level_capacity_component",
                "outcome": col,
                "predictor": "participant_level_variance",
                "n_rows": int(capacity[col].notna().sum()),
                "n_subjects": int(capacity["participant_id"].nunique()),
                "estimate": 1.0,
                "std_error": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "q_value": "",
                "effect_direction": "capacity_defined_between_participants",
                "control_status": "design_boundary",
                "claim_strength": "exploratory",
                "interpretation": "Capacity is estimated at participant level in this implementation; session-level capacity reliability is not claimed.",
                "source_table": "state_capacity_variance_decomposition.csv",
                "construct": "capacity",
                "component": col,
                **vc,
                "n_sessions": "",
                "n_session_tasks": "",
            }
        )
    out = pd.DataFrame(rows)
    out_path = NHB_TABLES / "state_capacity_variance_decomposition.csv"
    out.to_csv(out_path, index=False)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, [out_path])
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, [out_path], notes=f"Variance decomposition rows={len(out)}.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
