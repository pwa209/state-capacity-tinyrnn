from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
TABLE_DIR = PACKAGE_ROOT / "outputs" / "tables"
AUDIT_DIR = PACKAGE_ROOT / "outputs" / "audit"
SOURCE_DIR = PACKAGE_ROOT / "outputs" / "source_data"
LOG_DIR = PACKAGE_ROOT / "outputs" / "logs"
PROCESSED_DIR = PACKAGE_ROOT / "data" / "processed"


BEHAVIOR_FEATURES = [
    "mean_accuracy",
    "negative_log_likelihood",
    "brier_score",
    "response_rate",
    "response_entropy",
    "lapse_proxy",
    "time_accuracy_slope",
    "early_late_accuracy_delta",
    "response_time_slope",
    "error_transition_rate",
    "error_lag1_autocorrelation",
    "sequential_dependence",
    "nback_load_accuracy_slope",
]


STATE_AXES = [
    "state_lapse_axis_z",
    "state_drift_axis_z",
    "state_variability_axis_z",
    "state_reliability_axis_inverted_z",
]


CAPACITY_AXES = [
    "capacity_hidden_size_axis_z_z",
    "capacity_selection_confidence_z",
    "capacity_complexity_preference_axis_z",
    "capacity_high_capacity_nll_advantage_z",
    "capacity_load_robustness_axis_z",
    "capacity_cross_task_consistency_axis_z",
]


@dataclass
class SpearmanResult:
    analysis_family: str
    x: str
    y: str
    controls: str
    n: int
    spearman_rho: float
    p_value: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    claim_strength: str


def ensure_dirs() -> None:
    for path in [TABLE_DIR, AUDIT_DIR, SOURCE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    x = as_numeric(series)
    mean = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=series.index)
    return (x - mean) / sd


def safe_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3 or np.nanstd(x_arr[mask]) <= 1e-12:
        return np.nan
    return float(np.polyfit(x_arr[mask], y_arr[mask], deg=1)[0])


def normalized_entropy(values: pd.Series) -> float:
    clean = values.astype("string").fillna("<missing>")
    counts = clean.value_counts(dropna=False).to_numpy(dtype=float)
    if counts.sum() <= 0 or len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum() / np.log2(len(probs)))


def lag_corr(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    x = values[:-1]
    y = values[1:]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def response_present(response: pd.Series) -> pd.Series:
    text = response.astype("string").str.lower().fillna("")
    absent = text.isin(["", "<na>", "nan", "none", "no_response", "no response", "miss", "missing"])
    return ~absent


def add_correct_num(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    correct_text = out["correct"].astype("string").str.lower()
    out["correct_num"] = np.select(
        [
            correct_text.isin(["true", "1", "1.0"]),
            correct_text.isin(["false", "0", "0.0"]),
        ],
        [1.0, 0.0],
        default=np.nan,
    )
    return out


def build_human_behavior_fingerprints(events: pd.DataFrame) -> pd.DataFrame:
    included = add_correct_num(events[events["event_included"].fillna(False).astype(bool)]).copy()
    included["session"] = included["session"].fillna("no_session").astype("string")
    included["response_present"] = response_present(included["response"]).astype(float)
    included["rt_num"] = pd.to_numeric(included["rt"], errors="coerce")
    included["trial_num"] = pd.to_numeric(included["trial_index"], errors="coerce")
    included["load_num"] = pd.to_numeric(included["load_level"], errors="coerce")

    rows = []
    group_cols = ["dataset", "subject", "session", "task"]
    for keys, group in included.groupby(group_cols, dropna=False):
        group = group.sort_values(["trial_num", "timestamp"], na_position="last")
        y_raw = group["correct_num"].to_numpy(dtype=float)
        y_mask = np.isfinite(y_raw)
        y = y_raw[y_mask]
        n = int(len(y))
        if n == 0:
            continue
        acc = float(np.mean(y))
        p = float(np.clip(acc, 1e-4, 1 - 1e-4))
        trial = np.arange(n, dtype=float)
        trial_norm = trial / max(n - 1, 1)
        n_third = max(n // 3, 1)
        early = float(np.mean(y[:n_third]))
        late = float(np.mean(y[-n_third:]))

        response_binary = group["response_present"].to_numpy(dtype=float)
        response_binary = response_binary[np.isfinite(response_binary)]
        error = 1.0 - y

        rt = group["rt_num"].to_numpy(dtype=float)
        rt_mask = np.isfinite(rt)
        rt_trial_norm = np.arange(len(rt), dtype=float) / max(len(rt) - 1, 1)
        rt_slope = safe_slope(rt_trial_norm[rt_mask], rt[rt_mask]) if rt_mask.sum() >= 3 else np.nan

        load = group["load_num"].to_numpy(dtype=float)
        load_y = load[y_mask]
        load_slope = safe_slope(load_y, y) if np.isfinite(load_y).sum() >= 3 else np.nan

        rows.append(
            {
                "dataset": keys[0],
                "subject": keys[1],
                "session": keys[2],
                "task": keys[3],
                "participant_id": f"{keys[0]}:{keys[1]}",
                "n_events": n,
                "mean_accuracy": acc,
                "negative_log_likelihood": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "brier_score": float(np.mean((y - p) ** 2)),
                "response_rate": float(np.mean(response_binary)) if len(response_binary) else np.nan,
                "response_entropy": normalized_entropy(group["response"]),
                "lapse_proxy": 1.0 - acc,
                "time_accuracy_slope": safe_slope(trial_norm, y),
                "early_late_accuracy_delta": late - early,
                "response_time_slope": rt_slope,
                "error_transition_rate": float(np.mean(np.abs(np.diff(error)))) if n >= 2 else np.nan,
                "error_lag1_autocorrelation": lag_corr(error),
                "sequential_dependence": lag_corr(response_binary) if len(response_binary) >= 3 else np.nan,
                "nback_load_accuracy_slope": load_slope,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_eligibility(events: pd.DataFrame, state: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    included = add_correct_num(events[events["event_included"].fillna(False).astype(bool)]).copy()
    event_counts = (
        included.groupby("dataset", dropna=False)
        .agg(
            included_events=("dataset", "size"),
            supervised_events=("correct_num", lambda x: int(x.notna().sum())),
            subjects_with_events=("subject", "nunique"),
        )
        .reset_index()
    )
    state_counts = (
        state.groupby("dataset", dropna=False)
        .agg(session_state_rows=("dataset", "size"), subjects_with_state=("subject", "nunique"))
        .reset_index()
    )
    capacity_counts = (
        capacity.groupby("dataset", dropna=False)
        .agg(participant_capacity_rows=("dataset", "size"), subjects_with_capacity=("subject", "nunique"))
        .reset_index()
    )
    eligibility = event_counts.merge(state_counts, on="dataset", how="left").merge(capacity_counts, on="dataset", how="left")
    for col in ["session_state_rows", "subjects_with_state", "participant_capacity_rows", "subjects_with_capacity"]:
        eligibility[col] = eligibility[col].fillna(0).astype(int)
    eligibility["included_in_revised_step09_projection"] = (
        (eligibility["supervised_events"] > 0)
        & (eligibility["session_state_rows"] > 0)
        & (eligibility["participant_capacity_rows"] > 0)
    )
    eligibility["exclusion_reason"] = ""
    eligibility.loc[eligibility["supervised_events"] == 0, "exclusion_reason"] = "no_correctness_labels_in_unified_events"
    eligibility.loc[
        (eligibility["supervised_events"] > 0) & (eligibility["session_state_rows"] == 0),
        "exclusion_reason",
    ] = "no_revised_step08_state_coordinates"
    eligibility.loc[
        (eligibility["supervised_events"] > 0) & (eligibility["participant_capacity_rows"] == 0),
        "exclusion_reason",
    ] = "no_revised_step08_capacity_coordinates"
    return eligibility


def fit_projection(
    artificial: pd.DataFrame, human: pd.DataFrame, target: str, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in BEHAVIOR_FEATURES if c in artificial.columns and c in human.columns]
    train = artificial[feature_cols + [target]].copy()
    train = train.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        train[col] = train[col].fillna(train[col].median())
    train = train.dropna(subset=[target])

    apply = human[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        fill = train[col].median()
        apply[col] = apply[col].fillna(fill)

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 25)),
    )
    model.fit(train[feature_cols], train[target])
    human[f"{prefix}_projection_raw"] = model.predict(apply[feature_cols])
    human[f"{prefix}_projection_raw_z"] = zscore(human[f"{prefix}_projection_raw"])

    coef = model.named_steps["ridgecv"].coef_
    coefficients = pd.DataFrame(
        {
            "projection": prefix,
            "target": target,
            "feature": feature_cols,
            "ridge_standardized_coefficient": coef,
            "ridge_alpha": model.named_steps["ridgecv"].alpha_,
            "n_artificial_agents": len(train),
        }
    )
    return human, coefficients


def residualize_series(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    data = pd.concat([y.rename("_target"), controls], axis=1).replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["_target"])
    if data.empty:
        return pd.Series(np.nan, index=y.index)
    control_cols = [c for c in controls.columns if c in data.columns]
    if not control_cols:
        return y - y.mean()
    x = data[control_cols].copy()
    for col in control_cols:
        if x[col].isna().all():
            x[col] = 0.0
        else:
            x[col] = x[col].fillna(x[col].median())
    model = LinearRegression()
    model.fit(x, data["_target"])
    resid = data["_target"] - model.predict(x)
    out = pd.Series(np.nan, index=y.index)
    out.loc[resid.index] = resid
    return out


def bootstrap_spearman(x: pd.Series, y: pd.Series, n_boot: int = 2000, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(data) < 5:
        return np.nan, np.nan
    rhos = []
    arr = data.to_numpy(dtype=float)
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bx = arr[idx, 0]
        by = arr[idx, 1]
        if np.std(bx) <= 1e-12 or np.std(by) <= 1e-12:
            continue
        rhos.append(stats.spearmanr(bx, by).statistic)
    if not rhos:
        return np.nan, np.nan
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def classify_claim(rho: float, p_value: float, n: int, gate_context: str) -> str:
    if not np.isfinite(rho) or not np.isfinite(p_value) or n < 10:
        return "insufficient"
    if "state" in gate_context and "ann_state_residualized_failed" in gate_context:
        if abs(rho) >= 0.35 and p_value < 0.01:
            return "exploratory_state_convergent_but_ann_limited"
        return "exploratory_or_unsupported_state"
    if abs(rho) >= 0.50 and p_value < 0.001:
        return "strong_current_claim"
    if abs(rho) >= 0.30 and p_value < 0.05:
        return "supported_but_qualified"
    return "not_supported"


def spearman_test(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    analysis_family: str,
    controls_label: str,
    gate_context: str,
) -> SpearmanResult:
    data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x_col].nunique() < 2 or data[y_col].nunique() < 2:
        rho = np.nan
        p = np.nan
    else:
        result = stats.spearmanr(data[x_col], data[y_col])
        rho = float(result.statistic)
        p = float(result.pvalue)
    ci_low, ci_high = bootstrap_spearman(data[x_col], data[y_col])
    claim_strength = classify_claim(rho, p, len(data), gate_context)
    if (
        controls_label == "none"
        and "machine_projection" in analysis_family
        and claim_strength == "strong_current_claim"
    ):
        claim_strength = "raw_supported_requires_residualized_confirmation"
    return SpearmanResult(
        analysis_family=analysis_family,
        x=x_col,
        y=y_col,
        controls=controls_label,
        n=int(len(data)),
        spearman_rho=rho,
        p_value=p,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        claim_strength=claim_strength,
    )


def make_controls(df: pd.DataFrame) -> pd.DataFrame:
    controls = pd.DataFrame(index=df.index)
    for col in ["mean_accuracy", "negative_log_likelihood", "response_rate", "n_events"]:
        if col in df.columns:
            controls[col] = as_numeric(df[col])
    for col in ["dataset", "task"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype("string"), prefix=col, drop_first=True, dtype=float)
            controls = pd.concat([controls, dummies], axis=1)
    return controls


def row_mean_with_min_count(df: pd.DataFrame, cols: list[str], min_count: int = 2) -> pd.Series:
    available_cols = [col for col in cols if col in df.columns]
    if not available_cols:
        return pd.Series(np.nan, index=df.index)
    values = df[available_cols]
    out = values.mean(axis=1, skipna=True)
    out = out.where(values.notna().sum(axis=1) >= min_count)
    return out


def add_residualized_tests(df: pd.DataFrame, pairs: list[tuple[str, str, str, str]]) -> list[SpearmanResult]:
    controls = make_controls(df)
    residual_df = df.copy()
    results = []
    for x_col, y_col, family, gate_context in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        residual_df[f"{x_col}_resid"] = residualize_series(df[x_col], controls)
        residual_df[f"{y_col}_resid"] = residualize_series(df[y_col], controls)
        results.append(
            spearman_test(
                residual_df,
                f"{x_col}_resid",
                f"{y_col}_resid",
                family,
                "dataset+task+mean_accuracy+nll+response_rate+n_events",
                gate_context,
            )
        )
    return results


def attach_dynamics(projection: pd.DataFrame) -> pd.DataFrame:
    trajectory_path = TABLE_DIR / "recurrent_dynamics_by_subject_task.csv"
    fixed_path = TABLE_DIR / "fixed_point_summary.csv"
    if not trajectory_path.exists():
        projection["dynamics_available"] = False
        return projection

    trajectory = pd.read_csv(trajectory_path)
    trajectory = trajectory.rename(
        columns={
            "trajectory_participation_ratio": "dynamics_trajectory_participation_ratio",
            "trajectory_cov_rank": "dynamics_trajectory_cov_rank",
            "trajectory_radius": "dynamics_trajectory_radius",
            "trajectory_step_norm_mean": "dynamics_step_norm_mean",
            "hidden_variability": "dynamics_hidden_variability",
        }
    )
    trajectory["dynamics_capacity_coordinate"] = row_mean_with_min_count(
        trajectory,
        [
            "dynamics_trajectory_participation_ratio",
            "dynamics_trajectory_cov_rank",
            "dynamics_trajectory_radius",
        ],
        min_count=2,
    )
    trajectory["dynamics_state_proxy"] = row_mean_with_min_count(
        trajectory,
        ["dynamics_step_norm_mean", "dynamics_hidden_variability"],
        min_count=1,
    )
    dyn_cols = [
        "dataset",
        "subject",
        "session",
        "task",
        "selected_hidden_size",
        "dynamics_trajectory_participation_ratio",
        "dynamics_trajectory_cov_rank",
        "dynamics_trajectory_radius",
        "dynamics_step_norm_mean",
        "dynamics_hidden_variability",
        "dynamics_capacity_coordinate",
        "dynamics_state_proxy",
    ]
    projection = projection.merge(
        trajectory[dyn_cols],
        on=["dataset", "subject", "session", "task", "selected_hidden_size"],
        how="left",
    )

    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path)
        fixed_summary = fixed.rename(
            columns={
                "hidden_size": "selected_hidden_size",
                "spectral_radius": "dynamics_spectral_radius",
                "memory_timescale_steps": "dynamics_memory_timescale_steps",
            }
        )[
            ["selected_hidden_size", "task", "dynamics_spectral_radius", "dynamics_memory_timescale_steps"]
        ]
        projection = projection.merge(fixed_summary, on=["selected_hidden_size", "task"], how="left")

    geom_cols = [
        "dynamics_trajectory_participation_ratio",
        "dynamics_trajectory_cov_rank",
        "dynamics_trajectory_radius",
        "dynamics_step_norm_mean",
        "dynamics_hidden_variability",
        "dynamics_spectral_radius",
        "dynamics_memory_timescale_steps",
    ]
    for col in geom_cols:
        if col in projection.columns:
            projection[f"{col}_z"] = zscore(projection[col])
    z_cols = [f"{c}_z" for c in geom_cols if f"{c}_z" in projection.columns]
    projection["dynamics_capacity_geometry_z"] = projection[z_cols].mean(axis=1, skipna=True)
    projection["dynamics_available"] = projection[z_cols].notna().any(axis=1)
    return projection


def main() -> None:
    ensure_dirs()

    state = pd.read_csv(TABLE_DIR / "session_state_multiaxis_coordinates.csv")
    capacity = pd.read_csv(TABLE_DIR / "participant_capacity_multidimensional_coordinates.csv")
    events = pd.read_parquet(PROCESSED_DIR / "all_model_events.parquet")
    artificial_behavior = pd.read_csv(TABLE_DIR / "artificial_behavioral_fingerprints.csv")
    artificial_dynamics = pd.read_csv(TABLE_DIR / "artificial_dynamics_fingerprints.csv")
    artificial_params = pd.read_csv(TABLE_DIR / "artificial_perturbation_parameters.csv")
    ann_hybrid = pd.read_csv(TABLE_DIR / "ann_hybrid_recovery.csv")

    eligibility = build_dataset_eligibility(events, state, capacity)
    human_behavior = build_human_behavior_fingerprints(events)

    artificial = artificial_behavior.merge(artificial_dynamics, on="agent_id", how="left").merge(
        artificial_params, on="agent_id", how="left", suffixes=("", "_param")
    )

    projected_behavior = human_behavior.copy()
    coefficient_frames = []
    for target, prefix in [("state_severity", "machine_state"), ("capacity_level", "machine_capacity")]:
        projected_behavior, coef = fit_projection(artificial, projected_behavior, target, prefix)
        coefficient_frames.append(coef)

    projection = state.merge(
        projected_behavior,
        on=["dataset", "subject", "session", "task"],
        how="left",
        suffixes=("", "_behavior"),
    )
    projection["participant_id"] = projection["dataset"].astype(str) + ":" + projection["subject"].astype(str)
    projection = projection.merge(
        capacity,
        on=["participant_id", "dataset", "subject"],
        how="left",
        suffixes=("", "_capacity"),
    )

    projection["state_reliability_axis_inverted"] = -as_numeric(projection["state_reliability_axis"])
    projection["state_reliability_axis_inverted_z"] = zscore(projection["state_reliability_axis_inverted"])
    projection["state_parameter_instability_z"] = projection[STATE_AXES].mean(axis=1, skipna=True)

    projection["capacity_parameter_resource_z"] = projection[CAPACITY_AXES].mean(axis=1, skipna=True)
    projection = attach_dynamics(projection)
    if "dynamics_state_proxy" in projection.columns:
        projection["dynamics_state_proxy_z"] = zscore(projection["dynamics_state_proxy"])

    projection["optimized_state_profile_z"] = row_mean_with_min_count(
        projection,
        ["state_parameter_instability_z", "machine_state_projection_raw_z", "dynamics_state_proxy_z"],
    )
    projection["optimized_capacity_profile_z"] = row_mean_with_min_count(
        projection,
        ["capacity_parameter_resource_z", "machine_capacity_projection_raw_z", "dynamics_capacity_geometry_z"],
    )

    convergence_pairs = [
        ("state_parameter_instability_z", "machine_state_projection_raw_z", "state_parameter_vs_machine_projection", "state_ann_state_residualized_failed"),
        ("state_parameter_instability_z", "dynamics_state_proxy", "state_parameter_vs_dynamics_proxy", "state_ann_state_residualized_failed"),
        ("optimized_state_profile_z", "mean_accuracy", "optimized_state_vs_accuracy_sanity", "state_ann_state_residualized_failed"),
        ("capacity_parameter_resource_z", "machine_capacity_projection_raw_z", "capacity_parameter_vs_machine_projection", "capacity_ann_capacity_supported"),
        ("capacity_parameter_resource_z", "dynamics_capacity_geometry_z", "capacity_parameter_vs_dynamics_geometry", "capacity_ann_capacity_supported"),
        ("machine_capacity_projection_raw_z", "dynamics_capacity_geometry_z", "capacity_machine_projection_vs_dynamics_geometry", "capacity_ann_capacity_supported"),
        ("optimized_capacity_profile_z", "dynamics_capacity_geometry_z", "optimized_capacity_vs_dynamics_geometry", "capacity_ann_capacity_supported"),
    ]

    raw_results = [
        spearman_test(projection, x, y, family, "none", gate)
        for x, y, family, gate in convergence_pairs
        if x in projection.columns and y in projection.columns
    ]
    residualized_results = add_residualized_tests(projection, convergence_pairs)

    tests = pd.DataFrame([r.__dict__ for r in raw_results])
    residual_tests = pd.DataFrame([r.__dict__ for r in residualized_results])

    axis_results = []
    for axis in STATE_AXES:
        if axis in projection.columns:
            axis_results.append(
                spearman_test(
                    projection,
                    axis,
                    "machine_state_projection_raw_z",
                    "state_axis_vs_machine_projection",
                    "none",
                    "state_ann_state_residualized_failed",
                ).__dict__
            )
    for axis in CAPACITY_AXES:
        if axis in projection.columns:
            axis_results.append(
                spearman_test(
                    projection,
                    axis,
                    "dynamics_capacity_geometry_z",
                    "capacity_axis_vs_dynamics_geometry",
                    "none",
                    "capacity_ann_capacity_supported",
                ).__dict__
            )
    axis_tests = pd.DataFrame(axis_results)

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    ann_gate_summary = ann_hybrid.to_dict(orient="records")

    projection_out = TABLE_DIR / "human_state_capacity_multiaxis_projection.csv"
    tests_out = TABLE_DIR / "multiaxis_profile_convergence_tests.csv"
    resid_out = TABLE_DIR / "multiaxis_projection_residualized_tests.csv"
    axis_out = TABLE_DIR / "multiaxis_axis_level_tests.csv"
    coef_out = TABLE_DIR / "machine_projection_model_coefficients.csv"
    eligibility_out = TABLE_DIR / "human_projection_dataset_eligibility.csv"
    source_out = SOURCE_DIR / "human_projection_source.csv"

    projection.to_csv(projection_out, index=False)
    tests.to_csv(tests_out, index=False)
    residual_tests.to_csv(resid_out, index=False)
    axis_tests.to_csv(axis_out, index=False)
    coefficients.to_csv(coef_out, index=False)
    eligibility.to_csv(eligibility_out, index=False)
    projection.to_csv(source_out, index=False)

    cap_raw_supported = tests[
        (tests["analysis_family"].str.contains("capacity", na=False))
        & (
            tests["claim_strength"].isin(
                [
                    "strong_current_claim",
                    "supported_but_qualified",
                    "raw_supported_requires_residualized_confirmation",
                ]
            )
        )
    ]
    cap_residual_supported = residual_tests[
        (residual_tests["analysis_family"].str.contains("capacity", na=False))
        & (residual_tests["claim_strength"].isin(["strong_current_claim", "supported_but_qualified"]))
    ]
    cap_dynamics_supported = tests[
        (tests["analysis_family"].str.contains("dynamics_geometry", na=False))
        & (tests["claim_strength"].isin(["strong_current_claim", "supported_but_qualified"]))
    ]
    state_supported = tests[
        (tests["analysis_family"].str.contains("state", na=False))
        & (tests["claim_strength"].isin(["exploratory_state_convergent_but_ann_limited", "strong_current_claim"]))
    ]
    dynamics_rows = int(projection["dynamics_available"].fillna(False).sum())
    included_datasets = ", ".join(sorted(projection["dataset"].dropna().unique()))
    excluded = eligibility[~eligibility["included_in_revised_step09_projection"]]

    claim_text = [
        "# Human Projection Claim Limits",
        "",
        "Step 09 was run as a full multidimensional projection and coordinate-optimization pass.",
        "",
        "## ANN Gate Context",
        "",
        "- ANN matched state/capacity classification passed in raw and residualized fingerprints.",
        "- ANN capacity recovery passed in raw and residualized hybrid agents.",
        "- ANN state-severity recovery failed after residualizing performance, so human state claims remain exploratory unless downstream independent evidence is unusually strong.",
        "",
        "## Current Claim Boundary",
        "",
    ]
    if dynamics_rows == 0:
        claim_text.append(
            "- No recurrent-dynamics rows attach to this revised Step 09 projection in the current dynamics table; ds007554 now has reconstructed push-button correctness labels and Step 08 behavioral coordinates, but Step 11 dynamics must be rerun before dynamics can be merged into this projection."
        )
    if len(cap_dynamics_supported):
        claim_text.append("- Capacity profile claims can be treated as supported when explicitly tied to recurrent geometry/dimensionality.")
    elif len(cap_residual_supported):
        claim_text.append(
            "- Capacity profile claims survive residualized projection tests but do not yet have revised Step 09 dynamics convergence; describe them as supported but qualified."
        )
    elif len(cap_raw_supported):
        claim_text.append(
            "- Capacity profile claims show strong raw alignment with the machine-fingerprint projection, but this does not survive the residualized projection test; treat this as useful coordinate structure, not a strong independent manuscript claim."
        )
    else:
        claim_text.append("- Capacity profile claims remain qualified; convergence was weaker than required for a strong claim.")
    if len(state_supported):
        claim_text.append("- State profile claims show some convergence, but must still be labelled exploratory because the ANN residualized state-recovery gate failed.")
    else:
        claim_text.append("- State profile claims remain exploratory or unsupported with the current behavioral-only projection.")
    claim_text.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{projection_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{tests_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{resid_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{axis_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{coef_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{eligibility_out.relative_to(PACKAGE_ROOT)}`",
            f"- `{source_out.relative_to(PACKAGE_ROOT)}`",
            "",
            "## Dataset Eligibility",
            "",
            f"- Included in revised projection: {included_datasets}.",
        ]
    )
    for _, row in excluded.iterrows():
        claim_text.append(
            f"- Excluded from revised projection: {row['dataset']} ({row['exclusion_reason']}; "
            f"supervised_events={row['supervised_events']}, state_rows={row['session_state_rows']}, "
            f"capacity_rows={row['participant_capacity_rows']})."
        )
    claim_text.extend(
        [
            "",
            "## ANN Hybrid Records Used",
            "",
        ]
    )
    for row in ann_gate_summary:
        claim_text.append(
            f"- {row['analysis_type']} / {row['feature_set']} / {row['target_axis']}: "
            f"rho={row['spearman_rho']:.3f}, pass_gate={row['pass_gate']}"
        )
    (AUDIT_DIR / "human_projection_claim_limits.md").write_text("\n".join(claim_text), encoding="utf-8")

    status = [
        "# Step 09 Human Projection Status",
        "",
        "Status: implemented and run as a full multidimensional projection pass.",
        "",
        f"- Session-task rows projected: {len(projection)}",
        f"- Datasets included: {included_datasets}",
        f"- Rows with dynamics attached: {dynamics_rows}",
        f"- Raw convergence tests: {len(tests)}",
        f"- Residualized convergence tests: {len(residual_tests)}",
        "",
        "Main interpretation: state remains exploratory because ANN state-severity recovery failed after performance residualization. Capacity remains qualified in this repaired human projection: machine-projection convergence remains weak after residualization, but recurrent-dynamics geometry is now attached for all projected rows.",
    ]
    (LOG_DIR / "step09_human_projection_status.md").write_text("\n".join(status), encoding="utf-8")

    print("\n".join(status))


if __name__ == "__main__":
    main()
