from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"

RNG_SEED = 20260606
N_PERMUTATIONS = 5000
N_BOOTSTRAPS = 2000


@dataclass
class ModelSpec:
    model_name: str
    predictors: list[str]
    description: str
    claim_role: str


MODEL_SPECS = [
    ModelSpec("intercept_only", [], "Global mean accuracy only.", "baseline"),
    ModelSpec("task_dataset", ["dataset", "task"], "Dataset and task fixed effects.", "required_baseline"),
    ModelSpec(
        "one_dimensional_state_impairment",
        ["dataset", "task", "optimized_state_profile_z"],
        "Task/dataset plus optimized state profile.",
        "state_test",
    ),
    ModelSpec(
        "capacity_parameter",
        ["dataset", "task", "capacity_parameter_resource_z"],
        "Task/dataset plus multidimensional capacity parameter profile.",
        "capacity_test",
    ),
    ModelSpec(
        "additive_state_capacity",
        ["dataset", "task", "state_parameter_instability_z", "capacity_parameter_resource_z"],
        "Task/dataset plus separate state and capacity profiles.",
        "primary_coordinate_test",
    ),
    ModelSpec(
        "state_capacity_interaction",
        ["dataset", "task", "state_parameter_instability_z", "capacity_parameter_resource_z", "state_capacity_interaction"],
        "Additive state/capacity model plus their interaction.",
        "interaction_test",
    ),
    ModelSpec(
        "machine_projection_additive",
        ["dataset", "task", "machine_state_projection_raw_z", "machine_capacity_projection_raw_z"],
        "Task/dataset plus machine-projected state and capacity axes.",
        "projection_test",
    ),
    ModelSpec(
        "behavioral_descriptive",
        ["dataset", "task", "response_rate", "negative_log_likelihood", "n_events"],
        "Descriptive behavioral summaries; negative log likelihood is derived from observed accuracy.",
        "descriptive_sanity_check",
    ),
    ModelSpec(
        "random_axis_control",
        ["dataset", "task", "random_axis"],
        "Task/dataset plus a fixed random axis.",
        "negative_control",
    ),
    ModelSpec(
        "shuffled_coordinate_control",
        ["dataset", "task", "shuffled_state", "shuffled_capacity"],
        "Task/dataset plus shuffled state/capacity coordinates.",
        "negative_control",
    ),
]


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (x - x.mean()) / sd


def prepare_analysis_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    projection_path = TABLES / "human_state_capacity_multiaxis_projection.csv"
    events_path = PROCESSED / "all_model_events.parquet"
    if not projection_path.exists():
        raise FileNotFoundError("Step 10 requires Step 09 output human_state_capacity_multiaxis_projection.csv")
    if not events_path.exists():
        raise FileNotFoundError("Step 10 requires Step 03 output all_model_events.parquet")

    projection = pd.read_csv(projection_path)
    events = pd.read_parquet(events_path)
    included = events[events["event_included"].fillna(False).astype(bool)].copy()
    ds_eligibility = (
        included.groupby("dataset", dropna=False)
        .agg(
            included_events=("dataset", "size"),
            supervised_events=("correct", lambda s: int(s.notna().sum())),
            subjects=("subject", "nunique"),
            tasks=("task", "nunique"),
        )
        .reset_index()
    )
    coord_counts = (
        projection.groupby("dataset", dropna=False)
        .agg(projection_rows=("dataset", "size"), projection_subjects=("subject", "nunique"))
        .reset_index()
    )
    ds_eligibility = ds_eligibility.merge(coord_counts, on="dataset", how="left")
    ds_eligibility[["projection_rows", "projection_subjects"]] = ds_eligibility[
        ["projection_rows", "projection_subjects"]
    ].fillna(0).astype(int)
    ds_eligibility["discovery_eligible"] = (
        (ds_eligibility["supervised_events"] > 0) & (ds_eligibility["projection_rows"] > 0)
    )
    ds_eligibility["exclusion_reason"] = ""
    ds_eligibility.loc[ds_eligibility["supervised_events"] == 0, "exclusion_reason"] = (
        "no_correctness_labels_in_unified_events"
    )
    ds_eligibility.loc[
        (ds_eligibility["supervised_events"] > 0) & (ds_eligibility["projection_rows"] == 0),
        "exclusion_reason",
    ] = "no_step09_projection_coordinates"

    analysis = projection[projection["mean_accuracy"].notna()].copy()
    analysis["participant_id"] = analysis["dataset"].astype(str) + ":" + analysis["subject"].astype(str)
    analysis["state_capacity_interaction"] = (
        analysis["state_parameter_instability_z"] * analysis["capacity_parameter_resource_z"]
    )
    rng = np.random.default_rng(RNG_SEED)
    analysis["random_axis"] = rng.normal(size=len(analysis))
    analysis["shuffled_state"] = rng.permutation(analysis["state_parameter_instability_z"].fillna(0).to_numpy())
    analysis["shuffled_capacity"] = rng.permutation(analysis["capacity_parameter_resource_z"].fillna(0).to_numpy())
    return analysis, ds_eligibility


def make_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    if not predictors:
        return pd.DataFrame({"intercept_feature": np.ones(len(df))}, index=df.index)
    pieces = []
    for predictor in predictors:
        if predictor in {"dataset", "task"}:
            pieces.append(pd.get_dummies(df[predictor].astype("string"), prefix=predictor, drop_first=True, dtype=float))
        else:
            pieces.append(pd.to_numeric(df[predictor], errors="coerce").rename(predictor).to_frame())
    design = pd.concat(pieces, axis=1)
    for col in design.columns:
        if design[col].isna().all():
            design[col] = 0.0
        else:
            design[col] = design[col].fillna(design[col].median())
    if design.shape[1] == 0:
        design["intercept_feature"] = 1.0
    return design.astype(float)


def fit_predict_lopo(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    design = make_design(df, spec.predictors)
    y = pd.to_numeric(df["mean_accuracy"], errors="coerce").to_numpy(dtype=float)
    predictions = np.full(len(df), np.nan, dtype=float)
    participants = df["participant_id"].astype(str).to_numpy()
    unique_participants = np.unique(participants)
    x = design.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    for participant in unique_participants:
        val_mask = participants == participant
        train_mask = ~val_mask
        if train_mask.sum() < 10 or val_mask.sum() == 0:
            continue
        beta, *_ = np.linalg.lstsq(x[train_mask], y[train_mask], rcond=None)
        predictions[val_mask] = x[val_mask] @ beta
    out = df[["dataset", "subject", "session", "task", "participant_id", "mean_accuracy", "n_events"]].copy()
    out["model_name"] = spec.model_name
    out["prediction"] = np.clip(predictions, 0.0, 1.0)
    out["squared_error"] = (out["mean_accuracy"] - out["prediction"]) ** 2
    out["absolute_error"] = (out["mean_accuracy"] - out["prediction"]).abs()
    return out


def summarize_predictions(predictions: pd.DataFrame, spec: ModelSpec, scope_name: str) -> dict[str, object]:
    valid = predictions.dropna(subset=["prediction", "mean_accuracy"]).copy()
    y = valid["mean_accuracy"].to_numpy(dtype=float)
    pred = valid["prediction"].to_numpy(dtype=float)
    weights = pd.to_numeric(valid["n_events"], errors="coerce").fillna(1).to_numpy(dtype=float)
    weights = np.clip(weights, 1.0, None)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    weighted_rmse = float(np.sqrt(np.average((y - pred) ** 2, weights=weights)))
    return {
        "analysis_scope": scope_name,
        "model_name": spec.model_name,
        "description": spec.description,
        "claim_role": spec.claim_role,
        "n_rows": int(len(valid)),
        "n_participants": int(valid["participant_id"].nunique()),
        "n_datasets": int(valid["dataset"].nunique()),
        "lopo_rmse": rmse,
        "lopo_weighted_rmse": weighted_rmse,
        "lopo_mae": float(mean_absolute_error(y, pred)),
        "lopo_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "weighted_sse": float(np.sum(weights * (y - pred) ** 2)),
        "mean_observed_accuracy": float(np.mean(y)),
    }


def full_sample_sse(df: pd.DataFrame, predictors: list[str], y_col: str = "mean_accuracy") -> float:
    design = make_design(df, predictors)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    return design_sse(design, y)


def design_sse(design: pd.DataFrame, y: np.ndarray) -> float:
    x = design.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    return float(np.sum((y - pred) ** 2))


def permutation_test_gain(
    df: pd.DataFrame,
    reduced: list[str],
    full: list[str],
    shuffled_cols: list[str],
    label: str,
    scope_name: str,
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + len(label))
    y = pd.to_numeric(df["mean_accuracy"], errors="coerce").to_numpy(dtype=float)
    reduced_design = make_design(df, reduced)
    full_design = make_design(df, full)
    reduced_sse = design_sse(reduced_design, y)
    observed_gain = reduced_sse - design_sse(full_design, y)
    null_gains = []
    for _ in range(n_permutations):
        perm_design = full_design.copy()
        for col in shuffled_cols:
            if col in perm_design.columns:
                perm_design[col] = rng.permutation(perm_design[col].to_numpy())
        null_gains.append(reduced_sse - design_sse(perm_design, y))
    null = np.asarray(null_gains)
    p_value = float((np.sum(null >= observed_gain) + 1) / (len(null) + 1))
    return {
        "analysis_scope": scope_name,
        "test_name": label,
        "reduced_predictors": ",".join(reduced),
        "full_predictors": ",".join(full),
        "shuffled_columns": ",".join(shuffled_cols),
        "observed_sse_gain": float(observed_gain),
        "permutation_p95": float(np.percentile(null, 95)),
        "empirical_p_value": p_value,
        "n_permutations": int(n_permutations),
        "pass_nominal": bool(p_value < 0.05 and observed_gain > np.percentile(null, 95)),
    }


def bootstrap_spearman(df: pd.DataFrame, x: str, scope_name: str, y: str = "mean_accuracy") -> dict[str, object]:
    data = df[[x, y, "participant_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 5 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {
            "analysis_scope": scope_name,
            "effect_name": f"{x}_vs_{y}",
            "n_rows": int(len(data)),
            "n_participants": int(data["participant_id"].nunique()) if len(data) else 0,
            "spearman_rho": np.nan,
            "p_value": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }
    rho, p_value = stats.spearmanr(data[x], data[y])
    rng = np.random.default_rng(RNG_SEED + len(x))
    participants = data["participant_id"].drop_duplicates().to_numpy()
    x_values = data[x].to_numpy(dtype=float)
    y_values = data[y].to_numpy(dtype=float)
    group_indices = {
        participant: np.flatnonzero(data["participant_id"].to_numpy() == participant)
        for participant in participants
    }
    boot = []
    for _ in range(N_BOOTSTRAPS):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        idx = np.concatenate([group_indices[participant] for participant in sampled])
        bx = x_values[idx]
        by = y_values[idx]
        if np.unique(bx).size < 2 or np.unique(by).size < 2:
            continue
        boot.append(stats.spearmanr(bx, by).statistic)
    return {
        "analysis_scope": scope_name,
        "effect_name": f"{x}_vs_{y}",
        "n_rows": int(len(data)),
        "n_participants": int(data["participant_id"].nunique()),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "bootstrap_ci_low": float(np.percentile(boot, 2.5)) if boot else np.nan,
        "bootstrap_ci_high": float(np.percentile(boot, 97.5)) if boot else np.nan,
    }


def make_figure(model_comparison: pd.DataFrame, figure_data: Path) -> None:
    plot_df = model_comparison[model_comparison["lopo_rmse"].notna()].copy()
    primary = plot_df[plot_df["analysis_scope"].eq("ds007554_primary_reconstructed")].copy()
    if not primary.empty:
        plot_df = primary
    plot_df = plot_df.sort_values("lopo_rmse")
    plot_df.to_csv(figure_data, index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = ["#4C78A8" if "control" not in name else "#9A9A9A" for name in plot_df["model_name"]]
    ax.barh(plot_df["model_name"], plot_df["lopo_rmse"], color=colors, height=0.72)
    ax.set_xlabel("Leave-one-participant-out RMSE")
    ax.set_ylabel("")
    ax.set_title("Step 10 repaired supervised discovery")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_ds007554_discovery.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    analysis, eligibility = prepare_analysis_dataset()
    eligibility.to_csv(TABLES / "ds007554_discovery_eligibility.csv", index=False)

    predictions = []
    summaries = []
    scopes: list[tuple[str, pd.DataFrame]] = []
    ds_analysis = analysis[analysis["dataset"].eq("ds007554")].copy()
    ds_eligible = bool(
        "ds007554" in set(eligibility["dataset"])
        and eligibility.loc[eligibility["dataset"].eq("ds007554"), "discovery_eligible"].iloc[0]
        and len(ds_analysis) >= 10
    )
    if ds_eligible:
        scopes.append(("ds007554_primary_reconstructed", ds_analysis))
    scopes.append(("all_supervised_datasets_context", analysis.copy()))

    for scope_name, scope_df in scopes:
        for spec in MODEL_SPECS:
            pred = fit_predict_lopo(scope_df, spec)
            pred["analysis_scope"] = scope_name
            predictions.append(pred)
            summaries.append(summarize_predictions(pred, spec, scope_name))

    model_comparison = pd.DataFrame(summaries)

    prediction_df = pd.concat(predictions, ignore_index=True)
    prediction_df.to_csv(TABLES / "ds007554_discovery_predictions.csv", index=False)

    permutation_rows = []
    bootstrap_rows = []
    for scope_name, scope_df in scopes:
        permutation_rows.extend(
            [
            permutation_test_gain(
                scope_df,
                ["dataset", "task"],
                ["dataset", "task", "state_parameter_instability_z", "capacity_parameter_resource_z"],
                ["state_parameter_instability_z", "capacity_parameter_resource_z"],
                "additive_state_capacity_vs_task_dataset",
                scope_name,
            ),
            permutation_test_gain(
                scope_df,
                ["dataset", "task", "state_parameter_instability_z", "capacity_parameter_resource_z"],
                ["dataset", "task", "state_parameter_instability_z", "capacity_parameter_resource_z", "state_capacity_interaction"],
                ["state_capacity_interaction"],
                "interaction_vs_additive_state_capacity",
                scope_name,
            ),
            permutation_test_gain(
                scope_df,
                ["dataset", "task"],
                ["dataset", "task", "machine_state_projection_raw_z", "machine_capacity_projection_raw_z"],
                ["machine_state_projection_raw_z", "machine_capacity_projection_raw_z"],
                "machine_projection_vs_task_dataset",
                scope_name,
            ),
            permutation_test_gain(
                scope_df,
                ["dataset", "task"],
                ["dataset", "task", "capacity_parameter_resource_z"],
                ["capacity_parameter_resource_z"],
                "capacity_parameter_vs_task_dataset",
                scope_name,
            ),
            ]
        )
        bootstrap_rows.extend(
            [
                bootstrap_spearman(scope_df, "state_parameter_instability_z", scope_name),
                bootstrap_spearman(scope_df, "capacity_parameter_resource_z", scope_name),
                bootstrap_spearman(scope_df, "machine_state_projection_raw_z", scope_name),
                bootstrap_spearman(scope_df, "machine_capacity_projection_raw_z", scope_name),
                bootstrap_spearman(scope_df, "optimized_state_profile_z", scope_name),
                bootstrap_spearman(scope_df, "optimized_capacity_profile_z", scope_name),
            ]
        )
    permutation_tests = pd.DataFrame(permutation_rows)
    bootstrap_effects = pd.DataFrame(bootstrap_rows)

    model_comparison.to_csv(TABLES / "ds007554_discovery_model_comparison.csv", index=False)
    permutation_tests.to_csv(TABLES / "ds007554_permutation_tests.csv", index=False)
    bootstrap_effects.to_csv(TABLES / "ds007554_bootstrap_effects.csv", index=False)
    analysis.to_csv(SOURCE_DATA / "figure_ds007554_discovery_source.csv", index=False)
    make_figure(model_comparison, SOURCE_DATA / "figure_ds007554_discovery_model_comparison_source.csv")

    primary_scope = "ds007554_primary_reconstructed" if ds_eligible else "all_supervised_datasets_context"
    best = model_comparison[model_comparison["analysis_scope"].eq(primary_scope)].sort_values("lopo_rmse").head(1)
    best_text = "none"
    if len(best):
        best_text = f"{best.iloc[0]['model_name']} (LOPO RMSE={best.iloc[0]['lopo_rmse']:.4f})"
    task_row = model_comparison[
        model_comparison["analysis_scope"].eq(primary_scope) & model_comparison["model_name"].eq("task_dataset")
    ]
    additive_row = model_comparison[
        model_comparison["analysis_scope"].eq(primary_scope) & model_comparison["model_name"].eq("additive_state_capacity")
    ]
    additive_vs_task = np.nan
    if len(task_row) and len(additive_row):
        additive_vs_task = float(additive_row.iloc[0]["lopo_rmse"] - task_row.iloc[0]["lopo_rmse"])

    audit_lines = [
        "# Step 10 Discovery Claim Audit",
        "",
        "Step 10 was run after repairing HBN correctness labels.",
        "",
        "## ds007554 Primary Dataset",
        "",
        f"- ds007554 supervised discovery eligible after push-button reconstruction: {ds_eligible}.",
        f"- ds007554 reconstructed rows in Step 10: {len(ds_analysis)}; participants: {ds_analysis['participant_id'].nunique() if len(ds_analysis) else 0}.",
        "- The ds007554 labels are reconstructed from response timing, so claims should say reconstructed behavioral correctness rather than source-provided trial correctness.",
        "",
        "## Contextual Supervised Discovery",
        "",
        f"- Included context datasets: {', '.join(sorted(analysis['dataset'].dropna().unique()))}.",
        f"- Context rows: {len(analysis)}; participants: {analysis['participant_id'].nunique()}.",
        f"- Best LOPO model: {best_text}.",
        f"- Additive state/capacity RMSE minus task/dataset RMSE in primary scope `{primary_scope}`: {additive_vs_task:.4f}.",
        "",
        "## Interpretation",
        "",
        "- If the coordinate models do not beat task/dataset and descriptive baselines, the correct interpretation is not simply low power.",
        "- The repaired sample is large enough to detect modest raw associations; the weak residualized results point to dataset/task heterogeneity and construct mismatch.",
        "- State remains exploratory. Capacity remains qualified until independent dynamics or neurophysiology alignment supports it.",
    ]
    (AUDIT / "step10_discovery_claim_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    status = {
        "status": "implemented_and_run",
        "primary_scope": primary_scope,
        "n_rows": int(len(ds_analysis) if ds_eligible else len(analysis)),
        "n_participants": int(ds_analysis["participant_id"].nunique() if ds_eligible else analysis["participant_id"].nunique()),
        "datasets": sorted(analysis["dataset"].dropna().unique().tolist()),
        "ds007554_discovery_eligible": ds_eligible,
        "analysis_scopes": [name for name, _ in scopes],
        "best_lopo_model": best_text,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstraps": N_BOOTSTRAPS,
    }
    (LOGS / "step10_discovery_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP10_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
