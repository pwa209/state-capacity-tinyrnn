from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openneuro" / "ds005508"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / sd


def bh_q(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    m = len(valid)
    q = valid.to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def spearman_row(df: pd.DataFrame, x: str, y: str, family: str, label: str) -> dict[str, Any]:
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data[x].nunique() < 2 or data[y].nunique() < 2:
        rho, p_value = np.nan, np.nan
    else:
        rho, p_value = stats.spearmanr(data[x], data[y])
    return {
        "analysis": label,
        "family": family,
        "x": x,
        "y": y,
        "n": int(len(data)),
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
    }


def ols_row(df: pd.DataFrame, x: str, y: str, covariates: list[str], family: str, label: str) -> dict[str, Any]:
    cols = [x, y] + covariates
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < len(covariates) + 8 or data[x].nunique() < 2:
        return {
            "analysis": label,
            "family": family,
            "x": x,
            "y": y,
            "n": int(len(data)),
            "estimate": np.nan,
            "std_error": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
        }
    yv = pd.to_numeric(data[y], errors="coerce").to_numpy(dtype=float)
    pieces = [pd.Series(1.0, index=data.index, name="intercept").to_frame(), pd.to_numeric(data[x], errors="coerce").rename(x).to_frame()]
    for cov in covariates:
        pieces.append(pd.to_numeric(data[cov], errors="coerce").rename(cov).to_frame())
    design = pd.concat(pieces, axis=1).fillna(0.0)
    xv = design.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(xv)
    if len(yv) <= rank + 1:
        return {
            "analysis": label,
            "family": family,
            "x": x,
            "y": y,
            "n": int(len(data)),
            "estimate": np.nan,
            "std_error": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
        }
    beta = np.linalg.pinv(xv) @ yv
    pred = xv @ beta
    resid = yv - pred
    df_resid = max(1, len(yv) - rank)
    sigma2 = float(np.sum(resid**2) / df_resid)
    cov = sigma2 * np.linalg.pinv(xv.T @ xv)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    t_value = float(beta[1] / se) if se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_value), df=df_resid)) if np.isfinite(t_value) else np.nan
    sst = float(np.sum((yv - yv.mean()) ** 2))
    return {
        "analysis": label,
        "family": family,
        "x": x,
        "y": y,
        "n": int(len(data)),
        "estimate": float(beta[1]),
        "std_error": se,
        "t_value": t_value,
        "p_value": p_value,
        "r_squared": float(1 - np.sum(resid**2) / sst) if sst > 1e-12 else np.nan,
    }


def load_inputs() -> dict[str, pd.DataFrame]:
    events = pd.read_parquet(PROCESSED / "hbn_model_events.parquet")
    state = pd.read_csv(TABLES / "session_state_multiaxis_coordinates.csv")
    capacity = pd.read_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv")
    projection = pd.read_csv(TABLES / "human_state_capacity_multiaxis_projection.csv")
    selection = pd.read_csv(TABLES / "model_selection_by_subject.csv")
    dynamics = pd.read_csv(TABLES / "recurrent_dynamics_by_subject_task.csv")
    participants = pd.read_csv(RAW / "participants.tsv", sep="\t")
    participants["subject"] = participants["participant_id"].astype(str).str.replace("^sub-", "", regex=True)
    return {
        "events": events[events["dataset"].eq("hbn_release_4")].copy(),
        "state": state[state["dataset"].eq("hbn_release_4")].copy(),
        "capacity": capacity[capacity["dataset"].eq("hbn_release_4")].copy(),
        "projection": projection[projection["dataset"].eq("hbn_release_4")].copy(),
        "selection": selection[selection["dataset"].eq("hbn_release_4")].copy(),
        "dynamics": dynamics[dynamics["dataset"].eq("hbn_release_4")].copy(),
        "participants": participants,
    }


def build_coverage(inputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = inputs["events"]
    included = events[events["event_included"].fillna(False).astype(bool)].copy()
    event_summary = pd.DataFrame(
        [
            {
                "dataset": "hbn_release_4",
                "n_subjects_events": int(events["subject"].nunique()),
                "n_subjects_included": int(included["subject"].nunique()),
                "n_events": int(len(events)),
                "n_included_events": int(len(included)),
                "n_supervised_events": int(included["correct"].notna().sum()),
                "n_tasks": int(included["task"].nunique()),
                "n_state_rows": int(len(inputs["state"])),
                "n_capacity_rows": int(len(inputs["capacity"])),
                "n_projection_rows": int(len(inputs["projection"])),
                "n_dynamics_rows": int(len(inputs["dynamics"])),
            }
        ]
    )
    task_summary = (
        included.groupby("task", dropna=False)
        .agg(
            n_subjects=("subject", "nunique"),
            n_events=("dataset", "size"),
            n_supervised_events=("correct", lambda s: int(s.notna().sum())),
            mean_accuracy=("correct", "mean"),
            n_source_files=("source_file", "nunique"),
        )
        .reset_index()
        .sort_values(["n_subjects", "n_events"], ascending=False)
    )
    return event_summary, task_summary


def build_participant_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = inputs["events"]
    supervised = events[events["correct"].notna()].copy()
    supervised["correct_numeric"] = pd.to_numeric(supervised["correct"], errors="coerce")
    behavior = (
        supervised.groupby("subject", dropna=False)
        .agg(
            supervised_events=("correct_numeric", "size"),
            supervised_accuracy=("correct_numeric", "mean"),
            supervised_tasks=("task", "nunique"),
        )
        .reset_index()
    )
    nonsupervised = (
        events.groupby("subject", dropna=False)
        .agg(total_events=("dataset", "size"), total_tasks=("task", "nunique"), source_files=("source_file", "nunique"))
        .reset_index()
    )
    state = (
        inputs["state"].groupby(["subject"], dropna=False)
        .agg(
            state_rows=("dataset", "size"),
            state_mean=("state_multidimensional_summary_z", "mean"),
            state_sd=("state_multidimensional_summary_z", "std"),
            state_quality_high=("state_estimation_quality", lambda s: int((s == "high").sum())),
            state_ceiling_rows=("accuracy_ceiling_flag", "sum"),
        )
        .reset_index()
    )
    capacity = inputs["capacity"][
        [
            "subject",
            "selected_hidden_size",
            "capacity_multidimensional_summary_z",
            "capacity_selection_confidence",
            "capacity_cross_task_consistency_axis",
            "capacity_information_quality",
        ]
    ].copy()
    capacity["capacity_parameter_resource_z"] = capacity["capacity_multidimensional_summary_z"]
    selection = inputs["selection"][
        ["subject", "n_val_events", "val_nll", "val_accuracy", "nll_margin_to_next_best"]
    ].copy()
    dynamics = (
        inputs["dynamics"].groupby("subject", dropna=False)
        .agg(
            dynamics_rows=("dataset", "size"),
            trajectory_participation_ratio=("trajectory_participation_ratio", "mean"),
            trajectory_cov_rank=("trajectory_cov_rank", "mean"),
            trajectory_radius=("trajectory_radius", "mean"),
            trajectory_step_norm_mean=("trajectory_step_norm_mean", "mean"),
            hidden_variability=("hidden_variability", "mean"),
        )
        .reset_index()
    )
    participants = inputs["participants"][
        ["subject", "participant_id", "age", "sex", "p_factor", "attention", "internalizing", "externalizing", "full_pheno"]
    ].copy()
    table = participants.merge(nonsupervised, on="subject", how="left")
    for frame in [behavior, state, capacity, selection, dynamics]:
        table = table.merge(frame, on="subject", how="left")
    table["age_z"] = zscore(table["age"])
    table["capacity_z"] = pd.to_numeric(table["capacity_parameter_resource_z"], errors="coerce")
    table["state_z"] = pd.to_numeric(table["state_mean"], errors="coerce")
    return table


def build_state_task_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    state = inputs["state"].copy()
    capacity = inputs["capacity"][["subject", "capacity_multidimensional_summary_z"]].copy()
    capacity["capacity_parameter_resource_z"] = capacity["capacity_multidimensional_summary_z"]
    dynamics = inputs["dynamics"][
        [
            "subject",
            "session",
            "task",
            "trajectory_participation_ratio",
            "trajectory_cov_rank",
            "trajectory_radius",
            "trajectory_step_norm_mean",
            "hidden_variability",
            "accuracy",
        ]
    ].copy()
    out = state.merge(dynamics, on=["subject", "session", "task"], how="left", suffixes=("", "_dynamics"))
    out = out.merge(capacity[["subject", "capacity_parameter_resource_z"]], on="subject", how="left")
    out["state_parameter_instability_z"] = out["state_multidimensional_summary_z"]
    return out


def run_scalability_tests(participants: pd.DataFrame, task_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            spearman_row(participants, "capacity_parameter_resource_z", "trajectory_participation_ratio", "capacity_geometry", "participant_capacity_vs_pr"),
            spearman_row(participants, "capacity_parameter_resource_z", "trajectory_cov_rank", "capacity_geometry", "participant_capacity_vs_cov_rank"),
            spearman_row(participants, "capacity_parameter_resource_z", "trajectory_radius", "capacity_geometry", "participant_capacity_vs_radius"),
            spearman_row(participants, "capacity_parameter_resource_z", "val_accuracy", "model_performance", "participant_capacity_vs_val_accuracy"),
            spearman_row(participants, "capacity_parameter_resource_z", "val_nll", "model_performance", "participant_capacity_vs_val_nll"),
            spearman_row(participants, "state_mean", "supervised_accuracy", "state_behavior", "participant_state_vs_accuracy"),
            spearman_row(participants, "state_sd", "supervised_accuracy", "state_behavior", "participant_state_variability_vs_accuracy"),
            spearman_row(participants, "age", "capacity_parameter_resource_z", "developmental_exploratory", "age_vs_capacity"),
            spearman_row(participants, "age", "state_mean", "developmental_exploratory", "age_vs_state"),
            spearman_row(participants, "age", "val_accuracy", "developmental_exploratory", "age_vs_val_accuracy"),
            spearman_row(participants, "attention", "capacity_parameter_resource_z", "phenotype_exploratory", "attention_vs_capacity"),
            spearman_row(participants, "p_factor", "state_mean", "phenotype_exploratory", "pfactor_vs_state"),
        ]
    )
    rows.extend(
        [
            ols_row(task_table, "capacity_parameter_resource_z", "trajectory_cov_rank", ["accuracy"], "capacity_geometry_adjusted", "capacity_vs_cov_rank_adjust_accuracy"),
            ols_row(task_table, "capacity_parameter_resource_z", "trajectory_participation_ratio", ["accuracy"], "capacity_geometry_adjusted", "capacity_vs_pr_adjust_accuracy"),
            ols_row(task_table, "state_parameter_instability_z", "accuracy", ["trajectory_cov_rank"], "state_behavior_adjusted", "state_vs_accuracy_adjust_geometry"),
        ]
    )
    tests = pd.DataFrame(rows)
    tests["q_value"] = bh_q(tests["p_value"])
    tests["bh_significant_05"] = tests["q_value"] < 0.05
    return tests


def model_performance_tables(inputs: dict[str, pd.DataFrame], participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = inputs["selection"].copy()
    hidden_distribution = (
        selection.groupby("selected_hidden_size", dropna=False)
        .agg(n_subjects=("subject", "nunique"), median_val_nll=("val_nll", "median"), median_val_accuracy=("val_accuracy", "median"))
        .reset_index()
        .sort_values("selected_hidden_size")
    )
    model_perf = participants[
        [
            "subject",
            "participant_id",
            "age",
            "sex",
            "total_events",
            "supervised_events",
            "supervised_tasks",
            "selected_hidden_size",
            "val_nll",
            "val_accuracy",
            "nll_margin_to_next_best",
            "capacity_parameter_resource_z",
            "state_mean",
            "state_sd",
            "trajectory_cov_rank",
            "trajectory_participation_ratio",
        ]
    ].copy()
    return model_perf, hidden_distribution


def make_figure(
    task_summary: pd.DataFrame,
    hidden_distribution: pd.DataFrame,
    tests: pd.DataFrame,
    participant_table: pd.DataFrame,
) -> None:
    source_rows = []
    for _, row in task_summary.iterrows():
        rec = row.to_dict()
        rec["panel"] = "task_coverage"
        source_rows.append(rec)
    for _, row in hidden_distribution.iterrows():
        rec = row.to_dict()
        rec["panel"] = "hidden_size_distribution"
        source_rows.append(rec)
    for _, row in tests.iterrows():
        rec = row.to_dict()
        rec["panel"] = "scalability_tests"
        source_rows.append(rec)
    participant_table.to_csv(SOURCE_DATA / "hbn_participant_scalability_source.csv", index=False)
    pd.DataFrame(source_rows).to_csv(SOURCE_DATA / "figure_hbn_scalability_source.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.ravel()

    top_tasks = task_summary.sort_values("n_subjects", ascending=True).tail(10)
    axes[0].barh(top_tasks["task"], top_tasks["n_subjects"], color="#4C78A8")
    axes[0].set_xlabel("Subjects")
    axes[0].set_title("HBN task coverage")

    axes[1].bar(hidden_distribution["selected_hidden_size"].astype(str), hidden_distribution["n_subjects"], color="#F58518")
    axes[1].set_xlabel("Selected hidden size")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Capacity model selection")

    forest = tests[tests["family"].isin(["capacity_geometry", "capacity_geometry_adjusted", "developmental_exploratory"])].copy()
    forest["effect"] = forest["spearman_rho"].where(forest["spearman_rho"].notna(), forest["estimate"])
    forest = forest.dropna(subset=["effect"]).head(10)
    if not forest.empty:
        y = np.arange(len(forest))
        axes[2].scatter(forest["effect"], y, c=np.where(forest["bh_significant_05"], "#009E73", "#777777"), s=24)
        axes[2].axvline(0, color="#555555", linewidth=0.8)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(forest["analysis"])
    axes[2].set_xlabel("Effect")
    axes[2].set_title("Scalability effects")

    plot = participant_table.dropna(subset=["age", "capacity_parameter_resource_z"]).copy()
    if not plot.empty:
        axes[3].scatter(plot["age"], plot["capacity_parameter_resource_z"], s=10, alpha=0.55, color="#CC79A7", edgecolor="none")
        if plot["age"].nunique() > 2:
            slope, intercept, *_ = stats.linregress(plot["age"], plot["capacity_parameter_resource_z"])
            x = np.linspace(plot["age"].min(), plot["age"].max(), 50)
            axes[3].plot(x, intercept + slope * x, color="#222222", linewidth=1.0)
    axes[3].set_xlabel("Age (years)")
    axes[3].set_ylabel("Capacity profile z")
    axes[3].set_title("Developmental check")

    for ax in axes:
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_hbn_scalability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    inputs = load_inputs()
    event_summary, task_summary = build_coverage(inputs)
    participant_table = build_participant_table(inputs)
    state_task_table = build_state_task_table(inputs)
    tests = run_scalability_tests(participant_table, state_task_table)
    model_perf, hidden_distribution = model_performance_tables(inputs, participant_table)

    event_summary.to_csv(TABLES / "hbn_scalability_summary.csv", index=False)
    task_summary.to_csv(TABLES / "hbn_task_coverage.csv", index=False)
    participant_table.to_csv(TABLES / "hbn_participant_scalability.csv", index=False)
    state_task_table.to_csv(TABLES / "hbn_state_task_scalability.csv", index=False)
    model_perf.to_csv(TABLES / "hbn_model_performance.csv", index=False)
    hidden_distribution.to_csv(TABLES / "hbn_hidden_size_distribution.csv", index=False)
    tests.to_csv(TABLES / "hbn_scalability_tests.csv", index=False)
    make_figure(task_summary, hidden_distribution, tests, participant_table)

    cap_geom = tests[tests["analysis"].eq("participant_capacity_vs_cov_rank")].head(1)
    age_cap = tests[tests["analysis"].eq("age_vs_capacity")].head(1)
    summary = event_summary.iloc[0].to_dict()
    audit = [
        "# Step 15 HBN Scalability Audit",
        "",
        f"- HBN subjects in event table: {summary['n_subjects_events']}.",
        f"- Included events: {summary['n_included_events']}; supervised events: {summary['n_supervised_events']}.",
        f"- State rows: {summary['n_state_rows']}; capacity rows: {summary['n_capacity_rows']}; recurrent-dynamics rows: {summary['n_dynamics_rows']}.",
        f"- Tasks represented: {summary['n_tasks']}.",
        "",
        "## Interpretation Boundary",
        "",
        "- HBN is used as a scalability and developmental exploratory dataset.",
        "- It should not be written as direct proof of the adult N-back/PVT state-capacity theory because most HBN events are passive-video or EEG event markers, RT is absent in the unified table, and supervised correctness is task-limited.",
        "- Strong HBN claims are limited to whether the pipeline scales and whether capacity-like profiles align with model geometry across hundreds of participants.",
    ]
    if not cap_geom.empty:
        row = cap_geom.iloc[0]
        audit.append(f"- Capacity vs trajectory covariance rank: rho={row['spearman_rho']:.3g}, p={row['p_value']:.3g}, q={row['q_value']:.3g}.")
    if not age_cap.empty:
        row = age_cap.iloc[0]
        audit.append(f"- Exploratory age vs capacity: rho={row['spearman_rho']:.3g}, p={row['p_value']:.3g}, q={row['q_value']:.3g}.")
    (AUDIT / "step15_hbn_scalability_audit.md").write_text("\n".join(audit), encoding="utf-8")

    (SOURCE_DATA / "figure_hbn_scalability_data_dictionary.md").write_text(
        "\n".join(
            [
                "# Figure HBN scalability source data dictionary",
                "",
                "- `figure_hbn_scalability_source.csv`: panel-level source data for HBN task coverage, hidden-size distribution and scalability effects.",
                "- `hbn_participant_scalability_source.csv`: participant-level source data for age, model performance, state/capacity coordinates and recurrent geometry.",
                "- Key fields: `task`, `n_subjects`, `n_events`, `selected_hidden_size`, `n_subjects`, `analysis`, `spearman_rho`, `estimate`, `p_value`, `q_value`, `age`, `capacity_parameter_resource_z`.",
                "- Script: `state_capacity_tinyrnn/scripts/14_external_hbn/run_hbn_scalability.py`.",
            ]
        ),
        encoding="utf-8",
    )
    (SOURCE_DATA / "figure_hbn_scalability_script_used.txt").write_text(
        "state_capacity_tinyrnn/scripts/14_external_hbn/run_hbn_scalability.py",
        encoding="utf-8",
    )

    status = {
        "status": "implemented_and_run",
        "n_subjects_events": int(summary["n_subjects_events"]),
        "n_included_events": int(summary["n_included_events"]),
        "n_supervised_events": int(summary["n_supervised_events"]),
        "n_state_rows": int(summary["n_state_rows"]),
        "n_capacity_rows": int(summary["n_capacity_rows"]),
        "n_dynamics_rows": int(summary["n_dynamics_rows"]),
        "n_tests": int(len(tests)),
        "bh_significant_rows": int(tests["bh_significant_05"].fillna(False).sum()),
        "outputs": [
            "outputs/tables/hbn_scalability_summary.csv",
            "outputs/tables/hbn_model_performance.csv",
            "outputs/tables/hbn_scalability_tests.csv",
            "outputs/figures/figure_hbn_scalability.png",
            "outputs/source_data/figure_hbn_scalability_source.csv",
        ],
    }
    (LOGS / "step15_hbn_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP15_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
