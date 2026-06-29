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
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"
RNG_SEED = 20260610


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


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


def spearman(df: pd.DataFrame, x: str, y: str) -> tuple[int, float, float]:
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return int(len(data)), np.nan, np.nan
    rho, p_value = stats.spearmanr(data[x], data[y])
    return int(len(data)), float(rho), float(p_value)


def bootstrap_spearman(df: pd.DataFrame, x: str, y: str, group: str, n_boot: int = 2000) -> dict[str, float]:
    data = df[[x, y, group]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data[x].nunique() < 2:
        return {"bootstrap_median": np.nan, "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan}
    rng = np.random.default_rng(RNG_SEED + len(x) + len(y))
    groups = data[group].drop_duplicates().to_numpy()
    indices = {g: np.flatnonzero(data[group].to_numpy() == g) for g in groups}
    xs = data[x].to_numpy(dtype=float)
    ys = data[y].to_numpy(dtype=float)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([indices[g] for g in sampled])
        if np.unique(xs[idx]).size > 1 and np.unique(ys[idx]).size > 1:
            vals.append(stats.spearmanr(xs[idx], ys[idx]).statistic)
    return {
        "bootstrap_median": float(np.nanmedian(vals)) if vals else np.nan,
        "bootstrap_ci_low": float(np.nanpercentile(vals, 2.5)) if vals else np.nan,
        "bootstrap_ci_high": float(np.nanpercentile(vals, 97.5)) if vals else np.nan,
    }


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "claim_key": "",
        "construct": "",
        "robustness_family": "",
        "test_name": "",
        "scope": "",
        "n": np.nan,
        "effect": np.nan,
        "p_value": np.nan,
        "q_value": np.nan,
        "pass_robustness": False,
        "interpretation": "",
        "source_table": "",
    }
    base.update(kwargs)
    rows.append(base)


def leave_one_dataset_tests(projection: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("state_parameter_instability_z", "machine_state_projection_raw_z", "state_projection", "state"),
        ("capacity_parameter_resource_z", "dynamics_capacity_geometry_z", "capacity_geometry", "capacity"),
        ("capacity_parameter_resource_z", "machine_capacity_projection_raw_z", "capacity_projection", "capacity"),
        ("optimized_state_profile_z", "mean_accuracy", "state_accuracy", "state"),
        ("optimized_capacity_profile_z", "dynamics_capacity_geometry_z", "capacity_optimized_geometry", "capacity"),
    ]
    rows = []
    for x, y, claim, construct in pairs:
        n, rho, p_value = spearman(projection, x, y)
        add_row(
            rows,
            claim_key=claim,
            construct=construct,
            robustness_family="full_sample",
            test_name=f"{x}_vs_{y}",
            scope="all_datasets",
            n=n,
            effect=rho,
            p_value=p_value,
            pass_robustness=bool(np.isfinite(p_value) and p_value < 0.05),
            interpretation="full-sample association",
            source_table="human_state_capacity_multiaxis_projection.csv",
        )
        boot = bootstrap_spearman(projection, x, y, "participant_id")
        add_row(
            rows,
            claim_key=claim,
            construct=construct,
            robustness_family="participant_bootstrap",
            test_name=f"{x}_vs_{y}",
            scope="all_datasets",
            n=n,
            effect=boot["bootstrap_median"],
            p_value=np.nan,
            pass_robustness=bool(
                np.isfinite(boot["bootstrap_ci_low"])
                and np.isfinite(boot["bootstrap_ci_high"])
                and not (boot["bootstrap_ci_low"] <= 0 <= boot["bootstrap_ci_high"])
            ),
            interpretation=f"95% CI [{boot['bootstrap_ci_low']:.3g}, {boot['bootstrap_ci_high']:.3g}]",
            source_table="human_state_capacity_multiaxis_projection.csv",
            bootstrap_ci_low=boot["bootstrap_ci_low"],
            bootstrap_ci_high=boot["bootstrap_ci_high"],
        )
        for dataset in sorted(projection["dataset"].dropna().unique()):
            subset = projection[~projection["dataset"].eq(dataset)].copy()
            sn, srho, sp = spearman(subset, x, y)
            add_row(
                rows,
                claim_key=claim,
                construct=construct,
                robustness_family="leave_one_dataset_out",
                test_name=f"{x}_vs_{y}",
                scope=f"without_{dataset}",
                n=sn,
                effect=srho,
                p_value=sp,
                pass_robustness=bool(np.isfinite(sp) and sp < 0.05 and np.sign(srho) == np.sign(rho)),
                interpretation="passes if significant and same direction as full sample",
                source_table="human_state_capacity_multiaxis_projection.csv",
            )
        for task in sorted(projection["task"].dropna().astype(str).unique()):
            subset = projection[~projection["task"].astype(str).eq(task)].copy()
            if len(subset) < 30:
                continue
            sn, srho, sp = spearman(subset, x, y)
            add_row(
                rows,
                claim_key=claim,
                construct=construct,
                robustness_family="leave_one_task_out",
                test_name=f"{x}_vs_{y}",
                scope=f"without_{task}",
                n=sn,
                effect=srho,
                p_value=sp,
                pass_robustness=bool(np.isfinite(sp) and sp < 0.05 and np.sign(srho) == np.sign(rho)),
                interpretation="passes if significant and same direction as full sample",
                source_table="human_state_capacity_multiaxis_projection.csv",
            )
    return pd.DataFrame(rows)


def hidden_size_sensitivity() -> pd.DataFrame:
    metrics = pd.read_csv(TABLES / "subject_validation_metrics_by_hidden_size.csv")
    rows = []
    for dataset, group in metrics.groupby("dataset", dropna=False):
        for hidden, sub in group.groupby("hidden_size", dropna=False):
            add_row(
                rows,
                claim_key="hidden_size_sensitivity",
                construct="capacity",
                robustness_family="hidden_size_sensitivity",
                test_name="validation_nll_by_hidden_size",
                scope=str(dataset),
                n=int(len(sub)),
                effect=float(sub["val_nll"].median()),
                p_value=np.nan,
                pass_robustness=True,
                interpretation=f"median validation NLL for hidden_size={hidden}",
                source_table="subject_validation_metrics_by_hidden_size.csv",
                hidden_size=int(hidden),
                median_val_accuracy=float(sub["val_accuracy"].median()),
            )
    return pd.DataFrame(rows)


def baseline_and_control_tests() -> pd.DataFrame:
    rows = []
    discovery = pd.read_csv(TABLES / "ds007554_discovery_model_comparison.csv")
    for scope, group in discovery.groupby("analysis_scope", dropna=False):
        task = group[group["model_name"].eq("task_dataset")]
        additive = group[group["model_name"].eq("additive_state_capacity")]
        interaction = group[group["model_name"].eq("state_capacity_interaction")]
        descriptive = group[group["model_name"].eq("behavioral_descriptive")]
        random_axis = group[group["model_name"].eq("random_axis_control")]
        shuffled = group[group["model_name"].eq("shuffled_coordinate_control")]
        if not task.empty and not additive.empty:
            delta = float(additive.iloc[0]["lopo_rmse"] - task.iloc[0]["lopo_rmse"])
            add_row(
                rows,
                claim_key="additive_state_capacity_vs_task_baseline",
                construct="state_capacity",
                robustness_family="baseline_comparison",
                test_name="additive_rmse_minus_task_dataset_rmse",
                scope=scope,
                n=int(additive.iloc[0]["n_rows"]),
                effect=delta,
                pass_robustness=delta < 0,
                interpretation="negative values mean additive state/capacity improves over task/dataset baseline",
                source_table="ds007554_discovery_model_comparison.csv",
            )
        if not additive.empty and not descriptive.empty:
            delta = float(additive.iloc[0]["lopo_rmse"] - descriptive.iloc[0]["lopo_rmse"])
            add_row(
                rows,
                claim_key="descriptive_baseline_challenge",
                construct="state_capacity",
                robustness_family="baseline_comparison",
                test_name="additive_rmse_minus_descriptive_rmse",
                scope=scope,
                n=int(additive.iloc[0]["n_rows"]),
                effect=delta,
                pass_robustness=delta < 0,
                interpretation="positive values mean descriptive behavioral baseline outperforms coordinate model",
                source_table="ds007554_discovery_model_comparison.csv",
            )
        if not interaction.empty and not additive.empty:
            delta = float(interaction.iloc[0]["lopo_rmse"] - additive.iloc[0]["lopo_rmse"])
            add_row(
                rows,
                claim_key="state_capacity_interaction_prediction",
                construct="interaction",
                robustness_family="interaction_prediction",
                test_name="interaction_rmse_minus_additive_rmse",
                scope=scope,
                n=int(interaction.iloc[0]["n_rows"]),
                effect=delta,
                pass_robustness=delta < 0,
                interpretation="negative values mean interaction improves prediction over additive model",
                source_table="ds007554_discovery_model_comparison.csv",
            )
        for label, frame in [("random_axis_control", random_axis), ("shuffled_coordinate_control", shuffled)]:
            if not frame.empty and not additive.empty:
                delta = float(additive.iloc[0]["lopo_rmse"] - frame.iloc[0]["lopo_rmse"])
                add_row(
                    rows,
                    claim_key=label,
                    construct="control",
                    robustness_family="negative_control",
                    test_name="additive_rmse_minus_control_rmse",
                    scope=scope,
                    n=int(additive.iloc[0]["n_rows"]),
                    effect=delta,
                    pass_robustness=delta < 0,
                    interpretation="passes if coordinate model beats negative control",
                    source_table="ds007554_discovery_model_comparison.csv",
                )
    perms = pd.read_csv(TABLES / "ds007554_permutation_tests.csv")
    for _, row in perms.iterrows():
        add_row(
            rows,
            claim_key=row["test_name"],
            construct="interaction" if "interaction" in row["test_name"] else "state_capacity",
            robustness_family="permutation_control",
            test_name=row["test_name"],
            scope=row["analysis_scope"],
            n=np.nan,
            effect=float(row["observed_sse_gain"]),
            p_value=float(row["empirical_p_value"]),
            pass_robustness=bool(row["pass_nominal"]),
            interpretation="observed gain must exceed shuffled-coordinate null",
            source_table="ds007554_permutation_tests.csv",
            permutation_p95=float(row["permutation_p95"]),
        )
    return pd.DataFrame(rows)


def residualized_and_validation_tests() -> pd.DataFrame:
    rows = []
    residual = pd.read_csv(TABLES / "multiaxis_projection_residualized_tests.csv")
    for _, row in residual.iterrows():
        add_row(
            rows,
            claim_key=row["analysis_family"],
            construct="state" if "state" in row["analysis_family"] else "capacity",
            robustness_family="residualized_controls",
            test_name=f"{row['x']}_vs_{row['y']}",
            scope="all_datasets",
            n=int(row["n"]),
            effect=float(row["spearman_rho"]),
            p_value=float(row["p_value"]),
            pass_robustness=bool(row["p_value"] < 0.05),
            interpretation=str(row["claim_strength"]),
            source_table="multiaxis_projection_residualized_tests.csv",
            bootstrap_ci_low=row.get("bootstrap_ci_low", np.nan),
            bootstrap_ci_high=row.get("bootstrap_ci_high", np.nan),
        )
    for table, construct_hint in [
        ("cog_bci_validation_models.csv", "state_capacity"),
        ("tu_berlin_load_validation.csv", "state_capacity_interaction"),
        ("hbn_scalability_tests.csv", "state_capacity"),
    ]:
        df = pd.read_csv(TABLES / table)
        for _, row in df.iterrows():
            p = row.get("p_value", np.nan)
            q = row.get("q_value", np.nan)
            name = str(row.get("analysis_family", row.get("model_name", row.get("analysis", ""))))
            construct = "state" if "state" in name.lower() or str(row.get("x", "")).startswith("state") else construct_hint
            if "capacity" in name.lower() or str(row.get("x", "")).startswith("capacity"):
                construct = "capacity"
            if "pressure" in name.lower() or "load_x_capacity" in str(row.get("target_predictor", "")):
                construct = "interaction"
            add_row(
                rows,
                claim_key=name,
                construct=construct,
                robustness_family="external_validation",
                test_name=str(row.get("outcome", row.get("y", name))),
                scope=table.replace(".csv", ""),
                n=int(row.get("n", 0)) if pd.notna(row.get("n", np.nan)) else np.nan,
                effect=float(row.get("estimate", row.get("spearman_rho", np.nan))) if pd.notna(row.get("estimate", row.get("spearman_rho", np.nan))) else np.nan,
                p_value=float(p) if pd.notna(p) else np.nan,
                q_value=float(q) if pd.notna(q) else np.nan,
                pass_robustness=bool((pd.notna(q) and q < 0.05) or (pd.isna(q) and pd.notna(p) and p < 0.05)),
                interpretation="external validation or negative finding",
                source_table=table,
            )
    return pd.DataFrame(rows)


def make_figure(robustness: pd.DataFrame, falsification: pd.DataFrame) -> None:
    source = pd.concat([robustness.assign(source_panel="robustness"), falsification.assign(source_panel="falsification")], ignore_index=True)
    source.to_csv(SOURCE_DATA / "figure_robustness_source.csv", index=False)
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

    counts = robustness.groupby(["construct", "pass_robustness"]).size().reset_index(name="n")
    piv = counts.pivot(index="construct", columns="pass_robustness", values="n").fillna(0)
    piv.plot(kind="bar", stacked=True, ax=axes[0], color={False: "#999999", True: "#009E73"}, legend=False)
    axes[0].set_ylabel("Tests")
    axes[0].set_title("Robustness outcomes")

    loo = robustness[robustness["robustness_family"].eq("leave_one_dataset_out")].copy()
    if not loo.empty:
        plot = loo.groupby("claim_key")["pass_robustness"].mean().sort_values()
        axes[1].barh(plot.index, plot.values, color="#4C78A8")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Pass fraction")
    axes[1].set_title("Leave-one-dataset")

    hidden = robustness[robustness["robustness_family"].eq("hidden_size_sensitivity")].copy()
    if not hidden.empty:
        for dataset, group in hidden.groupby("scope"):
            axes[2].plot(group["hidden_size"], group["effect"], marker="o", linewidth=1.0, label=dataset)
    axes[2].set_xlabel("Hidden size")
    axes[2].set_ylabel("Median validation NLL")
    axes[2].set_title("Hidden-size sensitivity")

    controls = falsification[falsification["robustness_family"].isin(["negative_control", "permutation_control", "baseline_comparison"])].copy()
    if not controls.empty:
        plot = controls.head(14).copy()
        colors = np.where(plot["pass_robustness"], "#009E73", "#D55E00")
        axes[3].barh(plot["claim_key"].astype(str), plot["effect"], color=colors)
        axes[3].axvline(0, color="#555555", linewidth=0.8)
    axes[3].set_xlabel("Effect / delta")
    axes[3].set_title("Falsification checks")

    for ax in axes:
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_robustness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    projection = pd.read_csv(TABLES / "human_state_capacity_multiaxis_projection.csv")
    parts = [
        leave_one_dataset_tests(projection),
        hidden_size_sensitivity(),
        residualized_and_validation_tests(),
    ]
    robustness = pd.concat(parts, ignore_index=True)
    falsification = baseline_and_control_tests()
    for df in [robustness, falsification]:
        if "q_value" not in df:
            df["q_value"] = np.nan
        fill_mask = df["q_value"].isna()
        df.loc[fill_mask, "q_value"] = bh_q(df.loc[fill_mask, "p_value"])

    robustness.to_csv(TABLES / "robustness_master_table.csv", index=False)
    falsification.to_csv(TABLES / "falsification_tests.csv", index=False)
    make_figure(robustness, falsification)

    (SOURCE_DATA / "figure_robustness_data_dictionary.md").write_text(
        "\n".join(
            [
                "# Figure robustness source data dictionary",
                "",
                "- `figure_robustness_source.csv`: combined source data for Step 16 robustness and falsification panels.",
                "- `source_panel=robustness`: hidden-size sensitivity, leave-one-dataset/task, residualized and external-validation checks.",
                "- `source_panel=falsification`: baselines, random-axis/shuffled-coordinate controls and permutation controls.",
                "- Key fields: `claim_key`, `construct`, `robustness_family`, `test_name`, `scope`, `n`, `effect`, `p_value`, `q_value`, `pass_robustness`.",
                "- Script: `state_capacity_tinyrnn/scripts/15_baselines_robustness/run_robustness.py`.",
            ]
        ),
        encoding="utf-8",
    )
    (SOURCE_DATA / "figure_robustness_script_used.txt").write_text(
        "state_capacity_tinyrnn/scripts/15_baselines_robustness/run_robustness.py",
        encoding="utf-8",
    )

    audit = [
        "# Step 16 Robustness and Falsification Audit",
        "",
        f"- Robustness rows: {len(robustness)}.",
        f"- Falsification rows: {len(falsification)}.",
        f"- Robustness pass rows: {int(robustness['pass_robustness'].fillna(False).sum())}.",
        f"- Falsification pass rows: {int(falsification['pass_robustness'].fillna(False).sum())}.",
        "",
        "## Interpretation Boundary",
        "",
        "- State is not promoted to a strong claim because ANN residualized state recovery, TU Berlin state validation and HBN participant-level state validation remain weak.",
        "- Capacity is robust mainly when tied to recurrent geometry and capacity-pressure/load validation.",
        "- The ds007554 descriptive behavioral baseline still outperforms coordinate models, which must be stated in the manuscript.",
        "- Step 16 uses available GRU outputs; vanilla RNN/LSTM architecture variants are not present in the current trained-model outputs and therefore are not claimed.",
    ]
    (AUDIT / "step16_robustness_audit.md").write_text("\n".join(audit), encoding="utf-8")

    status = {
        "status": "implemented_and_run",
        "robustness_rows": int(len(robustness)),
        "falsification_rows": int(len(falsification)),
        "robustness_pass_rows": int(robustness["pass_robustness"].fillna(False).sum()),
        "falsification_pass_rows": int(falsification["pass_robustness"].fillna(False).sum()),
        "outputs": [
            "outputs/tables/robustness_master_table.csv",
            "outputs/tables/falsification_tests.csv",
            "outputs/figures/figure_robustness.png",
            "outputs/source_data/figure_robustness_source.csv",
        ],
    }
    (LOGS / "step16_robustness_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP16_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
