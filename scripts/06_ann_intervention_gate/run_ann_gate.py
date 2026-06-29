from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
SOURCE_DATA = PROJECT_ROOT / "outputs" / "source_data"
AUDIT = PROJECT_ROOT / "outputs" / "audit"
N_PERMUTATIONS = 1000
RNG_SEED = 20260527
PRIMARY_FEATURE_SET = "residualized_fingerprint"

BASE_FEATURES = [
    "brier_score",
    "response_rate",
    "response_entropy",
    "lapse_proxy",
    "time_accuracy_slope",
    "early_late_accuracy_delta",
    "entropy_time_slope",
    "response_time_slope",
    "probability_volatility",
    "confidence_volatility",
    "error_transition_rate",
    "error_lag1_autocorrelation",
    "sequential_dependence",
    "nback_load_accuracy_slope",
    "accuracy_nback",
    "accuracy_go_nogo",
    "accuracy_context_xor",
    "participation_ratio",
    "trajectory_radius",
    "mean_hidden_abs",
    "hidden_variability",
    "mean_hidden_step_norm",
    "sd_hidden_step_norm",
    "recurrent_rank_dyn",
    "spectral_radius",
    "local_memory_timescale",
    "latent_decoder_accuracy",
]
CONFOUNDS = ["mean_accuracy", "negative_log_likelihood"]


def load_agents() -> pd.DataFrame:
    required = [
        TABLES / "artificial_agent_registry.csv",
        TABLES / "artificial_behavioral_fingerprints.csv",
        TABLES / "artificial_dynamics_fingerprints.csv",
        TABLES / "performance_matched_agent_pairs.csv",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Step 07 requires Step 05 and Step 06 outputs first: " + ", ".join(missing))

    registry = pd.read_csv(TABLES / "artificial_agent_registry.csv")
    behavior = pd.read_csv(TABLES / "artificial_behavioral_fingerprints.csv")
    dynamics = pd.read_csv(TABLES / "artificial_dynamics_fingerprints.csv")
    agents = registry.merge(behavior, on="agent_id", validate="one_to_one")
    agents = agents.merge(dynamics, on="agent_id", validate="one_to_one", suffixes=("", "_dyn"))
    agents["tau_shift"] = np.abs(np.log(agents["tau"].clip(lower=1e-6)))
    agents["update_gain_loss"] = 1.0 - agents["update_gain"]
    agents["memory_decay_loss"] = 1.0 - agents["memory_decay"]
    return agents


def residualize_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    confounds = df[CONFOUNDS].copy()
    confounds = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(confounds), columns=CONFOUNDS)
    for column in feature_columns:
        y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        observed = ~np.isnan(y)
        residual = np.full_like(y, np.nan, dtype=float)
        if observed.sum() >= 3:
            model = LinearRegression()
            model.fit(confounds.loc[observed, CONFOUNDS], y[observed])
            residual[observed] = y[observed] - model.predict(confounds.loc[observed, CONFOUNDS])
        result[column] = residual
    return result


def feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "raw_fingerprint":
        return df[BASE_FEATURES].copy()
    if feature_set == "residualized_fingerprint":
        return residualize_features(df, BASE_FEATURES)[BASE_FEATURES].copy()
    raise ValueError(f"Unknown feature set {feature_set}")


def classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )


def grouped_classifier_score(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros_like(y, dtype=int)
    y_prob = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in splitter.split(X, y, groups):
        model = classifier_pipeline()
        model.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X.iloc[test_idx])
        y_prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) == 2 else np.nan
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "roc_auc": float(auc),
        "predicted_label": y_pred,
        "predicted_probability": y_prob,
    }


def swap_labels_within_pairs(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = y.copy()
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        if len(idx) == 2 and rng.random() < 0.5:
            shuffled[idx] = shuffled[idx[::-1]]
    return shuffled


def build_matched_long(agents: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(TABLES / "performance_matched_agent_pairs.csv")
    rows = []
    for record in pairs.to_dict("records"):
        rows.append(
            {
                "match_id": record["match_id"],
                "task_family": record["task_family"],
                "agent_id": record["state_agent_id"],
                "label": 1,
                "label_name": "state",
            }
        )
        rows.append(
            {
                "match_id": record["match_id"],
                "task_family": record["task_family"],
                "agent_id": record["capacity_agent_id"],
                "label": 0,
                "label_name": "capacity",
            }
        )
    return pd.DataFrame(rows).merge(agents, on="agent_id", how="left", validate="many_to_one")


def run_matched_classification(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    matched = build_matched_long(agents)
    result_rows = []
    source_rows = []
    permutation_rows = []

    for task_family, task_df in matched.groupby("task_family"):
        for feature_set in ["raw_fingerprint", "residualized_fingerprint"]:
            X = feature_frame(task_df, feature_set)
            y = task_df["label"].to_numpy(dtype=int)
            groups = task_df["match_id"].to_numpy()
            observed = grouped_classifier_score(X, y, groups)
            permuted_scores = []
            for permutation in range(N_PERMUTATIONS):
                y_perm = swap_labels_within_pairs(y, groups, rng)
                score = grouped_classifier_score(X, y_perm, groups)["balanced_accuracy"]
                permuted_scores.append(score)
                permutation_rows.append(
                    {
                        "analysis_type": "matched_state_capacity_classification",
                        "feature_set": feature_set,
                        "task_family": task_family,
                        "target_axis": "state_vs_capacity",
                        "permutation_index": permutation + 1,
                        "metric": "balanced_accuracy",
                        "metric_value": float(score),
                    }
                )

            p95 = float(np.quantile(permuted_scores, 0.95))
            empirical_p = float((np.sum(np.asarray(permuted_scores) >= observed["balanced_accuracy"]) + 1) / (N_PERMUTATIONS + 1))
            pass_gate = bool(observed["balanced_accuracy"] > p95)
            result_rows.append(
                {
                    "analysis_type": "matched_state_capacity_classification",
                    "feature_set": feature_set,
                    "task_family": task_family,
                    "n_samples": int(len(task_df)),
                    "n_groups": int(task_df["match_id"].nunique()),
                    "metric": "balanced_accuracy",
                    "observed": observed["balanced_accuracy"],
                    "roc_auc": observed["roc_auc"],
                    "permutation_p95": p95,
                    "empirical_p_value": empirical_p,
                    "pass_gate": pass_gate,
                }
            )

            for idx, row in task_df.reset_index(drop=True).iterrows():
                source_rows.append(
                    {
                        "analysis_type": "matched_state_capacity_classification",
                        "feature_set": feature_set,
                        "task_family": task_family,
                        "agent_id": row["agent_id"],
                        "match_id": row["match_id"],
                        "true_label": row["label_name"],
                        "predicted_probability_state": observed["predicted_probability"][idx],
                        "predicted_label": "state" if observed["predicted_label"][idx] == 1 else "capacity",
                    }
                )

    return pd.DataFrame(result_rows), pd.DataFrame(source_rows), pd.DataFrame(permutation_rows)


def regressor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                Ridge(alpha=1.0),
            ),
        ]
    )


def cv_regression_predictions(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    unique_groups = np.unique(groups)
    splitter = LeaveOneGroupOut() if len(unique_groups) <= 5 else GroupKFold(n_splits=5)
    predictions = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = regressor_pipeline()
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
    return predictions


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def run_pure_recovery(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    source_rows = []
    permutation_rows = []
    analyses = [
        ("state_severity_recovery", "state_perturbation", "state_severity"),
        ("state_parameter_recovery", "state_perturbation", "tau_shift"),
        ("state_parameter_recovery", "state_perturbation", "lapse"),
        ("state_parameter_recovery", "state_perturbation", "hidden_noise_sigma"),
        ("state_parameter_recovery", "state_perturbation", "update_gain_loss"),
        ("state_parameter_recovery", "state_perturbation", "memory_decay_loss"),
        ("capacity_level_recovery", "capacity_perturbation", "capacity_level"),
    ]

    for analysis_type, family, target_axis in analyses:
        data = agents[agents["family"] == family].copy()
        for feature_set in ["raw_fingerprint", "residualized_fingerprint"]:
            X = feature_frame(data, feature_set)
            y = data[target_axis].to_numpy(dtype=float)
            groups = data["seed"].to_numpy()
            predictions = cv_regression_predictions(X, y, groups)
            rho, p_value = safe_spearman(y, predictions)
            permuted_rhos = []
            for permutation in range(N_PERMUTATIONS):
                y_perm = rng.permutation(y)
                perm_predictions = cv_regression_predictions(X, y_perm, groups)
                perm_rho, _ = safe_spearman(y_perm, perm_predictions)
                metric = 0.0 if np.isnan(perm_rho) else abs(perm_rho)
                permuted_rhos.append(metric)
                permutation_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "feature_set": feature_set,
                        "task_family": "",
                        "target_axis": target_axis,
                        "permutation_index": permutation + 1,
                        "metric": "abs_spearman_rho",
                        "metric_value": float(metric),
                    }
                )
            p95 = float(np.quantile(permuted_rhos, 0.95))
            empirical_p = float((np.sum(np.asarray(permuted_rhos) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1))
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "feature_set": feature_set,
                    "task_family": "",
                    "target_axis": target_axis,
                    "n_samples": int(len(data)),
                    "n_groups": int(data["seed"].nunique()),
                    "metric": "spearman_rho",
                    "observed": rho,
                    "roc_auc": np.nan,
                    "permutation_p95": p95,
                    "empirical_p_value": empirical_p,
                    "pass_gate": bool(abs(rho) >= 0.70 and abs(rho) > p95),
                }
            )
            for agent_id, true_value, predicted_value in zip(data["agent_id"], y, predictions):
                source_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "feature_set": feature_set,
                        "task_family": "",
                        "agent_id": agent_id,
                        "match_id": "",
                        "true_label": target_axis,
                        "true_value": true_value,
                        "predicted_value": predicted_value,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(source_rows), pd.DataFrame(permutation_rows)


def train_predict_hybrid(
    train_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    target_axis: str,
    feature_set: str,
    y_override: np.ndarray | None = None,
) -> np.ndarray:
    X_train = feature_frame(train_df, feature_set)
    X_hybrid = feature_frame(hybrid_df, feature_set)
    y_train = train_df[target_axis].to_numpy(dtype=float) if y_override is None else y_override
    model = regressor_pipeline()
    model.fit(X_train, y_train)
    return model.predict(X_hybrid)


def run_hybrid_recovery(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 2)
    hybrid = agents[agents["family"] == "hybrid"].copy()
    state_train = agents[agents["family"] == "state_perturbation"].copy()
    capacity_train = agents[agents["family"] == "capacity_perturbation"].copy()
    rows = []
    source_rows = []
    permutation_rows = []

    analyses = [
        ("hybrid_state_axis", state_train, "state_severity"),
        ("hybrid_capacity_axis", capacity_train, "capacity_level"),
    ]

    for analysis_type, train_df, target_axis in analyses:
        for feature_set in ["raw_fingerprint", "residualized_fingerprint"]:
            y_true = hybrid[target_axis].to_numpy(dtype=float)
            predictions = train_predict_hybrid(train_df, hybrid, target_axis, feature_set)
            rho, p_value = safe_spearman(y_true, predictions)
            permuted_rhos = []
            train_y = train_df[target_axis].to_numpy(dtype=float)
            for permutation in range(N_PERMUTATIONS):
                y_perm = rng.permutation(train_y)
                perm_predictions = train_predict_hybrid(train_df, hybrid, target_axis, feature_set, y_override=y_perm)
                perm_rho, _ = safe_spearman(y_true, perm_predictions)
                metric = 0.0 if np.isnan(perm_rho) else abs(perm_rho)
                permuted_rhos.append(metric)
                permutation_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "feature_set": feature_set,
                        "task_family": "",
                        "target_axis": target_axis,
                        "permutation_index": permutation + 1,
                        "metric": "abs_spearman_rho",
                        "metric_value": float(metric),
                    }
                )
            p95 = float(np.quantile(permuted_rhos, 0.95))
            empirical_p = float((np.sum(np.asarray(permuted_rhos) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1))
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "feature_set": feature_set,
                    "target_axis": target_axis,
                    "n_hybrid_agents": int(len(hybrid)),
                    "spearman_rho": rho,
                    "nominal_p_value": p_value,
                    "permutation_p95_abs_rho": p95,
                    "empirical_p_value": empirical_p,
                    "pass_gate": bool(abs(rho) > p95 and empirical_p < 0.05),
                }
            )
            for agent_id, true_value, predicted_value in zip(hybrid["agent_id"], y_true, predictions):
                source_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "feature_set": feature_set,
                        "agent_id": agent_id,
                        "target_axis": target_axis,
                        "true_value": true_value,
                        "predicted_value": predicted_value,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(source_rows), pd.DataFrame(permutation_rows)


def make_figure(
    gate_results: pd.DataFrame,
    hybrid_results: pd.DataFrame,
    source_classification: pd.DataFrame,
    source_recovery: pd.DataFrame,
    source_hybrid: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    axes = axes.ravel()

    cls = gate_results[
        (gate_results["analysis_type"] == "matched_state_capacity_classification")
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ].copy()
    axes[0].bar(np.arange(len(cls)), cls["observed"], color="#0072B2", width=0.7, label="observed")
    axes[0].scatter(np.arange(len(cls)), cls["permutation_p95"], color="#D55E00", s=20, label="perm. 95th")
    axes[0].set_xticks(np.arange(len(cls)))
    axes[0].set_xticklabels(cls["task_family"], rotation=25, ha="right")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Matched state/capacity classifier")
    axes[0].legend(frameon=False, fontsize=6)

    rec = gate_results[
        gate_results["analysis_type"].isin(["state_severity_recovery", "capacity_level_recovery"])
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ].copy()
    labels = rec["analysis_type"].str.replace("_", " ").str.replace(" recovery", "", regex=False)
    axes[1].bar(np.arange(len(rec)), rec["observed"], color="#009E73", width=0.7)
    axes[1].scatter(np.arange(len(rec)), rec["permutation_p95"], color="#D55E00", s=20)
    axes[1].axhline(0.70, color="#333333", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(np.arange(len(rec)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Known-axis recovery")

    hybrid_primary = hybrid_results[hybrid_results["feature_set"] == PRIMARY_FEATURE_SET].copy()
    axes[2].bar(np.arange(len(hybrid_primary)), hybrid_primary["spearman_rho"], color="#CC79A7", width=0.7)
    axes[2].scatter(np.arange(len(hybrid_primary)), hybrid_primary["permutation_p95_abs_rho"], color="#D55E00", s=20)
    axes[2].set_xticks(np.arange(len(hybrid_primary)))
    axes[2].set_xticklabels(hybrid_primary["analysis_type"].str.replace("_", " "), rotation=25, ha="right")
    axes[2].set_ylabel("Spearman rho")
    axes[2].set_title("Hybrid recovery")

    for axis_name, color in [("state_severity", "#0072B2"), ("capacity_level", "#D55E00")]:
        data = source_hybrid[
            (source_hybrid["feature_set"] == PRIMARY_FEATURE_SET)
            & (source_hybrid["target_axis"] == axis_name)
        ]
        axes[3].scatter(data["true_value"], data["predicted_value"], s=12, alpha=0.75, color=color, label=axis_name)
    axes[3].set_xlabel("True hybrid axis value")
    axes[3].set_ylabel("Predicted value")
    axes[3].set_title("Hybrid projections")
    axes[3].legend(frameon=False, fontsize=6)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_ann_intervention_gate.{suffix}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df.to_dict("records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join(rows)


def write_source_dictionary() -> None:
    text = """# Figure ANN Intervention Gate Source Data

Source data combine three panels:

- matched state/capacity classification predictions;
- pure perturbation recovery predictions;
- hybrid perturbation recovery predictions.

Rows are tagged by `analysis_type` and `feature_set`.
"""
    (SOURCE_DATA / "figure_ann_intervention_gate_data_dictionary.md").write_text(text, encoding="utf-8")
    (SOURCE_DATA / "figure_ann_intervention_gate_script_used.txt").write_text(
        "scripts/06_ann_intervention_gate/run_ann_gate.py\n",
        encoding="utf-8",
    )


def write_decision(gate_results: pd.DataFrame, hybrid_results: pd.DataFrame) -> bool:
    primary_cls = gate_results[
        (gate_results["analysis_type"] == "matched_state_capacity_classification")
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    primary_state = gate_results[
        (gate_results["analysis_type"] == "state_severity_recovery")
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    primary_state_parameters = gate_results[
        (gate_results["analysis_type"] == "state_parameter_recovery")
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    primary_capacity = gate_results[
        (gate_results["analysis_type"] == "capacity_level_recovery")
        & (gate_results["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    primary_hybrid = hybrid_results[hybrid_results["feature_set"] == PRIMARY_FEATURE_SET]

    classifier_pass = bool(primary_cls["pass_gate"].all())
    composite_state_pass = bool((primary_state["observed"].abs() >= 0.70).all() and primary_state["pass_gate"].all())
    state_parameter_pass_count = int(
        ((primary_state_parameters["observed"].abs() >= 0.70) & primary_state_parameters["pass_gate"]).sum()
    )
    state_pass = bool(composite_state_pass or state_parameter_pass_count >= 3)
    capacity_pass = bool((primary_capacity["observed"].abs() >= 0.70).all() and primary_capacity["pass_gate"].all())
    hybrid_pass = bool(primary_hybrid["pass_gate"].all())
    overall_pass = classifier_pass and state_pass and capacity_pass and hybrid_pass

    lines = [
        "# Step 07 ANN Intervention Gate Decision",
        "",
        f"Decision: {'PASS' if overall_pass else 'FAIL'}",
        "",
        f"Primary feature set: `{PRIMARY_FEATURE_SET}`",
        "",
        f"- Classifier exceeds shuffled-label 95th percentile: {classifier_pass}",
        f"- Composite state recovery passes: {composite_state_pass}",
        f"- Recoverable individual state parameters passing: {state_parameter_pass_count}/5",
        f"- Revised multi-axis state recovery criterion passes: {state_pass}",
        f"- Capacity recovery Spearman >= 0.70 and beats shuffled labels: {capacity_pass}",
        f"- Hybrid recovery significant for both axes: {hybrid_pass}",
        "",
        "Primary classifier results:",
        "",
        simple_markdown_table(primary_cls),
        "",
        "Primary recovery results:",
        "",
        simple_markdown_table(pd.concat([primary_state, primary_state_parameters, primary_capacity], ignore_index=True)),
        "",
        "Primary hybrid results:",
        "",
        simple_markdown_table(primary_hybrid),
        "",
        "Protocol consequence:",
        "- If PASS, human analyses may proceed as planned.",
        "- If FAIL, human state-capacity claims must be explicitly downgraded to exploratory or negative.",
    ]
    (AUDIT / "ann_gate_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return overall_pass


def main() -> int:
    assert_full_run_allowed()
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    agents = load_agents()
    classification_results, classification_source, classification_perms = run_matched_classification(agents)
    recovery_results, recovery_source, recovery_perms = run_pure_recovery(agents)
    hybrid_results, hybrid_source, hybrid_perms = run_hybrid_recovery(agents)

    gate_results = pd.concat([classification_results, recovery_results], ignore_index=True)
    shuffled = pd.concat([classification_perms, recovery_perms, hybrid_perms], ignore_index=True)
    figure_source = pd.concat(
        [
            classification_source.assign(source_panel="classification"),
            recovery_source.assign(source_panel="pure_recovery"),
            hybrid_source.assign(source_panel="hybrid_recovery"),
        ],
        ignore_index=True,
        sort=False,
    )

    gate_results.to_csv(TABLES / "ann_intervention_gate_results.csv", index=False)
    hybrid_results.to_csv(TABLES / "ann_hybrid_recovery.csv", index=False)
    shuffled.to_csv(TABLES / "ann_shuffled_label_controls.csv", index=False)
    figure_source.to_csv(SOURCE_DATA / "figure_ann_intervention_gate_source.csv", index=False)
    write_source_dictionary()
    make_figure(gate_results, hybrid_results, classification_source, recovery_source, hybrid_source)
    passed = write_decision(gate_results, hybrid_results)

    summary = {
        "passed": bool(passed),
        "n_permutations": N_PERMUTATIONS,
        "n_gate_rows": int(len(gate_results)),
        "n_hybrid_rows": int(len(hybrid_results)),
        "n_shuffled_rows": int(len(shuffled)),
    }
    print("ANN_GATE complete " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
