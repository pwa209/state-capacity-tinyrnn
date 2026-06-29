from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUNDLED_SITE = Path(
    "C:/Users/Gebruiker/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages"
)
if BUNDLED_SITE.exists() and str(BUNDLED_SITE) not in sys.path:
    sys.path.append(str(BUNDLED_SITE))

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TABLES = ROOT / "outputs" / "nhb_revision" / "tables"
FANCY_SOURCE = ROOT / "outputs" / "nhb_revision" / "fancy_figures" / "source_data"
OUT = ROOT / "outputs" / "nhb_revision" / "mechanistic_insights"
OUT.mkdir(parents=True, exist_ok=True)

ANALYSIS_ID = "nhb_33_mechanistic_insight_analyses"
SCRIPT_NAME = "scripts/nhb_revision/33_mechanistic_insight_analyses.py"
RNG = np.random.default_rng(20260616)
N_BOOT = 120

ART_BEHAVIOR = [
    "mean_accuracy",
    "negative_log_likelihood",
    "brier_score",
    "response_entropy",
    "time_accuracy_slope",
    "early_late_accuracy_delta",
    "probability_volatility",
    "confidence_volatility",
    "error_transition_rate",
    "error_lag1_autocorrelation",
    "nback_load_accuracy_slope",
]
ART_DYNAMICS = [
    "participation_ratio",
    "trajectory_radius",
    "mean_hidden_abs",
    "hidden_variability",
    "recurrent_rank_dyn",
    "spectral_radius",
    "local_memory_timescale",
    "latent_decoder_accuracy",
    "mean_hidden_step_norm",
    "sd_hidden_step_norm",
]
HUMAN_DYNAMICS = [
    "dynamics_trajectory_participation_ratio_z",
    "dynamics_trajectory_cov_rank_z",
    "dynamics_trajectory_radius_z",
    "dynamics_step_norm_mean_z",
    "dynamics_hidden_variability_z",
    "dynamics_spectral_radius_z",
    "dynamics_memory_timescale_steps_z",
    "dynamics_capacity_geometry_z",
    "dynamics_state_proxy_z",
]
HUMAN_OUTCOMES = ["mean_accuracy", "rt_cv", "lapse_proxy", "brier_score", "negative_log_likelihood"]


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def design_matrix(df: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    pieces = []
    for c in covars:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            pieces.append(zscore(df[c]).rename(c))
        else:
            pieces.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def ols_coef(y: pd.Series, x: pd.Series, cov: pd.DataFrame | None = None) -> tuple[float, float, float]:
    dat = pd.DataFrame({"y": zscore(y), "x": zscore(x)})
    if cov is not None and not cov.empty:
        dat = pd.concat([dat, cov], axis=1)
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna()
    if len(dat) < 20 or dat["x"].std(ddof=0) == 0:
        return np.nan, np.nan, np.nan
    yv = dat["y"].to_numpy(float)
    X0 = dat.drop(columns=["y"]).to_numpy(float)
    X = np.column_stack([np.ones(len(dat)), X0])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        df = max(len(yv) - X.shape[1], 1)
        sigma2 = float((resid @ resid) / df)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
        coef = float(beta[1])
        coef_se = float(se[1])
        t = coef / coef_se if coef_se > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan
        return coef, coef_se, p
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan


def mediation_once(df: pd.DataFrame, x_col: str, m_col: str, y_col: str, covars: list[str]) -> dict[str, float]:
    cov = design_matrix(df, covars)
    a, a_se, a_p = ols_coef(df[m_col], df[x_col], cov)
    cov_b = pd.concat([cov, zscore(df[x_col]).rename(x_col)], axis=1)
    b, b_se, b_p = ols_coef(df[y_col], df[m_col], cov_b)
    c, _, c_p = ols_coef(df[y_col], df[x_col], cov)
    cov_cp = pd.concat([cov, zscore(df[m_col]).rename(m_col)], axis=1)
    cp, _, cp_p = ols_coef(df[y_col], df[x_col], cov_cp)
    indirect = a * b if np.isfinite(a) and np.isfinite(b) else np.nan
    prop = indirect / c if np.isfinite(indirect) and np.isfinite(c) and abs(c) > 1e-9 else np.nan
    return {
        "a_path": a,
        "a_p_value": a_p,
        "b_path": b,
        "b_p_value": b_p,
        "total_effect": c,
        "total_p_value": c_p,
        "direct_effect": cp,
        "direct_p_value": cp_p,
        "indirect_effect": indirect,
        "proportion_mediated": prop,
    }


def bootstrap_mediation(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    covars: list[str],
    cluster_col: str | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    obs = mediation_once(df, x_col, m_col, y_col, covars)
    boots = []
    if cluster_col and cluster_col in df.columns:
        clusters = df[cluster_col].dropna().unique()
        for _ in range(n_boot):
            sample_clusters = RNG.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat([df[df[cluster_col] == c] for c in sample_clusters], ignore_index=True)
            boots.append(mediation_once(sample, x_col, m_col, y_col, covars)["indirect_effect"])
    else:
        n = len(df)
        for _ in range(n_boot):
            sample = df.iloc[RNG.integers(0, n, size=n)].reset_index(drop=True)
            boots.append(mediation_once(sample, x_col, m_col, y_col, covars)["indirect_effect"])
    arr = np.asarray([b for b in boots if np.isfinite(b)], dtype=float)
    if len(arr) < 50:
        lo = hi = p = np.nan
    else:
        lo, hi = np.quantile(arr, [0.025, 0.975])
        p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        p = max(float(p), 1 / (len(arr) + 1))
    return {**obs, "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi), "bootstrap_p_value": float(p)}


def bh_q(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return pd.Series(q, index=p_values.index)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(adj, 0, 1)
    q[valid] = out
    return pd.Series(q, index=p_values.index)


def artificial_mediation(agents: pd.DataFrame) -> pd.DataFrame:
    df = agents[agents["perturbation_class"].isin(["state", "capacity"])].copy()
    df["is_capacity_family"] = (df["perturbation_class"] == "capacity").astype(float)
    covars = ["model_family", "seed", "mean_accuracy"]
    rows = []
    key_outcomes = [
        "nback_load_accuracy_slope",
        "negative_log_likelihood",
        "response_entropy",
        "probability_volatility",
        "error_transition_rate",
    ]
    key_mediators = [
        "participation_ratio",
        "recurrent_rank_dyn",
        "local_memory_timescale",
        "trajectory_radius",
        "hidden_variability",
        "mean_hidden_step_norm",
    ]
    for m in key_mediators:
        for y in key_outcomes:
            dat = df[["is_capacity_family", m, y, *covars]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(dat) < 60:
                continue
            res = bootstrap_mediation(dat, "is_capacity_family", m, y, covars, cluster_col=None)
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "domain": "artificial_agent",
                    "pathway": "imposed_family_to_behavior_via_dynamics",
                    "x": "capacity_family_vs_state_family",
                    "mediator": m,
                    "outcome": y,
                    "n_rows": len(dat),
                    **res,
                    "interpretation": "Artificial-agent mediation; perturbation family is imposed by construction.",
                }
            )
    out = pd.DataFrame(rows)
    out["bootstrap_q_value"] = bh_q(out["bootstrap_p_value"])
    out["passes_mechanistic_screen"] = (
        (out["bootstrap_q_value"] < 0.05)
        & (np.sign(out["bootstrap_ci_low"]) == np.sign(out["bootstrap_ci_high"]))
    )
    return out


def human_mediation(human: pd.DataFrame) -> pd.DataFrame:
    df = human[human["dynamics_available"].astype(bool)].copy()
    rows = []
    focal_mediators = [
        "dynamics_capacity_geometry_z",
        "dynamics_state_proxy_z",
        "dynamics_trajectory_participation_ratio_z",
    ]
    focal_outcomes = ["mean_accuracy", "rt_cv", "lapse_proxy"]
    specs = [
        ("capacity_parameter_resource_z", "state_parameter_instability_z", "capacity_profile"),
        ("state_parameter_instability_z", "capacity_parameter_resource_z", "state_instability_profile"),
    ]
    for x, complement, label in specs:
        covars = ["dataset", "task", complement]
        for m in focal_mediators:
            for y in focal_outcomes:
                if m not in df.columns or y not in df.columns:
                    continue
                dat = df[[x, complement, m, y, "dataset", "task", "participant_id"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(dat) < 100 or dat[m].std(ddof=0) == 0 or dat[y].std(ddof=0) == 0:
                    continue
                res = bootstrap_mediation(dat, x, m, y, covars, cluster_col="participant_id")
                rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "domain": "human_observational",
                        "pathway": "profile_to_behavior_via_fitted_dynamics",
                        "x": label,
                        "mediator": m,
                        "outcome": y,
                        "n_rows": len(dat),
                        "n_participants": dat["participant_id"].nunique(),
                        **res,
                        "interpretation": "Human row-level observational pathway; controls include dataset, task and complementary profile.",
                    }
                )
    out = pd.DataFrame(rows)
    out["bootstrap_q_value"] = bh_q(out["bootstrap_p_value"])
    out["passes_mechanistic_screen"] = (
        (out["bootstrap_q_value"] < 0.05)
        & (np.sign(out["bootstrap_ci_low"]) == np.sign(out["bootstrap_ci_high"]))
    )
    return out


def attribution_analysis(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = agents[agents["perturbation_class"].isin(["state", "capacity"])].copy()
    features = [c for c in ART_BEHAVIOR + ART_DYNAMICS if c in df.columns]
    dat = df[["perturbation_class", "model_family", "seed", *features]].copy()
    y = (dat["perturbation_class"] == "capacity").astype(int).to_numpy()
    X = dat[features]
    groups = (dat["model_family"].astype(str) + "_seed" + dat["seed"].astype(str)).to_numpy()
    cv = GroupKFold(n_splits=min(9, len(np.unique(groups))))
    pred = np.zeros(len(dat), dtype=float)
    fold_rows = []
    importances = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=20260616 + fold,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(X.iloc[tr], y[tr])
        pred[te] = clf.predict_proba(X.iloc[te])[:, 1]
        yhat = pred[te] >= 0.5
        fold_rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "balanced_accuracy": balanced_accuracy_score(y[te], yhat),
                "auc": roc_auc_score(y[te], pred[te]),
            }
        )
        perm = permutation_importance(
            clf,
            X.iloc[te],
            y[te],
            scoring="balanced_accuracy",
            n_repeats=30,
            random_state=20260700 + fold,
        )
        for f, mean, sd in zip(features, perm.importances_mean, perm.importances_std):
            importances.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "feature": f,
                    "fold": fold,
                    "permutation_importance_mean": mean,
                    "permutation_importance_sd": sd,
                    "feature_group": "dynamics" if f in ART_DYNAMICS else "behavior",
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    imp = pd.DataFrame(importances)
    summary = (
        imp.groupby(["feature", "feature_group"], as_index=False)
        .agg(mean_importance=("permutation_importance_mean", "mean"), sd_importance=("permutation_importance_mean", "std"))
        .sort_values("mean_importance", ascending=False)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    group_summary = (
        summary.groupby("feature_group", as_index=False)["mean_importance"]
        .sum()
        .rename(columns={"mean_importance": "summed_mean_importance"})
    )
    fold_df["overall_balanced_accuracy"] = balanced_accuracy_score(y, pred >= 0.5)
    fold_df["overall_auc"] = roc_auc_score(y, pred)
    return pd.concat([summary, group_summary], ignore_index=True, sort=False), fold_df


def artificial_counterfactuals(agents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = [
        "mean_accuracy",
        "nback_load_accuracy_slope",
        "participation_ratio",
        "recurrent_rank_dyn",
        "hidden_variability",
        "local_memory_timescale",
        "response_entropy",
        "probability_volatility",
    ]
    base = agents[agents["perturbation_class"] == "capacity"].set_index(["model_family", "seed", "hidden_size"])
    state = agents[agents["perturbation_class"] == "state"].copy()
    for _, r in state.iterrows():
        key = (r["model_family"], r["seed"], r["hidden_size"])
        if key not in base.index:
            continue
        b = base.loc[key]
        for y in outcomes:
            if y in r and y in b and pd.notna(r[y]) and pd.notna(b[y]):
                rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "domain": "artificial_agent",
                        "counterfactual_type": "graph_preserving_state_intervention",
                        "model_family": r["model_family"],
                        "seed": r["seed"],
                        "hidden_size": r["hidden_size"],
                        "perturbation_name": r["perturbation_name"],
                        "axis_value": r["state_severity"],
                        "outcome": y,
                        "baseline_value": b[y],
                        "perturbed_value": r[y],
                        "delta": r[y] - b[y],
                    }
                )
    cap = agents[agents["perturbation_class"] == "capacity"].copy()
    base_h1 = agents[
        (agents["perturbation_class"] == "capacity") & (agents["hidden_size"] == 1)
    ].set_index(["model_family", "seed"])
    for _, r in cap.iterrows():
        key = (r["model_family"], r["seed"])
        if key not in base_h1.index:
            continue
        b = base_h1.loc[key]
        for y in outcomes:
            if y in r and y in b and pd.notna(r[y]) and pd.notna(b[y]):
                rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "domain": "artificial_agent",
                        "counterfactual_type": "graph_altering_capacity_scale",
                        "model_family": r["model_family"],
                        "seed": r["seed"],
                        "hidden_size": r["hidden_size"],
                        "perturbation_name": r["perturbation_name"],
                        "axis_value": r["capacity_level"],
                        "outcome": y,
                        "baseline_value": b[y],
                        "perturbed_value": r[y],
                        "delta": r[y] - b[y],
                    }
                )
    deltas = pd.DataFrame(rows)
    summary_rows = []
    for (ctype, outcome), g in deltas.groupby(["counterfactual_type", "outcome"]):
        if g["axis_value"].nunique() > 2:
            rho, p = stats.spearmanr(g["axis_value"], g["delta"], nan_policy="omit")
        else:
            rho, p = stats.ttest_1samp(g["delta"], 0.0, nan_policy="omit")
            rho = float(g["delta"].mean())
        summary_rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "domain": "artificial_agent",
                "counterfactual_type": ctype,
                "outcome": outcome,
                "n_rows": len(g),
                "mean_delta": g["delta"].mean(),
                "median_delta": g["delta"].median(),
                "test_statistic": rho,
                "p_value": p,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["q_value"] = bh_q(summary["p_value"])
    return deltas, summary


def human_counterfactual_proxy(human: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("capacity_parameter_resource_z", "capacity_profile_quartile_proxy"),
        ("state_parameter_instability_z", "state_instability_quartile_proxy"),
    ]
    outcomes = [
        "mean_accuracy",
        "rt_cv",
        "lapse_proxy",
        "dynamics_capacity_geometry_z",
        "dynamics_state_proxy_z",
        "dynamics_trajectory_participation_ratio_z",
    ]
    for x, label in specs:
        dat = human[[x, "dataset", "task", "participant_id", *[o for o in outcomes if o in human.columns]]].replace([np.inf, -np.inf], np.nan)
        lo = dat[x].quantile(0.25)
        hi = dat[x].quantile(0.75)
        groups = {"low": dat[dat[x] <= lo], "high": dat[dat[x] >= hi]}
        for outcome in outcomes:
            if outcome not in dat.columns:
                continue
            a = groups["low"][outcome].dropna()
            b = groups["high"][outcome].dropna()
            if len(a) < 20 or len(b) < 20:
                continue
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "domain": "human_observational",
                    "counterfactual_type": label,
                    "outcome": outcome,
                    "n_low": len(a),
                    "n_high": len(b),
                    "low_mean": a.mean(),
                    "high_mean": b.mean(),
                    "high_minus_low": b.mean() - a.mean(),
                    "mannwhitney_u": stat,
                    "p_value": p,
                    "interpretation": "Observed high-vs-low profile contrast; this is a calibrated proxy, not an experimental intervention.",
                }
            )
    out = pd.DataFrame(rows)
    out["q_value"] = bh_q(out["p_value"])
    return out


def make_figure(
    art_med: pd.DataFrame,
    hum_med: pd.DataFrame,
    attribution: pd.DataFrame,
    cf_summary: pd.DataFrame,
    human_cf: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.2, 8.6), dpi=300)
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)

    ax1 = fig.add_subplot(gs[0, 0])
    aplot = art_med.pivot_table(
        index="mediator", columns="outcome", values="indirect_effect", aggfunc="mean"
    )
    im = ax1.imshow(aplot, cmap="RdBu_r", vmin=-np.nanmax(abs(aplot.values)), vmax=np.nanmax(abs(aplot.values)), aspect="auto")
    ax1.set_xticks(range(aplot.shape[1]))
    ax1.set_xticklabels([c.replace("_", "\n") for c in aplot.columns], rotation=0, fontsize=6.5)
    ax1.set_yticks(range(aplot.shape[0]))
    ax1.set_yticklabels([r.replace("_", " ") for r in aplot.index], fontsize=7)
    for i, mediator in enumerate(aplot.index):
        for j, outcome in enumerate(aplot.columns):
            row = art_med[(art_med["mediator"] == mediator) & (art_med["outcome"] == outcome)]
            if not row.empty and bool(row.iloc[0]["passes_mechanistic_screen"]):
                ax1.text(j, i, "*", ha="center", va="center", fontsize=11, color="black")
    ax1.set_title("a  Artificial-agent mediation: family -> dynamics -> behaviour", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02, label="indirect effect")

    ax2 = fig.add_subplot(gs[0, 1])
    subset = hum_med[
        hum_med["mediator"].isin(["dynamics_capacity_geometry_z", "dynamics_state_proxy_z", "dynamics_trajectory_participation_ratio_z"])
        & hum_med["outcome"].isin(["mean_accuracy", "rt_cv", "lapse_proxy"])
    ].copy()
    subset = subset.sort_values("indirect_effect", key=lambda x: x.abs(), ascending=False)
    if subset["passes_mechanistic_screen"].any():
        subset = subset[subset["passes_mechanistic_screen"]].copy()
    subset = subset.head(10).iloc[::-1].copy()
    subset["label"] = (
        subset["x"].map({"capacity_profile": "C", "state_instability_profile": "S"})
        + "-"
        + subset["mediator"]
        .str.replace("dynamics_", "", regex=False)
        .str.replace("_z", "", regex=False)
        .str.replace("trajectory_participation_ratio", "PR", regex=False)
        .str.replace("capacity_geometry", "G", regex=False)
        .str.replace("state_proxy", "SP", regex=False)
        + "-"
        + subset["outcome"]
        .str.replace("mean_accuracy", "Acc", regex=False)
        .str.replace("lapse_proxy", "Lapse", regex=False)
        .str.replace("rt_cv", "RT CV", regex=False)
    )
    y = np.arange(len(subset))
    colors = np.where(subset["passes_mechanistic_screen"], "#2C7A4B", "#B7B7B7")
    ax2.barh(y, subset["indirect_effect"], color=colors)
    ax2.axvline(0, color="#555555", lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(subset["label"], fontsize=7)
    ax2.set_xlabel("cluster-bootstrap indirect effect")
    ax2.set_title("b  Supported human observational pathways", loc="left", fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 0])
    top = attribution[attribution["feature"].notna()].sort_values("mean_importance", ascending=False).head(14).iloc[::-1]
    colors = top["feature_group"].map({"dynamics": "#0072B2", "behavior": "#D55E00"}).fillna("#777777")
    ax3.barh(np.arange(len(top)), top["mean_importance"], color=colors)
    ax3.set_yticks(np.arange(len(top)))
    ax3.set_yticklabels(top["feature"].str.replace("_", " "), fontsize=7)
    ax3.set_xlabel("permutation importance (balanced accuracy)")
    ax3.set_title("c  Fingerprint features carrying the family boundary", loc="left", fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 1])
    key = cf_summary[
        cf_summary["outcome"].isin(["participation_ratio", "nback_load_accuracy_slope", "hidden_variability", "mean_accuracy"])
    ].copy()
    key["label"] = key["counterfactual_type"].str.replace("graph_preserving_", "state: ").str.replace("graph_altering_", "capacity: ").str.replace("_", " ") + "\n" + key["outcome"].str.replace("_", " ")
    key = key.sort_values(["counterfactual_type", "outcome"]).iloc[::-1]
    yy = np.arange(len(key))
    ax4.barh(yy, key["median_delta"], color=np.where(key["q_value"] < 0.05, "#5A5A5A", "#C9C9C9"))
    ax4.axvline(0, color="#555555", lw=0.8)
    ax4.set_yticks(yy)
    ax4.set_yticklabels(key["label"], fontsize=6.8)
    ax4.set_xlabel("median counterfactual delta from baseline")
    ax4.set_title("d  Perturbation-response counterfactuals", loc="left", fontweight="bold")

    fig.text(
        0.01,
        0.01,
        "Mechanistic additions distinguish imposed artificial perturbation pathways from observational human profile pathways. "
        "Asterisks mark bootstrap/FDR-supported indirect effects.",
        fontsize=7,
    )
    fig.savefig(OUT / "fig_mechanistic_insights.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_mechanistic_insights.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_mechanistic_insights.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(
    art_med: pd.DataFrame,
    hum_med: pd.DataFrame,
    attribution: pd.DataFrame,
    folds: pd.DataFrame,
    cf_summary: pd.DataFrame,
    human_cf: pd.DataFrame,
) -> None:
    def compact_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No rows passed this filter._"
        return df[cols].to_csv(index=False).strip()

    art_hits = art_med[art_med["passes_mechanistic_screen"]].sort_values("bootstrap_q_value").head(8)
    hum_hits = hum_med[hum_med["passes_mechanistic_screen"]].sort_values("bootstrap_q_value").head(8)
    top_features = attribution[attribution["feature"].notna()].sort_values("mean_importance", ascending=False).head(8)
    folds0 = folds.iloc[0] if len(folds) else pd.Series(dtype=float)
    text = [
        "# Mechanistic insight analyses",
        "",
        "These analyses add mechanistic tests requested after the NHB-oriented draft review.",
        "",
        "## Analysis 1: mechanistic pathway / mediation",
        "",
        "Artificial-agent mediation treats perturbation family as imposed by construction. Human pathway tests are observational and control dataset, task and the complementary profile.",
        "",
        f"- Artificial supported indirect pathways: {int(art_med['passes_mechanistic_screen'].sum())} / {len(art_med)}.",
        f"- Human supported observational indirect pathways: {int(hum_med['passes_mechanistic_screen'].sum())} / {len(hum_med)}.",
        "",
        "Top artificial pathways:",
        compact_table(art_hits, ["mediator", "outcome", "indirect_effect", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_q_value"]),
        "",
        "Top human observational pathways:",
        compact_table(hum_hits, ["x", "mediator", "outcome", "indirect_effect", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_q_value"]),
        "",
        "## Analysis 2: feature-to-mechanism attribution",
        "",
        f"Grouped cross-validated family classification from fingerprint features achieved overall balanced accuracy = {folds0.get('overall_balanced_accuracy', np.nan):.3f} and AUC = {folds0.get('overall_auc', np.nan):.3f}.",
        "",
        "Top attribution features:",
        compact_table(top_features, ["feature", "feature_group", "mean_importance", "rank"]),
        "",
        "## Analysis 3: counterfactual perturbation response",
        "",
        "Artificial state interventions are graph-preserving perturbations applied to a trained model. Capacity counterfactuals compare graph-altering hidden-size scale against the h=1 baseline within architecture and seed. Human high-versus-low profile contrasts are calibrated observational proxies, not experimental interventions.",
        "",
        compact_table(cf_summary.sort_values("q_value").head(10), ["counterfactual_type", "outcome", "n_rows", "median_delta", "q_value"]),
        "",
        "## Manuscript language",
        "",
        "Recommended claim: capacity is mechanistically linked to recurrent trajectory geometry and load-sensitive behavioural cost; state is mechanistically linked to operating-regime variability/reliability. Avoid claiming that human observational pathways prove causal neural mechanisms.",
    ]
    (OUT / "mechanistic_insight_report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    agents = pd.read_csv(TABLES / "architecture_artificial_agent_fingerprints.csv")
    human = pd.read_csv(FANCY_SOURCE / "fancy_fig2_state_capacity_landscape_source_data.csv")

    art_med = artificial_mediation(agents)
    hum_med = human_mediation(human)
    attribution, folds = attribution_analysis(agents)
    cf_deltas, cf_summary = artificial_counterfactuals(agents)
    human_cf = human_counterfactual_proxy(human)

    art_med.to_csv(OUT / "mechanistic_artificial_mediation.csv", index=False)
    hum_med.to_csv(OUT / "mechanistic_human_observational_mediation.csv", index=False)
    attribution.to_csv(OUT / "mechanistic_feature_attribution.csv", index=False)
    folds.to_csv(OUT / "mechanistic_feature_attribution_cv.csv", index=False)
    cf_deltas.to_csv(OUT / "mechanistic_artificial_counterfactual_deltas.csv", index=False)
    cf_summary.to_csv(OUT / "mechanistic_artificial_counterfactual_summary.csv", index=False)
    human_cf.to_csv(OUT / "mechanistic_human_profile_contrast_proxy.csv", index=False)

    with pd.ExcelWriter(OUT / "mechanistic_insights_source_data.xlsx", engine="openpyxl") as writer:
        art_med.to_excel(writer, sheet_name="artificial_mediation", index=False)
        hum_med.to_excel(writer, sheet_name="human_observ_mediation", index=False)
        attribution.to_excel(writer, sheet_name="feature_attribution", index=False)
        folds.to_excel(writer, sheet_name="attribution_cv", index=False)
        cf_summary.to_excel(writer, sheet_name="art_counterfactual_summary", index=False)
        cf_deltas.to_excel(writer, sheet_name="art_counterfactual_deltas", index=False)
        human_cf.to_excel(writer, sheet_name="human_profile_proxy", index=False)

    make_figure(art_med, hum_med, attribution, cf_summary, human_cf)
    write_report(art_med, hum_med, attribution, folds, cf_summary, human_cf)
    print(f"Wrote mechanistic insights to {OUT}")


if __name__ == "__main__":
    main()
