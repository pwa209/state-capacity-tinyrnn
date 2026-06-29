from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"


def ensure_dirs() -> None:
    for path in [FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, name: str, source: pd.DataFrame, dictionary: str) -> None:
    source_path = SOURCE_DATA / f"{name}_source.csv"
    source.to_csv(source_path, index=False)
    (SOURCE_DATA / f"{name}_data_dictionary.md").write_text(dictionary, encoding="utf-8")
    (SOURCE_DATA / f"{name}_script_used.txt").write_text(
        "state_capacity_tinyrnn/scripts/16_statistics_figures/make_figures.py",
        encoding="utf-8",
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def barh(ax: plt.Axes, labels: pd.Series, values: pd.Series, color: str) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)


def figure_1() -> None:
    events = pd.read_csv(TABLES / "event_counts_by_dataset.csv")
    effects = pd.read_csv(TABLES / "master_effects_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    plot = events.sort_values("n_included_events")
    barh(axes[0], plot["dataset"], plot["n_included_events"], "#4C78A8")
    axes[0].set_xlabel("Included events")
    axes[0].set_title("Dataset scale")
    piv = effects.pivot_table(index="construct", columns="claim_strength", values="n_effects", aggfunc="sum").fillna(0)
    piv.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20c", legend=False)
    axes[1].set_ylabel("Effect rows")
    axes[1].set_title("Claim audit coverage")
    fig.tight_layout()
    src = pd.concat([events.assign(panel="dataset_scale"), effects.assign(panel="claim_coverage")], ignore_index=True, sort=False)
    save_figure(
        fig,
        "figure_1_pipeline_and_claim_coverage",
        src,
        "# Figure 1 data dictionary\n\nDataset event counts and master-effect claim-strength counts by construct.",
    )


def figure_2() -> None:
    gate = pd.read_csv(TABLES / "ann_intervention_gate_results.csv")
    hybrid = pd.read_csv(TABLES / "ann_hybrid_recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    clf = gate[gate["analysis_type"].eq("matched_state_capacity_classification") & gate["feature_set"].eq("residualized_fingerprint")]
    barh(axes[0], clf["task_family"], clf["observed"], "#009E73")
    axes[0].axvline(0.5, color="#555555", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Balanced accuracy")
    axes[0].set_title("State vs capacity classifier")
    hy = hybrid[hybrid["feature_set"].eq("residualized_fingerprint")].copy()
    colors = np.where(hy["pass_gate"], "#009E73", "#D55E00")
    axes[1].bar(hy["target_axis"], hy["spearman_rho"], color=colors)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Hybrid recovery")
    fig.tight_layout()
    src = pd.concat([gate.assign(panel="ann_gate"), hybrid.assign(panel="hybrid_recovery")], ignore_index=True, sort=False)
    save_figure(
        fig,
        "figure_2_ann_gate",
        src,
        "# Figure 2 data dictionary\n\nANN intervention gate and hybrid recovery rows. Colors indicate pass/fail for hybrid gates.",
    )


def figure_3() -> None:
    models = pd.read_csv(TABLES / "ds007554_discovery_model_comparison.csv")
    boot = pd.read_csv(TABLES / "ds007554_bootstrap_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    primary = models[models["analysis_scope"].eq("ds007554_primary_reconstructed")].sort_values("lopo_rmse")
    barh(axes[0], primary["model_name"], primary["lopo_rmse"], "#4C78A8")
    axes[0].set_xlabel("LOPO RMSE")
    axes[0].set_title("ds007554 discovery")
    b = boot[boot["analysis_scope"].eq("ds007554_primary_reconstructed")].copy()
    y = np.arange(len(b))
    axes[1].errorbar(b["spearman_rho"], y, xerr=[b["spearman_rho"] - b["bootstrap_ci_low"], b["bootstrap_ci_high"] - b["spearman_rho"]], fmt="o", color="#D55E00")
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(b["effect_name"])
    axes[1].set_xlabel("Spearman rho")
    axes[1].set_title("State/capacity associations")
    fig.tight_layout()
    src = pd.concat([models.assign(panel="model_comparison"), boot.assign(panel="bootstrap_effects")], ignore_index=True, sort=False)
    save_figure(fig, "figure_3_ds007554_discovery", src, "# Figure 3 data dictionary\n\nds007554 model comparison and bootstrap association effects.")


def figure_4() -> None:
    dyn = pd.read_csv(TABLES / "recurrent_dynamics_state_capacity_tests.csv")
    decoder = pd.read_csv(TABLES / "latent_decoder_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sel = dyn[dyn["predictor"].isin(["state_parameter_instability_z", "capacity_parameter_resource_z", "optimized_state_profile_z", "optimized_capacity_profile_z"])].copy()
    sel = sel[sel["outcome"].isin(["trajectory_cov_rank", "trajectory_radius", "hidden_variability", "trajectory_step_norm_mean"])]
    pivot = sel.pivot_table(index="outcome", columns="predictor", values="spearman_rho")
    axes[0].imshow(pivot.fillna(0), cmap="coolwarm", vmin=-0.8, vmax=0.8)
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_title("Recurrent geometry")
    ok = decoder[decoder["status"].eq("ok")]
    axes[1].bar(ok["target"], ok["balanced_accuracy"], color="#009E73")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Latent decoders")
    fig.tight_layout()
    src = pd.concat([dyn.assign(panel="geometry_tests"), decoder.assign(panel="latent_decoders")], ignore_index=True, sort=False)
    save_figure(fig, "figure_4_recurrent_dynamics", src, "# Figure 4 data dictionary\n\nRecurrent dynamics Spearman tests and latent decoder results.")


def figure_5() -> None:
    neuro = pd.read_csv(TABLES / "ds007554_neurophys_models.csv")
    direct = neuro[neuro["analysis_family"].eq("state_capacity_coordinate_model")].copy()
    direct["neglog10p"] = -np.log10(pd.to_numeric(direct["p_value"], errors="coerce"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    plot = direct.sort_values("p_value").head(24)
    colors = np.where(plot["predictor"].eq("state"), "#0072B2", "#D55E00")
    barh(ax, plot["feature"] + " / " + plot["predictor"], plot["neglog10p"], "#999999")
    for patch, color in zip(ax.patches, colors):
        patch.set_color(color)
    ax.axvline(-np.log10(0.05), color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("-log10 p")
    ax.set_title("ds007554 direct neurophysiology")
    fig.tight_layout()
    save_figure(fig, "figure_5_ds007554_neurophysiology", direct, "# Figure 5 data dictionary\n\nDirect ds007554 state/capacity neurophysiology model rows.")


def figure_6() -> None:
    cog = pd.read_csv(TABLES / "cog_bci_validation_models.csv")
    tested = cog[cog["claim_status"].astype(str).eq("tested")].copy()
    tested["neglog10q"] = -np.log10(pd.to_numeric(tested["q_value"], errors="coerce"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    state = tested[tested["x"].eq("state")].sort_values("q_value").head(8)
    cap = tested[tested["x"].eq("capacity")].sort_values("q_value").head(8)
    barh(axes[0], state["y"], state["neglog10q"], "#0072B2")
    axes[0].set_xlabel("-log10 q")
    axes[0].set_title("COG state")
    barh(axes[1], cap["y"], cap["neglog10q"], "#D55E00")
    axes[1].set_xlabel("-log10 q")
    axes[1].set_title("COG capacity")
    fig.tight_layout()
    save_figure(fig, "figure_6_cog_bci_validation", cog, "# Figure 6 data dictionary\n\nCOG-BCI validation models, including state behavior and capacity consistency/EEG rows.")


def figure_7() -> None:
    tu = pd.read_csv(TABLES / "tu_berlin_load_validation.csv")
    hbn = pd.read_csv(TABLES / "hbn_scalability_tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    tu_plot = tu.sort_values("q_value").head(8)
    tu_plot["neglog10q"] = -np.log10(pd.to_numeric(tu_plot["q_value"], errors="coerce"))
    barh(axes[0], tu_plot["model_name"], tu_plot["neglog10q"], "#4C78A8")
    axes[0].set_xlabel("-log10 q")
    axes[0].set_title("TU Berlin validation")
    hbn_plot = hbn.sort_values("q_value").head(8)
    hbn_plot["neglog10q"] = -np.log10(pd.to_numeric(hbn_plot["q_value"], errors="coerce"))
    barh(axes[1], hbn_plot["analysis"], hbn_plot["neglog10q"], "#009E73")
    axes[1].set_xlabel("-log10 q")
    axes[1].set_title("HBN scalability")
    fig.tight_layout()
    src = pd.concat([tu.assign(panel="tu_berlin"), hbn.assign(panel="hbn")], ignore_index=True, sort=False)
    save_figure(fig, "figure_7_tu_hbn_validation", src, "# Figure 7 data dictionary\n\nTU Berlin capacity-pressure/load validation and HBN scalability tests.")


def figure_8() -> None:
    rob = pd.read_csv(TABLES / "robustness_master_table.csv")
    fals = pd.read_csv(TABLES / "falsification_tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    counts = rob.groupby(["construct", "pass_robustness"]).size().reset_index(name="n")
    piv = counts.pivot(index="construct", columns="pass_robustness", values="n").fillna(0)
    piv.plot(kind="bar", stacked=True, ax=axes[0], color={False: "#999999", True: "#009E73"}, legend=False)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Robustness")
    f = fals.head(12)
    colors = np.where(f["pass_robustness"], "#009E73", "#D55E00")
    barh(axes[1], f["claim_key"], f["effect"], "#999999")
    for patch, color in zip(axes[1].patches, colors):
        patch.set_color(color)
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Falsification")
    fig.tight_layout()
    src = pd.concat([rob.assign(panel="robustness"), fals.assign(panel="falsification")], ignore_index=True, sort=False)
    save_figure(fig, "figure_8_robustness_falsification", src, "# Figure 8 data dictionary\n\nStep 16 robustness and falsification source rows.")


def main() -> int:
    ensure_dirs()
    set_style()
    makers: list[Callable[[], None]] = [figure_1, figure_2, figure_3, figure_4, figure_5, figure_6, figure_7, figure_8]
    for maker in makers:
        maker()
    audit = [
        "# Step 18 Figure Audit",
        "",
        "- Generated final Figures 1-8 as PNG, PDF and SVG.",
        "- Every final figure has a matching source CSV, data dictionary and script-used file.",
        "- Figures cover state, capacity, interactions, neurophysiology, external validation, scalability and robustness/falsification.",
    ]
    (AUDIT / "step18_figures_audit.md").write_text("\n".join(audit), encoding="utf-8")
    status = {
        "status": "implemented_and_run",
        "n_final_figures": 8,
        "figure_prefixes": [
            "figure_1_pipeline_and_claim_coverage",
            "figure_2_ann_gate",
            "figure_3_ds007554_discovery",
            "figure_4_recurrent_dynamics",
            "figure_5_ds007554_neurophysiology",
            "figure_6_cog_bci_validation",
            "figure_7_tu_hbn_validation",
            "figure_8_robustness_falsification",
        ],
    }
    (LOGS / "step18_figures_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP18_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
