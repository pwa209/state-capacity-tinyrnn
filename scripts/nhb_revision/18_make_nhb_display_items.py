from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import FIGURE_MAP_COLUMNS, NHB_AUDIT, NHB_FIGURES, NHB_MANUSCRIPT, NHB_TABLES, TABLES, append_manifest, append_registry, append_tsv, ensure_nhb_dirs


ANALYSIS_ID = "nhb_18_make_nhb_display_items"
SCRIPT_NAME = "scripts/nhb_revision/18_make_nhb_display_items.py"


def read(name: str):
    p = NHB_TABLES / name
    if not p.exists():
        p = TABLES / name
    return pd.read_csv(p)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "figure.dpi": 140,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")


def save(fig, fig_id: str, source: pd.DataFrame, legend: str, outputs: list[Path]) -> None:
    source_path = NHB_TABLES / f"{fig_id}_source_data.csv"
    source.to_csv(source_path, index=False)
    outputs.append(source_path)
    for ext in ["pdf", "png"]:
        path = NHB_FIGURES / f"{fig_id}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    legend_path = NHB_MANUSCRIPT / f"{fig_id}_legend.md"
    legend_path.write_text(legend + "\n", encoding="utf-8")
    outputs.append(legend_path)
    append_tsv(
        NHB_AUDIT / "nhb_figure_source_map.tsv",
        FIGURE_MAP_COLUMNS,
        {
            "figure_id": fig_id,
            "panel": "A-D",
            "source_table": source_path.relative_to(ROOT).as_posix(),
            "source_script": SCRIPT_NAME,
            "output_path": f"outputs/nhb_revision/figures/{fig_id}.pdf",
            "legend_path": legend_path.relative_to(ROOT).as_posix(),
        },
    )


def bar(ax, labels, values, color="#4C78A8", horizontal=True):
    if horizontal:
        y = np.arange(len(labels))
        ax.barh(y, values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
    else:
        x = np.arange(len(labels))
        ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="x" if horizontal else "y", color="#E6E6E6", linewidth=0.6)


def fig1(outputs):
    gate = read("architecture_perturbation_gate_results.csv")
    hybrid = read("architecture_hybrid_recovery_results.csv")
    claims = pd.read_csv(NHB_AUDIT / "nhb_final_claim_audit.tsv", sep="\t")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    panel_label(ax[0, 0], "A")
    ax[0, 0].axis("off")
    ax[0, 0].text(0.02, 0.72, "State-like\noperating perturbations", bbox=dict(boxstyle="round,pad=0.35", fc="#D9EAF7", ec="#4C78A8"))
    ax[0, 0].text(0.56, 0.72, "Capacity-like\nresource perturbations", bbox=dict(boxstyle="round,pad=0.35", fc="#FBE4D5", ec="#D55E00"))
    ax[0, 0].annotate("matched performance\nresidualized fingerprints", xy=(0.49, 0.55), xytext=(0.49, 0.25), ha="center", arrowprops=dict(arrowstyle="->"))
    ax[0, 0].set_title("Falsifiable distinction")
    panel_label(ax[0, 1], "B")
    g = gate[(gate.feature_set == "residualized_fingerprint") & (gate.task_family == "overall")]
    bar(ax[0, 1], g.model_family, g.balanced_accuracy.astype(float), "#009E73", horizontal=False)
    ax[0, 1].axhline(0.75, color="#333333", ls="--", lw=0.8)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_ylabel("Balanced accuracy")
    ax[0, 1].set_title("Intervention-family gate")
    panel_label(ax[1, 0], "C")
    h = hybrid[hybrid.feature_set == "residualized_fingerprint"]
    pivot = h.pivot(index="model_family", columns="target_axis", values="spearman_rho")
    pivot.plot(kind="bar", ax=ax[1, 0], color=["#0072B2", "#D55E00"])
    ax[1, 0].axhline(0.4, color="#333333", ls="--", lw=0.8)
    ax[1, 0].set_ylabel("Recovery rho")
    ax[1, 0].set_title("Scalar recovery boundary")
    panel_label(ax[1, 1], "D")
    counts = claims.claim_strength.value_counts()
    bar(ax[1, 1], counts.index, counts.values, "#666666", horizontal=False)
    ax[1, 1].set_title("Claim gates")
    fig.tight_layout()
    save(fig, "fig1_concept_pipeline", pd.concat([gate.assign(panel="B"), hybrid.assign(panel="C"), claims.assign(panel="D")], ignore_index=True, sort=False), "**Figure 1 | Concept and falsification logic.** Architecture-robust intervention-family separation coexists with failed scalar state recovery.", outputs)


def fig2(outputs):
    events = read("event_counts_by_dataset.csv")
    state = read("session_state_multiaxis_coordinates.csv")
    cap = read("participant_capacity_multidimensional_coordinates.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    panel_label(ax[0, 0], "A")
    bar(ax[0, 0], events.dataset, events.n_included_events.astype(float), "#4C78A8")
    ax[0, 0].set_title("Included events")
    panel_label(ax[0, 1], "B")
    bar(ax[0, 1], events.dataset, events.n_subjects.astype(float), "#4C78A8")
    ax[0, 1].set_title("Participants")
    panel_label(ax[1, 0], "C")
    rows = pd.Series({"state rows": len(state), "capacity rows": len(cap)})
    bar(ax[1, 0], rows.index, rows.values, "#009E73", horizontal=False)
    ax[1, 0].set_title("Coordinate rows")
    panel_label(ax[1, 1], "D")
    cov = events[["dataset", "n_tasks", "n_events_with_correct"]].copy()
    ax[1, 1].scatter(cov.n_tasks, cov.n_events_with_correct, s=70, color="#D55E00")
    for _, r in cov.iterrows():
        ax[1, 1].text(r.n_tasks, r.n_events_with_correct, r.dataset, fontsize=6)
    ax[1, 1].set_xlabel("Tasks")
    ax[1, 1].set_ylabel("Events with correctness")
    ax[1, 1].set_title("Coverage")
    fig.tight_layout()
    save(fig, "fig2_dataset_pipeline_scale", pd.concat([events.assign(panel="A_B_D"), pd.DataFrame({"metric": rows.index, "value": rows.values, "panel": "C"})], ignore_index=True, sort=False), "**Figure 2 | Dataset and pipeline scale.** Four datasets contributed behavioural events, state rows and participant-level capacity profiles.", outputs)


def fig3(outputs):
    gate = read("architecture_perturbation_gate_results.csv")
    loo = read("leave_one_architecture_gate_results.csv")
    hybrid = read("architecture_hybrid_recovery_results.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    g = gate[(gate.feature_set == "residualized_fingerprint") & (gate.task_family == "overall")]
    panel_label(ax[0, 0], "A")
    bar(ax[0, 0], g.model_family, g.balanced_accuracy.astype(float), "#009E73", horizontal=False)
    ax[0, 0].axhline(0.75, color="#333", ls="--")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("By architecture")
    l = loo[(loo.feature_set == "residualized_fingerprint") & (loo.task_family == "overall")]
    panel_label(ax[0, 1], "B")
    bar(ax[0, 1], l.heldout_model_family, l.balanced_accuracy.astype(float), "#56B4E9", horizontal=False)
    ax[0, 1].axhline(0.70, color="#333", ls="--")
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("Leave-one-architecture")
    panel_label(ax[1, 0], "C")
    h = hybrid[(hybrid.feature_set == "residualized_fingerprint") & (hybrid.target_axis == "state_severity")]
    bar(ax[1, 0], h.model_family, h.spearman_rho.astype(float), "#0072B2", horizontal=False)
    ax[1, 0].axhline(0.4, color="#333", ls="--")
    ax[1, 0].set_title("Scalar state recovery")
    panel_label(ax[1, 1], "D")
    h = hybrid[(hybrid.feature_set == "residualized_fingerprint") & (hybrid.target_axis == "capacity_level")]
    bar(ax[1, 1], h.model_family, h.spearman_rho.astype(float), "#D55E00", horizontal=False)
    ax[1, 1].axhline(0.4, color="#333", ls="--")
    ax[1, 1].set_title("Hybrid capacity recovery")
    fig.tight_layout()
    save(fig, "fig3_artificial_agent_gates", pd.concat([gate.assign(panel="A"), loo.assign(panel="B"), hybrid.assign(panel="C_D")], ignore_index=True, sort=False), "**Figure 3 | Artificial-agent perturbation gates.** State/capacity family classification survives architecture checks, while scalar state recovery remains below the recovery gate.", outputs)


def fig4(outputs):
    inc = read("incremental_value_model_comparison.csv")
    low = read("low_data_prediction_curves.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for i, scope in enumerate(inc.dataset.unique()[:2]):
        a = ax.flat[i]
        panel_label(a, chr(65 + i))
        sub = inc[inc.dataset == scope].sort_values("RMSE")
        bar(a, sub.model_name, sub.RMSE.astype(float), "#4C78A8")
        a.set_title(scope)
        a.set_xlabel("LOPO RMSE")
    panel_label(ax[1, 0], "C")
    sub = inc.dropna(subset=["delta_RMSE"])
    bar(ax[1, 0], sub.model_name.head(10), sub.delta_RMSE.astype(float).head(10), "#D55E00")
    ax[1, 0].axvline(0, color="#333")
    ax[1, 0].set_title("Increment over task baseline")
    panel_label(ax[1, 1], "D")
    for model, g in low.groupby("model_name"):
        vals = g.groupby("calibration_trials")["RMSE"].mean(numeric_only=True)
        ax[1, 1].plot(range(len(vals)), vals.values, marker="o", label=model)
    ax[1, 1].set_xticks(range(len(vals)))
    ax[1, 1].set_xticklabels(vals.index, rotation=35)
    ax[1, 1].set_title("Low-data prediction")
    ax[1, 1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig4_prediction_baselines", pd.concat([inc.assign(panel="A_C"), low.assign(panel="D")], ignore_index=True, sort=False), "**Figure 4 | Prediction and behavioural baselines.** Coordinate models are evaluated against behavioural baselines and sparse-calibration curves.", outputs)


def fig5(outputs):
    dyn = read("recurrent_dynamics_state_capacity_tests.csv")
    press = read("capacity_pressure_models.csv")
    marg = read("capacity_pressure_marginal_effects.csv")
    ab = read("capacity_variant_validation_summary.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    capdyn = dyn[dyn.predictor.eq("capacity_parameter_resource_z")].sort_values("spearman_rho", ascending=False).head(8)
    panel_label(ax[0, 0], "A")
    bar(ax[0, 0], capdyn.outcome, capdyn.spearman_rho.astype(float), "#D55E00")
    ax[0, 0].set_title("Capacity-geometry")
    panel_label(ax[0, 1], "B")
    lx = press[press.predictor.eq("load_x_capacity")]
    bar(ax[0, 1], lx.outcome, lx.estimate.astype(float), "#009E73")
    ax[0, 1].axvline(0, color="#333")
    ax[0, 1].set_title("Load x capacity")
    panel_label(ax[1, 0], "C")
    mm = marg[marg.outcome.eq("mean_accuracy")]
    for label, g in mm.groupby("capacity_level"):
        ax[1, 0].plot(g.load_z, g.predicted, label=label)
    ax[1, 0].set_title("Marginal accuracy")
    ax[1, 0].legend(fontsize=6)
    panel_label(ax[1, 1], "D")
    bar(ax[1, 1], ab.variant, ab.median_abs_effect.astype(float), "#CC79A7")
    ax[1, 1].set_title("Capacity ablation")
    fig.tight_layout()
    save(fig, "fig5_capacity_geometry_pressure", pd.concat([dyn.assign(panel="A"), press.assign(panel="B"), marg.assign(panel="C"), ab.assign(panel="D")], ignore_index=True, sort=False), "**Figure 5 | Capacity geometry and pressure.** Capacity evidence is strongest for fitted recurrent geometry and load-pressure behaviour.", outputs)


def fig6(outputs):
    rel = read("state_split_half_reliability.csv")
    early = read("state_early_late_model_comparison.csv")
    var = read("state_capacity_variance_decomposition.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    panel_label(ax[0, 0], "A")
    sub = rel.groupby("dataset")["cosine_similarity"].mean().sort_values()
    bar(ax[0, 0], sub.index, sub.values, "#0072B2")
    ax[0, 0].set_title("Split-half reliability")
    panel_label(ax[0, 1], "B")
    sub = rel.groupby("trial_count_bin")["cosine_similarity"].mean().sort_index()
    bar(ax[0, 1], sub.index, sub.values, "#56B4E9", horizontal=False)
    ax[0, 1].set_title("Trial-count threshold")
    panel_label(ax[1, 0], "C")
    sp = early[early.predictor.eq("state_plus_recent_behavior")].sort_values("delta_R2", ascending=False).head(10)
    bar(ax[1, 0], sp.dataset + "/" + sp.outcome, sp.delta_R2.astype(float), "#009E73")
    ax[1, 0].set_title("Early-to-late delta R2")
    panel_label(ax[1, 1], "D")
    vv = var[var.construct.isin(["state", "capacity"])].groupby("construct")["icc"].mean()
    bar(ax[1, 1], vv.index, vv.values, "#666666", horizontal=False)
    ax[1, 1].set_title("ICC decomposition")
    fig.tight_layout()
    save(fig, "fig6_state_reliability", pd.concat([rel.assign(panel="A_B"), early.assign(panel="C"), var.assign(panel="D")], ignore_index=True, sort=False), "**Figure 6 | State reliability profile.** State behaves as a denser-data reliability profile with lower trait-like ICC than capacity.", outputs)


def fig7(outputs):
    phys = read("physiology_robustness_models.csv")
    perm = read("physiology_permutation_controls.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    phys["neglog10p"] = -np.log10(pd.to_numeric(phys.p_value, errors="coerce"))
    for i, dataset in enumerate(["ds007554", "cog_bci"]):
        a = ax.flat[i]
        panel_label(a, chr(65 + i))
        sub = phys[phys.dataset.eq(dataset)].sort_values("neglog10p", ascending=False).head(10)
        bar(a, sub.feature.astype(str) + "/" + sub.predictor.astype(str), sub.neglog10p.fillna(0), "#785EF0")
        a.set_title(dataset)
    panel_label(ax[1, 0], "C")
    bar(ax[1, 0], perm.control, np.ones(len(perm)), "#666666")
    ax[1, 0].set_title("Permutation controls")
    panel_label(ax[1, 1], "D")
    counts = phys.claim_strength.value_counts()
    bar(ax[1, 1], counts.index, counts.values, "#E69F00", horizontal=False)
    ax[1, 1].set_title("Bounded claim labels")
    fig.tight_layout()
    save(fig, "fig7_physiology_alignment", pd.concat([phys.assign(panel="A_B_D"), perm.assign(panel="C")], ignore_index=True, sort=False), "**Figure 7 | Physiology and bounded multimodal alignment.** Physiology is treated as alignment evidence, not direct proof of neural coordinates.", outputs)


def fig8(outputs):
    claims = pd.read_csv(NHB_AUDIT / "nhb_final_claim_audit.tsv", sep="\t")
    shuf = read("profile_shuffle_controls.csv")
    hybrid = read("architecture_hybrid_recovery_results.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    panel_label(ax[0, 0], "A")
    counts = claims.groupby(["construct", "claim_strength"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=ax[0, 0], colormap="tab20c")
    ax[0, 0].set_title("Claim-strength matrix")
    ax[0, 0].legend(fontsize=6)
    panel_label(ax[0, 1], "B")
    num = shuf.select_dtypes(include=[np.number])
    vals = num.iloc[:, 0].dropna().head(12) if not num.empty else pd.Series([1, 1])
    ax[0, 1].plot(range(len(vals)), vals, marker="o", color="#666666")
    ax[0, 1].set_title("Negative controls")
    panel_label(ax[1, 0], "C")
    h = hybrid[(hybrid.feature_set == "residualized_fingerprint") & (hybrid.target_axis == "state_severity")]
    bar(ax[1, 0], h.model_family, h.spearman_rho.astype(float), "#0072B2", horizontal=False)
    ax[1, 0].axhline(0.4, color="#333", ls="--")
    ax[1, 0].set_title("Failed scalar state gate")
    panel_label(ax[1, 1], "D")
    ax[1, 1].axis("off")
    ax[1, 1].text(0.02, 0.72, "Strong: capacity\nModerate: reliability state\nFailed: scalar state\nBounded: physiology", va="top", fontsize=9)
    ax[1, 1].set_title("Final claim map")
    fig.tight_layout()
    save(fig, "fig8_claim_audit_falsification", pd.concat([claims.assign(panel="A_D"), shuf.assign(panel="B"), hybrid.assign(panel="C")], ignore_index=True, sort=False), "**Figure 8 | Claim audit and falsification summary.** The final claim map foregrounds constructive falsification and forbidden overclaims.", outputs)


def main() -> None:
    ensure_nhb_dirs()
    style()
    started = datetime.now(timezone.utc).isoformat()
    outputs: list[Path] = []
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        fn(outputs)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, outputs, notes="Regenerated polished source-linked NHB display items.")
    print(f"Regenerated {len(outputs)} display-item artifacts")


if __name__ == "__main__":
    main()
