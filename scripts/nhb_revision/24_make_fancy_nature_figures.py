from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB_TABLES, TABLES, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_24_fancy_nature_figures"
SCRIPT_NAME = "scripts/nhb_revision/24_make_fancy_nature_figures.py"
FIG_DIR = ROOT / "outputs" / "nhb_revision" / "fancy_figures"
SRC_DIR = FIG_DIR / "source_data"


PALETTE = {
    "ink": "#14213D",
    "muted": "#5C6672",
    "grid": "#D9DEE7",
    "state": "#2A9D8F",
    "capacity": "#E76F51",
    "interaction": "#6D597A",
    "control": "#7A8793",
    "exploratory": "#F4A261",
    "negative": "#9AA3AD",
    "blue": "#457B9D",
    "pale": "#F7F8FA",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def table(name: str, nhb: bool = True) -> pd.DataFrame:
    return read_csv((NHB_TABLES if nhb else TABLES) / name)


def save_source(name: str, df: pd.DataFrame) -> Path:
    path = SRC_DIR / f"{name}_source_data.csv"
    df.to_csv(path, index=False)
    return path


def finalize(fig: plt.Figure, name: str, source: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    source_path = save_source(name, source)
    paths.append(source_path)
    for ext in ("pdf", "svg", "png"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=450, bbox_inches="tight", facecolor="white")
        paths.append(out)
    plt.close(fig)
    return paths


def setup() -> None:
    ensure_nhb_dirs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.6,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=PALETTE["ink"])


def add_flow_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.18) -> None:
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        0.18,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + 0.09, text, ha="center", va="center", color=PALETTE["ink"], fontsize=7, linespacing=1.1)


def figure_1() -> list[Path]:
    counts = table("event_counts_by_dataset.csv", nhb=False)
    claims = read_csv(ROOT / "outputs" / "nhb_revision" / "audit" / "nhb_final_claim_audit.tsv")
    arch = table("architecture_perturbation_gate_results.csv")
    source = pd.concat(
        [
            counts.assign(panel="dataset_scale"),
            claims.assign(panel="claim_audit"),
            arch[arch.get("task_family", "") == "overall"].assign(panel="architecture_gate") if not arch.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.05, 1.05], hspace=0.45, wspace=0.45)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_axis_off()
    boxes = [
        (0.02, "Open datasets\nbehaviour + physiology", PALETTE["blue"]),
        (0.24, "Tiny RNN\nstate/capacity fits", PALETTE["ink"]),
        (0.46, "Artificial-agent\nperturbation gate", PALETTE["interaction"]),
        (0.68, "Validation\nstate, capacity, controls", PALETTE["state"]),
    ]
    for x, label, color in boxes:
        add_flow_box(ax_flow, (x, 0.42), label, color)
    for x in [0.20, 0.42, 0.64]:
        ax_flow.annotate("", xy=(x + 0.035, 0.51), xytext=(x, 0.51), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PALETTE["muted"]))
    ax_flow.text(0.5, 0.91, "Machine-defined profiles are tested before human claims are written", ha="center", color=PALETTE["ink"], fontsize=9, fontweight="bold")
    panel_label(ax_flow, "a")

    ax_counts = fig.add_subplot(gs[1, 0])
    if not counts.empty:
        c = counts.copy()
        xcol = "included_events" if "included_events" in c.columns else c.select_dtypes("number").columns[-1]
        c = c.sort_values(xcol, ascending=True)
        ax_counts.barh(c["dataset"].astype(str), c[xcol].astype(float), color=PALETTE["blue"], alpha=0.9)
        ax_counts.set_xlabel("Included events")
        ax_counts.grid(axis="x", color=PALETTE["grid"], lw=0.5)
    ax_counts.set_title("Dataset scale")
    panel_label(ax_counts, "b")

    ax_claim = fig.add_subplot(gs[1, 1])
    if not claims.empty:
        strength = claims["claim_strength"].value_counts()
        colors = [PALETTE["state"] if k == "strong" else PALETTE["blue"] if k == "moderate" else PALETTE["exploratory"] if k == "exploratory" else PALETTE["interaction"] for k in strength.index]
        ax_claim.pie(strength.values, labels=strength.index, startangle=90, colors=colors, wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(fontsize=6.5))
        ax_claim.text(0, 0, f"{strength.sum()}\nclaims", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
    ax_claim.set_title("Claim audit")
    panel_label(ax_claim, "c")

    ax_gate = fig.add_subplot(gs[1, 2])
    if not arch.empty:
        g = arch[(arch["task_family"] == "overall") & (arch["feature_set"] == "residualized_fingerprint")].copy()
        ax_gate.bar(g["model_family"], g["balanced_accuracy"].astype(float), color=[PALETTE["state"], PALETTE["capacity"], PALETTE["interaction"]], alpha=0.9)
        ax_gate.axhline(0.5, color=PALETTE["muted"], lw=0.8, ls="--")
        ax_gate.set_ylim(0.45, 1.0)
        ax_gate.set_ylabel("Balanced accuracy")
        ax_gate.tick_params(axis="x", rotation=25)
    ax_gate.set_title("True architecture gate")
    panel_label(ax_gate, "d")
    fig.suptitle("A falsifiable open-data pipeline for state and capacity", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig1_graphical_abstract", source)


def figure_2() -> list[Path]:
    proj = table("human_state_capacity_multiaxis_projection.csv", nhb=False)
    if proj.empty:
        proj = table("tu_berlin_coordinates.csv", nhb=False)
    source = proj.copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    x = "optimized_capacity_profile_z" if "optimized_capacity_profile_z" in proj.columns else "capacity_multidimensional_summary_z"
    y = "optimized_state_profile_z" if "optimized_state_profile_z" in proj.columns else "state_multidimensional_summary_z"
    if not proj.empty and x in proj and y in proj:
        datasets = sorted(proj["dataset"].dropna().astype(str).unique())
        cmap = [PALETTE["blue"], PALETTE["state"], PALETTE["capacity"], PALETTE["interaction"], PALETTE["control"]]
        for i, ds in enumerate(datasets):
            d = proj[proj["dataset"].astype(str) == ds]
            ax.scatter(d[x], d[y], s=12, alpha=0.45, label=ds, color=cmap[i % len(cmap)], edgecolor="none")
        ax.axhline(0, color=PALETTE["grid"], lw=0.8)
        ax.axvline(0, color=PALETTE["grid"], lw=0.8)
        ax.set_xlabel("Capacity profile (z)")
        ax.set_ylabel("State profile (z)")
        ax.legend(frameon=False, loc="best", markerscale=1.5)
    ax.set_title("Human state-capacity landscape")
    panel_label(ax, "a")

    ax = axes[1]
    if not proj.empty and "dataset" in proj and y in proj:
        order = proj.groupby("dataset")[y].median().sort_values().index
        vals = [proj.loc[proj["dataset"] == ds, y].dropna().to_numpy() for ds in order]
        ax.violinplot(vals, showextrema=False, showmedians=True)
        ax.set_xticks(range(1, len(order) + 1), [str(o).replace("_", "\n") for o in order], rotation=0)
        ax.set_ylabel("State profile")
    ax.set_title("State distribution")
    panel_label(ax, "b")

    ax = axes[2]
    if not proj.empty and "dataset" in proj and x in proj:
        order = proj.groupby("dataset")[x].median().sort_values().index
        vals = [proj.loc[proj["dataset"] == ds, x].dropna().to_numpy() for ds in order]
        ax.violinplot(vals, showextrema=False, showmedians=True)
        ax.set_xticks(range(1, len(order) + 1), [str(o).replace("_", "\n") for o in order], rotation=0)
        ax.set_ylabel("Capacity profile")
    ax.set_title("Capacity distribution")
    panel_label(ax, "c")
    fig.suptitle("State varies across sessions/tasks; capacity is profiled at participant level", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig2_state_capacity_landscape", source)


def figure_3() -> list[Path]:
    gate = table("architecture_perturbation_gate_results.csv")
    hybrid = table("architecture_hybrid_recovery_results.csv")
    leave = table("leave_one_architecture_gate_results.csv")
    source = pd.concat([gate.assign(panel="gate"), hybrid.assign(panel="hybrid"), leave.assign(panel="leave_one_architecture")], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]})
    ax = axes[0]
    g = gate[(gate["task_family"] == "overall") & (gate["feature_set"] == "residualized_fingerprint")].copy() if not gate.empty else pd.DataFrame()
    if not g.empty:
        ax.bar(g["model_family"], g["balanced_accuracy"].astype(float), color=PALETTE["ink"], alpha=0.86)
        for i, r in enumerate(g.itertuples()):
            ax.text(i, float(r.balanced_accuracy) + 0.015, f"{float(r.balanced_accuracy):.2f}", ha="center", fontsize=6.5)
        ax.axhline(0.5, color=PALETTE["muted"], lw=0.8, ls="--")
        ax.set_ylim(0.45, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Balanced accuracy")
    ax.set_title("Residualized family gate")
    panel_label(ax, "a")

    ax = axes[1]
    if not hybrid.empty:
        h = hybrid.pivot_table(index=["model_family", "analysis_type"], columns="feature_set", values="spearman_rho").reset_index()
        labels = [f"{r.model_family}\n{str(r.analysis_type).replace('hybrid_', '').replace('_axis', '')}" for r in h.itertuples()]
        xpos = np.arange(len(h))
        width = 0.38
        ax.bar(xpos - width / 2, h.get("raw_fingerprint", np.nan), width, color=PALETTE["state"], label="Raw")
        ax.bar(xpos + width / 2, h.get("residualized_fingerprint", np.nan), width, color=PALETTE["capacity"], label="Residualized")
        ax.axhline(0, color=PALETTE["muted"], lw=0.7)
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("Spearman rho")
        ax.legend(frameon=False, ncols=2)
    ax.set_title("Hybrid scalar recovery")
    panel_label(ax, "b")

    ax = axes[2]
    if not leave.empty:
        l = leave[leave["feature_set"] == "residualized_fingerprint"]
        ax.bar(l["heldout_model_family"], l["balanced_accuracy"].astype(float), color=PALETTE["interaction"], alpha=0.9)
        ax.axhline(0.5, color=PALETTE["muted"], lw=0.8, ls="--")
        ax.set_ylim(0.45, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Held-out accuracy")
    ax.set_title("Leave-one-architecture")
    panel_label(ax, "c")
    fig.suptitle("Architecture robustness supports family separation but qualifies scalar state", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig3_architecture_robustness", source)


def figure_4() -> list[Path]:
    grid = table("capacity_pressure_marginal_effects.csv")
    models = table("capacity_pressure_models.csv")
    source = pd.concat([grid.assign(panel="surface"), models.assign(panel="model")], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    cmap = LinearSegmentedColormap.from_list("nature_surface", ["#EDF6F9", "#83C5BE", "#006D77"])
    for idx, outcome in enumerate(["mean_accuracy", "rt_median"]):
        ax = axes[idx]
        d = grid[grid["outcome"] == outcome].copy() if not grid.empty else pd.DataFrame()
        if not d.empty:
            pivot = d.pivot_table(index="capacity_level", columns="load_z", values="predicted")
            pivot = pivot.reindex(["low_capacity", "median_capacity", "high_capacity"])
            im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, origin="lower")
            ax.set_yticks(range(len(pivot.index)), [i.replace("_", "\n") for i in pivot.index])
            ax.set_xticks(np.linspace(0, pivot.shape[1] - 1, 4), [f"{v:.1f}" for v in np.linspace(d["load_z"].min(), d["load_z"].max(), 4)])
            ax.set_xlabel("N-back load (z)")
            ax.set_title("Predicted accuracy" if outcome == "mean_accuracy" else "Predicted median RT")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        panel_label(ax, "ab"[idx])
    ax = axes[2]
    if not models.empty:
        m = models[models["predictor"].isin(["load_z", "capacity_parameter_resource_z", "load_x_capacity", "state_parameter_instability_z"])].copy()
        m = m[m["outcome"].isin(["mean_accuracy", "rt_median", "lapse_rate"])]
        m["neglog10q"] = -np.log10(pd.to_numeric(m["q_value"], errors="coerce").clip(lower=1e-300))
        top = m.sort_values("neglog10q", ascending=False).head(8)
        ax.barh(range(len(top)), top["neglog10q"], color=np.where(top["predictor"].eq("load_x_capacity"), PALETTE["capacity"], PALETTE["blue"]))
        ax.set_yticks(range(len(top)), [f"{r.outcome}\n{r.predictor}" for r in top.itertuples()])
        ax.invert_yaxis()
        ax.set_xlabel("-log10(q)")
    ax.set_title("Model evidence")
    panel_label(ax, "c")
    fig.suptitle("Capacity pressure emerges under increasing task load", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig4_capacity_pressure_surface", source)


def figure_5() -> list[Path]:
    var = table("state_capacity_variance_decomposition.csv")
    rel = table("state_bootstrap_reliability.csv")
    early = table("state_early_late_model_comparison.csv")
    source = pd.concat([var.assign(panel="variance"), rel.assign(panel="reliability"), early.assign(panel="early_late")], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]})
    ax = axes[0]
    if not var.empty:
        v = var[["construct", "component", "icc"]].dropna().copy()
        v = v.groupby("construct")["icc"].median().reset_index()
        ax.bar(v["construct"], v["icc"], color=[PALETTE["capacity"] if c == "capacity" else PALETTE["state"] for c in v["construct"]], alpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Median ICC")
    ax.set_title("Trait-like stability")
    panel_label(ax, "a")

    ax = axes[1]
    if not rel.empty:
        r = rel.copy()
        r["trial_count_bin"] = pd.Categorical(r["trial_count_bin"], ["lt_50", "50_99", "100_199", "200_plus"], ordered=True)
        data = [r.loc[r["trial_count_bin"] == b, "cosine_similarity"].dropna().to_numpy() for b in r["trial_count_bin"].cat.categories]
        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="#E8F3F1", color=PALETTE["state"]), medianprops=dict(color=PALETTE["ink"]), whiskerprops=dict(color=PALETTE["state"]), capprops=dict(color=PALETTE["state"]))
        ax.set_xticks(range(1, 5), ["<50", "50-99", "100-199", "200+"])
        ax.set_xlabel("Trials")
        ax.set_ylabel("Split-half cosine")
    ax.set_title("State reliability improves with data")
    panel_label(ax, "b")

    ax = axes[2]
    if not early.empty:
        e = early.copy()
        metric_col = "delta_rmse_vs_baseline" if "delta_rmse_vs_baseline" in e.columns else None
        if metric_col:
            top = e.sort_values(metric_col).head(10)
            ax.barh(range(len(top)), top[metric_col], color=PALETTE["state"])
            ax.set_yticks(range(len(top)), [str(x)[:28] for x in top.get("model_name", top.index).astype(str)])
            ax.invert_yaxis()
            ax.set_xlabel("Delta RMSE")
        else:
            ax.text(0.5, 0.5, "Early-late model comparison\nsource data available", ha="center", va="center")
    ax.set_title("Early-to-late state prediction")
    panel_label(ax, "c")
    fig.suptitle("State is a reliability profile, not a stable scalar trait", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig5_state_reliability_atlas", source)


def figure_6() -> list[Path]:
    phys = table("physiology_robustness_models.csv")
    claims = read_csv(ROOT / "outputs" / "nhb_revision" / "audit" / "nhb_final_claim_audit.tsv")
    source = pd.concat([phys.assign(panel="physiology"), claims.assign(panel="claims")], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.55, 0.7, 0.95]})
    fig.subplots_adjust(wspace=0.7)
    ax = axes[0]
    if not phys.empty:
        p = phys.copy()
        p["axis"] = np.where(p["predictor"].astype(str).str.contains("capacity"), "capacity", "state")
        top_features = p.groupby(["axis", "modality", "feature"])["estimate"].apply(lambda s: np.nanmax(np.abs(s.astype(float)))).reset_index().sort_values("estimate", ascending=False).head(10)
        heat = p[p["feature"].isin(top_features["feature"])].pivot_table(index="feature", columns="axis", values="estimate", aggfunc="mean")
        heat = heat.reindex(top_features["feature"].drop_duplicates())
        im = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_yticks(range(len(heat.index)), [str(i).replace("_", " ")[:34] for i in heat.index])
        ax.set_xticks(range(len(heat.columns)), heat.columns)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label("rho", fontsize=6)
        cb.ax.tick_params(labelsize=6)
    ax.set_title("Physiology alignment screen")
    panel_label(ax, "a")

    ax = axes[1]
    if not phys.empty and "permutation_q_value" in phys:
        sig = phys[pd.to_numeric(phys["permutation_q_value"], errors="coerce") < 0.05]
        counts = sig["modality"].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE["exploratory"])
        ax.set_ylabel("Permutation-FDR rows")
        ax.tick_params(axis="x", rotation=35)
    ax.set_title("Exploratory physiology", pad=12)
    ax.text(-0.20, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=PALETTE["ink"])

    ax = axes[2]
    if not claims.empty:
        order = ["strong", "moderate", "qualified", "exploratory", "negative"]
        colors = {"strong": PALETTE["capacity"], "moderate": PALETTE["blue"], "qualified": PALETTE["interaction"], "exploratory": PALETTE["exploratory"], "negative": PALETTE["negative"]}
        counts = claims["claim_strength"].value_counts().reindex(order).dropna()
        ax.pie(counts.values, labels=counts.index, colors=[colors[x] for x in counts.index], startangle=90, wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(fontsize=6.5))
        ax.text(0, 0, "bounded\nclaims", ha="center", va="center", fontsize=8)
    ax.set_title("Allowed language")
    panel_label(ax, "c")
    fig.suptitle("Physiology is supportive but bounded by the claim audit", x=0.02, y=1.01, ha="left", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return finalize(fig, "fancy_fig6_physiology_claim_audit", source)


def main() -> None:
    setup()
    started = pd.Timestamp.utcnow().isoformat()
    outputs: list[Path] = []
    for fn in [figure_1, figure_2, figure_3, figure_4, figure_5, figure_6]:
        outputs.extend(fn())
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, outputs, notes=f"Generated {len(outputs)} fancy figure/source-data artifacts.")
    print(f"Generated {len(outputs)} fancy figure/source-data artifacts in {FIG_DIR}")


if __name__ == "__main__":
    main()
