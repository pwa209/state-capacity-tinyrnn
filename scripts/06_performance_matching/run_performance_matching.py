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

TASK_FAMILIES = {
    "overall": "mean_accuracy",
    "nback": "accuracy_nback",
    "go_nogo": "accuracy_go_nogo",
    "context_xor": "accuracy_context_xor",
}
NLL_COLUMN = "negative_log_likelihood"
MIN_PAIRS_PER_TASK = 20
CALIPERS = [0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.25, 1.75, 2.50]


def load_agent_table() -> pd.DataFrame:
    required = [
        TABLES / "artificial_agent_registry.csv",
        TABLES / "artificial_behavioral_fingerprints.csv",
        TABLES / "artificial_dynamics_fingerprints.csv",
        TABLES / "artificial_perturbation_parameters.csv",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Step 06 requires Step 05 outputs first: " + ", ".join(missing))

    registry = pd.read_csv(TABLES / "artificial_agent_registry.csv")
    behavior = pd.read_csv(TABLES / "artificial_behavioral_fingerprints.csv")
    dynamics = pd.read_csv(TABLES / "artificial_dynamics_fingerprints.csv")
    perturbations = pd.read_csv(TABLES / "artificial_perturbation_parameters.csv")

    merged = registry.merge(behavior, on="agent_id", validate="one_to_one")
    merged = merged.merge(dynamics, on="agent_id", validate="one_to_one", suffixes=("", "_dyn"))
    merged = merged.merge(
        perturbations[["agent_id", "capacity_axis", "capacity_level", "state_severity"]],
        on="agent_id",
        validate="one_to_one",
        suffixes=("", "_pert"),
    )
    for column in ["capacity_axis", "capacity_level", "state_severity"]:
        pert_column = f"{column}_pert"
        if pert_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[pert_column])
            merged = merged.drop(columns=[pert_column])
    return merged


def zscore_frame(values: pd.DataFrame) -> pd.DataFrame:
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0).replace(0, 1.0)
    return (values - means) / stds


def pairwise_distances(state_values: pd.DataFrame, capacity_values: pd.DataFrame) -> np.ndarray:
    state_array = state_values.to_numpy(dtype=float)
    capacity_array = capacity_values.to_numpy(dtype=float)
    delta = state_array[:, None, :] - capacity_array[None, :, :]
    return np.sqrt(np.mean(delta**2, axis=2))


def greedy_match(
    state_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    task_family: str,
    accuracy_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    perf = pd.concat(
        [
            state_df[["agent_id", accuracy_column, NLL_COLUMN]],
            capacity_df[["agent_id", accuracy_column, NLL_COLUMN]],
        ],
        ignore_index=True,
    ).dropna()
    scaled = zscore_frame(perf[[accuracy_column, NLL_COLUMN]])
    perf_scaled = perf[["agent_id"]].join(scaled)

    state_scaled = state_df[["agent_id"]].merge(perf_scaled, on="agent_id", how="inner")
    capacity_scaled = capacity_df[["agent_id"]].merge(perf_scaled, on="agent_id", how="inner")
    state_usable = state_df.merge(state_scaled[["agent_id"]], on="agent_id", how="inner")
    capacity_usable = capacity_df.merge(capacity_scaled[["agent_id"]], on="agent_id", how="inner")

    distances = pairwise_distances(
        state_scaled[[accuracy_column, NLL_COLUMN]],
        capacity_scaled[[accuracy_column, NLL_COLUMN]],
    )
    candidate_pairs: list[dict[str, Any]] = []
    for i, state_row in state_usable.reset_index(drop=True).iterrows():
        for j, capacity_row in capacity_usable.reset_index(drop=True).iterrows():
            candidate_pairs.append(
                {
                    "state_i": i,
                    "capacity_j": j,
                    "state_agent_id": state_row["agent_id"],
                    "capacity_agent_id": capacity_row["agent_id"],
                    "distance": float(distances[i, j]),
                }
            )
    candidate_pairs = sorted(candidate_pairs, key=lambda row: row["distance"])

    selected: list[dict[str, Any]] = []
    selected_state: set[int] = set()
    selected_capacity: set[int] = set()
    chosen_caliper = CALIPERS[-1]
    for caliper in CALIPERS:
        selected = []
        selected_state = set()
        selected_capacity = set()
        for pair in candidate_pairs:
            if pair["distance"] > caliper:
                break
            if pair["state_i"] in selected_state or pair["capacity_j"] in selected_capacity:
                continue
            selected.append(pair)
            selected_state.add(pair["state_i"])
            selected_capacity.add(pair["capacity_j"])
        chosen_caliper = caliper
        if len(selected) >= MIN_PAIRS_PER_TASK:
            break

    pair_rows = []
    for match_index, pair in enumerate(selected, start=1):
        s = state_usable.iloc[pair["state_i"]]
        c = capacity_usable.iloc[pair["capacity_j"]]
        pair_rows.append(
            {
                "match_id": f"{task_family}_{match_index:03d}",
                "task_family": task_family,
                "performance_accuracy_column": accuracy_column,
                "state_agent_id": s["agent_id"],
                "capacity_agent_id": c["agent_id"],
                "state_seed": int(s["seed"]),
                "capacity_seed": int(c["seed"]),
                "state_accuracy": float(s[accuracy_column]),
                "capacity_accuracy": float(c[accuracy_column]),
                "state_nll": float(s[NLL_COLUMN]),
                "capacity_nll": float(c[NLL_COLUMN]),
                "accuracy_abs_delta": float(abs(s[accuracy_column] - c[accuracy_column])),
                "nll_abs_delta": float(abs(s[NLL_COLUMN] - c[NLL_COLUMN])),
                "z_distance": pair["distance"],
                "caliper_used": chosen_caliper,
                "state_severity": float(s["state_severity"]),
                "capacity_level": float(c["capacity_level"]),
                "capacity_axis": c["capacity_axis"],
                "state_checkpoint_path": s["checkpoint_path"],
                "capacity_checkpoint_path": c["checkpoint_path"],
            }
        )

    diagnostics = {
        "task_family": task_family,
        "performance_accuracy_column": accuracy_column,
        "n_state_candidates": int(len(state_usable)),
        "n_capacity_candidates": int(len(capacity_usable)),
        "n_pairs": int(len(pair_rows)),
        "min_required_pairs": MIN_PAIRS_PER_TASK,
        "pass_minimum_pair_count": bool(len(pair_rows) >= MIN_PAIRS_PER_TASK),
        "caliper_used": chosen_caliper,
        "median_z_distance": float(np.median([row["z_distance"] for row in pair_rows])) if pair_rows else np.nan,
        "max_z_distance": float(np.max([row["z_distance"] for row in pair_rows])) if pair_rows else np.nan,
        "median_accuracy_abs_delta": float(np.median([row["accuracy_abs_delta"] for row in pair_rows])) if pair_rows else np.nan,
        "median_nll_abs_delta": float(np.median([row["nll_abs_delta"] for row in pair_rows])) if pair_rows else np.nan,
    }
    return pd.DataFrame(pair_rows), diagnostics


def build_source_data(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in pairs.to_dict("records"):
        rows.append(
            {
                "match_id": record["match_id"],
                "task_family": record["task_family"],
                "agent_role": "state",
                "agent_id": record["state_agent_id"],
                "accuracy": record["state_accuracy"],
                "negative_log_likelihood": record["state_nll"],
                "z_distance": record["z_distance"],
                "accuracy_abs_delta": record["accuracy_abs_delta"],
                "nll_abs_delta": record["nll_abs_delta"],
                "capacity_axis": "",
            }
        )
        rows.append(
            {
                "match_id": record["match_id"],
                "task_family": record["task_family"],
                "agent_role": "capacity",
                "agent_id": record["capacity_agent_id"],
                "accuracy": record["capacity_accuracy"],
                "negative_log_likelihood": record["capacity_nll"],
                "z_distance": record["z_distance"],
                "accuracy_abs_delta": record["accuracy_abs_delta"],
                "nll_abs_delta": record["nll_abs_delta"],
                "capacity_axis": record["capacity_axis"],
            }
        )
    return pd.DataFrame(rows)


def make_figure(source: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
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
    colors = {"state": "#0072B2", "capacity": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), constrained_layout=True)
    axes = axes.ravel()

    for axis, task_family in zip(axes, TASK_FAMILIES):
        task_data = source[source["task_family"] == task_family]
        for _, group in task_data.groupby("match_id"):
            if len(group) == 2:
                axis.plot(
                    group["accuracy"],
                    group["negative_log_likelihood"],
                    color="#B0B0B0",
                    linewidth=0.35,
                    alpha=0.5,
                    zorder=1,
                )
        for role in ["state", "capacity"]:
            role_data = task_data[task_data["agent_role"] == role]
            axis.scatter(
                role_data["accuracy"],
                role_data["negative_log_likelihood"],
                s=12,
                color=colors[role],
                alpha=0.85,
                edgecolors="none",
                label=role if task_family == "overall" else None,
                zorder=2,
            )
        diag = diagnostics[diagnostics["task_family"] == task_family].iloc[0]
        axis.set_title(f"{task_family.replace('_', ' ')}: n={int(diag['n_pairs'])}", fontsize=8)
        axis.set_xlabel("Matched accuracy")
        axis.set_ylabel("Negative log-likelihood")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_agent_performance_matching.{suffix}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_data_dictionary() -> None:
    text = """# Figure Agent Performance Matching Source Data

Each row is one agent endpoint in a matched state-capacity pair.

- `match_id`: unique matched-pair identifier within task family.
- `task_family`: matching performance view: overall, nback, go_nogo or context_xor.
- `agent_role`: whether the row is the state-perturbed or capacity-limited member.
- `agent_id`: artificial agent identifier from the registry.
- `accuracy`: accuracy used for matching in that task family.
- `negative_log_likelihood`: global held-out NLL used for matching.
- `z_distance`: Euclidean distance between matched agents after z-scoring accuracy and NLL.
- `accuracy_abs_delta`: absolute raw accuracy difference within the pair.
- `nll_abs_delta`: absolute raw NLL difference within the pair.
- `capacity_axis`: capacity manipulation axis for capacity rows.
"""
    (SOURCE_DATA / "figure_agent_performance_matching_data_dictionary.md").write_text(text, encoding="utf-8")
    (SOURCE_DATA / "figure_agent_performance_matching_script_used.txt").write_text(
        "scripts/06_performance_matching/run_performance_matching.py\n",
        encoding="utf-8",
    )


def simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df.to_dict("records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join(rows)


def write_decision(diagnostics: pd.DataFrame) -> None:
    passed = bool(diagnostics["pass_minimum_pair_count"].all())
    lines = [
        "# Step 06 Performance Matching Decision",
        "",
        f"Decision: {'PASS' if passed else 'FAIL'}",
        "",
        f"Minimum required matched pairs per task family: {MIN_PAIRS_PER_TASK}",
        "",
        simple_markdown_table(diagnostics),
        "",
        "Interpretation:",
        "- PASS means Step 07 can test state-vs-capacity separability under matched performance.",
        "- A failed task family must be treated as unmatched and cannot support matched-performance claims.",
    ]
    (AUDIT / "performance_matching_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    assert_full_run_allowed()
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    agents = load_agent_table()
    state = agents[agents["family"] == "state_perturbation"].copy()
    capacity = agents[agents["family"] == "capacity_perturbation"].copy()
    if state.empty or capacity.empty:
        raise RuntimeError("Performance matching requires both state_perturbation and capacity_perturbation agents.")

    pair_tables = []
    diagnostics = []
    for task_family, accuracy_column in TASK_FAMILIES.items():
        pairs, diag = greedy_match(state, capacity, task_family, accuracy_column)
        pair_tables.append(pairs)
        diagnostics.append(diag)

    matched_pairs = pd.concat(pair_tables, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    source = build_source_data(matched_pairs)

    matched_pairs.to_csv(TABLES / "performance_matched_agent_pairs.csv", index=False)
    diagnostics_df.to_csv(TABLES / "performance_matching_diagnostics.csv", index=False)
    source.to_csv(SOURCE_DATA / "figure_agent_performance_matching_source.csv", index=False)
    write_data_dictionary()
    make_figure(source, diagnostics_df)
    write_decision(diagnostics_df)

    summary = {
        "n_pairs_total": int(len(matched_pairs)),
        "pairs_by_task": matched_pairs.groupby("task_family").size().to_dict(),
        "passed": bool(diagnostics_df["pass_minimum_pair_count"].all()),
    }
    print("PERFORMANCE_MATCHING complete " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
