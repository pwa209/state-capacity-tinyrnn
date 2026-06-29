from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB, TABLES, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_29_projection_figure_statistics"
SCRIPT_NAME = "scripts/nhb_revision/29_projection_figure_statistics.py"
OUT_DIR = NHB / "display_item_revision"
PROJECTION_CSV = TABLES / "human_state_capacity_multiaxis_projection.csv"
ELIGIBILITY_CSV = TABLES / "human_projection_dataset_eligibility.csv"
FIGURE_PLAN_CSV = OUT_DIR / "revised_figure_plan.csv"
CAPTION_CSV = OUT_DIR / "revised_figure_captions.csv"
OUTPUT_XLSX = OUT_DIR / "projection_figure_results_statistics_source_data.xlsx"

X_COL = "optimized_state_profile_z"
Y_COL = "optimized_capacity_profile_z"
CHI2_68 = 2.279  # chi-square(df=2) probability mass 0.68
CHI2_95 = 5.991  # chi-square(df=2) probability mass 0.95


PLOT_COLUMNS = [
    "dataset",
    "subject",
    "session",
    "task",
    "task_family",
    "participant_id",
    "state_estimation_quality",
    "capacity_claim_status",
    "n_calibration_events",
    "n_heldout_events",
    "n_events",
    "mean_accuracy",
    "calibration_accuracy",
    "rt_median",
    "accuracy_ceiling_flag",
    "dynamics_available",
    X_COL,
    Y_COL,
    "machine_state_projection_raw_z",
    "machine_capacity_projection_raw_z",
    "state_multidimensional_summary_z",
    "capacity_multidimensional_summary_z",
    "state_lapse_axis_z",
    "state_drift_axis_z",
    "state_variability_axis_z",
    "state_reliability_axis_z",
    "capacity_hidden_size_axis_z",
    "capacity_selection_confidence_z",
    "capacity_complexity_preference_axis_z",
    "capacity_high_capacity_nll_advantage_z",
    "capacity_load_robustness_axis_z",
    "capacity_cross_task_consistency_axis_z",
    "dynamics_capacity_geometry_z",
]


STATE_SUBAXES = [
    "state_lapse_axis_z",
    "state_drift_axis_z",
    "state_variability_axis_z",
    "state_reliability_axis_z",
    "state_multidimensional_summary_z",
    "machine_state_projection_raw_z",
    X_COL,
]

CAPACITY_SUBAXES = [
    "capacity_hidden_size_axis_z",
    "capacity_selection_confidence_z",
    "capacity_complexity_preference_axis_z",
    "capacity_high_capacity_nll_advantage_z",
    "capacity_load_robustness_axis_z",
    "capacity_cross_task_consistency_axis_z",
    "capacity_multidimensional_summary_z",
    "machine_capacity_projection_raw_z",
    "dynamics_capacity_geometry_z",
    Y_COL,
]

CATEGORICAL_COLUMNS = {
    "dataset",
    "subject",
    "session",
    "task",
    "task_family",
    "participant_id",
    "state_estimation_quality",
    "capacity_claim_status",
    "accuracy_ceiling_flag",
    "dynamics_available",
}


def task_family(task: object) -> str:
    text = str(task).lower()
    if "nback" in text or "n-back" in text:
        return "nback"
    if "flanker" in text:
        return "flanker"
    if "pvt" in text or "vigilance" in text:
        return "vigilance"
    if "go" in text or "nogo" in text:
        return "go_nogo"
    if "stroop" in text:
        return "stroop"
    if "rest" in text:
        return "rest"
    if "memory" in text:
        return "memory"
    if "motor" in text:
        return "motor"
    if "arithmetic" in text:
        return "arithmetic"
    return str(task)


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ci_summary(group: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [tuple([(), group])] if not group_cols else group.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_cols, key))
        for value_col in value_cols:
            values = pd.to_numeric(sub[value_col], errors="coerce").dropna()
            n = int(values.shape[0])
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            se = float(sd / math.sqrt(n)) if n > 1 else np.nan
            rows.append(
                {
                    **prefix,
                    "coordinate": value_col,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "ci95_low": mean - 1.96 * se if n > 1 else np.nan,
                    "ci95_high": mean + 1.96 * se if n > 1 else np.nan,
                    "median": float(values.median()) if n else np.nan,
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if n else np.nan,
                    "min": float(values.min()) if n else np.nan,
                    "max": float(values.max()) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def group_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_projection_rows": len(sub),
                "n_subjects": sub["subject"].nunique(dropna=True) if "subject" in sub else np.nan,
                "n_participants": sub["participant_id"].nunique(dropna=True)
                if "participant_id" in sub
                else np.nan,
                "n_sessions": sub["session"].nunique(dropna=True) if "session" in sub else np.nan,
                "n_tasks": sub["task"].nunique(dropna=True) if "task" in sub else np.nan,
                "n_high_state_quality": int((sub.get("state_estimation_quality") == "high").sum())
                if "state_estimation_quality" in sub
                else np.nan,
                "n_low_or_flagged_state_quality": int((sub.get("state_estimation_quality") != "high").sum())
                if "state_estimation_quality" in sub
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ellipse_parameters(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        xy = sub[[X_COL, Y_COL]].apply(pd.to_numeric, errors="coerce").dropna()
        row = dict(zip(group_cols, key))
        row["n"] = len(xy)
        if len(xy) < 3:
            rows.append({**row, "ellipse_status": "too_few_points"})
            continue
        cov = np.cov(xy[X_COL].to_numpy(), xy[Y_COL].to_numpy())
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        vals = np.maximum(vals, 0)
        angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
        rows.append(
            {
                **row,
                "ellipse_status": "ok",
                "center_x_state": float(xy[X_COL].mean()),
                "center_y_capacity": float(xy[Y_COL].mean()),
                "cov_xx": float(cov[0, 0]),
                "cov_xy": float(cov[0, 1]),
                "cov_yy": float(cov[1, 1]),
                "sd_x": float(xy[X_COL].std(ddof=1)),
                "sd_y": float(xy[Y_COL].std(ddof=1)),
                "pearson_r": float(xy[X_COL].corr(xy[Y_COL], method="pearson")),
                "eigenvalue_major": float(vals[0]),
                "eigenvalue_minor": float(vals[1]),
                "angle_degrees": angle,
                "radius_major_68": float(math.sqrt(CHI2_68 * vals[0])),
                "radius_minor_68": float(math.sqrt(CHI2_68 * vals[1])),
                "radius_major_95": float(math.sqrt(CHI2_95 * vals[0])),
                "radius_minor_95": float(math.sqrt(CHI2_95 * vals[1])),
            }
        )
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, scopes: list[list[str]]) -> pd.DataFrame:
    rows = []
    for scope in scopes:
        grouped = [(("all",), df)] if not scope else df.groupby(scope, dropna=False)
        for key, sub in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            xy = sub[[X_COL, Y_COL]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(xy)
            spearman = (
                xy[X_COL].rank(method="average").corr(xy[Y_COL].rank(method="average"), method="pearson")
                if n > 1
                else np.nan
            )
            rows.append(
                {
                    "scope": "overall" if not scope else "+".join(scope),
                    **dict(zip(scope or ["group"], key)),
                    "n": n,
                    "pearson_r": float(xy[X_COL].corr(xy[Y_COL], method="pearson"))
                    if n > 1
                    else np.nan,
                    "spearman_rho": float(spearman) if n > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def marginal_histograms(df: pd.DataFrame, coordinate: str, group_col: str = "dataset", n_bins: int = 30) -> pd.DataFrame:
    values = pd.to_numeric(df[coordinate], errors="coerce").dropna()
    lower = math.floor(float(values.min()) * 2) / 2
    upper = math.ceil(float(values.max()) * 2) / 2
    bins = np.linspace(lower, upper, n_bins + 1)
    rows = []
    scopes = [("all", df)]
    scopes.extend(list(df.groupby(group_col, dropna=False)))
    for group, sub in scopes:
        v = pd.to_numeric(sub[coordinate], errors="coerce").dropna().to_numpy()
        counts, edges = np.histogram(v, bins=bins)
        total = counts.sum()
        widths = np.diff(edges)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "coordinate": coordinate,
                    "group_col": "all" if group == "all" else group_col,
                    "group": group,
                    "bin_index": i + 1,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "bin_mid": float((edges[i] + edges[i + 1]) / 2),
                    "count": int(count),
                    "proportion": float(count / total) if total else 0.0,
                    "density": float(count / total / widths[i]) if total and widths[i] else 0.0,
                }
            )
    return pd.DataFrame(rows)


def density_2d(df: pd.DataFrame, group_col: str | None = None, n_bins: int = 30) -> pd.DataFrame:
    x = pd.to_numeric(df[X_COL], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    valid = df.loc[x.notna() & y.notna()].copy()
    lower_x = math.floor(float(valid[X_COL].min()) * 2) / 2
    upper_x = math.ceil(float(valid[X_COL].max()) * 2) / 2
    lower_y = math.floor(float(valid[Y_COL].min()) * 2) / 2
    upper_y = math.ceil(float(valid[Y_COL].max()) * 2) / 2
    x_bins = np.linspace(lower_x, upper_x, n_bins + 1)
    y_bins = np.linspace(lower_y, upper_y, n_bins + 1)
    rows = []
    groups = [("all", valid)] if group_col is None else list(valid.groupby(group_col, dropna=False))
    for group, sub in groups:
        counts, x_edges, y_edges = np.histogram2d(sub[X_COL], sub[Y_COL], bins=[x_bins, y_bins])
        total = counts.sum()
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                count = int(counts[i, j])
                if count == 0:
                    continue
                rows.append(
                    {
                        "group_col": "all" if group_col is None else group_col,
                        "group": group,
                        "x_bin": i + 1,
                        "y_bin": j + 1,
                        "x_left": float(x_edges[i]),
                        "x_right": float(x_edges[i + 1]),
                        "x_mid": float((x_edges[i] + x_edges[i + 1]) / 2),
                        "y_left": float(y_edges[j]),
                        "y_right": float(y_edges[j + 1]),
                        "y_mid": float((y_edges[j] + y_edges[j + 1]) / 2),
                        "count": count,
                        "proportion": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def source_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": X_COL,
                "role": "x axis",
                "description": "Optimized multidimensional state-profile score, z-scaled. Use as projection x-axis.",
            },
            {
                "field": Y_COL,
                "role": "y axis",
                "description": "Optimized multidimensional capacity-profile score, z-scaled. Use as projection y-axis.",
            },
            {
                "field": "dataset",
                "role": "colour/group",
                "description": "Open dataset source; recommended colour mapping.",
            },
            {
                "field": "task_family",
                "role": "shape/facet",
                "description": "Derived task family; useful for symbols or facets.",
            },
            {
                "field": "state_estimation_quality",
                "role": "alpha/filter",
                "description": "Use to identify ceiling/low-variance state estimates; do not silently remove unless specified.",
            },
            {
                "field": "accuracy_ceiling_flag",
                "role": "quality flag",
                "description": "True where accuracy is at/near ceiling and state estimation is less informative.",
            },
        ]
    )


def write_workbook(sheets: dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for column_cells in ws.columns:
                header = str(column_cells[0].value or "")
                width = max(10, min(36, len(header) + 4))
                ws.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    ensure_nhb_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    projection = pd.read_csv(PROJECTION_CSV, low_memory=False)
    projection["task_family"] = projection["task"].map(task_family)
    numeric_cols = [c for c in set(PLOT_COLUMNS + STATE_SUBAXES + CAPACITY_SUBAXES) if c not in CATEGORICAL_COLUMNS]
    projection = clean_numeric(projection, numeric_cols)
    plot_points = projection[[c for c in PLOT_COLUMNS if c in projection.columns]].copy()
    plot_points = plot_points.dropna(subset=[X_COL, Y_COL])

    eligibility = pd.read_csv(ELIGIBILITY_CSV) if ELIGIBILITY_CSV.exists() else pd.DataFrame()
    fig_plan = pd.read_csv(FIGURE_PLAN_CSV) if FIGURE_PLAN_CSV.exists() else pd.DataFrame()
    captions = pd.read_csv(CAPTION_CSV) if CAPTION_CSV.exists() else pd.DataFrame()
    fig2_plan = fig_plan.loc[fig_plan["display_id"].eq("Figure 2")].copy() if not fig_plan.empty else pd.DataFrame()
    fig2_caption = captions.loc[captions["figure"].eq("Figure 2")].copy() if not captions.empty else pd.DataFrame()

    readme = pd.DataFrame(
        [
            {
                "item": "purpose",
                "value": "Projection-figure statistics and source data for drawing Figure 2 in R or Python.",
            },
            {
                "item": "x_axis",
                "value": X_COL,
            },
            {
                "item": "y_axis",
                "value": Y_COL,
            },
            {
                "item": "n_projection_rows",
                "value": len(plot_points),
            },
            {
                "item": "n_datasets",
                "value": plot_points["dataset"].nunique(),
            },
            {
                "item": "recommended_main_panel",
                "value": "Scatter/projection map coloured by dataset, shaped or faceted by task family, with 68% or 95% dataset ellipses and marginal histograms.",
            },
            {
                "item": "claim_boundary",
                "value": "Plot profile scores, not validated scalar neural coordinates. Label as optimized state profile and optimized capacity profile.",
            },
            {
                "item": "generated_at",
                "value": datetime.now().isoformat(timespec="seconds"),
            },
            {
                "item": "source_csv",
                "value": str(PROJECTION_CSV),
            },
        ]
    )

    dataset_summary = group_counts(plot_points, ["dataset"]).merge(
        ci_summary(plot_points, ["dataset"], [X_COL, Y_COL]),
        on="dataset",
        how="right",
    )
    task_summary = group_counts(plot_points, ["task_family"]).merge(
        ci_summary(plot_points, ["task_family"], [X_COL, Y_COL]),
        on="task_family",
        how="right",
    )
    dataset_task_summary = group_counts(plot_points, ["dataset", "task_family"]).merge(
        ci_summary(plot_points, ["dataset", "task_family"], [X_COL, Y_COL]),
        on=["dataset", "task_family"],
        how="right",
    )
    quality_summary = group_counts(plot_points, ["dataset", "state_estimation_quality"]).merge(
        ci_summary(plot_points, ["dataset", "state_estimation_quality"], [X_COL, Y_COL]),
        on=["dataset", "state_estimation_quality"],
        how="right",
    )

    subaxis_rows = []
    for axis_family, cols in [("state", STATE_SUBAXES), ("capacity", CAPACITY_SUBAXES)]:
        present = [c for c in cols if c in plot_points.columns]
        for scope, group_cols in [("overall", []), ("dataset", ["dataset"]), ("task_family", ["task_family"])]:
            summary = ci_summary(plot_points, group_cols, present)
            summary.insert(0, "axis_family", axis_family)
            summary.insert(1, "scope", scope)
            subaxis_rows.append(summary)
    subaxis_summary = pd.concat(subaxis_rows, ignore_index=True)

    sheets = {
        "README": readme,
        "axis_dictionary": source_dictionary(),
        "caption_and_panel": pd.concat([fig2_plan, fig2_caption], ignore_index=True, sort=False),
        "projection_points": plot_points,
        "dataset_counts": eligibility,
        "dataset_summary": dataset_summary,
        "task_summary": task_summary,
        "dataset_task_summary": dataset_task_summary,
        "quality_summary": quality_summary,
        "coordinate_correlations": correlations(plot_points, [[], ["dataset"], ["task_family"], ["dataset", "task_family"]]),
        "ellipse_dataset_95": ellipse_parameters(plot_points, ["dataset"]),
        "ellipse_task_95": ellipse_parameters(plot_points, ["task_family"]),
        "ellipse_dataset_task": ellipse_parameters(plot_points, ["dataset", "task_family"]),
        "marginal_hist_state": marginal_histograms(plot_points, X_COL),
        "marginal_hist_capacity": marginal_histograms(plot_points, Y_COL),
        "density2d_all": density_2d(plot_points, None),
        "density2d_dataset": density_2d(plot_points, "dataset"),
        "subaxis_summary": subaxis_summary,
    }

    write_workbook(sheets, OUTPUT_XLSX)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, [OUTPUT_XLSX])
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        [OUTPUT_XLSX],
        status="complete",
        notes=f"Projection figure statistics from {PROJECTION_CSV.name}",
    )
    print(f"Wrote {OUTPUT_XLSX}")
    print(f"Projection rows: {len(plot_points)}")


if __name__ == "__main__":
    main()
