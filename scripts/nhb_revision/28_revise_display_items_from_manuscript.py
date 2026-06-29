from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_28_manuscript_display_item_revision"
SCRIPT_NAME = "scripts/nhb_revision/28_revise_display_items_from_manuscript.py"
MANUSCRIPT = Path(r"C:\Users\Gebruiker\Downloads\state_capacity_manuscript_NHB (1).docx")
OUT_DIR = NHB / "display_item_revision"
TABLE_DIR = NHB / "tables"
FIG_DIR = NHB / "figures"
FANCY_DIR = NHB / "fancy_figures"
FANCY_SOURCE_DIR = FANCY_DIR / "source_data"
ROOT_TABLES = ROOT / "outputs" / "tables"
AUDIT_DIR = NHB / "audit"

PAYLOAD_JSON = OUT_DIR / "display_item_workbook_payload.json"
REPORT_MD = OUT_DIR / "revised_display_item_report.md"
EXTRACT_MD = OUT_DIR / "manuscript_methods_results_extracted.md"
CAPTIONS_CSV = OUT_DIR / "revised_figure_captions.csv"
TABLE_PLAN_CSV = OUT_DIR / "revised_table_plan.csv"
FIGURE_PLAN_CSV = OUT_DIR / "revised_figure_plan.csv"


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def read_doc_paragraphs(path: Path) -> list[dict[str, str | int]]:
    doc = Document(path)
    rows: list[dict[str, str | int]] = []
    for i, paragraph in enumerate(doc.paragraphs):
        text = clean_text(paragraph.text)
        if not text:
            continue
        style = paragraph.style.name if paragraph.style is not None else "NoStyle"
        rows.append({"index": i, "style": style, "text": text})
    return rows


def is_heading(row: dict[str, str | int]) -> bool:
    style = str(row["style"]).lower()
    text = str(row["text"]).strip()
    return style.startswith("heading") or text.lower() in {
        "abstract",
        "introduction",
        "results",
        "discussion",
        "methods",
        "references",
        "figure legends",
    }


def extract_section(rows: list[dict[str, str | int]], start_name: str, stop_names: set[str]) -> list[str]:
    start = None
    for i, row in enumerate(rows):
        if str(row["text"]).strip().lower() == start_name.lower():
            start = i + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for row in rows[start:]:
        text = str(row["text"]).strip()
        lower = text.lower()
        if lower in stop_names and is_heading(row):
            break
        out.append(text)
    return out


def extract_captions(rows: list[dict[str, str | int]], label: str) -> list[dict[str, str]]:
    pattern = re.compile(rf"^{label}\s*(\d+)\s*\|\s*(.*)$", re.I)
    out: list[dict[str, str]] = []
    for row in rows:
        text = str(row["text"]).strip()
        match = pattern.match(text)
        if match:
            out.append(
                {
                    "item": f"{label.title()} {match.group(1)}",
                    "caption_in_manuscript": match.group(2).strip(),
                    "paragraph_index": str(row["index"]),
                }
            )
    return out


def path_text(path: Path) -> str:
    return str(path)


def source(path: Path, role: str) -> dict[str, str]:
    return {"path": path_text(path), "role": role}


FIGURE_PLANS: list[dict[str, Any]] = [
    {
        "display_id": "Figure 1",
        "status": "keep_main",
        "nhb_role": "Main Figure 1",
        "nmi_role": "Main Figure 1",
        "title": "Machine-defined state-capacity framework and intervention logic",
        "needs_projection": "no",
        "fancy_option": "Use fancy_fig1_graphical_abstract as a graphical abstract or top panel; use fig1_concept_pipeline for the quantitative pipeline.",
        "recommendation": "Keep, but make it a compact conceptual+pipeline figure. Do not make numerical claims here except sample scale and gate names.",
        "panels": "a conceptual state/capacity intervention schematic; b recurrent architecture/task battery; c analysis flow from artificial agents to human projection; d claim-strength legend.",
        "sources": [
            source(TABLE_DIR / "fig1_concept_pipeline_source_data.csv", "primary source-data table"),
            source(FANCY_SOURCE_DIR / "fancy_fig1_graphical_abstract_source_data.csv", "fancy graphical abstract source data"),
        ],
    },
    {
        "display_id": "Figure 2",
        "status": "keep_main_revise_projection",
        "nhb_role": "Main Figure 2",
        "nmi_role": "Main Figure 2 or Extended Data depending on space",
        "title": "Datasets, harmonisation and human projection onto the machine-defined profiles",
        "needs_projection": "yes",
        "fancy_option": "Use fancy_fig2_state_capacity_landscape as the projection landscape; this is the best candidate for the requested projection figure.",
        "recommendation": "Keep and revise. The current caption says projection exists, but the figure should visibly show projected human sessions/tasks in state-capacity/profile space, coloured by dataset and shaped by task.",
        "panels": "a dataset scale/inclusion flow; b event/task harmonisation; c 2D projection of human observations; d marginal distributions for optimized state and capacity profiles.",
        "sources": [
            source(TABLE_DIR / "fig2_dataset_pipeline_scale_source_data.csv", "dataset scale and projection row counts"),
            source(ROOT_TABLES / "human_state_capacity_multiaxis_projection.csv", "session/task projection coordinates"),
            source(FANCY_SOURCE_DIR / "fancy_fig2_state_capacity_landscape_source_data.csv", "state-capacity landscape source data"),
            source(ROOT_TABLES / "human_projection_dataset_eligibility.csv", "dataset eligibility for projection"),
        ],
    },
    {
        "display_id": "Figure 3",
        "status": "keep_main",
        "nhb_role": "Main Figure 3",
        "nmi_role": "Main Figure 3",
        "title": "Artificial-agent gates and scalar-state falsification",
        "needs_projection": "no",
        "fancy_option": "Use fancy_fig3_architecture_robustness as an architecture robustness inset.",
        "recommendation": "Keep. This is the central machine-validation figure. It should show both the positive family gate and the failed residualised scalar-state recovery.",
        "panels": "a perturbation-family balanced accuracy by architecture/task; b permutation/null threshold; c leave-one-architecture generalisation; d hybrid raw versus residualised scalar recovery.",
        "sources": [
            source(TABLE_DIR / "fig3_artificial_agent_gates_source_data.csv", "main Figure 3 source data"),
            source(TABLE_DIR / "architecture_perturbation_gate_results.csv", "architecture gate results"),
            source(TABLE_DIR / "architecture_hybrid_recovery_results.csv", "hybrid scalar recovery"),
            source(TABLE_DIR / "leave_one_architecture_gate_results.csv", "leave-one-architecture control"),
            source(FANCY_SOURCE_DIR / "fancy_fig3_architecture_robustness_source_data.csv", "fancy robustness source data"),
        ],
    },
    {
        "display_id": "Figure 4",
        "status": "keep_main",
        "nhb_role": "Main Figure 4",
        "nmi_role": "Main Figure 4",
        "title": "Incremental predictive value and negative controls",
        "needs_projection": "uses_projected_profiles",
        "fancy_option": "Keep visually restrained; a Nature-style slope/delta plot is stronger than decoration here.",
        "recommendation": "Keep, but caption must acknowledge that behavioural descriptive baselines remain strong. The point is explanatory/incremental value, not supreme prediction.",
        "panels": "a out-of-sample RMSE by baseline/model; b delta-RMSE against task/dataset baseline; c random/shuffled-axis controls; d state-capacity interaction delta.",
        "sources": [
            source(TABLE_DIR / "fig4_prediction_baselines_source_data.csv", "main Figure 4 source data"),
            source(TABLE_DIR / "incremental_value_model_comparison.csv", "incremental model comparison"),
            source(TABLE_DIR / "incremental_value_delta_metrics.csv", "delta metrics"),
            source(TABLE_DIR / "general_state_capacity_interaction.csv", "state-capacity interaction tests"),
            source(ROOT_TABLES / "ds007554_discovery_model_comparison.csv", "discovery model comparison"),
        ],
    },
    {
        "display_id": "Figure 5",
        "status": "keep_main",
        "nhb_role": "Main Figure 5",
        "nmi_role": "Main Figure 5",
        "title": "Capacity as load-pressure and recurrent-geometry profile",
        "needs_projection": "uses_capacity_profile",
        "fancy_option": "Use fancy_fig4_capacity_pressure_surface as a visually strong but still quantitative surface panel.",
        "recommendation": "Keep. This is one of the strongest result figures because capacity has convergent geometry and TU Berlin load-pressure support.",
        "panels": "a recurrent participation-ratio/capacity association; b TU Berlin load-by-capacity accuracy/RT pressure; c variant/ablation validation; d pressure surface.",
        "sources": [
            source(TABLE_DIR / "fig5_capacity_geometry_pressure_source_data.csv", "main Figure 5 source data"),
            source(TABLE_DIR / "capacity_pressure_models.csv", "capacity pressure model coefficients"),
            source(TABLE_DIR / "capacity_pressure_marginal_effects.csv", "capacity pressure marginal effects"),
            source(ROOT_TABLES / "recurrent_dynamics_state_capacity_tests.csv", "recurrent dynamics tests"),
            source(TABLE_DIR / "capacity_variant_validation_summary.csv", "capacity variant validation"),
            source(FANCY_SOURCE_DIR / "fancy_fig4_capacity_pressure_surface_source_data.csv", "fancy pressure surface source data"),
        ],
    },
    {
        "display_id": "Figure 6",
        "status": "keep_main",
        "nhb_role": "Main Figure 6",
        "nmi_role": "Main Figure 6 or Extended Data",
        "title": "State as within-person reliability and operating-profile signal",
        "needs_projection": "uses_state_profile",
        "fancy_option": "Use fancy_fig5_state_reliability_atlas as a summary atlas or Extended Data opener.",
        "recommendation": "Keep, because it prevents overclaiming. The figure should explicitly say state is useful as a reliability/operating profile, not a stable trait coordinate.",
        "panels": "a variance decomposition/ICC; b split-half reliability versus trial count; c bootstrap reliability; d early-late prediction curves.",
        "sources": [
            source(TABLE_DIR / "fig6_state_reliability_source_data.csv", "main Figure 6 source data"),
            source(TABLE_DIR / "state_capacity_variance_decomposition.csv", "variance decomposition"),
            source(TABLE_DIR / "state_split_half_reliability.csv", "split-half reliability"),
            source(TABLE_DIR / "state_bootstrap_reliability.csv", "bootstrap reliability"),
            source(FANCY_SOURCE_DIR / "fancy_fig5_state_reliability_atlas_source_data.csv", "fancy reliability atlas source data"),
        ],
    },
    {
        "display_id": "Figure 7",
        "status": "conditional_main",
        "nhb_role": "Main Figure 7 if physiology is central; otherwise Extended Data Figure",
        "nmi_role": "Extended Data Figure unless the editor wants human neuroscience validation",
        "title": "Bounded physiological alignment of behavioural profiles",
        "needs_projection": "uses_projected_profiles",
        "fancy_option": "Use fancy_fig6_physiology_claim_audit as a strong claim-audit visual, not as proof of neural mechanism.",
        "recommendation": "Use carefully. It is attractive for NHB, but the caption and labels must say exploratory physiological alignment rather than neural coordinate evidence.",
        "panels": "a EEG/ECG/fNIRS partial-correlation tiles; b blocked-permutation/FDR survival; c modality-specific examples; d claim-boundary audit.",
        "sources": [
            source(TABLE_DIR / "fig7_physiology_alignment_source_data.csv", "main Figure 7 source data"),
            source(TABLE_DIR / "physiology_robustness_models.csv", "physiology robustness models"),
            source(TABLE_DIR / "physiology_permutation_controls.csv", "blocked-permutation controls"),
            source(ROOT_TABLES / "ds007554_neurophys_models.csv", "ds007554 neurophysiology models"),
            source(ROOT_TABLES / "cog_bci_validation_models.csv", "COG-BCI validation models"),
            source(FANCY_SOURCE_DIR / "fancy_fig6_physiology_claim_audit_source_data.csv", "fancy physiology claim-audit source data"),
        ],
    },
    {
        "display_id": "Figure 8",
        "status": "move_to_extended_or_last_main",
        "nhb_role": "Extended Data Figure unless journal permits eight main figures",
        "nmi_role": "Extended Data/Supplementary Figure",
        "title": "Generalisation and claim-audited falsification map",
        "needs_projection": "no",
        "fancy_option": "Use as a graphical supplement or transparent robustness map.",
        "recommendation": "Scientifically important, but probably too meta for the main narrative. Keep the data and use the audit as Extended Data/Supplementary Table unless the editor asks for transparent claim grading in the main text.",
        "panels": "a leave-one-dataset-out results; b leave-one-task-out results; c profile shuffle controls; d claim-strength map.",
        "sources": [
            source(TABLE_DIR / "fig8_claim_audit_falsification_source_data.csv", "main Figure 8 source data"),
            source(TABLE_DIR / "leave_one_dataset_out_validation.csv", "leave-one-dataset validation"),
            source(TABLE_DIR / "leave_one_task_out_validation.csv", "leave-one-task validation"),
            source(TABLE_DIR / "profile_shuffle_controls.csv", "profile-shuffle controls"),
            source(AUDIT_DIR / "nhb_final_claim_audit.tsv", "final claim audit"),
        ],
    },
]


TABLE_RECOMMENDATIONS = [
    {
        "table": "Table 1",
        "recommendation": "keep_main",
        "reason": "Dataset inventory is needed for venue reviewers and anchors sample size/roles.",
        "preferred_location": "Main text Table 1",
    },
    {
        "table": "Table 2",
        "recommendation": "move_to_extended_data",
        "reason": "Architecture-family gate is better as Figure 3; detailed rows can be Extended Data.",
        "preferred_location": "Extended Data Table 1",
    },
    {
        "table": "Table 3",
        "recommendation": "merge_into_figure_3",
        "reason": "Hybrid scalar recovery is conceptually part of the artificial-agent falsification gate.",
        "preferred_location": "Figure 3 panel/table inset plus source-data workbook",
    },
    {
        "table": "Table 4",
        "recommendation": "move_to_extended_data",
        "reason": "Model coefficients should support Figure 5 but are too detailed for main text.",
        "preferred_location": "Extended Data Table 2",
    },
    {
        "table": "Table 5",
        "recommendation": "move_to_extended_data",
        "reason": "Variance components belong behind the state reliability figure.",
        "preferred_location": "Extended Data Table 3",
    },
    {
        "table": "Table 6",
        "recommendation": "move_to_extended_data",
        "reason": "Prediction deltas are better visualised in Figure 4; retain numeric table for reproducibility.",
        "preferred_location": "Extended Data Table 4",
    },
    {
        "table": "Table 7",
        "recommendation": "move_to_extended_data_or_supplement",
        "reason": "Physiology is exploratory; top alignments should not dominate the main text.",
        "preferred_location": "Extended Data Table 5",
    },
    {
        "table": "Table 8",
        "recommendation": "move_to_supplement",
        "reason": "The claim audit is essential for transparency but too meta as a main-text table.",
        "preferred_location": "Supplementary Table / Extended Data Table 6",
    },
]


REVISED_CAPTIONS = [
    {
        "figure": "Figure 1",
        "caption": "Machine-defined state-capacity framework. a, State-like interventions alter recurrent operating conditions, whereas capacity-like interventions alter representational resources. b, Vanilla RNN, GRU and LSTM agents were trained on the artificial task battery. c, Artificial-agent fingerprints define analysis axes that are then projected into human datasets. d, Claim-strength labels constrain the permitted interpretation of each downstream result.",
    },
    {
        "figure": "Figure 2",
        "caption": "Human projection across open datasets. a, Included events, participants, sessions and tasks across COG-BCI, ds007554, TU Berlin EEG-NIRS and HBN Release 4. b, Harmonisation of behavioural events into calibration and held-out records. c, Projection of human session/task observations into optimized state-profile and capacity-profile space; points should be coloured by dataset and shaped by task family. d, Marginal distributions and eligibility flags showing where state and capacity profiles are interpretable.",
    },
    {
        "figure": "Figure 3",
        "caption": "Artificial-agent gates separate intervention families but falsify a simple residualised scalar-state claim. a, Balanced accuracy for classifying state-like versus capacity-like intervention families across recurrent architectures and tasks. b, Permutation-null thresholds and leave-one-architecture generalisation. c, Hybrid scalar recovery from raw fingerprints. d, Collapse of residualised scalar-state recovery after architecture and capacity are controlled, motivating profile-level rather than scalar-state language.",
    },
    {
        "figure": "Figure 4",
        "caption": "Projected profiles add explanatory value beyond task and dataset baselines, with bounded predictive claims. a, Out-of-sample prediction error for task/dataset, descriptive behavioural, additive profile and interaction models. b, Delta error relative to the required task/dataset baseline. c, Random-axis and shuffled-profile controls. d, State-capacity interaction estimates, reported as incremental rather than decisive when behavioural baselines remain strong.",
    },
    {
        "figure": "Figure 5",
        "caption": "Capacity behaves as a load-pressure and recurrent-geometry profile. a, Recurrent-trajectory dimensionality and related geometry metrics by capacity profile. b, TU Berlin load-by-capacity pressure effects for accuracy and response time. c, Capacity validation across behaviour-only, model-only, geometry-blind and feature-ablated variants. d, Optional pressure-surface panel showing how load exposes capacity differences.",
    },
    {
        "figure": "Figure 6",
        "caption": "State behaves as a within-person operating and reliability profile. a, Variance decomposition and intraclass correlations for state sub-axes and capacity. b, Split-half reliability as a function of available trials. c, Bootstrap reliability across trial-count regimes. d, Early-late prediction results showing when state profiles are informative and when data are too sparse or ceiling-limited.",
    },
    {
        "figure": "Figure 7",
        "caption": "Bounded physiological alignment of behavioural profiles. a, Partial associations between projected profiles and EEG, ECG and fNIRS features after task, session and load adjustment. b, Blocked-permutation false-discovery-rate controls. c, Representative modality-specific effects. d, Claim-boundary panel making clear that these analyses support physiological alignment, not a direct neural coordinate.",
    },
    {
        "figure": "Figure 8",
        "caption": "Generalisation and claim-audited falsification map. a, Leave-one-dataset-out estimates for core claims. b, Leave-one-task-out estimates. c, Profile-shuffle and negative-control outcomes. d, Pre-specified claim audit linking each manuscript claim to allowed language, failed gates and surviving controls. This display is recommended for Extended Data unless eight main figures are acceptable.",
    },
]


FANCY_OPTIONS = [
    {
        "asset": "fancy_fig1_graphical_abstract",
        "recommended_use": "Graphical abstract or Figure 1 panel a",
        "venue_fit": "NMI/NHB",
        "note": "Strong opening visual, but keep quantitative claims in adjacent panels.",
    },
    {
        "asset": "fancy_fig2_state_capacity_landscape",
        "recommended_use": "Projection figure / Figure 2 panel c",
        "venue_fit": "NHB strongest; NMI acceptable as projection map",
        "note": "Best answer to the projection-figure need.",
    },
    {
        "asset": "fancy_fig3_architecture_robustness",
        "recommended_use": "Figure 3 inset or Extended Data opener",
        "venue_fit": "NMI strongest",
        "note": "Good for architecture robustness and model-centric venue framing.",
    },
    {
        "asset": "fancy_fig4_capacity_pressure_surface",
        "recommended_use": "Figure 5 panel d",
        "venue_fit": "NHB/NMI",
        "note": "Visually engaging and grounded in a strong result.",
    },
    {
        "asset": "fancy_fig5_state_reliability_atlas",
        "recommended_use": "Figure 6 panel summary or Extended Data",
        "venue_fit": "NHB",
        "note": "Use to clarify the state result without overstating trait stability.",
    },
    {
        "asset": "fancy_fig6_physiology_claim_audit",
        "recommended_use": "Figure 7 panel d or Extended Data",
        "venue_fit": "NHB if physiology is central; NMI as supplement",
        "note": "Must be labelled as bounded physiological alignment, not neural mechanism.",
    },
]


def read_source_table(path: Path, role: str, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [{"source_file": path_text(path), "source_role": role, "source_status": "missing"}]
        )
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "source_file": path_text(path),
                    "source_role": role,
                    "source_status": f"read_error: {exc}",
                }
            ]
        )
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
        df["source_row_limit_note"] = f"first {max_rows} rows retained for workbook readability"
    df.insert(0, "source_role", role)
    df.insert(0, "source_file", path.name)
    df.insert(0, "source_status", "ok")
    return df


def projection_subset(path: Path) -> pd.DataFrame:
    df = read_source_table(path, "projection coordinates")
    if "source_status" in df.columns and len(df) == 1 and df.loc[0, "source_status"] != "ok":
        return df
    wanted = [
        "source_status",
        "source_file",
        "source_role",
        "dataset",
        "subject",
        "session",
        "task",
        "n_calibration_events",
        "n_heldout_events",
        "calibration_accuracy",
        "mean_accuracy",
        "rt_median",
        "state_estimation_quality",
        "state_lapse_axis_z",
        "state_drift_axis_z",
        "state_variability_axis_z",
        "state_reliability_axis_z",
        "state_multidimensional_summary_z",
        "machine_state_projection_raw_z",
        "machine_capacity_projection_raw_z",
        "capacity_hidden_size_axis_z",
        "capacity_multidimensional_summary_z",
        "dynamics_capacity_geometry_z",
        "optimized_state_profile_z",
        "optimized_capacity_profile_z",
        "dynamics_available",
    ]
    return df[[c for c in wanted if c in df.columns]].copy()


def data_for_figure(plan: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in plan["sources"]:
        path = Path(src["path"])
        role = src["role"]
        if path.name == "human_state_capacity_multiaxis_projection.csv":
            frames.append(projection_subset(path))
        else:
            frames.append(read_source_table(path, role))
    return pd.concat(frames, ignore_index=True, sort=False)


def df_to_rows(df: pd.DataFrame) -> list[list[Any]]:
    if df.empty:
        return [["note"], ["No rows available"]]
    df = df.copy()
    df = df.where(pd.notna(df), None)
    rows: list[list[Any]] = [list(map(str, df.columns))]
    for record in df.itertuples(index=False, name=None):
        rows.append([clean_cell(v) for v in record])
    return rows


def clean_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        if isinstance(value, float) and (pd.isna(value)):
            return None
        return value
    return str(value)


def rows_from_records(records: list[dict[str, Any]]) -> list[list[Any]]:
    if not records:
        return [["note"], ["No records"]]
    headers = list(records[0].keys())
    return [headers] + [[clean_cell(record.get(h)) for h in headers] for record in records]


def build_report(
    methods: list[str],
    results: list[str],
    doc_figures: list[dict[str, str]],
    doc_tables: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("# Revised Display-Item Plan for NHB/NMI")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input manuscript: `{MANUSCRIPT}`")
    lines.append("")
    lines.append("## Main Verdict")
    lines.append(
        "The Methods and Results support a strong Nature-style display package, but the current manuscript is too table-heavy for the main text. Keep a small number of decisive main figures, retain one dataset table in the main text, and move most coefficient/audit tables to Extended Data or Supplementary Information."
    )
    lines.append("")
    lines.append("## Projection Figure")
    lines.append(
        "Yes, the paper should have a projection figure. It should be revised into Figure 2: a visible projection of human session/task observations into optimized state-profile and capacity-profile space, rather than only reporting that 1,283 state rows and 390 capacity rows were projected."
    )
    lines.append("")
    lines.append("## Recommended Main Figures")
    for plan in FIGURE_PLANS:
        lines.append(
            f"- **{plan['display_id']}** ({plan['status']}): {plan['title']}. {plan['recommendation']}"
        )
    lines.append("")
    lines.append("## Fancy Figure Options")
    for opt in FANCY_OPTIONS:
        lines.append(
            f"- **{opt['asset']}**: {opt['recommended_use']} ({opt['venue_fit']}). {opt['note']}"
        )
    lines.append("")
    lines.append("## Table Decisions")
    for rec in TABLE_RECOMMENDATIONS:
        lines.append(
            f"- **{rec['table']}**: {rec['recommendation']} -> {rec['preferred_location']}. {rec['reason']}"
        )
    lines.append("")
    lines.append("## Venue-Specific Framing")
    lines.append(
        "- For **Nature Human Behaviour**, foreground Figures 2, 5, 6 and a cautious Figure 7 because the human projection, reliability and physiology story matter."
    )
    lines.append(
        "- For **Nature Machine Intelligence**, foreground Figures 1, 3, 4 and 5 because the machine-defined intervention gate and model robustness are the cleanest venue match; move physiology to Extended Data."
    )
    lines.append("")
    lines.append("## Current Manuscript Inventory")
    lines.append(f"- Results paragraphs extracted: {len(results)}")
    lines.append(f"- Methods paragraphs extracted: {len(methods)}")
    lines.append(f"- Figure legends detected: {len(doc_figures)}")
    lines.append(f"- Table captions detected in Results: {len(doc_tables)}")
    lines.append("")
    lines.append("## Caption Boundary")
    lines.append(
        "All revised captions deliberately use profile language. Avoid: 'state is a validated scalar coordinate' and 'physiology proves a neural coordinate'. Use: 'state profile', 'capacity profile', 'bounded physiological alignment' and 'exploratory external alignment'."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_nhb_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_doc_paragraphs(MANUSCRIPT)
    results = extract_section(rows, "Results", {"discussion", "methods", "references", "figure legends"})
    methods = extract_section(rows, "Methods", {"references", "figure legends"})
    doc_figures = extract_captions(rows, "Fig.")
    if not doc_figures:
        doc_figures = extract_captions(rows, "Figure")
    doc_tables = extract_captions(rows, "Table")

    figure_plan_rows: list[dict[str, Any]] = []
    for plan in FIGURE_PLANS:
        figure_plan_rows.append(
            {
                "display_id": plan["display_id"],
                "status": plan["status"],
                "nhb_role": plan["nhb_role"],
                "nmi_role": plan["nmi_role"],
                "title": plan["title"],
                "needs_projection": plan["needs_projection"],
                "panels": plan["panels"],
                "recommendation": plan["recommendation"],
                "fancy_option": plan["fancy_option"],
                "source_files": "; ".join(Path(s["path"]).name for s in plan["sources"]),
            }
        )

    table_plan_rows: list[dict[str, Any]] = []
    table_caption_map = {row["item"]: row["caption_in_manuscript"] for row in doc_tables}
    for rec in TABLE_RECOMMENDATIONS:
        row = rec.copy()
        row["caption_in_manuscript"] = table_caption_map.get(rec["table"], "")
        table_plan_rows.append(row)

    caption_rows: list[dict[str, Any]] = []
    original_caption_map = {
        row["item"].replace("Fig.", "Figure"): row["caption_in_manuscript"] for row in doc_figures
    }
    for row in REVISED_CAPTIONS:
        caption_rows.append(
            {
                "figure": row["figure"],
                "revised_caption": row["caption"],
                "caption_in_manuscript": original_caption_map.get(row["figure"], ""),
            }
        )

    manuscript_audit = [
        {
            "audit_item": "projection_figure",
            "status": "needed_and_present_but_should_be_made_visible",
            "note": "Current Figure 2 mentions projection; revised plan makes projection a visible panel using human_state_capacity_multiaxis_projection.csv.",
        },
        {
            "audit_item": "main_text_tables",
            "status": "too_many",
            "note": "Eight in-text tables is heavy for NHB/NMI. Keep Table 1 main and move most model/audit tables to Extended Data/Supplement.",
        },
        {
            "audit_item": "physiology_claim",
            "status": "bounded",
            "note": "Figure 7 can be main for NHB, but must remain exploratory alignment, not direct neural-coordinate evidence.",
        },
        {
            "audit_item": "state_claim",
            "status": "qualified",
            "note": "State should be framed as an operating/reliability profile, not as a fully validated scalar trait coordinate.",
        },
        {
            "audit_item": "capacity_claim",
            "status": "strongest_axis",
            "note": "Capacity has the strongest convergence through recurrent geometry and TU Berlin load-pressure evidence.",
        },
    ]

    sheets: list[dict[str, Any]] = [
        {
            "name": "INDEX",
            "rows": rows_from_records(
                [
                    {
                        "sheet": "FIGURE_PLAN",
                        "description": "Revised figure decisions for NHB/NMI.",
                    },
                    {
                        "sheet": "TABLE_PLAN",
                        "description": "Which tables to keep, merge, or move.",
                    },
                    {
                        "sheet": "CAPTIONS_REVISED",
                        "description": "Revised figure captions and original captions.",
                    },
                    {
                        "sheet": "FANCY_OPTIONS",
                        "description": "Fancy figure assets and recommended use.",
                    },
                    {
                        "sheet": "MANUSCRIPT_AUDIT",
                        "description": "Main claims and overclaiming checks.",
                    },
                    *[
                        {
                            "sheet": f"Figure{i}_data",
                            "description": f"Source data for revised Figure {i}.",
                        }
                        for i in range(1, 9)
                    ],
                ]
            ),
        },
        {"name": "FIGURE_PLAN", "rows": rows_from_records(figure_plan_rows)},
        {"name": "TABLE_PLAN", "rows": rows_from_records(table_plan_rows)},
        {"name": "CAPTIONS_REVISED", "rows": rows_from_records(caption_rows)},
        {"name": "FANCY_OPTIONS", "rows": rows_from_records(FANCY_OPTIONS)},
        {"name": "MANUSCRIPT_AUDIT", "rows": rows_from_records(manuscript_audit)},
    ]

    for i, plan in enumerate(FIGURE_PLANS, start=1):
        df = data_for_figure(plan)
        sheets.append({"name": f"Figure{i}_data", "rows": df_to_rows(df)})

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_manuscript": path_text(MANUSCRIPT),
        "output_workbook": path_text(OUT_DIR / "state_capacity_NHB_NMI_revised_figure_table_source_data.xlsx"),
        "sheets": sheets,
    }
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(figure_plan_rows).to_csv(FIGURE_PLAN_CSV, index=False)
    pd.DataFrame(table_plan_rows).to_csv(TABLE_PLAN_CSV, index=False)
    pd.DataFrame(caption_rows).to_csv(CAPTIONS_CSV, index=False)
    EXTRACT_MD.write_text(
        "# Extracted Methods and Results\n\n"
        "## Results\n\n"
        + "\n\n".join(results)
        + "\n\n## Methods\n\n"
        + "\n\n".join(methods)
        + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(build_report(methods, results, doc_figures, doc_tables), encoding="utf-8")

    outputs = [PAYLOAD_JSON, REPORT_MD, FIGURE_PLAN_CSV, TABLE_PLAN_CSV, CAPTIONS_CSV, EXTRACT_MD]
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        datetime.now(timezone.utc).isoformat(),
        outputs,
        status="completed_payload_ready_for_workbook",
        notes=f"Input manuscript: {MANUSCRIPT}",
    )
    print(f"Wrote {PAYLOAD_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
