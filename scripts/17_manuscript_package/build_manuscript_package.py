from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"
MANUSCRIPT = ROOT / "outputs" / "manuscript_text"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"


def ensure_dirs() -> None:
    for path in [AUDIT, LOGS, MANUSCRIPT]:
        path.mkdir(parents=True, exist_ok=True)


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def fmt_p(value: float | int | str | None) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def get_one(df: pd.DataFrame, query: str) -> pd.Series:
    out = df.query(query)
    if out.empty:
        return pd.Series(dtype=object)
    return out.iloc[0]


def title_abstract(stats: dict[str, object]) -> str:
    return f"""# Title and Abstract

## Title

State and capacity in compact recurrent models of human task behavior: a multi-dataset validation and falsification study

## Abstract

Behavioral impairment can arise because a system is temporarily in a poor operating state, because it has limited representational capacity, or because both mechanisms interact. These alternatives are often confounded by overall accuracy. Here we reconstructed a state-capacity analysis pipeline around compact gated recurrent neural networks trained on directly downloadable human behavioral datasets with EEG, fNIRS and ECG extensions. Artificial-agent perturbations established that state-like and capacity-like manipulations can be classified under matched performance, but residualized recovery of a single state-severity axis failed. We therefore treated state and capacity as multidimensional profiles rather than as clean scalar traits.

Across COG-BCI, ds007554, TU Berlin and HBN Release 4, the strongest evidence supported capacity as a geometry- and pressure-related profile. Capacity aligned with recurrent trajectory geometry, predicted cross-task consistency in COG-BCI, moderated load-related decline in TU Berlin, and scaled to HBN Release 4. State showed meaningful but less stable evidence: it predicted within-person behavioral reliability and response-time variability in COG-BCI, showed exploratory neurophysiological alignment in ds007554, and appeared in some behavioral projections, but failed the artificial-agent residualized state gate and did not validate strongly in TU Berlin or HBN. The direct ds007554 state-capacity interaction was not robust, whereas pooled interaction and TU Berlin load-by-capacity pressure effects were stronger. These results support a cautious reframing: compact recurrent models can separate useful state-like reliability profiles from capacity-like geometry profiles, but current evidence justifies strong claims for capacity pressure more than for state as an independently recovered coordinate.
"""


def introduction() -> str:
    return """# Introduction

Momentary task failure is ambiguous. A person may respond poorly because the same cognitive system is operating in an unfavorable state, because the system lacks sufficient representational capacity for the task, or because task demand exposes a capacity limit only under certain states. Accuracy alone cannot distinguish these alternatives. This ambiguity matters for cognitive neuroscience, computational psychiatry and applied monitoring, where a lapse, fatigue-like state, developmental limitation or durable capacity constraint can produce similar behavioral impairment.

Recurrent neural networks provide a practical way to formalize this distinction. A compact recurrent model has an internal state trajectory, recurrent dynamics and a representational bottleneck. Changing operating parameters such as lapse, noise, gain or memory decay can mimic state perturbation without changing structural resources. Changing hidden size, rank or bottleneck resources can mimic capacity perturbation. If these manipulations leave distinct behavioral and dynamical fingerprints after matching performance, they provide an operational basis for projecting human behavior into state-like and capacity-like profiles.

The present study implemented this logic as a full reproducible pipeline. We first tested whether artificial perturbations could separate state and capacity under matched performance. We then trained compact GRU models on unified human event tables from COG-BCI, ds007554, TU Berlin EEG-NIRS and HBN Release 4. We derived multidimensional session-level state profiles and participant-level capacity profiles, attached recurrent geometry, extracted EEG/fNIRS/ECG features where available, and ran external validation, robustness and falsification analyses.

The guiding premise was conservative. State and capacity were not assumed to be single latent variables. Instead, each was represented as a multidimensional profile, and claims were graded according to convergence across artificial-agent recovery, human behavior, recurrent dynamics, physiology, external datasets and negative controls.
"""


def results(stats: dict[str, object]) -> str:
    return f"""# Results

## Dataset Scale and Unified Event Table

The reconstructed event table contained four directly downloadable datasets. COG-BCI contributed 55,854 included events from 29 participants, ds007554 contributed 31,053 events from 30 participants with 10,896 reconstructed supervised events, TU Berlin contributed 22,096 included events from 26 participants, and HBN Release 4 contributed 180,824 included events from 324 participants. Across the full training set, Step 08 produced 1,283 session-task state rows and 390 participant capacity rows.

## Artificial-Agent Gate

Artificial agents showed that state-like and capacity-like perturbations can be distinguished under matched performance. The residualized state-versus-capacity classifier remained above shuffled controls across task families, with overall balanced accuracy of {stats['ann_overall_ba']:.3f}. However, the stricter recovery criterion failed for state. Residualized state-severity recovery was only rho = {stats['ann_state_rho']:.3f}, and the hybrid state axis failed permutation testing. This failure drove the main claim boundary: human state findings are interpreted as exploratory unless independently validated.

## Human Projection and ds007554 Discovery

Human projection analyses showed weak but nonzero state-related convergence. The state parameter profile correlated with machine-projected state in the full sample, but residualized convergence remained small. In ds007554, reconstructed push-button correctness made supervised discovery eligible. In the ds007554 primary reconstructed scope, the additive state-capacity model improved over the task/dataset baseline, but the behavioral descriptive baseline was much better. The machine-projection model had LOPO RMSE = {stats['ds_machine_rmse']:.4f}, the additive state-capacity model had LOPO RMSE = {stats['ds_additive_rmse']:.4f}, and the behavioral descriptive baseline had LOPO RMSE = {stats['ds_behavioral_rmse']:.4f}. The direct ds007554 interaction did not robustly improve over the additive model.

## Recurrent Dynamics

Capacity-like profiles aligned strongly with fitted recurrent geometry. Across 1,283 trajectories and 115,979 hidden-state rows, capacity was associated with trajectory covariance rank and related geometry measures. The strongest reported capacity-geometry association was trajectory covariance rank, rho = {stats['dyn_cap_rank_rho']:.3f}, p = {fmt_p(stats['dyn_cap_rank_p'])}. These results support capacity as a recurrent-geometry profile, while still requiring the caveat that this is fitted-model geometry rather than direct neural dynamics.

## Neurophysiology

Step 12 extracted ds007554 EEG, fNIRS and ECG features and attached reconstructed state/capacity coordinates. Direct state/capacity physiology models were eligible after push-button reconstruction. State showed exploratory associations with EEG spectral entropy and several EEG band-power features, while capacity showed stronger ECG/fNIRS associations. These findings are useful external alignment evidence, but are qualified because ds007554 behavioral correctness was reconstructed from timing signals rather than distributed as explicit trial-correctness labels.

## COG-BCI External Validation

COG-BCI provided the strongest state validation. State varied more within person than between person and predicted behavioral reliability/instability. State predicted RT coefficient of variation with q = {fmt_p(stats['cog_state_rt_q'])}, RT IQR with q = {fmt_p(stats['cog_state_iqr_q'])}, accuracy with q = {fmt_p(stats['cog_state_acc_q'])}, and lapse rate with q = {fmt_p(stats['cog_state_lapse_q'])}. Capacity showed qualified support through cross-task consistency and EEG associations. This pattern supports state as a transient behavioral reliability profile, not as a fully recovered scalar state axis.

## TU Berlin Load and Capacity Pressure

TU Berlin validated capacity pressure. Increasing N-back load reduced accuracy and slowed RT. More importantly, capacity moderated load-related decline: the load-by-capacity interaction predicted accuracy with beta = {stats['tu_pressure_beta']:.3f}, q = {fmt_p(stats['tu_pressure_q'])}, and RT with q = {fmt_p(stats['tu_pressure_rt_q'])}. State effects on lapse and RT variability were not significant after correction, so TU Berlin supports capacity pressure more strongly than state.

## HBN Scalability

HBN Release 4 tested scalability across hundreds of participants. The pipeline scaled to 324 participants, 180,824 included events, 746 state rows, 306 capacity rows and 746 recurrent-dynamics rows. Capacity-like profiles aligned with trajectory radius (rho = {stats['hbn_radius_rho']:.3f}, q = {fmt_p(stats['hbn_radius_q'])}) and trajectory covariance rank (rho = {stats['hbn_rank_rho']:.3f}, q = {fmt_p(stats['hbn_rank_q'])}). HBN did not provide strong state validation: participant-level state versus accuracy, state variability versus accuracy, age versus state and p-factor versus state were not significant after correction.

## Robustness and Falsification

Step 16 produced 160 robustness rows and 18 falsification rows. Capacity evidence was robust mainly when tied to recurrent geometry and load/capacity-pressure validation. State remained limited by the failed ANN residualized state gate, nonsignificant TU Berlin state effects and weak HBN participant-level state tests. Random-axis and shuffled-coordinate controls did not reproduce the main coordinate gains, but the ds007554 behavioral descriptive baseline outperformed coordinate models. The master effects table contains 454 effect rows, including 279 state rows, 114 capacity rows and 32 interaction rows, and intentionally includes negative and failed-validation findings.
"""


def methods() -> str:
    return """# Methods

## Study Design

The study was implemented as a scripted reconstruction under `state_capacity_tinyrnn`. Each analysis step wrote tables, audit files, figures and source data. No empirical result was accepted into the manuscript unless it was produced by a script in the reconstructed folder and represented in the master claim audit.

## Datasets

The main datasets were OpenNeuro ds007554, COG-BCI, TU Berlin simultaneous EEG-NIRS and HBN Release 4. Datasets requiring manual login or uncontrolled access were excluded from confirmatory claims. A unified event schema represented dataset, subject, session, task, block, trial index, timestamp, condition, load, target, response, correctness, RT and inclusion flags.

## Artificial Agents

Tiny recurrent agents were perturbed along state-like dimensions, including lapse, noise, gain, memory decay and temporal parameters, and capacity-like dimensions, including hidden size and recurrent resources. State and capacity perturbations were performance-matched before classification and recovery tests. Hybrid agents crossed state and capacity perturbations.

## Human Recurrent Models

Human behavior was modeled with compact GRUs using hidden sizes 1, 2, 3, 4, 6 and 8. Splits included participant-level, session-blocked and odd/even mini-block validation. State profiles were session-task level and estimated from calibration windows while held-out events were not used for state estimation. Capacity profiles were participant level and summarized hidden-size selection, selection confidence, complexity preference, high-capacity advantage, load robustness and cross-task consistency.

## State and Capacity Profiles

State was treated as a multidimensional operating profile, not a single scalar. Axes included lapse-like error rate, drift, variability and reliability. Capacity was treated as a multidimensional resource profile, including selected hidden size, selection confidence, complexity preference, high-capacity advantage, load robustness and consistency. Composite summaries were used for modeling only alongside axis-level audits.

## Dynamics

Hidden trajectories from fitted GRUs were summarized by participation ratio, covariance rank, trajectory radius, step norm, hidden variability, fixed-point/Jacobian summaries and latent decoder performance. These quantities were interpreted as fitted-model dynamics, not direct neural measurements.

## Neurophysiology and External Validation

ds007554 EEG, fNIRS and ECG features were extracted and modeled against task load and repaired state/capacity coordinates. COG-BCI EEG features were extracted from EEGLAB marker trials. TU Berlin EEG and NIRS block windows were extracted from MATLAB archives and aligned to N-back load. HBN was used as scalability and developmental-exploratory evidence.

## Statistical Testing

Analyses included leave-one-participant-out prediction, Spearman associations, OLS models with task/session/subject controls where appropriate, permutation tests, bootstrap confidence intervals and Benjamini-Hochberg correction. Claim strength was graded as strong, moderate, exploratory, negative or failed validation based on convergence, correction, controls and preregistered gate outcomes.
"""


def discussion() -> str:
    return """# Discussion

This study supports a cautious distinction between state-like reliability profiles and capacity-like recurrent-geometry profiles. The strongest evidence concerns capacity: capacity profiles aligned with recurrent geometry, generalized across HBN at scale, predicted COG-BCI cross-task consistency and moderated TU Berlin load-related decline. This makes capacity the more mature construct in the current analysis.

State was meaningful but weaker. COG-BCI showed clear state associations with RT variability, lapse and accuracy, and ds007554 showed exploratory state-neurophysiology alignment. However, the artificial-agent state recovery gate failed after performance residualization, TU Berlin did not validate state effects after correction, and HBN participant-level state tests were weak. The correct interpretation is therefore not that state is absent, but that current state estimates are best described as exploratory behavioral reliability profiles rather than a validated scalar coordinate.

The interaction evidence is similarly mixed. The direct ds007554 state-capacity interaction did not robustly improve over the additive model, whereas pooled interaction tests and TU Berlin load-by-capacity pressure were stronger. Thus, the safest interaction claim is about capacity pressure under increasing task demand, not a general state-by-capacity law.

The study also shows the value of falsification. Random-axis and shuffled-coordinate controls did not explain the main coordinate effects, but simple descriptive behavioral summaries outperformed coordinate models in ds007554. This prevents overclaiming: compact recurrent coordinates are useful explanatory profiles, not automatically superior predictive models.

Overall, the work is best framed as a computational validation and boundary-setting study. It provides stronger support for capacity as a recurrent-geometry and load-pressure construct than for state as an independently recovered coordinate. The state construct remains scientifically useful, but its strongest current form is a transient reliability/instability profile that requires further validation.
"""


def limitations() -> str:
    return """# Limitations

- The ANN gate failed for residualized state recovery, so state claims must remain exploratory.
- ds007554 correctness was reconstructed from push-button timing, not provided as explicit trial-level correctness labels.
- The ds007554 behavioral descriptive baseline outperformed the coordinate models.
- Capacity is participant-level in the current implementation, limiting formal session-level capacity test-retest claims.
- HBN is useful for scalability but not direct adult N-back/PVT evidence; most HBN events are passive-video or EEG markers, RT is absent in the unified table and correctness is task-limited.
- Recurrent geometry is fitted-model geometry, not direct neural geometry.
- Vanilla RNN and LSTM architecture variants were not trained in the current full run, so architecture-variant robustness cannot be claimed.
- Neurophysiology analyses use derived features rather than full encoding models or source-localized neural measures.
"""


def figure_captions() -> str:
    return """# Figure Captions

**Figure 1. Study pipeline and claim coverage.** Dataset scale and master claim-audit coverage across state, capacity, interaction, load and control constructs.

**Figure 2. Artificial-agent intervention gate.** Matched state-versus-capacity classification succeeds under residualized fingerprints, but residualized state-severity and hybrid-state recovery fail the stricter gate.

**Figure 3. ds007554 reconstructed discovery analysis.** Leave-one-participant-out model comparison and bootstrap state/capacity associations using push-button reconstructed correctness.

**Figure 4. Recurrent dynamics.** Associations between state/capacity profiles and fitted GRU trajectory geometry, plus latent decoder performance.

**Figure 5. ds007554 neurophysiology.** Direct state/capacity associations with EEG, ECG, fNIRS and physiology-derived features after coordinate repair.

**Figure 6. COG-BCI validation.** State and capacity associations with COG-BCI behavior and EEG features, including strong state associations with RT variability and lapse.

**Figure 7. TU Berlin and HBN validation.** TU Berlin load/capacity-pressure effects and HBN scalability/geometry validation.

**Figure 8. Robustness and falsification.** Hidden-size sensitivity, residualized tests, leave-one-dataset/task checks, random/shuffled controls, permutation controls and baseline challenges.
"""


def availability() -> tuple[str, str, str]:
    reproducibility = """# Reproducibility Checklist

- All included datasets are directly downloadable or locally inventoried with provenance.
- Unified event tables are stored under `data/processed`.
- Every analysis step writes status logs under `outputs/logs`.
- Every figure has PNG, PDF, SVG, source CSV, data dictionary and script-used file.
- `outputs/audit/claim_audit.tsv` maps manuscript claims to scripts and outputs.
- `outputs/tables/master_effects_table.csv` is the single source of truth for reported effects.
- Known unresolved limitation: the current workspace has no `.git` directory and Git is not installed on PATH, documented in `outputs/audit/git_environment_status.md`.
"""
    data = """# Data Availability Statement

The analyses use directly downloadable public datasets: OpenNeuro ds007554, COG-BCI, TU Berlin simultaneous EEG-NIRS and HBN Release 4 / OpenNeuro ds005508. Processed event tables, derived feature tables, source data for figures and audit files are written under `state_capacity_tinyrnn/data/processed` and `state_capacity_tinyrnn/outputs`.
"""
    code = """# Code Availability Statement

All analysis scripts used for the reconstructed study are in `state_capacity_tinyrnn/scripts`. The principal reproducibility entry points are the step scripts for preprocessing, GRU training, projection, recurrent dynamics, neurophysiology, COG-BCI validation, TU Berlin validation, HBN scalability, robustness, master statistics and manuscript packaging.
"""
    return reproducibility, data, code


def overclaiming_report() -> str:
    return """# Overclaiming Risk Report

## Highest-risk claims

1. **State as a validated scalar coordinate.** Not supported. State should be described as an exploratory multidimensional reliability/instability profile.
2. **Direct ds007554 supervised behavioral discovery.** Must state that correctness was reconstructed from push-button timing.
3. **Coordinate models as best predictors.** Not true for ds007554, where the descriptive behavioral baseline is best.
4. **Neural mechanism claims from GRU geometry.** Recurrent geometry is fitted-model geometry, not direct neural geometry.
5. **General state-capacity interaction.** Direct ds007554 state-capacity interaction is weak; stronger evidence is for TU Berlin load-by-capacity pressure.

## Safer framing

- Strongest: capacity as a recurrent-geometry and capacity-pressure profile.
- Moderate/exploratory: state as transient reliability/RT-instability profile.
- Exploratory: state-neurophysiology alignment.
- Negative/limited: direct scalar state recovery and direct ds007554 state-capacity interaction.
"""


def final_decision(stats: dict[str, object]) -> str:
    return f"""# Final Decision Report

1. **Did the artificial-agent gate pass?** Partly. State-versus-capacity classification passed, but residualized state recovery failed. Overall gate consequence: state claims downgraded.
2. **Did simulation/hybrid recovery pass?** Capacity hybrid recovery passed; state hybrid recovery failed after residualization.
3. **Did ds007554 reproduce the discovery result?** Partly. Reconstructed ds007554 supports state/capacity and machine projection signals, but descriptive behavioral baseline is best and direct interaction is weak.
4. **Did capacity associate with recurrent geometry?** Yes. Strongly, especially trajectory covariance rank and radius.
5. **Did state associate with reliability/lapse/RT variability?** Yes in COG-BCI; weak or negative in TU Berlin and HBN.
6. **Did ds007554 neurophysiology align with state/capacity?** Partly. Direct coordinate models are eligible and show exploratory state and stronger capacity/physiology associations, qualified by reconstructed labels.
7. **Did COG-BCI validate the coordinate structure?** Yes for state reliability/RT variability and qualified capacity consistency/EEG effects.
8. **Did TU Berlin validate load/capacity pressure?** Yes. Load-by-capacity pressure effects on accuracy and RT were strong.
9. **Did HBN demonstrate scalability?** Yes. HBN scaled to 324 participants and supported capacity-geometry alignment.
10. **Did behavioral baselines outperform TinyRNN coordinates?** In ds007554, yes: the descriptive behavioral baseline was best.
11. **Which claims are strong, moderate, exploratory, negative or failed?** See `outputs/tables/master_effects_table.csv` and `outputs/audit/claim_audit.tsv`. Summary: capacity has the strongest evidence; state is mostly exploratory; direct ds007554 interaction is negative/weak; TU load-by-capacity pressure is strong.
12. **Journal framing.** The current result is better framed as a computational neuroscience/model-validation paper than as a definitive NHB-style behavioral-neural discovery. A Nature Machine Intelligence-style framing may be more natural if positioned around falsifiable recurrent profiles and public-dataset validation. For Nature Human Behaviour, the state claims would need stronger prospective validation.

## Recommended Manuscript Claim

Compact recurrent models provide reproducible multidimensional profiles that distinguish capacity-like geometry and load-pressure effects from state-like reliability fluctuations. Capacity is currently the stronger validated construct; state is meaningful but exploratory.
"""


def gather_stats() -> dict[str, object]:
    ann = read_table("ann_intervention_gate_results.csv")
    ds = read_table("ds007554_discovery_model_comparison.csv")
    dyn = read_table("recurrent_dynamics_state_capacity_tests.csv")
    cog = read_table("cog_bci_validation_models.csv")
    tu = read_table("tu_berlin_load_validation.csv")
    hbn = read_table("hbn_scalability_tests.csv")
    ann_overall = ann.query("analysis_type == 'matched_state_capacity_classification' and feature_set == 'residualized_fingerprint' and task_family == 'overall'").iloc[0]
    ann_state = ann.query("analysis_type == 'state_severity_recovery' and feature_set == 'residualized_fingerprint'").iloc[0]
    ds_primary = ds.query("analysis_scope == 'ds007554_primary_reconstructed'")
    get_model = lambda name: ds_primary.query("model_name == @name").iloc[0]
    dyn_rank = dyn.query("outcome == 'trajectory_cov_rank' and predictor == 'capacity_parameter_resource_z'").iloc[0]
    cog_state_rt = cog.query("x == 'state' and y == 'rt_cv_behavior'").iloc[0]
    cog_state_iqr = cog.query("x == 'state' and y == 'rt_iqr_behavior'").iloc[0]
    cog_state_acc = cog.query("x == 'state' and y == 'mean_accuracy'").iloc[0]
    cog_state_lapse = cog.query("x == 'state' and y == 'lapse_rate'").iloc[0]
    tu_pressure = tu.query("model_name == 'capacity_pressure_accuracy'").iloc[0]
    tu_pressure_rt = tu.query("model_name == 'capacity_pressure_rt'").iloc[0]
    hbn_rank = hbn.query("analysis == 'participant_capacity_vs_cov_rank'").iloc[0]
    hbn_radius = hbn.query("analysis == 'participant_capacity_vs_radius'").iloc[0]
    return {
        "ann_overall_ba": float(ann_overall["observed"]),
        "ann_state_rho": float(ann_state["observed"]),
        "ds_machine_rmse": float(get_model("machine_projection_additive")["lopo_rmse"]),
        "ds_additive_rmse": float(get_model("additive_state_capacity")["lopo_rmse"]),
        "ds_behavioral_rmse": float(get_model("behavioral_descriptive")["lopo_rmse"]),
        "dyn_cap_rank_rho": float(dyn_rank["spearman_rho"]),
        "dyn_cap_rank_p": float(dyn_rank["p_value"]),
        "cog_state_rt_q": float(cog_state_rt["q_value"]),
        "cog_state_iqr_q": float(cog_state_iqr["q_value"]),
        "cog_state_acc_q": float(cog_state_acc["q_value"]),
        "cog_state_lapse_q": float(cog_state_lapse["q_value"]),
        "tu_pressure_beta": float(tu_pressure["estimate"]),
        "tu_pressure_q": float(tu_pressure["q_value"]),
        "tu_pressure_rt_q": float(tu_pressure_rt["q_value"]),
        "hbn_rank_rho": float(hbn_rank["spearman_rho"]),
        "hbn_rank_q": float(hbn_rank["q_value"]),
        "hbn_radius_rho": float(hbn_radius["spearman_rho"]),
        "hbn_radius_q": float(hbn_radius["q_value"]),
    }


def main() -> int:
    ensure_dirs()
    stats = gather_stats()
    files = {
        "title_abstract.md": title_abstract(stats),
        "introduction.md": introduction(),
        "results.md": results(stats),
        "methods.md": methods(),
        "discussion.md": discussion(),
        "limitations.md": limitations(),
        "figure_captions.md": figure_captions(),
    }
    repro, data, code = availability()
    audit_files = {
        "reproducibility_checklist.md": repro,
        "data_availability_statement.md": data,
        "code_availability_statement.md": code,
        "overclaiming_risk_report.md": overclaiming_report(),
        "final_decision_report.md": final_decision(stats),
    }
    for name, text in files.items():
        (MANUSCRIPT / name).write_text(text, encoding="utf-8")
    for name, text in audit_files.items():
        (AUDIT / name).write_text(text, encoding="utf-8")

    full = "\n\n".join(files[name] for name in ["title_abstract.md", "introduction.md", "results.md", "methods.md", "discussion.md", "limitations.md", "figure_captions.md"])
    (MANUSCRIPT / "full_manuscript_draft.md").write_text(full, encoding="utf-8")

    status = {
        "status": "implemented_and_run",
        "manuscript_files": sorted(files.keys()) + ["full_manuscript_draft.md"],
        "audit_files": sorted(audit_files.keys()),
        "recommended_claim": "Capacity is strong as recurrent-geometry/load-pressure evidence; state is meaningful but exploratory.",
    }
    (LOGS / "step19_manuscript_package_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP19_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
