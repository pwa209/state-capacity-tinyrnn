from __future__ import annotations

import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


COLUMNS = [
    "theoretical_construct",
    "definition",
    "machine_operation",
    "human_measure",
    "expected_signature",
    "dataset",
    "script",
    "output_table",
    "claim_allowed",
]


ROWS = [
    {
        "theoretical_construct": "global computational state",
        "definition": "Transient operating regime of a system with otherwise available representational resources, represented as a multi-axis state profile rather than a single scalar.",
        "machine_operation": "Hold architecture and weights fixed; vary tau, lapse, hidden noise sigma, update gain and memory decay.",
        "human_measure": "Session-level multi-axis state profile estimated from held-out behavioral windows and, in later steps, projected behavioral/physiological fingerprints.",
        "expected_signature": "Within-person/session variation across lapse, reliability, update gain, memory decay and RT variability axes, with physiological vigilance markers beyond task and mean accuracy.",
        "dataset": "ds007554, COG-BCI, TU Berlin",
        "script": "scripts/07_train_tinyrnn/train_all.py; scripts/08_estimate_coordinates/estimate_coordinates.py; scripts/11_ds007554_neurophys/extract_and_model_neurophys.py; scripts/12_external_cog_bci/run_cog_bci_validation.py",
        "output_table": "outputs/tables/session_state_parameters.csv; outputs/tables/session_state_multiaxis_coordinates.csv; outputs/tables/session_state_quality_report.csv; outputs/tables/human_state_capacity_multiaxis_projection.csv; outputs/tables/ds007554_neurophys_models.csv",
        "claim_allowed": "Current ANN gate did not recover state as a reliable residualized coordinate; human state claims remain exploratory unless multi-axis convergence and controls pass in downstream projection.",
    },
    {
        "theoretical_construct": "structural capacity",
        "definition": "Durable representational or dynamical resource profile available to a task system, represented as a multi-axis capacity profile rather than a single scalar.",
        "machine_operation": "Hold operating regime fixed; vary hidden size, recurrent rank, memory window, bottleneck width and recurrence.",
        "human_measure": "Participant-level selected hidden size, selection confidence, complexity preference, high-capacity advantage, load robustness, cross-task consistency and recurrent geometry.",
        "expected_signature": "Between-person stability across multiple resource axes, cross-task consistency, trajectory geometry and local memory timescale beyond mean accuracy.",
        "dataset": "ds007554, COG-BCI, TU Berlin, HBN",
        "script": "scripts/07_train_tinyrnn/train_all.py; scripts/08_estimate_coordinates/estimate_coordinates.py; scripts/09_dynamics/run_dynamics.py",
        "output_table": "outputs/tables/participant_capacity_coordinates.csv; outputs/tables/participant_capacity_multidimensional_coordinates.csv; outputs/tables/recurrent_dynamics_by_subject_task.csv",
        "claim_allowed": "Capacity has partial ANN support but not a clean scalar-trait interpretation; claims must emphasize multi-axis proxy evidence unless geometry, projection and robustness tests converge.",
    },
    {
        "theoretical_construct": "hybrid manipulation",
        "definition": "Concurrent change in state severity and capacity level.",
        "machine_operation": "Factorially cross artificial state perturbations with artificial capacity restrictions.",
        "human_measure": "Human projection into state-profile and capacity-profile axes.",
        "expected_signature": "Hybrid artificial agents recover separable state and capacity profiles; human sessions can vary in state without erasing participant capacity profiles.",
        "dataset": "Artificial agents; ds007554; COG-BCI",
        "script": "scripts/05_artificial_agents/build_artificial_agents.py; scripts/06_ann_intervention_gate/run_ann_gate.py",
        "output_table": "outputs/tables/ann_hybrid_recovery.csv; outputs/tables/human_state_capacity_multiaxis_projection.csv",
        "claim_allowed": "Current ANN hybrid recovery failed for state and passed only for capacity; human hybrid claims are exploratory until downstream projection converges.",
    },
    {
        "theoretical_construct": "state perturbation",
        "definition": "A performance change produced by altered deployment of an intact trained system.",
        "machine_operation": "Increase tau, lapse, noise or memory decay; reduce update gain with fixed weights/resources.",
        "human_measure": "Session multi-axis state profile; within-session instability; physiological vigilance/workload features.",
        "expected_signature": "Similar mean impairment can occur with distinctive lapse, entropy, sequential-dependence and physiological signatures.",
        "dataset": "Artificial agents; ds007554; COG-BCI",
        "script": "scripts/05_artificial_agents/build_artificial_agents.py; scripts/06_ann_intervention_gate/run_ann_gate.py; scripts/11_ds007554_neurophys/extract_and_model_neurophys.py",
        "output_table": "outputs/tables/artificial_perturbation_parameters.csv; outputs/tables/ann_intervention_gate_results.csv",
        "claim_allowed": "Only after performance-matched ANN state/capacity classification beats permutation controls.",
    },
    {
        "theoretical_construct": "capacity perturbation",
        "definition": "A performance change produced by limiting representational or recurrent resources.",
        "machine_operation": "Reduce hidden size, recurrent rank, memory window, bottleneck width or remove recurrence.",
        "human_measure": "Participant multi-axis capacity profile; load sensitivity; recurrent geometry.",
        "expected_signature": "Reduced trajectory dimensionality, lower recurrent rank/timescale, steeper load-related decline.",
        "dataset": "Artificial agents; ds007554; TU Berlin; HBN",
        "script": "scripts/05_artificial_agents/build_artificial_agents.py; scripts/09_dynamics/run_dynamics.py; scripts/13_external_tu_berlin/run_tu_berlin_validation.py",
        "output_table": "outputs/tables/artificial_dynamics_fingerprints.csv; outputs/tables/recurrent_dynamics_by_subject_task.csv",
        "claim_allowed": "Only if capacity recovery passes ANN gate and human geometry association survives robustness tests.",
    },
    {
        "theoretical_construct": "state rescue",
        "definition": "Improvement caused by restoring operating regime without changing representational resources.",
        "machine_operation": "Move tau/lapse/sigma/gain/memory_decay from impaired values back toward baseline in fixed-capacity agents.",
        "human_measure": "Within-person improvement across sessions or conditions without a capacity-profile change.",
        "expected_signature": "Reliability improves while capacity-like geometry remains stable.",
        "dataset": "Artificial agents; COG-BCI repeated sessions",
        "script": "scripts/06_ann_intervention_gate/run_ann_gate.py; scripts/12_external_cog_bci/run_cog_bci_validation.py",
        "output_table": "outputs/tables/ann_hybrid_recovery.csv; outputs/tables/cog_bci_validation_models.csv",
        "claim_allowed": "Exploratory unless repeated-session data show state change with stable capacity.",
    },
    {
        "theoretical_construct": "capacity rescue",
        "definition": "Improvement caused by increasing representational resources rather than merely changing operating state.",
        "machine_operation": "Increase hidden size/rank/window/bottleneck/recurrence under fixed state parameters.",
        "human_measure": "Lower load-related decline in higher-capacity participants; stronger latent separability.",
        "expected_signature": "Performance under capacity pressure improves with geometry/timescale rather than only state reliability.",
        "dataset": "Artificial agents; TU Berlin; COG-BCI",
        "script": "scripts/05_artificial_agents/build_artificial_agents.py; scripts/13_external_tu_berlin/run_tu_berlin_validation.py",
        "output_table": "outputs/tables/artificial_behavioral_fingerprints.csv; outputs/tables/tu_berlin_load_validation.csv",
        "claim_allowed": "Only if load/capacity-pressure tests survive mean-performance controls.",
    },
    {
        "theoretical_construct": "behavioral fingerprint",
        "definition": "Multi-feature behavioral pattern used to distinguish perturbation families beyond mean accuracy.",
        "machine_operation": "Compute accuracy, NLL, entropy, lapse, slopes, load sensitivity, sequential dependence and response patterns.",
        "human_measure": "Session/task fingerprint projected into artificial state-profile and capacity-profile spaces.",
        "expected_signature": "State and capacity profiles remain separable after residualizing accuracy and NLL.",
        "dataset": "Artificial agents; all human datasets with behavioral events",
        "script": "scripts/05_artificial_agents/build_artificial_agents.py; scripts/08_estimate_coordinates/estimate_coordinates.py",
        "output_table": "outputs/tables/artificial_behavioral_fingerprints.csv; outputs/tables/multiaxis_projection_residualized_tests.csv",
        "claim_allowed": "Only if parameter-derived profiles and fingerprint projections converge across preregistered axes.",
    },
    {
        "theoretical_construct": "dynamical fingerprint",
        "definition": "Internal recurrent-state pattern used to identify resource-like versus state-like mechanisms.",
        "machine_operation": "Compute participation ratio, radius, step norm, rank, spectral radius, memory timescale and latent decoding.",
        "human_measure": "Dynamical summaries of fitted human TinyRNNs.",
        "expected_signature": "Capacity tracks participation ratio and/or local memory timescale under bootstrap and robustness tests.",
        "dataset": "Artificial agents; ds007554; HBN scalability",
        "script": "scripts/09_dynamics/run_dynamics.py",
        "output_table": "outputs/tables/recurrent_dynamics_by_subject_task.csv; outputs/tables/fixed_point_summary.csv",
        "claim_allowed": "Strong only if robust to hidden-size sensitivity, architecture variants and shuffled/random controls.",
    },
    {
        "theoretical_construct": "neurophysiology alignment",
        "definition": "Association between machine-defined state/capacity profiles and independent EEG/fNIRS/ECG features.",
        "machine_operation": "No direct machine operation; use physiology as external alignment evidence.",
        "human_measure": "EEG bands/entropy/slope, fNIRS HbO/HbR features, ECG HRV features.",
        "expected_signature": "State-profile axes align with reliability/vigilance physiology beyond task and mean accuracy; capacity-profile axes align with stable or load-related features.",
        "dataset": "ds007554, COG-BCI, TU Berlin",
        "script": "scripts/11_ds007554_neurophys/extract_and_model_neurophys.py; scripts/12_external_cog_bci/run_cog_bci_validation.py; scripts/13_external_tu_berlin/run_tu_berlin_validation.py",
        "output_table": "outputs/tables/ds007554_neurophys_models.csv; outputs/tables/cog_bci_validation_models.csv; outputs/tables/tu_berlin_load_validation.csv",
        "claim_allowed": "Only if effects survive task and mean-accuracy controls.",
    },
    {
        "theoretical_construct": "capacity pressure",
        "definition": "Task demand that selectively exposes resource limitations rather than generic poor state.",
        "machine_operation": "Increase load or memory demand under fixed state and compare capacity levels.",
        "human_measure": "Load-related decline, latent separability, workload physiology and state/capacity interaction.",
        "expected_signature": "Lower capacity predicts steeper load decline; state predicts within-load reliability variation.",
        "dataset": "TU Berlin, COG-BCI, ds007554 N-back arithmetic",
        "script": "scripts/13_external_tu_berlin/run_tu_berlin_validation.py; scripts/10_ds007554_discovery/run_discovery.py",
        "output_table": "outputs/tables/tu_berlin_load_validation.csv; outputs/tables/ds007554_discovery_model_comparison.csv",
        "claim_allowed": "Only if capacity-pressure evidence remains after controlling for mean performance.",
    },
    {
        "theoretical_construct": "consciousness relevance",
        "definition": "Broader theoretical motivation that unresponsiveness may reflect state, capacity or both.",
        "machine_operation": "None in this empirical study; mentioned only as future relevance.",
        "human_measure": "None; no clinical or consciousness diagnosis in current datasets.",
        "expected_signature": "Not tested here.",
        "dataset": "Not applicable",
        "script": "Not applicable",
        "output_table": "Not applicable",
        "claim_allowed": "No empirical consciousness claim allowed.",
    },
]


def main() -> None:
    assert_full_run_allowed()
    output = PROJECT_ROOT / "outputs" / "audit" / "theory_to_method_crosswalk.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"THEORY_CROSSWALK: wrote {len(ROWS)} constructs to {output}")


if __name__ == "__main__":
    main()
