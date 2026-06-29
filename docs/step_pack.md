# Step Pack: Revised State-Capacity TinyRNN Study

This step pack explains the reconstructed full-study protocol in plain language. It is the operational map for the new `state_capacity_tinyrnn` folder.

## Governing Rule

The previous project is now an archive and design reference. The new study must regenerate every result from scripts in this folder. No empirical claim can enter the manuscript unless it appears in `outputs/audit/claim_audit.tsv` with a script, output file, sample size and claim-strength label.

## Step 00: Audit Existing Work

Goal: preserve the previous Codex implementation without trusting it as final.

What it does:

- Finds previous step folders `00_project_setup` through `13_neurophys_extension`.
- Inventories scripts, tables, figures, processed files and model checkpoints.
- Marks pilots, debug outputs and full-run candidates.
- Archives previous outputs under `outputs/archive_previous_run`.
- Writes a migration map from old folders to the revised protocol.

Main outputs:

- `outputs/audit/existing_project_inventory.csv`
- `outputs/audit/previous_results_reproducibility_status.md`
- `outputs/audit/migration_plan.md`

Pass condition:

- Previous work is archived.
- Nothing from the previous project is accepted as final.

Current status: implemented and run.

## Step 01: Dataset Download

Goal: obtain only directly downloadable, reproducible datasets.

What it does:

- Downloads allowed main datasets:
  - OpenNeuro `ds007554`
  - COG-BCI Zenodo record `6874129`
  - TU Berlin simultaneous EEG-NIRS
  - HBN EEG Release 4 / OpenNeuro `ds005508`
- Records checksums, file sizes and download commands.
- Creates dataset eligibility table.
- Excludes login/manual/private/browser-only datasets from main analyses.

Main outputs:

- `outputs/manifests/download_manifest.json`
- `outputs/manifests/checksums.tsv`
- `outputs/logs/download_all.log`
- `outputs/audit/dataset_eligibility.tsv`

Pass condition:

- Every main dataset has a command-line provenance record.
- Any dataset requiring manual login or controlled access is excluded from confirmatory claims.

Current status: implemented and run.

## Step 02: Raw Data Inventory

Goal: prove what data were actually downloaded.

What it does:

- Counts subjects, sessions, tasks, behavioral files, EEG files, fNIRS files, ECG files and questionnaires.
- Compares observed counts with expected counts.
- Flags missing, corrupt or incomplete files.

Main outputs:

- `outputs/manifests/raw_file_inventory.csv`
- `outputs/tables/dataset_subject_session_counts.csv`
- `outputs/logs/raw_inventory_failures.log`

Pass condition:

- Included datasets have at least 95% of expected subjects unless a source-side reason is documented.

Current status: implemented and run.

## Step 03: Unified Behavioral Event Schema

Goal: convert all datasets into one common event-level table.

What it does:

- Converts every dataset into rows with shared fields:
  - dataset, subject, session, task, block, trial index, timestamp, condition, load, stimulus, target, response, correctness, RT and recent history.
- Records event exclusions and missingness.
- Prevents model training until event counts exist.

Main outputs:

- `data/processed/all_model_events.parquet`
- `data/processed/ds007554_model_events.parquet`
- `data/processed/cog_bci_model_events.parquet`
- `data/processed/tu_berlin_model_events.parquet`
- `data/processed/hbn_model_events.parquet`
- `outputs/tables/event_counts_by_dataset.csv`
- `outputs/tables/event_exclusion_counts.csv`
- `outputs/tables/event_missingness_report.csv`

Pass condition:

- All included datasets have event count, exclusion and missingness reports.

Current status: implemented and run.

## Step 04: Theory-To-Method Crosswalk

Goal: connect the theoretical manuscript to measurable operations, using multidimensional state and capacity profiles rather than treating either construct as one mandatory scalar coordinate.

What it does:

- Defines each theoretical construct.
- Maps each construct to a machine operation, human measure, expected signature, dataset, script and output table.
- Decides whether a claim is allowed and under what conditions.

Required constructs:

- global computational state as a multi-axis operating-state profile
- structural capacity as a multi-axis resource/capacity profile
- hybrid manipulation
- state perturbation
- capacity perturbation
- state rescue
- capacity rescue
- behavioral fingerprint
- dynamical fingerprint
- neurophysiology alignment
- capacity pressure
- consciousness relevance

Main output:

- `outputs/audit/theory_to_method_crosswalk.tsv`

Pass condition:

- No manuscript Introduction or Discussion should be finalized before this file exists.

Current status: implemented and run.

## Step 05: Machine Perturbation Library

Goal: define state and capacity first in artificial agents, before human interpretation.

What it does:

- Trains baseline TinyRNN/GRU agents.
- Creates state perturbations by changing operating regime with fixed resources:
  - tau
  - lapse
  - hidden noise sigma
  - update gain
  - memory decay
- Creates capacity perturbations by changing representational resources:
  - hidden size
  - recurrent rank
  - memory window
  - bottleneck width
  - recurrence type
- Creates hybrid agents crossing state severity and capacity level.

Main outputs:

- `outputs/tables/artificial_agent_registry.csv`
- `outputs/tables/artificial_perturbation_parameters.csv`
- `outputs/tables/artificial_behavioral_fingerprints.csv`
- `outputs/tables/artificial_dynamics_fingerprints.csv`
- `outputs/model_checkpoints/artificial_agents/`

Pass condition:

- Every artificial agent has a unique ID, family, seed, task set, parameters and checkpoint.

Current status: implemented and run.

## Step 06: Performance Matching

Goal: prevent state and capacity from collapsing into generic severity.

What it does:

- Matches state-perturbed and capacity-limited agents with similar mean accuracy and negative log-likelihood.
- Creates matched pairs for fair comparison.

Main outputs:

- `outputs/tables/performance_matched_agent_pairs.csv`
- `outputs/figures/figure_agent_performance_matching.png`
- `outputs/source_data/figure_agent_performance_matching_source.csv`

Pass condition:

- At least 20 matched state-capacity pairs per main task family, or a documented failure report.

Current status: implemented and run.

## Step 07: ANN-Only Intervention Gate

Goal: test whether the framework works in machines before projecting humans.

What it does:

- Classifies state versus capacity perturbation under matched performance.
- Recovers known state perturbation dimensions and their composite severity.
- Recovers known capacity-resource dimensions and their composite level.
- Recovers both multidimensional profiles in hybrid agents.
- Repeats after residualizing mean accuracy and NLL.
- Runs shuffled-label controls.

Main outputs:

- `outputs/tables/ann_intervention_gate_results.csv`
- `outputs/tables/ann_hybrid_recovery.csv`
- `outputs/tables/ann_shuffled_label_controls.csv`
- `outputs/figures/figure_ann_intervention_gate.png`
- `outputs/source_data/figure_ann_intervention_gate_source.csv`
- `outputs/audit/ann_gate_decision.md`

Pass condition:

- State/capacity classifier exceeds permutation 95th percentile.
- State recovery Spearman >= 0.70.
- Capacity recovery Spearman >= 0.70.
- Hybrid recovery significant for both axes.
- Residualized results beat shuffled-label controls.

If failed:

- Human state claims must be exploratory or negative.
- Human capacity claims may remain as partial proxy evidence only if projection, geometry and robustness steps converge.

Current status: implemented and run; gate failed under the preregistered residualized criterion.

## Step 08: Human TinyRNN Training

Goal: train task-conditioned recurrent models on human behavior.

What it does:

- Trains GRU models with hidden sizes `[1, 2, 3, 4, 6, 8]`.
- Uses leakage-controlled splits:
  - participant-level split
  - session-blocked split
  - odd/even mini-block split
- Estimates candidate state parameters:
  - log tau
  - logit lapse
  - log sigma
  - negative log gain
- Exports multidimensional session-state profiles:
  - response reliability
  - lapse/ceiling risk
  - RT instability
  - sequential instability
  - even-trial state summary
- Exports multidimensional participant-capacity profiles:
  - selected hidden size
  - selection confidence
  - complexity preference
  - high-capacity NLL advantage
  - load robustness
  - cross-task consistency

Main outputs:

- `outputs/tables/model_selection_by_subject.csv`
- `outputs/tables/session_state_parameters.csv`
- `outputs/tables/session_state_multiaxis_coordinates.csv`
- `outputs/tables/session_state_quality_report.csv`
- `outputs/tables/participant_capacity_coordinates.csv`
- `outputs/tables/participant_capacity_multidimensional_coordinates.csv`
- `outputs/tables/training_validation_metrics.csv`
- `outputs/model_checkpoints/human_models/`
- `outputs/logs/training_logs/`

Pass condition:

- Scripts prove held-out events were not used to estimate state parameters.

Current status: implemented and run.

## Step 09: Human Fingerprint Projection

Goal: project human behavior into the artificial perturbation space and test convergence for multidimensional state and capacity profiles.

What it does:

- Uses Step 08 parameter-derived state and capacity profiles as one measurement family.
- Projects human behavioral/dynamical fingerprints into the Step 05 artificial-agent perturbation profile space as an independent measurement family.
- Tests axis-wise and composite convergence between parameter-derived profiles and fingerprint-projected profiles.
- Tests whether profile axes remain after residualizing accuracy, RT, NLL, task and dataset.
- Carries forward the Step 07 negative state gate by marking state-profile claims exploratory unless this step shows strong convergent evidence.

Main outputs:

- `outputs/tables/human_state_capacity_multiaxis_projection.csv`
- `outputs/tables/multiaxis_profile_convergence_tests.csv`
- `outputs/tables/multiaxis_projection_residualized_tests.csv`
- `outputs/audit/human_projection_claim_limits.md`

Pass condition:

- Primary human claims require convergence between parameter-derived and fingerprint-projected profiles, not merely a useful summary coordinate.
- Confirmatory state claims require enough evidence to overcome the current ANN gate failure; otherwise state remains exploratory.
- Capacity claims require convergent multi-axis capacity evidence plus later geometry/robustness support.

Current status: implemented and rerun after repairing HBN labels, reconstructing ds007554 N-back/N-back-arithmetic correctness from push-button timing, expanding COG-BCI behavior with N-back/Flanker EEGLAB markers, and rerunning recurrent dynamics. The revised projection includes COG-BCI, ds007554, HBN Release 4, and TU Berlin with 1,283 / 1,283 rows carrying attached recurrent-dynamics geometry. Residualized tests still do not support a strong independent human projection claim; state remains exploratory and capacity remains qualified.

## Step 10: ds007554 Discovery

Goal: run the primary discovery analysis in the original multimodal dataset.

What it does:

- Compares:
  - task-only model
  - one-dimensional impairment model
  - additive state-capacity model
  - state-capacity interaction model
  - hierarchical logistic model
  - no-state GRU
  - behavioral descriptive baseline
  - random-axis control
  - shuffled-label control
- Reports LOPO RMSE, held-out NLL, weighted R2, permutation p-values, bootstrap CIs and BH-adjusted p-values.

Main outputs:

- `outputs/tables/ds007554_discovery_model_comparison.csv`
- `outputs/tables/ds007554_permutation_tests.csv`
- `outputs/tables/ds007554_bootstrap_effects.csv`
- `outputs/figures/figure_ds007554_discovery.png`
- `outputs/source_data/figure_ds007554_discovery_source.csv`

Pass condition:

- Task-only and behavioral baselines are reported even if they outperform TinyRNN.

Current status: implemented and rerun after the ds007554 push-button correctness repair, COG-BCI EEGLAB-marker expansion, recurrent-dynamics refresh, and Step 09 projection refresh. ds007554 is now analyzed as a primary reconstructed-correctness scope with 120 session-task rows from 29 participants, plus an all-supervised-datasets contextual scope. The descriptive behavioral baseline is the best ds007554 LOPO model, while machine-projection and additive state/capacity models improve over task/dataset-only baselines. Claims must say ds007554 correctness was reconstructed from timing signals.

## Step 11: Recurrent Dynamics

Goal: test whether capacity is a real dynamical geometry coordinate.

What it does:

- Extracts hidden trajectories.
- Computes:
  - participation ratio
  - trajectory radius
  - step norm
  - recurrent rank
  - fixed points
  - Jacobian spectral radius
  - local memory timescale
  - latent decoder accuracy

Main outputs:

- `outputs/tables/recurrent_dynamics_by_subject_task.csv`
- `outputs/tables/fixed_point_summary.csv`
- `outputs/tables/latent_decoder_results.csv`
- `outputs/figures/figure_recurrent_dynamics.png`
- `outputs/source_data/figure_recurrent_dynamics_source.csv`

Pass condition:

- Capacity-geometry claim is strong only if capacity associates with participation ratio and/or memory timescale under bootstrap and robustness tests.

Current status: implemented and rerun after Step 08 was regenerated on the expanded supervised event table. The current analysis extracted 1,283 session-task hidden trajectories, 115,979 hidden-state event rows, 54 fixed-point/Jacobian summaries, recurrent-weight geometry, latent decoder tests, and Nature-style figure exports. The strongest capacity/geometry association is trajectory covariance rank with capacity profile (rho = 0.713, p = 1.53e-199, n = 1,283), but this remains fitted-model geometry rather than direct neural dynamics.

## Step 12: ds007554 Neurophysiology

Goal: link model coordinates to independent EEG, fNIRS and ECG features.

What it does:

- Extracts EEG features:
  - delta, theta, alpha, beta power
  - theta/alpha ratio
  - frontal-midline theta if channels allow
  - spectral entropy
  - aperiodic slope
  - trialwise variability
- Extracts fNIRS features:
  - HbO/HbR mean
  - HbO/HbR slope
  - prefrontal task response
- Extracts ECG features:
  - heart rate
  - RMSSD
  - SDNN
  - pNN50
  - heart-rate slope
- Models physiology against state/capacity when behavioral coordinates are available; otherwise writes explicit ineligibility rows and runs task-load physiology checks with session/task controls.

Main outputs:

- `outputs/tables/ds007554_neurophys_features.csv`
- `outputs/tables/ds007554_eeg_features.csv`
- `outputs/tables/ds007554_physio_features.csv`
- `outputs/tables/ds007554_fnirs_features.csv`
- `outputs/tables/ds007554_fnirs_inventory.csv`
- `outputs/tables/ds007554_neurophys_models.csv`
- `outputs/figures/figure_ds007554_neurophys.png`
- `outputs/source_data/figure_ds007554_neurophys_source.csv`
- `outputs/logs/step12_neurophys_status.json`
- `outputs/audit/step12_neurophys_claim_audit.md`

Pass condition:

- Neurophysiology claims require effects beyond task and mean accuracy.

Current status: implemented and rerun after installing `mne`, `mne-nirs`, `neurokit2`, `pyedflib`, and `h5py` into the project environment. The run processed 515 / 515 EEG EDF files with `pyedflib`, 781 / 781 ECG/Biodex physiology files, and 519 / 519 fNIRS SNIRF files with HbO/HbR conversion. ds007554 N-back/N-back-arithmetic correctness was reconstructed from push-button timing, Step 08 coordinates were regenerated, and Step 12 now runs direct state/capacity neurophysiology models on 475 rows with attached coordinates. These direct models remain qualified because the ds007554 behavioral labels are reconstructed from timing signals rather than provided as explicit trial-correctness columns.

## Step 13: COG-BCI External Validation

Goal: perform the primary external validation.

What it does:

- Imports COG-BCI behavior, EEG, ECG and subjective measures.
- Tests whether state varies more within person than capacity.
- Tests whether capacity is more test-retest stable than state.
- Tests whether state predicts subjective vigilance/workload and physiological reliability markers.
- Tests whether capacity predicts cross-task consistency or load robustness.

Main outputs:

- `data/processed/cog_bci_model_events.parquet`
- `outputs/tables/cog_bci_eeg_features.csv`
- `outputs/tables/cog_bci_coordinates.csv`
- `outputs/tables/cog_bci_validation_models.csv`
- `outputs/figures/figure_cog_bci_validation.png`
- `outputs/source_data/figure_cog_bci_validation_source.csv`
- `outputs/logs/step13_cog_bci_status.json`
- `outputs/audit/step13_cog_bci_validation_audit.md`

Pass condition:

- This is the primary external validation. If it fails, the manuscript claim must be downgraded.

Current status: implemented and run full. COG-BCI preprocessing now includes PVT behavior plus N-back and Flanker trials reconstructed from EEGLAB event markers. Step 13 extracted EEG features for 435 / 435 task EEG recordings across 29 subjects, 3 sessions, and PVT/Flanker/N-back tasks, merged them with 261 state/capacity coordinate rows, and ran 21 validation-model rows. Results support state as a within-person/session-varying reliability/RT-variability coordinate and give qualified support for capacity via cross-task consistency and selected EEG associations. ECG and subjective workload were not available in the local COG-BCI files parsed here, and participant-level capacity limits formal test-retest claims.

## Step 14: TU Berlin EEG-NIRS Validation

Goal: test direct N-back load and multimodal physiology validation.

What it does:

- Imports TU Berlin simultaneous EEG-NIRS N-back data.
- Tests whether increasing load raises difficulty and changes physiology.
- Tests whether lower capacity predicts steeper load-related decline.
- Tests whether state predicts within-load reliability/lapse/RT variability.

Main outputs:

- `data/processed/tu_berlin_model_events.parquet`
- `outputs/tables/tu_berlin_coordinates.csv`
- `outputs/tables/tu_berlin_load_validation.csv`
- `outputs/figures/figure_tu_berlin_validation.png`
- `outputs/source_data/figure_tu_berlin_validation_source.csv`

Pass condition:

- TU Berlin supports load/capacity-pressure only if load-related decline or physiology aligns with capacity beyond mean performance.

Current status: implemented and run full. Step 14 uses TU Berlin N-back behavior from 26 subjects, 234 subject-session-load behavioral coordinate rows, 702 EEG N-back block windows and 702 NIRS N-back block windows. It confirms strong load pressure on accuracy and RT, a BH-significant load-by-capacity interaction for accuracy and RT, and load-related EEG theta/alpha, EEG aperiodic slope and NIRS HbR effects. State effects on lapse and RT variability are not BH-significant here, so TU Berlin supports capacity-pressure more strongly than state.

## Step 15: HBN Scalability

Goal: test whether the pipeline scales to hundreds of participants.

What it does:

- Runs HBN Release 4 as supplementary scalability analysis.
- Tests model stability, state-like reliability and capacity-like dimensionality.
- Reports age/developmental effects as exploratory only.

Main outputs:

- `outputs/tables/hbn_scalability_summary.csv`
- `outputs/tables/hbn_model_performance.csv`
- `outputs/figures/figure_hbn_scalability.png`
- `outputs/source_data/figure_hbn_scalability_source.csv`

Pass condition:

- HBN demonstrates scale and stability, not direct proof of the core adult N-back/PVT theory.

Current status: implemented and run full. Step 15 uses HBN Release 4 as a scalability and developmental-exploratory dataset: 324 subjects, 180,824 included events, 27,133 supervised events, 10 tasks, 746 state rows, 306 capacity rows and 746 recurrent-dynamics rows. The pipeline scales to hundreds of participants. Capacity-like profiles align strongly with recurrent geometry, especially trajectory covariance rank (rho = 0.533, p = 7.87e-24, q = 3.94e-23) and trajectory radius (rho = 0.729, p = 5.56e-52, q = 4.17e-51). HBN should not be written as direct adult N-back/PVT evidence because most events are passive-video or EEG markers, RT is absent in the unified table and correctness is task-limited. Developmental effects are exploratory; age versus capacity is not BH-significant.

## Step 16: Robustness and Falsification

Goal: try to break the results.

What it does:

- Tests hidden-size sensitivity.
- Tests GRU, vanilla RNN and LSTM variants.
- Runs random-axis controls.
- Runs shuffled-label controls.
- Residualizes severity.
- Reverses odd/even splits.
- Runs leave-one-task-out, leave-one-dataset-out, participant bootstrap and dataset bootstrap.

Main outputs:

- `outputs/tables/robustness_master_table.csv`
- `outputs/tables/falsification_tests.csv`
- `outputs/figures/figure_robustness.png`
- `outputs/source_data/figure_robustness_source.csv`

Pass condition:

- If shuffled labels or random axes reproduce the main effects, relevant claims fail.

Current status: implemented and run full. Step 16 produced 160 robustness rows and 18 falsification rows. It covers hidden-size sensitivity, residualized controls, leave-one-dataset/task tests, participant bootstrap, external-validation checks, random/shuffled controls, permutation controls and baseline comparisons. Robustness is strongest for capacity when tied to recurrent geometry and capacity-pressure/load validation. State remains limited by failed ANN residualized state recovery, nonsignificant TU Berlin state tests and weak HBN participant-level state validation. The ds007554 descriptive behavioral baseline still outperforms coordinate models and must be reported. Vanilla RNN/LSTM architecture variants are not present in the current trained-model outputs, so Step 16 does not claim architecture-variant robustness beyond the available GRU hidden-size sensitivity.

## Step 17: Master Statistics Table

Goal: create the single source of truth for all empirical claims.

What it does:

- Combines all tested effects into one table.
- Adds sample sizes, scripts, output files, splits, p-values and claim strengths.

Main output:

- `outputs/tables/master_effects_table.csv`

Required claim strengths:

- strong
- moderate
- exploratory
- negative
- failed_validation

Pass condition:

- Every manuscript claim must map to this table and the claim audit.

Current status: implemented and run full. Step 17 produced `master_effects_table.csv` with 454 effect rows and `claim_audit.tsv` as the manuscript-facing audit table. It intentionally includes positive, exploratory, negative and failed-validation results across state, capacity, interaction, controls, load, neurophysiology, COG-BCI, TU Berlin, HBN and robustness/falsification. Coverage includes 279 state rows, 114 capacity rows and 32 interaction rows. State effects are mostly exploratory or negative; capacity has the strongest support when tied to recurrent geometry, COG validation, TU capacity pressure and HBN scalability.

## Step 18: Figures

Goal: generate journal-level figures with reproducible source data.

Figures:

- Figure 1: theory-to-method pipeline and machine perturbation logic
- Figure 2: ANN perturbation gate and performance matching
- Figure 3: ds007554 discovery coordinates
- Figure 4: recurrent dynamics and capacity geometry
- Figure 5: ds007554 neurophysiology alignment
- Figure 6: COG-BCI external validation
- Figure 7: TU Berlin/HBN validation and scalability
- Figure 8: robustness, baselines and falsification

Every figure must include:

- PNG
- PDF
- SVG
- `source_data.csv`
- `data_dictionary.md`
- `script_used.txt`

Pass condition:

- No figure is accepted without source data and a data dictionary.

Current status: implemented and run full. Step 18 generated final Figures 1-8 as PNG, PDF and SVG, each with a source CSV, data dictionary and script-used file. The final figures cover pipeline/claim coverage, ANN gate, ds007554 discovery, recurrent dynamics, ds007554 neurophysiology, COG-BCI validation, TU Berlin/HBN validation and robustness/falsification. The figure set is reproducible from `scripts/16_statistics_figures/make_figures.py` and is designed to show state, capacity, interactions, validation failures and controls rather than only capacity-positive results.

## Step 19: Manuscript Package

Goal: generate the final manuscript materials only after all gates pass or fail honestly.

What it does:

- Writes title/abstract, Introduction, Results, Methods, Discussion, Limitations and figure captions.
- Writes claim audit, reproducibility checklist, data availability, code availability and overclaiming report.
- Writes final decision report.

Main outputs:

- `outputs/manuscript_text/title_abstract.md`
- `outputs/manuscript_text/introduction.md`
- `outputs/manuscript_text/results.md`
- `outputs/manuscript_text/methods.md`
- `outputs/manuscript_text/discussion.md`
- `outputs/manuscript_text/limitations.md`
- `outputs/manuscript_text/figure_captions.md`
- `outputs/audit/claim_audit.tsv`
- `outputs/audit/reproducibility_checklist.md`
- `outputs/audit/data_availability_statement.md`
- `outputs/audit/code_availability_statement.md`
- `outputs/audit/overclaiming_risk_report.md`
- `outputs/audit/final_decision_report.md`

Pass condition:

- The manuscript may not be finalized until `final_decision_report.md` exists.

Current status: implemented and run full. Step 19 generated a manuscript draft package under `outputs/manuscript_text`, including title/abstract, Introduction, Results, Methods, Discussion, Limitations, figure captions and `full_manuscript_draft.md`. It also generated the final decision report, overclaiming risk report, reproducibility checklist and data/code availability statements under `outputs/audit`. The recommended framing is conservative: capacity is strong as recurrent-geometry/load-pressure evidence; state is meaningful but exploratory; direct ds007554 state-capacity interaction is weak; TU Berlin load-by-capacity pressure is strong.

## Final Decision Gate

The final decision report must answer:

1. Did the artificial-agent gate pass?
2. Did simulation/hybrid recovery pass?
3. Did ds007554 reproduce the discovery result?
4. Did capacity associate with recurrent geometry?
5. Did state associate with reliability/lapse/RT variability?
6. Did ds007554 neurophysiology align with state/capacity?
7. Did COG-BCI validate the coordinate structure?
8. Did TU Berlin validate load/capacity-pressure?
9. Did HBN demonstrate scalability?
10. Did behavioral baselines outperform TinyRNN?
11. Which claims are strong, moderate, exploratory, negative or failed?
12. Is the manuscript suitable for Nature Human Behaviour, Nature Machine Intelligence, or should it be reframed?

## Practical Execution Order

Use this order:

1. `make audit_existing`
2. `make download_all`
3. `make inventory`
4. `make preprocess_all`
5. `make theory_crosswalk`
6. `make artificial_agents`
7. `make ann_gate`
8. Continue human training and validation only if the ANN gate passes, or explicitly reframe as exploratory/negative.
