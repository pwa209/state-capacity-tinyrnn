PY := ..\.venv\Scripts\python.exe

.PHONY: setup test smoke audit_existing download_all inventory preprocess_all theory_crosswalk artificial_agents performance_matching ann_gate train_all estimate_coordinates discovery dynamics neurophys external_all robustness figures manuscript_package full

setup:
	$(PY) -m pip install -e .

test:
	$(PY) -m unittest discover -s tests

smoke:
	$(PY) scripts/00_audit_existing/audit_existing.py --smoke

audit_existing:
	$(PY) scripts/00_audit_existing/audit_existing.py

download_all:
	$(PY) scripts/01_download/download_all.py

inventory:
	$(PY) scripts/02_inventory/raw_inventory.py

preprocess_all:
	$(PY) scripts/03_preprocess_behavior/preprocess_all.py

theory_crosswalk:
	$(PY) scripts/04_theory_crosswalk/build_crosswalk.py

artificial_agents:
	$(PY) scripts/05_artificial_agents/build_artificial_agents.py

performance_matching:
	$(PY) scripts/06_performance_matching/run_performance_matching.py

ann_gate:
	$(PY) scripts/06_ann_intervention_gate/run_ann_gate.py

train_all:
	$(PY) scripts/07_train_tinyrnn/train_all.py

estimate_coordinates:
	$(PY) scripts/08_estimate_coordinates/estimate_coordinates.py

discovery:
	$(PY) scripts/10_ds007554_discovery/run_discovery.py

dynamics:
	$(PY) scripts/09_dynamics/run_dynamics.py

neurophys:
	$(PY) scripts/11_ds007554_neurophys/extract_and_model_neurophys.py

external_all:
	$(PY) scripts/12_external_cog_bci/run_cog_bci_validation.py
	$(PY) scripts/13_external_tu_berlin/run_tu_berlin_validation.py
	$(PY) scripts/14_external_hbn/run_hbn_scalability.py

robustness:
	$(PY) scripts/15_baselines_robustness/run_robustness.py

figures:
	$(PY) scripts/16_statistics_figures/make_figures.py

manuscript_package:
	$(PY) scripts/17_manuscript_package/build_manuscript_package.py

full: audit_existing download_all inventory preprocess_all theory_crosswalk artificial_agents performance_matching ann_gate train_all estimate_coordinates discovery dynamics neurophys external_all robustness figures manuscript_package
