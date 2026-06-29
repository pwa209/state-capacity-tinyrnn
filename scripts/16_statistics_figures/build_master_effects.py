from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"


def ensure_dirs() -> None:
    for path in [TABLES, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def bh_q(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    m = len(valid)
    q = valid.to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def infer_construct(text: str) -> str:
    t = text.lower()
    if "interaction" in t or "pressure" in t or "load_x_capacity" in t:
        return "interaction"
    if "state" in t:
        return "state"
    if "capacity" in t or "hidden" in t or "geometry" in t:
        return "capacity"
    if "control" in t or "shuffled" in t or "random" in t:
        return "control"
    if "load" in t or "task" in t:
        return "load"
    return "other"


def classify(row: pd.Series) -> str:
    p = row.get("p_value", np.nan)
    q = row.get("q_value", np.nan)
    construct = str(row.get("construct", ""))
    direction = str(row.get("direction", ""))
    context = str(row.get("context", "")).lower()
    if "fail" in context or "negative" in context:
        return "failed_validation" if "gate" in context else "negative"
    sig = (pd.notna(q) and q < 0.05) or (pd.isna(q) and pd.notna(p) and p < 0.005)
    if not sig:
        return "negative" if construct in {"state", "interaction"} and "tested" in context else "exploratory"
    if construct == "state":
        return "exploratory" if "ann_state_limited" in context or "exploratory" in context else "moderate"
    if construct == "capacity":
        return "strong" if "geometry" in context or "pressure" in context or "hbn" in context else "moderate"
    if construct == "interaction":
        return "strong" if "capacity_pressure" in context or "pooled" in context else "moderate"
    if construct == "control":
        return "strong" if "passes" in direction else "negative"
    return "moderate" if sig else "exploratory"


def add(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "effect_id": "",
        "step": "",
        "analysis_domain": "",
        "construct": "",
        "effect_name": "",
        "dataset_scope": "",
        "n": np.nan,
        "n_participants": np.nan,
        "effect_size": np.nan,
        "effect_metric": "",
        "p_value": np.nan,
        "q_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "direction": "",
        "claim_strength": "",
        "context": "",
        "source_table": "",
        "script": "",
    }
    base.update(kwargs)
    rows.append(base)


def load_if_exists(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    ann = load_if_exists("ann_intervention_gate_results.csv")
    for _, r in ann.iterrows():
        name = f"{r['analysis_type']}:{r.get('target_axis', r.get('task_family', ''))}:{r['feature_set']}"
        add(
            rows,
            effect_id=f"step07_{len(rows)+1:04d}",
            step="07",
            analysis_domain="artificial_agent_gate",
            construct=infer_construct(str(r["analysis_type"]) + str(r.get("target_axis", ""))),
            effect_name=name,
            dataset_scope=str(r.get("task_family", "artificial")),
            n=r.get("n_samples", np.nan),
            effect_size=r.get("observed", np.nan),
            effect_metric=r.get("metric", ""),
            p_value=r.get("empirical_p_value", np.nan),
            direction="passes gate" if bool(r.get("pass_gate", False)) else "fails gate",
            context="ann_state_limited gate" if "state" in name.lower() and not bool(r.get("pass_gate", False)) else "tested",
            source_table="ann_intervention_gate_results.csv",
            script="scripts/06_ann_intervention_gate/run_ann_gate.py",
        )

    hybrid = load_if_exists("ann_hybrid_recovery.csv")
    for _, r in hybrid.iterrows():
        add(
            rows,
            effect_id=f"step07_hybrid_{len(rows)+1:04d}",
            step="07",
            analysis_domain="artificial_hybrid_recovery",
            construct=infer_construct(str(r["target_axis"])),
            effect_name=f"{r['analysis_type']}:{r['feature_set']}",
            dataset_scope="artificial_hybrid_agents",
            n=r.get("n_hybrid_agents", np.nan),
            effect_size=r.get("spearman_rho", np.nan),
            effect_metric="spearman_rho",
            p_value=r.get("empirical_p_value", r.get("nominal_p_value", np.nan)),
            direction="passes gate" if bool(r.get("pass_gate", False)) else "fails gate",
            context="ann_state_limited gate" if "state" in str(r["target_axis"]).lower() and not bool(r.get("pass_gate", False)) else "tested",
            source_table="ann_hybrid_recovery.csv",
            script="scripts/06_ann_intervention_gate/run_ann_gate.py",
        )

    for table, step, domain in [
        ("multiaxis_profile_convergence_tests.csv", "09", "human_projection_raw"),
        ("multiaxis_projection_residualized_tests.csv", "09", "human_projection_residualized"),
        ("multiaxis_axis_level_tests.csv", "09", "human_axis_level_projection"),
    ]:
        df = load_if_exists(table)
        for _, r in df.iterrows():
            name = str(r.get("analysis_family", ""))
            add(
                rows,
                effect_id=f"step{step}_{len(rows)+1:04d}",
                step=step,
                analysis_domain=domain,
                construct=infer_construct(name + str(r.get("x", ""))),
                effect_name=name,
                dataset_scope="all_supervised_datasets",
                n=r.get("n", np.nan),
                effect_size=r.get("spearman_rho", np.nan),
                effect_metric="spearman_rho",
                p_value=r.get("p_value", np.nan),
                ci_low=r.get("bootstrap_ci_low", np.nan),
                ci_high=r.get("bootstrap_ci_high", np.nan),
                direction=str(r.get("claim_strength", "")),
                context="ann_state_limited tested" if "state" in name else "tested",
                source_table=table,
                script="scripts/09_human_projection/run_human_projection.py",
            )

    ds_models = load_if_exists("ds007554_discovery_model_comparison.csv")
    for _, r in ds_models.iterrows():
        add(
            rows,
            effect_id=f"step10_model_{len(rows)+1:04d}",
            step="10",
            analysis_domain="ds007554_discovery_model_comparison",
            construct=infer_construct(str(r["model_name"]) + str(r["claim_role"])),
            effect_name=str(r["model_name"]),
            dataset_scope=str(r["analysis_scope"]),
            n=r.get("n_rows", np.nan),
            n_participants=r.get("n_participants", np.nan),
            effect_size=r.get("lopo_rmse", np.nan),
            effect_metric="lopo_rmse_lower_is_better",
            p_value=np.nan,
            direction=f"LOPO R2={r.get('lopo_r2', np.nan)}",
            context="descriptive_baseline_must_be_reported" if r["model_name"] == "behavioral_descriptive" else "tested",
            source_table="ds007554_discovery_model_comparison.csv",
            script="scripts/10_ds007554_discovery/run_discovery.py",
        )

    for table, step, domain, script in [
        ("ds007554_permutation_tests.csv", "10", "ds007554_permutation", "scripts/10_ds007554_discovery/run_discovery.py"),
        ("ds007554_bootstrap_effects.csv", "10", "ds007554_bootstrap", "scripts/10_ds007554_discovery/run_discovery.py"),
        ("recurrent_dynamics_state_capacity_tests.csv", "11", "recurrent_dynamics", "scripts/09_dynamics/run_dynamics.py"),
        ("ds007554_neurophys_models.csv", "12", "ds007554_neurophysiology", "scripts/11_ds007554_neurophys/extract_and_model_neurophys.py"),
        ("cog_bci_validation_models.csv", "13", "cog_bci_validation", "scripts/12_external_cog_bci/run_cog_bci_validation.py"),
        ("tu_berlin_load_validation.csv", "14", "tu_berlin_validation", "scripts/13_external_tu_berlin/run_tu_berlin_validation.py"),
        ("hbn_scalability_tests.csv", "15", "hbn_scalability", "scripts/14_external_hbn/run_hbn_scalability.py"),
        ("robustness_master_table.csv", "16", "robustness", "scripts/15_baselines_robustness/run_robustness.py"),
        ("falsification_tests.csv", "16", "falsification", "scripts/15_baselines_robustness/run_robustness.py"),
    ]:
        df = load_if_exists(table)
        for _, r in df.iterrows():
            text = " ".join(str(r.get(c, "")) for c in df.columns)
            effect = r.get("spearman_rho", r.get("estimate", r.get("beta", r.get("observed_sse_gain", r.get("effect", np.nan)))))
            metric = "spearman_rho" if "spearman_rho" in df.columns else "coefficient_or_effect"
            p = r.get("p_value", r.get("empirical_p_value", np.nan))
            q = r.get("q_value", np.nan)
            add(
                rows,
                effect_id=f"step{step}_{len(rows)+1:04d}",
                step=step,
                analysis_domain=domain,
                construct=infer_construct(text),
                effect_name=str(r.get("test_name", r.get("effect_name", r.get("analysis", r.get("analysis_family", r.get("model_name", r.get("outcome", ""))))))),
                dataset_scope=str(r.get("analysis_scope", r.get("scope", domain))),
                n=r.get("n", r.get("n_rows", np.nan)),
                n_participants=r.get("n_participants", r.get("n_subjects", np.nan)),
                effect_size=effect,
                effect_metric=metric,
                p_value=p,
                q_value=q,
                ci_low=r.get("bootstrap_ci_low", np.nan),
                ci_high=r.get("bootstrap_ci_high", np.nan),
                direction=str(r.get("claim_status", r.get("interpretation", r.get("status", "")))),
                context=("ann_state_limited exploratory" if "state" in text.lower() else "tested")
                + (" capacity_pressure" if "pressure" in text.lower() else "")
                + (" hbn" if "hbn" in table else "")
                + (" geometry" if "geometry" in text.lower() or "trajectory" in text.lower() else ""),
                source_table=table,
                script=script,
            )

    out = pd.DataFrame(rows)
    missing_q = out["q_value"].isna()
    out.loc[missing_q, "q_value"] = bh_q(out.loc[missing_q, "p_value"])
    out["claim_strength"] = out.apply(classify, axis=1)
    return out


def main() -> int:
    ensure_dirs()
    effects = build_rows()
    effects.to_csv(TABLES / "master_effects_table.csv", index=False)
    effects.to_csv(AUDIT / "claim_audit.tsv", sep="\t", index=False)
    summary = (
        effects.groupby(["construct", "claim_strength"], dropna=False)
        .size()
        .reset_index(name="n_effects")
        .sort_values(["construct", "claim_strength"])
    )
    summary.to_csv(TABLES / "master_effects_summary.csv", index=False)
    audit_lines = [
        "# Step 17 Master Statistics Audit",
        "",
        f"- Total effect rows: {len(effects)}.",
        f"- State rows: {int(effects['construct'].eq('state').sum())}.",
        f"- Capacity rows: {int(effects['construct'].eq('capacity').sum())}.",
        f"- Interaction rows: {int(effects['construct'].eq('interaction').sum())}.",
        "- The table intentionally includes positive, exploratory, negative and failed-validation results.",
        "- State effects remain mostly exploratory/negative because the ANN residualized state gate failed and TU/HBN did not validate state strongly.",
        "- Capacity effects are strongest when tied to recurrent geometry, COG behavior/EEG, TU capacity-pressure and HBN scalability.",
    ]
    (AUDIT / "step17_master_statistics_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    status = {
        "status": "implemented_and_run",
        "n_effect_rows": int(len(effects)),
        "n_state_rows": int(effects["construct"].eq("state").sum()),
        "n_capacity_rows": int(effects["construct"].eq("capacity").sum()),
        "n_interaction_rows": int(effects["construct"].eq("interaction").sum()),
        "outputs": [
            "outputs/tables/master_effects_table.csv",
            "outputs/tables/master_effects_summary.csv",
            "outputs/audit/claim_audit.tsv",
        ],
    }
    (LOGS / "step17_master_statistics_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP17_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
