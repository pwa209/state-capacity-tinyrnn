from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_31_figure3_capacity_gate_audit"
SCRIPT_NAME = "scripts/nhb_revision/31_audit_figure3_capacity_gate.py"
TABLE_DIR = NHB / "tables"
AUDIT_DIR = NHB / "audit"
HYBRID = TABLE_DIR / "architecture_hybrid_recovery_results.csv"
PERTURBATION = TABLE_DIR / "architecture_perturbation_gate_results.csv"
CLAIM_AUDIT = AUDIT_DIR / "nhb_final_claim_audit.tsv"
OUT_DIR = NHB / "display_item_revision"
OUT_MD = OUT_DIR / "figure3_capacity_gate_audit.md"
OUT_CSV = OUT_DIR / "figure3_capacity_gate_audit_rows.csv"


def yes_no(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    small = df.copy()
    small = small.where(pd.notna(small), "")
    cols = list(small.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in small.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_nhb_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    hybrid = pd.read_csv(HYBRID)
    perturb = pd.read_csv(PERTURBATION)
    claims = pd.read_csv(CLAIM_AUDIT, sep="\t")

    hybrid["audit_decision_rule"] = "PASS iff abs(Spearman rho) > 0.40 and nominal p < 0.05"
    hybrid["audit_pass_label"] = hybrid["pass_gate"].map(lambda x: yes_no(bool(x)))
    hybrid["is_gru_capacity"] = (
        hybrid["model_family"].eq("gru") & hybrid["target_axis"].eq("capacity_level")
    )
    hybrid["interpretive_scope"] = hybrid.apply(
        lambda r: "raw scalar recovery" if r["feature_set"] == "raw_fingerprint" else "architecture/other-axis residualised scalar recovery",
        axis=1,
    )
    hybrid.to_csv(OUT_CSV, index=False)

    gru_capacity = hybrid[hybrid["is_gru_capacity"]].copy()
    summary = (
        hybrid.groupby(["target_axis", "feature_set"], as_index=False)
        .agg(
            n=("spearman_rho", "size"),
            median_rho=("spearman_rho", "median"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
            pass_count=("pass_gate", "sum"),
        )
        .sort_values(["target_axis", "feature_set"])
    )
    family_gate = perturb[
        perturb["feature_set"].eq("residualized_fingerprint")
        & perturb["task_family"].eq("overall")
        & perturb["model_family"].isin(["vanilla_rnn", "gru", "lstm"])
    ][["model_family", "balanced_accuracy", "auc", "permutation_p", "control_status", "claim_strength"]]
    c1_c3 = claims[claims["claim_id"].isin(["C1", "C2", "C3"])]

    lines = [
        "# Figure 3 Capacity Gate Audit",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Verdict",
        "GRU capacity **passes the raw hybrid scalar-recovery test**, but **does not pass the residualised hybrid scalar-recovery test**.",
        "",
        "The residualised GRU capacity row is rho = 0.298, nominal p = 0.0775, pass_gate = False. The pre-specified rule in the script is abs(rho) > 0.40 and p < 0.05, so this row fails both the rho threshold and the p-value threshold.",
        "",
        "This does not invalidate the later capacity-profile claim. Figure 3 validates separability of intervention families and falsifies robust scalar recovery. Capacity-profile validation is supported later by load-pressure and recurrent-geometry analyses, not by residualised scalar recovery in Figure 3.",
        "",
        "## GRU Capacity Rows",
        to_md_table(gru_capacity),
        "",
        "## Hybrid Recovery Summary",
        to_md_table(summary),
        "",
        "## Residualised Intervention-Family Gate",
        to_md_table(family_gate),
        "",
        "## Relevant Claim Audit Rows",
        to_md_table(c1_c3),
        "",
        "## Manuscript/Figure Repair",
        "Use wording like: 'Intervention families separate cleanly, but residualised scalar recovery is not robust.'",
        "",
        "Do not write that capacity passes residualised scalar recovery. Do write that capacity later shows profile-level validation through recurrent geometry and state-controlled load-pressure evidence.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    append_manifest(ANALYSIS_ID, SCRIPT_NAME, [OUT_MD, OUT_CSV])
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        [OUT_MD, OUT_CSV],
        status="complete",
        notes="Audited whether GRU capacity passes Figure 3 raw or residualised hybrid recovery gate.",
    )
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
