from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "07_train_tinyrnn"
for path in [SCRIPT_DIR, TRAIN_SCRIPT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_all import prepare_training_events, robust_slope  # type: ignore
from nhb_utils import NHB_TABLES, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_03_state_reliability"
SCRIPT_NAME = "scripts/nhb_revision/03_state_reliability.py"
RNG_SEED = 20260611
MIN_TRIALS = 20
N_BOOT = 10
MAX_BOOTSTRAP_GROUPS_PER_DATASET_TASK_BIN = 4
PROFILE_COLUMNS = ["error_rate", "rt_cv", "rt_iqr", "time_accuracy_slope", "error_transition_rate", "log_rt_variance"]


def state_profile(df: pd.DataFrame) -> np.ndarray:
    correct = pd.to_numeric(df["correct_numeric"], errors="coerce").to_numpy(float)
    rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
    rt = rt[np.isfinite(rt)]
    errors = 1.0 - correct[np.isfinite(correct)]
    error_rate = float(np.nanmean(errors)) if len(errors) else np.nan
    rt_cv = float(np.nanstd(rt) / max(np.nanmean(rt), 1e-6)) if len(rt) > 2 else np.nan
    rt_iqr = float(np.nanquantile(rt, 0.75) - np.nanquantile(rt, 0.25)) if len(rt) > 2 else np.nan
    slope = robust_slope(pd.to_numeric(df["trial_index"], errors="coerce").to_numpy(float), correct)
    if len(errors) > 2:
        trans = float(np.mean((errors[:-1] == 1) & (errors[1:] == 1)))
    else:
        trans = np.nan
    log_rt_var = float(np.nanvar(np.log(np.clip(rt, 1e-6, None)))) if len(rt) > 2 else np.nan
    return np.asarray([error_rate, rt_cv, rt_iqr, slope, trans, log_rt_var], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return np.nan
    return float(np.dot(aa, bb) / denom)


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(a[mask], b[mask]).statistic)
    return float(spearmanr(a[mask], b[mask]).statistic)


def icc_two_profiles(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    data = np.vstack([a[mask], b[mask]]).T
    n, k = data.shape
    mean_targets = data.mean(axis=1)
    mean_raters = data.mean(axis=0)
    grand = data.mean()
    ss_between = k * np.sum((mean_targets - grand) ** 2)
    ss_rater = n * np.sum((mean_raters - grand) ** 2)
    ss_total = np.sum((data - grand) ** 2)
    ss_error = ss_total - ss_between - ss_rater
    ms_between = ss_between / max(n - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    denom = ms_between + (k - 1) * ms_error
    return float((ms_between - ms_error) / denom) if denom != 0 else np.nan


def compare_profiles(a_df: pd.DataFrame, b_df: pd.DataFrame) -> dict[str, float]:
    a = state_profile(a_df)
    b = state_profile(b_df)
    return {
        "pearson_r": safe_corr(a, b, "pearson"),
        "spearman_rho": safe_corr(a, b, "spearman"),
        "cosine_similarity": cosine(a, b),
        "icc": icc_two_profiles(a, b),
    }


def split_group(group: pd.DataFrame, split_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values("trial_index")
    if split_type == "odd_even":
        return group[group["trial_index"] % 2 == 1], group[group["trial_index"] % 2 == 0]
    halfway = len(group) // 2
    return group.iloc[:halfway], group.iloc[halfway:]


def trial_bin(n: int) -> str:
    if n < 30:
        return "20_29"
    if n < 50:
        return "30_49"
    if n < 100:
        return "50_99"
    if n < 200:
        return "100_199"
    return "200_plus"


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    trainable, _ = prepare_training_events()
    rows = []
    boot_rows = []
    boot_counts: dict[tuple[str, str, str, str], int] = {}
    rng = np.random.default_rng(RNG_SEED)
    group_cols = ["dataset", "subject", "session_id", "task"]
    for (dataset, subject, session, task), group in trainable.groupby(group_cols, dropna=False):
        group = group.sort_values("trial_index").copy()
        if len(group) < MIN_TRIALS:
            continue
        participant_id = f"{dataset}:{subject}"
        for split_type in ["odd_even", "first_second"]:
            a, b = split_group(group, split_type)
            if len(a) < 5 or len(b) < 5:
                continue
            sims = compare_profiles(a, b)
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "script_name": SCRIPT_NAME,
                    "dataset": dataset,
                    "task": task,
                    "subject_id": subject,
                    "participant_id": participant_id,
                    "session": session,
                    "split": split_type,
                    "model_family": "not_model_specific",
                    "hidden_size": "",
                    "state_definition": "six_component_behavioral_reliability_profile",
                    "capacity_definition": "",
                    "outcome": "state_profile_similarity",
                    "predictor": split_type,
                    "n_rows": len(group),
                    "n_subjects": 1,
                    "estimate": sims["cosine_similarity"],
                    "std_error": "",
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": "",
                    "q_value": "",
                    "effect_direction": "higher_is_more_reliable",
                    "control_status": "split_half_reliability",
                    "claim_strength": "exploratory",
                    "interpretation": "Independent split-half agreement of state-like behavioural reliability profile.",
                    "source_table": "state_split_half_reliability.csv",
                    "split_type": split_type,
                    "n_trials": len(group),
                    "trial_count_bin": trial_bin(len(group)),
                    **sims,
                }
            )
            boot_vals = []
            boot_key = (str(dataset), str(task), split_type, trial_bin(len(group)))
            boot_count = boot_counts.get(boot_key, 0)
            if len(group) >= 2 * MIN_TRIALS and boot_count < MAX_BOOTSTRAP_GROUPS_PER_DATASET_TASK_BIN:
                idx = np.arange(len(group))
                for _ in range(N_BOOT):
                    draw_a = rng.choice(idx, size=len(group) // 2, replace=True)
                    draw_b = rng.choice(idx, size=len(group) // 2, replace=True)
                    sim = compare_profiles(group.iloc[draw_a], group.iloc[draw_b])
                    boot_vals.append(sim["cosine_similarity"])
                boot_counts[boot_key] = boot_count + 1
            if boot_vals:
                boot_rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "script_name": SCRIPT_NAME,
                        "dataset": dataset,
                        "task": task,
                        "subject_id": subject,
                        "participant_id": participant_id,
                        "session": session,
                        "split_type": split_type,
                        "n_trials": len(group),
                        "trial_count_bin": trial_bin(len(group)),
                        "pearson_r": sims["pearson_r"],
                        "spearman_rho": sims["spearman_rho"],
                        "cosine_similarity": sims["cosine_similarity"],
                        "icc": sims["icc"],
                        "bootstrap_ci_low": float(np.nanpercentile(boot_vals, 2.5)),
                        "bootstrap_ci_high": float(np.nanpercentile(boot_vals, 97.5)),
                        "bootstrap_median": float(np.nanmedian(boot_vals)),
                        "source_table": "state_bootstrap_reliability.csv",
                    }
                )
    rel = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    rel_path = NHB_TABLES / "state_split_half_reliability.csv"
    boot_path = NHB_TABLES / "state_bootstrap_reliability.csv"
    rel.to_csv(rel_path, index=False)
    boot.to_csv(boot_path, index=False)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, [rel_path, boot_path])
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        [rel_path, boot_path],
        notes=f"Computed split-half reliability for {len(rel)} session-task split rows.",
    )
    print(f"Wrote {rel_path}")
    print(f"Wrote {boot_path}")


if __name__ == "__main__":
    main()
