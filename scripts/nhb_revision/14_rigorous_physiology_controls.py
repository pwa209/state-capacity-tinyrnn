from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB_TABLES, TABLES, append_manifest, append_registry, bh_q, ensure_nhb_dirs


ANALYSIS_ID = "nhb_14_rigorous_physiology_controls"
SCRIPT_NAME = "scripts/nhb_revision/14_rigorous_physiology_controls.py"
RNG_SEED = 20260611
N_PERMUTATIONS = 200


PREDICTOR_CANDIDATES = [
    "state_multidimensional_summary_z",
    "state_parameter_instability_z",
    "optimized_state_profile_z",
    "capacity_multidimensional_summary_z",
    "capacity_parameter_resource_z",
    "optimized_capacity_profile_z",
]


FEATURE_PREFIXES = (
    "eeg_",
    "cog_eeg_",
    "fnirs_",
    "nirs_",
    "ecg_",
    "biodex_",
)

FEATURE_KEYWORDS = (
    "theta_alpha_ratio",
    "relative_theta",
    "relative_alpha",
    "spectral_entropy",
    "aperiodic_slope",
    "channel_variability",
    "frontal_theta_power",
    "hbo_task_response",
    "hbr_task_response",
    "hbo_hbr_difference_mean",
    "ecg_hr_mean",
    "ecg_rmssd",
    "ecg_sdnn",
)


def read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def z_residualize_by_blocks(values: pd.Series, blocks: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=y.index, dtype=float)
    block_cols = [c for c in blocks.columns if c in blocks and blocks[c].notna().any()]
    if not block_cols:
        return y
    keys = blocks[block_cols].astype(str).agg("::".join, axis=1)
    for _, idx in keys.groupby(keys).groups.items():
        idx = list(idx)
        vals = y.loc[idx]
        if vals.notna().sum() == 0:
            continue
        out.loc[idx] = vals - vals.mean(skipna=True)
    return out


def permutation_p(x: pd.Series, y: pd.Series, rng: np.random.Generator, groups: pd.Series | None = None) -> tuple[float, float, float]:
    data = pd.DataFrame({"x": x, "y": y, "group": groups if groups is not None else "all"}).dropna()
    if len(data) < 20 or data["x"].nunique() < 3 or data["y"].nunique() < 3:
        return np.nan, np.nan, np.nan
    obs = float(stats.spearmanr(data["x"], data["y"]).statistic)
    null = np.zeros(N_PERMUTATIONS, dtype=float)
    yv = data["y"].to_numpy()
    xv = data["x"].to_numpy()
    group_indices = [np.asarray(idx, dtype=int) for idx in data.groupby("group").indices.values()]
    for i in range(N_PERMUTATIONS):
        shuffled = xv.copy()
        for idx in group_indices:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[i] = abs(float(stats.spearmanr(shuffled, yv).statistic))
    p = float((np.sum(null >= abs(obs)) + 1) / (len(null) + 1))
    return obs, p, float(np.quantile(null, 0.95))


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in FEATURE_PREFIXES):
            continue
        if not any(keyword in col for keyword in FEATURE_KEYWORDS):
            continue
        if col.endswith("_z") or "coordinate" in col or "claim_status" in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def modality_for(feature: str) -> str:
    if "fnirs" in feature or feature.startswith("nirs_"):
        return "fnirs"
    if feature.startswith("ecg_"):
        return "ecg"
    if feature.startswith("biodex_"):
        return "biodex"
    return "eeg"


def dataset_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    ds = read_csv("ds007554_neurophys_features.csv")
    if not ds.empty:
        frames["ds007554"] = ds
    tu = read_csv("tu_berlin_eeg_nirs_features.csv")
    if not tu.empty:
        frames["tu_berlin_eeg_nirs"] = tu
    cog = read_csv("cog_bci_coordinates.csv")
    if not cog.empty:
        frames["cog_bci"] = cog
    return frames


def run_dataset(dataset: str, df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(RNG_SEED + sum(ord(ch) for ch in dataset))
    predictors = [c for c in PREDICTOR_CANDIDATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    features = feature_columns(df)
    rows: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    if not predictors or not features:
        return rows, controls
    blocks = pd.DataFrame({c: df[c] for c in ["task", "session", "load_level"] if c in df.columns})
    block_cols = [c for c in ["task", "session", "load_level"] if c in df.columns]
    permutation_group = df[block_cols].astype(str).agg("::".join, axis=1) if block_cols else None
    for predictor in predictors:
        observed_abs: list[float] = []
        null95: list[float] = []
        for feature in features:
            x = z_residualize_by_blocks(df[predictor], blocks)
            y = z_residualize_by_blocks(df[feature], blocks)
            data = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(data) < 20 or data["x"].nunique() < 3 or data["y"].nunique() < 3:
                continue
            obs, p_perm, n95 = permutation_p(x, y, rng, groups=permutation_group)
            rho, p_nominal = stats.spearmanr(data["x"], data["y"])
            observed_abs.append(abs(float(rho)))
            if np.isfinite(n95):
                null95.append(n95)
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "script_name": SCRIPT_NAME,
                    "dataset": dataset,
                    "modality": modality_for(feature),
                    "feature": feature,
                    "predictor": predictor,
                    "n_rows": len(data),
                    "n_subjects": df["subject"].nunique() if "subject" in df.columns else np.nan,
                    "estimate": float(rho),
                    "p_value": float(p_nominal),
                    "permutation_p_value": p_perm,
                    "permutation_abs95": n95,
                    "control_status": "task_session_residualized_subject_permutation",
                    "claim_strength": "exploratory" if np.isfinite(p_perm) and p_perm < 0.05 else "negative",
                    "interpretation": "Physiology alignment screen; not direct neural-coordinate evidence.",
                    "source_table": "physiology_robustness_models.csv",
                }
            )
        controls.append(
            {
                "analysis_id": ANALYSIS_ID,
                "dataset": dataset,
                "predictor": predictor,
                "control": "subject_label_permutation_with_task_session_blocks",
                "n_feature_tests": len(observed_abs),
                "max_abs_observed_rho": max(observed_abs) if observed_abs else np.nan,
                "median_null_abs95": float(np.nanmedian(null95)) if null95 else np.nan,
                "status": "computed",
                "claim_strength": "bounded_exploratory",
            }
        )
    return rows, controls


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    all_rows: list[dict[str, object]] = []
    all_controls: list[dict[str, object]] = []
    for dataset, df in dataset_frames().items():
        rows, controls = run_dataset(dataset, df)
        all_rows.extend(rows)
        all_controls.extend(controls)
    models = pd.DataFrame(all_rows)
    if not models.empty:
        models["q_value"] = bh_q(models["p_value"])
        models["permutation_q_value"] = bh_q(models["permutation_p_value"])
        models["claim_strength"] = np.where(models["permutation_q_value"] < 0.05, "exploratory_permutation_fdr", models["claim_strength"])
        models = models.sort_values(["permutation_p_value", "p_value", "dataset", "predictor", "feature"])
    controls = pd.DataFrame(all_controls)
    outputs = []
    model_path = NHB_TABLES / "physiology_robustness_models.csv"
    models.to_csv(model_path, index=False)
    outputs.append(model_path)
    control_path = NHB_TABLES / "physiology_permutation_controls.csv"
    controls.to_csv(control_path, index=False)
    outputs.append(control_path)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, outputs, notes=f"Computed row-level physiology controls for {len(models)} feature-coordinate tests.")
    print(f"Wrote {len(models)} physiology tests and {len(controls)} controls")


if __name__ == "__main__":
    main()
