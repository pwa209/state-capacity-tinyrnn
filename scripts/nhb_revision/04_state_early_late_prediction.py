from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "07_train_tinyrnn"
for path in [SCRIPT_DIR, TRAIN_SCRIPT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_all import prepare_training_events, robust_slope  # type: ignore
from nhb_utils import NHB_TABLES, append_manifest, append_registry, bh_q, ensure_nhb_dirs


ANALYSIS_ID = "nhb_04_state_early_late_prediction"
SCRIPT_NAME = "scripts/nhb_revision/04_state_early_late_prediction.py"
FRACTIONS = [0.30, 0.40, 0.50]
OUTCOMES = ["accuracy", "lapse", "rt_cv", "rt_iqr", "mean_rt", "log_rt_variance"]
RNG_SEED = 20260611


def summarize_window(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    correct = pd.to_numeric(df["correct_numeric"], errors="coerce").to_numpy(float)
    rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
    rt = rt[np.isfinite(rt)]
    acc = float(np.nanmean(correct)) if len(correct) else np.nan
    lapse = 1.0 - acc if np.isfinite(acc) else np.nan
    rt_cv = float(np.nanstd(rt) / max(np.nanmean(rt), 1e-6)) if len(rt) > 2 else np.nan
    rt_iqr = float(np.nanquantile(rt, 0.75) - np.nanquantile(rt, 0.25)) if len(rt) > 2 else np.nan
    mean_rt = float(np.nanmean(rt)) if len(rt) else np.nan
    log_rt_variance = float(np.nanvar(np.log(np.clip(rt, 1e-6, None)))) if len(rt) > 2 else np.nan
    slope = robust_slope(pd.to_numeric(df["trial_index"], errors="coerce").to_numpy(float), correct)
    errors = 1.0 - correct[np.isfinite(correct)]
    error_transition = float(np.mean((errors[:-1] == 1) & (errors[1:] == 1))) if len(errors) > 2 else np.nan
    reliability = acc - (0 if not np.isfinite(error_transition) else error_transition)
    variability = (0 if not np.isfinite(rt_cv) else rt_cv) + np.sqrt(max(acc * (1 - acc), 0)) if np.isfinite(acc) else np.nan
    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_lapse": lapse,
        f"{prefix}_rt_cv": rt_cv,
        f"{prefix}_rt_iqr": rt_iqr,
        f"{prefix}_mean_rt": mean_rt,
        f"{prefix}_log_rt_variance": log_rt_variance,
        f"{prefix}_state_lapse_axis": lapse,
        f"{prefix}_state_drift_axis": -slope if np.isfinite(slope) else np.nan,
        f"{prefix}_state_variability_axis": variability,
        f"{prefix}_state_reliability_axis": reliability,
        f"{prefix}_n_trials": len(df),
    }


def make_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, subject, session, task), group in events.groupby(["dataset", "subject", "session_id", "task"], dropna=False):
        group = group.sort_values("trial_index").copy()
        if len(group) < 30:
            continue
        for frac in FRACTIONS:
            cut = max(5, int(np.floor(len(group) * frac)))
            early = group.iloc[:cut]
            late = group.iloc[cut:]
            if len(late) < 5:
                continue
            early_summary = summarize_window(early, "early")
            late_summary = summarize_window(late, "late")
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "script_name": SCRIPT_NAME,
                    "dataset": dataset,
                    "task": task,
                    "subject_id": subject,
                    "participant_id": f"{dataset}:{subject}",
                    "session": session,
                    "split": f"first_{int(frac*100)}pct_to_late",
                    "fraction": frac,
                    **early_summary,
                    **late_summary,
                }
            )
    return pd.DataFrame(rows)


def model_spec(model_name: str) -> tuple[list[str], list[str]]:
    controls_cat = ["task"]
    control_num = ["early_n_trials"]
    recent = ["early_accuracy", "early_lapse", "early_rt_cv", "early_rt_iqr", "early_mean_rt", "early_log_rt_variance"]
    state = ["early_state_lapse_axis", "early_state_drift_axis", "early_state_variability_axis", "early_state_reliability_axis"]
    if model_name == "baseline_controls":
        return control_num, controls_cat
    if model_name == "recent_behavior_only":
        return recent, []
    if model_name == "state_only":
        return state, []
    if model_name == "state_plus_controls":
        return state + control_num, controls_cat
    if model_name == "state_plus_recent_behavior":
        return state + recent, []
    if model_name == "shuffled_state_control":
        return [f"shuffled_{c}" for c in state], []
    raise ValueError(model_name)


def fit_predict(df: pd.DataFrame, outcome: str, model_name: str) -> np.ndarray:
    data = df.copy()
    state_cols = ["early_state_lapse_axis", "early_state_drift_axis", "early_state_variability_axis", "early_state_reliability_axis"]
    rng = np.random.default_rng(RNG_SEED + len(outcome) + len(model_name))
    for col in state_cols:
        data[f"shuffled_{col}"] = rng.permutation(data[col].to_numpy())
    num_cols, cat_cols = model_spec(model_name)
    features = num_cols + cat_cols
    y = data[outcome].to_numpy(float)
    groups = data["participant_id"].to_numpy()
    pred = np.full(len(data), np.nan)
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    for train_idx, test_idx in splitter.split(data[features], y, groups):
        pipe.fit(data.iloc[train_idx][features], y[train_idx])
        pred[test_idx] = pipe.predict(data.iloc[test_idx][features])
    return pred


def compare_models(rows: pd.DataFrame) -> pd.DataFrame:
    model_names = [
        "baseline_controls",
        "recent_behavior_only",
        "state_only",
        "state_plus_controls",
        "state_plus_recent_behavior",
        "shuffled_state_control",
    ]
    out_rows = []
    for (dataset, fraction), scope in rows.groupby(["dataset", "fraction"], dropna=False):
        for outcome in OUTCOMES:
            target = f"late_{outcome}"
            data = scope.dropna(subset=[target]).copy()
            if len(data) < 25 or data[target].nunique() < 3:
                continue
            preds = {}
            for model_name in model_names:
                try:
                    preds[model_name] = fit_predict(data, target, model_name)
                except Exception:
                    continue
            baseline_pred = preds.get("baseline_controls")
            recent_pred = preds.get("recent_behavior_only")
            for model_name, pred in preds.items():
                valid = np.isfinite(pred) & np.isfinite(data[target].to_numpy(float))
                y = data[target].to_numpy(float)[valid]
                p = pred[valid]
                rmse = float(np.sqrt(mean_squared_error(y, p)))
                mae = float(mean_absolute_error(y, p))
                r2 = float(r2_score(y, p))
                delta_r2 = np.nan
                p_value = np.nan
                if baseline_pred is not None:
                    b = baseline_pred[valid]
                    delta_r2 = r2 - float(r2_score(y, b))
                if recent_pred is not None and model_name in ["state_plus_recent_behavior", "state_only"]:
                    r = recent_pred[valid]
                    if len(y) > 2:
                        p_value = float(ttest_rel((y - r) ** 2, (y - p) ** 2, nan_policy="omit").pvalue)
                out_rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "script_name": SCRIPT_NAME,
                        "dataset": dataset,
                        "task": "all",
                        "subject_id": "",
                        "participant_id": "",
                        "session": "",
                        "split": f"first_{int(fraction*100)}pct_to_late",
                        "model_family": "ridge_cv",
                        "hidden_size": "",
                        "state_definition": "early_window_behavioral_reliability_profile",
                        "capacity_definition": "",
                        "outcome": outcome,
                        "predictor": model_name,
                        "n_rows": int(len(y)),
                        "n_subjects": int(data["participant_id"].nunique()),
                        "estimate": delta_r2,
                        "std_error": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_value": p_value,
                        "q_value": "",
                        "effect_direction": "higher_delta_r2_is_better",
                        "control_status": "early_late_prediction",
                        "claim_strength": "exploratory",
                        "interpretation": "Cross-validated prediction of late-window behaviour from early-window state and behaviour summaries.",
                        "source_table": "state_early_late_model_comparison.csv",
                        "fraction": fraction,
                        "model_name": model_name,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2,
                        "delta_R2": delta_r2,
                    }
                )
    result = pd.DataFrame(out_rows)
    if not result.empty:
        result["q_value"] = bh_q(result["p_value"])
        result["claim_strength"] = np.where(
            (result["predictor"].eq("state_plus_recent_behavior")) & (pd.to_numeric(result["q_value"], errors="coerce") < 0.05) & (result["delta_R2"] > 0),
            "strong",
            np.where((pd.to_numeric(result["p_value"], errors="coerce") < 0.05) & (result["delta_R2"] > 0), "exploratory", "negative"),
        )
    return result


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    trainable, _ = prepare_training_events()
    rows = make_rows(trainable)
    comparison = compare_models(rows)
    pred_path = NHB_TABLES / "state_early_late_prediction.csv"
    comp_path = NHB_TABLES / "state_early_late_model_comparison.csv"
    rows.to_csv(pred_path, index=False)
    comparison.to_csv(comp_path, index=False)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, [pred_path, comp_path])
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, [pred_path, comp_path], notes=f"Early-late rows={len(rows)}, model rows={len(comparison)}.")
    print(f"Wrote {pred_path}")
    print(f"Wrote {comp_path}")


if __name__ == "__main__":
    main()
