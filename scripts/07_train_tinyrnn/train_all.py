from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
AUDIT = PROJECT_ROOT / "outputs" / "audit"
CHECKPOINTS = PROJECT_ROOT / "outputs" / "model_checkpoints" / "human_models"
TRAIN_LOGS = PROJECT_ROOT / "outputs" / "logs" / "training_logs"

HIDDEN_SIZES = [1, 2, 3, 4, 6, 8]
SPLIT_STRATEGIES = ["participant_level", "session_blocked", "odd_even_miniblock"]
SEQ_LEN = 64
EPOCHS = 24
BATCH_SIZE = 64
LR = 3e-3
RNG_SEED = 20260529


@dataclass
class SplitData:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame


class HumanGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.out(hidden).squeeze(-1)


def ensure_dirs() -> None:
    for path in [TABLES, AUDIT, CHECKPOINTS, TRAIN_LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def nll_from_prob(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return -(y * np.log(np.clip(prob, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - prob, eps, 1 - eps)))


def prepare_training_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED / "all_model_events.parquet"
    if not path.exists():
        raise FileNotFoundError("Step 08 requires Step 03 output data/processed/all_model_events.parquet")
    events = pd.read_parquet(path)
    events = events[events["event_included"].astype(bool)].copy()
    events["has_supervised_target"] = events["correct"].notna()

    exclusion = (
        events.groupby(["dataset", "task"], dropna=False)
        .agg(
            n_events=("dataset", "size"),
            n_supervised_events=("has_supervised_target", "sum"),
            n_subjects=("subject", "nunique"),
        )
        .reset_index()
    )
    exclusion["included_in_step08_training"] = exclusion["n_supervised_events"] > 0
    exclusion["exclusion_reason"] = np.where(
        exclusion["included_in_step08_training"],
        "",
        "no_correct_or_response_target_in_unified_events",
    )

    trainable = events[events["has_supervised_target"]].copy()
    trainable["correct_numeric"] = trainable["correct"].astype("boolean").astype(float)
    trainable["participant_id"] = trainable["dataset"].astype(str) + ":" + trainable["subject"].astype(str)
    trainable["session_id"] = trainable["session"].fillna("no_session").astype(str)
    trainable["block_id"] = trainable["block"].fillna("no_block").astype(str)
    trainable["task"] = trainable["task"].astype(str)
    trainable["condition"] = trainable["condition"].fillna("missing").astype(str)
    trainable["load_level"] = pd.to_numeric(trainable["load_level"], errors="coerce").fillna(0.0)
    trainable["trial_index"] = pd.to_numeric(trainable["trial_index"], errors="coerce").fillna(0).astype(int)
    trainable["time_on_task"] = pd.to_numeric(trainable["time_on_task"], errors="coerce").fillna(0.0)
    trainable["rt"] = pd.to_numeric(trainable["rt"], errors="coerce")
    trainable = trainable.sort_values(["dataset", "subject", "session_id", "task", "timestamp", "trial_index"]).reset_index(drop=True)

    return trainable, exclusion


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    group_cols = ["dataset", "subject", "session_id", "task"]
    data["trial_norm"] = data.groupby(group_cols, dropna=False)["trial_index"].transform(
        lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1.0)
    )
    data["time_norm"] = data.groupby(group_cols, dropna=False)["time_on_task"].transform(
        lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1.0)
    )
    data["load_norm"] = data["load_level"] / max(float(data["load_level"].max()), 1.0)
    data["is_tu"] = (data["dataset"] == "tu_berlin_eeg_nirs").astype(float)
    data["is_cog"] = (data["dataset"] == "cog_bci").astype(float)

    categorical = pd.get_dummies(
        data[["dataset", "task", "condition"]],
        columns=["dataset", "task", "condition"],
        dtype=float,
    )
    numeric_cols = ["trial_norm", "time_norm", "load_norm", "is_tu", "is_cog"]
    data = pd.concat([data, categorical], axis=1)
    feature_cols = numeric_cols + list(categorical.columns)
    data[feature_cols] = data[feature_cols].astype(float)
    return data, feature_cols


def make_splits(df: pd.DataFrame) -> list[SplitData]:
    participants = sorted(df["participant_id"].unique())
    participant_val = set(participants[::5])
    participant_split = SplitData(
        "participant_level",
        df[~df["participant_id"].isin(participant_val)].copy(),
        df[df["participant_id"].isin(participant_val)].copy(),
    )

    session_val_mask = pd.Series(False, index=df.index)
    for _, group in df.groupby("participant_id"):
        sessions = sorted(group["session_id"].unique())
        val_session = sessions[-1]
        session_val_mask.loc[group[group["session_id"] == val_session].index] = True
    session_split = SplitData(
        "session_blocked",
        df[~session_val_mask].copy(),
        df[session_val_mask].copy(),
    )

    odd_val = df["trial_index"] % 2 == 1
    odd_even_split = SplitData(
        "odd_even_miniblock",
        df[~odd_val].copy(),
        df[odd_val].copy(),
    )
    return [participant_split, session_split, odd_even_split]


def build_sequences(df: pd.DataFrame, feature_cols: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    group_cols = ["dataset", "subject", "session_id", "task"]
    for group_key, group in df.groupby(group_cols, dropna=False, sort=True):
        group = group.sort_values(["timestamp", "trial_index"]).reset_index(drop=True)
        for start in range(0, len(group), SEQ_LEN):
            chunk = group.iloc[start : start + SEQ_LEN]
            x = np.zeros((SEQ_LEN, len(feature_cols)), dtype=np.float32)
            y = np.zeros(SEQ_LEN, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            n = len(chunk)
            x[:n] = chunk[feature_cols].to_numpy(dtype=np.float32)
            y[:n] = chunk["correct_numeric"].to_numpy(dtype=np.float32)
            mask[:n] = 1.0
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            for offset, (_, row) in enumerate(chunk.iterrows()):
                meta_rows.append(
                    {
                        "sequence_index": len(xs) - 1,
                        "sequence_offset": offset,
                        "dataset": row["dataset"],
                        "subject": row["subject"],
                        "participant_id": row["participant_id"],
                        "session": row["session_id"],
                        "task": row["task"],
                        "trial_index": row["trial_index"],
                        "correct": row["correct_numeric"],
                        "rt": row["rt"],
                    }
                )

    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        torch.tensor(np.stack(masks), dtype=torch.float32),
        pd.DataFrame(meta_rows),
    )


def train_model(
    split_name: str,
    hidden_size: int,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    input_dim: int,
) -> tuple[HumanGRU, dict[str, Any], np.ndarray]:
    torch.manual_seed(RNG_SEED + hidden_size)
    model = HumanGRU(input_dim=input_dim, hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    x_train, y_train, mask_train = train_tensors
    x_val, y_val, mask_val = val_tensors

    pos = float((y_train * mask_train).sum())
    neg = float(((1 - y_train) * mask_train).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    rng = np.random.default_rng(RNG_SEED + hidden_size + len(split_name))
    history = []
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(x_train.shape[0])
        epoch_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            logits = model(x_train[idx])
            loss_raw = criterion(logits, y_train[idx])
            loss = (loss_raw * mask_train[idx]).sum() / mask_train[idx].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        history.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_prob = torch.sigmoid(val_logits).cpu().numpy()
        train_logits = model(x_train)
        train_prob = torch.sigmoid(train_logits).cpu().numpy()

    val_mask_np = mask_val.cpu().numpy().astype(bool)
    train_mask_np = mask_train.cpu().numpy().astype(bool)
    val_y_np = y_val.cpu().numpy()
    train_y_np = y_train.cpu().numpy()
    val_prob_flat = val_prob[val_mask_np]
    val_y_flat = val_y_np[val_mask_np]
    train_prob_flat = train_prob[train_mask_np]
    train_y_flat = train_y_np[train_mask_np]

    metrics = {
        "split_strategy": split_name,
        "hidden_size": hidden_size,
        "n_train_events": int(train_mask_np.sum()),
        "n_val_events": int(val_mask_np.sum()),
        "train_nll": float(nll_from_prob(train_prob_flat, train_y_flat).mean()),
        "val_nll": float(nll_from_prob(val_prob_flat, val_y_flat).mean()),
        "train_accuracy": float(((train_prob_flat >= 0.5) == train_y_flat).mean()),
        "val_accuracy": float(((val_prob_flat >= 0.5) == val_y_flat).mean()),
        "final_training_loss": float(history[-1]),
        "epochs": EPOCHS,
    }
    return model, metrics, val_prob


def flatten_predictions(val_meta: pd.DataFrame, val_prob: np.ndarray, hidden_size: int, split_name: str) -> pd.DataFrame:
    rows = []
    for _, row in val_meta.iterrows():
        prob = float(val_prob[int(row["sequence_index"]), int(row["sequence_offset"])])
        y = float(row["correct"])
        rows.append(
            {
                **row.to_dict(),
                "split_strategy": split_name,
                "hidden_size": hidden_size,
                "predicted_correct_probability": prob,
                "event_nll": float(nll_from_prob(np.array([prob]), np.array([y]))[0]),
                "predicted_correct": bool(prob >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, hidden, participant), group in predictions.groupby(["split_strategy", "hidden_size", "participant_id"]):
        y = group["correct"].to_numpy(dtype=float)
        prob = group["predicted_correct_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "split_strategy": split,
                "hidden_size": hidden,
                "participant_id": participant,
                "dataset": ",".join(sorted(group["dataset"].astype(str).unique())),
                "subject": ",".join(sorted(group["subject"].astype(str).unique())),
                "n_val_events": int(len(group)),
                "val_nll": float(nll_from_prob(prob, y).mean()),
                "val_accuracy": float(((prob >= 0.5) == y).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_model_selection(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = subject_metrics(predictions)
    primary = metrics[metrics["split_strategy"] == "odd_even_miniblock"].copy()
    selections = []
    for participant, group in primary.groupby("participant_id"):
        ranked = group.sort_values(["val_nll", "hidden_size"]).reset_index(drop=True)
        best = ranked.iloc[0].to_dict()
        margin = float(ranked.iloc[1]["val_nll"] - ranked.iloc[0]["val_nll"]) if len(ranked) > 1 else np.nan
        best.update(
            {
                "selected_hidden_size": int(best["hidden_size"]),
                "selection_rule": "lowest odd_even_miniblock validation NLL",
                "nll_margin_to_next_best": margin,
            }
        )
        selections.append(best)
    selection_df = pd.DataFrame(selections)
    return selection_df, metrics


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return values * 0.0
    return (values - values.mean()) / std


def build_capacity_coordinates(
    selection_df: pd.DataFrame,
    trainable: pd.DataFrame,
    subject_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    multidim_rows = []
    coord = np.log2(selection_df["selected_hidden_size"].astype(float))
    z = (coord - coord.mean()) / (coord.std(ddof=0) if coord.std(ddof=0) > 0 else 1.0)
    for (_, row), coordinate, zcoord in zip(selection_df.iterrows(), coord, z):
        participant_events = trainable[trainable["participant_id"] == row["participant_id"]]
        validation = subject_validation[
            (subject_validation["participant_id"] == row["participant_id"])
            & (subject_validation["split_strategy"] == "odd_even_miniblock")
        ].copy()
        validation["log2_hidden"] = np.log2(validation["hidden_size"].astype(float))
        if len(validation) >= 3:
            nll_complexity_slope = robust_slope(validation["log2_hidden"].to_numpy(float), -validation["val_nll"].to_numpy(float))
            low = validation[validation["hidden_size"].isin([1, 2, 3])]["val_nll"].min()
            high = validation[validation["hidden_size"].isin([6, 8])]["val_nll"].min()
            high_capacity_nll_advantage = float(low - high) if not (np.isnan(low) or np.isnan(high)) else np.nan
        else:
            nll_complexity_slope = np.nan
            high_capacity_nll_advantage = np.nan
        task_accuracy = participant_events.groupby("task")["correct_numeric"].mean()
        cross_task_consistency = float(1.0 - task_accuracy.std(ddof=0)) if len(task_accuracy) > 1 else np.nan
        nback = participant_events[participant_events["task"] == "nback"]
        if len(nback) and nback["load_level"].nunique() > 1:
            load_robustness = robust_slope(nback["load_level"].to_numpy(float), nback["correct_numeric"].to_numpy(float))
        else:
            load_robustness = np.nan
        selection_confidence = float(row["nll_margin_to_next_best"])
        rows.append(
            {
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "subject": row["subject"],
                "selected_hidden_size": int(row["selected_hidden_size"]),
                "capacity_coordinate_log2_hidden": float(coordinate),
                "capacity_coordinate_z": float(zcoord),
                "selection_val_nll": float(row["val_nll"]),
                "selection_val_accuracy": float(row["val_accuracy"]),
                "nll_margin_to_next_best": selection_confidence,
                "n_training_eligible_events": int(len(participant_events)),
                "n_sessions": int(participant_events["session_id"].nunique()),
                "n_tasks": int(participant_events["task"].nunique()),
                "ann_gate_context": "capacity_interpretable_state_exploratory_ann_gate_failed",
            }
        )
        multidim_rows.append(
            {
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "subject": row["subject"],
                "selected_hidden_size": int(row["selected_hidden_size"]),
                "capacity_hidden_size_axis": float(coordinate),
                "capacity_hidden_size_axis_z": float(zcoord),
                "capacity_selection_confidence": selection_confidence,
                "capacity_complexity_preference_axis": nll_complexity_slope,
                "capacity_high_capacity_nll_advantage": high_capacity_nll_advantage,
                "capacity_load_robustness_axis": load_robustness,
                "capacity_cross_task_consistency_axis": cross_task_consistency,
                "capacity_information_quality": "high" if selection_confidence > 0.02 else "low_margin",
                "capacity_claim_status": "multidimensional_capacity_proxy_ann_capacity_partial_support",
            }
        )
    multidim = pd.DataFrame(multidim_rows)
    axis_cols = [
        "capacity_hidden_size_axis_z",
        "capacity_selection_confidence",
        "capacity_complexity_preference_axis",
        "capacity_high_capacity_nll_advantage",
        "capacity_load_robustness_axis",
        "capacity_cross_task_consistency_axis",
    ]
    for col in axis_cols:
        multidim[f"{col}_z"] = zscore(multidim[col])
    multidim["capacity_multidimensional_summary_z"] = multidim[[f"{col}_z" for col in axis_cols]].mean(axis=1, skipna=True)
    return pd.DataFrame(rows), multidim


def logit(value: float) -> float:
    value = min(max(value, 1e-4), 1 - 1e-4)
    return float(math.log(value / (1 - value)))


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3 or np.nanstd(x[valid]) == 0:
        return np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    x_valid = (x_valid - x_valid.min()) / max(x_valid.max() - x_valid.min(), 1e-9)
    return float(np.polyfit(x_valid, y_valid, 1)[0])


def build_state_parameters(trainable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, subject, session, task), group in trainable.groupby(["dataset", "subject", "session_id", "task"], dropna=False):
        group = group.sort_values(["timestamp", "trial_index"]).copy()
        calibration = group[group["trial_index"] % 2 == 0].copy()
        heldout = group[group["trial_index"] % 2 == 1].copy()
        if len(calibration) == 0:
            continue
        correct = calibration["correct_numeric"].to_numpy(dtype=float)
        accuracy = float(np.mean(correct))
        error = 1.0 - accuracy
        n_errors = int(np.sum(correct == 0))
        n_correct = int(np.sum(correct == 1))
        rt = calibration["rt"].to_numpy(dtype=float)
        rt_valid = rt[~np.isnan(rt)]
        rt_median = float(np.median(rt_valid)) if len(rt_valid) else np.nan
        rt_iqr = float(np.quantile(rt_valid, 0.75) - np.quantile(rt_valid, 0.25)) if len(rt_valid) else np.nan
        rt_cv = float(np.std(rt_valid) / max(np.mean(rt_valid), 1e-6)) if len(rt_valid) > 1 else np.nan
        time = calibration["time_on_task"].to_numpy(dtype=float)
        time_slope = robust_slope(time, correct)
        trial = calibration["trial_index"].to_numpy(dtype=float)
        early_cut = np.nanquantile(trial, 0.33) if len(trial) else np.nan
        late_cut = np.nanquantile(trial, 0.67) if len(trial) else np.nan
        early = correct[trial <= early_cut] if not np.isnan(early_cut) else np.array([])
        late = correct[trial >= late_cut] if not np.isnan(late_cut) else np.array([])
        early_late_accuracy_delta = float(np.mean(late) - np.mean(early)) if len(early) and len(late) else np.nan
        errors = 1 - correct
        if len(errors) > 1:
            error_transition = float(np.mean((errors[:-1] == 1) & (errors[1:] == 1)))
            if np.std(errors[:-1]) > 0 and np.std(errors[1:]) > 0:
                error_autocorr = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
            else:
                error_autocorr = np.nan
        else:
            error_transition = np.nan
            error_autocorr = np.nan
        sigma_proxy = np.nanstd(correct) + (0 if np.isnan(rt_cv) else rt_cv)
        outcome_variance = float(np.var(correct))
        correctness_information_score = float(outcome_variance * np.log1p(len(calibration)))
        rt_information_score = float((0 if np.isnan(rt_cv) else min(rt_cv, 2.0)) * np.log1p(len(rt_valid)))
        state_information_score = correctness_information_score + 0.25 * rt_information_score
        ceiling_flag = bool(accuracy >= 0.95 and n_errors < 5)
        if ceiling_flag or state_information_score < 0.05:
            state_quality = "low_ceiling_or_low_variance"
        elif state_information_score < 0.20:
            state_quality = "medium"
        else:
            state_quality = "high"
        lapse_axis = logit(error)
        drift_axis = float(-(time_slope if not np.isnan(time_slope) else 0.0) - (early_late_accuracy_delta if not np.isnan(early_late_accuracy_delta) else 0.0))
        variability_axis = float((rt_cv if not np.isnan(rt_cv) else 0.0) + np.sqrt(max(outcome_variance, 0.0)))
        reliability_axis = float(accuracy - (error_transition if not np.isnan(error_transition) else 0.0))
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "session": session,
                "task": task,
                "n_calibration_events": int(len(calibration)),
                "n_heldout_events": int(len(heldout)),
                "n_calibration_errors": n_errors,
                "n_calibration_correct": n_correct,
                "calibration_rule": "even trial_index only",
                "heldout_rule": "odd trial_index; never used for state estimation",
                "heldout_used_for_state_estimation": False,
                "calibration_accuracy": accuracy,
                "calibration_error_rate": error,
                "rt_median": rt_median,
                "rt_iqr": rt_iqr,
                "rt_cv": rt_cv,
                "time_accuracy_slope": time_slope,
                "early_late_accuracy_delta": early_late_accuracy_delta,
                "error_transition_rate": error_transition,
                "error_lag1_autocorrelation": error_autocorr,
                "correctness_information_score": correctness_information_score,
                "rt_information_score": rt_information_score,
                "state_information_score": state_information_score,
                "accuracy_ceiling_flag": ceiling_flag,
                "state_estimation_quality": state_quality,
                "state_lapse_axis": lapse_axis,
                "state_drift_axis": drift_axis,
                "state_variability_axis": variability_axis,
                "state_reliability_axis": reliability_axis,
                "log_tau_proxy": float(np.log1p(max(0.0, -time_slope if not np.isnan(time_slope) else 0.0) + (rt_cv if not np.isnan(rt_cv) else 0.0))),
                "logit_lapse_proxy": lapse_axis,
                "log_sigma_proxy": float(np.log(max(sigma_proxy, 1e-4))),
                "negative_log_gain_proxy": float(-np.log(min(max(accuracy, 1e-4), 0.9999))),
                "state_claim_status": "exploratory_ann_gate_failed",
            }
        )
    return pd.DataFrame(rows)


def build_state_quality_report(state: pd.DataFrame) -> pd.DataFrame:
    quality = (
        state.groupby(["dataset", "task", "state_estimation_quality"], dropna=False)
        .agg(
            n_session_task_rows=("dataset", "size"),
            median_information_score=("state_information_score", "median"),
            median_accuracy=("calibration_accuracy", "median"),
            median_errors=("n_calibration_errors", "median"),
        )
        .reset_index()
    )
    return quality.sort_values(["dataset", "task", "state_estimation_quality"]).reset_index(drop=True)


def build_multiaxis_state_coordinates(state: pd.DataFrame) -> pd.DataFrame:
    output = state.copy()
    axis_cols = ["state_lapse_axis", "state_drift_axis", "state_variability_axis", "state_reliability_axis"]
    for col in axis_cols:
        output[f"{col}_z"] = zscore(output[col])
    output["state_multidimensional_summary_z"] = output[[f"{col}_z" for col in axis_cols]].mean(axis=1, skipna=True)
    return output


def write_leakage_audit(trainable: pd.DataFrame, splits: list[SplitData]) -> None:
    lines = [
        "# Step 08 Leakage-Control Audit",
        "",
        "Step 08 uses supervised events only where `correct` is available in the unified event table.",
        "",
        "Excluded datasets/tasks without supervised targets are reported in `human_training_event_exclusions.csv`.",
        "",
        "Model-validation splits:",
    ]
    for split in splits:
        train_ids = set(split.train_df.index)
        val_ids = set(split.val_df.index)
        overlap = len(train_ids & val_ids)
        lines.append(f"- `{split.name}`: train events={len(split.train_df)}, validation events={len(split.val_df)}, index overlap={overlap}")
    lines.extend(
        [
            "",
            "State-parameter estimation:",
            "- Uses only even `trial_index` events within each dataset/subject/session/task.",
            "- Odd `trial_index` events are held out and counted but never used to estimate state proxies.",
            "- `session_state_parameters.csv` contains `heldout_used_for_state_estimation=False` for every row.",
            "",
            "Current ANN-gate context:",
            "- Step 07 failed the quantitative state-axis gate.",
            "- Capacity coordinates are retained for downstream modeling as multidimensional proxies.",
            "- State parameters are exploratory multidimensional proxies and cannot support confirmatory claims alone.",
            "- `session_state_quality_report.csv` flags ceiling/low-variance sessions.",
            "- `participant_capacity_multidimensional_coordinates.csv` keeps hidden-size, complexity, load-robustness and consistency axes separate.",
        ]
    )
    (AUDIT / "step08_leakage_control.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    assert_full_run_allowed()
    ensure_dirs()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    trainable_raw, exclusions = prepare_training_events()
    trainable, feature_cols = add_features(trainable_raw)
    splits = make_splits(trainable)
    exclusions.to_csv(TABLES / "human_training_event_exclusions.csv", index=False)
    write_leakage_audit(trainable, splits)

    all_metrics = []
    all_predictions = []
    for split in splits:
        if split.train_df.empty or split.val_df.empty:
            continue
        x_train, y_train, mask_train, _ = build_sequences(split.train_df, feature_cols)
        x_val, y_val, mask_val, val_meta = build_sequences(split.val_df, feature_cols)
        for hidden_size in HIDDEN_SIZES:
            model, metrics, val_prob = train_model(
                split.name,
                hidden_size,
                (x_train, y_train, mask_train),
                (x_val, y_val, mask_val),
                input_dim=len(feature_cols),
            )
            checkpoint = CHECKPOINTS / f"human_gru_{split.name}_h{hidden_size}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                    "input_dim": len(feature_cols),
                    "feature_columns": feature_cols,
                    "split_strategy": split.name,
                    "metrics": metrics,
                    "ann_gate_context": "capacity_interpretable_state_exploratory_ann_gate_failed",
                },
                checkpoint,
            )
            metrics["checkpoint_path"] = checkpoint.relative_to(PROJECT_ROOT).as_posix()
            all_metrics.append(metrics)
            all_predictions.append(flatten_predictions(val_meta, val_prob, hidden_size, split.name))
            (TRAIN_LOGS / f"{split.name}_h{hidden_size}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(
                f"STEP08_MODEL split={split.name} hidden={hidden_size} "
                f"val_nll={metrics['val_nll']:.4f} val_acc={metrics['val_accuracy']:.4f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    model_selection, subject_validation = build_model_selection(predictions)
    capacity, capacity_multidim = build_capacity_coordinates(model_selection, trainable, subject_validation)
    state = build_state_parameters(trainable)
    state_multiaxis = build_multiaxis_state_coordinates(state)
    state_quality = build_state_quality_report(state)

    metrics_df.to_csv(TABLES / "training_validation_metrics.csv", index=False)
    subject_validation.to_csv(TABLES / "subject_validation_metrics_by_hidden_size.csv", index=False)
    model_selection.to_csv(TABLES / "model_selection_by_subject.csv", index=False)
    capacity.to_csv(TABLES / "participant_capacity_coordinates.csv", index=False)
    capacity_multidim.to_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv", index=False)
    state.to_csv(TABLES / "session_state_parameters.csv", index=False)
    state_multiaxis.to_csv(TABLES / "session_state_multiaxis_coordinates.csv", index=False)
    state_quality.to_csv(TABLES / "session_state_quality_report.csv", index=False)

    summary = {
        "n_supervised_events": int(len(trainable)),
        "n_participants": int(trainable["participant_id"].nunique()),
        "n_models": int(len(metrics_df)),
        "n_checkpoints": len(list(CHECKPOINTS.glob("human_gru_*.pt"))),
        "n_capacity_rows": int(len(capacity)),
        "n_capacity_multidim_rows": int(len(capacity_multidim)),
        "n_state_rows": int(len(state)),
        "n_state_multiaxis_rows": int(len(state_multiaxis)),
        "hidden_sizes": HIDDEN_SIZES,
        "split_strategies": SPLIT_STRATEGIES,
    }
    (TRAIN_LOGS / "step08_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STEP08_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
