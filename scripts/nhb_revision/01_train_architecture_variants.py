from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SRC = ROOT / "src"
TRAIN_SCRIPT = ROOT / "scripts" / "07_train_tinyrnn"
for path in [SRC, TRAIN_SCRIPT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_all import (  # type: ignore
    BATCH_SIZE,
    HIDDEN_SIZES,
    LR,
    SEQ_LEN,
    add_features,
    build_sequences,
    make_splits,
    nll_from_prob,
    prepare_training_events,
)
from nhb_utils import (
    NHB_MODELS,
    NHB_TABLES,
    append_exclusion,
    append_manifest,
    append_registry,
    ensure_nhb_dirs,
    safe_auc,
    safe_balanced_accuracy,
)


ANALYSIS_ID = "nhb_01_train_architecture_variants"
SCRIPT_NAME = "scripts/nhb_revision/01_train_architecture_variants.py"
MODEL_FAMILIES = ["vanilla_rnn", "gru", "lstm"]
SPLIT_STRATEGIES = ["participant_level", "session_blocked", "odd_even_miniblock"]
EPOCHS = int(os.environ.get("NHB_ARCH_EPOCHS", "12"))
RNG_SEED = 20260611


class CompactRecurrent(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, model_family: str):
        super().__init__()
        self.model_family = model_family
        self.hidden_size = hidden_size
        if model_family == "vanilla_rnn":
            self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True, nonlinearity="tanh")
        elif model_family == "gru":
            self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True)
        elif model_family == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        return self.out(hidden).squeeze(-1)


def train_one(
    model_family: str,
    split_name: str,
    hidden_size: int,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    input_dim: int,
) -> tuple[CompactRecurrent, dict[str, object], np.ndarray]:
    torch.manual_seed(RNG_SEED + hidden_size + 17 * MODEL_FAMILIES.index(model_family))
    model = CompactRecurrent(input_dim=input_dim, hidden_size=hidden_size, model_family=model_family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    x_train, y_train, mask_train = train_tensors
    x_val, y_val, mask_val = val_tensors

    pos = float((y_train * mask_train).sum())
    neg = float(((1 - y_train) * mask_train).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    rng = np.random.default_rng(RNG_SEED + hidden_size + len(split_name) + len(model_family))
    history = []
    start_time = time.time()
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(x_train.shape[0])
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            logits = model(x_train[idx])
            loss_raw = criterion(logits, y_train[idx])
            loss = (loss_raw * mask_train[idx]).sum() / mask_train[idx].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x_train)).cpu().numpy()
        val_prob = torch.sigmoid(model(x_val)).cpu().numpy()

    train_mask = mask_train.cpu().numpy().astype(bool)
    val_mask = mask_val.cpu().numpy().astype(bool)
    y_train_np = y_train.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    train_prob_flat = train_prob[train_mask]
    val_prob_flat = val_prob[val_mask]
    y_train_flat = y_train_np[train_mask]
    y_val_flat = y_val_np[val_mask]
    val_pred = val_prob_flat >= 0.5

    metrics = {
        "analysis_id": ANALYSIS_ID,
        "script_name": SCRIPT_NAME,
        "split": split_name,
        "model_family": model_family,
        "hidden_size": hidden_size,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "n_test": 0,
        "accuracy": float((val_pred == y_val_flat).mean()),
        "balanced_accuracy": safe_balanced_accuracy(y_val_flat, val_pred.astype(float)),
        "auc": safe_auc(y_val_flat, val_prob_flat),
        "cross_entropy": float(nll_from_prob(val_prob_flat, y_val_flat).mean()),
        "rmse": float(np.sqrt(np.mean((val_prob_flat - y_val_flat) ** 2))),
        "train_cross_entropy": float(nll_from_prob(train_prob_flat, y_train_flat).mean()),
        "final_training_loss": float(history[-1]),
        "epochs": EPOCHS,
        "converged": bool(np.isfinite(history[-1])),
        "training_time_seconds": float(time.time() - start_time),
    }
    return model, metrics, val_prob


def flatten_predictions(
    val_meta: pd.DataFrame,
    val_prob: np.ndarray,
    hidden_size: int,
    split_name: str,
    model_family: str,
) -> pd.DataFrame:
    rows = []
    for _, row in val_meta.iterrows():
        prob = float(val_prob[int(row["sequence_index"]), int(row["sequence_offset"])])
        y = float(row["correct"])
        rows.append(
            {
                **row.to_dict(),
                "analysis_id": ANALYSIS_ID,
                "script_name": SCRIPT_NAME,
                "split": split_name,
                "model_family": model_family,
                "hidden_size": hidden_size,
                "predicted_correct_probability": prob,
                "event_nll": float(nll_from_prob(np.array([prob]), np.array([y]))[0]),
                "predicted_correct": bool(prob >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "task", "split", "model_family", "hidden_size"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        dataset, task, split, model_family, hidden_size = keys
        y = group["correct"].to_numpy(dtype=float)
        prob = group["predicted_correct_probability"].to_numpy(dtype=float)
        pred = (prob >= 0.5).astype(float)
        rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "script_name": SCRIPT_NAME,
                "dataset": dataset,
                "task": task,
                "subject_id": "",
                "participant_id": "",
                "session": "",
                "split": split,
                "model_family": model_family,
                "hidden_size": int(hidden_size),
                "state_definition": "not_estimated_in_architecture_training",
                "capacity_definition": "hidden_size_architecture_variant",
                "outcome": "correct",
                "predictor": "compact_recurrent_prediction",
                "n_rows": int(len(group)),
                "n_subjects": int(group["participant_id"].nunique()),
                "estimate": float((pred == y).mean()),
                "std_error": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "q_value": "",
                "effect_direction": "higher_is_better",
                "control_status": "architecture_variant",
                "claim_strength": "moderate",
                "interpretation": "Validation performance for architecture robustness; compare across model families.",
                "source_table": "architecture_variant_prediction_metrics.csv",
                "accuracy": float((pred == y).mean()),
                "balanced_accuracy": safe_balanced_accuracy(y, pred),
                "auc": safe_auc(y, prob),
                "cross_entropy": float(nll_from_prob(prob, y).mean()),
                "rmse": float(np.sqrt(np.mean((prob - y) ** 2))),
                "n_train": "",
                "n_validation": int(len(group)),
                "n_test": 0,
                "converged": True,
                "training_time_seconds": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    model_dir = NHB_MODELS / "architecture_variants"
    model_dir.mkdir(parents=True, exist_ok=True)

    trainable, exclusion = prepare_training_events()
    trainable, feature_cols = add_features(trainable)
    splits = {split.name: split for split in make_splits(trainable) if split.name in SPLIT_STRATEGIES}
    training_rows = []
    all_predictions = []

    for _, row in exclusion.iterrows():
        if not bool(row["included_in_step08_training"]):
            append_exclusion(
                ANALYSIS_ID,
                SCRIPT_NAME,
                str(row["dataset"]),
                str(row["task"]),
                "all",
                str(row["exclusion_reason"]),
                "info",
            )

    for split_name, split in splits.items():
        train_tensors = build_sequences(split.train_df, feature_cols)[:3]
        x_val, y_val, mask_val, val_meta = build_sequences(split.val_df, feature_cols)
        val_tensors = (x_val, y_val, mask_val)
        for model_family in MODEL_FAMILIES:
            for hidden_size in HIDDEN_SIZES:
                try:
                    model, metrics, val_prob = train_one(
                        model_family,
                        split_name,
                        hidden_size,
                        train_tensors,
                        val_tensors,
                        input_dim=len(feature_cols),
                    )
                    checkpoint = model_dir / f"{model_family}_{split_name}_h{hidden_size}.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_family": model_family,
                            "hidden_size": hidden_size,
                            "split": split_name,
                            "feature_cols": feature_cols,
                            "seq_len": SEQ_LEN,
                            "epochs": EPOCHS,
                        },
                        checkpoint,
                    )
                    metrics["checkpoint_path"] = checkpoint.relative_to(ROOT).as_posix()
                    training_rows.append(metrics)
                    all_predictions.append(flatten_predictions(val_meta, val_prob, hidden_size, split_name, model_family))
                    print(f"trained {model_family} {split_name} h={hidden_size}")
                except Exception as exc:
                    append_exclusion(
                        ANALYSIS_ID,
                        SCRIPT_NAME,
                        "all",
                        "all",
                        model_family,
                        f"{split_name} hidden_size={hidden_size}: {exc}",
                        "error",
                    )

    summary = pd.DataFrame(training_rows)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    pred_metrics = prediction_metrics(predictions) if not predictions.empty else pd.DataFrame()

    summary_path = NHB_TABLES / "architecture_variant_training_summary.csv"
    metrics_path = NHB_TABLES / "architecture_variant_prediction_metrics.csv"
    summary.to_csv(summary_path, index=False)
    pred_metrics.to_csv(metrics_path, index=False)

    outputs = [summary_path, metrics_path]
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        outputs,
        notes=f"Trained {len(summary)} architecture/split/hidden-size cells with EPOCHS={EPOCHS}.",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
