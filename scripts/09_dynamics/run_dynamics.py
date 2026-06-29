from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SRC = ROOT / "src"
TRAIN_SCRIPT = ROOT / "scripts" / "07_train_tinyrnn"
for path in [SRC, TRAIN_SCRIPT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_all import HumanGRU, add_features, prepare_training_events


TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"
CHECKPOINTS = ROOT / "outputs" / "model_checkpoints" / "human_models"

RNG_SEED = 20260610
MAX_DECODER_ROWS = 30000


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def load_models() -> dict[int, tuple[HumanGRU, list[str]]]:
    models: dict[int, tuple[HumanGRU, list[str]]] = {}
    for checkpoint in sorted(CHECKPOINTS.glob("human_gru_odd_even_miniblock_h*.pt")):
        payload = torch.load(checkpoint, map_location="cpu")
        hidden_size = int(payload["hidden_size"])
        feature_cols = list(payload["feature_columns"])
        model = HumanGRU(input_dim=int(payload["input_dim"]), hidden_size=hidden_size)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[hidden_size] = (model, feature_cols)
    if not models:
        raise FileNotFoundError("No odd_even_miniblock human GRU checkpoints found.")
    return models


def align_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    aligned = df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[feature_cols].astype(float).to_numpy(dtype=np.float32)


def participation_ratio(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] == 0:
        return np.nan
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    denom = float(np.sum(eig**2))
    if denom <= 1e-12:
        return np.nan
    return float((np.sum(eig) ** 2) / denom)


def covariance_rank(matrix: np.ndarray) -> int:
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return 0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    return int(np.linalg.matrix_rank(np.atleast_2d(cov), tol=1e-6))


def recurrent_weight_geometry(model: HumanGRU, hidden_size: int) -> dict[str, float]:
    weight = model.gru.weight_hh_l0.detach().cpu().numpy()
    singular = np.linalg.svd(weight, compute_uv=False)
    pr = float((singular.sum() ** 2) / np.sum(singular**2)) if np.sum(singular**2) > 1e-12 else np.nan
    return {
        "hidden_size": hidden_size,
        "recurrent_weight_rank": int(np.linalg.matrix_rank(weight)),
        "recurrent_weight_top_singular": float(singular[0]) if len(singular) else np.nan,
        "recurrent_weight_participation_ratio": pr,
        "recurrent_weight_frobenius_norm": float(np.linalg.norm(weight)),
    }


def extract_hidden_trajectories(
    trainable_features: pd.DataFrame,
    selection: pd.DataFrame,
    models: dict[int, tuple[HumanGRU, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selection.set_index("participant_id")["selected_hidden_size"].to_dict()
    hidden_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for group_key, group in trainable_features.groupby(["dataset", "subject", "session_id", "task"], dropna=False):
        dataset, subject, session, task = group_key
        participant_id = f"{dataset}:{subject}"
        hidden_size = int(selected.get(participant_id, np.nan)) if participant_id in selected else None
        if hidden_size is None or hidden_size not in models:
            continue
        model, feature_cols = models[hidden_size]
        group = group.sort_values(["timestamp", "trial_index"]).reset_index(drop=True)
        x_np = align_features(group, feature_cols)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            hidden, logits = model.gru(x)
            pred = torch.sigmoid(model.out(hidden).squeeze(-1)).squeeze(0).cpu().numpy()
        h = hidden.squeeze(0).cpu().numpy()
        correct = group["correct_numeric"].to_numpy(dtype=float)
        steps = np.diff(h, axis=0) if len(h) > 1 else np.empty((0, h.shape[1]))
        step_norm = np.linalg.norm(steps, axis=1) if len(steps) else np.array([])
        centered = h - h.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(centered, axis=1)

        metric_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "session": session,
                "task": task,
                "participant_id": participant_id,
                "selected_hidden_size": hidden_size,
                "n_events": int(len(group)),
                "accuracy": float(np.mean(correct)),
                "mean_predicted_correct_probability": float(np.mean(pred)),
                "trajectory_participation_ratio": participation_ratio(h),
                "trajectory_cov_rank": covariance_rank(h),
                "trajectory_radius": float(np.mean(radius)),
                "trajectory_radius_sd": float(np.std(radius)),
                "trajectory_step_norm_mean": float(np.mean(step_norm)) if len(step_norm) else np.nan,
                "trajectory_step_norm_sd": float(np.std(step_norm)) if len(step_norm) else np.nan,
                "trajectory_total_length": float(np.sum(step_norm)) if len(step_norm) else 0.0,
                "mean_hidden_abs": float(np.mean(np.abs(h))),
                "hidden_variability": float(np.mean(np.std(h, axis=0))) if h.shape[0] > 1 else np.nan,
            }
        )

        for i, row in group.reset_index(drop=True).iterrows():
            hidden_record = {
                "dataset": dataset,
                "subject": subject,
                "session": session,
                "task": task,
                "participant_id": participant_id,
                "trial_index": int(row["trial_index"]),
                "correct": float(row["correct_numeric"]),
                "predicted_correct_probability": float(pred[i]),
                "hidden_size": hidden_size,
            }
            for dim in range(h.shape[1]):
                hidden_record[f"h{dim + 1}"] = float(h[i, dim])
            hidden_rows.append(hidden_record)

    return pd.DataFrame(metric_rows), pd.DataFrame(hidden_rows)


def fixed_point_for_model(model: HumanGRU, x_mean: np.ndarray, hidden_size: int) -> dict[str, float]:
    x = torch.tensor(x_mean, dtype=torch.float32).view(1, 1, -1)
    h = torch.zeros(1, 1, hidden_size, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(300):
            _, h_next = model.gru(x, h)
            if torch.norm(h_next - h).item() < 1e-7:
                h = h_next
                break
            h = h_next
    h_star = h.detach().view(-1)

    def step(flat_h: torch.Tensor) -> torch.Tensor:
        h0 = flat_h.view(1, 1, hidden_size)
        _, h_next = model.gru(x, h0)
        return h_next.view(-1)

    jac = torch.autograd.functional.jacobian(step, h_star).detach().cpu().numpy()
    eig = np.linalg.eigvals(jac)
    spectral_radius = float(np.max(np.abs(eig))) if len(eig) else np.nan
    if np.isfinite(spectral_radius) and 0 < spectral_radius < 1:
        timescale = float(-1.0 / math.log(max(spectral_radius, 1e-8)))
    elif np.isfinite(spectral_radius) and spectral_radius == 0:
        timescale = 0.0
    else:
        timescale = np.inf
    return {
        "fixed_point_norm": float(torch.norm(h_star).item()),
        "fixed_point_mean_abs": float(torch.mean(torch.abs(h_star)).item()),
        "jacobian_rank": int(np.linalg.matrix_rank(jac, tol=1e-6)),
        "jacobian_frobenius_norm": float(np.linalg.norm(jac)),
        "spectral_radius": spectral_radius,
        "memory_timescale_steps": timescale,
        "max_real_eigenvalue": float(np.max(np.real(eig))) if len(eig) else np.nan,
        "max_imag_abs_eigenvalue": float(np.max(np.abs(np.imag(eig)))) if len(eig) else np.nan,
    }


def compute_fixed_points(trainable_features: pd.DataFrame, models: dict[int, tuple[HumanGRU, list[str]]]) -> pd.DataFrame:
    rows = []
    for hidden_size, (model, feature_cols) in models.items():
        for task, group in trainable_features.groupby("task", dropna=False):
            x_mean = align_features(group, feature_cols).mean(axis=0)
            row = {
                "hidden_size": hidden_size,
                "task": task,
                "n_events_for_mean_input": int(len(group)),
            }
            row.update(fixed_point_for_model(model, x_mean, hidden_size))
            rows.append(row)
    return pd.DataFrame(rows)


def latent_decoder_results(hidden: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    hidden_cols = [c for c in hidden.columns if re.fullmatch(r"h[0-9]+", c)]
    decoder_df = hidden.dropna(subset=["participant_id"]).copy()
    if len(decoder_df) > MAX_DECODER_ROWS:
        decoder_df = decoder_df.iloc[rng.choice(len(decoder_df), MAX_DECODER_ROWS, replace=False)].copy()
    groups = decoder_df["participant_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    results = []
    targets = {
        "task": decoder_df["task"].astype(str),
        "correct": decoder_df["correct"].astype(int).astype(str),
        "dataset": decoder_df["dataset"].astype(str),
        "hidden_size": decoder_df["hidden_size"].astype(int).astype(str),
    }
    x = decoder_df[hidden_cols].fillna(0).to_numpy(dtype=float)
    for target_name, y_series in targets.items():
        y = y_series.to_numpy()
        if len(np.unique(y)) < 2 or n_splits < 2:
            results.append(
                {
                    "target": target_name,
                    "status": "insufficient_classes",
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "n_events": int(len(decoder_df)),
                    "n_classes": int(len(np.unique(y))),
                }
            )
            continue
        preds = np.empty(len(y), dtype=object)
        ok = np.zeros(len(y), dtype=bool)
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(x, y, groups=groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(x[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(x[test_idx])
            ok[test_idx] = True
        if not ok.any():
            status = "failed_cv"
            ba = np.nan
            f1 = np.nan
        else:
            status = "ok"
            ba = float(balanced_accuracy_score(y[ok], preds[ok]))
            f1 = float(f1_score(y[ok], preds[ok], average="macro"))
        results.append(
            {
                "target": target_name,
                "status": status,
                "balanced_accuracy": ba,
                "macro_f1": f1,
                "n_events": int(ok.sum()),
                "n_classes": int(len(np.unique(y))),
            }
        )
    return pd.DataFrame(results)


def spearman_effects(metrics: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(
        projection[
            [
                "dataset",
                "subject",
                "session",
                "task",
                "state_parameter_instability_z",
                "capacity_parameter_resource_z",
                "optimized_state_profile_z",
                "optimized_capacity_profile_z",
                "machine_state_projection_raw_z",
                "machine_capacity_projection_raw_z",
                "mean_accuracy",
            ]
        ],
        on=["dataset", "subject", "session", "task"],
        how="left",
    )
    outcomes = [
        "trajectory_participation_ratio",
        "trajectory_cov_rank",
        "trajectory_radius",
        "trajectory_step_norm_mean",
        "hidden_variability",
        "mean_hidden_abs",
    ]
    predictors = [
        "state_parameter_instability_z",
        "capacity_parameter_resource_z",
        "optimized_state_profile_z",
        "optimized_capacity_profile_z",
        "machine_state_projection_raw_z",
        "machine_capacity_projection_raw_z",
        "mean_accuracy",
    ]
    rows = []
    for outcome in outcomes:
        for predictor in predictors:
            data = merged[[outcome, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 5 or data[outcome].nunique() < 2 or data[predictor].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = stats.spearmanr(data[predictor], data[outcome])
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": int(len(data)),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    return pd.DataFrame(rows), merged


def make_figure(metrics: pd.DataFrame, fixed: pd.DataFrame, decoder: pd.DataFrame, effects: pd.DataFrame) -> None:
    source = effects.copy()
    source.to_csv(SOURCE_DATA / "figure_recurrent_dynamics_source.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    cap_effects = effects[
        effects["predictor"].eq("capacity_parameter_resource_z")
        & effects["outcome"].isin(["trajectory_participation_ratio", "trajectory_radius", "trajectory_step_norm_mean"])
    ].copy()
    axes[0].barh(cap_effects["outcome"], cap_effects["spearman_rho"], color="#4C78A8")
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_xlabel("Spearman rho")
    axes[0].set_title("Capacity vs geometry")

    fixed_plot = fixed.groupby("hidden_size", dropna=False)["spectral_radius"].mean().reset_index()
    axes[1].plot(fixed_plot["hidden_size"], fixed_plot["spectral_radius"], marker="o", color="#F58518")
    axes[1].set_xlabel("Hidden size")
    axes[1].set_ylabel("Mean spectral radius")
    axes[1].set_title("Local dynamics")

    ok = decoder[decoder["status"].eq("ok")].copy()
    axes[2].bar(ok["target"], ok["balanced_accuracy"], color="#54A24B")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Balanced accuracy")
    axes[2].set_title("Latent decoders")
    axes[2].tick_params(axis="x", rotation=35)

    for ax in axes:
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_recurrent_dynamics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    trainable_raw, _ = prepare_training_events()
    trainable_features, _ = add_features(trainable_raw)
    selection = pd.read_csv(TABLES / "model_selection_by_subject.csv")
    projection = pd.read_csv(TABLES / "human_state_capacity_multiaxis_projection.csv")
    models = load_models()

    metrics, hidden = extract_hidden_trajectories(trainable_features, selection, models)
    fixed = compute_fixed_points(trainable_features, models)
    decoder = latent_decoder_results(hidden)
    weight_geometry = pd.DataFrame(
        [recurrent_weight_geometry(model, hidden_size) for hidden_size, (model, _) in sorted(models.items())]
    )
    effects, merged_metrics = spearman_effects(metrics, projection)

    metrics.to_csv(TABLES / "recurrent_dynamics_by_subject_task.csv", index=False)
    fixed.to_csv(TABLES / "fixed_point_summary.csv", index=False)
    decoder.to_csv(TABLES / "latent_decoder_results.csv", index=False)
    weight_geometry.to_csv(TABLES / "recurrent_weight_geometry.csv", index=False)
    effects.to_csv(TABLES / "recurrent_dynamics_state_capacity_tests.csv", index=False)
    merged_metrics.to_csv(SOURCE_DATA / "recurrent_dynamics_merged_source.csv", index=False)
    hidden.to_parquet(ROOT / "outputs" / "tables" / "hidden_state_trajectories.parquet", index=False)
    make_figure(metrics, fixed, decoder, effects)

    cap_geometry = effects[
        effects["predictor"].eq("capacity_parameter_resource_z")
        & effects["outcome"].isin(
            [
                "trajectory_participation_ratio",
                "trajectory_cov_rank",
                "trajectory_radius",
                "trajectory_step_norm_mean",
            ]
        )
    ].copy()
    strongest = cap_geometry.reindex(cap_geometry["spearman_rho"].abs().sort_values(ascending=False).index).head(1)
    strongest_text = "none"
    if len(strongest):
        row = strongest.iloc[0]
        strongest_text = f"{row['outcome']} rho={row['spearman_rho']:.3f}, p={row['p_value']:.3g}, n={int(row['n'])}"
    audit = [
        "# Step 11 Recurrent Dynamics Claim Audit",
        "",
        "Step 11 was run on repaired supervised Step 08 models and coordinates.",
        "",
        f"- Session-task trajectories: {len(metrics)}.",
        f"- Hidden-state event rows: {len(hidden)}.",
        f"- Fixed-point rows: {len(fixed)}.",
        f"- Latent decoder targets: {len(decoder)}.",
        "",
        "## Capacity Geometry",
        "",
        f"- Strongest capacity/geometry association: {strongest_text}.",
        "- Because models are pooled by hidden size, participant dynamics should be interpreted as fitted-model geometry under participant/session/task input streams, not as individually trained neural dynamics.",
        "",
        "## State Boundary",
        "",
        "- State remains exploratory unless effects survive independent neurophysiology or stronger ANN residualized state recovery.",
    ]
    (AUDIT / "step11_recurrent_dynamics_claim_audit.md").write_text("\n".join(audit), encoding="utf-8")

    status = {
        "status": "implemented_and_run",
        "n_trajectories": int(len(metrics)),
        "n_hidden_rows": int(len(hidden)),
        "n_fixed_point_rows": int(len(fixed)),
        "n_decoder_targets": int(len(decoder)),
        "datasets": sorted(metrics["dataset"].dropna().unique().tolist()),
        "strongest_capacity_geometry": strongest_text,
    }
    (LOGS / "step11_recurrent_dynamics_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP11_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
