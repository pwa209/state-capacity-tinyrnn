from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "tu_berlin_eeg_nirs"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / sd


def bh_q(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return q
    m = len(valid)
    ranked = valid.to_numpy() * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q.loc[valid.index] = np.clip(ranked, 0.0, 1.0)
    return q


def safe_cv(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return np.nan
    mean = float(values.mean())
    if abs(mean) <= 1e-12:
        return np.nan
    return float(values.std(ddof=1) / abs(mean))


def load_behavior_with_coordinates() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(PROCESSED / "tu_berlin_model_events.parquet")
    events = events[events["event_included"].fillna(False).astype(bool)].copy()
    events["correct_numeric"] = pd.to_numeric(events["correct"], errors="coerce")
    events["rt_numeric"] = pd.to_numeric(events["rt"], errors="coerce")

    nback = events[events["task"].eq("nback")].copy()
    behavior = (
        nback.groupby(["dataset", "subject", "session", "task", "load_level"], dropna=False)
        .agg(
            n_events=("correct_numeric", "size"),
            mean_accuracy=("correct_numeric", "mean"),
            lapse_rate=("correct_numeric", lambda s: float(1.0 - np.nanmean(s))),
            rt_median=("rt_numeric", "median"),
            rt_iqr=("rt_numeric", lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))),
            rt_cv=("rt_numeric", safe_cv),
        )
        .reset_index()
    )

    state = pd.read_csv(TABLES / "session_state_multiaxis_coordinates.csv")
    capacity = pd.read_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv")
    projection_path = TABLES / "human_state_capacity_multiaxis_projection.csv"
    projection_cols = [
        "dataset",
        "subject",
        "session",
        "task",
        "state_parameter_instability_z",
        "capacity_parameter_resource_z",
        "optimized_state_profile_z",
        "optimized_capacity_profile_z",
        "dynamics_capacity_geometry_z",
        "dynamics_available",
    ]
    projection = pd.read_csv(projection_path)[projection_cols] if projection_path.exists() else pd.DataFrame()

    coords = behavior.merge(
        state,
        on=["dataset", "subject", "session", "task"],
        how="left",
        suffixes=("", "_state"),
    )
    coords["participant_id"] = coords["dataset"].astype(str) + ":" + coords["subject"].astype(str)
    coords = coords.merge(capacity, on=["participant_id", "dataset", "subject"], how="left", suffixes=("", "_capacity"))
    if not projection.empty:
        coords = coords.merge(projection, on=["dataset", "subject", "session", "task"], how="left")
    if "state_parameter_instability_z" not in coords:
        state_axes = [c for c in ["state_lapse_axis_z", "state_drift_axis_z", "state_variability_axis_z", "state_reliability_axis_z"] if c in coords]
        coords["state_parameter_instability_z"] = coords[state_axes].mean(axis=1, skipna=True) if state_axes else np.nan
    if "capacity_parameter_resource_z" not in coords:
        cap_axes = [
            c
            for c in [
                "capacity_hidden_size_axis_z_z",
                "capacity_selection_confidence_z",
                "capacity_complexity_preference_axis_z",
                "capacity_high_capacity_nll_advantage_z",
                "capacity_load_robustness_axis_z",
                "capacity_cross_task_consistency_axis_z",
            ]
            if c in coords
        ]
        coords["capacity_parameter_resource_z"] = coords[cap_axes].mean(axis=1, skipna=True) if cap_axes else np.nan
    coords["load_z"] = zscore(coords["load_level"])
    coords["load_x_capacity"] = coords["load_z"] * pd.to_numeric(coords["capacity_parameter_resource_z"], errors="coerce")
    coords["load_x_optimized_capacity"] = coords["load_z"] * pd.to_numeric(
        coords.get("optimized_capacity_profile_z", np.nan), errors="coerce"
    )
    return events, coords


def design_matrix(df: pd.DataFrame, predictors: list[str], categorical: list[str]) -> tuple[np.ndarray, list[str]]:
    pieces = [pd.Series(1.0, index=df.index, name="intercept").to_frame()]
    names = ["intercept"]
    for col in predictors:
        if col in categorical:
            dummies = pd.get_dummies(df[col].astype("string"), prefix=col, drop_first=True, dtype=float)
            if not dummies.empty:
                pieces.append(dummies)
                names.extend(dummies.columns.tolist())
        else:
            values = pd.to_numeric(df[col], errors="coerce")
            fill = float(values.median()) if values.notna().any() else 0.0
            pieces.append(values.fillna(fill).rename(col).to_frame())
            names.append(col)
    x = pd.concat(pieces, axis=1).to_numpy(dtype=float)
    return x, names


def ols_test(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    target: str,
    model_name: str,
    family: str,
    categorical: list[str] | None = None,
) -> dict[str, Any]:
    categorical = categorical or []
    needed = [outcome] + [p for p in predictors if p not in categorical]
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in needed if c in df.columns]).copy()
    if len(data) < 8 or outcome not in data or target not in predictors:
        return {
            "model_name": model_name,
            "family": family,
            "outcome": outcome,
            "target_predictor": target,
            "n": int(len(data)),
            "estimate": np.nan,
            "std_error": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "status": "insufficient_data",
        }
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(dtype=float)
    x, names = design_matrix(data, predictors, categorical)
    rank = np.linalg.matrix_rank(x)
    if len(y) <= rank + 1 or target not in names:
        return {
            "model_name": model_name,
            "family": family,
            "outcome": outcome,
            "target_predictor": target,
            "n": int(len(data)),
            "estimate": np.nan,
            "std_error": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "status": "rank_or_target_failure",
        }
    beta = np.linalg.pinv(x) @ y
    pred = x @ beta
    resid = y - pred
    df_resid = max(1, len(y) - rank)
    sigma2 = float(np.sum(resid**2) / df_resid)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    idx = names.index(target)
    se = float(np.sqrt(max(cov[idx, idx], 0.0)))
    t_value = float(beta[idx] / se) if se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_value), df=df_resid)) if np.isfinite(t_value) else np.nan
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(resid**2) / sst) if sst > 1e-12 else np.nan
    return {
        "model_name": model_name,
        "family": family,
        "outcome": outcome,
        "target_predictor": target,
        "n": int(len(data)),
        "n_subjects": int(data["subject"].nunique()) if "subject" in data else np.nan,
        "estimate": float(beta[idx]),
        "std_error": se,
        "t_value": t_value,
        "p_value": p_value,
        "r_squared": r2,
        "status": "ok",
    }


def load_from_class_name(name: str) -> int | None:
    match = re.search(r"([023])-back", str(name))
    return int(match.group(1)) if match else None


def marker_blocks(mrk: Any) -> list[dict[str, Any]]:
    times = np.ravel(np.asarray(mrk.time, dtype=float))
    y = np.asarray(mrk.y)
    class_names = np.ravel(np.asarray(mrk.className, dtype=object))
    rows = []
    for class_idx, class_name in enumerate(class_names):
        if "session" not in str(class_name):
            continue
        load = load_from_class_name(str(class_name))
        if load is None:
            continue
        for event_idx in np.flatnonzero(y[class_idx] == 1):
            rows.append({"time_ms": float(times[event_idx]), "load_level": load})
    rows = sorted(rows, key=lambda row: row["time_ms"])
    for idx, row in enumerate(rows):
        row["session"] = str(idx // 9 + 1)
        row["block"] = str(idx % 9 + 1)
        row["next_time_ms"] = float(rows[idx + 1]["time_ms"]) if idx + 1 < len(rows) else np.nan
    return rows


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def eeg_window_features(window: np.ndarray, fs: float, channel_names: list[str]) -> dict[str, float]:
    if window.ndim != 2 or window.shape[0] < int(fs * 5):
        return {}
    values = window.astype(float)
    values = values - np.nanmedian(values, axis=0, keepdims=True)
    values = np.nan_to_num(values, copy=False)
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(1024, len(values)), axis=0)
    mean_psd = np.nanmean(psd, axis=1)
    total = band_power(freqs, mean_psd, 1.0, 30.0)
    out: dict[str, float] = {"eeg_total_power": total}
    for name, (low, high) in BANDS.items():
        out[f"eeg_{name}_power"] = band_power(freqs, mean_psd, low, high)
        out[f"eeg_relative_{name}"] = out[f"eeg_{name}_power"] / total if total and np.isfinite(total) else np.nan
    out["eeg_theta_alpha_ratio"] = out["eeg_theta_power"] / out["eeg_alpha_power"] if out["eeg_alpha_power"] else np.nan
    norm = mean_psd[(freqs >= 1.0) & (freqs <= 30.0)].copy()
    norm = norm / np.nansum(norm) if np.nansum(norm) > 0 else norm
    out["eeg_spectral_entropy"] = float(-np.nansum(norm * np.log2(norm + 1e-12)) / math.log2(len(norm))) if len(norm) else np.nan
    slope_mask = (freqs >= 2.0) & (freqs <= 30.0) & (mean_psd > 0)
    out["eeg_aperiodic_slope"] = (
        float(np.polyfit(np.log10(freqs[slope_mask]), np.log10(mean_psd[slope_mask]), 1)[0])
        if slope_mask.sum() >= 5
        else np.nan
    )
    frontal = [i for i, ch in enumerate(channel_names) if str(ch).upper().startswith(("FP", "AF", "F", "FC"))]
    if frontal:
        frontal_psd = np.nanmean(psd[:, frontal], axis=1)
        out["eeg_frontal_theta_power"] = band_power(freqs, frontal_psd, 4.0, 7.0)
        out["eeg_frontal_alpha_power"] = band_power(freqs, frontal_psd, 8.0, 12.0)
    return out


def nirs_window_features(oxy: np.ndarray, deoxy: np.ndarray, fs: float) -> dict[str, float]:
    if oxy.ndim != 2 or oxy.shape[0] < int(fs * 5):
        return {}
    x = np.arange(oxy.shape[0], dtype=float) / fs
    hbo_mean = np.nanmean(oxy, axis=1)
    hbr_mean = np.nanmean(deoxy, axis=1)
    return {
        "nirs_hbo_mean": float(np.nanmean(hbo_mean)),
        "nirs_hbo_sd": float(np.nanstd(hbo_mean)),
        "nirs_hbo_slope": float(np.polyfit(x, hbo_mean, 1)[0]) if len(x) >= 3 else np.nan,
        "nirs_hbo_end_minus_start": float(np.nanmean(hbo_mean[-int(fs * 5) :]) - np.nanmean(hbo_mean[: int(fs * 5)])),
        "nirs_hbr_mean": float(np.nanmean(hbr_mean)),
        "nirs_hbr_sd": float(np.nanstd(hbr_mean)),
        "nirs_hbr_slope": float(np.polyfit(x, hbr_mean, 1)[0]) if len(x) >= 3 else np.nan,
        "nirs_hbr_end_minus_start": float(np.nanmean(hbr_mean[-int(fs * 5) :]) - np.nanmean(hbr_mean[: int(fs * 5)])),
    }


def extract_eeg_features() -> pd.DataFrame:
    archive = RAW / "EEG_01-26_MATLAB.zip"
    rows = []
    if not archive.exists():
        return pd.DataFrame(rows)
    with zipfile.ZipFile(archive) as zf:
        subjects = sorted({name.split("/")[0].replace("-EEG", "") for name in zf.namelist() if name.endswith("cnt_nback.mat")})
        for subject in subjects:
            prefix = f"{subject}-EEG"
            try:
                cnt = loadmat(io.BytesIO(zf.read(f"{prefix}/cnt_nback.mat")), squeeze_me=True, struct_as_record=False)["cnt_nback"]
                mrk = loadmat(io.BytesIO(zf.read(f"{prefix}/mrk_nback.mat")), squeeze_me=True, struct_as_record=False)["mrk_nback"]
            except Exception as exc:
                rows.append({"subject": subject, "modality": "eeg", "status": "failed", "error": str(exc)})
                continue
            data = np.asarray(cnt.x, dtype=float)
            fs = float(cnt.fs)
            channel_names = [str(ch) for ch in np.ravel(cnt.clab)]
            for block in marker_blocks(mrk):
                start = int(round(block["time_ms"] / 1000.0 * fs))
                next_stop = int(round(block["next_time_ms"] / 1000.0 * fs)) if np.isfinite(block["next_time_ms"]) else len(data)
                stop = min(len(data), start + int(60 * fs), max(start, next_stop - int(2 * fs)))
                row = {
                    "dataset": "tu_berlin_eeg_nirs",
                    "subject": subject,
                    "session": block["session"],
                    "block": block["block"],
                    "load_level": block["load_level"],
                    "modality": "eeg",
                    "window_samples": max(0, stop - start),
                    "status": "ok",
                }
                row.update(eeg_window_features(data[start:stop], fs, channel_names))
                rows.append(row)
    return pd.DataFrame(rows)


def extract_nirs_features() -> pd.DataFrame:
    archive = RAW / "NIRS_01-26_MATLAB.zip"
    rows = []
    if not archive.exists():
        return pd.DataFrame(rows)
    with zipfile.ZipFile(archive) as zf:
        subjects = sorted({name.split("/")[0].replace("-NIRS", "") for name in zf.namelist() if name.endswith("cnt_nback.mat")})
        for subject in subjects:
            prefix = f"{subject}-NIRS"
            try:
                cnt = loadmat(io.BytesIO(zf.read(f"{prefix}/cnt_nback.mat")), squeeze_me=True, struct_as_record=False)["cnt_nback"]
                mrk = loadmat(io.BytesIO(zf.read(f"{prefix}/mrk_nback.mat")), squeeze_me=True, struct_as_record=False)["mrk_nback"]
            except Exception as exc:
                rows.append({"subject": subject, "modality": "nirs", "status": "failed", "error": str(exc)})
                continue
            oxy = np.asarray(cnt.oxy.x, dtype=float)
            deoxy = np.asarray(cnt.deoxy.x, dtype=float)
            fs = float(cnt.oxy.fs)
            for block in marker_blocks(mrk):
                start = int(round(block["time_ms"] / 1000.0 * fs))
                next_stop = int(round(block["next_time_ms"] / 1000.0 * fs)) if np.isfinite(block["next_time_ms"]) else len(oxy)
                stop = min(len(oxy), start + int(60 * fs), max(start, next_stop - int(2 * fs)))
                row = {
                    "dataset": "tu_berlin_eeg_nirs",
                    "subject": subject,
                    "session": block["session"],
                    "block": block["block"],
                    "load_level": block["load_level"],
                    "modality": "nirs",
                    "window_samples": max(0, stop - start),
                    "status": "ok",
                }
                row.update(nirs_window_features(oxy[start:stop], deoxy[start:stop], fs))
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_physiology(eeg: pd.DataFrame, nirs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df in [eeg, nirs]:
        if df.empty:
            continue
        ok = df[df["status"].eq("ok")].copy()
        numeric_cols = [
            col
            for col in ok.columns
            if col not in {"dataset", "subject", "session", "block", "load_level", "modality", "status", "error"}
            and pd.api.types.is_numeric_dtype(ok[col])
        ]
        frames.append(
            ok.groupby(["dataset", "subject", "session", "load_level"], dropna=False)[numeric_cols]
            .mean()
            .reset_index()
        )
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["dataset", "subject", "session", "load_level"], how="outer")
    return out


def run_models(coords: pd.DataFrame, physio_coords: pd.DataFrame) -> pd.DataFrame:
    subject_session_controls = ["load_z", "subject", "session"]
    pressure_controls = ["load_z", "capacity_parameter_resource_z", "load_x_capacity", "state_parameter_instability_z", "subject", "session"]
    state_controls = ["load_z", "state_parameter_instability_z", "capacity_parameter_resource_z", "subject", "session"]
    rows = [
        ols_test(coords, "mean_accuracy", subject_session_controls, "load_z", "behavior_accuracy_load_effect", "behavior", ["subject", "session"]),
        ols_test(coords, "rt_median", subject_session_controls, "load_z", "behavior_rt_load_effect", "behavior", ["subject", "session"]),
        ols_test(coords, "mean_accuracy", pressure_controls, "load_x_capacity", "capacity_pressure_accuracy", "capacity_pressure", ["subject", "session"]),
        ols_test(coords, "rt_median", pressure_controls, "load_x_capacity", "capacity_pressure_rt", "capacity_pressure", ["subject", "session"]),
        ols_test(coords, "lapse_rate", state_controls, "state_parameter_instability_z", "state_lapse_reliability", "state", ["subject", "session"]),
        ols_test(coords, "rt_cv", state_controls, "state_parameter_instability_z", "state_rt_variability", "state", ["subject", "session"]),
    ]
    if not physio_coords.empty:
        physio_coords = physio_coords.copy()
        physio_coords["load_z"] = zscore(physio_coords["load_level"])
        physio_coords["load_x_capacity"] = physio_coords["load_z"] * pd.to_numeric(
            physio_coords["capacity_parameter_resource_z"], errors="coerce"
        )
        for outcome in ["eeg_theta_alpha_ratio", "eeg_spectral_entropy", "eeg_aperiodic_slope", "nirs_hbo_mean", "nirs_hbo_slope", "nirs_hbr_mean"]:
            if outcome in physio_coords.columns:
                rows.append(
                    ols_test(
                        physio_coords,
                        outcome,
                        ["load_z", "subject", "session"],
                        "load_z",
                        f"physiology_{outcome}_load_effect",
                        "physiology_load",
                        ["subject", "session"],
                    )
                )
                rows.append(
                    ols_test(
                        physio_coords,
                        outcome,
                        pressure_controls,
                        "load_x_capacity",
                        f"physiology_{outcome}_capacity_pressure",
                        "physiology_capacity_pressure",
                        ["subject", "session"],
                    )
                )
    models = pd.DataFrame(rows)
    models["q_value"] = bh_q(models["p_value"])
    models["bh_significant_05"] = models["q_value"] < 0.05
    return models


def make_figure(coords: pd.DataFrame, physio_coords: pd.DataFrame, models: pd.DataFrame) -> None:
    source_rows = []
    load_summary = (
        coords.groupby("load_level", dropna=False)
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            sem_accuracy=("mean_accuracy", lambda s: float(stats.sem(s, nan_policy="omit"))),
            rt_median=("rt_median", "mean"),
            sem_rt=("rt_median", lambda s: float(stats.sem(s, nan_policy="omit"))),
        )
        .reset_index()
    )
    for _, row in load_summary.iterrows():
        record = row.to_dict()
        record["panel"] = "behavior_load"
        source_rows.append(record)
    model_plot = models[models["status"].eq("ok")].copy()
    for _, row in model_plot.iterrows():
        rec = row.to_dict()
        rec["panel"] = "model_coefficients"
        source_rows.append(rec)
    if not physio_coords.empty:
        phys_summary = (
            physio_coords.groupby("load_level", dropna=False)
            .agg(
                eeg_theta_alpha_ratio=("eeg_theta_alpha_ratio", "mean"),
                nirs_hbo_mean=("nirs_hbo_mean", "mean"),
            )
            .reset_index()
        )
        for _, row in phys_summary.iterrows():
            record = row.to_dict()
            record["panel"] = "physiology_load"
            source_rows.append(record)
    pd.DataFrame(source_rows).to_csv(SOURCE_DATA / "figure_tu_berlin_validation_source.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axes = axes.ravel()
    axes[0].errorbar(load_summary["load_level"], load_summary["mean_accuracy"], yerr=load_summary["sem_accuracy"], marker="o", color="#0072B2")
    axes[0].set_xlabel("N-back load")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Load pressure")

    axes[1].errorbar(load_summary["load_level"], load_summary["rt_median"], yerr=load_summary["sem_rt"], marker="o", color="#D55E00")
    axes[1].set_xlabel("N-back load")
    axes[1].set_ylabel("Median RT (s)")
    axes[1].set_title("Response slowing")

    forest = model_plot[
        model_plot["model_name"].isin(
            [
                "capacity_pressure_accuracy",
                "capacity_pressure_rt",
                "state_lapse_reliability",
                "state_rt_variability",
            ]
        )
    ].copy()
    if not forest.empty:
        y = np.arange(len(forest))
        axes[2].errorbar(forest["estimate"], y, xerr=1.96 * forest["std_error"], fmt="o", color="#009E73")
        axes[2].axvline(0, color="#555555", linewidth=0.8)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(forest["model_name"])
    axes[2].set_xlabel("Coefficient")
    axes[2].set_title("State/capacity tests")

    if not physio_coords.empty and {"eeg_theta_alpha_ratio", "nirs_hbo_mean"}.issubset(physio_coords.columns):
        phys_summary = (
            physio_coords.groupby("load_level", dropna=False)
            .agg(eeg_theta_alpha_ratio=("eeg_theta_alpha_ratio", "mean"), nirs_hbo_mean=("nirs_hbo_mean", "mean"))
            .reset_index()
        )
        ax2 = axes[3].twinx()
        axes[3].plot(phys_summary["load_level"], phys_summary["eeg_theta_alpha_ratio"], marker="o", color="#4C78A8")
        ax2.plot(phys_summary["load_level"], phys_summary["nirs_hbo_mean"], marker="s", color="#CC79A7")
        axes[3].set_ylabel("EEG theta/alpha", color="#4C78A8")
        ax2.set_ylabel("NIRS HbO mean", color="#CC79A7")
    axes[3].set_xlabel("N-back load")
    axes[3].set_title("Multimodal physiology")

    for ax in axes:
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_tu_berlin_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    _, coords = load_behavior_with_coordinates()
    coords.to_csv(TABLES / "tu_berlin_coordinates.csv", index=False)
    coords.to_csv(TABLES / "tu_berlin_behavior_load_summary.csv", index=False)

    eeg = extract_eeg_features()
    nirs = extract_nirs_features()
    eeg.to_csv(TABLES / "tu_berlin_eeg_features.csv", index=False)
    nirs.to_csv(TABLES / "tu_berlin_nirs_features.csv", index=False)
    physio = aggregate_physiology(eeg, nirs)
    if not physio.empty:
        physio_coords = physio.merge(
            coords.drop(columns=[c for c in ["load_z", "load_x_capacity"] if c in coords.columns]),
            on=["dataset", "subject", "session", "load_level"],
            how="left",
            suffixes=("", "_behavior"),
        )
    else:
        physio_coords = pd.DataFrame()
    physio_coords.to_csv(TABLES / "tu_berlin_eeg_nirs_features.csv", index=False)

    models = run_models(coords, physio_coords)
    models.to_csv(TABLES / "tu_berlin_load_validation.csv", index=False)
    make_figure(coords, physio_coords, models)

    load_row = models[models["model_name"].eq("behavior_accuracy_load_effect")].head(1)
    pressure_row = models[models["model_name"].eq("capacity_pressure_accuracy")].head(1)
    physiology_ok = int((eeg.get("status", pd.Series(dtype=str)).eq("ok")).sum()) + int(
        (nirs.get("status", pd.Series(dtype=str)).eq("ok")).sum()
    )
    audit = [
        "# Step 14 TU Berlin EEG-NIRS Validation Audit",
        "",
        f"- Behavioral load-coordinate rows: {len(coords)}.",
        f"- Subjects with behavior: {coords['subject'].nunique()}.",
        f"- EEG block windows extracted: {int((eeg.get('status', pd.Series(dtype=str)).eq('ok')).sum())}.",
        f"- NIRS block windows extracted: {int((nirs.get('status', pd.Series(dtype=str)).eq('ok')).sum())}.",
        f"- Tested model rows: {len(models)}.",
        "",
        "## Interpretation Boundary",
        "",
        "- TU Berlin is a load/capacity-pressure validation, not a new training dataset.",
        "- Capacity-pressure claims require the load-by-capacity interaction to survive task/session/subject controls and BH correction.",
        "- State claims are still exploratory because the ANN state gate failed, even when behavioral or physiology associations are nominally significant.",
    ]
    if not load_row.empty:
        row = load_row.iloc[0]
        audit.append(f"- Load effect on accuracy: beta={row['estimate']:.4g}, p={row['p_value']:.3g}, q={row['q_value']:.3g}.")
    if not pressure_row.empty:
        row = pressure_row.iloc[0]
        audit.append(
            f"- Capacity-pressure effect on accuracy: beta={row['estimate']:.4g}, p={row['p_value']:.3g}, q={row['q_value']:.3g}."
        )
    (AUDIT / "step14_tu_berlin_validation_audit.md").write_text("\n".join(audit), encoding="utf-8")

    status = {
        "status": "implemented_and_run",
        "behavior_rows": int(len(coords)),
        "subjects": int(coords["subject"].nunique()),
        "eeg_windows_ok": int((eeg.get("status", pd.Series(dtype=str)).eq("ok")).sum()),
        "nirs_windows_ok": int((nirs.get("status", pd.Series(dtype=str)).eq("ok")).sum()),
        "physiology_windows_ok": physiology_ok,
        "model_rows": int(len(models)),
        "bh_significant_rows": int(models["bh_significant_05"].fillna(False).sum()),
        "outputs": [
            "outputs/tables/tu_berlin_coordinates.csv",
            "outputs/tables/tu_berlin_load_validation.csv",
            "outputs/figures/figure_tu_berlin_validation.png",
            "outputs/source_data/figure_tu_berlin_validation_source.csv",
        ],
    }
    (LOGS / "step14_tu_berlin_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP14_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
