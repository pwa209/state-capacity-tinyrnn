from __future__ import annotations

import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "cog_bci"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"

EEG_TASK_FILES = {
    "pvt": ["PVT"],
    "flanker": ["Flanker"],
    "nback": ["zeroBACK", "oneBACK", "twoBACK"],
}
EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
N_PERMUTATIONS = 5000
RNG_SEED = 20260610


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def spectral_entropy(psd: np.ndarray) -> float:
    clean = np.asarray(psd, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    if len(clean) <= 1:
        return np.nan
    p = clean / clean.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(len(p)))


def aperiodic_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    mask = (freqs >= 2.0) & (freqs <= 40.0) & np.isfinite(psd) & (psd > 0)
    if mask.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), deg=1)[0])


def extract_eeg_features_from_raw(raw: mne.io.BaseRaw, row_meta: dict[str, Any]) -> dict[str, Any]:
    raw.load_data(verbose="ERROR")
    raw.pick_types(eeg=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()
    data = data - np.nanmedian(data, axis=1, keepdims=True)
    nperseg = min(int(sfreq * 4), data.shape[1])
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    mean_psd = np.nanmedian(psd, axis=0)
    powers = {name: band_power(freqs, mean_psd, low, high) for name, (low, high) in EEG_BANDS.items()}
    total_power = band_power(freqs, mean_psd, 1.0, 40.0)
    out = {
        **row_meta,
        "modality": "eeg",
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "sampling_frequency": sfreq,
        "recording_duration_sec": float(data.shape[1] / sfreq),
        "cog_eeg_delta_power": powers["delta"],
        "cog_eeg_theta_power": powers["theta"],
        "cog_eeg_alpha_power": powers["alpha"],
        "cog_eeg_beta_power": powers["beta"],
        "cog_eeg_theta_alpha_ratio": powers["theta"] / powers["alpha"] if powers["alpha"] and powers["alpha"] > 0 else np.nan,
        "cog_eeg_relative_theta": powers["theta"] / total_power if total_power and total_power > 0 else np.nan,
        "cog_eeg_relative_alpha": powers["alpha"] / total_power if total_power and total_power > 0 else np.nan,
        "cog_eeg_spectral_entropy": spectral_entropy(mean_psd[(freqs >= 1.0) & (freqs <= 40.0)]),
        "cog_eeg_aperiodic_slope": aperiodic_slope(freqs, mean_psd),
        "cog_eeg_channel_variability": float(np.nanmedian(np.nanstd(data, axis=1))),
        "feature_status": "ok",
        "feature_error": "",
    }
    return out


def extract_cog_eeg_features() -> pd.DataFrame:
    cache = TABLES / "cog_bci_eeg_features.csv"
    rows: list[dict[str, Any]] = []
    archives = sorted(RAW.glob("sub-*.zip"))
    for archive_index, archive in enumerate(archives, start=1):
        subject = archive.stem
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for session in ["ses-S1", "ses-S2", "ses-S3"]:
                for task, stems in EEG_TASK_FILES.items():
                    for stem in stems:
                        set_member = f"{subject}/{session}/eeg/{stem}.set"
                        fdt_member = f"{subject}/{session}/eeg/{stem}.fdt"
                        meta = {
                            "dataset": "cog_bci",
                            "subject": subject,
                            "session": session,
                            "task": task,
                            "eeg_block": stem,
                            "source_file": f"{archive.name}:{set_member}",
                        }
                        if set_member not in names or fdt_member not in names:
                            rows.append({**meta, "modality": "eeg", "feature_status": "missing_eeglab_pair", "feature_error": ""})
                            continue
                        tmpdir = Path(tempfile.mkdtemp())
                        try:
                            set_path = tmpdir / set_member
                            fdt_path = tmpdir / fdt_member
                            set_path.parent.mkdir(parents=True, exist_ok=True)
                            set_path.write_bytes(zf.read(set_member))
                            fdt_path.write_bytes(zf.read(fdt_member))
                            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
                            rows.append(extract_eeg_features_from_raw(raw, meta))
                        except Exception as exc:
                            rows.append(
                                {
                                    **meta,
                                    "modality": "eeg",
                                    "feature_status": "failed",
                                    "feature_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        finally:
                            shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"STEP13_COG_EEG archive={archive_index}/{len(archives)} subject={subject}", flush=True)
    eeg = pd.DataFrame(rows)
    eeg.to_csv(cache, index=False)
    return eeg


def add_correct_num(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["correct_num"] = out["correct"].astype("boolean").astype(float)
    out["rt_num"] = pd.to_numeric(out["rt"], errors="coerce")
    out["trial_num"] = pd.to_numeric(out["trial_index"], errors="coerce")
    return out


def behavioral_summaries(events: pd.DataFrame) -> pd.DataFrame:
    data = add_correct_num(events[events["event_included"].astype(bool)]).copy()
    rows = []
    for keys, group in data.groupby(["dataset", "subject", "session", "task"], dropna=False):
        group = group.sort_values(["trial_num", "timestamp"], na_position="last")
        y = group["correct_num"].to_numpy(dtype=float)
        rt = group["rt_num"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        rt_ok = rt[np.isfinite(rt)]
        if len(y) == 0:
            continue
        n_third = max(len(y) // 3, 1)
        rows.append(
            {
                "dataset": keys[0],
                "subject": keys[1],
                "session": keys[2],
                "task": keys[3],
                "n_events": int(len(y)),
                "mean_accuracy": float(np.mean(y)),
                "lapse_rate": float(1.0 - np.mean(y)),
                "rt_median_behavior": float(np.nanmedian(rt_ok)) if len(rt_ok) else np.nan,
                "rt_iqr_behavior": float(np.nanpercentile(rt_ok, 75) - np.nanpercentile(rt_ok, 25)) if len(rt_ok) else np.nan,
                "rt_cv_behavior": float(np.nanstd(rt_ok) / max(np.nanmean(rt_ok), 1e-9)) if len(rt_ok) else np.nan,
                "early_late_accuracy_delta_behavior": float(np.mean(y[-n_third:]) - np.mean(y[:n_third])),
            }
        )
    return pd.DataFrame(rows)


def load_coordinates() -> tuple[pd.DataFrame, pd.DataFrame]:
    state = pd.read_csv(TABLES / "session_state_multiaxis_coordinates.csv")
    capacity = pd.read_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv")
    state = state[state["dataset"].eq("cog_bci")].copy()
    capacity = capacity[capacity["dataset"].eq("cog_bci")].copy()
    return state, capacity


def build_coordinate_table(events: pd.DataFrame, eeg: pd.DataFrame) -> pd.DataFrame:
    state, capacity = load_coordinates()
    behavior = behavioral_summaries(events)
    coord = state.merge(capacity, on=["dataset", "subject"], how="left", suffixes=("", "_capacity"))
    coord = coord.merge(behavior, on=["dataset", "subject", "session", "task"], how="left", suffixes=("", "_behavior"))
    eeg_ok = eeg[eeg["feature_status"].eq("ok")].copy()
    eeg_task = (
        eeg_ok.groupby(["dataset", "subject", "session", "task"], dropna=False)
        .agg(
            cog_eeg_theta_alpha_ratio=("cog_eeg_theta_alpha_ratio", "mean"),
            cog_eeg_relative_theta=("cog_eeg_relative_theta", "mean"),
            cog_eeg_relative_alpha=("cog_eeg_relative_alpha", "mean"),
            cog_eeg_spectral_entropy=("cog_eeg_spectral_entropy", "mean"),
            cog_eeg_aperiodic_slope=("cog_eeg_aperiodic_slope", "mean"),
            cog_eeg_channel_variability=("cog_eeg_channel_variability", "mean"),
            eeg_blocks=("eeg_block", "nunique"),
        )
        .reset_index()
    )
    coord = coord.merge(eeg_task, on=["dataset", "subject", "session", "task"], how="left")
    coord.to_csv(TABLES / "cog_bci_coordinates.csv", index=False)
    return coord


def spearman_row(df: pd.DataFrame, x: str, y: str, family: str, controls: str = "none") -> dict[str, Any]:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 8 or subset[x].nunique() < 3 or subset[y].nunique() < 3:
        return {
            "analysis_family": family,
            "x": x,
            "y": y,
            "controls": controls,
            "n": int(len(subset)),
            "estimate": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "claim_status": "not_eligible_low_variance_or_n",
        }
    rho, p = stats.spearmanr(subset[x], subset[y])
    return {
        "analysis_family": family,
        "x": x,
        "y": y,
        "controls": controls,
        "n": int(len(subset)),
        "estimate": float(rho),
        "p_value": float(p),
        "r_squared": np.nan,
        "claim_status": "tested",
    }


def ols_rows(df: pd.DataFrame, outcome: str, family: str) -> list[dict[str, Any]]:
    predictors = ["state_multidimensional_summary_z", "capacity_multidimensional_summary_z"]
    required = [outcome, "subject", "session", "task"] + predictors
    subset = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 20 or subset[outcome].nunique() < 3:
        return [
            {
                "analysis_family": family,
                "x": predictor,
                "y": outcome,
                "controls": "session+task+other_coordinate",
                "n": int(len(subset)),
                "estimate": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "claim_status": "not_eligible_low_variance_or_n",
            }
            for predictor in ["state", "capacity"]
        ]
    design = pd.get_dummies(subset[["session", "task"]].astype("string"), drop_first=True, dtype=float)
    for predictor in predictors:
        design[predictor] = subset[predictor].to_numpy(dtype=float)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(subset[outcome].astype(float), design).fit(cov_type="HC3")
    rows = []
    for label, predictor in [("state", predictors[0]), ("capacity", predictors[1])]:
        rows.append(
            {
                "analysis_family": family,
                "x": label,
                "y": outcome,
                "controls": "session+task+other_coordinate",
                "n": int(len(subset)),
                "estimate": float(model.params.get(predictor, np.nan)),
                "p_value": float(model.pvalues.get(predictor, np.nan)),
                "r_squared": float(model.rsquared),
                "claim_status": "tested",
            }
        )
    return rows


def variance_decomposition(coord: pd.DataFrame) -> pd.DataFrame:
    rows = []
    state_col = "state_multidimensional_summary_z"
    subject_means = coord.groupby("subject")[state_col].mean()
    within = coord.groupby("subject")[state_col].std(ddof=1).dropna()
    rows.append(
        {
            "analysis_family": "state_within_between_variance",
            "x": "within_subject_sd",
            "y": state_col,
            "controls": "none",
            "n": int(within.shape[0]),
            "estimate": float(within.mean()),
            "p_value": np.nan,
            "r_squared": np.nan,
            "claim_status": "descriptive",
        }
    )
    rows.append(
        {
            "analysis_family": "state_within_between_variance",
            "x": "between_subject_sd",
            "y": state_col,
            "controls": "none",
            "n": int(subject_means.shape[0]),
            "estimate": float(subject_means.std(ddof=1)),
            "p_value": np.nan,
            "r_squared": np.nan,
            "claim_status": "descriptive",
        }
    )
    return pd.DataFrame(rows)


def run_validation_models(coord: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(variance_decomposition(coord).to_dict("records"))
    rows.append(
        spearman_row(
            coord.drop_duplicates("subject"),
            "capacity_multidimensional_summary_z",
            "capacity_cross_task_consistency_axis",
            "capacity_cross_task_consistency",
        )
    )
    for outcome in [
        "mean_accuracy",
        "lapse_rate",
        "rt_cv_behavior",
        "rt_iqr_behavior",
        "cog_eeg_theta_alpha_ratio",
        "cog_eeg_relative_theta",
        "cog_eeg_spectral_entropy",
        "cog_eeg_aperiodic_slope",
        "cog_eeg_channel_variability",
    ]:
        if outcome in coord.columns:
            family = "cog_bci_behavior_coordinate_model" if not outcome.startswith("cog_eeg") else "cog_bci_eeg_coordinate_model"
            rows.extend(ols_rows(coord, outcome, family))
    models = pd.DataFrame(rows)
    mask = models["p_value"].notna()
    if mask.any():
        p = models.loc[mask, "p_value"].to_numpy(dtype=float)
        order = np.argsort(p)
        q = np.empty_like(p)
        ranked = p[order]
        m = len(p)
        running = 1.0
        for i in range(m - 1, -1, -1):
            running = min(running, ranked[i] * m / (i + 1))
            q[order[i]] = running
        models.loc[mask, "q_value"] = q
    else:
        models["q_value"] = np.nan
    models.to_csv(TABLES / "cog_bci_validation_models.csv", index=False)
    return models


def make_figure(coord: pd.DataFrame, models: pd.DataFrame) -> None:
    coord.to_csv(SOURCE_DATA / "figure_cog_bci_validation_source.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    state_col = "state_multidimensional_summary_z"
    coord.boxplot(column=state_col, by="task", ax=axes[0], grid=False, color="#333333")
    axes[0].set_title("State by task")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("State coordinate")
    fig.suptitle("")

    subj = coord.drop_duplicates("subject")
    axes[1].scatter(
        subj["capacity_multidimensional_summary_z"],
        subj["capacity_cross_task_consistency_axis"],
        s=18,
        color="#4C78A8",
        alpha=0.85,
    )
    axes[1].set_xlabel("Capacity")
    axes[1].set_ylabel("Cross-task consistency")

    plot_models = models[models["p_value"].notna()].copy()
    plot_models["neglog10_p"] = -np.log10(plot_models["p_value"].clip(lower=1e-300))
    plot_models = plot_models.sort_values("neglog10_p", ascending=False).head(8)
    labels = plot_models["analysis_family"].str.replace("cog_bci_", "", regex=False) + "\n" + plot_models["x"] + " -> " + plot_models["y"]
    axes[2].barh(labels, plot_models["neglog10_p"], color="#54A24B")
    axes[2].axvline(-math.log10(0.05), color="#666666", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("-log10 p")
    axes[2].set_title("Validation models")
    axes[2].grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_cog_bci_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_audit(events: pd.DataFrame, eeg: pd.DataFrame, coord: pd.DataFrame, models: pd.DataFrame) -> None:
    tested = models[models["claim_status"].eq("tested")]
    significant = tested[(tested["q_value"].notna()) & (tested["q_value"] < 0.05)]
    lines = [
        "# Step 13 COG-BCI Validation Audit",
        "",
        f"- Processed behavioral events: {len(events)}.",
        f"- Subjects: {events['subject'].nunique()}; sessions: {events['session'].nunique()}; tasks: {events['task'].nunique()}.",
        f"- EEG feature rows: {len(eeg)}; successful EEG rows: {int(eeg['feature_status'].eq('ok').sum())}.",
        f"- Coordinate rows: {len(coord)}.",
        f"- Tested validation rows: {len(tested)}; BH-significant rows: {len(significant)}.",
        "",
        "## Claim Boundary",
        "",
        "- COG-BCI validation now uses PVT behavior plus N-back/Flanker trials reconstructed from EEGLAB event markers.",
        "- ECG and subjective workload were not available in the local COG-BCI files parsed here; Step 13 validates behavior and EEG.",
        "- Capacity test-retest is limited because the current Step 08 capacity coordinate is participant-level rather than session-level.",
    ]
    (AUDIT / "step13_cog_bci_validation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    events = pd.read_parquet(PROCESSED / "cog_bci_model_events.parquet")
    eeg = extract_cog_eeg_features()
    coord = build_coordinate_table(events, eeg)
    models = run_validation_models(coord)
    make_figure(coord, models)
    write_audit(events, eeg, coord, models)
    status = {
        "status": "implemented_and_run_full_cog_bci_validation",
        "behavior_events": int(len(events)),
        "subjects": int(events["subject"].nunique()),
        "sessions": int(events["session"].nunique()),
        "tasks": int(events["task"].nunique()),
        "eeg_rows": int(len(eeg)),
        "eeg_ok": int(eeg["feature_status"].eq("ok").sum()),
        "coordinate_rows": int(len(coord)),
        "validation_model_rows": int(len(models)),
    }
    (LOGS / "step13_cog_bci_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STEP13_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
