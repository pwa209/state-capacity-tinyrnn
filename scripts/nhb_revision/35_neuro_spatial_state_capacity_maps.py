from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import append_manifest, append_registry, bh_q


ANALYSIS_ID = "nhb_35_neuro_spatial_state_capacity_maps"
SCRIPT_NAME = "scripts/nhb_revision/35_neuro_spatial_state_capacity_maps.py"

RAW = ROOT / "data" / "raw" / "openneuro" / "ds007554"
TABLES = ROOT / "outputs" / "tables"
OUT_DIR = ROOT / "outputs" / "nhb_revision" / "neuro_spatial_maps"

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

PRIMARY_BANDS = ["theta", "alpha", "beta"]
PREDICTORS = {
    "state": "state_multidimensional_summary_z",
    "capacity": "capacity_multidimensional_summary_z",
}


try:
    import pyedflib

    HAVE_PYEDFLIB = True
    PYEDFLIB_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on project extras
    HAVE_PYEDFLIB = False
    PYEDFLIB_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import mne

    HAVE_MNE = True
    MNE_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on project extras
    HAVE_MNE = False
    MNE_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import surfplot  # noqa: F401

    HAVE_SURFPLOT = True
    SURFPLOT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional visual package
    HAVE_SURFPLOT = False
    SURFPLOT_ERROR = f"{type(exc).__name__}: {exc}"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_bids(path: Path) -> dict[str, str]:
    text = path.as_posix()
    out: dict[str, str] = {}
    for key in ["sub", "ses", "task", "run", "recording"]:
        match = re.search(rf"{key}-([A-Za-z0-9]+)", text)
        if match:
            out[key] = match.group(1)
    return out


def normalize_channel_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip()).upper()


def read_edf(path: Path) -> tuple[list[str], np.ndarray, float]:
    if not HAVE_PYEDFLIB:
        raise RuntimeError(f"pyedflib unavailable: {PYEDFLIB_ERROR}")
    reader = pyedflib.EdfReader(str(path))
    try:
        n_signals = reader.signals_in_file
        labels = [str(x).strip() for x in reader.getSignalLabels()]
        rates = np.asarray(reader.getSampleFrequencies(), dtype=float)
        signals = [np.asarray(reader.readSignal(i), dtype=float) for i in range(n_signals)]
    finally:
        reader.close()
    min_len = min(len(x) for x in signals)
    data = np.vstack([x[:min_len] for x in signals])
    sfreq = float(np.nanmedian(rates))
    keep = np.isfinite(data).mean(axis=1) > 0.95
    data = data[keep]
    labels = [label for label, ok in zip(labels, keep) if ok]
    return labels, data, sfreq


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_channel_bandpower(path: Path) -> list[dict[str, Any]]:
    entities = parse_bids(path)
    labels, data, sfreq = read_edf(path)
    data = data - np.nanmedian(data, axis=1, keepdims=True)
    nperseg = max(32, min(int(round(sfreq * 4)), data.shape[1]))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    rows: list[dict[str, Any]] = []
    total = np.asarray([band_power(freqs, p, 1.0, 40.0) for p in psd], dtype=float)
    for ch_i, label in enumerate(labels):
        channel = normalize_channel_name(label)
        if not channel or channel.upper() in {"STATUS", "TRIGGER", "STI", "ANNOTATIONS"}:
            continue
        for band, (low, high) in EEG_BANDS.items():
            power = band_power(freqs, psd[ch_i], low, high)
            rows.append(
                {
                    "dataset": "ds007554",
                    "subject": entities.get("sub"),
                    "session": entities.get("ses"),
                    "task": entities.get("task"),
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "channel": channel,
                    "band": band,
                    "sampling_frequency": sfreq,
                    "n_samples": int(data.shape[1]),
                    "band_power": power,
                    "log_band_power": math.log1p(power) if np.isfinite(power) and power >= 0 else np.nan,
                    "relative_band_power": power / total[ch_i] if np.isfinite(power) and np.isfinite(total[ch_i]) and total[ch_i] > 0 else np.nan,
                    "feature_status": "ok",
                }
            )
    return rows


def extract_or_load_bandpower(force_extract: bool = False) -> pd.DataFrame:
    path = OUT_DIR / "ds007554_eeg_channel_bandpower.csv"
    if path.exists() and not force_extract:
        return pd.read_csv(path, dtype={"subject": "string", "session": "string"})
    rows: list[dict[str, Any]] = []
    files = sorted(RAW.rglob("*_eeg.edf"))
    failures: list[dict[str, Any]] = []
    for i, edf_path in enumerate(files, start=1):
        try:
            rows.extend(extract_channel_bandpower(edf_path))
        except Exception as exc:
            entities = parse_bids(edf_path)
            failures.append(
                {
                    "dataset": "ds007554",
                    "subject": entities.get("sub"),
                    "session": entities.get("ses"),
                    "task": entities.get("task"),
                    "source_file": edf_path.relative_to(ROOT).as_posix(),
                    "feature_status": "failed",
                    "feature_error": f"{type(exc).__name__}: {exc}",
                }
            )
        if i % 50 == 0 or i == len(files):
            print(f"STEP35_EEG_CHANNELS {i}/{len(files)}", flush=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["subject"] = out["subject"].astype("string").str.zfill(3)
        out["session"] = out["session"].astype("string").str.zfill(2)
    out.to_csv(path, index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "ds007554_eeg_channel_bandpower_failures.csv", index=False)
    return out


def attach_coordinates(bandpower: pd.DataFrame) -> pd.DataFrame:
    state = pd.read_csv(TABLES / "session_state_multiaxis_coordinates.csv", dtype={"subject": "string", "session": "string"})
    capacity = pd.read_csv(TABLES / "participant_capacity_multidimensional_coordinates.csv", dtype={"subject": "string"})
    state_cols = [
        "dataset",
        "subject",
        "session",
        "task",
        "state_multidimensional_summary_z",
        "state_lapse_axis_z",
        "state_drift_axis_z",
        "state_variability_axis_z",
        "state_reliability_axis_z",
        "state_estimation_quality",
    ]
    capacity_cols = [
        "dataset",
        "subject",
        "capacity_multidimensional_summary_z",
        "capacity_hidden_size_axis_z_z",
        "capacity_selection_confidence_z",
        "capacity_load_robustness_axis_z",
        "capacity_cross_task_consistency_axis_z",
        "capacity_information_quality",
    ]
    state = state[state["dataset"].eq("ds007554")][[c for c in state_cols if c in state.columns]].copy()
    capacity = capacity[capacity["dataset"].eq("ds007554")][[c for c in capacity_cols if c in capacity.columns]].copy()
    for frame in [bandpower, state, capacity]:
        if "subject" in frame:
            frame["subject"] = frame["subject"].astype("string").str.zfill(3)
        if "session" in frame:
            frame["session"] = frame["session"].astype("string").str.zfill(2)
        if "task" in frame:
            frame["task"] = frame["task"].astype("string")
    merged = bandpower.merge(state, on=["dataset", "subject", "session", "task"], how="left")
    merged = merged.merge(capacity, on=["dataset", "subject"], how="left")
    merged["coordinate_status"] = np.where(
        merged["state_multidimensional_summary_z"].notna() & merged["capacity_multidimensional_summary_z"].notna(),
        "state_capacity_coordinates_attached",
        "missing_state_or_capacity_coordinate",
    )
    return merged


def zscore(values: pd.Series) -> pd.Series:
    y = pd.to_numeric(values, errors="coerce").astype(float)
    sd = y.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return y * 0.0
    return (y - y.mean()) / sd


def fit_channel_models(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = data[data["coordinate_status"].eq("state_capacity_coordinates_attached")].copy()
    for band, band_df in eligible.groupby("band", dropna=False):
        for channel, group in band_df.groupby("channel", dropna=False):
            cols = ["log_band_power", "subject", "session", "task"] + list(PREDICTORS.values())
            g = group[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if len(g) < 40 or g["subject"].nunique() < 8 or g["log_band_power"].nunique() < 5:
                continue
            y = zscore(g["log_band_power"])
            design = pd.DataFrame(index=g.index)
            for _, col in PREDICTORS.items():
                design[col] = zscore(g[col])
            dummies = pd.get_dummies(g[["task", "session"]].astype("string"), drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
            design = sm.add_constant(design, has_constant="add")
            try:
                model = sm.OLS(y, design).fit(cov_type="HC3")
            except Exception as exc:
                rows.append(
                    {
                        "dataset": "ds007554",
                        "channel": channel,
                        "band": band,
                        "predictor": "model",
                        "n_rows": len(g),
                        "n_subjects": g["subject"].nunique(),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for label, predictor_col in PREDICTORS.items():
                rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "script_name": SCRIPT_NAME,
                        "dataset": "ds007554",
                        "modality": "eeg",
                        "channel": channel,
                        "band": band,
                        "outcome": "log_band_power_z",
                        "predictor": label,
                        "predictor_column": predictor_col,
                        "n_rows": int(len(g)),
                        "n_subjects": int(g["subject"].nunique()),
                        "estimate": float(model.params.get(predictor_col, np.nan)),
                        "std_error": float(model.bse.get(predictor_col, np.nan)),
                        "p_value": float(model.pvalues.get(predictor_col, np.nan)),
                        "r_squared": float(model.rsquared),
                        "control_status": "task_session_adjusted_hc3",
                        "claim_strength": "spatial_exploratory",
                        "interpretation": "EEG sensor-space association; not source-localized cortical evidence.",
                        "status": "ok",
                        "error": "",
                    }
                )
    models = pd.DataFrame(rows)
    if not models.empty and "p_value" in models:
        ok = models["status"].eq("ok")
        models.loc[ok, "q_value"] = bh_q(models.loc[ok, "p_value"])
        models.loc[ok & (models["q_value"] < 0.05), "claim_strength"] = "sensor_space_fdr"
        models["effect_direction"] = np.where(models["estimate"] > 0, "positive", np.where(models["estimate"] < 0, "negative", "zero"))
    return models


def montage_table(channels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not HAVE_MNE:
        return pd.DataFrame(
            [{"channel": ch, "montage_status": "missing_mne", "x": np.nan, "y": np.nan, "z": np.nan} for ch in channels]
        )
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = {normalize_channel_name(k): np.asarray(v, dtype=float) for k, v in montage.get_positions()["ch_pos"].items()}
    for ch in channels:
        pos = ch_pos.get(normalize_channel_name(ch))
        if pos is None:
            rows.append({"channel": ch, "montage_status": "missing_standard_1005_position", "x": np.nan, "y": np.nan, "z": np.nan})
        else:
            rows.append({"channel": ch, "montage_status": "standard_1005_position", "x": pos[0], "y": pos[1], "z": pos[2]})
    return pd.DataFrame(rows)


def make_topomap_figure(models: pd.DataFrame, positions: pd.DataFrame) -> tuple[Path | None, str]:
    if not HAVE_MNE:
        return None, f"blocked_missing_mne: {MNE_ERROR}"
    ok = models[models["status"].eq("ok")].copy()
    ok = ok[ok["band"].isin(PRIMARY_BANDS) & ok["predictor"].isin(PREDICTORS)]
    pos_ok = positions[positions["montage_status"].eq("standard_1005_position")].copy()
    channels = [ch for ch in sorted(ok["channel"].dropna().unique()) if ch in set(pos_ok["channel"])]
    if len(channels) < 8:
        return None, "blocked_too_few_channels_with_standard_positions"

    montage = mne.channels.make_standard_montage("standard_1005")
    # Restore common capitalization for MNE while keeping model joins uppercase.
    montage_names = {normalize_channel_name(ch): ch for ch in montage.ch_names}
    mne_names = [montage_names.get(normalize_channel_name(ch), ch) for ch in channels]
    info = mne.create_info(mne_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), constrained_layout=True)
    vmax = np.nanmax(np.abs(ok.loc[ok["channel"].isin(channels), "estimate"].to_numpy(dtype=float)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.25
    vmax = max(vmax, 0.05)
    for row_i, predictor in enumerate(["state", "capacity"]):
        for col_i, band in enumerate(PRIMARY_BANDS):
            ax = axes[row_i, col_i]
            sub = ok[(ok["predictor"].eq(predictor)) & (ok["band"].eq(band))].set_index("channel")
            values = np.asarray([sub["estimate"].get(ch, np.nan) for ch in channels], dtype=float)
            mask = np.asarray([bool(sub["q_value"].get(ch, np.nan) < 0.05) for ch in channels], dtype=bool)
            im, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                contours=0,
                sensors=True,
                mask=mask,
                mask_params={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "black", "linewidth": 0, "markersize": 7},
            )
            ax.set_title(f"{predictor.capitalize()} - {band}", fontsize=10)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Task/session-adjusted beta (z outcome)", fontsize=9)
    fig.suptitle("Sensor-space EEG state-capacity maps (no FDR-significant sensors)", fontsize=13, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"fig_neuro_spatial_eeg_topomaps.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUT_DIR / "fig_neuro_spatial_eeg_topomaps.png", "created"


def fnirs_surface_eligibility() -> pd.DataFrame:
    snirf_files = sorted(RAW.rglob("*_nirs.snirf"))
    has_bids_coords = bool(
        list(RAW.rglob("*_coordsystem.json")) or list(RAW.rglob("*_optodes.tsv")) or list(RAW.rglob("*_electrodes.tsv"))
    )
    rows: list[dict[str, Any]] = []
    checked = 0
    raw_loc_channels = 0
    raw_loc_finite = 0
    if HAVE_MNE:
        for path in snirf_files[:10]:
            try:
                raw = mne.io.read_raw_snirf(str(path), preload=False, verbose="ERROR")
                checked += 1
                for ch in raw.info.get("chs", []):
                    raw_loc_channels += 1
                    loc = np.asarray(ch.get("loc", []), dtype=float)
                    if loc.size >= 3 and np.isfinite(loc[:3]).all() and np.linalg.norm(loc[:3]) > 1e-9:
                        raw_loc_finite += 1
            except Exception:
                continue
    surface_eligible = bool(has_bids_coords or raw_loc_finite > 0)
    rows.append(
        {
            "dataset": "ds007554",
            "modality": "fnirs",
            "snirf_files": len(snirf_files),
            "checked_snirf_files": checked,
            "bids_coordsystem_or_optodes_file_present": has_bids_coords,
            "channels_checked_for_locations": raw_loc_channels,
            "channels_with_finite_nonzero_locations": raw_loc_finite,
            "surfplot_available": HAVE_SURFPLOT,
            "mne_available": HAVE_MNE,
            "surface_projection_eligible": surface_eligible,
            "eligibility_status": "eligible_for_surface_projection" if surface_eligible else "not_eligible_no_optode_or_source_space_coordinates",
            "interpretation": (
                "fNIRS surface projection can be attempted with available coordinates."
                if surface_eligible
                else "Dataset supports fNIRS feature analysis, but not honest surfplot cortical projection without optode/source-space coordinates."
            ),
            "surfplot_error": SURFPLOT_ERROR,
            "mne_error": MNE_ERROR,
        }
    )
    return pd.DataFrame(rows)


def make_source_workbook(
    bandpower: pd.DataFrame,
    models: pd.DataFrame,
    topomap_source: pd.DataFrame,
    positions: pd.DataFrame,
    eligibility: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path | None:
    path = OUT_DIR / "neuro_spatial_state_capacity_source_data.xlsx"
    try:
        import openpyxl  # noqa: F401

        engine = "openpyxl"
    except Exception as exc:
        skip_path = OUT_DIR / "neuro_spatial_state_capacity_source_data_workbook_skipped.md"
        skip_path.write_text(
            "\n".join(
                [
                    "# Neuro-Spatial Workbook Skipped",
                    "",
                    f"The CSV source tables were written, but the Excel workbook was skipped in this Python environment because `openpyxl` is unavailable: {type(exc).__name__}: {exc}",
                    "",
                    "Use a Python environment with pandas and openpyxl to combine the CSV source tables into `neuro_spatial_state_capacity_source_data.xlsx`.",
                ]
            ),
            encoding="utf-8",
        )
        return None
    with pd.ExcelWriter(path, engine=engine) as writer:
        bandpower.to_excel(writer, sheet_name="eeg_channel_bandpower", index=False)
        models.to_excel(writer, sheet_name="eeg_channel_models", index=False)
        topomap_source.to_excel(writer, sheet_name="topomap_source", index=False)
        positions.to_excel(writer, sheet_name="eeg_montage_positions", index=False)
        eligibility.to_excel(writer, sheet_name="surface_eligibility", index=False)
        summary.to_excel(writer, sheet_name="run_summary", index=False)
    return path


def write_plan() -> Path:
    path = OUT_DIR / "step35_neuro_spatial_visualization_plan.md"
    path.write_text(
        "\n".join(
            [
                "# Step 35: Neuro-Spatial State-Capacity Visualization",
                "",
                "Purpose: test whether the already-estimated state and capacity profiles have spatially structured neurophysiology signatures.",
                "",
                "Concrete workflow:",
                "1. Reuse ds007554 raw EEG EDF files and repaired state/capacity coordinates from the existing analysis.",
                "2. Extract per-channel EEG band powers for delta, theta, alpha and beta bands.",
                "3. Attach session-level state coordinates and participant-level capacity coordinates.",
                "4. Fit channel-wise models: log band power ~ state + capacity + task + session, with HC3 robust errors.",
                "5. Correct the channel/band/predictor map with Benjamini-Hochberg FDR.",
                "6. Draw EEG scalp topographies for state and capacity effects in theta, alpha and beta bands.",
                "7. Audit fNIRS/surfplot eligibility from BIDS/SNIRF geometry metadata. Do not draw cortical surfaces unless geometry exists.",
                "8. Export every figure source table, model table, spatial metadata table, Excel workbook and a concise interpretation report.",
                "",
                "Claim boundary: maps are sensor-space visualizations unless explicit optode/source-space geometry is available. They should be written as exploratory spatial neurophysiology alignment, not causal brain localization of state or capacity.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def frame_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}" if np.isfinite(val) else "")
            else:
                vals.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    models: pd.DataFrame,
    eligibility: pd.DataFrame,
    fig_status: str,
) -> Path:
    path = OUT_DIR / "step35_neuro_spatial_visualization_report.md"
    ok = models[models["status"].eq("ok")].copy() if not models.empty else pd.DataFrame()
    sig = ok[ok.get("q_value", pd.Series(dtype=float)) < 0.05].copy() if not ok.empty else pd.DataFrame()
    top = ok.sort_values("p_value").head(12) if not ok.empty else pd.DataFrame()
    lines = [
        "# Step 35 Neuro-Spatial Visualization Report",
        "",
        "## Run Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Spatial Claim Boundary",
        "",
        "- EEG outputs are scalp sensor-space maps using standard 10-05 montage positions.",
        "- fNIRS surface/surfplot visualization is attempted only if optode/source-space geometry is available.",
        "- These outputs are exploratory neurophysiology alignment evidence, not causal source localization.",
        "",
        "## fNIRS/Surfplot Eligibility",
        "",
        frame_to_markdown(eligibility),
        "",
        f"EEG figure status: `{fig_status}`.",
        "",
        "## FDR-Supported Sensor Effects",
        "",
    ]
    if sig.empty:
        lines.append("No channel-level EEG state/capacity effects survived FDR correction in this exploratory map.")
    else:
        lines.append(frame_to_markdown(sig.sort_values("q_value"), max_rows=30))
    lines.extend(["", "## Strongest Nominal Sensor Effects", ""])
    lines.append(frame_to_markdown(top) if not top.empty else "No fitted channel models were available.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true", help="Reopen EDF files even if cached channel bandpower exists.")
    args = parser.parse_args()

    ensure_dirs()
    started = datetime.now(timezone.utc).isoformat()
    plan_path = write_plan()

    bandpower = extract_or_load_bandpower(force_extract=args.force_extract)
    attached = attach_coordinates(bandpower)
    attached_path = OUT_DIR / "ds007554_eeg_channel_bandpower_with_coordinates.csv"
    attached.to_csv(attached_path, index=False)

    models = fit_channel_models(attached)
    model_path = OUT_DIR / "ds007554_eeg_channel_spatial_models.csv"
    models.to_csv(model_path, index=False)

    positions = montage_table(sorted(attached["channel"].dropna().unique()))
    positions_path = OUT_DIR / "ds007554_eeg_standard1005_channel_positions.csv"
    positions.to_csv(positions_path, index=False)

    topomap_source = models.merge(positions, on="channel", how="left") if not models.empty else pd.DataFrame()
    topomap_path = OUT_DIR / "fig_neuro_spatial_eeg_topomap_source_data.csv"
    topomap_source.to_csv(topomap_path, index=False)

    eligibility = fnirs_surface_eligibility()
    eligibility_path = OUT_DIR / "fnirs_surfplot_eligibility.csv"
    eligibility.to_csv(eligibility_path, index=False)

    figure_path, fig_status = make_topomap_figure(models, positions)
    figure_outputs = [OUT_DIR / f"fig_neuro_spatial_eeg_topomaps.{suffix}" for suffix in ["png", "pdf", "svg"]]
    figure_outputs = [p for p in figure_outputs if p.exists()]

    ok_models = models[models["status"].eq("ok")] if not models.empty else pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "analysis_id": ANALYSIS_ID,
                "eeg_bandpower_rows": int(len(bandpower)),
                "eeg_recordings": int(attached["source_file"].nunique()) if "source_file" in attached else 0,
                "eeg_channels": int(attached["channel"].nunique()) if "channel" in attached else 0,
                "coordinate_attached_rows": int(attached["coordinate_status"].eq("state_capacity_coordinates_attached").sum()),
                "channel_models_ok": int(len(ok_models)),
                "fdr_supported_sensor_effects": int((ok_models["q_value"] < 0.05).sum()) if "q_value" in ok_models else 0,
                "mne_available": HAVE_MNE,
                "pyedflib_available": HAVE_PYEDFLIB,
                "surfplot_available": HAVE_SURFPLOT,
                "figure_status": fig_status,
            }
        ]
    )
    summary_path = OUT_DIR / "step35_run_summary.csv"
    summary.to_csv(summary_path, index=False)

    workbook_path = make_source_workbook(attached, models, topomap_source, positions, eligibility, summary)
    report_path = write_report(summary, models, eligibility, fig_status)

    outputs = [
        plan_path,
        attached_path,
        model_path,
        positions_path,
        topomap_path,
        eligibility_path,
        summary_path,
        report_path,
        *figure_outputs,
    ]
    if workbook_path is not None:
        outputs.insert(7, workbook_path)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        outputs,
        status="complete",
        notes="Exploratory EEG sensor-space state/capacity topographies; fNIRS/surfplot gated by geometry eligibility.",
    )
    print(json.dumps({"summary": summary.to_dict(orient="records")[0], "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
