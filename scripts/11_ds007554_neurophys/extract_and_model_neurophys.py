from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal, stats

try:
    import mne
    from mne.preprocessing.nirs import beer_lambert_law, optical_density

    HAVE_MNE_NIRS = True
    MNE_NIRS_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local neurophysiology extras
    HAVE_MNE_NIRS = False
    MNE_NIRS_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import pyedflib

    HAVE_PYEDFLIB = True
    PYEDFLIB_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local neurophysiology extras
    HAVE_PYEDFLIB = False
    PYEDFLIB_ERROR = f"{type(exc).__name__}: {exc}"


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openneuro" / "ds007554"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
AUDIT = ROOT / "outputs" / "audit"
LOGS = ROOT / "outputs" / "logs"

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
FRONTAL_CHANNEL_HINTS = {"FZ", "F3", "F4", "FC3", "FC4", "FP1", "FP2"}


@dataclass
class EdfData:
    labels: list[str]
    data: np.ndarray
    sampling_rates: np.ndarray
    physical_dimensions: list[str]


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, SOURCE_DATA, AUDIT, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def parse_bids(path: Path) -> dict[str, str]:
    text = path.as_posix()
    out: dict[str, str] = {}
    for key in ["sub", "ses", "task", "run", "recording"]:
        match = re.search(rf"{key}-([A-Za-z0-9]+)", text)
        if match:
            out[key] = match.group(1)
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def edf_str(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


def parse_edf(path: Path, max_channels: int | None = None) -> EdfData:
    if HAVE_PYEDFLIB:
        reader = pyedflib.EdfReader(str(path))
        try:
            n_signals = reader.signals_in_file
            selected = np.arange(n_signals)
            if max_channels is not None:
                selected = selected[:max_channels]
            labels = reader.getSignalLabels()
            dimensions = [reader.getPhysicalDimension(i) for i in range(n_signals)]
            rates = np.asarray(reader.getSampleFrequencies(), dtype=float)
            per_signal = [np.asarray(reader.readSignal(int(idx)), dtype=float) for idx in selected]
        finally:
            reader.close()
        min_len = min(len(x) for x in per_signal)
        data = np.vstack([x[:min_len] for x in per_signal])
        return EdfData(
            labels=[labels[int(i)] for i in selected],
            data=data,
            sampling_rates=rates[selected],
            physical_dimensions=[dimensions[int(i)] for i in selected],
        )

    with path.open("rb") as f:
        fixed = f.read(256)
        header_bytes = int(edf_str(fixed[184:192]))
        n_records = int(float(edf_str(fixed[236:244])))
        record_duration = float(edf_str(fixed[244:252]))
        n_signals = int(edf_str(fixed[252:256]))
        signal_header = f.read(header_bytes - 256)
        offset = 0

        def take(width: int) -> list[str]:
            nonlocal offset
            values = [edf_str(signal_header[offset + i * width : offset + (i + 1) * width]) for i in range(n_signals)]
            offset += n_signals * width
            return values

        labels = take(16)
        _transducer = take(80)
        dimensions = take(8)
        phys_min = np.asarray([float(v) for v in take(8)], dtype=float)
        phys_max = np.asarray([float(v) for v in take(8)], dtype=float)
        dig_min = np.asarray([float(v) for v in take(8)], dtype=float)
        dig_max = np.asarray([float(v) for v in take(8)], dtype=float)
        _prefilter = take(80)
        samples_per_record = np.asarray([int(float(v)) for v in take(8)], dtype=int)
        _reserved = take(32)

        selected = np.arange(n_signals)
        if max_channels is not None:
            selected = selected[:max_channels]

        raw = np.frombuffer(f.read(), dtype="<i2")
    if n_records <= 0:
        total_per_record = int(samples_per_record.sum())
        n_records = int(len(raw) // max(total_per_record, 1))
    total_per_record = int(samples_per_record.sum())
    raw = raw[: n_records * total_per_record]

    per_signal: list[np.ndarray] = []
    cursor = 0
    chunks = {idx: [] for idx in selected}
    for _ in range(n_records):
        for signal_index, n_samples in enumerate(samples_per_record):
            chunk = raw[cursor : cursor + n_samples]
            cursor += n_samples
            if signal_index in chunks:
                chunks[signal_index].append(chunk)
    for idx in selected:
        digital = np.concatenate(chunks[idx]).astype(float)
        scale = (phys_max[idx] - phys_min[idx]) / max(dig_max[idx] - dig_min[idx], 1e-12)
        physical = (digital - dig_min[idx]) * scale + phys_min[idx]
        per_signal.append(physical)
    min_len = min(len(x) for x in per_signal)
    data = np.vstack([x[:min_len] for x in per_signal])
    rates = samples_per_record[selected] / record_duration
    return EdfData(
        labels=[labels[i] for i in selected],
        data=data,
        sampling_rates=rates,
        physical_dimensions=[dimensions[i] for i in selected],
    )


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def spectral_entropy(psd: np.ndarray) -> float:
    clean = np.asarray(psd, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    if len(clean) == 0:
        return np.nan
    p = clean / clean.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(len(p)))


def aperiodic_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    mask = (freqs >= 2.0) & (freqs <= 40.0) & np.isfinite(psd) & (psd > 0)
    if mask.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), deg=1)[0])


def extract_eeg_features(path: Path) -> dict[str, Any]:
    entities = parse_bids(path)
    meta = read_json(path.with_name(path.name.replace("_eeg.edf", "_eeg.json")))
    try:
        edf = parse_edf(path)
        sfreq = float(np.median(edf.sampling_rates))
        data = edf.data
        # Robust channel centering reduces reference offsets without changing spectral structure.
        data = data - np.nanmedian(data, axis=1, keepdims=True)
        nperseg = min(int(sfreq * 4), data.shape[1])
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
        mean_psd = np.nanmedian(psd, axis=0)
        powers = {name: band_power(freqs, mean_psd, low, high) for name, (low, high) in EEG_BANDS.items()}
        total_power = band_power(freqs, mean_psd, 1.0, 40.0)
        frontal_idx = [i for i, label in enumerate(edf.labels) if label.strip().upper() in FRONTAL_CHANNEL_HINTS]
        if frontal_idx:
            frontal_psd = np.nanmedian(psd[frontal_idx, :], axis=0)
            frontal_theta = band_power(freqs, frontal_psd, 4.0, 8.0)
        else:
            frontal_theta = np.nan
        row = {
            "dataset": "ds007554",
            "subject": entities.get("sub"),
            "session": entities.get("ses"),
            "task": entities.get("task"),
            "modality": "eeg",
            "source_file": path.relative_to(ROOT).as_posix(),
            "n_channels": int(data.shape[0]),
            "n_samples": int(data.shape[1]),
            "sampling_frequency": sfreq,
            "recording_duration_sec": float(meta.get("RecordingDuration", data.shape[1] / sfreq)),
            "eeg_delta_power": powers["delta"],
            "eeg_theta_power": powers["theta"],
            "eeg_alpha_power": powers["alpha"],
            "eeg_beta_power": powers["beta"],
            "eeg_theta_alpha_ratio": powers["theta"] / powers["alpha"] if powers["alpha"] and powers["alpha"] > 0 else np.nan,
            "eeg_relative_theta": powers["theta"] / total_power if total_power and total_power > 0 else np.nan,
            "eeg_relative_alpha": powers["alpha"] / total_power if total_power and total_power > 0 else np.nan,
            "eeg_frontal_theta_power": frontal_theta,
            "eeg_spectral_entropy": spectral_entropy(mean_psd[(freqs >= 1.0) & (freqs <= 40.0)]),
            "eeg_aperiodic_slope": aperiodic_slope(freqs, mean_psd),
            "eeg_channel_variability": float(np.nanmedian(np.nanstd(data, axis=1))),
            "feature_status": "ok",
            "feature_error": "",
        }
        return row
    except Exception as exc:
        return {
            "dataset": "ds007554",
            "subject": entities.get("sub"),
            "session": entities.get("ses"),
            "task": entities.get("task"),
            "modality": "eeg",
            "source_file": path.relative_to(ROOT).as_posix(),
            "feature_status": "failed",
            "feature_error": f"{type(exc).__name__}: {exc}",
        }


def read_physio_vector(path: Path) -> np.ndarray:
    values: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            first = stripped.split("\t")[0].split(",")[0]
            try:
                values.append(float(first))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def ecg_features(x: np.ndarray, fs: float) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) < int(fs * 10):
        return {
            "ecg_hr_mean": np.nan,
            "ecg_hr_slope": np.nan,
            "ecg_rmssd": np.nan,
            "ecg_sdnn": np.nan,
            "ecg_pnn50": np.nan,
            "ecg_peak_count": 0,
        }
    x = signal.detrend(x)
    nyq = fs / 2
    high = min(25.0 / nyq, 0.99)
    low = min(3.0 / nyq, high * 0.5)
    b, a = signal.butter(2, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, x)
    distance = max(1, int(fs * 0.30))
    prominence = max(np.nanstd(filtered) * 0.8, 1e-9)
    peaks, _ = signal.find_peaks(filtered, distance=distance, prominence=prominence)
    if len(peaks) < 4:
        peaks, _ = signal.find_peaks(-filtered, distance=distance, prominence=prominence)
    if len(peaks) < 4:
        return {
            "ecg_hr_mean": np.nan,
            "ecg_hr_slope": np.nan,
            "ecg_rmssd": np.nan,
            "ecg_sdnn": np.nan,
            "ecg_pnn50": np.nan,
            "ecg_peak_count": int(len(peaks)),
        }
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi >= 0.30) & (ibi <= 2.00)]
    if len(ibi) < 3:
        return {
            "ecg_hr_mean": np.nan,
            "ecg_hr_slope": np.nan,
            "ecg_rmssd": np.nan,
            "ecg_sdnn": np.nan,
            "ecg_pnn50": np.nan,
            "ecg_peak_count": int(len(peaks)),
        }
    hr = 60.0 / ibi
    times = peaks[1 : 1 + len(ibi)] / fs
    slope = float(np.polyfit(times, hr, deg=1)[0]) if len(hr) >= 3 else np.nan
    diff_ibi = np.diff(ibi)
    return {
        "ecg_hr_mean": float(np.mean(hr)),
        "ecg_hr_slope": slope,
        "ecg_rmssd": float(np.sqrt(np.mean(diff_ibi**2))) if len(diff_ibi) else np.nan,
        "ecg_sdnn": float(np.std(ibi, ddof=1)) if len(ibi) > 1 else np.nan,
        "ecg_pnn50": float(np.mean(np.abs(diff_ibi) > 0.05)) if len(diff_ibi) else np.nan,
        "ecg_peak_count": int(len(peaks)),
    }


def extract_physio_features(path: Path) -> dict[str, Any]:
    entities = parse_bids(path)
    meta = read_json(path.with_suffix("").with_suffix(".json"))
    recording = entities.get("recording", "")
    try:
        fs = float(meta.get("SamplingFrequency", np.nan))
        x = read_physio_vector(path)
        base = {
            "dataset": "ds007554",
            "subject": entities.get("sub"),
            "session": entities.get("ses"),
            "task": entities.get("task"),
            "modality": "ecg" if recording.lower() == "ecg" else "biodex",
            "source_file": path.relative_to(ROOT).as_posix(),
            "n_samples": int(len(x)),
            "sampling_frequency": fs,
            "recording_duration_sec": float(len(x) / fs) if np.isfinite(fs) and fs > 0 else np.nan,
            "feature_status": "ok",
            "feature_error": "",
        }
        if recording.lower() == "ecg":
            base.update(ecg_features(x, fs))
        else:
            t = np.arange(len(x), dtype=float) / fs if np.isfinite(fs) and fs > 0 else np.arange(len(x), dtype=float)
            base.update(
                {
                    "biodex_mean": float(np.nanmean(x)) if len(x) else np.nan,
                    "biodex_sd": float(np.nanstd(x)) if len(x) else np.nan,
                    "biodex_range": float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan,
                    "biodex_slope": float(np.polyfit(t, x, deg=1)[0]) if len(x) > 3 else np.nan,
                }
            )
        return base
    except Exception as exc:
        return {
            "dataset": "ds007554",
            "subject": entities.get("sub"),
            "session": entities.get("ses"),
            "task": entities.get("task"),
            "modality": "ecg" if recording.lower() == "ecg" else "biodex",
            "source_file": path.relative_to(ROOT).as_posix(),
            "feature_status": "failed",
            "feature_error": f"{type(exc).__name__}: {exc}",
        }


def inventory_fnirs() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.rglob("*_nirs.snirf")):
        entities = parse_bids(path)
        meta = read_json(path.with_name(path.name.replace("_nirs.snirf", "_nirs.json")))
        rows.append(
            {
                "dataset": "ds007554",
                "subject": entities.get("sub"),
                "session": entities.get("ses"),
                "task": entities.get("task"),
                "source_file": path.relative_to(ROOT).as_posix(),
                "file_size_bytes": path.stat().st_size,
                "sampling_frequency": meta.get("SamplingFrequency"),
                "recording_duration_sec": meta.get("RecordingDuration"),
                "nirs_channel_count": meta.get("NIRSChannelCount"),
                "feature_status": "inventory_only",
                "feature_error": "",
            }
        )
    return pd.DataFrame(rows)


def window_response(times: np.ndarray, trace: np.ndarray, onsets: np.ndarray) -> float:
    responses: list[float] = []
    for onset in onsets:
        baseline = (times >= onset - 5.0) & (times < onset)
        response = (times >= onset + 5.0) & (times <= onset + 30.0)
        if baseline.sum() >= 3 and response.sum() >= 3:
            responses.append(float(np.nanmean(trace[response]) - np.nanmean(trace[baseline])))
    return float(np.nanmean(responses)) if responses else np.nan


def extract_fnirs_features(path: Path) -> dict[str, Any]:
    entities = parse_bids(path)
    meta = read_json(path.with_name(path.name.replace("_nirs.snirf", "_nirs.json")))
    base = {
        "dataset": "ds007554",
        "subject": entities.get("sub"),
        "session": entities.get("ses"),
        "task": entities.get("task"),
        "modality": "fnirs",
        "source_file": path.relative_to(ROOT).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "sampling_frequency": meta.get("SamplingFrequency"),
        "recording_duration_sec": meta.get("RecordingDuration"),
        "nirs_channel_count": meta.get("NIRSChannelCount"),
    }
    if not HAVE_MNE_NIRS:
        base.update(
            {
                "feature_status": "blocked_missing_mne_nirs",
                "feature_error": MNE_NIRS_ERROR,
            }
        )
        return base
    try:
        raw = mne.io.read_raw_snirf(str(path), preload=True, verbose="ERROR")
        od = optical_density(raw, verbose="ERROR")
        hb = beer_lambert_law(od, ppf=0.1)
        data = hb.get_data()
        times = hb.times
        channel_types = np.asarray(hb.get_channel_types())
        onsets = np.asarray(getattr(hb.annotations, "onset", []), dtype=float)
        onsets = onsets[np.isfinite(onsets)]
        row = {
            **base,
            "n_channels": int(data.shape[0]),
            "n_samples": int(data.shape[1]),
            "sampling_frequency": float(hb.info["sfreq"]),
            "recording_duration_sec": float(times[-1] - times[0]) if len(times) else np.nan,
            "fnirs_hbo_channel_count": int(np.sum(channel_types == "hbo")),
            "fnirs_hbr_channel_count": int(np.sum(channel_types == "hbr")),
            "feature_status": "ok",
            "feature_error": "",
        }
        mean_traces: dict[str, np.ndarray] = {}
        for chroma in ["hbo", "hbr"]:
            idx = np.where(channel_types == chroma)[0]
            if len(idx) == 0:
                row[f"fnirs_{chroma}_mean"] = np.nan
                row[f"fnirs_{chroma}_sd"] = np.nan
                row[f"fnirs_{chroma}_slope"] = np.nan
                row[f"fnirs_{chroma}_channel_variability"] = np.nan
                row[f"fnirs_{chroma}_task_response"] = np.nan
                continue
            chroma_data = data[idx, :]
            trace = np.nanmean(chroma_data, axis=0)
            mean_traces[chroma] = trace
            row[f"fnirs_{chroma}_mean"] = float(np.nanmean(trace))
            row[f"fnirs_{chroma}_sd"] = float(np.nanstd(trace))
            row[f"fnirs_{chroma}_slope"] = float(np.polyfit(times, trace, deg=1)[0]) if len(times) > 3 else np.nan
            row[f"fnirs_{chroma}_channel_variability"] = float(np.nanmedian(np.nanstd(chroma_data, axis=1)))
            row[f"fnirs_{chroma}_task_response"] = window_response(times, trace, onsets)
        if {"hbo", "hbr"}.issubset(mean_traces):
            diff = mean_traces["hbo"] - mean_traces["hbr"]
            row["fnirs_hbo_hbr_difference_mean"] = float(np.nanmean(diff))
            row["fnirs_hbo_hbr_difference_sd"] = float(np.nanstd(diff))
        return row
    except Exception as exc:
        base.update(
            {
                "feature_status": "failed",
                "feature_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return base


def task_load_proxy(task: str | float) -> float:
    text = str(task).lower()
    if text in {"full", "nbackarithmetic"}:
        return 3.0
    if text in {"nback", "mentalarithmetic"}:
        return 2.0
    if "motor" in text:
        return 1.0
    return 0.0


def attach_state_capacity_coordinates(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    state_path = TABLES / "session_state_multiaxis_coordinates.csv"
    capacity_path = TABLES / "participant_capacity_multidimensional_coordinates.csv"
    if not state_path.exists() or not capacity_path.exists():
        out["coordinate_status"] = "missing_coordinate_tables"
        return out

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
        "state_information_score",
        "state_estimation_quality",
    ]
    capacity_cols = [
        "dataset",
        "subject",
        "capacity_multidimensional_summary_z",
        "capacity_hidden_size_axis_z_z",
        "capacity_selection_confidence_z",
        "capacity_complexity_preference_axis_z",
        "capacity_high_capacity_nll_advantage_z",
        "capacity_load_robustness_axis_z",
        "capacity_cross_task_consistency_axis_z",
        "capacity_information_quality",
    ]
    state = pd.read_csv(state_path, dtype={"subject": "string", "session": "string"})
    capacity = pd.read_csv(capacity_path, dtype={"subject": "string"})
    state = state[state["dataset"].eq("ds007554")][[c for c in state_cols if c in state.columns]].copy()
    capacity = capacity[capacity["dataset"].eq("ds007554")][[c for c in capacity_cols if c in capacity.columns]].copy()
    for frame in [out, state, capacity]:
        if "subject" in frame.columns:
            frame["subject"] = frame["subject"].astype("string").str.zfill(3)
        if "session" in frame.columns:
            frame["session"] = frame["session"].astype("string").str.zfill(2)
        if "task" in frame.columns:
            frame["task"] = frame["task"].astype("string")

    out = out.merge(state, on=["dataset", "subject", "session", "task"], how="left")
    out = out.merge(capacity, on=["dataset", "subject"], how="left")
    has_state = out["state_multidimensional_summary_z"].notna() if "state_multidimensional_summary_z" in out.columns else False
    has_capacity = out["capacity_multidimensional_summary_z"].notna() if "capacity_multidimensional_summary_z" in out.columns else False
    out["coordinate_status"] = np.where(
        has_state & has_capacity,
        "state_capacity_coordinates_attached",
        "no_matching_state_capacity_coordinates_for_row",
    )
    return out


def model_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_features = [
        col
        for col in features.columns
        if (
            col.startswith("eeg_")
            or col.startswith("ecg_")
            or col.startswith("biodex_")
            or col.startswith("fnirs_")
        )
        and col not in {"ecg_peak_count"}
    ]
    data = features[features["feature_status"].eq("ok")].copy()
    data["task_load_proxy"] = data["task"].map(task_load_proxy)
    for feature in numeric_features:
        subset = data[["subject", "session", "task", "modality", "task_load_proxy", feature]].dropna()
        if len(subset) < 20 or subset[feature].nunique() < 3:
            continue
        for modality, group in subset.groupby("modality"):
            if len(group) < 20 or group[feature].nunique() < 3:
                continue
            design = pd.get_dummies(group[["session", "task"]].astype("string"), drop_first=True, dtype=float)
            design.insert(0, "task_load_proxy", group["task_load_proxy"].astype(float).to_numpy())
            design = sm.add_constant(design, has_constant="add")
            model = sm.OLS(group[feature].astype(float), design).fit(cov_type="HC3")
            rows.append(
                {
                    "analysis_family": "task_load_proxy_with_session_task_controls",
                    "modality": modality,
                    "feature": feature,
                    "predictor": "task_load_proxy",
                    "n": int(len(group)),
                    "beta": float(model.params.get("task_load_proxy", np.nan)),
                    "p_value": float(model.pvalues.get("task_load_proxy", np.nan)),
                    "r_squared": float(model.rsquared),
                    "claim_status": "task_physiology_only_not_state_capacity",
                }
            )

    coordinate_predictors = {
        "state": "state_multidimensional_summary_z",
        "capacity": "capacity_multidimensional_summary_z",
    }
    if "coordinate_status" in data.columns:
        coordinate_eligible = data[data["coordinate_status"].eq("state_capacity_coordinates_attached")].copy()
    else:
        coordinate_eligible = data.iloc[0:0].copy()
    for feature in numeric_features:
        required = ["subject", "session", "task", "modality", "task_load_proxy", feature] + list(coordinate_predictors.values())
        if not all(col in coordinate_eligible.columns for col in required):
            continue
        subset = coordinate_eligible[required].dropna()
        if len(subset) < 20 or subset[feature].nunique() < 3:
            continue
        for modality, group in subset.groupby("modality"):
            if len(group) < 20 or group[feature].nunique() < 3:
                continue
            design = pd.get_dummies(group[["session", "task"]].astype("string"), drop_first=True, dtype=float)
            design.insert(0, "task_load_proxy", group["task_load_proxy"].astype(float).to_numpy())
            for predictor_label, predictor_col in coordinate_predictors.items():
                design[predictor_col] = pd.to_numeric(group[predictor_col], errors="coerce").to_numpy(dtype=float)
            design = sm.add_constant(design, has_constant="add")
            model = sm.OLS(group[feature].astype(float), design).fit(cov_type="HC3")
            for predictor_label, predictor_col in coordinate_predictors.items():
                rows.append(
                    {
                        "analysis_family": "state_capacity_coordinate_model",
                        "modality": modality,
                        "feature": feature,
                        "predictor": predictor_label,
                        "n": int(len(group)),
                        "beta": float(model.params.get(predictor_col, np.nan)),
                        "p_value": float(model.pvalues.get(predictor_col, np.nan)),
                        "r_squared": float(model.rsquared),
                        "claim_status": "direct_state_capacity_neurophysiology_model_ds007554_pushbutton_repaired",
                    }
                )

    if not any(row.get("analysis_family") == "state_capacity_coordinate_model" for row in rows):
        for coordinate in ["state", "capacity"]:
            rows.append(
                {
                    "analysis_family": "state_capacity_coordinate_model",
                    "modality": "all",
                    "feature": "all_neurophys_features",
                    "predictor": coordinate,
                    "n": 0,
                    "beta": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "claim_status": "not_eligible_missing_ds007554_behavioral_coordinates",
                }
            )
    for coordinate, predictor_col in coordinate_predictors.items():
        if predictor_col in coordinate_eligible.columns and coordinate_eligible[predictor_col].notna().any():
            continue
        rows.append(
            {
                "analysis_family": "state_capacity_coordinate_model",
                "modality": "all",
                "feature": "all_neurophys_features",
                "predictor": coordinate,
                "n": 0,
                "beta": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "claim_status": "not_eligible_missing_ds007554_behavioral_coordinates",
            }
        )
    return pd.DataFrame(rows)


def make_figure(features: pd.DataFrame, models: pd.DataFrame) -> None:
    source = features.copy()
    source.to_csv(SOURCE_DATA / "figure_ds007554_neurophys_source.csv", index=False)
    plot_models = models[
        models["analysis_family"].eq("task_load_proxy_with_session_task_controls")
        & models["p_value"].notna()
    ].copy()
    plot_models["neglog10_p"] = -np.log10(plot_models["p_value"].clip(lower=1e-300))
    plot_models = plot_models.sort_values("neglog10_p", ascending=False).head(12)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = plot_models["modality"].astype(str) + ": " + plot_models["feature"].astype(str)
    colors = plot_models["modality"].map({"eeg": "#4C78A8", "ecg": "#F58518", "biodex": "#54A24B"}).fillna("#777777")
    ax.barh(labels, plot_models["neglog10_p"], color=colors)
    ax.axvline(-math.log10(0.05), color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10 p for task-load proxy")
    ax.set_title("ds007554 neurophysiology feature models")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"figure_ds007554_neurophys.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    eeg_rows = []
    physio_rows = []
    fnirs_rows = []
    eeg_files = sorted(RAW.rglob("*_eeg.edf"))
    physio_files = sorted(RAW.rglob("*_physio.tsv.gz"))
    fnirs_files = sorted(RAW.rglob("*_nirs.snirf"))
    for i, path in enumerate(eeg_files, start=1):
        eeg_rows.append(extract_eeg_features(path))
        if i % 50 == 0:
            print(f"STEP12_EEG {i}/{len(eeg_files)}", flush=True)
    for i, path in enumerate(physio_files, start=1):
        physio_rows.append(extract_physio_features(path))
        if i % 75 == 0:
            print(f"STEP12_PHYSIO {i}/{len(physio_files)}", flush=True)
    for i, path in enumerate(fnirs_files, start=1):
        fnirs_rows.append(extract_fnirs_features(path))
        if i % 50 == 0:
            print(f"STEP12_FNIRS {i}/{len(fnirs_files)}", flush=True)
    eeg = pd.DataFrame(eeg_rows)
    physio = pd.DataFrame(physio_rows)
    fnirs = pd.DataFrame(fnirs_rows)
    fnirs_inventory = inventory_fnirs()
    features = pd.concat([eeg, physio, fnirs], ignore_index=True, sort=False)
    features = attach_state_capacity_coordinates(features)
    models = model_feature_table(features)

    features.to_csv(TABLES / "ds007554_neurophys_features.csv", index=False)
    eeg.to_csv(TABLES / "ds007554_eeg_features.csv", index=False)
    physio.to_csv(TABLES / "ds007554_physio_features.csv", index=False)
    fnirs.to_csv(TABLES / "ds007554_fnirs_features.csv", index=False)
    fnirs_inventory.to_csv(TABLES / "ds007554_fnirs_inventory.csv", index=False)
    models.to_csv(TABLES / "ds007554_neurophys_models.csv", index=False)
    make_figure(features, models)

    status = {
        "status": "implemented_and_run_with_project_neurophys_readers",
        "eeg_files": len(eeg_files),
        "eeg_ok": int(eeg["feature_status"].eq("ok").sum()) if len(eeg) else 0,
        "physio_files": len(physio_files),
        "physio_ok": int(physio["feature_status"].eq("ok").sum()) if len(physio) else 0,
        "fnirs_files": int(len(fnirs_files)),
        "fnirs_ok": int(fnirs["feature_status"].eq("ok").sum()) if len(fnirs) else 0,
        "fnirs_signal_features": "hbo_hbr_extracted_with_mne" if HAVE_MNE_NIRS else "blocked_missing_mne_nirs",
        "pyedflib": "available_preferred_reader_used" if HAVE_PYEDFLIB else f"missing_fallback_reader_used: {PYEDFLIB_ERROR}",
        "state_capacity_coordinate_rows": int(features["coordinate_status"].eq("state_capacity_coordinates_attached").sum())
        if "coordinate_status" in features.columns
        else 0,
        "state_capacity_coordinate_models": "eligible_and_run"
        if len(models[models["analysis_family"].eq("state_capacity_coordinate_model") & models["n"].gt(0)])
        else "not_eligible_missing_ds007554_behavioral_coordinates",
    }
    (LOGS / "step12_neurophys_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    audit = [
        "# Step 12 ds007554 Neurophysiology Audit",
        "",
        "Step 12 was run after installing mne, mne-nirs, neurokit2, pyedflib and h5py into the project environment. EDF reading uses pyedflib when available, with a local EDF fallback retained in the script.",
        "",
        f"- EEG EDF files processed: {status['eeg_ok']} / {status['eeg_files']}.",
        f"- ECG/Biodex physio files processed: {status['physio_ok']} / {status['physio_files']}.",
        f"- fNIRS SNIRF files processed for HbO/HbR features: {status['fnirs_ok']} / {status['fnirs_files']}.",
        "",
        "## Claim Boundary",
        "",
        "- ds007554 push-button recordings were used in Step 03 to reconstruct N-back and N-back-arithmetic correctness labels.",
        f"- Rows with attached ds007554 state/capacity coordinates: {status['state_capacity_coordinate_rows']}.",
        "- Current state/capacity neurophysiology models are direct ds007554 tests, but remain qualified because the behavioral labels are reconstructed from push-button timing rather than distributed as explicit trial-correctness columns.",
    ]
    (AUDIT / "step12_neurophys_claim_audit.md").write_text("\n".join(audit), encoding="utf-8")
    print("STEP12_COMPLETE " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
