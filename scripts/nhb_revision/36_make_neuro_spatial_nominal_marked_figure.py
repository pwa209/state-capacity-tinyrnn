from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "nhb_revision" / "neuro_spatial_maps"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import append_manifest, append_registry


ANALYSIS_ID = "nhb_36_neuro_spatial_nominal_marked_figure"
SCRIPT_NAME = "scripts/nhb_revision/36_make_neuro_spatial_nominal_marked_figure.py"
PRIMARY_BANDS = ["theta", "alpha", "beta"]


def normalize_channel_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def main() -> None:
    import mne
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc).isoformat()
    models = pd.read_csv(OUT_DIR / "ds007554_eeg_channel_spatial_models.csv")
    positions = pd.read_csv(OUT_DIR / "ds007554_eeg_standard1005_channel_positions.csv")
    ok = models[models["status"].eq("ok")].copy()
    ok = ok[ok["band"].isin(PRIMARY_BANDS) & ok["predictor"].isin(["state", "capacity"])]
    ok["significance_marker"] = np.select(
        [ok["q_value"] < 0.05, ok["p_value"] < 0.05],
        ["FDR_q_lt_0_05", "nominal_p_lt_0_05"],
        default="not_marked",
    )
    source = ok.merge(positions, on="channel", how="left")
    source_path = OUT_DIR / "fig_neuro_spatial_eeg_topomaps_nominal_marked_source_data.csv"
    source.to_csv(source_path, index=False)

    pos_ok = positions[positions["montage_status"].eq("standard_1005_position")].copy()
    channels = [ch for ch in sorted(ok["channel"].dropna().unique()) if ch in set(pos_ok["channel"])]
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_names = {normalize_channel_name(ch): ch for ch in montage.ch_names}
    mne_names = [montage_names.get(normalize_channel_name(ch), ch) for ch in channels]
    info = mne.create_info(mne_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    vmax = np.nanmax(np.abs(ok.loc[ok["channel"].isin(channels), "estimate"].to_numpy(dtype=float)))
    vmax = max(float(vmax) if np.isfinite(vmax) else 0.25, 0.05)
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8), constrained_layout=True)
    last_im = None
    for row_i, predictor in enumerate(["state", "capacity"]):
        for col_i, band in enumerate(PRIMARY_BANDS):
            ax = axes[row_i, col_i]
            sub = ok[(ok["predictor"].eq(predictor)) & (ok["band"].eq(band))].set_index("channel")
            values = np.asarray([sub["estimate"].get(ch, np.nan) for ch in channels], dtype=float)
            fdr_mask = np.asarray([bool(sub["q_value"].get(ch, np.nan) < 0.05) for ch in channels], dtype=bool)
            nominal_mask = np.asarray([bool((sub["p_value"].get(ch, np.nan) < 0.05) and not (sub["q_value"].get(ch, np.nan) < 0.05)) for ch in channels], dtype=bool)
            marker_mask = nominal_mask | fdr_mask
            last_im, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                contours=0,
                sensors=False,
                mask=marker_mask,
                mask_params={
                    "marker": "o",
                    "linestyle": "None",
                    "markerfacecolor": "none",
                    "markeredgecolor": "black",
                    "color": "black",
                    "linewidth": 1.1,
                    "markersize": 7.5,
                },
            )
            ax.set_title(f"{predictor.capitalize()} - {band}", fontsize=10)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Task/session-adjusted beta (z outcome)", fontsize=9)
    fig.suptitle("EEG sensor maps with exploratory nominal markers", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "Open circles mark nominal p < .05 only; no sensor survived FDR q < .05. Stars would indicate FDR-significant sensors.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    outputs = [source_path]
    for suffix in ["png", "pdf", "svg"]:
        path = OUT_DIR / f"fig_neuro_spatial_eeg_topomaps_nominal_marked.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, outputs)
    append_registry(
        ANALYSIS_ID,
        SCRIPT_NAME,
        started,
        outputs,
        notes="Nominal p < .05 sensor markers added for visualization only; no FDR-significant EEG sensors.",
    )
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
