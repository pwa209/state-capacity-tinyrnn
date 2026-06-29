from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "nhb_revision" / "neuro_spatial_maps"


SHEETS = {
    "eeg_channel_bandpower": "ds007554_eeg_channel_bandpower_with_coordinates.csv",
    "eeg_channel_models": "ds007554_eeg_channel_spatial_models.csv",
    "topomap_source": "fig_neuro_spatial_eeg_topomap_source_data.csv",
    "eeg_montage_positions": "ds007554_eeg_standard1005_channel_positions.csv",
    "surface_eligibility": "fnirs_surfplot_eligibility.csv",
    "run_summary": "step35_run_summary.csv",
}


def main() -> None:
    output = OUT_DIR / "neuro_spatial_state_capacity_source_data.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, name in SHEETS.items():
            pd.read_csv(OUT_DIR / name).to_excel(writer, sheet_name=sheet, index=False)
    print(output)


if __name__ == "__main__":
    main()
