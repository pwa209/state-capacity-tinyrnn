from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "nhb_revision" / "placeholder_sensitivity"
DOC_AUDIT = ROOT / "outputs" / "nhb_revision" / "manuscript_placeholder_audit"
SOURCE_DOCX = Path(r"C:\Users\Gebruiker\Downloads\State_capacity_PW_revised.docx")
UPDATED_DOCX = OUT / "State_capacity_PW_revised_placeholders_resolved.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)

CAPACITY_COMPONENTS_MINUS_LOAD = [
    "capacity_hidden_size_axis_z_z",
    "capacity_selection_confidence_z",
    "capacity_complexity_preference_axis_z",
    "capacity_high_capacity_nll_advantage_z",
    "capacity_cross_task_consistency_axis_z",
]


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (x - x.mean()) / sd


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
) -> dict[str, object]:
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


def spearman_result(df: pd.DataFrame, x: str, y: str, label: str) -> dict[str, object]:
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        rho, p = np.nan, np.nan
    else:
        res = stats.spearmanr(data[x], data[y])
        rho, p = float(res.statistic), float(res.pvalue)
    return {"analysis": label, "x": x, "y": y, "n": int(len(data)), "spearman_rho": rho, "p_value": p}


def residualize_ranked(df: pd.DataFrame, target: str, controls: list[str], categorical: list[str]) -> pd.Series:
    data = df[[target] + controls].replace([np.inf, -np.inf], np.nan).copy()
    y = data[target].rank(method="average", na_option="keep")
    y_name = f"{target}_rank"
    data[y_name] = y
    data = data.dropna(subset=[y_name])
    x, _ = design_matrix(data, controls, categorical)
    beta = np.linalg.pinv(x) @ data[y_name].to_numpy(dtype=float)
    resid = data[y_name].to_numpy(dtype=float) - x @ beta
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[data.index] = resid
    return out


def partial_spearman(
    df: pd.DataFrame,
    x: str,
    y: str,
    controls: list[str],
    categorical: list[str],
    label: str,
) -> dict[str, object]:
    working = df[[x, y] + controls].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y]).copy()
    if len(working) < 8:
        return {"analysis": label, "x": x, "y": y, "n": int(len(working)), "spearman_rho": np.nan, "p_value": np.nan}
    working[f"{x}_resid_rank"] = residualize_ranked(working, x, controls, categorical)
    working[f"{y}_resid_rank"] = residualize_ranked(working, y, controls, categorical)
    data = working[[f"{x}_resid_rank", f"{y}_resid_rank"]].dropna()
    if len(data) < 8 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        rho, p = np.nan, np.nan
    else:
        res = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
        rho, p = float(res.statistic), float(res.pvalue)
    return {
        "analysis": label,
        "x": x,
        "y": y,
        "controls": "+".join(controls),
        "n": int(len(data)),
        "spearman_rho": rho,
        "p_value": p,
    }


def within_hidden_size_meta(df: pd.DataFrame, x: str, y: str) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for hidden_size, group in df.groupby("selected_hidden_size", dropna=False):
        data = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 8 or data[x].nunique() < 2 or data[y].nunique() < 2:
            rho, p = np.nan, np.nan
        else:
            res = stats.spearmanr(data[x], data[y])
            rho, p = float(res.statistic), float(res.pvalue)
        rows.append(
            {
                "selected_hidden_size": hidden_size,
                "n": int(len(data)),
                "spearman_rho": rho,
                "p_value": p,
            }
        )
    strata = pd.DataFrame(rows)
    meta_data = strata.dropna(subset=["spearman_rho"]).copy()
    meta_data = meta_data[meta_data["n"] > 3]
    if meta_data.empty:
        meta = {"analysis": f"{x}_vs_{y}_within_hidden_size_meta", "n_strata": 0, "spearman_rho": np.nan, "p_value": np.nan}
    else:
        clipped = np.clip(meta_data["spearman_rho"].to_numpy(dtype=float), -0.999999, 0.999999)
        weights = (meta_data["n"].to_numpy(dtype=float) - 3).clip(min=1)
        z_bar = float(np.sum(weights * np.arctanh(clipped)) / np.sum(weights))
        se = 1.0 / math.sqrt(float(np.sum(weights)))
        z_stat = z_bar / se
        meta = {
            "analysis": f"{x}_vs_{y}_within_hidden_size_meta",
            "n_strata": int(len(meta_data)),
            "n": int(meta_data["n"].sum()),
            "spearman_rho": float(np.tanh(z_bar)),
            "p_value": float(2 * stats.norm.sf(abs(z_stat))),
        }
    return strata, meta


def run_capacity_geometry_control() -> tuple[pd.DataFrame, pd.DataFrame]:
    projection = pd.read_csv(TABLES / "human_state_capacity_multiaxis_projection.csv")
    projection = projection[projection["dynamics_available"].astype(str).str.lower().isin(["true", "1"])].copy()

    rows: list[dict[str, object]] = []
    for y in ["dynamics_trajectory_participation_ratio", "dynamics_capacity_geometry_z"]:
        rows.append(spearman_result(projection, "capacity_parameter_resource_z", y, f"raw_capacity_vs_{y}"))
        rows.append(
            partial_spearman(
                projection,
                "capacity_parameter_resource_z",
                y,
                ["selected_hidden_size"],
                ["selected_hidden_size"],
                f"hidden_size_residualized_capacity_vs_{y}",
            )
        )
        rows.append(
            partial_spearman(
                projection,
                "capacity_parameter_resource_z",
                y,
                ["selected_hidden_size", "dataset", "task"],
                ["selected_hidden_size", "dataset", "task"],
                f"hidden_size_dataset_task_residualized_capacity_vs_{y}",
            )
        )
        strata, meta = within_hidden_size_meta(projection, "capacity_parameter_resource_z", y)
        strata["outcome"] = y
        strata.to_csv(OUT / f"capacity_geometry_within_hidden_size_{y}.csv", index=False)
        rows.append(meta)

    results = pd.DataFrame(rows)
    results["q_value"] = bh_q(results["p_value"])
    results["bh_significant_05"] = results["q_value"] < 0.05
    source_cols = [
        "dataset",
        "subject",
        "session",
        "task",
        "selected_hidden_size",
        "capacity_parameter_resource_z",
        "dynamics_trajectory_participation_ratio",
        "dynamics_capacity_geometry_z",
        "mean_accuracy",
    ]
    source = projection[[c for c in source_cols if c in projection.columns]].copy()
    return results, source


def add_capacity_minus_load(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "participant_id" not in out.columns and {"dataset", "subject"}.issubset(out.columns):
        out["participant_id"] = out["dataset"].astype(str) + ":" + out["subject"].astype(str)
    if "load_z" not in out.columns and "load_level" in out.columns:
        out["load_z"] = zscore(out["load_level"])
    available = [c for c in CAPACITY_COMPONENTS_MINUS_LOAD if c in out.columns]
    out["capacity_minus_load_profile_raw"] = out[available].mean(axis=1, skipna=True)
    participant_profile = (
        out[["participant_id", "capacity_minus_load_profile_raw"]]
        .drop_duplicates("participant_id")
        .set_index("participant_id")["capacity_minus_load_profile_raw"]
    )
    participant_profile_z = zscore(participant_profile)
    out["capacity_minus_load_profile_z"] = out["participant_id"].map(participant_profile_z)
    out["load_x_capacity_minus_load"] = pd.to_numeric(out["load_z"], errors="coerce") * pd.to_numeric(
        out["capacity_minus_load_profile_z"], errors="coerce"
    )
    return out


def run_load_profile_control() -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = pd.read_csv(TABLES / "tu_berlin_coordinates.csv")
    if "participant_id" not in coords.columns:
        coords["participant_id"] = coords["dataset"].astype(str) + ":" + coords["subject"].astype(str)
    coords = add_capacity_minus_load(coords)

    rows = [
        ols_test(
            coords,
            "mean_accuracy",
            ["load_z", "capacity_minus_load_profile_z", "load_x_capacity_minus_load", "state_parameter_instability_z", "subject", "session"],
            "load_x_capacity_minus_load",
            "capacity_minus_load_pressure_accuracy",
            "capacity_minus_load_pressure",
            ["subject", "session"],
        ),
        ols_test(
            coords,
            "rt_median",
            ["load_z", "capacity_minus_load_profile_z", "load_x_capacity_minus_load", "state_parameter_instability_z", "subject", "session"],
            "load_x_capacity_minus_load",
            "capacity_minus_load_pressure_rt",
            "capacity_minus_load_pressure",
            ["subject", "session"],
        ),
    ]

    physio_path = TABLES / "tu_berlin_eeg_nirs_features.csv"
    if physio_path.exists():
        physio = add_capacity_minus_load(pd.read_csv(physio_path))
        for outcome in [
            "eeg_theta_alpha_ratio",
            "eeg_spectral_entropy",
            "eeg_aperiodic_slope",
            "nirs_hbo_mean",
            "nirs_hbo_slope",
            "nirs_hbr_mean",
        ]:
            if outcome in physio.columns:
                rows.append(
                    ols_test(
                        physio,
                        outcome,
                        [
                            "load_z",
                            "capacity_minus_load_profile_z",
                            "load_x_capacity_minus_load",
                            "state_parameter_instability_z",
                            "subject",
                            "session",
                        ],
                        "load_x_capacity_minus_load",
                        f"physiology_{outcome}_capacity_minus_load_pressure",
                        "physiology_capacity_minus_load_pressure",
                        ["subject", "session"],
                    )
                )

    models = pd.DataFrame(rows)
    models["q_value"] = bh_q(models["p_value"])
    models["bh_significant_05"] = models["q_value"] < 0.05

    source_cols = [
        "dataset",
        "subject",
        "session",
        "task",
        "load_level",
        "load_z",
        "mean_accuracy",
        "rt_median",
        "state_parameter_instability_z",
        "capacity_parameter_resource_z",
        "capacity_load_robustness_axis_z",
        "capacity_minus_load_profile_z",
        "load_x_capacity",
        "load_x_capacity_minus_load",
    ]
    source = coords[[c for c in source_cols if c in coords.columns]].copy()
    return models, source


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10**exp)
        return f"{mant:.2f} x 10^{exp}"
    return f"{p:.3f}"


def fmt_num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_replacement_text(geometry: pd.DataFrame, load_models: pd.DataFrame) -> dict[str, str]:
    def pick(analysis: str) -> pd.Series:
        hit = geometry[geometry["analysis"].eq(analysis)]
        if hit.empty:
            raise RuntimeError(f"Missing geometry analysis: {analysis}")
        return hit.iloc[0]

    broad_raw = pick("raw_capacity_vs_dynamics_capacity_geometry_z")
    broad_hs = pick("hidden_size_residualized_capacity_vs_dynamics_capacity_geometry_z")
    broad_hs_dt = pick("hidden_size_dataset_task_residualized_capacity_vs_dynamics_capacity_geometry_z")
    tpr_hs = pick("hidden_size_residualized_capacity_vs_dynamics_trajectory_participation_ratio")
    broad_meta = pick("capacity_parameter_resource_z_vs_dynamics_capacity_geometry_z_within_hidden_size_meta")

    def load_pick(outcome: str) -> pd.Series:
        hit = load_models[(load_models["outcome"].eq(outcome)) & (load_models["family"].eq("capacity_minus_load_pressure"))]
        if hit.empty:
            raise RuntimeError(f"Missing load model outcome: {outcome}")
        return hit.iloc[0]

    acc = load_pick("mean_accuracy")
    rt = load_pick("rt_median")

    geometry_text = (
        "The association weakened but remained positive after controlling selected hidden size. "
        f"For the broad recurrent-geometry summary, the raw association was rho = {fmt_num(float(broad_raw['spearman_rho']))}, "
        f"P = {fmt_p(float(broad_raw['p_value']))}, n = {int(broad_raw['n'])}; after rank-residualising both variables by selected hidden-size category, "
        f"the association remained positive (partial rho = {fmt_num(float(broad_hs['spearman_rho']))}, "
        f"P = {fmt_p(float(broad_hs['p_value']))}, n = {int(broad_hs['n'])}). "
        f"A fixed-hidden-size meta-analysis also remained positive "
        f"(rho = {fmt_num(float(broad_meta['spearman_rho']))}, P = {fmt_p(float(broad_meta['p_value']))}), "
        f"whereas adding dataset and task controls produced a small sign reversal "
        f"(partial rho = {fmt_num(float(broad_hs_dt['spearman_rho']))}, P = {fmt_p(float(broad_hs_dt['p_value']))}). "
        f"The trajectory-participation-ratio result was also positive after hidden-size control "
        f"(partial rho = {fmt_num(float(tpr_hs['spearman_rho']))}, P = {fmt_p(float(tpr_hs['p_value']))}). "
        "Thus the geometry result is not explained solely by the architectural component of the capacity profile, but it should be interpreted as a composition-sensitive profile relationship rather than as a hidden-size-free capacity resource."
    )

    load_text = (
        "The load-pressure pattern was not solely a restatement of the load-robustness component. "
        "After removing the load-robustness axis from the omnibus capacity profile and re-standardising the reduced profile, "
        f"the load x capacity-minus-load interaction remained positive for accuracy "
        f"(beta = {fmt_num(float(acc['estimate']))}, s.e. = {fmt_num(float(acc['std_error']))}, "
        f"t = {fmt_num(float(acc['t_value']))}, P = {fmt_p(float(acc['p_value']))}, q = {fmt_p(float(acc['q_value']))}, "
        f"n = {int(acc['n'])}) and negative for median response time "
        f"(beta = {fmt_num(float(rt['estimate']))}, s.e. = {fmt_num(float(rt['std_error']))}, "
        f"t = {fmt_num(float(rt['t_value']))}, P = {fmt_p(float(rt['p_value']))}, q = {fmt_p(float(rt['q_value']))}, "
        f"n = {int(rt['n'])}). "
        "Therefore, load exposed broader capacity-profile differences rather than only the component explicitly defined from load robustness."
    )
    return {
        "capacity_geometry_placeholder_replacement": geometry_text,
        "load_profile_placeholder_replacement": load_text,
    }


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS))


def set_paragraph_text(p: ET.Element, text: str) -> None:
    texts = p.findall(".//w:t", NS)
    if not texts:
        run = ET.SubElement(p, f"{{{W_NS}}}r")
        t = ET.SubElement(run, f"{{{W_NS}}}t")
        t.text = text
        return
    texts[0].text = text
    for t in texts[1:]:
        t.text = ""


def replace_docx_placeholders(replacements: dict[str, str]) -> None:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(SOURCE_DOCX)
    placeholder_patterns = [
        (
            re.compile(r"\[PLACEHOLDER.*?result pending finalised sensitivity analysis.*?hidden-size-free capacity resource.*?\]", re.I | re.S),
            replacements["capacity_geometry_placeholder_replacement"],
        ),
        (
            re.compile(r"\[PLACEHOLDER.*?Report whether the load.*?robustness term\.\]", re.I | re.S),
            replacements["load_profile_placeholder_replacement"],
        ),
    ]

    with zipfile.ZipFile(SOURCE_DOCX, "r") as zin:
        xml = zin.read("word/document.xml")
        root = ET.fromstring(xml)
        replaced = []
        for p in root.findall(".//w:p", NS):
            text = paragraph_text(p)
            if not text:
                continue
            for pattern, replacement in placeholder_patterns:
                if pattern.search(text):
                    set_paragraph_text(p, replacement)
                    replaced.append(text[:120])
                    break
        if len(replaced) != 2:
            raise RuntimeError(f"Expected 2 placeholder replacements, made {len(replaced)}: {replaced}")
        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        temp_docx = UPDATED_DOCX.with_suffix(".tmp.docx")
        with zipfile.ZipFile(temp_docx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
        shutil.move(str(temp_docx), str(UPDATED_DOCX))


def main() -> None:
    ensure_dirs()

    geometry, geometry_source = run_capacity_geometry_control()
    load_models, load_source = run_load_profile_control()

    geometry.to_csv(OUT / "capacity_geometry_hidden_size_control.csv", index=False)
    geometry_source.to_csv(OUT / "capacity_geometry_hidden_size_control_source.csv", index=False)
    load_models.to_csv(OUT / "tu_berlin_capacity_minus_load_models.csv", index=False)
    load_source.to_csv(OUT / "tu_berlin_capacity_minus_load_source.csv", index=False)

    combined = pd.concat(
        [
            geometry.assign(result_family="capacity_geometry_hidden_size_control"),
            load_models.assign(result_family="tu_berlin_capacity_minus_load"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(OUT / "placeholder_sensitivity_results.csv", index=False)
    try:
        with pd.ExcelWriter(OUT / "placeholder_sensitivity_source_data.xlsx") as writer:
            geometry.to_excel(writer, sheet_name="geometry_tests", index=False)
            geometry_source.to_excel(writer, sheet_name="geometry_source", index=False)
            load_models.to_excel(writer, sheet_name="capacity_minus_load_models", index=False)
            load_source.to_excel(writer, sheet_name="capacity_minus_load_source", index=False)
            combined.to_excel(writer, sheet_name="combined_results", index=False)
    except Exception as exc:
        (OUT / "xlsx_write_failed.txt").write_text(str(exc), encoding="utf-8")

    replacements = build_replacement_text(geometry, load_models)
    (OUT / "placeholder_replacement_text.md").write_text(
        "\n\n".join(f"## {key}\n\n{value}" for key, value in replacements.items()),
        encoding="utf-8",
    )
    (OUT / "placeholder_sensitivity_summary.json").write_text(
        json.dumps(
            {
                "source_docx": str(SOURCE_DOCX),
                "updated_docx": str(UPDATED_DOCX),
                "outputs": {
                    "geometry_tests": str(OUT / "capacity_geometry_hidden_size_control.csv"),
                    "geometry_source": str(OUT / "capacity_geometry_hidden_size_control_source.csv"),
                    "load_models": str(OUT / "tu_berlin_capacity_minus_load_models.csv"),
                    "load_source": str(OUT / "tu_berlin_capacity_minus_load_source.csv"),
                    "combined_results": str(OUT / "placeholder_sensitivity_results.csv"),
                    "source_workbook": str(OUT / "placeholder_sensitivity_source_data.xlsx"),
                },
                "replacements": replacements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    replace_docx_placeholders(replacements)

    if DOC_AUDIT.exists():
        (OUT / "input_placeholder_audit_report.md").write_text(
            (DOC_AUDIT / "placeholder_audit_report.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(json.dumps({"updated_docx": str(UPDATED_DOCX), "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
