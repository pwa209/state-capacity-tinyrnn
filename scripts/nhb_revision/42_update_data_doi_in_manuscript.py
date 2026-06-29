from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "nhb_revision" / "placeholder_sensitivity"
INPUT_DOCX = OUT_DIR / "State_capacity_PW_revised_all_placeholders_resolved_provisional.docx"
OUTPUT_DOCX = OUT_DIR / "State_capacity_PW_revised_data_doi_resolved.docx"
SUMMARY_JSON = OUT_DIR / "data_doi_manuscript_update_summary.json"
REPLACEMENT_MD = OUT_DIR / "data_doi_replacement_text.md"

ZENODO_RECORD = "https://zenodo.org/records/21028379"
ZENODO_DOI = "10.5281/zenodo.21028379"
GITHUB_REPO = "https://github.com/pwa209/state-capacity-tinyrnn"


DATA_AVAILABILITY_TEXT = (
    "All analyses used secondary, de-identified, publicly available datasets obtained "
    "under the providers' terms. The original studies were conducted under their own "
    "ethics approvals and consent procedures. No new human-participant data were "
    "collected. Raw datasets are available from their original providers: "
    "COG-BCI from Zenodo record 6874129 (https://zenodo.org/records/6874129); "
    "the TU Berlin simultaneous EEG-NIRS N-back dataset from "
    "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/; the CMx7-MM multimodal "
    "dataset from OpenNeuro ds007554 version 1.0.0 "
    "(https://doi.org/10.18112/openneuro.ds007554.v1.0.0); and the Healthy Brain "
    "Network EEG Release 4 from OpenNeuro ds005508 version 1.0.1 "
    "(https://doi.org/10.18112/openneuro.ds005508.v1.0.1). The derived de-identified "
    "state/capacity coordinates, harmonised behavioural tables, statistical result "
    "tables, figure source-data workbooks, neurophysiology feature tables, exclusion "
    f"logs, reconstruction records and manifest files are deposited on Zenodo at {ZENODO_RECORD}, "
    f"DOI {ZENODO_DOI}. Raw dataset files are not redistributed in that archive."
)

CODE_AVAILABILITY_TEXT = (
    "All analysis code is available in the public GitHub repository "
    f"{GITHUB_REPO}, with staged scripts for artificial-agent perturbations, "
    "human recurrent-model fitting, state and capacity projection, neurophysiological "
    "feature extraction, statistical analysis, figure generation, manuscript assembly "
    "and placeholder-resolution sensitivity analyses. The project includes environment.yml, "
    "pyproject.toml, configuration files, random-seed settings, output manifests and "
    "source-data tables linking manuscript results to generating scripts. Before final "
    "journal submission, the GitHub repository should be released as version v1.0.0 and "
    "archived on Zenodo; the resulting software archive DOI, release tag and commit hash "
    "should then be inserted into the final Code availability statement."
)


def _xml_text(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _replace_paragraph_text(document_xml: str, heading: str, replacement: str) -> tuple[str, int]:
    paragraphs = re.findall(r"<w:p[\s\S]*?</w:p>", document_xml)
    replacements = 0
    for paragraph in paragraphs:
        plain = "".join(re.findall(r"<w:t[^>]*>([\s\S]*?)</w:t>", paragraph))
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain.startswith(heading):
            continue

        p_start = re.match(r"(<w:p[^>]*>)", paragraph)
        p_pr = re.search(r"(<w:pPr[\s\S]*?</w:pPr>)", paragraph)
        if p_start is None:
            raise RuntimeError(f"Could not parse paragraph wrapper for {heading!r}.")
        new_paragraph = (
            p_start.group(1)
            + (p_pr.group(1) if p_pr else "")
            + f"<w:r><w:t>{_xml_text(replacement)}</w:t></w:r></w:p>"
        )
        document_xml = document_xml.replace(paragraph, new_paragraph, 1)
        replacements += 1
    return document_xml, replacements


def main() -> None:
    if not INPUT_DOCX.exists():
        raise FileNotFoundError(INPUT_DOCX)

    with zipfile.ZipFile(INPUT_DOCX, "r") as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        updated_xml, n_data = _replace_paragraph_text(
            document_xml, "All analyses used secondary", DATA_AVAILABILITY_TEXT
        )
        if n_data != 1:
            raise RuntimeError(f"Expected one Data availability paragraph, found {n_data}.")
        updated_xml, n_code = _replace_paragraph_text(
            updated_xml, "At the current draft stage", CODE_AVAILABILITY_TEXT
        )
        if n_code != 1:
            raise RuntimeError(f"Expected one Code availability paragraph, found {n_code}.")
        entries = {
            item.filename: archive.read(item.filename)
            for item in archive.infolist()
            if item.filename != "word/document.xml"
        }

    with zipfile.ZipFile(OUTPUT_DOCX, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr("word/document.xml", updated_xml)

    with zipfile.ZipFile(OUTPUT_DOCX, "r") as archive:
        archive.testzip()
        final_xml = archive.read("word/document.xml").decode("utf-8")

    summary = {
        "input_docx": str(INPUT_DOCX),
        "output_docx": str(OUTPUT_DOCX),
        "zenodo_record": ZENODO_RECORD,
        "zenodo_doi": ZENODO_DOI,
        "github_repo": GITHUB_REPO,
        "data_availability_replacements": n_data,
        "code_availability_replacements": n_code,
        "doi_present": ZENODO_DOI in final_xml,
        "record_present": ZENODO_RECORD in final_xml,
        "github_repo_present": GITHUB_REPO in final_xml,
        "remaining_data_pending_phrase": "derived-data package should be deposited" in final_xml,
        "software_doi_still_pending": "software archive DOI" in final_xml,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPLACEMENT_MD.write_text(
        DATA_AVAILABILITY_TEXT + "\n\n" + CODE_AVAILABILITY_TEXT + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
