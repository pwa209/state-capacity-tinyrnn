from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nhb_revision" / "placeholder_sensitivity"
SOURCE_DOCX = OUT / "State_capacity_PW_revised_placeholders_resolved.docx"
UPDATED_DOCX = OUT / "State_capacity_PW_revised_all_placeholders_resolved_provisional.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)


DATA_AVAILABILITY = (
    "All analyses used secondary, de-identified, publicly available datasets obtained under the providers' terms. "
    "The original studies were conducted under their own ethics approvals and consent procedures. No new human-participant data were collected. "
    "Raw datasets are available from their original providers: COG-BCI was downloaded from Zenodo record 6874129 "
    "(https://zenodo.org/records/6874129); the TU Berlin simultaneous EEG-NIRS N-back dataset was downloaded from the official TU Berlin site "
    "(https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/; EEG_01-26_MATLAB.zip, NIRS_01-26_MATLAB.zip, behaviour files and participant archives); "
    "the CMx7-MM multimodal dataset is available on OpenNeuro as ds007554, version 1.0.0, DOI 10.18112/openneuro.ds007554.v1.0.0; "
    "and Healthy Brain Network EEG Release 4 is available on OpenNeuro as ds005508, version 1.0.1, DOI 10.18112/openneuro.ds005508.v1.0.1. "
    "The current local reproducibility package contains the derived de-identified state and capacity coordinates, locked artificial-agent coordinate system, "
    "harmonised event tables, row-level result tables, panel-level source-data files for all display items, exclusion logs, reconstruction flags and claim-audit records. "
    "Before journal submission, this derived-data package should be deposited in a public Zenodo or OSF archive and the resulting DOI inserted in the final manuscript."
)

CODE_AVAILABILITY = (
    "At the current draft stage, all analysis code is contained in the local reproducibility package under state_capacity_tinyrnn/scripts/, "
    "with staged scripts for artificial-agent perturbations, human recurrent-model fitting, state and capacity projection, neurophysiological feature extraction, "
    "statistical analysis, figure generation, manuscript assembly and placeholder-resolution sensitivity analyses. "
    "The project includes environment.yml, pyproject.toml, configuration files, random-seed settings, output manifests and source-data tables linking manuscript results to generating scripts. "
    "Before journal submission, the code package should be mirrored to a public GitHub repository and archived on Zenodo; the public repository URL, archive DOI, release tag, commit hash and final licence should then be inserted in the final manuscript. "
    "No Docker or Singularity image has yet been created, so end-to-end reproduction currently relies on the provided environment files rather than a container image."
)


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


def update_docx() -> dict[str, object]:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(SOURCE_DOCX)

    replacements = []
    with zipfile.ZipFile(SOURCE_DOCX, "r") as zin:
        root = ET.fromstring(zin.read("word/document.xml"))
        for p in root.findall(".//w:p", NS):
            text = paragraph_text(p)
            if "[accession/DOI/version]" in text or "[Zenodo/OSF DOI]" in text:
                set_paragraph_text(p, DATA_AVAILABILITY)
                replacements.append("data_availability")
            elif "[GitHub URL]" in text or "[Zenodo DOI]" in text or "vX.X.X" in text or "[hash]" in text:
                set_paragraph_text(p, CODE_AVAILABILITY)
                replacements.append("code_availability")

        if sorted(replacements) != ["code_availability", "data_availability"]:
            raise RuntimeError(f"Expected data/code replacements, got {replacements}")

        xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        tmp = UPDATED_DOCX.with_suffix(".tmp.docx")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
        shutil.move(str(tmp), str(UPDATED_DOCX))

    summary = {
        "source_docx": str(SOURCE_DOCX),
        "updated_docx": str(UPDATED_DOCX),
        "replacements": replacements,
        "remaining_required_before_submission": [
            "Deposit derived-data/source-data package to Zenodo or OSF and insert DOI.",
            "Mirror code to public GitHub repository and insert URL.",
            "Archive code release to Zenodo and insert DOI, release tag and commit hash.",
            "Choose and declare final licence.",
            "Optionally create Docker/Singularity container; otherwise retain environment.yml/pyproject.toml reproduction route.",
        ],
    }
    (OUT / "availability_placeholder_update_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "availability_replacement_text.md").write_text(
        f"## Data availability\n\n{DATA_AVAILABILITY}\n\n## Code availability\n\n{CODE_AVAILABILITY}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = update_docx()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
