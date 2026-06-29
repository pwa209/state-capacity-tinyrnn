from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "nhb_revision" / "record_package"
OUT_FILE = OUT_DIR / "state_capacity_all_python_scripts_20260625.py"
MANIFEST = OUT_DIR / "state_capacity_all_python_scripts_20260625_manifest.csv"


EXCLUDE_PARTS = {
    ".git",
    ".venv",
    ".agents",
    ".codex",
    "__pycache__",
    "data",
    "node_modules",
}

EXCLUDE_PREFIXES = {
    "outputs/",
    "state_capacity_tinyrnn/data/",
    "state_capacity_tinyrnn/outputs/archive_previous_run/",
    "state_capacity_tinyrnn/outputs/",
}

EXCLUDE_NAMES = {
    OUT_FILE.name,
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if path.name in EXCLUDE_NAMES:
        return False
    if any(part in EXCLUDE_PARTS for part in path.relative_to(ROOT).parts):
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    return path.suffix == ".py"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*.py") if should_include(path))
    rows: list[dict[str, object]] = []
    sources: list[tuple[str, str]] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        sources.append((rel, text))
        rows.append(
            {
                "relative_path": rel,
                "bytes_utf8": len(text.encode("utf-8", errors="replace")),
                "lines": text.count("\n") + (1 if text else 0),
                "sha256": sha256_text(text),
            }
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "bytes_utf8", "lines", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    generated_at = datetime.now(timezone.utc).isoformat()
    total_lines = sum(int(row["lines"]) for row in rows)
    total_bytes = sum(int(row["bytes_utf8"]) for row in rows)
    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as f:
        f.write('"""Single-file source compendium for the TinyRNN state-capacity project.\n\n')
        f.write("This file is generated for record/reference use. It is valid Python, but it is not intended to run the whole study directly.\n")
        f.write("Use SCRIPT_INDEX to inspect the included files, or write_scripts(target_dir) to reconstruct the source tree.\n")
        f.write('"""\n\n')
        f.write("from __future__ import annotations\n\n")
        f.write("from pathlib import Path\n\n")
        f.write(f"GENERATED_AT_UTC = {generated_at!r}\n")
        f.write("PROJECT_ROOT_AT_GENERATION = r\"C:\\\\Users\\\\Gebruiker\\\\Documents\\\\TinyRNN State and Capacity\"\n")
        f.write(f"N_SCRIPTS = {len(rows)!r}\n")
        f.write(f"TOTAL_SOURCE_LINES = {total_lines!r}\n")
        f.write(f"TOTAL_SOURCE_BYTES_UTF8 = {total_bytes!r}\n\n")
        f.write("SCRIPT_INDEX = [\n")
        for row in rows:
            f.write(
                "    "
                + repr(
                    {
                        "relative_path": row["relative_path"],
                        "bytes_utf8": row["bytes_utf8"],
                        "lines": row["lines"],
                        "sha256": row["sha256"],
                    }
                )
                + ",\n"
            )
        f.write("]\n\n")
        f.write("SCRIPT_SOURCES = {\n")
        for rel, text in sources:
            f.write(f"    {rel!r}: {text!r},\n")
        f.write("}\n\n")
        f.write(
            "def write_scripts(target_dir: str | Path, overwrite: bool = False) -> list[Path]:\n"
            "    \"\"\"Reconstruct all archived scripts under target_dir.\"\"\"\n"
            "    target = Path(target_dir)\n"
            "    written: list[Path] = []\n"
            "    for relative_path, source in SCRIPT_SOURCES.items():\n"
            "        out = target / relative_path\n"
            "        if out.exists() and not overwrite:\n"
            "            raise FileExistsError(out)\n"
            "        out.parent.mkdir(parents=True, exist_ok=True)\n"
            "        out.write_text(source, encoding='utf-8', newline='\\n')\n"
            "        written.append(out)\n"
            "    return written\n\n"
            "def print_index() -> None:\n"
            "    for item in SCRIPT_INDEX:\n"
            "        print(f\"{item['relative_path']}\\t{item['lines']} lines\\t{item['sha256']}\")\n\n"
            "if __name__ == '__main__':\n"
            "    print(f\"TinyRNN state-capacity Python compendium: {N_SCRIPTS} scripts, {TOTAL_SOURCE_LINES} lines\")\n"
            "    print_index()\n"
        )

    print(OUT_FILE)
    print(MANIFEST)
    print(f"scripts={len(rows)} lines={total_lines} bytes={total_bytes}")


if __name__ == "__main__":
    main()
