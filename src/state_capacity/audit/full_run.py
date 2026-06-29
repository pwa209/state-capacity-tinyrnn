from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


FORBIDDEN_FULL_RUN_FLAGS = {
    "--sample",
    "--max-subjects",
    "--max_subjects",
    "--nrows",
    "--head",
    "--debug",
    "--pilot",
    "--limit",
    "--early-break",
    "--early_break",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_simple_yaml(path: Path) -> dict[str, object]:
    """Load the simple key/value YAML used by run_mode.yaml without extra deps."""
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def forbidden_flags_present(argv: Iterable[str]) -> list[str]:
    present: list[str] = []
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in FORBIDDEN_FULL_RUN_FLAGS:
            present.append(arg)
    return present


def assert_full_run_allowed(
    argv: Iterable[str] | None = None,
    config_path: Path | None = None,
    allow_smoke: bool = False,
) -> dict[str, object]:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = config_path or project_root() / "configs" / "run_mode.yaml"
    config = load_simple_yaml(config_path)

    if allow_smoke and "--smoke" in argv:
        print("FULL_RUN_GUARD: smoke mode requested; no empirical claim outputs will be created.")
        return config

    if config.get("mode") != "full":
        raise RuntimeError(f"Full-run mode required, found mode={config.get('mode')!r}")
    if config.get("allow_pilot") is not False:
        raise RuntimeError("Full-run protocol requires allow_pilot: false")
    if config.get("allow_manual_data") is not False:
        raise RuntimeError("Full-run protocol requires allow_manual_data: false")
    if config.get("require_claim_audit") is not True:
        raise RuntimeError("Full-run protocol requires require_claim_audit: true")
    if config.get("require_source_data") is not True:
        raise RuntimeError("Full-run protocol requires require_source_data: true")

    forbidden = forbidden_flags_present(argv)
    if forbidden:
        raise RuntimeError(
            "Full-run mode forbids debug sampling or subject caps: " + ", ".join(forbidden)
        )

    print(
        "FULL_RUN_GUARD: mode=full; allow_pilot=false; "
        "allow_manual_data=false; claim_audit_required=true"
    )
    return config


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--smoke", action="store_true", help="Run only non-claim smoke checks.")
    return parser

