from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from state_capacity.audit.full_run import assert_full_run_allowed
from download_all import download_item, tu_berlin_items, write_manifest


def main() -> None:
    assert_full_run_allowed()
    items = tu_berlin_items()
    for item in items:
        download_item(item)
    write_manifest(items)
    failures = [item for item in items if item.status == "failed"]
    if failures:
        raise SystemExit(f"TU Berlin download failed for {len(failures)} files")


if __name__ == "__main__":
    main()
