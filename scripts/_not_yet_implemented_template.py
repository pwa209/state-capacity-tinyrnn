from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


def fail(step_name: str, required_outputs: list[str]) -> None:
    assert_full_run_allowed()
    outputs = "\n".join(f"- {item}" for item in required_outputs)
    raise SystemExit(
        f"{step_name} is scaffolded but not implemented yet.\n"
        "This is intentional: the revised protocol forbids carrying forward previous results as final.\n"
        "Required outputs before this step can pass:\n"
        f"{outputs}\n"
    )

