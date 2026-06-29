from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import forbidden_flags_present


class FullRunGuardTests(unittest.TestCase):
    def test_forbidden_sampling_flags_are_detected(self) -> None:
        self.assertEqual(forbidden_flags_present(["--max-subjects", "10"]), ["--max-subjects"])
        self.assertEqual(forbidden_flags_present(["--sample=true"]), ["--sample=true"])

    def test_ordinary_flags_pass_detection(self) -> None:
        self.assertEqual(forbidden_flags_present(["--record-id", "6874129"]), [])


if __name__ == "__main__":
    unittest.main()
