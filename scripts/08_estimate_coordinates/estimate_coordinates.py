from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _not_yet_implemented_template import fail

fail("Step 09 human multidimensional fingerprint projection", [
    "outputs/tables/human_state_capacity_multiaxis_projection.csv",
    "outputs/tables/multiaxis_profile_convergence_tests.csv",
    "outputs/tables/multiaxis_projection_residualized_tests.csv",
    "outputs/audit/human_projection_claim_limits.md",
])
