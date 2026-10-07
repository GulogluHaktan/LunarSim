"""The per-step reward must be continuous and dense over the WHOLE release envelope,
not only at the boundaries a test was written for.

tests/test_reward.py checks continuity where it was known to matter: across the touchdown
success boundary (a 411-point step, fixed), through zero vertical speed, through free
fall. Those are point checks, and this project's reward has acquired a clip, two max()
floors, a quadratic edge barrier and a regime-dependent angular-rate term since they were
written -- any of which could have introduced a cliff somewhere nobody looked.

This runs scripts/diag_reward_continuity.py, which differences every sampled state at two
step sizes a decade apart and so separates a cliff from a kink: a clip has a bounded
one-sided slope and its slope ratio stays near 1, while a jump's slope grows like 1/step.
The script is invoked rather than reimplemented so there is one copy of the scan.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diag_reward_continuity.py"


@pytest.fixture(scope="module")
def scan():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--samples", "600"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_no_discontinuities_anywhere_in_the_envelope(scan):
    m = re.search(r"DISCONTINUITIES .*?: (\d+)", scan)
    assert m, scan
    assert int(m.group(1)) == 0, (
        "the per-step reward jumps somewhere in the orbit_descent envelope:\n" + scan)


@pytest.mark.parametrize("dim", ["vx", "vy", "vz", "wx", "wy", "wz"])
def test_the_graded_dimensions_have_a_gradient_almost_everywhere(scan, dim):
    """Dense means the policy gets a signal at (nearly) every state, not just near a
    limit. Every dimension the touchdown criteria grade is checked: v_xy, v_z and omega.
    Tilt is graded too but deliberately shaped weakly -- the controller tilts 8-13 deg
    while braking, so a strong dense tilt term would charge it for what it must do -- so
    it is held to a lower bar below rather than listed here.
    """
    m = re.search(rf"^\s+{dim}\s+([\d.]+)%", scan, re.M)
    assert m, scan
    assert float(m.group(1)) > 99.0, f"{dim} responds in only {m.group(1)}% of states"


def test_tilt_and_altitude_still_respond_almost_everywhere(scan):
    for dim in ("tilt_x", "tilt_y", "altitude_m"):
        m = re.search(rf"^\s+{dim}\s+([\d.]+)%", scan, re.M)
        assert m, scan
        assert float(m.group(1)) > 95.0, f"{dim}: {m.group(1)}%"


def test_position_is_unshaped_in_the_interior_on_purpose(scan):
    """x/y respond in only ~38% of sampled states, and that is intended rather than a
    gap: the only position term is the edge barrier, which is zero inside 88% of the
    half-extent because this task's success criteria contain no position term at all.
    Pinned so that a future change which starts charging for position everywhere has to
    come here and say so.
    """
    fracs = [float(re.search(rf"^\s+{d}\s+([\d.]+)%", scan, re.M).group(1))
             for d in ("x", "y")]
    for f in fracs:
        assert 20.0 < f < 60.0, f"position response moved to {f}% -- intended?"
