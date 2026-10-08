"""The reward function must run on EVERY stage, including the ones with ranged spawns.

`land_handoff` is the first stage to give `spawn_altitude_m` a range -- [8, 25] m, so the
handoff an approach policy produces is covered rather than assumed -- and the reward
divided by `max(release_alt_m, 1.0)` with the tuple still in it:

    TypeError: '>' not supported between instances of 'float' and 'tuple'

That killed the landing specialist 17 seconds into its first run, and the term it crashed
in is weighted 0.0, so the arithmetic could never have affected the reward. The defect was
not in the formula: adding ranged parameters to the curriculum changed the TYPE of
something other modules read as a scalar, and nothing checked that every stage could still
be rewarded.

So this walks the whole curriculum, specialist stages included, rather than testing the one
stage that broke. A new ranged parameter on any stage fails here instead of 17 seconds into
a training run.
"""
import numpy as np
import pytest

from lunarsim.rl.curriculum import SPECIALIST_STAGES, STAGES, STAGES_BY_NAME
from lunarsim.rl.reward import default_reward_fn
from lunarsim.rl.spawn_range import upper

ALL = [s.name for s in list(STAGES) + list(SPECIALIST_STAGES)]


class _Tile:
    def __init__(self, size):
        self.size_m = size


class _Env:
    def __init__(self, stage, state):
        self.params = stage.params
        self.state = state
        self.tile = _Tile(stage.tile_size_m)


def _state(alt, vz=-2.0, lat=3.0):
    return {"x": 4.0, "y": -3.0, "z": alt, "vx": lat, "vy": 0.0, "vz": vz,
            "tilt_x": 0.05, "tilt_y": -0.04, "wx": 0.02, "wy": -0.01, "wz": 0.0,
            "fuel_kg": 8000.0, "rcs_fuel_kg": 280.0}


def test_at_least_one_stage_actually_uses_a_range():
    """Otherwise this file would pass while testing nothing."""
    ranged = [s.name for s in list(STAGES) + list(SPECIALIST_STAGES)
              if isinstance(s.params.spawn_altitude_m, (tuple, list))]
    assert ranged, "no stage has a ranged spawn_altitude_m; this test has stopped testing"


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("phase", ["flight", "terminated", "truncated"])
def test_reward_runs_and_is_finite_on_every_stage(name, phase):
    stage = STAGES_BY_NAME[name]
    alt = {"flight": 0.5 * upper(stage.params.spawn_altitude_m),
           "terminated": 0.0, "truncated": 2.0}[phase]
    info = {"altitude_m": alt, "action": np.zeros(4, dtype=np.float32),
            "terminated": phase == "terminated",
            "truncated": phase == "truncated",
            "timed_out": phase == "truncated",
            "landing_margins": {}}
    r = default_reward_fn(_Env(stage, _state(alt)), info)
    assert np.isfinite(r), f"{name}/{phase} produced {r}"
    assert np.isfinite(info["reward_terminal"]) and np.isfinite(info["reward_shaping"])
    assert info["reward_terminal"] + info["reward_shaping"] == pytest.approx(r, abs=1e-6)


@pytest.mark.parametrize("name", ALL)
def test_reward_runs_across_the_whole_release_envelope(name):
    """Sweeping altitude matters: the crash was in a term that only the release altitude
    reached, and a single mid-altitude sample would have missed a bound-only defect."""
    stage = STAGES_BY_NAME[name]
    top = upper(stage.params.spawn_altitude_m)
    for alt in np.linspace(0.0, top, 9):
        for vz in (-8.0, -1.0, 0.0, 1.0):
            info = {"altitude_m": float(alt), "action": np.zeros(4, dtype=np.float32),
                    "terminated": False, "truncated": False}
            r = default_reward_fn(_Env(stage, _state(float(alt), vz=vz)), info)
            assert np.isfinite(r), f"{name} alt={alt:.1f} vz={vz} -> {r}"
