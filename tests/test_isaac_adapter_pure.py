"""Only tests the parts of the Isaac adapter that don't touch `pxr`/`omni`
(no Isaac Sim install available in this environment)."""
from dataclasses import dataclass

import numpy as np

from lunarsim.adapters.isaac.rocks import cap_rock_count
from lunarsim.core.terrain.rocks import RockField


@dataclass
class _FakeTile:
    rocks: RockField


def test_cap_rock_count_keeps_largest_and_respects_max():
    rng = np.random.default_rng(0)
    n = 100
    rocks = RockField(
        x_m=rng.uniform(-10, 10, n),
        y_m=rng.uniform(-10, 10, n),
        diameter_m=np.linspace(0.1, 5.0, n),
    )
    tile = _FakeTile(rocks)

    capped = cap_rock_count(tile, max_count=10, rng=rng)
    assert len(capped.rocks.diameter_m) == 10
    # the 10 largest of a linspace(0.1, 5.0, 100) are all > 4.5
    assert capped.rocks.diameter_m.min() > 4.5


def test_cap_rock_count_noop_when_under_limit():
    rng = np.random.default_rng(0)
    rocks = RockField(x_m=np.zeros(5), y_m=np.zeros(5), diameter_m=np.ones(5))
    tile = _FakeTile(rocks)
    capped = cap_rock_count(tile, max_count=10, rng=rng)
    assert capped is tile
