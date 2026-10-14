"""`_edge_penalty` must be live in the env the trainer actually uses.

`reward._edge_penalty` reads the tile off the env object, so it silently
returned 0.0 for every run trained through `IsaacLanderVecEnv`: that env hands
`reward_fn` an `_EnvView`, and `_EnvView` had no `tile` attribute, so
`getattr(getattr(env, "tile", None), "size_m", 0.0)` fell through to 0.0 and
the term's own first guard short-circuited.

The term is not decorative -- `reward.py` records a learner exploiting the fact
that `left_tile` truncation pays no timeout penalty, which made flying off the
map a free exit from a crash. These tests pin the plumbing rather than the
coefficient, so a future refactor of `_EnvView` cannot quietly drop it again.

Deliberately Isaac-free: it exercises `_EnvView` and `_edge_penalty` directly,
so it runs in the core test suite instead of needing a GPU and a USD stage.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from lunarsim.rl.reward import RewardWeights, _edge_penalty

# `_EnvView` lives in the Isaac adapter module, but that module imports Isaac at
# call time, not import time, for exactly this kind of pure-Python reuse.
_vec = pytest.importorskip("lunarsim.adapters.isaac.isaac_lander_vec_env")


def test_env_view_carries_the_tile():
    """The attribute `_edge_penalty` reaches for has to exist on the view."""
    assert "tile" in _vec._EnvView.__slots__


class _Tile:
    def __init__(self, size_m):
        self.size_m = size_m


def test_edge_penalty_is_nonzero_through_the_view():
    """A vehicle well outside the safe fraction must be charged.

    0.88 of a 1680 m tile's half-extent is 739.2 m, so 800 m is outside it and
    the penalty must be positive. With the tile missing it was exactly 0.0,
    which is the regression this guards.
    """
    w = RewardWeights()
    view = _vec._EnvView(None, {}, _Tile(1680.0))
    charged = _edge_penalty(w, 800.0, 0.0,
                            getattr(getattr(view, "tile", None), "size_m", 0.0))
    assert charged > 0.0

    blind = _vec._EnvView(None, {})
    assert _edge_penalty(w, 800.0, 0.0,
                         getattr(getattr(blind, "tile", None), "size_m", 0.0)) == 0.0


def test_edge_penalty_is_zero_inside_the_safe_fraction():
    """Nothing is charged where the vehicle is supposed to fly, so this is a
    barrier and not a drift the policy has to fight everywhere."""
    w = RewardWeights()
    half = 1680.0 / 2.0
    inside = half * w.edge_safe_frac * 0.5
    assert _edge_penalty(w, inside, 0.0, 1680.0) == 0.0


def test_default_weights_actually_price_the_edge():
    """If `edge_k` is ever defaulted back to 0 the plumbing above is moot, so
    assert the coefficient too -- the term guards a measured exploit."""
    assert RewardWeights().edge_k > 0.0
    assert 0.0 < RewardWeights().edge_safe_frac < 1.0


def test_reward_weights_has_the_fields_the_term_reads():
    names = {f.name for f in fields(RewardWeights)}
    assert {"edge_k", "edge_safe_frac", "edge_cap"} <= names
