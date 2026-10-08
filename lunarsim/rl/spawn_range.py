"""Randomised entry conditions, for training a landing specialist.

WHY. The measured case for splitting this task in two, which three independent sources
agree on:

  1. This project's own altitude-resolved measurement of the critic's action gradient:

        altitude     toward the expert    |dQ/da|
         0-2 m            54.3%            1.297     decisive, and the gradient is noise
         2-5 m            53.8%            0.585
        15-40 m           63.1%            0.257
        80-150 m          68.6%            0.118     informative, and barely pushes

     The actor is pushed hardest exactly where the critic knows least. And the buffer is
     65% cruise-phase samples (40-210 m) against 14% in the decisive 0-5 m band, so most of
     what the critic learns is about the regime that does not decide the outcome.

  2. The reference paper for this problem (Gaudet/Linares/Furfaro, arXiv:1810.08719) already
     switches regime at 15 m -- aim at the pad above it, pure vertical descent below.

  3. Apollo's own descent guidance was phase-split: P63 braking, P64 approach, P66 terminal
     descent, with different gains in each.

So a landing specialist trains only in the regime it must be precise in, where its buffer is
100% decisive-band data -- which is the condition under which this project's own history
records from-scratch RL working ("a RANDOM policy touches down 40/40 ... the success region
is immediately adjacent to random behaviour").

The specialist is only useful if it can accept whatever an approach policy hands it, so its
entry conditions have to be a DISTRIBUTION rather than a point. These helpers let a stage
declare a range for altitude and descent rate the way it already can for horizontal speed,
without changing the meaning of a scalar, so every existing stage is untouched.
"""
from __future__ import annotations

import numpy as np


def sample_range(value, rng: np.random.Generator) -> float:
    """A scalar stays a scalar; a (lo, hi) pair is sampled uniformly.

    Keeping scalars exact matters: every stage in the curriculum declares scalars today and
    none of their behaviour may shift because this was added.
    """
    if isinstance(value, (tuple, list)):
        lo, hi = float(value[0]), float(value[1])
        if hi < lo:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi))
    return float(value)


def describe(value) -> str:
    if isinstance(value, (tuple, list)):
        return f"[{float(value[0]):.2f}, {float(value[1]):.2f}]"
    return f"{float(value):.2f}"


def upper(value) -> float:
    """The TOP of a range, or the value itself when it is already a scalar.

    For code that needs one number out of a possibly-ranged spawn parameter and means
    "the widest case" -- a release altitude to normalise progress against, a LiDAR range
    to allocate. A scalar passes through unchanged, so nothing that worked before moves.

    This exists because a crash found it. `land_handoff` is the first stage to give
    `spawn_altitude_m` a range, and `reward._descent_progress_reward` divided by
    `max(release_alt_m, 1.0)` with the tuple still in it:

        TypeError: '>' not supported between instances of 'float' and 'tuple'

    which killed the landing specialist 17 seconds into its first run -- and the term it
    crashed in is weighted 0.0, so the arithmetic was never going to affect the reward at
    all. Adding ranged parameters to the curriculum changed the TYPE of something four
    other modules read as a scalar; `sample_range` covered the envs, and this covers the
    readers that want a bound rather than a draw.
    """
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("empty range")
        return float(max(value))
    return float(value)
