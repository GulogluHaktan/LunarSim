"""Is the per-step reward DENSE and CONTINUOUS everywhere the policy flies, or only
where a test happened to look?

tests/test_reward.py checks continuity at the places it was known to matter -- across
the touchdown success boundary, through zero vertical speed, through free fall. Those are
point checks. This sweeps the whole `orbit_descent` release envelope and asks the question
that distinguishes a CLIFF from a KINK:

    a kink  (a clip, a max(), an abs()) has a bounded one-sided slope, so
            |dr|/|dx| stays put as the step shrinks
    a jump  (a branch that changes the value) has unbounded slope, so
            |dr|/|dx| grows like 1/step as the step shrinks

So each sampled state is differenced at two step sizes a decade apart. A ratio near 1
means Lipschitz-continuous at that point; a ratio near 10 means a discontinuity sits
inside the interval. That test cannot be fooled by a large-but-finite gradient, which is
what a single-step scan would report as suspicious.

Dense is the other half of the question and it is reported alongside: the fraction of
sampled states where the per-step reward responds at all to each state dimension. A
dimension that moves nothing over the envelope is a dimension the policy gets no gradient
in, however continuous it is.

Run (no Isaac needed):
    .venv/bin/python scripts/diag_reward_continuity.py
"""
from __future__ import annotations

import argparse

import numpy as np

from lunarsim.rl.curriculum import STAGES_BY_NAME
from lunarsim.rl.reward import make_apollo_reward_fn

p = argparse.ArgumentParser()
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--samples", type=int, default=4000)
p.add_argument("--step", type=float, default=1e-3,
               help="the coarse difference step, in each dimension's own units")
p.add_argument("--ratio-flag", type=float, default=3.0,
               help="slope(step/10)/slope(step) above this is reported as a jump")
args = p.parse_args()

stage = STAGES_BY_NAME[args.stage]
params = stage.params
reward_fn = make_apollo_reward_fn()
TILE = 150.0


class _Tile:
    size_m = TILE


class _Env:
    """The `.params` / `.state` / `.tile` triple is all `reward_fn` touches."""
    def __init__(self, state):
        self.params = params
        self.state = state
        self.tile = _Tile()


# The envelope the policy actually flies on this stage, widened a little so the scan
# includes the states a bad episode reaches rather than only the intended ones.
DIMS = {
    "x":      (-TILE / 2, TILE / 2),
    "y":      (-TILE / 2, TILE / 2),
    "vx":     (-30.0, 30.0),
    "vy":     (-30.0, 30.0),
    "vz":     (-30.0, 5.0),
    "tilt_x": (-0.6, 0.6),
    "tilt_y": (-0.6, 0.6),
    "wx":     (-1.0, 1.0),
    "wy":     (-1.0, 1.0),
    "wz":     (-1.0, 1.0),
}
ALT = (0.0, 200.0)


def _reward(state, alt):
    info = {"altitude_m": alt, "terminated": False, "truncated": False,
            "action": np.zeros(4, dtype=np.float32)}
    return float(reward_fn(_Env(dict(state)), info))


def main():
    rng = np.random.default_rng(0)
    # What a single step of this reward is typically worth, measured rather than assumed,
    # so the jump floor below is in the reward's own units whatever the scaling becomes.
    probe = []
    for _ in range(512):
        st = {k: float(rng.uniform(*r)) for k, r in DIMS.items()}
        st["z"] = 0.0
        probe.append(abs(_reward(st, float(rng.uniform(*ALT)))))
    typical_r = float(np.median(probe))
    jump_floor = 1e-3 * typical_r
    worst = {k: (0.0, None) for k in list(DIMS) + ["altitude_m"]}
    moved = {k: 0 for k in list(DIMS) + ["altitude_m"]}
    slopes = {k: [] for k in list(DIMS) + ["altitude_m"]}
    jumps = []

    for _ in range(args.samples):
        base = {k: float(rng.uniform(*rng_)) for k, rng_ in DIMS.items()}
        base["z"] = 0.0
        alt = float(rng.uniform(*ALT))
        r0 = _reward(base, alt)
        for k in list(DIMS) + ["altitude_m"]:
            for h in (args.step, args.step / 10.0):
                if k == "altitude_m":
                    r1 = _reward(base, alt + h)
                else:
                    pert = dict(base)
                    pert[k] = base[k] + h
                    r1 = _reward(pert, alt)
                slope = abs(r1 - r0) / h
                if h == args.step:
                    coarse = slope
                else:
                    fine = slope
            slopes[k].append(coarse)
            if coarse > 1e-12:
                moved[k] += 1
            if coarse > worst[k][0]:
                worst[k] = (coarse, dict(base, altitude_m=alt))
            # the discontinuity test: does the slope grow as the step shrinks?
            # A RATIO ALONE IS NOT ENOUGH, and the first version of this scan reported
            # noise because of it: wherever a term's slope passes through zero (the
            # angular-rate penalty's in-flight regime, for one) two nearly equal doubles
            # are differenced and the ratio of the results is arbitrary. Those points had
            # slopes of 1e-6 against a typical 8e-3 -- four orders down.
            #
            # So the ratio is paired with the SIZE of the jump it implies. A
            # discontinuity of height d shows up as a fine slope of d/(step/10), so the
            # implied height is `fine * step/10`, and it only matters if it is a
            # meaningful fraction of what a step of this reward is worth at all.
            implied_jump = fine * (args.step / 10.0)
            if implied_jump > jump_floor and fine / coarse > args.ratio_flag:
                jumps.append((k, fine / coarse, coarse, fine,
                              dict(base, altitude_m=alt)))

    print(f"[reward-continuity] stage={args.stage} samples={args.samples} "
          f"step={args.step:g} (and {args.step / 10:g})")
    print(f"  a typical per-step reward here is {typical_r:.4g}, so a value jump is "
          f"only counted once it exceeds {jump_floor:.3g}\n")
    print(f"  {'dimension':<12} {'states that respond':>20} {'median |dr/dx|':>16} "
          f"{'max |dr/dx|':>14}")
    for k in list(DIMS) + ["altitude_m"]:
        s = np.asarray(slopes[k])
        print(f"  {k:<12} {moved[k] / args.samples:>19.1%} "
              f"{np.median(s):>16.4g} {s.max():>14.4g}")

    print(f"\n  DISCONTINUITIES (slope grew by more than {args.ratio_flag}x when the "
          f"step shrank 10x): {len(jumps)}")
    for k, ratio, coarse, fine, st in sorted(jumps, key=lambda t: -t[1])[:8]:
        where = ", ".join(f"{n}={st[n]:+.2f}" for n in ("altitude_m", "vz", "vx", "x"))
        print(f"    {k:<10} ratio {ratio:>6.1f}x  slope {coarse:.3g} -> {fine:.3g}   "
              f"at {where}")
    if not jumps:
        print("    none -- every sampled point is Lipschitz at both step sizes, so the "
              "per-step reward is continuous over the envelope (clips and max() show as "
              "a bounded kink, which is what the flat slope ratios above say).")

    dead = [k for k in slopes if moved[k] / args.samples < 0.01]
    if dead:
        print(f"\n  DIMENSIONS THE REWARD IGNORES: {dead} -- no dense gradient there.")


main()
