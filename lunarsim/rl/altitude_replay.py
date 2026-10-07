"""Sample the replay buffer where the landing is DECIDED, not where the time is spent.

THE MEASUREMENT THIS EXISTS FOR. On `orbit_descent` the critic's action gradient was
resolved by altitude, and it is worst exactly where the episode is won or lost and best
where it barely matters:

    altitude band     fraction of the buffer    grad points to expert    |grad_a Q|
    0 - 2 m                      14%                   54.3%               1.297
    80 - 150 m                   65%                   68.6%               0.118

A 200 m descent spends most of its steps cruising, so uniform sampling spends most of
the critic's capacity on the cruise. The decisive last two metres -- where the touchdown
criteria (v_z, v_xy, tilt, omega) are actually graded -- get 14% of the gradient steps,
and that is the band the critic is wrong in.

WHY THIS AND NOT PER. Prioritized experience replay (Schaul et al., arXiv:1511.05952)
prioritises by TD error, which needs per-transition priorities maintained through
training and introduces its own importance-sampling correction. The imbalance here is
not a TD-error accident -- it is a known, structural property of the task's geometry, and
it is observable directly from the stored observation (slot 2 IS the altitude). A fixed
stratification by altitude targets it without any priority bookkeeping, and it cannot
drift out of step with the critic the way stale priorities can.

ON THE BIAS. Reweighting the sampling distribution does bias the gradient away from the
on-policy visitation distribution, which is what PER's importance weights correct for.
That reweighting is the POINT here rather than a side effect: the Bellman target of each
individual transition is unchanged and unbiased, and we are deliberately choosing to
spend more of the critic's capacity on the band whose values decide the outcome. No IS
correction is applied, and this paragraph exists so that choice is visible rather than
accidental. What it costs is accuracy in the cruise band, which the measurement above
says is the band the critic is already good at.
"""
from __future__ import annotations

import numpy as np

from lunarsim.rl.dual_gamma import DualRewardReplayBuffer

# Observation slot 2 is the altitude, divided by this on its way into the policy
# (lunarsim/rl/obs_norm.OBS_SCALE[2]). Imported rather than hard-coded so a change to
# the scale cannot silently turn this buffer's altitude threshold into a different one.
from lunarsim.rl.obs_norm import OBS_SCALE

_ALT_SLOT = 2
_ALT_SCALE = float(OBS_SCALE[_ALT_SLOT])


class AltitudeStratifiedReplayBuffer(DualRewardReplayBuffer):
    """A dual-gamma buffer that guarantees the decisive band a share of every batch.

    `decisive_frac` of each batch is drawn from transitions below `decisive_alt_m`, the
    rest from above it. When the buffer does not hold enough low-altitude transitions to
    fill the quota (early in a run, before anything has reached the ground) the shortfall
    is filled from the rest of the buffer and the batch stays the requested size -- a
    short batch would change the effective learning rate instead of the distribution,
    which is not the intended knob.
    """

    def __init__(self, *args, decisive_alt_m: float = 5.0, decisive_frac: float = 0.5,
                 oversample: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < decisive_frac < 1.0:
            raise ValueError(f"decisive_frac must be in (0,1), got {decisive_frac}")
        self.decisive_alt_m = float(decisive_alt_m)
        self.decisive_frac = float(decisive_frac)
        self.oversample = int(oversample)
        self._short_warned = False
        if self.optimize_memory_usage:
            # With optimize_memory_usage the next observation is the NEXT position's
            # observation, so position `self.pos` must be excluded from any draw -- a
            # constraint the parent's sample() enforces and this override would have to
            # reimplement. Refuse rather than quietly sample a wrong next_obs.
            raise ValueError(
                "AltitudeStratifiedReplayBuffer does not support "
                "optimize_memory_usage=True (it would need the parent's pos-exclusion "
                "logic to stay correct); pass optimize_memory_usage=False")

    def _altitudes(self, pos_inds, env_inds) -> np.ndarray:
        """Altitude in METRES for the given (position, env) pairs.

        Read back out of the stored observation and un-normalised, so there is no second
        copy of the altitude to drift out of step with the one the policy sees.
        """
        return self.observations[pos_inds, env_inds, _ALT_SLOT] * _ALT_SCALE

    def sample(self, batch_size: int, env=None):  # type: ignore[override]
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return super().sample(batch_size, env=env)

        n_cand = max(batch_size * self.oversample, batch_size)
        cand_pos = np.random.randint(0, upper, size=n_cand)
        cand_env = np.random.randint(0, self.n_envs, size=n_cand)
        alt = self._altitudes(cand_pos, cand_env)
        is_low = alt <= self.decisive_alt_m

        want_low = int(round(batch_size * self.decisive_frac))
        low_idx = np.flatnonzero(is_low)
        high_idx = np.flatnonzero(~is_low)

        take_low = min(want_low, low_idx.size)
        take_high = min(batch_size - take_low, high_idx.size)
        chosen = np.concatenate([low_idx[:take_low], high_idx[:take_high]])

        if take_low < want_low and not self._short_warned:
            # THE QUOTA WENT UNMET. This is the condition worth announcing, not a short
            # batch: when the decisive band cannot fill its share the high band quietly
            # fills the rest, the batch comes back the right size, and the run looks
            # exactly like one where stratification was applied. Saying so once is the
            # difference between "no low-altitude data yet" and "stratification is on".
            print(f"[altitude-replay] quota UNMET: wanted {want_low}/{batch_size} below "
                  f"{self.decisive_alt_m} m, found {low_idx.size} among {n_cand} "
                  f"candidates. Expected before anything reaches the ground, "
                  f"suspicious after.", flush=True)
            self._short_warned = True

        short = batch_size - chosen.size
        if short > 0:
            # Both bands short among the candidates (only reachable with a tiny buffer).
            # Fill uniformly so the batch keeps its size: a short batch would change the
            # effective learning rate rather than the distribution.
            extra_pos = np.random.randint(0, upper, size=short)
            extra_env = np.random.randint(0, self.n_envs, size=short)
            pos_inds = np.concatenate([cand_pos[chosen], extra_pos])
            env_inds = np.concatenate([cand_env[chosen], extra_env])
            return self._get_samples_pairs(pos_inds, env_inds, env)
        return self._get_samples_pairs(cand_pos[chosen], cand_env[chosen], env)
