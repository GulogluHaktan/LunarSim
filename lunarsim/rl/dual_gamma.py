"""Two discount rates: a low one for shaping, a high one for the terminal.

WHY. arXiv:1810.08719 (Gaudet, Linares, Furfaro) lists multiple discount rates as a
primary contribution and is blunt about the cost of going without:

    "it is advantageous to use a relatively large discount rate [with a terminal
     reward]. However, it is also advantageous to use a lower discount rate for the
     shaping rewards... Without the use of multiple discount rates, the performance
     was actually worsened by including the terminal reward term."

That is this project's gamma sweep, described from the outside before we ran it:

    gamma    terminal's share of V    outcome
    0.990    0.0109 over 450 steps    critic bounded in [-14,+6], terminal invisible
    0.995    0.1048                   measured, still failed
    0.998    0.4062                   terminal visible, critic diverged to +100..+1972

Three single-gamma settings, each trading the critic against the terminal, and the
reason no value worked is that one gamma has to serve two reward streams with opposite
requirements. The terminal arrives once, ~450 steps out, so it needs gamma close to 1 to
survive the discount. The shaping arrives every step, so a high gamma amplifies its
per-step error by 1/(1-gamma) -- 200 at gamma 0.995 -- which is the divergence.

HOW. The reward function publishes `info["reward_terminal"]` and
`info["reward_shaping"]`; this module stores the terminal stream alongside the total in
the replay buffer, and splits SAC's critic ensemble into two independent halves that
back up their own stream with their own gamma. The actor maximises the SUM, which is the
quantity the environment actually pays.

The entropy bonus is attached to the SHAPING head only. SAC's maximum-entropy objective
adds -alpha*log(pi) at every step, which is a dense per-step quantity and belongs on the
dense stream; putting it on both would double-count it, and putting it on the terminal
head would discount it by the terminal's gamma, which is wrong for a per-step term.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer


class DualRewardSamples(NamedTuple):
    """SB3's ReplayBufferSamples with the terminal stream added.

    Field names and order match `ReplayBufferSamples` so that anything unpacking or
    attribute-accessing the standard fields is unaffected; `rewards_terminal` is appended
    after them.
    """
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor | None
    rewards_terminal: th.Tensor
    # The EXACT terminal-stream value, as two ingredients rather than a number, because
    # the buffer does not know gamma_terminal: the episode's terminal reward and how many
    # steps remained until it arrived. `mc_valid` is 0 for transitions whose episode had
    # not ended when they were sampled, which have no exact target yet.
    mc_terminal_r: th.Tensor | None = None
    mc_steps: th.Tensor | None = None
    mc_valid: th.Tensor | None = None


class DualRewardReplayBuffer(ReplayBuffer):
    """Stores the terminal reward stream next to the total.

    Only the TERMINAL stream is stored; shaping is recovered as
    `rewards - rewards_terminal`. Keeping the total authoritative means a split that
    fails to sum cannot silently drift, which matters because `reward_fn` is the single
    place the two are computed.

    `_get_samples` is SB3 2.9.0's body reimplemented rather than wrapped, because the
    parent draws its `env_indices` INSIDE the method: calling super() and then drawing
    again would pair each terminal reward with a different transition than the one it
    belongs to, and the result would look like a plausible-but-wrong terminal head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards_terminal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._missing_info_warned = False
        # ---- the EXACT terminal-stream value, filled in when the episode ends ----
        #
        # WHY. The terminal stream has NO intermediate rewards: its value is exactly
        # r_term * gamma_t^(steps remaining). Bootstrapping it is therefore solving by
        # iteration something that is available in closed form, and at gamma_t = 0.999
        # that iteration contracts by 0.1% per step -- it barely contracts at all over a
        # 700-step episode, so any per-step bias lands in the value multiplied by 1000.
        # That is measured: the conservative term's +0.0123/step showed up as +12.3 in
        # the terminal head while the true value there is about +0.5, and the terminal
        # head was 89% of the critic's total error.
        #
        # So the ingredients of the exact target are retro-filled once the episode's
        # outcome is known. `mc_valid` marks the transitions that have one.
        self.mc_terminal_r = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.mc_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.mc_valid = np.zeros((self.buffer_size, self.n_envs), dtype=np.uint8)
        # positions written since each env's last episode boundary, oldest first
        self._pending: list[list[int]] = [[] for _ in range(self.n_envs)]

    def add(self, obs, next_obs, action, reward, done, infos):  # type: ignore[override]
        pos = self.pos                      # super().add advances it
        term = np.zeros(self.n_envs, dtype=np.float32)
        for i in range(self.n_envs):
            info = infos[i] if infos is not None and i < len(infos) else {}
            if "reward_terminal" in info:
                term[i] = float(info["reward_terminal"])
            elif not self._missing_info_warned:
                # Loud, not silent. A missing key would leave the terminal head learning
                # from an all-zero stream and the run would read as "dual gamma did not
                # help" -- the exact class of mistake that invalidated four conclusions
                # in this project.
                print("[dual-gamma] WARNING: info carries no 'reward_terminal'; the "
                      "terminal head will see zeros. Is this reward_fn current?",
                      flush=True)
                self._missing_info_warned = True
        super().add(obs, next_obs, action, reward, done, infos)
        self.rewards_terminal[pos] = term

        # This position now belongs to an in-flight episode, so whatever exact target it
        # held from a previous pass through the ring is no longer valid.
        self.mc_valid[pos, :] = 0
        done_arr = np.atleast_1d(done)
        for i in range(self.n_envs):
            self._pending[i].append(pos)
            # A pending list longer than the buffer means the ring has wrapped inside a
            # single episode and the oldest entries have been overwritten by it; drop
            # them rather than filling a position that no longer belongs to this episode.
            if len(self._pending[i]) > self.buffer_size:
                del self._pending[i][:-self.buffer_size]
            if not bool(done_arr[i] if i < len(done_arr) else False):
                continue
            seq = self._pending[i]
            r_fin = float(term[i])
            for k, pp in enumerate(reversed(seq)):
                self.mc_terminal_r[pp, i] = r_fin
                self.mc_steps[pp, i] = k
                self.mc_valid[pp, i] = 1
            self._pending[i] = []

    def _get_samples(self, batch_inds, env=None):  # type: ignore[override]
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        return self._get_samples_pairs(batch_inds, env_indices, env)

    def _get_samples_pairs(self, batch_inds, env_indices, env=None):
        """The body above, but with the (position, env) PAIRS given rather than drawn.

        A transition in this buffer is identified by a pair, not by a position: the
        parent draws `env_indices` itself, so any subclass that wants to CHOOSE which
        transitions to sample -- `AltitudeStratifiedReplayBuffer` does -- has to supply
        both halves or it would re-randomise the env half and pair each terminal reward
        with a different transition than the one it belongs to.
        """
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices]
             * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        tensors = tuple(map(self.to_torch, data))
        term = self.to_torch(
            self._normalize_reward(
                self.rewards_terminal[batch_inds, env_indices].reshape(-1, 1), env))
        mc_r = self.to_torch(
            self._normalize_reward(
                self.mc_terminal_r[batch_inds, env_indices].reshape(-1, 1), env))
        mc_n = self.to_torch(
            self.mc_steps[batch_inds, env_indices].astype(np.float32).reshape(-1, 1))
        mc_v = self.to_torch(
            self.mc_valid[batch_inds, env_indices].astype(np.float32).reshape(-1, 1))
        return DualRewardSamples(*tensors, discounts=None, rewards_terminal=term,
                                 mc_terminal_r=mc_r, mc_steps=mc_n, mc_valid=mc_v)
