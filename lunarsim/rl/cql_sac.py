"""SAC with CQL's conservative critic term.

WHY THIS EXISTS, measured. Twenty-two interventions in this project went into the
ACTOR -- learning rates, optimisers, freezes, policy delay, two-timescale updates,
and finally a parameter-space trust region around the warm start. The trust region
is the only one that held anything, and its two ends bracket the problem without
solving it:

    beta    pull half-life    outcome
    1e-2         69           frozen at the warm start: 53% vs the clone's 55%
    1e-3        693           dead in ONE 10k window (625 actor updates)

The proxy log from the 1e-3 run says why, and it is not an actor problem:

    step 50000  landed 47.4%  proxy  14.07  Q~-2   <- actor frozen: this is the clone
    step 60000  landed  0.0%  proxy -10.03  Q~+1
    step 80000  landed  0.0%  proxy -21.08  Q~+5
    step 90000  landed  0.0%  proxy -44.83  Q~+6

Q is POSITIVE while the true shaped return is -10 to -45 -- wrong in sign, not
merely in scale -- while critic_loss stays at 0.2-2.3. So this is not the
divergence earlier runs showed. The critic is quietly, confidently wrong about the
actions the actor has just begun proposing, the actor ascends that phantom
gradient, and competence is gone in 625 updates. A trust region cannot fix a critic
that misranks the actions at the end of the walk; it can only shorten the walk,
which is exactly the all-or-nothing behaviour the two beta ends show.

CQL (Kumar et al., arXiv:2006.04779) attacks that directly: push Q DOWN on actions
the policy proposes and on uniform samples, while holding it UP on the actions
actually present in the buffer. The out-of-distribution values the actor would
chase stop being free.

CQL(H), continuous-action form (the paper's Appendix F), per critic:

    loss += alpha * ( logsumexp over sampled a of [Q(s,a) - log mu(a)]
                      - Q(s, a_buffer) )

with the sampled set drawn half from the current policy (importance weight
log pi(a|s)) and half uniform over the action box (log mu = -d*log 2). Cal-QL
(arXiv:2303.05479) refines this by clamping the OOD values from below at a
reference value so the penalty cannot become so conservative that online
improvement stalls.

MEASURED OUTCOME of plain CQL here, which is why the clamp now exists. The collapse
was solved: deterministic 96-episode rates went from 0.0%/4.2% without the term to
49.5% pooled at alpha=5, and Q never turned positive in the release window again.
But there was no improvement on the warm start (clone 50.5% pooled), and the reason
was visible -- Q ended at -21 (alpha=1) and -32 (alpha=5) while the true shaped
return was +13 to +20. The demonstrator's own discounted V(s0), over the 48 recorded
ramp_35m episodes in training reward units (reward_total_scale=0.1, gamma=0.995), is
mean +5.34 (median 5.94, p10 0.86, max 7.53). So CQL had pushed Q 26 to 37 units
BELOW what the reference policy actually achieves, leaving no gradient worth
climbing.

`calql_ref` is a SCALAR, which is an approximation: the paper uses a state-dependent
V_ref from Monte-Carlo returns. It is defensible here because the measured spread of
V(s0) (-0.21 to 7.53) is small next to the 26-37 unit error being corrected, and a
scalar needs no second network to go wrong. 0.0 is the conservative choice, keeping
CQL's lower bound almost everywhere while removing the pathology; the measured mean
5.34 is the aggressive one.
"""
from __future__ import annotations

import math

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class CQLSAC(SAC):
    """SAC whose critic loss carries CQL(H)'s conservative term.

    `train()` is SB3 2.9.0's SAC.train() copied verbatim with two insertions --
    the conservative term and its logging -- so that nothing else in the update
    silently diverges from the base class this project's other results were
    produced with.
    """

    def __init__(self, *args, cql_alpha: float = 5.0, cql_n_samples: int = 10,
                 calql_ref: float | None = None,
                 dual_gamma: tuple[float, float] | None = None,
                 td3bc_alpha: float | None = None,
                 pex_temperature: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = float(cql_alpha)
        self.cql_n_samples = int(cql_n_samples)
        # None disables the Cal-QL clamp and leaves plain CQL(H).
        self.calql_ref = None if calql_ref is None else float(calql_ref)
        # (gamma_shaping, gamma_terminal) or None for single-gamma SAC. See
        # lunarsim/rl/dual_gamma.py for the measurement that motivates it.
        self.dual_gamma = None if dual_gamma is None else (float(dual_gamma[0]),
                                                           float(dual_gamma[1]))
        # PEX (Zhang, Xu, Yu -- arXiv:2302.00935). `pex_actor` is a FROZEN copy of the
        # warm-start actor; `pex_temperature` is the softmax temperature for choosing
        # between its action and the learnable actor's. None disables the whole thing.
        self.pex_temperature = None if pex_temperature is None else float(pex_temperature)
        self.pex_actor = None
        # TD3+BC's actor regulariser (Fujimoto & Gu, arXiv:2106.06860), in ACTION space.
        # This is the paper's ALPHA, and the direction is counter-intuitive enough that it
        # cost me a wrong test: lambda = alpha / mean|Q| scales the Q TERM, so a LARGER
        # alpha means MORE Q and a WEAKER behaviour constraint. The paper uses 2.5. None
        # disables the term.
        self.td3bc_alpha = None if td3bc_alpha is None else float(td3bc_alpha)
        if self.dual_gamma is not None:
            n = len(self.critic.q_networks)
            if n < 4 or n % 2 != 0:
                raise ValueError(
                    f"dual_gamma needs an even n_critics >= 4 so each stream keeps a "
                    f"pair to take the min over; got n_critics={n}")
            self._n_half = n // 2

    def _conservative_term(self, observations, current_q_values) -> th.Tensor:
        """logsumexp over OOD actions minus Q on the buffer's own actions."""
        batch = observations.shape[0]
        n = self.cql_n_samples
        dim = int(np.prod(self.action_space.shape))
        obs_rep = observations.repeat_interleave(n, dim=0)

        # No grad through the sampling: this term updates the CRITIC, and letting
        # it reach the actor would turn the penalty into an actor objective too.
        with th.no_grad():
            a_pi, logp_pi = self.actor.action_log_prob(obs_rep)
            a_unif = th.rand(batch * n, dim, device=observations.device) * 2.0 - 1.0
        log_unif = -dim * math.log(2.0)

        q_pi_all = self.critic(obs_rep, a_pi)
        q_unif_all = self.critic(obs_rep, a_unif)

        total = th.zeros((), device=observations.device)
        for q_pi, q_unif, q_data in zip(q_pi_all, q_unif_all, current_q_values):
            # Cal-QL: clamp the OOD values from BELOW at a reference before the
            # logsumexp, so the penalty cannot drive them under what the reference
            # policy actually achieves. Plain CQL drove Q to -21 and -32 here while
            # the demonstrator's own discounted V(s0) is +5.34 (mean over 48
            # recorded episodes, training reward units) -- 26 to 37 units of pure
            # over-conservatism, which left the actor nothing to climb and produced
            # exactly the measured "no collapse, no improvement" result.
            if self.calql_ref is not None:
                q_pi = th.clamp(q_pi, min=self.calql_ref)
                q_unif = th.clamp(q_unif, min=self.calql_ref)
            # subtract the proposal log-density, per CQL(H)'s importance weighting
            qp = q_pi.view(batch, n) - logp_pi.view(batch, n)
            qu = q_unif.view(batch, n) - log_unif
            lse = th.logsumexp(th.cat([qp, qu], dim=1), dim=1, keepdim=True)
            total = total + (lse - q_data).mean()
        return total

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        cql_terms = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            if self.dual_gamma is None:
                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values)
                                        for current_q in current_q_values)
            else:
                # ---- DUAL DISCOUNT ----
                # The ensemble is split in half: the first `_n_half` networks back up the
                # SHAPING stream at gamma_shaping, the rest back up the TERMINAL stream at
                # gamma_terminal. Each half keeps a pair so clipped-double-Q still applies
                # within a stream.
                #
                # The entropy bonus goes on the SHAPING target only. It is a dense
                # per-step quantity, so discounting it at the terminal's gamma would be
                # wrong, and adding it to both would count it twice.
                g_s, g_t = self.dual_gamma
                r_term = replay_data.rewards_terminal
                r_shape = replay_data.rewards - r_term
                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    nq = self.critic_target(replay_data.next_observations, next_actions)
                    nq_s, _ = th.min(th.cat(nq[:self._n_half], dim=1), dim=1, keepdim=True)
                    nq_t, _ = th.min(th.cat(nq[self._n_half:], dim=1), dim=1, keepdim=True)
                    nq_s = nq_s - ent_coef * next_log_prob.reshape(-1, 1)
                    tgt_s = r_shape + (1 - replay_data.dones) * g_s * nq_s
                    tgt_t = r_term + (1 - replay_data.dones) * g_t * nq_t

                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * (
                    sum(F.mse_loss(q, tgt_s) for q in current_q_values[:self._n_half])
                    + sum(F.mse_loss(q, tgt_t) for q in current_q_values[self._n_half:]))

            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # ---- INSERTION 1: CQL(H)'s conservative term ----
            if self.cql_alpha > 0.0:
                cql_term = self._conservative_term(replay_data.observations, current_q_values)
                critic_loss = critic_loss + self.cql_alpha * cql_term
                cql_terms.append(cql_term.item())
            # -------------------------------------------------

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            qpi = self.critic(replay_data.observations, actions_pi)
            if self.dual_gamma is None:
                min_qf_pi, _ = th.min(th.cat(qpi, dim=1), dim=1, keepdim=True)
            else:
                # The actor maximises the SUM of the two streams, which is the quantity
                # the environment actually pays. Each stream contributes its own min.
                qs, _ = th.min(th.cat(qpi[:self._n_half], dim=1), dim=1, keepdim=True)
                qt, _ = th.min(th.cat(qpi[self._n_half:], dim=1), dim=1, keepdim=True)
                min_qf_pi = qs + qt
            if self.td3bc_alpha is not None:
                # TD3+BC (arXiv:2106.06860): keep the actor near actions the DATA
                # supports, in action space, rather than near the parameters it started
                # from. The two are not interchangeable and this project only had the
                # second (`--bc-anchor`, a proximal pull on the weights).
                #
                # Measured reason for needing this. At v65's 60k checkpoint the critic's
                # VALUE is calibrated -- Q0 against the realised discounted return reads
                # +0.02 to +0.96 after dual-gamma, CQL and the Cal-QL clamp brought it
                # down from +20..+51 -- and yet the policy fell from 75% to 4.5% in one
                # window. Measuring the gradient instead of the value says why:
                #
                #   cos(grad_a Q, a_expert - pi(s))  mean -0.094, median -0.097
                #   fraction pointing toward the expert: 40.8%  (a coin flip is 50%)
                #   the clone's own untrained critic: -0.025, 47.6%
                #
                # So forty thousand steps of critic fitting left the action gradient
                # LESS informative about the 91% behaviour than an untrained critic's,
                # while the values were nearly exact. A calibrated value is not a correct
                # gradient: Q is fitted to predict returns ON the data, but its SHAPE in
                # action space around each state's single recorded action is almost
                # unconstrained by that data. This is the standard offline-RL difficulty
                # and the reason BCQ and TD3+BC constrain the action instead of trusting
                # grad_a Q.
                #
                # lambda normalises by the Q scale, as in the paper, so alpha means the
                # same thing whatever the reward scale is -- which matters here because
                # reward_total_scale has already moved twice today. LOWER alpha = stronger
                # behaviour constraint.
                lam = self.td3bc_alpha / (min_qf_pi.abs().mean().detach() + 1e-6)
                bc_err = ((actions_pi - replay_data.actions) ** 2).sum(dim=1).mean()
                actor_loss = (ent_coef * log_prob - lam * min_qf_pi).mean() + bc_err
            else:
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # ---- INSERTION 2: so the penalty's size is visible in the proxy log ----
        if cql_terms:
            self.logger.record("train/cql_term", np.mean(cql_terms))


    # ------------------------------------------------------------------ #
    # PEX: policy expansion (arXiv:2302.00935)
    # ------------------------------------------------------------------ #

    def install_pex(self, temperature: float) -> None:
        """Freeze a copy of the current actor as the retained offline policy.

        WHY THIS SHAPE, from this project's own measurements. Every intervention that
        stopped the warm start from being destroyed also stopped it from improving:

            anchor beta=1e-2 (69-update half-life)  frozen at the clone, no learning
            anchor beta=3e-3, 1e-3                  dead in one 10k window
            CQL alpha=5                             no collapse, no improvement
                                                    (49.5% against the clone's 50.5%)

        That is a systematic trade-off rather than bad luck, and the cause is structural:
        ONE network had to be both the competent fallback and the explorer. Constrain it
        and it cannot explore; release it and an uninformative action gradient destroys it.
        There is no setting in between because the problem is not in the knob.

        PEX names this exactly -- it calls what we were doing the "Direct" method, "which
        has the potential of destroying useful behaviors learned offline" -- and splits the
        two roles instead: freeze pi_beta, add a learnable pi_theta, and choose between
        their proposals per state with a categorical distribution over Q.

        The reason it fits OUR measurements specifically, which is what makes it more than
        one more method to try:

            the critic's VALUE    is calibrated    Q0 vs realised discounted return ~ 0
            the critic's GRADIENT is not           40.8% toward the expert, vs 47.6% for
                                                   the clone's own untrained critic

        Selection needs Q only to RANK two concrete actions. It never differentiates Q with
        respect to the action. So PEX uses precisely the capability we measured ourselves to
        have and avoids precisely the one we measured ourselves to lack -- where the anchor,
        CQL and TD3+BC all trust grad_a Q and merely try to restrain it.

        Structural consequences, none of which need tuning: the clone cannot be degraded,
        so collapse is impossible by construction; pi_theta may be arbitrarily bad at no
        cost because it simply is not selected; and the floor is the clone's own
        performance. The paper's ablation confirms the freeze carries the result -- training
        pi_beta alongside pi_theta gives "a clear performance drop".
        """
        import copy
        self.pex_actor = copy.deepcopy(self.actor)
        self.pex_actor.set_training_mode(False)
        for prm in self.pex_actor.parameters():
            prm.requires_grad_(False)
        self.pex_temperature = float(temperature)

    def _pex_q(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """The same scalar the actor ascends, so selection and learning agree."""
        qs = self.critic(obs, act)
        if self.dual_gamma is not None:
            return (th.cat(qs[:self._n_half], dim=1).min(dim=1).values
                    + th.cat(qs[self._n_half:], dim=1).min(dim=1).values)
        return th.cat(qs, dim=1).min(dim=1).values

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if self.pex_actor is None or self.pex_temperature is None:
            return super().predict(observation, state, episode_start, deterministic)
        # Both policies propose, the critic ranks, a categorical over Q picks. Equation (5)
        # of the paper; `deterministic` is honoured WITHIN each member policy, and the
        # choice between them stays stochastic because that is what lets pi_theta be tried
        # at all. At temperature -> 0 it becomes argmax.
        obs_t, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            a_beta = self.pex_actor(obs_t, deterministic=deterministic)
            a_theta = self.actor(obs_t, deterministic=deterministic)
            q_beta = self._pex_q(obs_t, a_beta)
            q_theta = self._pex_q(obs_t, a_theta)
            logits = th.stack([q_beta, q_theta], dim=1) / max(self.pex_temperature, 1e-6)
            pick = th.distributions.Categorical(logits=logits).sample()
            chosen = th.where(pick.unsqueeze(1).bool(), a_theta, a_beta)
        actions = chosen.cpu().numpy().reshape((-1, *self.action_space.shape))
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _excluded_save_params(self):
        return super()._excluded_save_params()

    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        if self.pex_actor is not None:
            # so a saved checkpoint is a self-contained COMPOSITE policy; without this an
            # evaluation would silently load pi_theta alone and measure the wrong thing
            state_dicts = list(state_dicts) + ["pex_actor"]
        return state_dicts, tensors
