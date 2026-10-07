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
                 calql_ref: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = float(cql_alpha)
        self.cql_n_samples = int(cql_n_samples)
        # None disables the Cal-QL clamp and leaves plain CQL(H).
        self.calql_ref = None if calql_ref is None else float(calql_ref)

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

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
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

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
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
