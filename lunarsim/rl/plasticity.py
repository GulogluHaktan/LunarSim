"""Plasticity tooling for SAC: LayerNorm policies, periodic partial resets,
and the dormant-neuron metric.

Why these three, and in this order. handover.md records four refuted knobs
(gamma, replay ratio, learning rate, entropy coefficient) and then a literature
review (rl-cokus-literatur-taramasi.md) that reframed the problem: the Klein et
al. (2024) survey puts overestimation bias DOWNSTREAM of plasticity loss, and
its headline practical finding is that general regularisation usually beats
domain-specific interventions -- hence LayerNorm first.

Measured caveat, recorded here so it is not forgotten: on this project's own
checkpoints the dormant-neuron fraction does NOT separate the good policy
(22.9%, actor 26.0% dormant) from the collapsed ones (0%, 25.0-28.9%), and the
v30 checkpoint collapsed with its weight norm and feature rank unchanged. So
ReDo-style recycling is not expected to be the fix here; `dormant_fraction` is
provided as a DIAGNOSTIC to keep measuring, not as a promise.

References:
  Nikishin et al. 2022, The Primacy Bias in Deep RL (arXiv:2205.07802)
  Sokar et al. 2023, The Dormant Neuron Phenomenon / ReDo (arXiv:2302.12902)
  Lyle et al. 2024, Normalization and effective learning rates (arXiv:2407.01800)
  Klein et al. 2024, Plasticity Loss in Deep RL: A Survey (arXiv:2411.04832)
  McCutcheon et al. 2026, Calibrated Partial Resets (arXiv:2607.24996)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1. LayerNorm
# --------------------------------------------------------------------------- #

def layer_norm_after_relu(module: nn.Module) -> nn.Module:
    """Insert LayerNorm after every Linear->activation pair in an nn.Sequential.

    SB3's `create_mlp` emits `[Linear, act, Linear, act, ...]` with no
    normalisation, and MlpPolicy exposes no option to add it, so the layers
    have to be woven in after construction.
    """
    if not isinstance(module, nn.Sequential):
        return module
    out: list[nn.Module] = []
    for i, layer in enumerate(module):
        out.append(layer)
        is_act = isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.GELU, nn.SiLU))
        prev_linear = i > 0 and isinstance(module[i - 1], nn.Linear)
        if is_act and prev_linear:
            out.append(nn.LayerNorm(module[i - 1].out_features))
    return nn.Sequential(*out)


def add_layer_norm_to_sac(model) -> int:
    """Rebuild a SAC model's actor and critic trunks with LayerNorm.

    Call this on a FRESH model, before any training: it reinitialises the
    inserted layers, and the optimizers have to be rebuilt to see the new
    parameters.
    """
    n = 0
    actor = model.policy.actor
    if hasattr(actor, "latent_pi"):
        actor.latent_pi = layer_norm_after_relu(actor.latent_pi)
        n += 1
    for critic in (model.policy.critic, model.policy.critic_target):
        for i, qnet in enumerate(critic.q_networks):
            new = layer_norm_after_relu(qnet)
            critic.q_networks[i] = new
            setattr(critic, f"qf{i}", new)
            n += 1
    model.policy.critic_target.load_state_dict(model.policy.critic.state_dict())
    # the optimizers were built over the OLD parameter list; rebuild or the
    # LayerNorm parameters never receive a gradient step
    lr = model.lr_schedule(1.0)
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr)
    return n


# --------------------------------------------------------------------------- #
# 2. Dormant neurons (diagnostic)
# --------------------------------------------------------------------------- #

def dormant_fraction(module: nn.Module, *inputs, tau: float = 0.025) -> float:
    """ReDo's metric: the fraction of ReLU units whose mean |activation| over a
    batch is below `tau` times their layer's mean.

    Feed it REAL rollout observations. Measured on random Gaussian inputs it
    still returned a stable number on this project's checkpoints, but the
    metric is defined on the on-policy distribution and random inputs are a
    weaker probe.
    """
    acts: dict[str, torch.Tensor] = {}
    hooks = []

    def mk(name):
        def hook(_m, _i, o):
            acts[name] = o.detach()
        return hook

    for name, m in module.named_modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        module(*inputs)
    for h in hooks:
        h.remove()
    if not acts:
        return float("nan")
    fracs = []
    for a in acts.values():
        score = a.abs().mean(0)
        mean = score.mean()
        fracs.append(1.0 if mean <= 0 else float((score <= tau * mean).float().mean()))
    return float(np.mean(fracs))


# --------------------------------------------------------------------------- #
# 3. Periodic partial resets
# --------------------------------------------------------------------------- #

def _shrink_perturb(layer: nn.Linear, alpha: float, gen: torch.Generator) -> None:
    """Pull weights toward a fresh initialisation by `alpha`.

    alpha=1 is a full reset, alpha=0 a no-op. Partial pulls are what Calibrated
    Partial Resets (2026) uses to avoid the destabilisation -- and the policy
    collapse -- that full reinitialisation can cause.
    """
    fresh = nn.Linear(layer.in_features, layer.out_features)
    with torch.no_grad():
        w = torch.empty_like(layer.weight)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=gen)
        layer.weight.mul_(1.0 - alpha).add_(w, alpha=alpha)
        if layer.bias is not None:
            layer.bias.mul_(1.0 - alpha).add_(fresh.bias.to(layer.bias.device),
                                              alpha=alpha)


def partial_reset_sac(model, alpha: float = 1.0, scope: str = "last",
                      seed: int = 0) -> int:
    """Reset part of a SAC agent.

    `scope`:
      "last"   -- the final layer of the actor and of every critic head. The
                  cheapest useful version: Nikishin et al. reset a PART of the
                  agent, and last-layer resetting on its own is reported to
                  help continual and transfer learning (arXiv:2310.07996).
      "critic" -- every critic head in full, leaving the actor alone. Ma et al.
                  (2023) find the modules differ in how much their plasticity
                  matters, so this is separable on purpose.
      "all"    -- actor trunk and critic heads.

    Returns the number of layers touched. The caller is responsible for the
    snapshot discipline around it: a reset costs a temporary performance drop
    (AltNet 2026 says so explicitly), so the best checkpoint must be saved
    BEFORE one.
    """
    gen = torch.Generator().manual_seed(seed)
    touched = 0

    def last_linear(seq):
        lin = [m for m in seq.modules() if isinstance(m, nn.Linear)]
        return lin[-1] if lin else None

    if scope in ("last", "all"):
        if getattr(model.policy.actor, "mu", None) is not None:
            _shrink_perturb(model.policy.actor.mu, alpha, gen)
            touched += 1
        for q in model.policy.critic.q_networks:
            lin = last_linear(q)
            if lin is not None:
                _shrink_perturb(lin, alpha, gen)
                touched += 1
    if scope in ("critic", "all"):
        for q in model.policy.critic.q_networks:
            for m in q.modules():
                if isinstance(m, nn.Linear):
                    _shrink_perturb(m, alpha, gen)
                    touched += 1
    if scope == "all":
        for m in model.policy.actor.latent_pi.modules():
            if isinstance(m, nn.Linear):
                _shrink_perturb(m, alpha, gen)
                touched += 1

    model.policy.critic_target.load_state_dict(model.policy.critic.state_dict())
    return touched


try:
    from stable_baselines3.common.callbacks import BaseCallback

    class PeriodicResetCallback(BaseCallback):
        """Reset part of the agent every `every` steps during training.

        Pairs with --checkpoint-every: snapshots taken between resets are what
        you actually evaluate, because the steps right after a reset are
        expected to be bad.
        """

        def __init__(self, every: int, alpha: float = 1.0, scope: str = "last",
                     seed: int = 0, verbose: int = 1):
            super().__init__(verbose)
            self.every = int(every)
            self.alpha = float(alpha)
            self.scope = scope
            self.seed = seed
            self._next = None

        def _on_training_start(self) -> None:
            self._next = self.model.num_timesteps + self.every

        def _on_step(self) -> bool:
            if self.every <= 0 or self._next is None:
                return True
            if self.model.num_timesteps >= self._next:
                n = partial_reset_sac(self.model, self.alpha, self.scope,
                                      self.seed + self.model.num_timesteps)
                if self.verbose:
                    print(f"[reset] step {self.model.num_timesteps}: "
                          f"{self.scope} alpha={self.alpha} -> {n} layers",
                          flush=True)
                self._next = self.model.num_timesteps + self.every
            return True

except Exception:  # pragma: no cover - SB3 absent (e.g. the docs venv)
    PeriodicResetCallback = None  # type: ignore
