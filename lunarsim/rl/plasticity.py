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


class ActorFreezeCallback(BaseCallback):  # type: ignore[misc]
    """Hold the actor still for the first `steps` env steps of training.

    A warm start pairs a COMPETENT actor with a RANDOMLY INITIALISED critic. The
    critic outputs ~0 while the true value range spans hundreds (terminal
    rewards here are -360..+450 against per-step shaping of ~0.5), so its first
    TD targets are enormous and its early Q-landscape is noise. SAC updates the
    actor against that critic from the very first gradient step, so a good
    policy is gradient-ascended on noise before the critic knows anything.

    Measured on this project: clones landing 31%, 38% and 58% were each warm
    started into SAC and each hit 0.0% in EVERY logged window, collapsing inside
    the first 10k steps. Neither an on-policy buffer warmup nor restoring the
    policy's action coverage prevented it.

    Freezing sets `requires_grad=False` on the actor's parameters. Zeroing the
    actor optimizer's learning rate does NOT work and silently does nothing:
    SB3 calls `_update_learning_rate` at the top of every `train()` call, which
    writes the schedule's value back into every param group, and a callback can
    only re-zero it between ENV steps -- so all `gradient_steps` updates in
    between run at the normal rate. A "frozen" run done that way is not frozen,
    which cost one experiment here before it was caught.
    """

    def __init__(self, steps: int, verbose: int = 1):
        super().__init__(verbose)
        self.steps = int(steps)
        self._until = None
        self._frozen = False
        self._saved_lr_schedule = None

    def _on_training_start(self) -> None:
        self._until = self.model.num_timesteps + self.steps

    def _set_actor_grad(self, flag: bool) -> None:
        for prm in self.model.actor.parameters():
            prm.requires_grad_(flag)

    def _on_step(self) -> bool:
        if self._until is None:
            return True
        if not self._frozen and self.model.num_timesteps < self._until:
            self._set_actor_grad(False)
            self._frozen = True
            if self.verbose:
                print(f"[freeze] actor held until step {self._until}", flush=True)
        elif self._frozen and self.model.num_timesteps >= self._until:
            self._set_actor_grad(True)
            self._frozen = False
            if self.verbose:
                print(f"[freeze] actor released at step "
                      f"{self.model.num_timesteps}", flush=True)
        return True


class ActorLearningRateCallback(BaseCallback):  # type: ignore[misc]
    """Give the actor its own, slower learning rate than the critic.

    SAC drives both networks from one `lr_schedule`, so actor and critic move on
    the SAME timescale. Actor-critic convergence results assume the opposite:
    the actor must be the slower of the two (two-timescale stochastic
    approximation -- Borkar; Konda & Tsitsiklis), so that it ascends a value
    function which has had time to settle rather than chasing a moving one.

    Measured on this project: after 60k steps of CRITIC-ONLY training the critic
    still valued a landing policy at Q ~ -14, when its true return is positive
    and in the hundreds. Releasing the actor then flipped Q to +100 while the
    landing rate went from 52% to 0% within a single 10k-step window -- the
    actor found what the critic overvalued, and the critic followed it there.
    Lowering the single shared rate to protect the policy had been crippling the
    critic at the same time, which is why neither direction helped.

    Setting the param groups from a callback does NOT work, and this is the same
    trap that made an earlier actor-freeze silently do nothing: SB3 calls
    `_update_learning_rate` at the TOP of every `train()`, after every callback
    hook has already fired, and writes `lr_schedule`'s value into every group.
    So the override is installed by wrapping that method -- SB3 sets both rates,
    then the wrapper puts the actor's back.
    """

    def __init__(self, actor_lr: float, verbose: int = 1):
        super().__init__(verbose)
        self.actor_lr = float(actor_lr)
        self._installed = False

    def _on_training_start(self) -> None:
        if self._installed:
            return
        # Patch the CLASS, not the instance. Assigning a closure to
        # `model._update_learning_rate` puts it in the model's __dict__, and
        # SB3's save() pickles that -- which fails with
        # "cannot pickle 'omni.kit.app._app.IApp'" because the closure reaches
        # the Isaac app through the env. That killed a run at its first
        # checkpoint. The instance carries only a float, which pickles fine.
        cls = type(self.model)
        self.model._actor_lr_override = self.actor_lr
        if not getattr(cls, "_actor_lr_patched", False):
            original = cls._update_learning_rate

            def patched(self_model, optimizers):
                original(self_model, optimizers)
                lr = getattr(self_model, "_actor_lr_override", None)
                if lr is not None:
                    for g in self_model.actor.optimizer.param_groups:
                        g["lr"] = lr

            cls._update_learning_rate = patched
            cls._actor_lr_patched = True
        self._installed = True
        if self.verbose:
            crit = self.model.lr_schedule(1.0)
            print(f"[two-timescale] critic lr {crit:.2e}, actor lr "
                  f"{self.actor_lr:.2e} "
                  f"(actor is {crit / max(self.actor_lr, 1e-12):.0f}x slower)",
                  flush=True)

    def _on_step(self) -> bool:
        return True
