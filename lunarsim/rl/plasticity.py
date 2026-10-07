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

    The new LayerNorms are placed on the SAME DEVICE as the module they are woven into.
    Without that they are created on CPU while the policy is on cuda and the first forward
    pass dies with

        RuntimeError: Expected all tensors to be on the same device, but got weight is on
        cpu, different from other tensors on cuda:0

    which is how v67 died 28 seconds in. Both callers hit it, so `--layer-norm` had never
    worked on GPU in this project at all -- it was only ever exercised on CPU, where the
    bug is invisible.
    """
    if not isinstance(module, nn.Sequential):
        return module
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = None
    out: list[nn.Module] = []
    for i, layer in enumerate(module):
        out.append(layer)
        is_act = isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.GELU, nn.SiLU))
        prev_linear = i > 0 and isinstance(module[i - 1], nn.Linear)
        if is_act and prev_linear:
            ln = nn.LayerNorm(module[i - 1].out_features)
            if device is not None:
                ln = ln.to(device)
            out.append(ln)
    seq = nn.Sequential(*out)
    return seq.to(device) if device is not None else seq


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


def add_layer_norm_to_critic(model) -> int:
    """LayerNorm in the CRITIC ONLY, which is what a warm start can accept.

    `add_layer_norm_to_sac` rebuilds the actor's `latent_pi` as well and says so: "Call
    this on a FRESH model, before any training: it reinitialises the inserted layers."
    On a warm start that would destroy the behaviour clone, which IS the warm start -- so
    `--layer-norm` being confined to the fresh-model branch was correct rather than an
    oversight, and this is the piece that was missing.

    The critic is a different case. A BC clone is fitted by regression on the actor alone;
    its critic has never seen a Bellman target, so the weights `SAC.load` restores for it
    carry no information and reinitialising them costs nothing. The run then spends
    `--critic-only-steps` fitting the critic before the actor is released, which is
    exactly the phase a fresh critic needs.

    Why bother: RLPD (arXiv:2302.02948) has a section titled "Layer Normalization
    Mitigates Catastrophic Overestimation" -- with LayerNorm before the output the critic
    is bounded by the final layer's weights however far out of distribution the queried
    action is. That is this project's measured failure mode, and
    tests/test_layernorm_bounds_ood_q.py confirms the property holds in this
    implementation: |Q| grows under 3x over a 1000x input range against over 10x
    un-normalised, while staying discriminative in distribution.
    """
    n = 0
    for critic in (model.policy.critic, model.policy.critic_target):
        for i, qnet in enumerate(critic.q_networks):
            new = layer_norm_after_relu(qnet)
            critic.q_networks[i] = new
            setattr(critic, f"qf{i}", new)
            n += 1
    model.policy.critic_target.load_state_dict(model.policy.critic.state_dict())
    # the optimizer was built over the OLD parameter list, so the LayerNorm parameters
    # would never receive a gradient step
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(),
                                              lr=model.lr_schedule(1.0))
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


class PolicyDelayCallback(BaseCallback):  # type: ignore[misc]
    """Update the actor once every `delay` gradient steps (TD3's policy_delay).

    This is the two-timescale condition implemented as a RATE of updates rather
    than a learning rate, and the distinction turned out to be the whole point.
    Lowering the shared learning rate does not slow the actor the way it looks
    like it should, because Adam normalises gradient magnitude: each step moves
    parameters by roughly `lr` regardless of the gradient, so the 10,000 actor
    updates that follow a release (gradient_steps=-1 with 16 envs gives 16 per
    vec step) displace the parameters by about 10,000 * lr. At lr 1e-5 that is
    0.1 in parameter space, which is enormous for weights of order 0.1-1 -- and
    it is why making the actor "30x slower" changed nothing.

    Measured, with everything else already fixed: a policy landing 52% with the
    actor frozen drops to 3% in the first window after release and 0% in the
    second, at every learning rate and critic-ensemble size tried.

    Implemented by wrapping the actor optimizer's `step`, which SB3 cannot
    overwrite (unlike param-group learning rates, which `_update_learning_rate`
    rewrites at the top of every `train()`), and which is not part of what
    `save()` pickles.
    """

    def __init__(self, delay: int, verbose: int = 1):
        super().__init__(verbose)
        self.delay = max(1, int(delay))
        self._installed = False

    def _on_training_start(self) -> None:
        if self._installed or self.delay <= 1:
            return
        opt = self.model.actor.optimizer
        original_step = opt.step
        state = {"n": 0}
        delay = self.delay

        def step(*a, **kw):
            state["n"] += 1
            if state["n"] % delay == 0:
                return original_step(*a, **kw)
            return None

        opt.step = step
        self._installed = True
        if self.verbose:
            print(f"[policy-delay] actor steps once every {delay} critic steps",
                  flush=True)

    def _on_step(self) -> bool:
        return True


class ActorSGDCallback(BaseCallback):  # type: ignore[misc]
    """Swap the actor's optimizer from Adam to SGD.

    Adam normalises by the gradient's running magnitude, so a step is about
    `lr` in parameter space whatever the gradient actually is. For a SATURATED
    tanh policy that is exactly wrong: at |a| -> 1 the Jacobian `1 - a^2`
    vanishes, so the gradient reaching the pre-tanh mean is tiny and dominated
    by noise -- and Adam rescales that noise back up to a full-size step. The
    policy then random-walks at `lr` per update regardless of whether there is
    any signal.

    Measured on this project: the policies here are ~96% saturated, and a 47-52%
    policy is destroyed within one 10k-step window of releasing the actor --
    which is 10,000 updates. It happens at every learning rate tried, which is
    the signature of step size being set by the optimizer rather than by the
    gradient. With SGD a vanishing gradient means a vanishing step.

    The optimizer object itself is not pickled by SB3 (only its state_dict), so
    replacing it is safe across checkpoints.
    """

    def __init__(self, lr: float, momentum: float = 0.0, verbose: int = 1):
        super().__init__(verbose)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._installed = False

    def _on_training_start(self) -> None:
        if self._installed:
            return
        self.model.actor.optimizer = torch.optim.SGD(
            self.model.actor.parameters(), lr=self.lr, momentum=self.momentum)
        self._installed = True
        if self.verbose:
            print(f"[actor-sgd] actor optimizer -> SGD(lr={self.lr:.1e}, "
                  f"momentum={self.momentum})", flush=True)

    def _on_step(self) -> bool:
        return True


class BCAnchorCallback(BaseCallback):  # type: ignore[misc]
    """Keep the actor in a trust region around the policy it started from.

    This is the proximal form of the BC-regularised actor loss that
    offline-to-online methods use (TD3+BC's `-Q + alpha*(pi(s) - a_data)^2`,
    AWAC's advantage-weighted constraint): instead of adding a term to the loss,
    pull the parameters back toward the anchor after each update,
    `theta <- theta + beta*(theta_0 - theta)`. Both implement "improve, but do
    not leave the region the demonstrations cover"; the proximal version needs
    no surgery on SB3's train().

    Why this project needs it, measured. The policies here are ~96% saturated,
    so competence depends on the pre-tanh mean being large: moving it from ~5 to
    ~1 takes actions from 1.0 to 0.76, an enormous behavioural change for a
    small parameter change. A 47-52% policy therefore dies in ONE 10k-step
    window after the actor is released -- and it does so identically under Adam
    and SGD, at every learning rate, with a bounded critic and with a diverging
    one. Nineteen interventions failed to stop it because none of them held the
    policy anywhere; they only changed how fast it left.

    `beta` is per actor update. With 10k updates per window, beta=1e-3 gives a
    pull half-life of ~700 updates, so the policy can move but cannot run.

    The anchor covers the MEAN network only. It originally walked the whole
    `state_dict()`, which silently included the `log_std` head -- and that pinned
    the exploration scale to the BC fit's residual for the entire run. Measured
    across every checkpoint of two 410k-step runs, `log_std.bias` moved from
    -2.647 to -2.645 and the head weights stayed at |W|~5e-4, i.e. std was frozen
    at [0.076, 0.135, 0.124, 0.020] -- identical to the clone's, to three
    decimals, after 410k steps. SAC's entropy term was left pushing against a
    parameter that could not move, and `ent_coef` auto-tuning collapsed from 1.0
    to 0.005 in response. A trust region is supposed to constrain WHICH ACTIONS
    the policy takes, not how much it is allowed to explore around them; holding
    log_std as well turns the run into near-deterministic policy iteration inside
    a small ball around the clone, which is not the algorithm being tested.
    """

    # Substring match rather than exact keys: SB3 names this head `log_std` when
    # it is a bare parameter and `log_std.weight` / `log_std.bias` when it is
    # state-dependent, and this project has checkpoints of both shapes.
    _FREE_KEY_MARKERS = ("log_std",)

    def __init__(self, beta: float, anchor_log_std: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.beta = float(beta)
        self.anchor_log_std = bool(anchor_log_std)
        self._anchor = None
        self._installed = False

    def _is_free(self, key: str) -> bool:
        if self.anchor_log_std:
            return False
        return any(m in key for m in self._FREE_KEY_MARKERS)

    def _on_training_start(self) -> None:
        if self._installed:
            return
        self._anchor = {k: v.detach().clone()
                        for k, v in self.model.actor.state_dict().items()
                        if not self._is_free(k)}
        free = [k for k in self.model.actor.state_dict() if self._is_free(k)]
        opt = self.model.actor.optimizer
        original_step = opt.step
        anchor = self._anchor
        actor = self.model.actor
        beta = self.beta

        def step(*a, **kw):
            out = original_step(*a, **kw)
            with torch.no_grad():
                for k, v in actor.state_dict().items():
                    ref = anchor.get(k)
                    if ref is not None and v.dtype.is_floating_point:
                        v.mul_(1.0 - beta).add_(ref, alpha=beta)
            return out

        opt.step = step
        self._installed = True
        if self.verbose:
            half = (0.693 / beta) if beta > 0 else float("inf")
            print(f"[bc-anchor] pulling actor toward its start, beta={beta:.1e} "
                  f"(half-life ~{half:.0f} updates); "
                  f"{len(anchor)} tensors held, free={free or 'none'}", flush=True)

    def _on_step(self) -> bool:
        return True
