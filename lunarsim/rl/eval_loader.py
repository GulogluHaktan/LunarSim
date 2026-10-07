"""Load a checkpoint for EVALUATION without caring what was done to its critic.

Why this exists. v68's six evaluation cells all came back empty: its critic carries
LayerNorm layers woven in after construction, so the saved state_dict does not match the
critic a default `SAC.load` rebuilds --

    Missing key(s):    critic.qf0.4.weight ...
    Unexpected key(s): critic.qf0.5.weight, critic.qf0.3.weight ...
    size mismatch for critic.qf0.2.weight: [512] from checkpoint vs [512, 512]

and the whole load raises. The run was fine; it simply could not be measured, and the
`2>/dev/null` in the evaluation script turned that into six blank lines.

Evaluation only ever needs the ACTOR -- `predict` does not touch the critic. So this
rebuilds the model from the checkpoint's own saved `policy_kwargs` (which is how net_arch
comes back) and loads the actor's parameters alone, leaving the critic freshly initialised
and unused. That makes an evaluation immune to any critic surgery: LayerNorm, a different
ensemble size, dual-gamma heads, anything added later.

It deliberately does NOT reconstruct the critic. If a future caller needs Q at evaluation
time it should say so explicitly rather than inherit a silently half-loaded one.
"""
from __future__ import annotations

import warnings
import zipfile

import torch as th


def load_actor_for_eval(path: str, device: str = "cpu"):
    """Return a SAC model whose ACTOR is the checkpoint's. The critic is NOT loaded.

    Falls back to a normal full load when that works, so an ordinary checkpoint behaves
    exactly as before and this cannot change any existing number.
    """
    from stable_baselines3 import PPO, SAC
    # PPO and SAC checkpoints are not interchangeable (ActorCriticPolicy against a separate
    # actor and critic), and loading the wrong class raises on the state dict rather than
    # producing a wrong policy -- but the message is unhelpful, so detect it from the saved
    # data instead of guessing from the filename.
    from stable_baselines3.common.save_util import load_from_zip_file
    try:
        _data, _params, _ = load_from_zip_file(path, device=device)
        is_ppo = any(k.startswith("action_net") or k.startswith("mlp_extractor")
                     for k in _params.get("policy", {}))
    except Exception:
        is_ppo = False
    if is_ppo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PPO.load(path, device=device)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SAC.load(path, device=device)
    except (RuntimeError, ValueError, KeyError) as exc:
        if "state_dict" not in str(exc) and "size mismatch" not in str(exc):
            raise

    # The critic could not be reconstructed. Build just the POLICY from the checkpoint's
    # own saved data and transplant the actor, rather than going through the algorithm's
    # _setup_model -- that path wants n_envs, a replay buffer and a train_freq it has no
    # reason to need for a forward pass, and fails without an env.
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.sac.policies import SACPolicy
    data, params, _ = load_from_zip_file(path, device=device)
    policy_kwargs = dict(data.get("policy_kwargs") or {})
    policy = SACPolicy(data["observation_space"], data["action_space"],
                       data["lr_schedule"], **policy_kwargs).to(device)
    actor_sd = {k[len("actor."):]: v for k, v in params["policy"].items()
                if k.startswith("actor.")}
    if not actor_sd:
        raise RuntimeError(f"{path}: no actor parameters found under 'policy'")
    missing, unexpected = policy.actor.load_state_dict(actor_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: the ACTOR does not match either -- missing {list(missing)[:4]}, "
            f"unexpected {list(unexpected)[:4]}. The policy architecture itself changed, "
            f"so fix the loader rather than evaluating a partially loaded actor.")
    policy.set_training_mode(False)
    print(f"[eval-loader] {path}: critic could not be rebuilt (LayerNorm or a changed "
          f"ensemble); loaded the ACTOR only, which is all predict() uses", flush=True)

    class _ActorOnly:
        """Just enough surface for an evaluation: `predict`."""
        def __init__(self, pol):
            self.policy = pol
            self.observation_space = pol.observation_space
            self.action_space = pol.action_space

        def predict(self, observation, state=None, episode_start=None,
                    deterministic=False):
            return self.policy.predict(observation, state, episode_start, deterministic)

    return _ActorOnly(policy)


def load_with_critic_for_diag(path: str, device: str = "cpu"):
    """Like `load_actor_for_eval`, but the CRITIC is loaded too -- strictly.

    The module docstring says a caller needing Q at evaluation time should say so
    explicitly rather than inherit a silently half-loaded critic. This is that caller:
    measuring whether a split critic's VALUE is calibrated needs the critic's own
    weights, and a freshly initialised one would answer a different question while
    looking identical.

    Returns `(model_like, dual_gamma)` where `model_like` carries `.critic` (a list of
    Q networks' outputs, SB3's `ContinuousCritic` call signature) and `.predict`, and
    `dual_gamma` is the `(gamma_shaping, gamma_terminal)` the checkpoint was trained with
    or None. Raises rather than loading the critic partially: a diagnostic that silently
    measures half a critic is worse than no diagnostic.
    """
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.sac.policies import SACPolicy

    data, params, _ = load_from_zip_file(path, device=device)
    policy_kwargs = dict(data.get("policy_kwargs") or {})
    policy = SACPolicy(data["observation_space"], data["action_space"],
                       data["lr_schedule"], **policy_kwargs).to(device)
    sd = params["policy"]

    def _sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    critic_sd = _sub("critic.")
    if not critic_sd:
        raise RuntimeError(f"{path}: no critic parameters under 'policy'")

    try:
        policy.critic.load_state_dict(critic_sd, strict=True)
    except RuntimeError:
        # The checkpoint's critic carries LayerNorm woven in after construction (see
        # lunarsim/rl/plasticity.add_layer_norm_to_critic). Rebuild it the SAME way the
        # training run did, then load strictly -- the retry is what makes "strict" mean
        # something here instead of just failing on a known, reproducible difference.
        from lunarsim.rl.plasticity import add_layer_norm_to_critic

        # The surgery also rebuilds the critic's optimizer (the LayerNorm parameters
        # would otherwise never get a gradient step), so the shim has to carry
        # `.critic` and `.lr_schedule` as well -- a diagnostic load never steps that
        # optimizer, but constructing it is part of matching the training-time critic.
        class _Shim:
            pass
        shim = _Shim()
        shim.policy = policy
        shim.critic = policy.critic
        shim.lr_schedule = data["lr_schedule"]
        add_layer_norm_to_critic(shim)
        policy.critic.load_state_dict(critic_sd, strict=True)
        print(f"[eval-loader] {path}: critic rebuilt WITH LayerNorm to match the "
              f"checkpoint, loaded strictly", flush=True)

    actor_sd = _sub("actor.")
    policy.actor.load_state_dict(actor_sd, strict=True)
    policy.set_training_mode(False)

    dual = data.get("dual_gamma")
    dual = None if dual is None else (float(dual[0]), float(dual[1]))

    class _WithCritic:
        def __init__(self, pol):
            self.policy = pol
            self.critic = pol.critic
            self.observation_space = pol.observation_space
            self.action_space = pol.action_space
            self.device = device

        def predict(self, observation, state=None, episode_start=None,
                    deterministic=False):
            return self.policy.predict(observation, state, episode_start, deterministic)

    return _WithCritic(policy), dual
