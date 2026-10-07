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
    from stable_baselines3 import SAC
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
