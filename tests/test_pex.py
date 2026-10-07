"""Does policy expansion actually retain the frozen policy and select by Q?

Written before any PEX run, because this project has now had seven interventions that
were accepted by a flag and absent from the mechanism -- including --layer-norm, which was
inert three different ways on the same day. The failure mode to guard here is specific: if
selection never picks the frozen policy, PEX degenerates to the Direct method it is meant
to replace, and the run would read as "PEX did not help".
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
import gymnasium as gym

from lunarsim.rl.cql_sac import CQLSAC


class _Env(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, **kw):
        return np.zeros(4, np.float32), {}

    def step(self, a):
        return np.zeros(4, np.float32), 0.0, False, False, {}


def _model(**kw):
    th.manual_seed(0)
    np.random.seed(0)
    return CQLSAC("MlpPolicy", _Env(), device="cpu", verbose=0, seed=0,
                  cql_alpha=0.0, learning_starts=10, batch_size=16, **kw)


def test_pex_off_by_default_leaves_predict_untouched():
    m = _model()
    assert m.pex_actor is None and m.pex_temperature is None
    a, _ = m.predict(np.zeros(4, np.float32), deterministic=True)
    assert a.shape == (2,)


def test_the_frozen_policy_never_changes_while_the_learnable_one_does():
    """The guarantee the whole method rests on: the warm start cannot be degraded."""
    m = _model()
    m.install_pex(temperature=1.0)
    before = {k: v.clone() for k, v in m.pex_actor.state_dict().items()}
    theta_before = {k: v.clone() for k, v in m.actor.state_dict().items()}
    # learn() rather than a bare train(): train() alone leaves SB3's _logger unset and
    # raises before touching a single parameter, which would pass as a false negative here
    m.learn(total_timesteps=400, log_interval=100000)

    for k, v in m.pex_actor.state_dict().items():
        assert th.equal(v, before[k]), f"frozen policy moved at {k}"
    moved = any(not th.equal(v, theta_before[k]) for k, v in m.actor.state_dict().items())
    assert moved, "the learnable actor did not move, so the test proves nothing"
    assert all(not p.requires_grad for p in m.pex_actor.parameters())


def test_selection_follows_Q_and_can_pick_either_policy():
    """Selection must track Q, not a fixed branch.

    Degenerating to always-pi_theta turns PEX back into the Direct method; always-pi_beta
    makes learning pointless. Both are silent failures, so the critic is stubbed to prefer
    a known action and the choice is checked against it.
    """
    m = _model()
    m.install_pex(temperature=0.01)        # near-argmax

    # install_pex deep-copies the actor, so immediately afterwards the two propose the
    # SAME action and selection is unobservable -- the guard below caught exactly that on
    # the first run. Perturb the learnable one so there is a real choice to make.
    with th.no_grad():
        for prm in m.actor.mu.parameters():
            prm.add_(th.randn_like(prm) * 0.5)

    obs = np.zeros((16, 4), np.float32)
    with th.no_grad():
        ot, _ = m.policy.obs_to_tensor(obs)
        a_beta = m.pex_actor(ot, deterministic=True)
        a_theta = m.actor(ot, deterministic=True)
    assert not th.allclose(a_beta, a_theta), (
        "the two policies propose identical actions, so selection cannot be observed")

    for prefer, name in ((a_beta, "beta"), (a_theta, "theta")):
        target = prefer.clone()

        def fake_q(o, a, _t=target):
            # high value for whichever action is closest to the preferred one
            d = ((a - _t) ** 2).sum(dim=1, keepdim=True)
            return (-d,) * len(m.critic.q_networks)

        m.critic.forward = fake_q            # type: ignore[assignment]
        got, _ = m.predict(obs, deterministic=True)
        got_t = th.as_tensor(got, dtype=th.float32)
        d_pref = ((got_t - target) ** 2).sum(1).mean()
        other = a_theta if name == "beta" else a_beta
        d_other = ((got_t - other) ** 2).sum(1).mean()
        assert d_pref < d_other, (
            f"critic was stubbed to prefer pi_{name} and selection did not follow it "
            f"(dist {d_pref:.4f} vs {d_other:.4f})")


def test_a_saved_checkpoint_carries_the_frozen_policy(tmp_path):
    """Otherwise an evaluation loads pi_theta alone and measures the wrong policy."""
    m = _model()
    m.install_pex(temperature=0.5)
    ref = {k: v.clone() for k, v in m.pex_actor.state_dict().items()}
    path = tmp_path / "pex.zip"
    m.save(str(path))

    m2 = CQLSAC.load(str(path), device="cpu")
    assert getattr(m2, "pex_actor", None) is not None, (
        "the frozen policy was not saved; a loaded checkpoint is NOT the composite policy")
    for k, v in m2.pex_actor.state_dict().items():
        assert th.allclose(v, ref[k]), f"frozen policy changed across save/load at {k}"
