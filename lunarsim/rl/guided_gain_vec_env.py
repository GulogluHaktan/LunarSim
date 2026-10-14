"""`GuidedGainEnv`'s construction, on the real Isaac backend and vectorized.

Read `lunarsim/rl/guided_gain_env.py` first: it carries the measurements and
the citations that chose this action space. This module is the same idea
against `IsaacLanderVecEnv`, which is the env `scripts/train_sac_isaac.py`
actually trains on and the one every number in `handover.md` was measured
through.

THREE THINGS HERE ARE EASY TO GET WRONG, AND ALL THREE HAVE BITTEN THIS
PROJECT BEFORE.

1. ALTITUDE. `IsaacLanderVecEnv.state["z"]` is the rigid body's CENTRE, placed
   at `ground_z + alt + half_height_m` with `half_height_m = 3.52`. The
   controller computes `alt = state["z"] - _ground_z(x, y)`, which is right for
   the zero-height point model in `AnalyticLanderEnv` and wrong here by
   3.52 m at zero tilt and 4.62 m at 15 deg. `collect_zemzev_demos.py`
   documents finding exactly this bug in the demo collector, and
   `diag_controller_stage_rate.py` still had it as of this session -- which is
   where the "controller 12.0% on land_handoff" bar came from. `_EnvSlice`
   below feeds the controller `ground_z + clearance_m` from the env's own
   `_contact(i)`, so "altitude 0" means the same thing to the controller, to
   `info["altitude_m"]`, to observation slot 2 and to the termination test.

2. PAD-RELATIVE COORDINATES. The envs are laid out side by side at
   `env_spacing_m` along +x, so world x for env i carries an
   `env_origins[i][0]` offset. The controller reads `s["x"]`/`s["y"]` as pad
   offsets, so env 7 would otherwise be handed a 1400 m offset from a pad it
   is sitting on. `IsaacLanderVecEnv` already subtracts the origin before
   calling `reward_fn`; this does the same before calling the controller.

3. MID-HOLD TERMINATION. One decision spans `hold_steps` control steps, and
   PhysX advances every vehicle together, so an env that touches down at
   sub-step j cannot be frozen for the remaining k-j. `handover.md` flags the
   consequence for an action-repeat `VecEnvWrapper`: the wrapper keeps stepping
   the terminated env, the underlying env auto-resets it, and the first k-j
   steps of the NEXT episode are silently consumed -- and those are the release
   states, the part of the distribution that decides the outcome.

   This env does not accept that trade. When env i reports done at sub-step j
   it is marked finished for the decision: its accumulated reward and its
   terminal info are frozen there, and everything it flies for the rest of the
   hold is DISCARDED. At the end of the hold each finished env is reset once
   more via `_reset_one(i)`, so the observation handed back is a true episode
   start that the policy's next decision acts on. The cost is up to
   `hold_steps - 1` wasted physics steps per episode end -- about 2.7% at
   k=20 on a 750-step episode -- and in exchange the release distribution is
   exactly the stage's own.
"""
from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Optional, Sequence

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains
from lunarsim.rl.guided_gain_env import _EXTRA_OBS, GAIN_SPEC, GainKnob


class _EnvSlice:
    """What `ZemZevController.act` needs, for one env of a vectorized backend.

    `act()` touches `.params`, `.state` and `._ground_z(x, y)` and nothing
    else, so this is the whole interface. `sync()` must be called before every
    `act()` because the controller reads live state.
    """

    __slots__ = ("params", "state", "_venv", "_i", "_ground_z_m")

    def __init__(self, venv, i: int):
        self._venv = venv
        self._i = i
        self.params = venv.params
        self.state: dict = {}
        self._ground_z_m = 0.0

    def sync(self) -> None:
        venv, i = self._venv, self._i
        clearance_m, ground_z = venv._contact(i)
        s = venv.state
        self.state = {k: float(v[i]) for k, v in s.items() if hasattr(v, "__len__")}
        ox, oy, _ = venv.env_origins[i]
        # Pad-relative, matching what the reward and observation slots 0-1 use.
        self.state["x"] = float(s["x"][i]) - float(ox)
        self.state["y"] = float(s["y"][i]) - float(oy)
        # Belly clearance, not body centre -- see note 1 in the module docstring.
        self.state["z"] = ground_z + clearance_m
        self._ground_z_m = ground_z

    def _ground_z(self, x: float, y: float) -> float:
        return self._ground_z_m


class GuidedGainVecEnv(VecEnv):
    """Policy sets ZEM/ZEV gains; `hold_steps` controller steps per decision."""

    def __init__(
        self,
        venv,
        gain_spec: Sequence[GainKnob] = GAIN_SPEC,
        base_gains: Optional[ZemZevGains] = None,
        hold_steps: int = 20,
        residual_scale: float = 0.0,
        hazard_obs: Optional[bool] = None,
    ):
        self.venv = venv
        import os
        self.hazard_obs = (os.environ.get("LUNARSIM_HAZ_OBS", "0") == "1") if hazard_obs is None else bool(hazard_obs)
        self.aim_scale = float(os.environ.get("LUNARSIM_AIM_SCALE", "0"))   # m/s per unit of the 2 aim action dims
        self.gain_spec = tuple(gain_spec)
        self.base_gains = base_gains or ZemZevGains()
        self.hold_steps = int(hold_steps)
        self.residual_scale = float(residual_scale)
        self.params = venv.params

        valid = {f.name for f in fields(ZemZevGains)}
        bad = [k.name for k in self.gain_spec if k.name not in valid]
        if bad:
            raise ValueError(f"not ZemZevGains fields: {bad}")
        self._nominals = np.array(
            [float(getattr(self.base_gains, k.name)) for k in self.gain_spec],
            dtype=np.float64)

        n = int(venv.num_envs)
        self.controllers = [ZemZevController(venv.params, self.base_gains)
                            for _ in range(n)]
        self.views = [_EnvSlice(venv, i) for i in range(n)]

        from gymnasium import spaces
        n_gain = len(self.gain_spec)
        n_act = n_gain + (4 if self.residual_scale > 0.0 else 0) + (2 if self.aim_scale > 0.0 else 0)
        obs_dim = int(venv.observation_space.shape[0]) + _EXTRA_OBS
        if self.hazard_obs:
            from lunarsim.evo.hazfeat import N_HAZ
            obs_dim += N_HAZ + (2 if self.aim_scale > 0.0 else 0)
        super().__init__(
            n,
            spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32),
        )
        self._actions: Optional[np.ndarray] = None
        self._in_final = np.zeros(n, dtype=bool)

    # -- observation ------------------------------------------------------

    def _augment(self, base_obs: np.ndarray) -> np.ndarray:
        s = self.venv.state
        n = self.num_envs
        extra = np.zeros((n, _EXTRA_OBS), dtype=np.float32)
        lim = max(self.params.safe_landing_v_xy_m_s, 1e-6)
        for i in range(n):
            tgo = self.controllers[i].tgo
            extra[i, 0] = 0.0 if tgo is None else float(tgo) / 30.0
            extra[i, 1] = 1.0 if self._in_final[i] else 0.0
            extra[i, 2] = math.hypot(float(s["vx"][i]), float(s["vy"][i])) / lim
        parts = [np.asarray(base_obs, dtype=np.float32), extra]
        if self.hazard_obs:
            from lunarsim.core.terrain.hazard import attach_hazard
            from lunarsim.evo.hazfeat import N_HAZ, haz_features
            hz = np.zeros((n, N_HAZ + (2 if self.aim_scale > 0.0 else 0)), dtype=np.float32)
            for i in range(n):
                tile = attach_hazard(self.venv._tiles[i])
                ox, oy = float(self.venv.env_origins[i][0]), float(self.venv.env_origins[i][1])
                hz[i, :N_HAZ] = haz_features(float(s["x"][i]) - ox, float(s["y"][i]) - oy,
                                     float(s["vx"][i]), float(s["vy"][i]), tile.haz_dist, tile.res_m)
                if self.aim_scale > 0.0:
                    from lunarsim.evo.hazfeat import safe_dir
                    hz[i, N_HAZ:] = safe_dir(float(s["x"][i]) - ox, float(s["y"][i]) - oy, tile.haz_dist, tile.res_m)
            parts.append(hz)
        return np.concatenate(parts, axis=1)

    # -- gains ------------------------------------------------------------

    def _gains_from_action(self, a: np.ndarray) -> ZemZevGains:
        upd = {k.name: k.value(float(self._nominals[j]), float(a[j]))
               for j, k in enumerate(self.gain_spec)}
        return replace(self.base_gains, **upd)

    def _commands(self, active: np.ndarray, resid) -> np.ndarray:
        n = self.num_envs
        cmd = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            self.views[i].sync()
            c = self.controllers[i].act(self.views[i])
            if resid is not None:
                c = np.clip(np.asarray(c, dtype=np.float64) + resid[i], -1.0, 1.0)
            cmd[i] = np.asarray(c, dtype=np.float32)
            self._in_final[i] = self.controllers[i].tgo is None
        return cmd

    # -- VecEnv API -------------------------------------------------------

    def reset(self):
        obs = self.venv.reset()
        for c in self.controllers:
            c.reset()
            c.g = self.base_gains
        self._in_final[:] = False
        return self._augment(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float64)

    def step_wait(self):
        n = self.num_envs
        a = self._actions
        n_gain = len(self.gain_spec)
        for i in range(n):
            self.controllers[i].g = self._gains_from_action(a[i, :n_gain])
        resid = (a[:, n_gain:n_gain + 4] * self.residual_scale) \
            if self.residual_scale > 0.0 else None
        if self.aim_scale > 0.0:
            from lunarsim.core.terrain.hazard import attach_hazard
            from lunarsim.evo.hazfeat import aim_point
            sv = self.venv.state
            for i in range(n):
                ai = np.clip(a[i, -2:], -1.0, 1.0) * self.aim_scale
                tl = attach_hazard(self.venv._tiles[i]); ox, oy = float(self.venv.env_origins[i][0]), float(self.venv.env_origins[i][1])
                self.controllers[i].aim_pos = aim_point(float(sv["x"][i]) - ox, float(sv["y"][i]) - oy, tl.haz_dist, tl.res_m,
                                                        (float(ai[0]), float(ai[1])))
                self.controllers[i].pos_fade_alt_m = 30.0
                self.controllers[i].final_pos_kp = 0.05

        rew = np.zeros(n, dtype=np.float64)
        finished = np.zeros(n, dtype=bool)
        out_infos: list[dict] = [{} for _ in range(n)]

        import os
        lam = float(os.environ.get("LUNARSIM_HAZ_LAMBDA_PPO", "0"))
        for _ in range(self.hold_steps):
            cmd = self._commands(~finished, resid)
            if lam > 0.0:   # touchdown shaping: snapshot the position / tile BEFORE the step (it may auto-reset)
                sx_, sy_ = self.venv.state["x"].copy(), self.venv.state["y"].copy()
                tiles_prev = list(self.venv._tiles)
            obs, r, d, infos = self.venv.step(cmd)
            for i in range(n):
                if finished[i]:
                    continue          # already decided; this sub-step is discarded
                rew[i] += float(r[i])
                if d[i]:
                    if lam > 0.0:
                        from lunarsim.core.terrain.hazard import attach_hazard
                        from lunarsim.evo.hazfeat import _bilerp
                        tl = attach_hazard(tiles_prev[i])
                        ox, oy = float(self.venv.env_origins[i][0]), float(self.venv.env_origins[i][1])
                        dd = float(_bilerp(tl.haz_dist, tl.res_m, np.array([sx_[i] - ox]), np.array([sy_[i] - oy]))[0])
                        rew[i] -= lam * min(dd, 150.0) / 150.0
                    finished[i] = True
                    out_infos[i] = dict(infos[i])
                    self.controllers[i].reset()
            if finished.all():
                break

        # A fresh start for every env that finished inside the hold, so the
        # observation the policy acts on is a real release state rather than
        # whatever the discarded sub-steps left behind (note 3 above).
        if finished.any():
            for i in np.flatnonzero(finished):
                self.venv._reset_one(int(i))
                self.controllers[int(i)].reset()
                self.controllers[int(i)].g = self.base_gains
                self._in_final[int(i)] = False
            obs = np.stack([self.venv._obs_one(i) for i in range(n)])

        D = int(self.observation_space.shape[0])
        for inf in out_infos:    # bootstrap target for truncated episodes must have the full observation width
            t = inf.get("terminal_observation")
            if t is not None and np.asarray(t).shape[-1] != D:
                t = np.asarray(t, dtype=np.float32).reshape(-1)
                inf["terminal_observation"] = np.concatenate([t, np.zeros(D - t.shape[0], dtype=np.float32)])
        return self._augment(obs), rew, finished.copy(), out_infos

    def close(self):
        self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs \
            if hasattr(self, attr_name) else self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return self.venv.env_method(method_name, *args, indices=indices, **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return self.venv.seed(seed) if hasattr(self.venv, "seed") else None
