"""Sampling-based Model Predictive Control (MPPI) for the lander, batched over envs on GPU.

At every replan the controller (1) copies each env's measured state, (2) rolls K
candidate control sequences forward through a copy of the env's own point-mass +
attitude dynamics (`AnalyticLanderEnv.step`, flat ground under the vehicle),
(3) scores each rollout with a cost, (4) updates the mean control sequence by
softmin-weighting the samples (MPPI), (5) applies the first segment and shifts the
sequence for warm start. It reads ONLY the measured state -- no ZEM/ZEV, no policy.

The cost uses the project's guidance field (`reward.target_velocity` constants) as
the descent/lateral speed reference, plus a touchdown-severity term built from the
same four limits `landed_safely` tests. That reference is disclosed prior knowledge,
the same one the ES policy sees in observation slots 16-19.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from lunarsim.core.vehicle.apollo_lm import G0, moment_of_inertia
from lunarsim.rl.action_map import THROTTLE_EXPO

# log10 of the cost weights ES tunes, in this order.
PARAM_NAMES = ("w_lat", "w_vz", "w_tilt", "w_rate", "w_u", "w_td", "w_T", "k_ref", "k_lat", "w_time", "w_haz")
DEFAULT_LOG = np.array([1.5, 0.0, -0.5, 0.0, -1.0, 2.0, 1.5, 0.1, -0.3, 1.0, 0.0])
HAZ_SCALE_M = float(os.environ.get('LUNARSIM_HAZ_SCALE', '30.0'))      # hazard cost = w_haz * min(dist_to_safe / HAZ_SCALE_M, HAZ_CAP)
HAZ_CAP = 5.0


@dataclass
class MPCConfig:
    K: int = 512            # samples per env
    M: int = 20             # control segments in the horizon
    hold: int = 12          # sim steps per segment (dt_s each): horizon = M*hold*dt_s = 12 s
    exec_steps: int = 4     # sim steps applied between replans
    roll_sub: int = 1       # sim steps merged into one rollout (Euler) step; must divide hold
    margin: float = 1.0     # planner plans to margin * (touchdown limits): robustness to model error
    iters: int = 2
    sigma: float = 0.35     # sample noise in action units
    lam: float = 0.1        # softmin temperature on min-max normalised cost
    hover_a0: float = -0.02
    seed: int = 0


class MPPIMPC:
    def __init__(self, lander_params, reward_weights, cfg: MPCConfig | None = None,
                 log_params=None, device: str = "cuda"):
        self.p, self.c = lander_params, cfg or MPCConfig()
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rw = reward_weights
        lp = np.asarray(DEFAULT_LOG if log_params is None else log_params, dtype=np.float64)
        if lp.shape[0] == len(PARAM_NAMES) - 1:             # pre-hazard parameter vector: pad w_haz
            lp = np.concatenate([lp, DEFAULT_LOG[-1:]])
        assert lp.shape == (len(PARAM_NAMES),), lp.shape
        self._haz = None; self._haz_half = 0.0
        self.W = dict(zip(PARAM_NAMES, (10.0 ** lp).tolist()))
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(self.c.seed)
        self.U = None  # (n, M, 4) mean sequence
        self._since_shift = 0
        p = self.p
        self.max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple
        assert self.c.hold % self.c.roll_sub == 0
        self.dt_r = p.dt_s * self.c.roll_sub
        self.damp = max(0.0, 1.0 - p.angular_damping_per_s * self.dt_r)

    # ------------------------------------------------------------------ public
    def reset(self, n: int, mask: np.ndarray | None = None):
        if self.U is None or self.U.shape[0] != n:
            self.U = torch.zeros(n, self.c.M, 4, device=self.dev)
            self.U[..., 0] = self.c.hover_a0
        if mask is not None:
            m = torch.as_tensor(mask, device=self.dev)
            self.U[m] = 0.0
            self.U[m, :, 0] = self.c.hover_a0

    def set_hazard(self, dist_maps, size_m: float, res_m: float, height_maps=None):
        """Per-env distance-to-safe-cell maps (n, N, N) [m]; index order i <-> x, j <-> y.
        `height_maps` (same layout, metres) lets the rollout track the ground height under the predicted
        position, so the touchdown time is right on sloping terrain (flat-ground assumption otherwise)."""
        m = torch.as_tensor(np.asarray(dist_maps, dtype=np.float32), device=self.dev)
        self._haz = m[:, None]                              # (n, 1, N, N)
        self._haz_half = (m.shape[-1] - 1) / 2.0 * res_m
        self._hgt = None
        if height_maps is not None and os.environ.get("LUNARSIM_MPC_TERRAIN", "1") != "0":
            self._hgt = torch.as_tensor(np.asarray(height_maps, dtype=np.float32), device=self.dev)[:, None]

    def _hgt_at(self, px, py):
        g = torch.stack([py / self._haz_half, px / self._haz_half], dim=-1).clamp(-1.0, 1.0)
        return torch.nn.functional.grid_sample(self._hgt, g[:, :, None, :], mode="bilinear",
                                               padding_mode="border", align_corners=True)[:, 0, :, 0]

    def _drift(self, alt):
        """Ground distance still to be covered while descending along the guidance field from `alt`."""
        if getattr(self, "_drift_tab", None) is None or self._drift_key != (self.W["k_ref"], self.W["k_lat"]):
            w = self.rw
            a = np.linspace(0.0, 500.0, 1001)
            lat = np.maximum(w.vfield_vxy_c * a ** w.vfield_vxy_p, w.vfield_vxy_floor_m_s) * self.W["k_lat"]
            kk = self.W["k_ref"]
            if os.environ.get("LUNARSIM_MPC_KBLEND", "1") != "0":
                kk = 1.0 + (kk - 1.0) * np.clip((a - 10.0) / 40.0, 0.0, 1.0)
            vzm = np.maximum(w.vfield_vz_c * a ** w.vfield_vz_p, w.vfield_vz_floor_m_s) * kk
            r = lat / np.maximum(vzm, 0.05)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(a))])
            self._drift_tab = torch.as_tensor(cum, dtype=torch.float32, device=self.dev)
            self._drift_key = (self.W["k_ref"], self.W["k_lat"])
        x = (torch.nan_to_num(alt, nan=0.0, posinf=499.0, neginf=0.0).clamp(0.0, 499.99) * 2.0)
        i0 = x.floor().long(); f = x - i0.float()
        return self._drift_tab[i0] * (1 - f) + self._drift_tab[(i0 + 1).clamp(max=1000)] * f

    def _haz_cost(self, px, py):
        """px, py: (n, K) tile-frame metres -> (n, K) smooth hazard cost in [0, HAZ_CAP]."""
        # grid_sample: x_grid indexes W (= j <-> y), y_grid indexes H (= i <-> x)
        g = torch.stack([py / self._haz_half, px / self._haz_half], dim=-1).clamp(-1.0, 1.0)  # (n, K, 2)
        d = torch.nn.functional.grid_sample(self._haz, g[:, :, None, :], mode="bilinear",
                                            padding_mode="border", align_corners=True)       # (n, 1, K, 1)
        return (d[:, 0, :, 0] / HAZ_SCALE_M).clamp(0.0, HAZ_CAP)

    def act(self, s: dict) -> np.ndarray:
        """`s`: dict of (n,) arrays: alt, vx, vy, vz, tilt_x, tilt_y, yaw, wx, wy, wz, fuel_kg."""
        n = len(s["alt"])
        if self.U is None or self.U.shape[0] != n:
            self.reset(n)
        st = {k: torch.as_tensor(np.asarray(s[k], dtype=np.float32), device=self.dev) for k in
              ("alt", "vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz", "fuel_kg")}
        t_left = torch.as_tensor(np.asarray(s.get("t_left", np.full(n, 1e9)), dtype=np.float32), device=self.dev)
        if self._haz is not None and "px" in s:
            st["px"] = torch.as_tensor(np.asarray(s["px"], dtype=np.float32), device=self.dev)
            st["py"] = torch.as_tensor(np.asarray(s["py"], dtype=np.float32), device=self.dev)
        K, M = self.c.K, self.c.M
        U = self.U
        for _ in range(self.c.iters):
            eps = torch.randn(n, K, M, 4, device=self.dev, generator=self.gen) * self.c.sigma
            eps[:, 0] = 0.0                                  # keep the current mean as a sample
            cand = (U[:, None] + eps).clamp(-1.0, 1.0)
            cost = self._rollout(st, cand, t_left)           # (n, K)
            cn = (cost - cost.min(1, keepdim=True).values) / (
                cost.max(1, keepdim=True).values - cost.min(1, keepdim=True).values + 1e-6)
            w = torch.softmax(-cn / self.c.lam, dim=1)       # (n, K)
            U = (w[:, :, None, None] * cand).sum(1)
        a = U[:, 0].clone()
        self._since_shift += self.c.exec_steps
        if self._since_shift >= self.c.hold:                 # the executed segment has elapsed
            self._since_shift -= self.c.hold
            U = torch.cat([U[:, 1:], U[:, -1:]], dim=1)      # shift for warm start
        self.U = U
        return a.cpu().numpy()

    # ------------------------------------------------------------------ model
    def _target(self, alt):
        w = self.rw; k = self.W["k_ref"]; kl = self.W["k_lat"]
        a = alt.clamp(min=0.0)
        lat_t = torch.clamp(w.vfield_vxy_c * a ** w.vfield_vxy_p, min=w.vfield_vxy_floor_m_s) * kl
        if os.environ.get("LUNARSIM_MPC_KBLEND", "1") != "0":
            # the descent-speed gain applies up high only: near the ground the reference is the stock (soft) one,
            # otherwise raising k_ref also raises the touchdown speed target
            k = 1.0 + (k - 1.0) * ((a - 10.0) / 40.0).clamp(0.0, 1.0)
        vz_t = -torch.clamp(w.vfield_vz_c * a ** w.vfield_vz_p, min=w.vfield_vz_floor_m_s) * k
        return lat_t, vz_t

    def _rollout(self, st, U, t_left):
        p, W, dev = self.p, self.W, self.dev
        n, K = U.shape[0], U.shape[1]
        ex = lambda t: t[:, None].expand(n, K).clone()      # noqa: E731
        alt, vx, vy, vz = ex(st["alt"]), ex(st["vx"]), ex(st["vy"]), ex(st["vz"])
        tx, ty, yaw = ex(st["tilt_x"]), ex(st["tilt_y"]), ex(st["yaw"])
        wx, wy, wz, fuel = ex(st["wx"]), ex(st["wy"]), ex(st["wz"]), ex(st["fuel_kg"])
        use_haz = self._haz is not None and "px" in st
        if use_haz:
            px, py = ex(st["px"]), ex(st["py"])
        use_hgt = use_haz and getattr(self, "_hgt", None) is not None
        if use_hgt:
            h0 = self._hgt_at(px, py); alt0 = alt.clone(); dz = torch.zeros_like(alt)
        alive = torch.ones(n, K, dtype=torch.bool, device=dev)
        cost = torch.zeros(n, K, device=dev)
        dt, g = self.dt_r, p.gravity_m_s2
        mg = self.c.margin
        lim_vz, lim_vxy, lim_tilt, lim_w = (mg * p.safe_landing_v_z_m_s, mg * p.safe_landing_v_xy_m_s,
                                            mg * p.safe_landing_tilt_rad, mg * p.safe_landing_w_rad_s)
        step_i = 0
        for m in range(U.shape[2]):
            a0, a1, a2, a3 = U[:, :, m, 0], U[:, :, m, 1], U[:, :, m, 2], U[:, :, m, 3]
            e = THROTTLE_EXPO
            thr = (0.5 + 0.5 * (e * a0 ** 3 + (1 - e) * a0)).clamp(0.0, 1.0)
            thrust = torch.where(thr <= 0, torch.zeros_like(thr),
                                 p.dps_thrust_min_n + thr * (p.dps_thrust_max_n - p.dps_thrust_min_n))
            thrust = torch.where(fuel <= 0, torch.zeros_like(thrust), thrust)
            for _ in range(self.c.hold // self.c.roll_sub):
                mass = p.dry_mass_kg + fuel
                i_tilt = mass * (3.0 * p.body_radius_m ** 2 + p.body_height_m ** 2) / 12.0
                i_yaw = mass * p.body_radius_m ** 2 / 2.0
                wx = (wx + a1 * self.max_torque / i_tilt * dt) * self.damp
                wy = (wy + a2 * self.max_torque / i_tilt * dt) * self.damp
                wz = (wz + a3 * self.max_torque / i_yaw * dt) * self.damp
                tx = tx + wx * dt; ty = ty + wy * dt; yaw = yaw + wz * dt
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                fwd = thrust * torch.sin(ty); lat = -thrust * torch.sin(tx)
                ax = (cy * fwd - sy * lat) / mass
                ay = (sy * fwd + cy * lat) / mass
                az = thrust * torch.cos(tx) * torch.cos(ty) / mass - g
                vx = vx + ax * dt; vy = vy + ay * dt; vz = vz + az * dt
                alt = alt + vz * dt
                if use_haz:
                    px = px + vx * dt; py = py + vy * dt
                if use_hgt:
                    dz = dz + vz * dt
                    alt = alt0 + dz - (self._hgt_at(px, py) - h0)
                fuel = (fuel - thrust / (p.dps_isp_s * G0) * dt).clamp(min=0.0)

                tilt = torch.sqrt(tx * tx + ty * ty)
                vxy = torch.sqrt(vx * vx + vy * vy)
                lat_t, vz_t = self._target(alt)
                run = (W["w_lat"] * ((vxy - lat_t) / 5.0) ** 2 + W["w_vz"] * ((vz - vz_t) / 2.0) ** 2
                       + W["w_tilt"] * (tilt / 0.26) ** 2
                       + W["w_rate"] * (wx * wx + wy * wy + wz * wz) / 0.25
                       + W["w_u"] * (a1 * a1 + a2 * a2 + a3 * a3) + W["w_time"]) * dt
                cost = cost + torch.where(alive, run, torch.zeros_like(run))

                touched = alive & (alt <= 0.0)
                lost = alive & (tilt > p.loss_of_control_tilt_rad)
                w_mag = torch.sqrt(wx * wx + wy * wy + wz * wz)
                sev = torch.maximum(torch.maximum(vz.abs() / lim_vz, vxy / lim_vxy),
                                    torch.maximum(tilt / lim_tilt, w_mag / lim_w))
                td = W["w_td"] * (0.2 * sev ** 2 + 10.0 * torch.relu(sev - 1.0) ** 2)
                if use_haz:
                    td = td + W["w_haz"] * self._haz_cost(px, py)
                cost = cost + torch.where(touched, td, torch.zeros_like(cost))
                cost = cost + torch.where(lost, W["w_td"] * 30.0 * torch.ones_like(cost), torch.zeros_like(cost))
                alive = alive & ~touched & ~lost
                step_i += 1
                # the episode budget ends: still airborne = the same failure as a hard crash
                late = alive & ((step_i * dt) >= t_left)[:, None]
                cost = cost + torch.where(late, 20.0 * W["w_td"] * torch.ones_like(cost), torch.zeros_like(cost))
                alive = alive & ~late
        # horizon end, still airborne: guidance error + tilt
        lat_t, vz_t = self._target(alt)
        vxy = torch.sqrt(vx * vx + vy * vy); tilt = torch.sqrt(tx * tx + ty * ty)
        term = W["w_T"] * (((vxy - lat_t) / 5.0) ** 2 + ((vz - vz_t) / 2.0) ** 2 + (tilt / 0.26) ** 2)
        cost = cost + torch.where(alive, term, torch.zeros_like(term))
        if use_haz:
            # still airborne at the horizon end: where does the vehicle come down if it keeps going?
            # lateral speed is bled off along the descent, so the mean drift to touchdown is ~half of it
            vxy_e = torch.sqrt(vx * vx + vy * vy)
            ux, uy = vx / vxy_e.clamp(min=0.1), vy / vxy_e.clamp(min=0.1)
            lat_e, _ = self._target(alt)
            drift = self._drift(alt) + 4.0 * (vxy_e - lat_e).clamp(min=0.0)   # braking tail ~ 0.5 * 8 s * excess
            cost = cost + torch.where(alive, W["w_haz"] * self._haz_cost(px + ux * drift, py + uy * drift),
                                      torch.zeros_like(cost))
        return cost
