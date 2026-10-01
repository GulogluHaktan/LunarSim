"""Renders an MP4 of a successful `ZemZevController` (hand-designed, non-RL
guidance+attitude) landing rollout on `AnalyticLanderEnv`: no Isaac Sim
needed, matplotlib only (a 3D flight path over the terrain + live
altitude/vz/vxy/tilt telemetry). Deliberately does NOT touch Isaac Sim --
this is meant to be safe to run alongside a live GPU training job (see
`scripts/render_landing_video.py`'s docstring for the same rationale, this
is its ZemZevController-driven sibling).

Usage: .venv/bin/python scripts/render_controller_landing_video.py \
    --stage orbit_descent --out out/orbit_descent_controller_landing.mp4
"""
from __future__ import annotations

import argparse
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

sys.path.insert(0, ".")
from scripts.test_landing_feasibility import STAGES, _terrain_config  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv  # noqa: E402
from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains  # noqa: E402


def rollout(stage_name: str, seed: int, max_seeds_to_try: int = 60):
    stage = STAGES[stage_name]
    gains = ZemZevGains()

    for attempt in range(max_seeds_to_try):
        s = seed + attempt
        tile = generate_tile(_terrain_config(stage, s))
        env = AnalyticLanderEnv(tile=tile, params=stage.params, reward_fn=lambda e, i: 0.0)
        ctrl = ZemZevController(stage.params, gains)
        obs, _ = env.reset(seed=s)
        ctrl.reset()
        history = []
        for _ in range(2000):
            action = ctrl.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            st = env.state
            ground_z = env._ground_z(st["x"], st["y"])
            history.append(dict(
                x=st["x"], y=st["y"], z=st["z"], ground_z=ground_z,
                vx=st["vx"], vy=st["vy"], vz=st["vz"],
                tilt_x=st["tilt_x"], tilt_y=st["tilt_y"],
                throttle=st["throttle"],
                t=info["t_s"],
            ))
            if terminated or truncated:
                landed = info.get("landed_safely", False)
                print(f"seed {s}: landed_safely={landed} (t={info['t_s']:.1f}s, {len(history)} steps)")
                if landed or attempt == max_seeds_to_try - 1:
                    return history, landed, env
                break
    raise RuntimeError("no landing found")


def render(history: list[dict], landed: bool, tile, stage_name: str, out_path: str, fps: int = 30, speedup: int = 2):
    frames = history[::speedup]
    t = np.array([f["t"] for f in frames])
    x = np.array([f["x"] for f in frames])
    y = np.array([f["y"] for f in frames])
    z = np.array([f["z"] for f in frames])
    ground_z = np.array([f["ground_z"] for f in frames])
    alt = z - ground_z
    vz = np.array([f["vz"] for f in frames])
    vxy = np.array([math.hypot(f["vx"], f["vy"]) for f in frames])
    tilt_deg = np.array([math.degrees(math.hypot(f["tilt_x"], f["tilt_y"])) for f in frames])

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_alt = fig.add_subplot(4, 2, 2)
    ax_vz = fig.add_subplot(4, 2, 4)
    ax_vxy = fig.add_subplot(4, 2, 6)
    ax_tilt = fig.add_subplot(4, 2, 8)

    pad = max(20.0, 1.2 * max(np.max(np.abs(x)), np.max(np.abs(y)), 10.0))
    n = 40
    gx = np.linspace(-pad, pad, n)
    gy = np.linspace(-pad, pad, n)
    GX, GY = np.meshgrid(gx, gy)
    from lunarsim.core.terrain.rocks import sample_height_at
    GZ = sample_height_at(tile.height, tile.res_m, GX.ravel(), GY.ravel()).reshape(GX.shape)
    ax3d.plot_surface(GX, GY, GZ, color="0.75", alpha=0.6, linewidth=0, antialiased=True)

    ax3d.plot(x, y, z, color="tab:blue", lw=1.5, alpha=0.6)
    point, = ax3d.plot([x[0]], [y[0]], [z[0]], "o", color="tab:red", markersize=8)
    trail, = ax3d.plot([], [], [], color="tab:orange", lw=2.5)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_title(f"ZemZevController ({stage_name}) -- {'LANDED SAFELY' if landed else 'touchdown'}")

    def strip(ax, ydata, label, limit, color):
        ax.plot(t, ydata, color=color, lw=1.2)
        ax.axhline(limit, color="red", ls="--", lw=0.8)
        ax.axhline(-limit, color="red", ls="--", lw=0.8)
        cursor = ax.axvline(t[0], color="k", lw=1)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=7)
        return cursor

    c_alt = strip(ax_alt, alt, "alt (m)", 1e9, "tab:green")
    ax_alt.set_ylim(0, max(alt) * 1.1 + 1)
    c_vz = strip(ax_vz, vz, "vz (m/s)", 1.0, "tab:blue")
    c_vxy = strip(ax_vxy, vxy, "vxy (m/s)", 1.2, "tab:purple")
    c_tilt = strip(ax_tilt, tilt_deg, "tilt (deg)", 15.0, "tab:brown")
    ax_tilt.set_xlabel("t (s)")

    fig.tight_layout()

    def update(i):
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        lo = max(0, i - 40)
        trail.set_data(x[lo:i + 1], y[lo:i + 1])
        trail.set_3d_properties(z[lo:i + 1])
        for c in (c_alt, c_vz, c_vxy, c_tilt):
            c.set_xdata([t[i], t[i]])
        return point, trail, c_alt, c_vz, c_vxy, c_tilt

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(out_path, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="orbit_descent", choices=list(STAGES))
    parser.add_argument("--out", type=str, default="out/controller_landing.mp4")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    history, landed, env = rollout(args.stage, args.seed)
    render(history, landed, env.tile, args.stage, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
