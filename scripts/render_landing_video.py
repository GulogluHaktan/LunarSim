"""Renders an MP4 of a successful `AnalyticLanderEnv` landing rollout using
a trained SAC policy: no Isaac Sim needed, matplotlib only (a 3D flight
path over the terrain + live altitude/vz/vxy/tilt/throttle telemetry).

This is NOT a photorealistic render (that's `lunarsim.adapters.isaac` +
Isaac Sim) -- it's a fast, dependency-light way to see a trained policy's
actual flight profile end to end.

Usage: .venv/bin/python scripts/render_landing_video.py \
    --checkpoint /tmp/hover_checkpoint_v11.zip --stage hover_only \
    --out out/hover_only_landing.mp4
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from stable_baselines3 import SAC

sys.path.insert(0, "scripts")
from train_sac_curriculum import STAGES, make_env  # noqa: E402


def rollout(checkpoint: str, stage_name: str, seed: int, max_seeds_to_try: int = 200):
    """Runs episodes until it finds one that actually lands safely (or
    gives up and returns the last one), recording full state history."""
    stage = next(s for s in STAGES if s.name == stage_name)
    model = SAC.load(checkpoint, device="cpu")

    for attempt in range(max_seeds_to_try):
        env = make_env(stage, seed=seed + attempt)()
        obs, _ = env.reset(seed=seed + attempt)
        history = []
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            s = env.state
            ground_z = env._ground_z(s["x"], s["y"])
            history.append(dict(
                x=s["x"], y=s["y"], z=s["z"], ground_z=ground_z,
                vx=s["vx"], vy=s["vy"], vz=s["vz"],
                tilt_x=s["tilt_x"], tilt_y=s["tilt_y"],
                throttle=s["throttle"],
                rcs_pitch=s["rcs_pitch"], rcs_roll=s["rcs_roll"], rcs_yaw=s["rcs_yaw"],
                t=info["t_s"],
            ))
            if terminated or truncated:
                landed = info.get("landed_safely", False)
                print(f"seed {seed + attempt}: landed_safely={landed} "
                      f"(t={info['t_s']:.1f}s, {len(history)} steps)")
                if landed or attempt == max_seeds_to_try - 1:
                    return history, landed, env
                break
    raise RuntimeError("no landing found")


def render(history: list[dict], landed: bool, tile, out_path: str, fps: int = 30, speedup: int = 2):
    frames = history[::speedup]
    t = np.array([f["t"] for f in frames])
    x = np.array([f["x"] for f in frames])
    y = np.array([f["y"] for f in frames])
    z = np.array([f["z"] for f in frames])
    ground_z = np.array([f["ground_z"] for f in frames])
    alt = z - ground_z
    vz = np.array([f["vz"] for f in frames])
    vxy = np.array([np.hypot(f["vx"], f["vy"]) for f in frames])
    tilt_deg = np.array([np.degrees(np.hypot(f["tilt_x"], f["tilt_y"])) for f in frames])
    throttle = np.array([f["throttle"] for f in frames])

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_alt = fig.add_subplot(4, 2, 2)
    ax_vz = fig.add_subplot(4, 2, 4)
    ax_vxy = fig.add_subplot(4, 2, 6)
    ax_tilt = fig.add_subplot(4, 2, 8)

    # terrain surface under the flight path's footprint
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
    ax3d.set_title(f"Apollo LM analytic landing -- {'LANDED SAFELY' if landed else 'touchdown'}")

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
    c_vxy = strip(ax_vxy, vxy, "vxy (m/s)", 0.5, "tab:purple")
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage", type=str, default="hover_only")
    parser.add_argument("--out", type=str, default="out/landing.mp4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    history, landed, env = rollout(args.checkpoint, args.stage, args.seed)
    render(history, landed, env.tile, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
