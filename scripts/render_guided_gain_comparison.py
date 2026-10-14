"""Side-by-side video: the ZEM/ZEV controller vs the guided-gain RL policy, same
episode, same terrain, same release state -- recorded on Isaac by
`scripts/record_guided_gain_isaac.py`, whose recordings reproduce the paired
validation wave episode for episode.

Not a photoreal render. It draws what the physics recorded: the 3D path over
the actual terrain tile each env flew, the vehicle's body axis at its recorded
tilt, and live telemetry with the two touchdown limits that decide most
outcomes (lateral speed 1.2 m/s, vertical 1.0 m/s).

Colours are the dataviz reference palette (dark surface), validated with
`validate_palette.js`: controller = slot 2 orange, policy = slot 1 blue.
Identity is never colour alone -- every series is direct-labelled.

Usage:
  .venv/bin/python scripts/render_guided_gain_comparison.py \
      out/recordings/wave90000_controller.npz out/recordings/wave90000_policy.npz \
      --env 10 --out out/videos/orbit_descent_controller_vs_rl.mp4
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

SURFACE = "#1a1a19"
PLANE = "#0d0d0d"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#383835"
POLICY = "#3987e5"
CONTROLLER = "#d95926"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
V_XY_LIMIT = 1.2
V_Z_LIMIT = 1.0


def load(path, i):
    d = np.load(path)
    g = lambda k: d[f"e{i}_{k}"]
    out = {k: g(k) for k in ("x", "y", "alt", "ground_z", "vx", "vy", "vz",
                            "tilt_x", "tilt_y", "yaw", "throttle")}
    out["z"] = out["ground_z"] + out["alt"]
    out["vxy"] = np.hypot(out["vx"], out["vy"])
    out["tilt"] = np.degrees(np.hypot(out["tilt_x"], out["tilt_y"]))
    out["outcome"] = json.loads(str(g("outcome")))
    out["gains"] = g("gains")
    out["gain_names"] = [str(s) for s in d["gain_names"]]
    out["tile"] = (g("tile_height"), float(g("tile_res")), float(g("tile_size")))
    out["dt"] = float(d["dt"])
    out["hold"] = int(d["hold_steps"])
    out["t"] = np.arange(len(out["x"])) * out["dt"]
    return out


def outcome_text(run):
    o, imp = run["outcome"], run["outcome"]["impact"]
    vxy = float(np.hypot(imp.get("vx", np.nan), imp.get("vy", np.nan)))
    vz = abs(float(imp.get("vz", np.nan)))
    if o["landed"]:
        return f"İNDİ  ✓   yanal {vxy:.2f} m/s · dikey {vz:.2f} m/s", GOOD
    why = (f"yanal {vxy:.2f} > {V_XY_LIMIT} m/s" if vxy > V_XY_LIMIT
           else f"dikey {vz:.2f} > {V_Z_LIMIT} m/s")
    return f"ÇAKILDI  ✕   {why}", CRIT


def body_axis(run, k, length):
    tx, ty, yaw = run["tilt_x"][k], run["tilt_y"][k], run["yaw"][k]
    # same small-angle body-axis direction the env uses for thrust
    fwd, lat = np.sin(ty), -np.sin(tx)
    ux = np.cos(yaw) * fwd - np.sin(yaw) * lat
    uy = np.sin(yaw) * fwd + np.cos(yaw) * lat
    uz = np.cos(tx) * np.cos(ty)
    v = np.array([ux, uy, uz])
    v /= np.linalg.norm(v)
    return v * length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller_npz")
    ap.add_argument("policy_npz")
    ap.add_argument("--env", type=int, default=10)
    ap.add_argument("--out", default="out/videos/orbit_descent_controller_vs_rl.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speedup", type=float, default=2.0)
    ap.add_argument("--slow-below-m", type=float, default=12.0,
                    help="play in real time below this altitude, where the outcome is decided")
    ap.add_argument("--hold-s", type=float, default=3.0)
    args = ap.parse_args()

    C = load(args.controller_npz, args.env)
    P = load(args.policy_npz, args.env)
    runs = [("ZEM/ZEV kontrolcü", C, CONTROLLER), ("RL politika (PPO, kazanç ayarlı)", P, POLICY)]

    # Frame schedule in simulated time: fast during the long cruise, real time
    # once either vehicle is below `slow_below_m`, then a hold on the result.
    T = max(C["t"][-1], P["t"][-1])
    alt_min = np.minimum(np.interp(np.arange(0, T, 0.01), C["t"], C["alt"]),
                         np.interp(np.arange(0, T, 0.01), P["t"], P["alt"]))
    t_slow = float(np.arange(0, T, 0.01)[np.argmax(alt_min < args.slow_below_m)])
    times, t = [], 0.0
    while t < T:
        times.append(t)
        t += (args.speedup if t < t_slow else 1.0) / args.fps
    times += [T] * int(args.hold_s * args.fps)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 10})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=PLANE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.28, wspace=0.22,
                          left=0.05, right=0.98, top=0.90, bottom=0.07)
    fig.text(0.05, 0.955, "Ay inişi · Isaac Sim fiziği · orbit_descent · aynı başlangıç, aynı arazi",
             fontsize=15, color=INK, weight="bold")
    fig.text(0.05, 0.925, "Doğrulama dalgası 90000, env %d — eğitimin hiç görmediği bir episod. "
             "192 taze episodda: kontrolcü %%52.6, RL politika %%89.6." % args.env,
             fontsize=10.5, color=INK2)

    # ---- two 3D scenes ----
    xs = np.concatenate([C["x"], P["x"]]); ys = np.concatenate([C["y"], P["y"]])
    pad = 40.0
    xlo, xhi = xs.min() - pad, xs.max() + pad
    ylo, yhi = ys.min() - pad, ys.max() + pad
    span = max(xhi - xlo, yhi - ylo)
    xc, yc = (xlo + xhi) / 2, (ylo + yhi) / 2
    xlo, xhi, ylo, yhi = xc - span / 2, xc + span / 2, yc - span / 2, yc + span / 2

    # The 3D pair takes the first two columns; the third holds the decisive
    # close-up.
    scenes = [fig.add_subplot(gs[0, col], projection="3d") for col in range(2)]
    for ax, (label, run, color) in zip(scenes, runs):
        h, res, size = run["tile"]
        n = h.shape[0]
        ax_coords = (np.arange(n) - (n - 1) / 2) * res
        ix = np.where((ax_coords >= xlo - res) & (ax_coords <= xhi + res))[0]
        iy = np.where((ax_coords >= ylo - res) & (ax_coords <= yhi + res))[0]
        X, Y = np.meshgrid(ax_coords[ix], ax_coords[iy], indexing="ij")
        Z = h[np.ix_(ix, iy)]
        ax.plot_surface(X, Y, Z, cmap="gray", alpha=0.55, linewidth=0, antialiased=True,
                        vmin=Z.min() - 2, vmax=Z.max() + 2)
        ax.set_facecolor(SURFACE)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.10, 0.10, 0.10, 1.0))
            a._axinfo["grid"]["color"] = GRID
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_zlim(Z.min() - 2, max(C["z"].max(), P["z"].max()) + 5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-58)
        ax.set_title(label, color=color, fontsize=13, weight="bold", pad=2)
        run["_path"], = ax.plot([], [], [], color=color, lw=2)
        run["_body"], = ax.plot([], [], [], color=INK, lw=3, solid_capstyle="round")
        run["_dot"], = ax.plot([], [], [], "o", color=color, ms=8, mec=SURFACE, mew=2)
        run["_hud"] = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                                color=INK2, va="top", family="DejaVu Sans Mono")
        run["_result"] = ax.text2D(0.5, 0.06, "", transform=ax.transAxes, fontsize=14,
                                   weight="bold", ha="center")

    # ---- decisive close-up: lateral speed over the last metres ----
    axz = fig.add_subplot(gs[0, 2])
    axz.set_facecolor(SURFACE)
    axz.axhline(V_XY_LIMIT, color=INK2, lw=1, ls=(0, (4, 3)))
    axz.text(14.6, V_XY_LIMIT + 0.05, "temas limiti 1.2 m/s", color=INK2, fontsize=9, ha="right")
    axz.set_xlim(15, 0); axz.set_ylim(0, 3.2)
    axz.set_xlabel("irtifa (m)  — sağa doğru yere yaklaşıyor")
    axz.set_ylabel("yanal hız (m/s)")
    axz.set_title("Kararın verildiği yer: son 15 m", color=INK, fontsize=12, weight="bold")
    axz.grid(color=GRID, lw=0.5)
    for _, run, color in runs:
        run["_zl"], = axz.plot([], [], color=color, lw=2)
        run["_zd"], = axz.plot([], [], "o", color=color, ms=8, mec=SURFACE, mew=2)

    # ---- telemetry strips ----
    strips = [("irtifa (m)", "alt", None, None),
              ("yanal hız (m/s)", "vxy", V_XY_LIMIT, "1.2 limit"),
              ("yatış (°)", "tilt", None, None)]
    strip_axes = []
    for k, (ylabel, key, lim, limlabel) in enumerate(strips):
        ax = fig.add_subplot(gs[1, k])
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, T); ax.grid(color=GRID, lw=0.5)
        ymax = max(C[key].max(), P[key].max()) * 1.08
        ax.set_ylim(0, ymax)
        ax.set_xlabel("zaman (s)"); ax.set_ylabel(ylabel)
        if lim is not None:
            ax.axhline(lim, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(T * 0.99, lim + ymax * 0.02, limlabel, color=INK2, fontsize=8.5, ha="right")
        lines = []
        for name, run, color in runs:
            ln, = ax.plot([], [], color=color, lw=2)
            lines.append((run, ln))
        strip_axes.append((ax, key, lines))
    # direct labels on the first strip (identity never colour-alone)
    ax0 = strip_axes[0][0]
    ax0.text(0.98, 0.92, "turuncu: kontrolcü", transform=ax0.transAxes, ha="right",
             color=INK2, fontsize=9)
    ax0.text(0.98, 0.82, "mavi: RL politika", transform=ax0.transAxes, ha="right",
             color=INK2, fontsize=9)

    def idx(run, tt):
        return int(min(np.searchsorted(run["t"], tt, side="right") - 1, len(run["t"]) - 1))

    def update(f):
        tt = times[f]
        for name, run, color in runs:
            k = idx(run, tt)
            run["_path"].set_data_3d(run["x"][:k + 1], run["y"][:k + 1], run["z"][:k + 1])
            p = np.array([run["x"][k], run["y"][k], run["z"][k]])
            b = body_axis(run, k, 14.0)
            run["_body"].set_data_3d([p[0], p[0] + b[0]], [p[1], p[1] + b[1]], [p[2], p[2] + b[2]])
            run["_dot"].set_data_3d([p[0]], [p[1]], [p[2]])
            gtxt = ""
            if run is P and len(run["gains"]):
                d = min(k // run["hold"], len(run["gains"]) - 1)
                names = run["gain_names"]
                g = run["gains"][d]
                pick = {"kd_xy_final": "yanal fren", "max_tilt_cmd_final_rad": "son yatış",
                        "tilt_taper_v_xy_m_s": "yatış eşiği"}
                gtxt = "\n" + "\n".join(f"{pick[nm]:<11}×{g[j]:.2f}"
                                        for j, nm in enumerate(names) if nm in pick)
            run["_hud"].set_text(
                f"t {run['t'][k]:5.1f} s\nirtifa {run['alt'][k]:6.1f} m\n"
                f"yanal  {run['vxy'][k]:5.2f} m/s\ndikey  {abs(run['vz'][k]):5.2f} m/s\n"
                f"yatış  {run['tilt'][k]:5.1f}°" + gtxt)
            if k == len(run["t"]) - 1:
                txt, col = outcome_text(run)
                run["_result"].set_text(txt); run["_result"].set_color(col)
            else:
                run["_result"].set_text("")
            m = run["alt"][:k + 1] <= 15
            run["_zl"].set_data(run["alt"][:k + 1][m], run["vxy"][:k + 1][m])
            if run["alt"][k] <= 15:
                run["_zd"].set_data([run["alt"][k]], [run["vxy"][k]])
            else:
                run["_zd"].set_data([], [])
        for ax, key, lines in strip_axes:
            for run, ln in lines:
                k = idx(run, tt)
                ln.set_data(run["t"][:k + 1], run[key][:k + 1])
        return []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=6000,
                                                      extra_args=["-pix_fmt", "yuv420p"]),
              savefig_kwargs={"facecolor": PLANE})
    # a still of the final frame, for the report
    update(len(times) - 1)
    fig.savefig(os.path.splitext(args.out)[0] + "_final.png", facecolor=PLANE, dpi=110)
    print(f"wrote {args.out} ({len(times)} frames, {len(times) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
