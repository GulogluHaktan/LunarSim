"""Isaac Sim render of the RL landing + a live telemetry dashboard, side by side.

Reads the frames `isaaclab_replay_recording.py` rendered (and its index.csv,
which maps every frame to the recording step it shows) and the recording .npz
itself, then draws a dashboard for each frame from the SAME step, so the plots
and the picture can never drift apart.

Dashboard: altitude, lateral speed against its 1.2 m/s touchdown limit,
vertical speed against its 1.0 m/s limit, tilt -- all as progressive traces --
and the gain multipliers the policy is commanding at that moment, against the
controller's nominal 1.0. Colours: dataviz reference palette, dark surface.

Usage:
  ~/isaac-env/bin/python scripts/compose_rl_dashboard_video.py \
      --recording out/recordings/wave90000_policy.npz --env 10 \
      --frames out/videos/policy_env10_frames --out out/videos/rl_landing_dashboard.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SURF, PLANE, INK, INK2, GRID = "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "#383835"
POL, CRIT, GOOD = "#3987e5", "#d03b3b", "#0ca30c"
LABELS = {  # the knobs that govern the controller's touchdown failure mode
    "tilt_taper_v_xy_m_s": "yatış eşiği",
    "max_tilt_cmd_rad": "maks. yatış (seyir)",
    "final_approach_alt_m": "son yaklaşma irt.",
    "kd_xy_final": "yanal fren (son)",
    "max_tilt_cmd_final_rad": "maks. yatış (son)",
    "kp_vz_final": "dikey kazanç (son)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--still", default=None, help="also save the final composite frame here")
    ap.add_argument("--pattern", default="f{:05d}.png",
                    help="frame file name pattern; compose_capture_video's dashboard uses frame_{:04d}.png")
    ap.add_argument("--index-csv", default=None,
                    help="frame->step index; default <frames>/index.csv. A capture's frames_index.csv "
                         "(frame_index, step, t_s) is accepted too")
    args = ap.parse_args()

    d = np.load(args.recording)
    g = lambda k: d[f"e{args.env}_{k}"]  # noqa: E731
    dt = float(d["dt"]); hold = int(d["hold_steps"])
    alt, vx, vy, vz = g("alt"), g("vx"), g("vy"), g("vz")
    tilt = np.degrees(np.hypot(g("tilt_x"), g("tilt_y")))
    vxy = np.hypot(vx, vy)
    t = np.arange(len(alt)) * dt
    gains = g("gains")
    names = [str(s) for s in d["gain_names"]]
    LABELS.update({"throttle": "gaz (throttle)", "pitch": "pitch torku", "roll": "roll torku", "yaw": "yaw torku"})
    show = [j for j, nm in enumerate(names) if nm in LABELS]
    outcome = json.loads(str(g("outcome")))

    raw = list(csv.DictReader(open(args.index_csv or os.path.join(args.frames, "index.csv"))))
    rows = [{"frame": r.get("frame", r.get("frame_index")), "step": r["step"]} for r in raw]
    rows = [r for r in rows if os.path.exists(os.path.join(args.frames, args.pattern.format(int(r["frame"]))))]
    first = Image.open(os.path.join(args.frames, args.pattern.format(int(rows[0]["frame"]))))
    H = first.height
    W_DASH = int(round(H * 0.9))

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "font.size": 9})
    _rn = os.path.basename(args.recording)
    _TITLE = ("Telemetri — MPC (öğrenmesiz, model tabanlı), kraterli arazi" if _rn.startswith("mpc") else
              "Telemetri — PPO (kontrolcü kazançlarını ayarlar), kraterli arazi" if _rn.startswith("ppo") else
              "Telemetri — ES politikası (kontrolcüsüz), Isaac Sim")
    fig = plt.figure(figsize=(W_DASH / 100, H / 100), dpi=100, facecolor=PLANE)
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 0.28, 1.4], hspace=0.55,
                          left=0.24, right=0.95, top=0.93, bottom=0.05)
    fig.text(0.04, 0.965, _TITLE, fontsize=11, weight="bold")

    strips = [("irtifa (m)", alt, None), ("yanal hız (m/s)", vxy, 1.2),
              ("dikey hız (m/s)", np.abs(vz), 1.0), ("yatış (°)", tilt, None)]
    artists = []
    for r, (lab, y, lim) in enumerate(strips):
        ax = fig.add_subplot(gs[r])
        ax.set_facecolor(SURF)
        ax.set_xlim(0, t[-1]); ax.set_ylim(0, float(np.nanmax(y)) * 1.1)
        ax.grid(color=GRID, lw=0.5)
        ax.set_ylabel(lab, fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if lim is not None:
            ax.axhline(lim, color=CRIT, lw=1, ls=(0, (4, 3)))
            ax.text(t[-1], lim, f" limit {lim}", color=CRIT, fontsize=7.5, va="bottom", ha="right")
        if r == 3:
            ax.set_xlabel("zaman (s)", fontsize=8.5)
        else:
            ax.tick_params(labelbottom=False)
        line, = ax.plot([], [], color=POL, lw=1.8)
        dot, = ax.plot([], [], "o", color=POL, ms=6, mec=SURF, mew=1.5)
        val = ax.text(1.0, 1.04, "", transform=ax.transAxes, fontsize=9, color=INK,
                      ha="right", va="bottom", weight="bold")
        artists.append((y, line, dot, val, lim))

    axg = fig.add_subplot(gs[5])
    axg.set_facecolor(SURF)
    for s in ("top", "right"):
        axg.spines[s].set_visible(False)
    yy = np.arange(len(show))
    bars = axg.barh(yy, [0.0] * len(show), color=POL, height=0.62)
    axg.axvline(0.0, color=INK2, lw=1, ls=(0, (4, 3)))
    axg.set_yticks(yy)
    axg.set_yticklabels([LABELS[names[j]] for j in show], fontsize=8)
    axg.invert_yaxis()
    axg.set_xlim(-1.1, 1.1)
    axg.set_title("politikanın doğrudan eyleyici komutları ([-1, 1], kontrolcü yok)",
                  fontsize=8.5, color=INK2, loc="right")
    gtxt = [axg.text(0, j, "", va="center", fontsize=7.5, color=INK) for j in yy]

    tmp = os.path.splitext(args.out)[0] + "_composite"
    os.makedirs(tmp, exist_ok=True)
    last = None
    n_out = 0
    for row in rows:
        f, k = int(row["frame"]), int(row["step"])
        for y, line, dot, val, lim in artists:
            line.set_data(t[:k + 1], y[:k + 1])
            dot.set_data([t[k]], [y[k]])
            over = lim is not None and alt[k] < 15 and y[k] > lim
            val.set_text(f"{y[k]:.2f}")
            val.set_color(CRIT if over else INK)
        dec = min(k // hold, len(gains) - 1)
        for b, txt, j in zip(bars, gtxt, show):
            b.set_width(float(gains[dec, j]))
            txt.set_text(f" {gains[dec, j]:+.2f}")
            txt.set_x(max(float(gains[dec, j]), 0.0))
        fig.canvas.draw()
        dash = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frame = np.asarray(Image.open(os.path.join(args.frames, args.pattern.format(f))).convert("RGB"))
        comp = np.concatenate([frame, dash[:frame.shape[0]]], axis=1)
        Image.fromarray(comp).save(os.path.join(tmp, f"c{n_out:05d}.png"))
        n_out += 1
        last = comp
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                    os.path.join(tmp, "c%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out], check=True)
    if args.still and last is not None:
        Image.fromarray(last).save(args.still)
    print(f"wrote {args.out} ({len(rows)} frames); outcome landed={outcome['landed']}")


if __name__ == "__main__":
    main()
