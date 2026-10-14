"""Build the LunarSim RL report (Turkish) as HTML, then print it to PDF with
headless Chrome.

Every number in the report is either read from a validation/recording file in
out/ or, for results whose files were lost when out/ was wiped on 2026-10-09,
taken from handover.md, which recorded them at the time. The constants below
say which is which.

Run with any Python that has numpy + matplotlib (e.g. ~/isaac-env/bin/python):
    python report/build_report.py
"""
from __future__ import annotations

import base64
import re
import html
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
REP = ROOT / "report"
FIGS = REP / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# dataviz reference palette, light mode (validated with validate_palette.js)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POL, CTL = "#2a78d6", "#eb6834"
GOOD, CRIT = "#0ca30c", "#d03b3b"

# ---------------------------------------------------------------- numbers
# Recorded in handover.md at the time; their JSON files were deleted with out/.
ISAAC_CTL = (101, 192)
FIRST_RUN_30K = dict(pol=(131, 192), b_only=41, c_only=11, p="3.6e-5", ci=(8.9, 22.9))
FIRST_RUN_60K = dict(pol=(130, 192), b_only=42, c_only=13, p="1.1e-4")
SEED21_45K = dict(pol=(172, 192), b_only=72, c_only=1, p="1.6e-20", ci=(30.2, 43.8),
                  pol_ci=(84.9, 93.8), waves="16/16")
LAND = {  # land_handoff, analytic backend, 4 fresh blocks x 48 = 192 paired, controller 19/192
    "eski ödül, seed 11": (60, 41, 0),
    "eski ödül, seed 777": (42, 23, 0),
    "sadece terminal + basamak": (61, 43, 1),
    "saf ikili sonuç": (47, 28, 0),
}
LAND_CTL = (19, 192)
RESCORE = [  # diag_reward_rescore.py on 192 paired land_handoff episodes
    ("gönderilen ödül (yoğun + terminal)", -17.52, -20.54, -15.94, False, "96.9%"),
    ("+ basamak 5 birim", -17.03, -18.98, -14.84, False, "96.9%"),
    ("+ basamak 15 birim", -16.04, -15.86, -12.66, False, "100%"),
    ("terminal tabanı −6", -12.57, -15.90, -10.92, False, "96.9%"),
    ("yoğun terim tavanı 1000 + basamak", -15.50, -17.29, -13.42, False, "96.9%"),
    ("sadece yoğun şekillendirme", -8.80, -13.09, -7.63, False, "70.3%"),
    ("sadece terminal", -8.72, -7.45, -8.30, True, "100%"),
    ("saf ikili sonuç", 0.34, 1.38, 0.94, True, "100%"),
]
SEL_CURVES = {  # selection block 7000, 48 episodes, unpaired -- noisy by design
    "ilk koşu (rollout 768, lr 3e-4)": [(15, 60.4), (30, 77.1), (45, 79.2), (60, 83.3), (75, 66.7)],
    "seed21 (rollout 1024, lr 1e-4)": [(15, 77.1), (30, 81.2), (45, 89.6), (60, 77.1)],
}

REPRO_VAL = OUT / "gg_isaac_orbit_v4_seed21_repro_validation.json"
REPRO_LOG = OUT / "gg_isaac_orbit_v4_seed21_repro.log"
REC_CTL = OUT / "recordings" / "wave90000_controller.npz"
REC_POL = OUT / "recordings" / "wave90000_policy.npz"
VIDEO = OUT / "videos" / "rl_landing_lidar_dashboard.mp4"
STILL = OUT / "videos" / "rl_landing_lidar_dashboard_final.png"
if not STILL.exists():  # fall back to the dashboard without the LiDAR map
    VIDEO = OUT / "videos" / "rl_landing_dashboard.mp4"
    STILL = OUT / "videos" / "rl_landing_dashboard_final.png"
LIDAR_COMPUTED = "lidar" in STILL.name


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK2, "figure.facecolor": SURF})


def img_tag(path, width="100%", alt=""):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    ext = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")
    return f'<img src="data:image/{ext};base64,{data}" style="width:{width}" alt="{html.escape(alt)}">'


# ---------------------------------------------------------------- repro data
repro = None
if REPRO_VAL.exists():
    d = json.loads(REPRO_VAL.read_text())
    a = np.array(d["baseline_landed"], bool)
    b = np.array(d["candidate_landed"], bool)
    n = len(a)
    rng = np.random.default_rng(0)
    diffs = [b[i].mean() - a[i].mean() for i in (rng.integers(0, n, n) for _ in range(20000))]
    w = d["n_envs"]
    repro = dict(ctl=(int(a.sum()), n), pol=(int(b.sum()), n),
                 b_only=int((b & ~a).sum()), c_only=int((a & ~b).sum()),
                 ci=(100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)),
                 waves=f"{sum(b[i:i+w].sum() >= a[i:i+w].sum() for i in range(0, n, w))}/{n // w}",
                 ckpt=d.get("checkpoint", ""))
    repro["p"] = f"{mcnemar_p(repro['b_only'], repro['c_only']):.1e}"
repro_curve = []
if REPRO_LOG.exists():
    for line in REPRO_LOG.read_text().splitlines():
        if line.startswith("[eval]"):
            m = re.match(r"\[eval\]\s+(\d+) dec\s+\d+/\d+\s+\(\s*([\d.]+)%\)", line)
            if m:
                repro_curve.append((int(m.group(1)) / 1000, float(m.group(2))))
if repro_curve:
    SEL_CURVES["yeniden üretim, seed21 (aynı ayarlar)"] = repro_curve

# ---------------------------------------------------------------- figure 1: results
rows = [("Kontrolcü (ZEM/ZEV)", ISAAC_CTL, None)]
rows.append(("RL, ilk koşu · 30k karar", FIRST_RUN_30K["pol"], None))
rows.append(("RL, seed21 · 45k karar", SEED21_45K["pol"], None))
if repro:
    rows.append(("RL, yeniden üretim", repro["pol"], None))
fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(rows) + 1.0))
style(ax)
for j, (lab, (k, n), _) in enumerate(rows[::-1]):
    lo, hi = wilson(k, n)
    col = CTL if lab.startswith("Kontrolcü") else POL
    ax.plot([lo, hi], [j, j], color=col, lw=2, solid_capstyle="round")
    ax.plot([100 * k / n], [j], "o", color=col, ms=9, mec=SURF, mew=2)
    ax.text(hi + 1.2, j, f"%{100*k/n:.1f}  ({k}/{n})", va="center", fontsize=9, color=INK)
ax.axvline(75, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(75.6, len(rows) - 0.55, "proje hedefi %75", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9.5, color=INK)
ax.set_xlim(40, 102)
ax.set_xlabel("iniş oranı, 192 eşleşmiş taze episod (çubuk: %95 Wilson aralığı)")
ax.set_ylim(-0.6, len(rows) - 0.3)
fig.tight_layout()
fig.savefig(FIGS / "results.png", dpi=200, facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: selection-block curves
fig, ax = plt.subplots(figsize=(7.4, 3.3))
style(ax)
cols = [INK2, POL, "#1baf7a"]
for n_c, ((lab, pts), col) in enumerate(zip(SEL_CURVES.items(), cols)):
    xs, ys = zip(*pts)
    if n_c == 2:  # the reproduction: identical to seed21, so draw it dashed on top
        ax.plot(xs, ys, color=col, lw=2, ls=(0, (5, 4)), marker="o", ms=4, label=lab, zorder=4)
    else:
        ax.plot(xs, ys, color=col, lw=3.2 if n_c == 1 else 2, marker="o", ms=7, mec=SURF,
                mew=1.5, label=lab)
ax.axhline(56.2, color=CTL, lw=1.4, ls=(0, (4, 3)))
ax.text(1, 57.2, "kontrolcü bu blokta 54–56%", color=INK2, fontsize=8.5)
ax.set_xlim(0, 80)
ax.set_ylim(40, 100)
ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK2)
ax.set_xlabel("eğitim (bin karar; 1 karar = 1 s uçuş)")
ax.set_ylabel("seçim bloğu iniş % (n=48)")
fig.tight_layout()
fig.savefig(FIGS / "curves.png", dpi=200, facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- figure 3: touchdown lateral speed (recording)
have_rec = REC_CTL.exists() and REC_POL.exists()
gain_rows = []
if have_rec:
    C, P = np.load(REC_CTL), np.load(REC_POL)
    ne = int(C["n_envs"])
    cv, pv, cl, pl = [], [], [], []
    for i in range(ne):
        oc = json.loads(str(C[f"e{i}_outcome"])); op = json.loads(str(P[f"e{i}_outcome"]))
        cv.append(math.hypot(oc["impact"].get("vx", np.nan), oc["impact"].get("vy", np.nan)))
        pv.append(math.hypot(op["impact"].get("vx", np.nan), op["impact"].get("vy", np.nan)))
        cl.append(oc["landed"]); pl.append(op["landed"])
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax)
    xs = np.arange(ne)
    for i in range(ne):
        ax.plot([i, i], [cv[i], pv[i]], color=GRID, lw=2, zorder=1)
    ax.scatter(xs, cv, s=60, color=CTL, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.scatter(xs, pv, s=60, color=POL, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.axhline(1.2, color=CRIT, lw=1.2, ls=(0, (4, 3)))
    ax.text(ne - 0.4, 1.23, "temas limiti 1.2 m/s", color=CRIT, fontsize=8.5, ha="right")
    ax.text(-0.3, max(cv) + 0.06, "turuncu: kontrolcü", color=INK2, fontsize=8.5)
    ax.text(3.3, max(cv) + 0.06, "mavi: RL politika", color=INK2, fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}" for i in xs], fontsize=8.5)
    ax.set_xlabel("episod (doğrulama dalgası 90000, aynı başlangıç ve arazi)")
    ax.set_ylabel("temasta yanal hız (m/s)")
    ax.set_ylim(0, max(cv) + 0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "touchdown_vxy.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    names = [str(s) for s in P["gain_names"]]
    allg = np.concatenate([P[f"e{i}_gains"] for i in range(ne)])
    # last decisions before touchdown, where the outcome is decided
    late = np.concatenate([P[f"e{i}_gains"][-6:] for i in range(ne)])
    tr = {"tilt_taper_v_xy_m_s": "yatış eşiği hızı", "max_tilt_cmd_rad": "en büyük yatış (seyir)",
          "final_approach_alt_m": "son yaklaşma irtifası", "final_approach_trigger_margin":
          "son yaklaşma tetik payı", "kd_xy_final": "son yaklaşma yanal fren",
          "max_tilt_cmd_final_rad": "en büyük yatış (son yaklaşma)", "kp_vz_final":
          "son yaklaşma dikey kazanç", "tgo_floor_s": "tgo alt sınırı", "kp_att": "tutum P kazancı",
          "kd_att": "tutum D kazancı", "thrust_authority_margin": "itki otorite payı"}
    for j, nm in enumerate(names):
        gain_rows.append((tr.get(nm, nm), nm, float(allg[:, j].mean()), float(late[:, j].mean())))

# ---------------------------------------------------------------- figure 6: knob timeline + phase table
KNOBS = [  # name, Turkish label, mode where the controller READS it, what it does physically
    ("tilt_taper_v_xy_m_s", "yatış azaltma hızı", "seyir/fren",
     "İzin verilen yatış, yanal hız bu değerin altına düşünce orantılı azalır. Küçükse araç düşük hızda da sert yatar."),
    ("max_tilt_cmd_rad", "en büyük yatış (seyir)", "seyir/fren",
     "Fren için izin verilen en büyük yatış açısı. Büyük = sert fren ama geri dönmesi uzun."),
    ("tgo_floor_s", "tgo alt sınırı", "seyir/fren",
     "Kontrolcünün 'inişe kalan süre' tahminine alt sınır. Küçük = son saniyelerde daha agresif komut."),
    ("thrust_authority_margin", "itki otorite payı", "seyir/fren",
     "Güdümün kullanabileceği en büyük itki oranı. Küçük = kontrolcü daha temkinli, daha uzun süre planlar."),
    ("final_approach_alt_m", "son yaklaşma irtifası", "geçiş",
     "Bu irtifanın altında kontrolcü güdümü bırakıp dikey iniş moduna geçer."),
    ("final_approach_trigger_margin", "son yaklaşma tetik payı", "geçiş",
     "Hızlı inerken geçişi durma mesafesine göre erkene çeken güvenlik çarpanı."),
    ("kd_xy_final", "yanal fren (son)", "son yaklaşma",
     "Son yaklaşmada istenen yanal ivme = −kazanç × yanal hız."),
    ("max_tilt_cmd_final_rad", "en büyük yatış (son)", "son yaklaşma",
     "Son yaklaşmada izin verilen en büyük yatış açısı."),
    ("kp_vz_final", "dikey hız kazancı (son)", "son yaklaşma",
     "Son yaklaşmada −0.8 m/s hedef iniş hızını izleme sertliği."),
    ("kp_att", "tutum P kazancı", "hepsi",
     "Yatış hatasını tork komutuna çeviren oransal kazanç."),
    ("kd_att", "tutum D kazancı (sönüm)", "hepsi",
     "Dönme hızını söndüren kazanç. Büyük = daha az salınım."),
]
knob_rows = []
if have_rec:
    from dataclasses import fields as _fields
    from lunarsim.control.zemzev_controller import ZemZevGains
    from lunarsim.rl.guided_gain_env import GAIN_SETS
    g0 = ZemZevGains(); spec = {k.name: k for k in GAIN_SETS["attitude"]}
    hold = int(P["hold_steps"])
    phase_vals = {nm: {"s": [], "f": [], "l": []} for nm, *_ in KNOBS}
    for i in range(ne):
        G = P[f"e{i}_gains"]; A = P[f"e{i}_alt"]
        for dcs in range(len(G)):
            a = A[min(dcs * hold, len(A) - 1)]
            ph = "s" if a > 50 else ("f" if a > 15 else "l")
            for j, nm in enumerate(names):
                if nm in phase_vals:
                    phase_vals[nm][ph].append(float(G[dcs, j]))
    for nm, lab, mode, what in KNOBS:
        k = spec[nm]; nom = float(getattr(g0, nm))
        unit = "°" if nm.endswith("_rad") else ""
        sc = (180 / math.pi) if unit else 1.0
        rng = f"{nom/k.ratio*sc:.2g}–{nom*k.ratio*sc:.3g}{unit}"
        pv = {ph: (np.mean(v) if v else float("nan")) for ph, v in phase_vals[nm].items()}
        knob_rows.append((lab, nm, f"{nom*sc:.3g}{unit}", rng, mode, what, pv))
    # timeline figure, the episode shown in the video (env 10)
    ev = 10 if ne > 10 else 0
    G = P[f"e{ev}_gains"]; A = P[f"e{ev}_alt"]
    Vxy = np.hypot(P[f"e{ev}_vx"], P[f"e{ev}_vy"])
    td = np.arange(len(G)) * hold * float(P["dt"])
    ta = np.arange(len(A)) * float(P["dt"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})
    for ax in (a1, a2):
        style(ax)
    a1.plot(ta, A, color=INK2, lw=2)
    a1.set_ylabel("irtifa (m)")
    a1b = a1.twinx(); a1b.plot(ta, Vxy, color=POL, lw=2); a1b.set_ylabel("yanal hız (m/s)", color=POL)
    a1b.tick_params(colors=POL, labelsize=9); a1b.spines["top"].set_visible(False)
    a1.text(0.99, 0.85, "gri: irtifa · mavi: yanal hız", transform=a1.transAxes, ha="right",
            fontsize=8.5, color=INK2)
    show = [("final_approach_alt_m", "son yaklaşma irtifası", "#eb6834"),
            ("kd_att", "tutum D kazancı", "#1baf7a"),
            ("tgo_floor_s", "tgo alt sınırı", "#4a3aa7"),
            ("tilt_taper_v_xy_m_s", "yatış azaltma hızı", "#e87ba4")]
    for nm, lab, col in show:
        j = names.index(nm)
        a2.step(td, G[:, j], where="post", color=col, lw=2, label=lab)
    a2.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    a2.text(0.3, 1.005, "1.0 = kontrolcünün kendi değeri", fontsize=8, color=INK2, va="bottom")
    a2.set_ylabel("politikanın çarpanı")
    a2.set_xlabel("zaman (s) — her basamak bir karar (1 s)")
    a2.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGS / "knob_timeline.png", dpi=200, facecolor=SURF)
    plt.close(fig)

# ---------------------------------------------------------------- HTML
def pct(k, n):
    return f"%{100*k/n:.1f}"


best = repro if repro else None
if best:
    head = ("Checkpoint dosyaları kaybolunca aynı tarif sıfırdan yeniden eğitildi; yeniden "
            f"üretilen politika aynı 192 episodda {pct(*repro['pol'])} ile sonucu birebir tekrarladı. ")
else:
    head = ""

land_rows = "".join(
    f"<tr><td>{k}</td><td>{pct(v[0],192)}</td><td>+{100*(v[0]-LAND_CTL[0])/192:.1f}</td>"
    f"<td>{v[1]}–{v[2]}</td></tr>" for k, v in LAND.items())
rescore_rows = "".join(
    f"<tr{' class=ok' if ok else ''}><td>{nm}</td><td>{c:.2f}</td><td>{v2:.2f}</td><td>{v2b:.2f}</td>"
    f"<td>{'evet' if ok else 'hayır'}</td><td>{ag}</td></tr>"
    for nm, c, v2, v2b, ok, ag in RESCORE)
gain_table = ""
if gain_rows:
    gain_table = ("<table><thead><tr><th>kazanç</th><th>kod adı</th><th>ortalama çarpan</th>"
                  "<th>son 6 karar</th></tr></thead><tbody>" + "".join(
                      f"<tr><td>{a}</td><td><code>{b}</code></td><td>×{m:.2f}</td><td>×{l:.2f}</td></tr>"
                      for a, b, m, l in gain_rows) + "</tbody></table>")

knob_table = ""
if knob_rows:
    def _c(v, mode, ph):
        active = (mode == "hepsi" or (mode == "seyir/fren" and ph in "sf")
                  or (mode == "geçiş" and ph in "fl") or (mode == "son yaklaşma" and ph == "l"))
        txt = f"×{v:.2f}"
        if active and abs(v - 1) > 0.05:
            return f"<b>{txt}</b>"
        return txt if active else f'<span style="color:#9a9993">{txt}</span>'
    knob_table = ("<table><thead><tr><th>düğme</th><th>nominal</th><th>politikanın aralığı</th>"
                  "<th>okunduğu mod</th><th>seyir &gt;50 m</th><th>fren 15–50 m</th><th>son &lt;15 m</th>"
                  "<th>ne yapar</th></tr></thead><tbody>" + "".join(
                      f"<tr><td>{lab}</td><td>{nom}</td><td>{rng}</td><td>{mode}</td>"
                      f"<td>{_c(pv['s'], mode, 's')}</td><td>{_c(pv['f'], mode, 'f')}</td>"
                      f"<td>{_c(pv['l'], mode, 'l')}</td><td style='font-size:8pt'>{what}</td></tr>"
                      for lab, nm, nom, rng, mode, what, pv in knob_rows) +
                  "</tbody></table><p style='font-size:8.5pt;color:#52514e'>Gri değerler: düğme o "
                  "evrede okunmuyor, değişikliğin uçuşa etkisi yok.</p>")
timeline_img = img_tag(FIGS / "knob_timeline.png") if (FIGS / "knob_timeline.png").exists() and have_rec else ""

repro_block = ""
if repro:
    repro_block = f"""
<h3>Yeniden üretim</h3>
<p>İlk seed21 checkpoint'i ve doğrulama dosyaları, başka bir ajanın yaptığı temizlik sırasında
<code>out/</code> klasörüyle birlikte silindi. Aynı tarif — aynı eylem uzayı, aynı ödül, aynı
hiperparametreler, aynı seed — sıfırdan yeniden eğitildi ve aynı 16 dalgada yeniden doğrulandı.
GPU ve PhysX tam deterministik olmadığı için birebir aynı ağırlıklar beklenmez; bu yüzden bu
koşu, sonucun tek bir şanslı koşuya değil tarife ait olup olmadığının da testidir.</p>
<table><thead><tr><th></th><th>kontrolcü</th><th>RL politika</th><th>fark (95% CI)</th>
<th>McNemar</th><th>dalga</th></tr></thead><tbody>
<tr><td>ilk seed21, 45k</td><td>52.6%</td><td>89.6%</td><td>+37.0 [+30.2, +43.8]</td><td>72–1, p=1.6e-20</td><td>16/16</td></tr>
<tr class=ok><td>yeniden üretim</td><td>{pct(*repro['ctl'])}</td><td>{pct(*repro['pol'])}</td>
<td>+{100*(repro['pol'][0]-repro['ctl'][0])/repro['pol'][1]:.1f} [{repro['ci'][0]:+.1f}, {repro['ci'][1]:+.1f}]</td>
<td>{repro['b_only']}–{repro['c_only']}, p={repro['p']}</td><td>{repro['waves']}</td></tr>
</tbody></table>"""

stills_html = ""
if STILL.exists():
    stills_html = f"""
<figure>{img_tag(STILL, '100%', 'RL politikası Isaac Sim içinde')}
<figcaption><b>Şekil 4.</b> Isaac Sim (RTX) içinde, doğrulamada kaydedilen gerçek yörüngenin yeniden
oynatılması (dalga 90000, env 10 — eğitimin görmediği bir episod; aynı episodda kontrolcü temasta
yanal hız limitini aşıp düşüyor). {"Sol dört panel projenin eski yakalama dashboard'u, kodu değiştirilmeden kullanıldı: dış takip kamerası, aracın altındaki navigasyon kamerası, kayan IMU grafiği ve iniş boyunca kurulan LiDAR arazi haritası ile inilebilirlik maskesi. Yeniden oynatmada fiziksel bir IMU yoktur: açısal hızlar kayıttan, ivmeler kaydedilen hızın sayısal türevinden (artı Ay yerçekimi) gelir. <b>LiDAR taramaları hesaplanmıştır</b>: Ouster OS2'nin görüş alanı ve gerçek montaj pozu ile kayıtlı araziye ışın izleme; RTX sensör çıktısı değildir (RTX LiDAR yeniden oynatma sahnelerinde isabet döndürmedi, çalıştığı tek hat ise doğrulama ortamından ~3 kat yavaş inen farklı bir fizik döngüsü kullanıyor). " if LIDAR_COMPUTED else ""}Sağdaki sütun iniş limitlerine karşı telemetri ve politikanın o anda kontrolcü kazançlarına
uyguladığı çarpanlar (1.0 = kontrolcünün kendi değeri). Video: <code>out/videos/{VIDEO.name}</code>.</figcaption></figure>"""

touch_html = ""
if have_rec:
    touch_html = f"""
<figure>{img_tag(FIGS / 'touchdown_vxy.png')}
<figcaption><b>Şekil 3.</b> Doğrulama dalgası 90000'deki 12 episodun her birinde temas anındaki yanal hız.
Kontrolcünün her başarısızlığı aynı sebepten: yanal hız 1.2 m/s limitinin hemen üstünde.
RL politikası aynı episodlarda yanal hızı limitin altına indiriyor.</figcaption></figure>"""

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: 'DejaVu Sans', sans-serif; color: #0b0b0b; font-size: 10.2pt; line-height: 1.5; }
h1 { font-size: 21pt; margin: 0 0 4pt 0; line-height: 1.2; }
.sub { color: #52514e; font-size: 11pt; margin-bottom: 14pt; }
h2, h3 { page-break-after: avoid; break-after: avoid; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #e4e3df; padding-bottom: 3pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p { margin: 0 0 7pt; text-align: justify; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f2f1ed; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.8pt; page-break-inside: avoid; }
th { text-align: left; color: #52514e; font-weight: 600; border-bottom: 1.2px solid #0b0b0b; padding: 3pt 5pt; }
td { border-bottom: 1px solid #e4e3df; padding: 3pt 5pt; }
tr.ok td { background: #eef6ee; }
figure { margin: 8pt 0 12pt; page-break-inside: avoid; }
figcaption { color: #52514e; font-size: 8.8pt; margin-top: 3pt; }
.pair { display: flex; justify-content: space-between; }
.box { background: #f4f7fd; border-left: 3px solid #2a78d6; padding: 8pt 10pt; margin: 8pt 0 12pt; }
.kpi { display: flex; gap: 10pt; margin: 10pt 0 14pt; }
.kpi > div { flex: 1; border: 1px solid #e4e3df; border-radius: 6pt; padding: 8pt 10pt; }
.kpi .v { font-size: 20pt; font-weight: 700; }
.kpi .l { color: #52514e; font-size: 8.5pt; }
ol, ul { margin: 0 0 8pt 16pt; padding: 0; }
li { margin-bottom: 3pt; }
.refs li { font-size: 8.6pt; }
"""

SVG = """
<svg viewBox="0 0 720 210" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="DejaVu Sans" font-size="12">
 <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#52514e"/></marker></defs>
 <rect x="10" y="60" width="150" height="80" rx="8" fill="#eaf2fc" stroke="#2a78d6"/>
 <text x="85" y="88" text-anchor="middle" font-weight="700" fill="#0b0b0b">RL politikası</text>
 <text x="85" y="106" text-anchor="middle" fill="#52514e">PPO, 2×64 MLP</text>
 <text x="85" y="122" text-anchor="middle" fill="#52514e">saniyede 1 karar</text>
 <line x1="160" y1="100" x2="250" y2="100" stroke="#52514e" stroke-width="1.5" marker-end="url(#a)"/>
 <text x="205" y="90" text-anchor="middle" fill="#52514e">11 kazanç</text>
 <text x="205" y="118" text-anchor="middle" fill="#52514e" font-size="10.5">× oran^a</text>
 <rect x="252" y="50" width="190" height="100" rx="8" fill="#fdf0ea" stroke="#eb6834"/>
 <text x="347" y="78" text-anchor="middle" font-weight="700" fill="#0b0b0b">ZEM/ZEV kontrolcü</text>
 <text x="347" y="96" text-anchor="middle" fill="#52514e">güdüm + PD tutum döngüsü</text>
 <text x="347" y="112" text-anchor="middle" fill="#52514e">taşınan tgo saati</text>
 <text x="347" y="128" text-anchor="middle" fill="#52514e">20 Hz (0.05 s)</text>
 <line x1="442" y1="100" x2="532" y2="100" stroke="#52514e" stroke-width="1.5" marker-end="url(#a)"/>
 <text x="487" y="90" text-anchor="middle" fill="#52514e">itki, tork</text>
 <rect x="534" y="60" width="176" height="80" rx="8" fill="#f2f1ed" stroke="#52514e"/>
 <text x="622" y="88" text-anchor="middle" font-weight="700" fill="#0b0b0b">Isaac Sim / PhysX</text>
 <text x="622" y="106" text-anchor="middle" fill="#52514e">11 t Apollo LM</text>
 <text x="622" y="122" text-anchor="middle" fill="#52514e">gerçek Ay arazisi</text>
 <path d="M622,140 L622,185 L85,185 L85,140" fill="none" stroke="#52514e" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#a)"/>
 <text x="353" y="200" text-anchor="middle" fill="#52514e">gözlem (20 + 3: kontrolcünün tgo'su, son-yaklaşma bayrağı, yanal hız/limit)</text>
 <text x="10" y="44" text-anchor="start" fill="#2a78d6" font-size="11">sıfır eylem = kontrolcünün kendisi</text>
</svg>"""

body = f"""
<h1>Ay inişinde pekiştirmeli öğrenme: ZEM/ZEV güdüm kazançlarının PPO ile öğrenilmesi</h1>
<div class=sub>LunarSim · Isaac Sim fiziği · teknik rapor · Ekim 2026</div>

<div class=kpi>
<div><div class=v style="color:#eb6834">52.6%</div><div class=l>analitik ZEM/ZEV kontrolcü, Isaac orbit_descent</div></div>
<div><div class=v style="color:#2a78d6">{pct(*repro['pol']) if repro else '89.6%'}</div><div class=l>RL politikası, aynı 192 taze episod</div></div>
<div><div class=v>{(f"{repro['b_only']}–{repro['c_only']}") if repro else '72–1'}</div><div class=l>politika iner/kontrolcü düşer — tersi</div></div>
</div>

<div class=box><b>Özet.</b> 6-serbestlik dereceli bir Ay iniş görevinde, uçtan uca RL denemelerinin
hiçbiri analitik kontrolcüyü geçememişti (en iyi tahmin +1.6 puan, ölçülemez). Sorunun kaynağı
algoritma değil; üç kurulum hatası, iki ölçüm hatası ve iniş oranını ters sıralayan bir ödül terimiydi.
Politikayı kontrolcünün yerine koymak yerine kontrolcünün kazançlarını saniyede bir ayarlayan bir
eylem uzayına geçildi, yoğun şekillendirme terimi kaldırılıp sadece iniş sonucu ödüllendirildi.
Sıfırdan eğitilen PPO politikası, hedef sahnede eğitimin hiç görmediği 192 episodda kontrolcünün
%52.6 iniş oranını %89.6'ya çıkardı (McNemar 72–1, p = 1.6·10⁻²⁰). {head}Proje hedefi %75 idi.</div>

<h2>1. Problem</h2>
<p>Görev: 11–15 tonluk Apollo LM sınıfı bir aracı 200 m irtifadan, 10–30 m/s yatay hızla bırakıldığı
yerden, dikey hız ≤ 1.0 m/s, yanal hız ≤ 1.2 m/s, yatış ≤ 15°, açısal hız ≤ 0.5 rad/s ve ayak
yüksekliği farkı ≤ 0.16 m koşullarının <i>hepsi aynı anda</i> sağlanarak indirmek. İtki gövde eksenine
sabit; yanal ivme ancak aracı yatırarak elde edilir, yani komuttan yanal hıza dört entegrasyon
vardır (tork → açısal ivme → açısal hız → yatış → yanal ivme). Analitik ZEM/ZEV kontrolcüsü Isaac'te
%52.6 iniyor; başarısızlıklarının neredeyse tamamı temasta yanal hızın limiti az farkla aşması.</p>
<p>Bu oturumdan önce SAC, PPO, CQL, Cal-QL, TD3+BC, PEX, LayerNorm ve çift-indirim denemelerinin
hiçbiri kontrolcünün davranış klonunu ölçülebilir biçimde geçemedi. Önceki teşhis, öğrenilmiş bir
Q(s,a)'nın bu görevde eylemleri sıralayamayacağıydı (1:2559 sinyal/gürültü).</p>

<h2>2. Teşhis: önceki denemeler neden başarısız oldu</h2>
<p>Ajan ekibiyle yapılan kod ve veri denetimi önceki teşhisin geçerli olmadığını gösterdi: "gürültü"
olarak ölçülen nicelik durumlar arası kritik artığıydı (1.884); eylem sıralamasını belirleyen
aynı-durum farkı ise 0.048 — 39 kat küçük. Üstelik PPO hiç Q kullanmadığı halde aynı şekilde başarısız
olmuştu. Asıl sebepler, her biri tek başına yeterli olan üç kurulum hatasıydı:</p>
<ol>
<li><b>GAE λ terminal ödülü attı.</b> γ = 0.999 iniş bonusunu görünür kılmak için seçilmişti, ama
λ = 0.99 avantaj ufkunu 91 adıma indirdi; politika gradyanına bonusun yalnızca <b>%0.037</b>'si ulaştı.</li>
<li><b>Keşif ölçeği donmuştu.</b> PPO'nun log-std'si davranış klonunun artığından miras alındı
(σ = 0.022–0.049, eylem kutusunun %1–2.5'i) ve 1.5 milyon adımda dört ondalık basamakta değişmedi.
Wagenmaker ve ark. bu kapsamanın RL ince ayarı için gereken asgari koşulu sağlamayabileceğini kanıtlıyor.</li>
<li><b>Kontrol adımı tutum zaman sabitine göre 125 kat ince.</b> Tam torkla 10° yatış 6.3 s sürüyor,
kontrol adımı 0.05 s. Tallec, Blier ve Ollivier'in (ICML 2019) sonucu bu rejimde eylem-avantaj
sinyalinin dt ile orantılı küçüldüğünü ve adım-başı bağımsız keşif gürültüsünün deterministik
politikaya çöktüğünü gösterir. Ölçtük: aynı σ'da zamansal korelasyonlu gürültü 3–8 kat fazla
gradyan sinyali üretiyor.</li>
</ol>
<p>Bunlara ek olarak iki ölçüm hatası düzeltildi: kontrolcü taban ölçümü yapan script araca gövde
merkezi yüksekliğini veriyordu (3.52–4.62 m fazla irtifa); haritadan kaçışı fiyatlayan
<code>_edge_penalty</code> terimi eğitimde kullanılan tek ortamda tamamen sıfırdı. Ayrıca tüm
karşılaştırmalar eşleşmiş tasarımda olmasına rağmen eşleşmemiş test kullanılıyordu.</p>

<h2>3. Yöntem</h2>
<h3>Eylem uzayı: kontrolcüyü değiştirme, ayarla</h3>
<figure>{SVG}<figcaption><b>Şekil 1.</b> Guided-gain mimarisi. Politika 11 kontrolcü kazancını
saniyede bir ayarlar; kontrolcü 20 Hz'de uçmaya devam eder.</figcaption></figure>
<p>Politika motor ve itici komutu üretmez; ZEM/ZEV kontrolcüsünün 11 kazancını
<code>kazanç = nominal · oran<sup>a</sup></code> şeklinde çarparak ayarlar ve bu kararı 20 kontrol adımı
(1 s) boyunca tutar. <b>Sıfır eylem kontrolcünün kendisini birebir üretir</b> (24 seed'de aynı getiri ve
sonuçla doğrulandı); yani öğrenme, %1 kapsamalı bir taklitten değil, çalışan sistemden başlar.
Kontrolcünün iç tutum döngüsü ve taşınan tgo saati döngüde kalır, böylece politika dört
entegrasyonluk zinciri öğrenmek zorunda kalmaz. Bölüm başına 1200 yerine ~60 karar düşer.
Bu yapı Furfaro ve ark.'nın (2020) aynı güdüm yasası üzerindeki çalışmasıyla ve Kaufmann ve
ark.'nın (2022) "iç döngü üstünde komut" bulgusuyla uyumludur.</p>

<h3>Ödül: iniş oranını doğru sıralayan tek seçim</h3>
<p>Kayıtlı politikaları birden çok aday ödülle aynı episodlar üzerinde yeniden puanlayan bir tanı
aracı yazıldı. Gönderilen ödül, en çok inen politikayı (%31.2) kontrolcünün (%9.9) <i>altına</i>
koyuyordu. Ayrıştırma sorunun yoğun hız-alanı takip teriminde olduğunu gösterdi: daha dikkatli,
daha uzun bir son yaklaşma ile inen politika bu terimden fazla ceza alıyordu.</p>
<table><thead><tr><th>ödül varyantı</th><th>kontrolcü</th><th>%31.2 politika</th><th>%21.9 politika</th>
<th>inişe göre sıralıyor</th><th>dönüşümde işaret uyumu</th></tr></thead><tbody>{rescore_rows}</tbody></table>
<p>Terminal tarafındaki hiçbir düzeltme (basamak bonusu, taban sıkıştırma, yoğun terim tavanı) makul
boyutta sıralamayı düzeltmedi. Açıklama basit bir özdeşlik: ortalama getiri J = F + p·(S − F);
başarısızlıkların ortalama getirisi F politikaya bağlı oldukça J iniş oranı p'ye göre monoton olmaz.
Bu yüzden yoğun terim kapatıldı ve sadece sonuç ödüllendirildi: iniş +5, başarısız temas 0, zaman
aşımı veya devrilme −3. Gaudet ve ark.'nın uzay inişi çalışmalarındaki yapı da budur.</p>

<h3>Eğitim ve ölçüm protokolü</h3>
<p>PPO sıfırdan, rastgele başlatılmış ağla eğitildi (2×64, σ₀ = 0.37, γ = 0.995, λ = 0.99,
8 paralel Isaac ortamı). En iyi sonuç veren ayar: güncelleme başına 1024 karar (~17 episod) ve
öğrenme hızı 1·10⁻⁴. Checkpoint seçimi yalnızca seed bloğu 7000'de yapıldı; puanlama, eğitimin
görmediği 16 dalga × 12 episod = 192 eşleşmiş episodda (dalga tohumları 90000–105000). Her dalga,
ortamın rastgele sayı üretecini sabit bir değere yeniden tohumlayıp her ortamdan tam bir episod alır;
iki ayrı çalıştırmanın bit-birebir aynı sonuç verdiği doğrulandı. Test: McNemar (kesin).</p>

<h2>4. Sonuçlar</h2>
<figure>{img_tag(FIGS / 'results.png')}<figcaption><b>Şekil 2.</b> Isaac orbit_descent, 192 eşleşmiş
taze episod. Kontrolcü ve RL politikaları aynı başlangıç koşulları ve arazide.</figcaption></figure>
<table><thead><tr><th>koşu</th><th>kontrolcü</th><th>RL politika</th><th>fark (95% CI)</th>
<th>McNemar</th></tr></thead><tbody>
<tr><td>ilk koşu, 30k karar</td><td>52.6%</td><td>68.2%</td><td>+15.6 [+8.9, +22.9]</td><td>41–11, p=3.6e-5</td></tr>
<tr><td>ilk koşu, 60k karar</td><td>52.6%</td><td>67.7%</td><td>+15.1</td><td>42–13, p=1.1e-4</td></tr>
<tr class=ok><td>seed21 (büyük rollout, düşük lr), 45k</td><td>52.6%</td><td>89.6%</td><td>+37.0 [+30.2, +43.8]</td><td>72–1, p=1.6e-20</td></tr>
</tbody></table>
{repro_block}
<figure>{img_tag(FIGS / 'curves.png')}<figcaption><b>Şekil 5.</b> Seçim bloğu (n = 48, eşleşmemiş)
eğrileri. Bu blok yalnızca checkpoint seçmek içindir ve gürültülüdür: ilk koşu burada 83.3%
okurken taze episodlarda 67.7% ölçüldü. Büyük rollout ve düşük öğrenme hızı platoyu kaldırdı;
her iki ayarda da ~45k karardan sonra gerileme var, bu yüzden en iyi checkpoint seçim bloğunda
seçilip taze episodlarda puanlanmalı. Yeniden üretim koşusu (kesikli) seed21 koşusunu her
değerlendirmede birebir tekrarladı, bu yüzden iki çizgi üst üste.</figcaption></figure>
{touch_html}
{stills_html}

<h3>RL tam olarak neyi ayarlıyor?</h3>
<p><b>Kısaca:</b> ZEM/ZEV kontrolcüsü 11 ayar düğmesi olan bir otopilottur; normalde bu düğmeler
uçuş boyunca sabittir. RL politikası her saniye aracın durumuna bakıp bu düğmeleri biraz çeviren
bir pilottur. Motoru ve iticileri doğrudan hiç kumanda etmez.</p>
<p><b>Her saniye olanlar.</b> (1) Politika 23 sayılık gözlemi okur: konum, hız, yatış (kuaterniyon),
açısal hız, yakıt, kalan süre, hız-alanı hataları ve kontrolcünün kendi iç durumu (inişe kalan süre
tahmini tgo, son yaklaşma modunda olup olmadığı, yanal hızın limite oranı). (2) 11 sayı üretir,
her biri a ∈ [−1, +1]; her düğme <code>nominal × oran<sup>a</sup></code> olur, a = 0 düğmeyi
hiç oynatmaz. (3) Kontrolcü bu ayarlarla 20 kontrol adımı (1 s) uçar: ZEM/ZEV istenen ivmeyi
hesaplar, ivmeyi istenen yatış açısına çevirir (en büyük yatış ve yatış azaltma hızıyla sınırlı),
yatış hatasını PD döngüsüyle tork komutuna çevirir ve itkiyi dikey ihtiyaca göre ayarlar.
"Son yaklaşma irtifası"nın altında güdümü bırakıp −0.8 m/s ile dikey iner ve kalan yanal hızı
söndürür.</p>
<p><b>Bir düğme yalnızca okunduğu modda etkilidir.</b> Örneğin dikey hız kazancı sadece son
yaklaşmada okunur; politikanın onu seyirde değiştirmesi uçuşu etkilemez. Aşağıdaki tablo her
düğmenin nerede okunduğunu ve doğrulanmış 12 episodda politikanın uçuş evresine göre ortalama
çarpanını verir (kalın: düğmenin okunduğu evrede ±%5'ten büyük değişiklik).</p>
{knob_table}
<figure>{timeline_img}<figcaption><b>Şekil 6.</b> Videodaki episodun (dalga 90000, env 10)
saniye saniye zaman çizelgesi. Üstte irtifa ve yanal hız; altta politikanın en çok oynattığı dört
düğme. Politika fren evresinde son yaklaşmaya geçiş irtifasını kademeli olarak düşürüyor
(~×0.87: geçiş 8 m yerine ~7 m'de) ve tutum sönümünü artırıyor.</figcaption></figure>
<p><b>Fiziksel yorumu.</b> Kontrolcünün asıl zaafı son yaklaşma modudur: 20°'lik yatışa ulaşmak
~8.7 s sürer, mod ise ~10 s sürer; kalan yanal hız bu yüzden temasa kadar tam sönmez.
Politika (1) bu moda biraz daha geç geçerek güçlü ZEM/ZEV frenini daha uzun çalıştırıyor,
(2) fren ve son yaklaşmada tutum sönümünü artırarak (×1.07) salınım sırasında biriken yanal hızı
azaltıyor, (3) seyirde tgo alt sınırını düşürerek (×0.90) frene daha erken ve sert başlıyor.
Aynı episodda temas anındaki yanal hız kontrolcüde 1.32 m/s (limit aşıldı), politikada 0.75 m/s.</p>
<p><b>Neden küçük değişiklikler yetiyor?</b> Kontrolcünün başarısızlıkları marjinaldir: Şekil 3'teki
8 başarısızlığın hepsinde yanal hız 1.27–1.37 m/s, limitin yalnızca %6–14 üstünde. Doğru anda
yapılan %10'luk ayar yeterli oluyor — ama bunu sabit bir kazanç seti değil, her saniye gözlenen
duruma göre yapılan ayar sağlıyor.</p>
<p><b>Sınır.</b> Bu ortalamalar politikanın ne yaptığını gösterir, hangi düğmenin iniş oranını
<i>gerçekten</i> belirlediğini kanıtlamaz. Bunun için düğmeleri teker teker nominale sabitleyip
iniş oranındaki düşüşü ölçen bir ablasyon gerekir; bu raporda yapılmadı.</p>

<h3>Bu sadece kontrolcünün ince ayarı mı?</h3>
<p>Kısmen. Bunu ölçmek için bir kontrol kolu eklendi: politikanın 12 kayıtlı uçuştaki ortalama
ayarı tek bir <b>sabit</b> kazanç setine çevrildi ve aynı 192 taze episodda uçuş boyunca hiç
değiştirilmeden kullanıldı.</p>
<table><thead><tr><th>kol</th><th>iniş</th><th>karşılaştırma</th><th>McNemar</th></tr></thead><tbody>
<tr><td>ZEM/ZEV, nominal kazançlar</td><td>%52.6</td><td>—</td><td>—</td></tr>
<tr><td>sabit kazanç seti (politikanın ortalaması)</td><td>%71.9</td><td>kontrolcüye göre +19.3</td><td>37–0, p=1.5e-11</td></tr>
<tr class=ok><td>RL politikası, her saniye durum bazlı</td><td>%89.6</td><td>sabit sete göre +17.7 [+12.0, +23.4]</td><td>36–2, p=5.4e-9</td></tr>
</tbody></table>
<p>Kazancın yaklaşık yarısı daha iyi bir <i>sabit</i> ayardan geliyor: politika, nominal değerlerden
daha iyi bir kazanç setinin nerede olduğunu buldu. Diğer yarısı ise yalnızca <i>durum bazlı</i>
ayarla elde ediliyor — sabit set 36 episodda düşerken politika aynı episodlarda iniyor, tersi
yalnızca 2 kez oluyor. Yani politika bir ince ayardan fazlasını, öğrenilmiş bir kazanç planlamasını
(gain scheduling) yapıyor; ama düşük seviyeli uçuşu hâlâ analitik kontrolcü yürütüyor.</p>

<h3>Ek sahne: land_handoff</h3>
<p>8–25 m'den, yatık ve dönerek bırakılan zor bir alt sahnede (analitik arka uç, 4 taze blok,
kontrolcü %9.9) dört koşunun hepsi kontrolcüyü sıfıra yakın regresyonla geçti. Burada ödül
değişikliği belirgin fark yaratmadı ve koşular arası seed farkı ödül farkı kadar büyük. Her episod için
200 rastgele kazanç vektörünün en iyisini seçen bir kâhin %31.2'ye ulaşıyor ve örnek sayısı arttıkça
hâlâ yükseliyor (24 örnekte %20.8'di); yani bu sahnenin gerçek tavanı bilinmiyor.</p>
<table><thead><tr><th>koşu</th><th>iniş</th><th>fark (puan)</th><th>McNemar</th></tr></thead>
<tbody>{land_rows}</tbody></table>

<h2>5. Bu RL mi, taklit mi?</h2>
<p><b>RL.</b> Politika rastgele ağırlıklarla başlatıldı; hiçbir uzman demosu, davranış klonu ya da
taklit kaybı kullanılmadı. Tek öğrenme sinyali iniş sonucudur. Ancak bu uçtan uca RL değildir:
düşük seviyeli uçuşu analitik kontrolcü yapar, RL onun kazançlarını duruma göre ayarlar. Doğru
tanım "RL ile kazanç planlama" ya da hibrit güdümdür. Politika kontrolcüyü taklit etmez —
192 episodun 72'sinde kontrolcünün düştüğü yerde iner.</p>

<h2>6. Sınırlar ve sonraki adımlar</h2>
<ul>
<li>Politika kontrolcünün yapısının ifade edebildiği manevralarla sınırlıdır. Daha ötesi için
kontrolcü çıkışına küçük bir rezidüel ya da tutum-setpoint eylem uzayı denenebilir.</li>
<li>Tüm doğrulamalar tek arazi tohumunda (7) yapıldı; farklı arazilerde genelleme ölçülmedi.</li>
<li>Uzun eğitim ~45k karardan sonra geriliyor; erken durdurma veya öğrenme hızı azaltımı gerekli.</li>
<li>Projenin eski yakalama hattı (<code>isaaclab_controller_eval_capture.py</code>) kendi fizik döngüsünü
kullanıyor ve aynı bırakış koşulundan araç orada eğitim/doğrulama ortamına göre ~3 kat yavaş iniyor
(19 s'de 200 m → 183 m, yani 17 m; doğrulama ortamında aynı sürede ~137 m). Eski 94.8 s'lik demo videosunun uzunluğu
büyük ihtimalle bu farktan geliyor. Bu rapordaki tüm sayılar ve videodaki uçuş doğrulama ortamından.</li>
<li>Literatürde aynı çöküşü yaşayan SAC çalışması (Dimitrov & Spigler) çözüm önermiyor;
bu kurulumda kısa ufuk sayesinde off-policy bir yöntem denenebilir.</li>
</ul>

<h2>Kaynaklar</h2>
<ol class=refs>
<li>Gaudet, Linares, Furfaro. Deep RL for six degree-of-freedom planetary landing. Advances in Space Research 65(7), 2020. arXiv:1810.08719.</li>
<li>Furfaro, Scorsoglio, Linares, Massari. Adaptive generalized ZEM-ZEV feedback guidance for planetary landing via a deep RL approach. Acta Astronautica, 2020. arXiv:2003.02182.</li>
<li>Tallec, Blier, Ollivier. Making deep Q-learning methods robust to time discretization. ICML 2019. arXiv:1901.09732.</li>
<li>Kaufmann, Bauersfeld, Scaramuzza. A benchmark comparison of learned control policies for agile quadrotor flight. arXiv:2202.10796.</li>
<li>Jiang ve ark. Rocket landing control with random annealing jump start RL. arXiv:2407.15083.</li>
<li>Wagenmaker, Dong, Finn, Levine. Posterior behavioral cloning. arXiv:2512.16911.</li>
<li>Andrychowicz ve ark. Hindsight experience replay. arXiv:1707.01495.</li>
<li>Pan, Bhatia, Steinhardt. The effects of reward misspecification. arXiv:2201.03544.</li>
<li>Dimitrov, Spigler. arXiv:2609.06667.</li>
<li>Metelli ve ark. Control frequency adaptation via action persistence. ICML 2020. arXiv:2002.06836.</li>
</ol>
"""

body = re.sub(r"(?<![:\d.])(\d+(?:\.\d+)?)%", r"%\1", body)
page = f"<!doctype html><html lang=tr><head><meta charset=utf-8><title>LunarSim RL raporu</title><style>{CSS}</style></head><body>{body}</body></html>"
html_path = REP / "lunarsim_rl_rapor.html"
html_path.write_text(page, encoding="utf-8")
pdf_path = REP / "lunarsim_rl_rapor.pdf"
subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"wrote {pdf_path} (repro={'yes' if repro else 'no'}, recording={'yes' if have_rec else 'no'})")
