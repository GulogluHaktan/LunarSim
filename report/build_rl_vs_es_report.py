"""RL (guided-gain PPO) vs ES (controller-free) comparison report, Turkish.
Reads existing result files only; never runs a simulation. Numbers come from:
  ~/lunarsim_artifacts/gg_isaac_orbit_v4_seed21_repro_validation.json  (PPO, Isaac, 192 paired)
  out/evo/isaac_orbit_val192.json                                      (ES + controller, Isaac, 192 paired)
  out/gg_isaac_orbit_v4_seed21_repro{.log,/curve.csv}                  (PPO training)
  out/evo/an_n1, out/evo/an_ladder_*/curve.csv, out/evo/ladder.log     (ES training, analytic)
Run: ~/isaac-env/bin/python report/build_rl_vs_es_report.py
"""
from __future__ import annotations
import base64, csv, json, math, os, re, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVO = ROOT / "out" / "evo"; ART = Path.home() / "lunarsim_artifacts"
FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CTL, PPO, ES, GOOD, CRIT, WARN = "#eb6834", "#7a4fd1", "#2a78d6", "#0ca30c", "#d03b3b", "#c98a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": "white", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6})

# ------------------------------------------------------------------ data
ppo = json.load(open(ART / "gg_isaac_orbit_v4_seed21_repro_validation.json"))
es = json.load(open(EVO / "isaac_orbit_val192.json"))
c_ppo = np.array(ppo["baseline_landed"], bool); c_es = np.array(es["controller"]["landed"], bool)
P = np.array(ppo["candidate_landed"], bool); E = np.array(es["es_policy"]["landed"], bool)
PAIRED = bool((c_ppo == c_es).all()) and ppo["wave_seeds"] == es["seeds"] and ppo["n_envs"] == 12
C = c_es; N = len(C); NW = N // 12
assert N == 192 and len(P) == N and len(E) == N

def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h

def mcn(x, y):  # exact two-sided; returns (y-only, x-only, p)
    b = int((y & ~x).sum()); c = int((x & ~y).sum()); n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def boot(a, b, B=20000, seed=0):  # wave-block bootstrap of rate(b)-rate(a), in points
    rng = np.random.default_rng(seed)
    A = a.reshape(NW, 12).sum(1); Bv = b.reshape(NW, 12).sum(1)
    idx = rng.integers(0, NW, (B, NW))
    d = 100 * (Bv[idx].sum(1) - A[idx].sum(1)) / (12 * NW)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

kC, kP, kE = int(C.sum()), int(P.sum()), int(E.sum())
ci = {n: wil(k, N) for n, k in (("C", kC), ("P", kP), ("E", kE))}
mP = mcn(C, P); mE = mcn(C, E); mEP = mcn(P, E)       # (policy-only, ctrl-only, p) ; (ES-only, PPO-only, p)
bP, bE, bEP = boot(C, P), boot(C, E), boot(P, E)
both = int((P & E).sum()); only_p = int((P & ~E).sum()); only_e = int((E & ~P).sum()); neither = int((~P & ~E).sum())
wC, wP, wE = (v.reshape(NW, 12).sum(1) for v in (C, P, E))
waves_e_ge_p = int((wE >= wP).sum()); waves_e_gt_p = int((wE > wP).sum()); waves_p_gt_e = int((wP > wE).sum())
# paired half-width at the observed discordance (normal approx. on difference of paired proportions)
disc = (only_e + only_p) / N
half_width_pts = 100 * 1.96 * math.sqrt(disc / N)

# PPO training: selection blocks
ppo_curve = list(csv.DictReader(open(ROOT / "out" / "gg_isaac_orbit_v4_seed21_repro" / "curve.csv")))
ppo_dec = np.array([float(r["decisions"]) for r in ppo_curve]); ppo_sel = np.array([float(r["landed"]) / float(r["episodes"]) for r in ppo_curve])
ppo_log = (ROOT / "out" / "gg_isaac_orbit_v4_seed21_repro.log").read_text(errors="ignore")
m = re.search(r"\[done\] best on the SELECTION block: (\d+)/(\d+) vs controller (\d+)/(\d+)", ppo_log)
ppo_sel_best = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
fps = [int(x) for x in re.findall(r"\|\s+fps\s+\|\s+(\d+)", ppo_log)]
chunk_s = max(int(x) for x in re.findall(r"time_elapsed\s+\|\s+(\d+)", ppo_log))
tm = re.search(r"\[(\d+\.\d+)s\] Simulation App Shutting Down", ppo_log); ppo_wall_s = float(tm.group(1)) if tm else float("nan")
args_line = re.search(r"Passing the following args to the base kit application:\s+(\[.*\])", ppo_log)
ppo_args = eval(args_line.group(1)) if args_line else []
ppo_envs = int(ppo_args[ppo_args.index("--n-envs") + 1]) if "--n-envs" in ppo_args else None
ppo_budget = int(ppo_args[ppo_args.index("--steps") + 1]) if "--steps" in ppo_args else None
best_ppo_dec = 45000  # snap_45000 is the selected snapshot (43/48 on the selection block)
assert ppo_dec[np.argmax(ppo_sel)] == best_ppo_dec

# ES training: ladder
def curve(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda x: float(x) if x not in ("", "nan", None) else np.nan  # noqa: E731
    return {k: np.array([f(r[k]) for r in rows]) for k in ("gen", "episodes", "mean_fit", "mean_land", "best_land", "sel_land", "sel_fit")}
stages = [("ramp_35m", EVO / "an_n1" / "curve.csv")] + [(s, EVO / f"an_ladder_{s}" / "curve.csv") for s in ("ramp_50m", "ramp_100m", "ramp_150m", "orbit_descent")]
ladder_log = (EVO / "ladder.log").read_text(errors="ignore")
land_t = [float(x) for x in re.findall(r"es_policy .*landed: .*? t=([\d.]+)s", ladder_log)]  # ramp_50m..orbit_descent
assert len(land_t) == 4, land_t
STAGE_T = {"ramp_35m": 25.0, "ramp_50m": land_t[0], "ramp_100m": land_t[1], "ramp_150m": land_t[2], "orbit_descent": land_t[3]}  # 35m: stage budget (upper bound)
es_stage = {}
for name, path in stages:
    c = curve(path); ok = ~np.isnan(c["sel_fit"])
    i = int(np.nanargmax(np.where(ok, c["sel_fit"], -1e18)))
    es_stage[name] = dict(c=c, gens=int(c["gen"][-1]) + 1, total_ep=int(c["episodes"][-1]), best_gen=int(c["gen"][i]),
                          ep_to_best=int(c["episodes"][i]), steps_per_ep=STAGE_T[name] * 20.0)
es_total_ep = sum(v["ep_to_best"] for v in es_stage.values())
es_total_steps = sum(v["ep_to_best"] * v["steps_per_ep"] for v in es_stage.values())
es_orbit_steps = es_stage["orbit_descent"]["ep_to_best"] * es_stage["orbit_descent"]["steps_per_ep"]
ppo_steps = best_ppo_dec * 20.0
an_n1_log = (EVO / "an_n1.log").read_text(errors="ignore")
t_an = [int(x) for x in re.findall(r"t=(\d+)s", an_n1_log)]
mt = lambda p: p.stat().st_mtime  # noqa: E731
bt = lambda p: float(subprocess.check_output(["stat", "-c", "%W", str(p)]).decode().strip())  # noqa: E731  (creation time, GNU stat)
t_ladder_start = bt(EVO / "ladder.log")                      # ladder.sh created its log when it started
t_ladder_end = mt(EVO / "an_ladder_orbit_descent" / "best_on_selection.npz")
es_wall_ladder_min = (t_ladder_end - t_ladder_start) / 60.0  # includes the per-rung validation runs
es_wall_35_min = (mt(EVO / "an_n1" / "curve.csv") - bt(EVO / "an_n1" / "curve.csv")) / 60.0

def box(ax, x, y, w, h, text, fc="#eef4fc", ec=ES, fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            color=INK, linespacing=1.25)

def arrow(ax, p0, p1, color=INK2, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))

def save(fig, name):
    p = FIG / f"rlvses_{name}.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); return p

# ------------------------------------------------------------------ figures
def fig_flow():
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.2))
    for ax in axs: ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    ax = axs[0]
    ax.text(0.0, 3.2, "RL: guided-gain PPO — kontrolcü uçmaya devam eder, PPO kazançlarını ayarlar", fontsize=10.5, weight="bold", color=PPO)
    box(ax, 0.0, 1.2, 1.6, 1.4, "Gözlem (23)\nham 20 + 3 ek\n(t_go, final_mi,\nv_xy)", fc="#f4f3ef", ec=INK2)
    box(ax, 2.0, 1.2, 1.8, 1.4, "PPO politikası\n64×64 tanh MLP\n23 → 11 eylem\n(1 sn'de bir karar)", fc="#f1ecfb", ec=PPO)
    box(ax, 4.2, 1.2, 1.8, 1.4, "11 kazanç çarpanı\nkazanç = nominal ×\noran^a,  a∈[-1,1]\n(a = 0 → kontrolcü)", fc="#f1ecfb", ec=PPO)
    box(ax, 6.4, 1.2, 1.7, 1.4, "ZEM/ZEV kontrolcüsü\nUÇAR: gaz + 3 tork\n(20 Hz, 20 adım\nsabit kazanç)", fc="#fdf0e9", ec=CTL, bold=False)
    box(ax, 8.4, 1.2, 1.5, 1.4, "Uçuş sonucu\niniş / çarpma\n→ ödül\n(+5 / 0 / −3)", fc="#eaf6ea", ec=GOOD)
    for a, b in ((1.62, 1.98), (3.82, 4.18), (6.02, 6.38), (8.12, 8.38)): arrow(ax, (a, 1.9), (b, 1.9))
    arrow(ax, (9.1, 1.15), (3.0, 0.7), color=CRIT, rad=-0.15)
    ax.text(5.2, 0.15, "PPO (gradyan): ödülden politika ve değer ağı güncellenir; kontrolcü içeride kalır", ha="center", fontsize=8.8, color=CRIT)
    ax = axs[1]
    ax.text(0.0, 3.2, "ES: kontrolcüsüz — 84 parametreli lineer politika doğrudan motor komutu verir", fontsize=10.5, weight="bold", color=ES)
    box(ax, 0.0, 1.2, 1.6, 1.4, "Gözlem (20)\nçalışan ort./std\nile normalize\n(rehber slotlar 16-19)", fc="#f4f3ef", ec=INK2)
    box(ax, 2.0, 1.2, 1.8, 1.4, "Lineer politika\neylem = W·gözlem + b\n(84 sayı)\n(20 Hz, her adım)", fc="#eef4fc", ec=ES)
    box(ax, 4.2, 1.2, 1.8, 1.4, "4 ham komut\ngaz · pitch\nroll · yaw\n(kontrolcü YOK)", fc="#eef4fc", ec=ES)
    box(ax, 6.4, 1.2, 1.7, 1.4, "Araç dinamiği\n(analitik eğitim,\nIsaac doğrulama)", fc="#f4f3ef", ec=INK2)
    box(ax, 8.4, 1.2, 1.5, 1.4, "Uçuş sonucu\n+ çarpma şiddeti\n→ uygunluk\n(sadece terminal)", fc="#eaf6ea", ec=GOOD)
    for a, b in ((1.62, 1.98), (3.82, 4.18), (6.02, 6.38), (8.12, 8.38)): arrow(ax, (a, 1.9), (b, 1.9))
    arrow(ax, (9.1, 1.15), (3.0, 0.7), color=CRIT, rad=-0.15)
    ax.text(5.2, 0.15, "ES (gradyansız): 96 aday gürültülü θ ± σε, sonuç sıralaması → Adam ile θ güncellenir", ha="center", fontsize=8.8, color=CRIT)
    fig.tight_layout()
    return save(fig, "flow")

def fig_bars():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    items = [("ZEM/ZEV\nkontrolcüsü", kC, CTL, ci["C"]), ("RL: guided-gain\nPPO", kP, PPO, ci["P"]), ("ES\n(kontrolcüsüz)", kE, ES, ci["E"])]
    for i, (lab, k, col, (lo, hi)) in enumerate(items):
        ax.bar(i, 100 * k / N, color=col, yerr=[[100 * k / N - 100 * lo], [100 * hi - 100 * k / N]], capsize=5, ecolor=INK2, width=0.6)
        ax.text(i, 100 * hi + 1.5, f"{k}/{N}\n%{100 * k / N:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([i[0] for i in items]); ax.set_ylim(0, 118)
    ax.set_ylabel("iniş oranı (%)"); ax.set_title("Isaac orbit_descent, 192 eşleştirilmiş taze bölüm (%95 Wilson GA)", fontsize=10)
    return save(fig, "bars")

def fig_waves():
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(NW); ax = axs[0]
    ax.plot(x, wC, "o-", color=CTL, lw=1.6, label="kontrolcü"); ax.plot(x, wP, "s-", color=PPO, lw=1.8, label="PPO")
    ax.plot(x, wE, "^-", color=ES, lw=1.8, label="ES")
    ax.set_xlabel("dalga (tohum 90000 … 105000)"); ax.set_ylabel("12 bölümden iniş sayısı"); ax.set_ylim(0, 13)
    ax.set_title("Dalga bazında iniş sayısı", fontsize=10); ax.legend(fontsize=8.5, loc="lower right", ncol=3)
    ax = axs[1]; d = wE - wP
    ax.bar(x, d, color=[ES if v > 0 else PPO if v < 0 else GRID for v in d])
    ax.axhline(0, color=INK2, lw=1); ax.set_xlabel("dalga"); ax.set_ylabel("ES − PPO (iniş sayısı)")
    ax.set_title(f"ES>PPO: {waves_e_gt_p}, ES<PPO: {waves_p_gt_e}, eşit: {NW - waves_e_gt_p - waves_p_gt_e}", fontsize=10)
    return save(fig, "waves")

def fig_disc():
    fig, ax = plt.subplots(figsize=(4.6, 3.6)); ax.grid(False)
    M = np.array([[both, only_p], [only_e, neither]])
    ax.imshow(np.array([[1, 0.55], [0.55, 0.05]]), cmap="Blues", vmin=0, vmax=1.2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=20, weight="bold",
                    color="white" if (i, j) == (0, 0) else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ES indi", "ES çarptı"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["PPO indi", "PPO çarptı"])
    ax.set_title(f"192 bölüm: PPO × ES\nMcNemar (ES-only {only_e} / PPO-only {only_p}) p={mEP[2]:.2f}", fontsize=10)
    return save(fig, "disc")

def fig_learn():
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axs[0]
    ax.plot(ppo_dec * 20 / 1e6, 100 * ppo_sel, "s-", color=PPO, lw=2, ms=6, label="PPO, seçim bloğu 48 bölüm (Isaac)")
    ob = es_stage["orbit_descent"]; c = ob["c"]; m = ~np.isnan(c["sel_land"])
    x_es = c["episodes"][m] * ob["steps_per_ep"] / 1e6
    ax.plot(x_es, 100 * c["sel_land"][m], "^-", color=ES, lw=2, ms=6, label="ES, yalnızca orbit_descent aşaması, 24 bölüm (analitik, warm-start)")
    ax.set_xlabel("harcanan kontrol adımı (milyon, 20 Hz)"); ax.set_ylabel("seçim bloğu iniş oranı (%)")
    ax.set_title("Aşama içi öğrenme (bloklar ve backend'ler farklı!)", fontsize=9.5); ax.legend(fontsize=7.8, loc="lower right"); ax.set_ylim(0, 105)
    ax = axs[1]
    names = [n for n, _ in stages]; tot = [es_stage[n]["ep_to_best"] * es_stage[n]["steps_per_ep"] / 1e6 for n in names]
    left = 0.0; cols = ["#9db8e4", "#7da3d9", "#5d8dce", "#3d78c3", ES]
    for n, t_, cc in zip(names, tot, cols):
        ax.barh(1, t_, left=left, color=cc, ec="white"); left += t_
    ax.barh(0, ppo_steps / 1e6, color=PPO)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["PPO\n(45 bin karar)", "ES\n(5 basamak)"]); ax.set_xscale("log"); ax.set_xlim(0.3, 300)
    ax.set_xlabel("toplam kontrol adımı (milyon, log)")
    ax.text(ppo_steps / 1e6 * 1.15, 0, f"≈{ppo_steps / 1e6:.1f} M", va="center", fontsize=9)
    ax.text(es_total_steps / 1e6 * 1.1, 1, f"≈{es_total_steps / 1e6:.0f} M", va="center", fontsize=9)
    ax.set_title("Eğitim bütçesi (seçilen kontrol noktasına kadar)", fontsize=9.5)
    return save(fig, "learn")

def fig_why():
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    cr = {"kontrolcü": 192 - kC, "PPO": 192 - kP, "ES": 192 - kE}
    ax.bar(list(cr), list(cr.values()), color=[CTL, PPO, ES], width=0.55)
    for i, v in enumerate(cr.values()): ax.text(i, v + 2, f"{v}", ha="center", fontsize=10)
    ax.set_ylabel("başarısız bölüm sayısı (192 içinde)"); ax.set_ylim(0, 110)
    ax.set_title("Başarısız bölümler (tümü çarpma; PPO için neden kaydı yok)", fontsize=10)
    return save(fig, "fail")

figs = {n: f() for n, f in (("flow", fig_flow), ("bars", fig_bars), ("waves", fig_waves), ("disc", fig_disc),
                            ("learn", fig_learn), ("fail", fig_why))}
img = lambda p, w="100%": f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" style="width:{w}">'  # noqa: E731
es_why = es["es_policy"]["why"]; c_why = es["controller"]["why"]

CSS = """@page{size:A4;margin:15mm 14mm}
body{font-family:'DejaVu Sans',sans-serif;font-size:10pt;line-height:1.5;color:#0b0b0b}
h1{font-size:20pt;margin:0 0 4px}h2{page-break-after:avoid;font-size:13.5pt;margin:18px 0 6px;border-bottom:2px solid #2a78d6;padding-bottom:3px}
.sub{color:#52514e;margin-bottom:14px}.kpi{display:flex;gap:10px;margin:10px 0}.kpi div{flex:1;border:1px solid #e4e3df;border-radius:8px;padding:8px 10px;background:#fcfcfb;font-size:9pt}
.kpi b{font-size:16pt;display:block}.note{background:#fff6e0;border-left:4px solid #c98a00;padding:8px 12px;margin:10px 0}
figure{margin:8px 0 12px;page-break-inside:avoid}figcaption{color:#52514e;font-size:9pt;margin-top:2px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9pt;page-break-inside:avoid}td,th{border:1px solid #e4e3df;padding:4px 7px;text-align:left;vertical-align:top}
th{background:#f4f3ef}code{background:#f4f3ef;padding:1px 4px;border-radius:3px;font-size:8.8pt}"""
pct = lambda k: f"%{100 * k / N:.1f}"  # noqa: E731
ci_s = lambda t: f"[{100 * t[0]:.1f}, {100 * t[1]:.1f}]"  # noqa: E731
sc = es_stage
body = f"""
<h1>RL (guided-gain PPO) ve evrimsel strateji (ES): aynı iniş görevinde karşılaştırma</h1>
<div class=sub>LunarSim · Apollo LM · Isaac orbit_descent (200 m, 10–30 m/s yanal hız) · tüm sayılar <code>out/</code> ve <code>~/lunarsim_artifacts/</code> dosyalarından okunmuştur</div>
<div class=kpi>
 <div><b>{pct(kC)}</b>ZEM/ZEV kontrolcüsü<br>{kC}/{N} · GA {ci_s(ci['C'])}</div>
 <div><b style="color:{PPO}">{pct(kP)}</b>RL: guided-gain PPO<br>{kP}/{N} · GA {ci_s(ci['P'])}</div>
 <div><b style="color:{ES}">{pct(kE)}</b>ES (kontrolcüsüz)<br>{kE}/{N} · GA {ci_s(ci['E'])}</div>
 <div><b>{only_e}–{only_p}</b>ES-only – PPO-only<br>McNemar p = {mEP[2]:.2f}</div>
</div>
<p><b>Özet.</b> İki yöntem de kontrolcüyü anlamlı biçimde geçiyor (PPO {mP[0]}–{mP[1]}, p={mP[2]:.1e}; ES {mE[0]}–{mE[1]}, p={mE[2]:.1e}). <b>ES ile PPO arasında istatistiksel fark yok</b>: ES {kE}/{N}, PPO {kP}/{N}; eşleştirilmiş farkın (ES − PPO) dalga-bloku bootstrap %95 aralığı [{bEP[0]:+.1f}, {bEP[1]:+.1f}] puan, dalgaların {waves_e_ge_p}/{NW}'inde ES ≥ PPO ({waves_e_gt_p} dalgada ES, {waves_p_gt_e} dalgada PPO önde). Bu bir "ES PPO'dan iyi" iddiası <i>değildir</i>. Kontrolcü vektörleri iki dosyada bölüm bölüm birebir aynıdır (aynı tohum, aynı başlangıçlar), bu yüzden üç kol eşleştirilmiş karşılaştırılmıştır.</p>

<h2>1. İkisi nasıl çalışıyor</h2>
<figure>{img(figs['flow'])}<figcaption>Şekil 1. Üstte RL: kontrolcü döngüde kalır, PPO yalnızca 11 kazanç çarpanını 1 sn'de bir ayarlar. Altta ES: hiç kontrolcü yok, 84 parametreli lineer politika doğrudan gaz ve üç eksen torku verir.</figcaption></figure>
<table>
<tr><th></th><th>RL: guided-gain PPO</th><th>ES (kontrolcüsüz)</th></tr>
<tr><td>Ne öğreniliyor</td><td>kontrolcünün 11 kazancının çarpanları (durumdan)</td><td>doğrudan eyleyici komutu (durumdan)</td></tr>
<tr><td>Kontrolcü döngüde mi</td><td><b>Evet</b>: ZEM/ZEV her adım uçar; eylem 0 = kontrolcünün kendisi</td><td><b>Hayır</b></td></tr>
<tr><td>Eylem uzayı / frekans</td><td>11 boyut, 1 sn'de bir karar, 20 adım tutma</td><td>4 boyut (gaz, pitch, roll, yaw), 20 Hz</td></tr>
<tr><td>Politika / parametre</td><td>64×64 tanh MLP, 23→11 (≈6.4 bin ağırlık, + değer ağı)</td><td>lineer, 20→4 (84 parametre)</td></tr>
<tr><td>Gözlem</td><td>20 ham + 3 ek (t_go, final-mi: kontrolcünün iç durumundan; v_xy)</td><td>20 (çalışan ort./std normalize); slot 16–19 rehber alanı</td></tr>
<tr><td>Ödül / uygunluk</td><td>sadece sonuç: +5 iniş, 0 başarısız temas, −3 zaman aşımı/devrilme (<code>reward_scale=0</code>)</td><td>sadece sonuç, derecelendirilmiş: temas şiddeti (cap 480), zaman aşımı −480</td></tr>
<tr><td>Optimizer</td><td>PPO (γ=0.995, λ=0.99, lr 1e-4, 8 ortam)</td><td>OpenAI-ES (48 antitetik çift, sıralama şekillendirme, Adam)</td></tr>
<tr><td>Eğitim backend'i</td><td>Isaac ortamı (<code>IsaacLanderVecEnv</code>), orbit_descent doğrudan</td><td>analitik backend, 5 basamaklı müfredat (warm-start); Isaac'te sıfır-atış</td></tr>
<tr><td>Doğrulama</td><td>Isaac, 192 eşleştirilmiş, tohum 90000–105000</td><td>aynı</td></tr>
</table>

<h2>2. Sonuçlar (Isaac, 192 eşleştirilmiş taze bölüm)</h2>
<figure>{img(figs['bars'], '62%')}<figcaption>Şekil 2. İniş oranları ve %95 Wilson güven aralıkları. Aralıklar büyük ölçüde örtüşüyor.</figcaption></figure>
<table>
<tr><th>karşılaştırma</th><th>sadece ilki indi</th><th>sadece ikincisi indi</th><th>McNemar p</th><th>fark, %95 bootstrap (puan)</th></tr>
<tr><td>kontrolcü → PPO</td><td>{mP[1]}</td><td>{mP[0]}</td><td>{mP[2]:.1e}</td><td>{100 * (kP - kC) / N:+.1f} [{bP[0]:+.1f}, {bP[1]:+.1f}]</td></tr>
<tr><td>kontrolcü → ES</td><td>{mE[1]}</td><td>{mE[0]}</td><td>{mE[2]:.1e}</td><td>{100 * (kE - kC) / N:+.1f} [{bE[0]:+.1f}, {bE[1]:+.1f}]</td></tr>
<tr><td><b>PPO → ES</b></td><td>{only_p}</td><td>{only_e}</td><td><b>{mEP[2]:.2f}</b></td><td>{100 * (kE - kP) / N:+.1f} [{bEP[0]:+.1f}, {bEP[1]:+.1f}]</td></tr>
</table>
<figure>{img(figs['waves'])}<figcaption>Şekil 3. Dalga bazında iniş sayıları ve ES − PPO farkı. Fark küçük ve işaret değiştiriyor; tek bir yönde sistematik üstünlük görülmüyor (16 dalgada 8 / 5 / 3).</figcaption></figure>
<figure>{img(figs['disc'], '44%')}<figcaption>Şekil 4. PPO ve ES'in bölüm bazında örtüşmesi. İkisinin de çarptığı bölüm <b>{neither}</b>: PPO'nun {only_e} hatasını ES, ES'in {only_p} hatasını PPO çözüyor. Yani hatalar farklı bölümlerde; bu iki yöntemin aynı zor bölümlerde takılmadığını gösterir, ama bir "topluluk" sonucu olarak raporlanmadı.</figcaption></figure>
<figure>{img(figs['fail'], '55%')}<figcaption>Şekil 5. Başarısız bölümler. ES: {es_why}; kontrolcü: {c_why}. PPO doğrulama dosyası sonlanma nedenini kaydetmiyor.</figcaption></figure>

<h2>3. Örnek verimliliği ve bütçe</h2>
<figure>{img(figs['learn'])}<figcaption>Şekil 6. Solda seçim bloğundaki iniş oranı, sağda toplam harcanan kontrol adımı (log ölçek). Birimler: 1 PPO kararı = 20 kontrol adımı (1 sn). ES için bölüm sayıları eğri dosyalarından, bölüm süresi ise doğrulamadaki iniş süresinden alınmış tahmindir.</figcaption></figure>
<table>
<tr><th></th><th>PPO</th><th>ES</th></tr>
<tr><td>Seçilen kontrol noktası</td><td>{best_ppo_dec:,} karar (8 ortam), seçim bloğu {ppo_sel_best[0] if ppo_sel_best else '?'}/{ppo_sel_best[1] if ppo_sel_best else '?'}; bütçe {ppo_budget:,} karar</td><td>müfredatın 5 basamağı, seçilen nesle kadar toplam {es_total_ep:,} bölüm ({', '.join(f"{n} {v['ep_to_best']:,}" for n, v in sc.items())})</td></tr>
<tr><td>Kontrol adımı (20 Hz)</td><td>≈ {ppo_steps / 1e6:.2f} M</td><td>≈ {es_total_steps / 1e6:.0f} M toplam (≈ {es_orbit_steps / 1e6:.1f} M yalnızca orbit_descent)</td></tr>
<tr><td>Duvar saati</td><td>{ppo_wall_s:.0f} s toplam süreç (≈ {ppo_wall_s / 60:.0f} dk; SB3 eğitim parçası ≈ 4 × {chunk_s / 60:.0f} dk + 4 değerlendirme; ≈ {fps[-1] if fps else '?'} karar/sn)</td><td>≈ {es_wall_35_min:.0f} dk (ramp_35m) + ≈ {es_wall_ladder_min:.0f} dk (4 basamak), 16 çekirdek CPU, toplam ≈ {es_wall_35_min + es_wall_ladder_min:.0f} dk; dosya oluşturma/değiştirme zaman damgalarından, müfredat süresi basamak başı doğrulama koşularını içerir</td></tr>
<tr><td>Donanım</td><td>1 GPU (Isaac)</td><td>16 çekirdek CPU (analitik)</td></tr>
</table>
<p>Kontrol adımı bakımından ES, PPO'nun yaklaşık {es_total_steps / ppo_steps:.0f} katı simülasyon harcıyor (yalnızca son aşama için yaklaşık {es_orbit_steps / ppo_steps:.0f} kat; ramp_35m için bölüm süresi 25 sn bütçesinden alındığından toplam bir üst sınırdır). Buna karşılık analitik backend, Isaac ortamından çok daha hızlı; duvar saati bu yüzden iki yöntemde aynı mertebede. Bu nedenle "ES örnek açısından verimsiz, saat açısından rekabetçi" demek ölçümle uyumludur, "ES daha verimli" demek değildir.</p>

<h2>4. Dürüst sınırlar</h2>
<div class=note><ul style="margin:0 0 0 16px;padding:0">
<li><b>Tek tohum, tek koşu.</b> PPO tek tohumdu (seed 21; bu, tekrar üretilmiş koşu); ES de tek tohum. Tekrarlanabilirlik ölçülmedi. Geçmiş PPO koşularında tohumlar arası sonuçlar çok oynadı (handover.md §0: land_handoff'ta iki tohum %21.9 ve %31.2).</li>
<li><b>Çözünürlük.</b> 192 bölümle bu eşleştirilmiş deneyde farkın %95 yarı-genişliği yaklaşık ±{half_width_pts:.1f} puan; ES–PPO farkı ({100 * (kE - kP) / N:+.1f} puan) bunun çok içinde.</li>
<li><b>Farklı backend'ler.</b> PPO Isaac ortamında, ES analitik ortamda eğitildi ve Isaac'te sıfır-atış doğrulandı. Seçim blokları farklı (PPO: tohum 7000, 48 bölüm, Isaac; ES: tohum 150000+, 24 bölüm, analitik); eğri karşılaştırması yalnızca yön verir.</li>
<li><b>Seçim yanlılığı.</b> İki yöntemde de kontrol noktası kendi seçim bloğunda seçildi; sayılar taze doğrulama bloğundandır. PPO'nun 45 bin kararlık noktası dört değerlendirmeden (15k/30k/45k/60k: %77, %81, %90, %77) en iyisi; ara değerler gürültülü.</li>
<li><b>PPO kontrolcüye ihtiyaç duyar</b> ve kontrolcünün iç durumunu (t_go, final-mi) gözlem olarak görür; ES kontrolcüsüzdür ama gözlemindeki rehber slotlar (16–19) ödül ağırlıklarından hesaplanır. İkisi de rehber bilgiyi girdi olarak kullanıyor.</li>
<li><b>ES eğitimi kısa kesildi.</b> orbit_descent aşaması {sc['orbit_descent']['gens']} nesilde durdu (24 bölümlük seçim bloğu iki kez ≥%97); daha uzun eğitim denenmedi.</li>
<li><b>Ödül farkı.</b> PPO ödülü ikili sonuç (iniş/başarısız), ES uygunluğu derecelendirilmiş şiddet; "aynı hedef" değil, "aynı sonuç ölçütü" (landed_safely).</li>
</ul></div>

<h2>5. Veri kaynakları</h2>
<table>
<tr><td>PPO doğrulama</td><td><code>~/lunarsim_artifacts/gg_isaac_orbit_v4_seed21_repro_validation.json</code> (+ <code>.log</code>); eğitim: <code>out/gg_isaac_orbit_v4_seed21_repro.log</code>, <code>…/curve.csv</code></td></tr>
<tr><td>ES doğrulama</td><td><code>out/evo/isaac_orbit_val192.json</code>; eğitim: <code>out/evo/an_n1/</code>, <code>out/evo/an_ladder_*/curve.csv</code>, <code>out/evo/ladder.log</code></td></tr>
<tr><td>Eşleştirme kontrolü</td><td>kontrolcü vektörleri iki dosyada {'özdeş' if PAIRED else 'FARKLI'}, dalga tohumları {'özdeş' if ppo['wave_seeds'] == es['seeds'] else 'farklı'}</td></tr>
</table>
"""
if not PAIRED:
    raise SystemExit("controller vectors differ between the two files: comparison would be unpaired")
html_path = ROOT / "report" / "lunarsim_rl_vs_es_rapor.html"
html_path.write_text(f"<!doctype html><html lang=tr><head><meta charset=utf-8><title>RL vs ES</title><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
pdf = ROOT / "report" / "lunarsim_rl_vs_es_rapor.pdf"
subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf)
print(dict(kC=kC, kP=kP, kE=kE, mP=mP, mE=mE, mEP=mEP, bP=bP, bE=bE, bEP=bEP, both=both, only_p=only_p, only_e=only_e,
           neither=neither, waves_e_ge_p=waves_e_ge_p, half_width_pts=round(half_width_pts, 1),
           es_total_ep=es_total_ep, es_total_steps_M=round(es_total_steps / 1e6, 1), es_orbit_steps_M=round(es_orbit_steps / 1e6, 1),
           ppo_steps_M=ppo_steps / 1e6, ppo_wall_s=ppo_wall_s, fps=fps[-1:] , es_wall_ladder_min=round(es_wall_ladder_min, 1),
           es_wall_35_min=round(es_wall_35_min, 1), land_t=land_t, ppo_sel_best=ppo_sel_best, ppo_envs=ppo_envs))
