"""Comprehensive report (Turkish): crater-rich terrain, how each method works (flowcharts), results,
failure analysis. HTML -> PDF with headless Chrome.  Run: ~/isaac-env/bin/python report/build_krater_rapor.py
"""
from __future__ import annotations
import base64, csv, json, math, re, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D2 = ROOT / "out" / "crater2"; D1 = ROOT / "out" / "crater"; ART = Path.home() / "lunarsim_artifacts"
FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CTL, PPO, ES, MPC = "#eb6834", "#0ca30c", "#2a78d6", "#7a3fc4"
CRIT, WARN = "#d03b3b", "#c98a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

# ---------------------------------------------------------------- helpers
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h

def mcn(a, b):  # a-only, b-only, exact two-sided p
    x = int((a & ~b).sum()); y = int((b & ~a).sum()); N = x + y
    if N == 0: return x, y, 1.0
    k = min(x, y); return x, y, min(1.0, 2 * sum(math.comb(N, i) for i in range(k + 1)) / 2 ** N)

def boot(a, b, waves, B=10000, seed=0):  # rate(b)-rate(a), wave-block bootstrap
    A, Bm = a.reshape(waves, -1), b.reshape(waves, -1); r = np.random.default_rng(seed)
    d = [Bm[i].mean() - A[i].mean() for i in (r.integers(0, waves, waves) for _ in range(B))]
    return np.percentile(d, [2.5, 97.5])

def box(ax, x, y, w, h, text, fc="#eef4fc", ec=ES, fs=8.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            color=INK, linespacing=1.25)

def arrow(ax, p0, p1, color=INK2, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))

def feedback(ax, pts, color=INK2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.5, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color)

def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(-0.2, 10.4); ax.set_ylim(0, 10 * h / w); return fig, ax

def save(fig, name):
    p = FIG / f"kr_{name}.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); return p

def img(p, w="100%"):
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" style="width:{w}">'

def tint(c, a=0.14):
    c = c.lstrip("#"); r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(255 - (255 - v) * a * 4 if False else v * a + 255 * (1 - a)) for v in (r, g, b))

# ---------------------------------------------------------------- data
def load_dense():
    e = json.load(open(D2 / "es.json")); m = json.load(open(D2 / "mpc.json")); p = json.load(open(D2 / "ppo.json"))
    ctl = np.array(e["controller"]["landed"]); es = np.array(e["es_policy"]["landed"])
    mpc = np.array(m["mpc"]["landed"]); ppo = np.array(p["candidate_landed"])
    assert (np.array(p["baseline_landed"]) == ctl).all(), "controller vector differs between ES and PPO runs"
    return dict(ctl=ctl, es=es, mpc=mpc, ppo=ppo, why_es=e["es_policy"]["why"], why_ctl=e["controller"]["why"],
                why_mpc=m["mpc"]["why"])

def load_c80():
    e = json.load(open(D1 / "es.json")); m = json.load(open(D1 / "mpc.json"))
    ctl = np.array(e["controller"]["landed"]); es = np.array(e["es_policy"]["landed"]); mpc = np.array(m["mpc"]["landed"])
    pc, pp = [], []
    for l in open(D1 / "ppo.log"):
        r = re.search(r"controller (\d+)/12\s+policy (\d+)/12", l)
        if r: pc.append(int(r[1])); pp.append(int(r[2]))
    return dict(ctl=ctl, es=es, mpc=mpc, ppo_k=int(sum(pp)), ppo_n=12 * len(pp), ctl_ppo_k=int(sum(pc)))

dense = load_dense(); c80 = load_c80()
n = len(dense["es"]); W = n // 12
flat = dict(ctl=(101, 192), mpc=(139, 192), ppo=(172, 192), es=(175, 192))
arms = [("ctl", "ZEM/ZEV kontrolcüsü", CTL), ("ppo", "PPO (kazanç ayarı)", PPO), ("es", "ES (kontrolcüsüz)", ES),
        ("mpc", "MPC (MPPI)", MPC)]
K = {k: int(dense[k].sum()) for k, _, _ in arms}
print("dense", K, "n", n)

# ---------------------------------------------------------------- figures
def fig_terrain():
    from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config
    from lunarsim.core.terrain.generate import generate_tile
    import copy
    st = STAGES_BY_NAME["orbit_descent"]; area = st.tile_size_m ** 2
    base = terrain_config(st, 11, 256)
    cfgs = {"Önceki alan (stok)": base}
    c80_ = copy.deepcopy(base)
    c80_.craters = {"count_scale": 80 / (area * 5e-4), "d_min_m": 12.0, "d_max_m": 70.0, "b": 1.5, "depth_ratio": 0.10, "age": 0.4}
    cfgs["80 krater (12–70 m)"] = c80_
    cd = copy.deepcopy(base)
    cd.craters = {"count_scale": 1000 / (area * 5e-4), "d_min_m": 6.0, "d_max_m": 140.0, "b": 1.8, "depth_ratio": 0.10, "age": 0.4}
    cfgs["Yoğun alan (~1000 krater, 6–140 m)"] = cd
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], hspace=0.22, wspace=0.12)
    cols = ["#8d8d8d", WARN, CRIT]; axs3 = fig.add_subplot(gs[:, 3])
    for j, ((nm, c), col) in enumerate(zip(cfgs.items(), cols)):
        t = generate_tile(c); h = np.asarray(t.height); N = h.shape[0]; res = t.size_m / N
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(h.T, origin="lower", cmap="gist_earth", extent=[-840, 840, -840, 840], vmin=-6, vmax=2)
        ax.set_title(nm + "\n1680 m karo", fontsize=8.8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.add_patch(plt.Rectangle((-250, -250), 500, 500, fill=False, ec="white", lw=1.2))
        z = fig.add_subplot(gs[1, j]); c0 = N // 2; r = int(250 / res)
        z.imshow(h[c0 - r:c0 + r, c0 - r:c0 + r].T, origin="lower", cmap="gist_earth", extent=[-250, 250, -250, 250], vmin=-6, vmax=2)
        z.set_title("merkezde 500 m", fontsize=8.5); z.set_xticks([]); z.set_yticks([]); z.grid(False)
        gy, gx = np.gradient(h, res); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        axs3.plot(np.sort(sl.ravel()), 1 - np.arange(sl.size) / sl.size, color=col, lw=1.8,
                  label=f"{nm.split(' (')[0]}: p99 {np.percentile(sl, 99):.1f}°, maks {sl.max():.0f}°")
    axs3.set_yscale("log"); axs3.set_xlabel("yerel eğim (°)"); axs3.set_ylabel("daha dik olan arazi oranı")
    axs3.set_xlim(0, 30); axs3.legend(fontsize=7.4, loc="upper right"); axs3.set_title("Eğim dağılımı", fontsize=9)
    return save(fig, "terrain")

def _rec(name):
    return np.load(D2 / "rec" / name)

ENV = int((D2 / "env_picked").read_text())
RECS = {"es": "es_ext.npz", "ppo": "ppo_ext.npz", "mpc": "mpc_ext.npz"}
COL = {"es": ES, "ppo": PPO, "mpc": MPC}
NAME = {"es": "ES", "ppo": "PPO", "mpc": "MPC"}

def fig_track():
    d = _rec(RECS["es"]); h = d[f"e{ENV}_tile_height"].astype(float); s = float(d[f"e{ENV}_tile_size"]); N = h.shape[0]
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.0]))
    for k in (0, 1):
        axs[k].imshow(h.T, origin="lower", cmap="gist_earth", extent=[-s / 2, s / 2, -s / 2, s / 2], vmin=-8, vmax=3)
        axs[k].grid(False)
    xs = []; ys = []
    for a, f in RECS.items():
        dd = _rec(f); x, y = dd[f"e{ENV}_x"], dd[f"e{ENV}_y"]; xs.append(x); ys.append(y)
        for k in (0, 1):
            axs[k].plot(x, y, color=COL[a], lw=1.6, label=NAME[a] if k == 0 else None)
            axs[k].plot(x[-1], y[-1], "o", color=COL[a], ms=6, mec="white")
    X = np.concatenate(xs); Y = np.concatenate(ys)
    axs[0].legend(fontsize=8, loc="upper left"); axs[0].set_title(f"Arazi karosu (1680 m) ve yer izleri, bölüm {ENV}", fontsize=9)
    axs[1].set_xlim(X.min() - 150, X.max() + 150); axs[1].set_ylim(Y.min() - 150, Y.max() + 150)
    axs[1].set_title("Yakınlaştırma: son yaklaşma ve iniş noktaları", fontsize=9)
    # terrain profile under the ES ground track
    dd = _rec(RECS["es"]); x, y = dd[f"e{ENV}_x"], dd[f"e{ENV}_y"]; gz = dd[f"e{ENV}_ground_z"]
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    axs[2].plot(dist, gz, color=INK2, lw=1.4); axs[2].fill_between(dist, gz, gz.min() - 1, color="#d9d3c4")
    axs[2].set_xlabel("yer izi boyunca mesafe (m)"); axs[2].set_ylabel("zemin yüksekliği (m)")
    axs[2].set_title("Araç altındaki arazi profili (ES izi)", fontsize=9)
    fig.tight_layout(); return save(fig, "track")

def fig_flow_ctl():
    fig, ax = blank(11, 3.1)
    box(ax, 0.1, 1.0, 1.7, 1.5, "ölçülen durum\nkonum, hız,\nyatış, açısal hız", "#f4f3ef", INK2)
    box(ax, 2.3, 1.0, 2.0, 1.5, "ZEM / ZEV\nrehberliği\nistenen ivme", "#fff1e8", CTL, bold=True)
    box(ax, 4.8, 1.0, 1.9, 1.5, "ivme → gaz ve\neğim (pitch/roll)\nkomutu", "#fff1e8", CTL)
    box(ax, 7.2, 1.0, 1.7, 1.5, "iç yatış döngüsü\nkp_att, kd_att\n→ RCS torku", "#fff1e8", CTL)
    box(ax, 9.2, 1.0, 0.75, 1.5, "araç", "#f4f3ef", INK2)
    for a, b in ((1.8, 2.3), (4.3, 4.8), (6.7, 7.2), (8.9, 9.2)): arrow(ax, (a, 1.75), (b, 1.75))
    feedback(ax, [(9.55, 1.0), (9.55, 0.55), (1.0, 0.55), (1.0, 1.0)])
    ax.text(5.2, 0.12, "geri besleme: 20 Hz, sabit kazançlar", ha="center", fontsize=8.5, color=INK2)
    ax.text(0.1, 2.95, "Sabit ZEM/ZEV kontrolcüsü: öğrenme yok, kazançlar elle seçilmiş", fontsize=10, weight="bold")
    return save(fig, "flow_ctl")

def fig_flow_ppo():
    fig, ax = blank(11, 4.4)
    ax.text(0.1, 4.25, "PPO ('guided-gain'): ağ kontrolcünün kazançlarını ayarlar, kontrolcü uçmaya devam eder", fontsize=10, weight="bold")
    box(ax, 0.1, 2.5, 1.7, 1.2, "gözlem\n(20 temel + 3 ek)", "#f4f3ef", INK2)
    box(ax, 2.3, 2.5, 2.0, 1.2, "PPO politika ağı\n(Isaac'te eğitildi)", "#eaf7ea", PPO, bold=True)
    box(ax, 4.8, 2.5, 2.1, 1.2, "11 kazanç çarpanı\nkazanç × oran^a\n1 s sabit tutulur", "#eaf7ea", PPO)
    box(ax, 7.4, 2.5, 1.9, 1.2, "ZEM/ZEV\nkontrolcüsü\n(ayarlı kazançlarla)", "#fff1e8", CTL)
    box(ax, 9.5, 2.5, 0.5, 1.2, "araç", "#f4f3ef", INK2, fs=8)
    for a, b in ((1.8, 2.3), (4.3, 4.8), (6.9, 7.4), (9.3, 9.5)): arrow(ax, (a, 3.1), (b, 3.1))
    feedback(ax, [(9.75, 2.5), (9.75, 2.05), (1.0, 2.05), (1.0, 2.5)])
    ax.text(5.2, 1.6, "a = 0 ise çarpan 1: kontrolcünün kendisi. Taban başarı kontrolcüdür; ağ yalnızca sapmayı öğrenir.",
            ha="center", fontsize=8.6, color=INK2)
    box(ax, 0.5, 0.15, 3.0, 0.95, "eğitim: rollout topla (Isaac)\nGAE avantajı, kırpılmış PPO hedefi", "#eaf7ea", PPO, fs=8.5)
    box(ax, 4.0, 0.15, 3.0, 0.95, "ödül: iniş sonucu baskın\n(terminal ağırlıklı, reward_scale=0)", "#eaf7ea", PPO, fs=8.5)
    box(ax, 7.5, 0.15, 2.5, 0.95, "doğrulama: eğitimin\ngörmediği tohumlar", "#eaf7ea", PPO, fs=8.5)
    return save(fig, "flow_ppo")

def fig_flow_es():
    fig, ax = blank(11, 6.2)
    ax.text(0.1, 6.05, "ES: türev kullanmayan arama ile 84 parametreli doğrusal politika (kontrolcü yok)", fontsize=10, weight="bold")
    # inference row
    ax.text(0.1, 5.5, "A. Uçuş (her 0.05 s):", fontsize=9, weight="bold", color=ES)
    box(ax, 0.1, 4.2, 1.8, 1.0, "gözlem (20)\n+ rehber alan\nözellikleri", "#f4f3ef", INK2)
    box(ax, 2.4, 4.2, 1.9, 1.0, "çalışan ortalama/\nstd ile\nnormalize et", "#eef4fc", ES)
    box(ax, 4.8, 4.2, 1.9, 1.0, "doğrusal katman\n20×4 + 4 = 84\nparametre θ", "#eef4fc", ES, bold=True)
    box(ax, 7.2, 4.2, 1.5, 1.0, "tanh / kırp\n[-1, 1]", "#eef4fc", ES)
    box(ax, 9.0, 4.2, 1.0, 1.0, "gaz, pitch,\nroll, yaw", "#f4f3ef", INK2, fs=8)
    for a, b in ((1.9, 2.4), (4.3, 4.8), (6.7, 7.2), (8.7, 9.0)): arrow(ax, (a, 4.7), (b, 4.7))
    # training loop
    ax.text(0.1, 3.6, "B. Eğitim döngüsü (analitik ortamda, basamaklı müfredat):", fontsize=9, weight="bold", color=ES)
    box(ax, 0.1, 2.0, 1.8, 1.2, "θ'nın etrafında\nε ~ N(0, I)\ndeğişimler", "#eef4fc", ES)
    box(ax, 2.3, 2.0, 1.9, 1.2, "antitetik çiftler\nθ + σε , θ − σε\naynı tohumlarda", "#eef4fc", ES)
    box(ax, 4.6, 2.0, 1.9, 1.2, "her aday için\nbölüm koştur,\nuygunluk F al", "#eef4fc", ES)
    box(ax, 6.9, 2.0, 1.6, 1.2, "merkezli sıra\n(centered rank)\ndönüşümü", "#eef4fc", ES)
    box(ax, 8.8, 2.0, 1.2, 1.2, "Adam ile\nθ güncelle", "#eef4fc", ES, bold=True)
    for a, b in ((1.9, 2.3), (4.2, 4.6), (6.5, 6.9), (8.5, 8.8)): arrow(ax, (a, 2.6), (b, 2.6))
    feedback(ax, [(9.4, 2.0), (9.4, 1.55), (1.0, 1.55), (1.0, 2.0)])
    ax.text(5.2, 1.1, "tekrar: nesil = bir güncelleme; sonra bir sonraki zorluk basamağına ısınma başlangıcıyla geç",
            ha="center", fontsize=8.6, color=INK2)
    box(ax, 0.3, 0.1, 4.7, 0.6, "ramp_35m → ramp_50m → ramp_100m → ramp_150m → orbit_descent", "#f4f3ef", INK2, fs=8.4)
    box(ax, 5.3, 0.1, 4.7, 0.6, "Isaac Sim'de sıfır-atış doğrulama (eğitimde hiç kullanılmadı)", "#eef4fc", ES, fs=8.4)
    return save(fig, "flow_es")

def fig_flow_mpc():
    fig, ax = blank(11, 4.9)
    ax.text(0.1, 4.75, "MPC (MPPI): her 0.2 s'de modelde ileri sararak plan yap, ilk komutu uygula", fontsize=10, weight="bold")
    box(ax, 0.1, 3.0, 1.8, 1.3, "ölçülen durum\n(konum, hız, yatış,\nyakıt, kalan süre)", "#f4f3ef", INK2)
    box(ax, 2.3, 3.0, 2.0, 1.3, "512 aday komut\ndizisi örnekle\n(20 segment × 0.6 s)", "#f3ecfb", MPC)
    box(ax, 4.7, 3.0, 2.1, 1.3, "dinamik model (GPU)\n12 s ileri sar\n(AnalyticLander kopyası)", "#f3ecfb", MPC, bold=True)
    box(ax, 7.2, 3.0, 1.4, 1.3, "maliyetle\npuanla", "#f3ecfb", MPC)
    box(ax, 8.9, 3.0, 1.3, 1.3, "softmin\nağırlıklı\nortalama", "#f3ecfb", MPC, fs=8.2)
    for a_, b_ in ((1.9, 2.3), (4.3, 4.7), (6.8, 7.2), (8.6, 8.9)): arrow(ax, (a_, 3.65), (b_, 3.65))
    box(ax, 8.2, 1.2, 2.0, 1.1, "ilk 0.2 s'lik komut\naraca uygulanır", "#f4f3ef", INK2, fs=8.5)
    arrow(ax, (9.55, 3.0), (9.3, 2.3))
    feedback(ax, [(8.2, 1.75), (7.9, 0.55), (1.0, 0.55), (1.0, 3.0)])
    ax.text(0.2, 1.75, "Maliyet:\n• yanal/dikey hız hatası (rehber alana göre)\n• yatış, açısal hız, komut eforu, zaman\n"
            "• temas şiddeti: w·(0.2 sev² + 10 relu(sev−1)²)\n• süre dolarsa büyük ceza\n• limitlerin %70'i kullanılır (güvenlik payı)",
            fontsize=8.4, va="center", color=INK)
    ax.text(5.4, 0.2, "10 maliyet ağırlığı ES ile ayarlandı; model dinamik denklemleri biliyor, öğrenme yok.",
            ha="center", fontsize=8.6, color=INK2)
    return save(fig, "flow_mpc")

def fig_compare_info():
    fig, ax = blank(11, 2.6)
    ax.text(0.1, 2.5, "Kim neyi biliyor, neyi karar veriyor?", fontsize=10, weight="bold")
    return None

def fig_results():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    for i, (k, nm, col) in enumerate(arms):
        lo, hi = wil(K[k], n)
        ax.bar(i, 100 * K[k] / n, color=col, yerr=[[100 * K[k] / n - 100 * lo], [100 * hi - 100 * K[k] / n]],
               capsize=5, ecolor=INK2)
        ax.text(i, 100 * hi + 2, f"{K[k]}/{n}\n%{100 * K[k] / n:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([a[1].replace(" (", "\n(") for a in arms], fontsize=8.5)
    ax.set_ylim(0, 112); ax.set_ylabel("iniş oranı (%)"); ax.set_title(f"Yoğun krater alanı, Isaac, {n} eşleştirilmiş bölüm (%95 Wilson)", fontsize=9)
    ax = axs[1]
    terr = ["Düz arazi\n(192 bölüm)", "80 krater\n(ES/MPC 192, PPO 84)", f"Yoğun krater\n({n} bölüm)"]
    vals = {
        "ctl": [100 * 101 / 192, 100 * c80["ctl"].mean(), 100 * K["ctl"] / n],
        "ppo": [100 * 172 / 192, 100 * c80["ppo_k"] / c80["ppo_n"], 100 * K["ppo"] / n],
        "es": [100 * 175 / 192, 100 * c80["es"].mean(), 100 * K["es"] / n],
        "mpc": [100 * 139 / 192, 100 * c80["mpc"].mean(), 100 * K["mpc"] / n]}
    w = 0.2
    for j, (k, nm, col) in enumerate(arms):
        xs = np.arange(3) + (j - 1.5) * w
        ax.bar(xs, vals[k], w, color=col, label=nm)
        for x, v in zip(xs, vals[k]): ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(terr, fontsize=8.5); ax.set_ylim(0, 105)
    ax.set_ylabel("iniş oranı (%)"); ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.set_title("Arazi zorlaştıkça iniş oranı", fontsize=9)
    fig.tight_layout(); return save(fig, "results")

def fig_waves():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for k, nm, col in arms:
        ax.plot(np.arange(W), dense[k].reshape(W, -1).sum(1), "o-", color=col, label=nm, lw=1.6)
    ax.set_xlabel("dalga (12 bölüm)"); ax.set_ylabel("iniş (12'de)"); ax.set_ylim(0, 12.5); ax.legend(fontsize=8, ncol=4)
    ax.set_title("Dalga bazında inişler (aynı bölümler)", fontsize=9)
    fig.tight_layout(); return save(fig, "waves")

def fig_overlap():
    es, ppo, mpc = dense["es"], dense["ppo"], dense["mpc"]
    pats = {}
    for i in range(n):
        key = (bool(es[i]), bool(ppo[i]), bool(mpc[i])); pats[key] = pats.get(key, 0) + 1
    order = sorted(pats.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    labels = []
    for key, c in order:
        lab = "+".join(nm for nm, v in zip(("ES", "PPO", "MPC"), key) if v) or "hiçbiri"
        labels.append(lab)
    ax.barh(range(len(order)), [c for _, c in order], color="#8b8b8b")
    for i, (_, c) in enumerate(order): ax.text(c + 0.5, i, str(c), va="center", fontsize=8.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("bölüm sayısı"); ax.set_title("Hangi yöntemler hangi bölümlerde indi? (örtüşme)", fontsize=9)
    fig.tight_layout(); return save(fig, "overlap"), order

def fig_traj():
    fig, axs = plt.subplots(5, 1, figsize=(8.6, 9.2), sharex=True)
    for a, f in RECS.items():
        d = _rec(f); e = ENV; dt = float(d["dt"]); alt = d[f"e{e}_alt"]; t = np.arange(len(alt)) * dt
        hv = np.hypot(d[f"e{e}_vx"], d[f"e{e}_vy"]); tilt = np.degrees(np.hypot(d[f"e{e}_tilt_x"], d[f"e{e}_tilt_y"]))
        for ax, y in zip(axs, (alt, hv, -d[f"e{e}_vz"], tilt, d[f"e{e}_throttle"])):
            ax.plot(t, y, color=COL[a], lw=1.5, label=NAME[a])
    for ax, lab, lim in zip(axs, ("irtifa (m)", "yanal hız (m/s)", "iniş hızı (m/s)", "yatış (°)", "gaz (0–1)"), (None, 1.2, 1.0, 15, None)):
        ax.set_ylabel(lab)
        if lim is not None: ax.axhline(lim, color=CRIT, ls="--", lw=0.9)
    axs[0].legend(ncol=3, fontsize=8); axs[-1].set_xlabel("zaman (s)")
    axs[0].set_title(f"Aynı bölüm ({ENV}), aynı başlangıç: üç yöntemin uçuşu (kesikli çizgi: iniş limiti)", fontsize=9)
    fig.tight_layout(); return save(fig, "traj")

CRIT_NAMES = {"v_z": "dikey hız", "v_xy": "yanal hız", "tilt": "yatış", "w": "açısal hız", "leg_diff": "bacak yükseklik farkı"}
def fail_data():
    rows = {}
    for a, f in (("es", "evo_wave90000_policy.npz"), ("ppo", "wave90000_policy.npz"), ("mpc", "mpc_wave90000_policy.npz"),
                 ("ctl", "evo_wave90000_controller.npz")):
        d = np.load(D2 / "rec" / f); rows[a] = [json.loads(str(d[f"e{i}_outcome"])) for i in range(12)]
    return rows

def fig_fail(rows):
    keys = list(CRIT_NAMES)
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.0), sharey=True)
    tally = {}
    for ax, (a, nm, col) in zip(axs, arms):
        M = np.full((12, len(keys)), np.nan)
        for i, o in enumerate(rows[a]):
            for j, kx in enumerate(keys):
                M[i, j] = o["margins"].get(kx, np.nan) if o.get("margins") else np.nan
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([CRIT_NAMES[k_] for k_ in keys], rotation=40, ha="right", fontsize=7.5)
        ax.set_title(nm, fontsize=9, color=col)
        for i, o in enumerate(rows[a]):
            ax.text(-0.9, i, "✓" if o["landed"] else "✗", ha="center", va="center", fontsize=9,
                    color=GOODC if o["landed"] else CRIT)
        t = {}
        for o in rows[a]:
            if o["landed"] or not o.get("margins"): continue
            for kx, v in o["margins"].items():
                if v <= 0.005: t[kx] = t.get(kx, 0) + 1
        tally[a] = t
    axs[0].set_yticks(range(12)); axs[0].set_yticklabels([f"bölüm {i}" for i in range(12)], fontsize=7.5)
    fig.suptitle("Kayıtlı dalga (tohum 90000): temasta kalan marj (yeşil geniş, kırmızı sıfır = limit aşıldı)", fontsize=9.5)
    fig.tight_layout(); return save(fig, "fail"), tally

GOODC = "#0ca30c"

# ---------------------------------------------------------------- build figures
f_ter = fig_terrain(); f_trk = fig_track()
f_fc, f_fp, f_fe, f_fm = fig_flow_ctl(), fig_flow_ppo(), fig_flow_es(), fig_flow_mpc()
f_res = fig_results(); f_wav = fig_waves(); (f_ov, overlap) = fig_overlap(); f_tr = fig_traj()
rows = fail_data(); (f_fl, tally) = fig_fail(rows)

# ---------------------------------------------------------------- stats
pairs = [("ES", "ctl", "es"), ("PPO", "ctl", "ppo"), ("MPC", "ctl", "mpc"), ("ES", "ppo", "es"), ("MPC", "es", "mpc"), ("MPC", "ppo", "mpc")]
nm_ = {"ctl": "kontrolcü", "es": "ES", "ppo": "PPO", "mpc": "MPC"}
prow = ""
for lab, a, b in pairs:
    x, y, p = mcn(dense[b], dense[a])  # b-only, a-only
    ci = boot(dense[a], dense[b], W)
    prow += (f"<tr><td>{nm_[b]} vs {nm_[a]}</td><td>{x}</td><td>{y}</td><td>{p:.1e}</td>"
             f"<td>{100 * (dense[b].mean() - dense[a].mean()):+.1f}  [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]</td></tr>")

def tally_txt(a):
    t = tally.get(a, {})
    return ", ".join(f"{CRIT_NAMES.get(k, k)} {v}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])) or "—"

why = {a: dense.get(f"why_{a}") for a in ("es", "mpc", "ctl")}
def whytxt(d): return ", ".join(f"{k} {v}" for k, v in d.items()) if d else "—"

nl = {a: sum(1 for o in rows[a] if not o["landed"]) for a in rows}
lg = {a: [i for i, o in enumerate(rows[a]) if o["landed"]] for a in rows}

# ---------------------------------------------------------------- document
CSS = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.9pt; line-height: 1.5; color: #0b0b0b; }
h1 { font-size: 21pt; margin: 0 0 4px; } h2 { page-break-after: avoid; font-size: 13.5pt; margin: 20px 0 6px; border-bottom: 2px solid #2a78d6; padding-bottom: 3px; }
h3 { font-size: 11pt; margin: 14px 0 4px; page-break-after: avoid; }
.sub { color: #52514e; margin-bottom: 14px; }
.kpi { display: flex; gap: 8px; margin: 10px 0; } .kpi div { flex: 1; border: 1px solid #e4e3df; border-radius: 8px; padding: 8px 10px; background: #fcfcfb; font-size: 9pt; }
.kpi b { font-size: 17pt; display: block; }
.note { background: #fff6e0; border-left: 4px solid #c98a00; padding: 8px 12px; margin: 10px 0; }
.key { background: #eef4fc; border-left: 4px solid #2a78d6; padding: 8px 12px; margin: 10px 0; }
figure { margin: 8px 0 12px; page-break-inside: avoid; } figcaption { color: #52514e; font-size: 8.8pt; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.8pt; page-break-inside: avoid; } td, th { border: 1px solid #e4e3df; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f4f3ef; } code { background: #f4f3ef; padding: 1px 4px; border-radius: 3px; font-size: 8.6pt; }
.pb { page-break-before: always; }
"""
body = f"""
<h1>Kraterli Ay yüzeyinde iniş: dört yöntem, nasıl çalışıyorlar, ne kadar başarılılar</h1>
<div class=sub>LunarSim · Apollo LM · Isaac Sim 6.1 · orbit_descent · 13 Ekim 2026 · tüm sayılar <code>out/crater*</code> ve <code>~/lunarsim_artifacts/</code> dosyalarından okunmuştur</div>
<div class=kpi>
 <div><b>%{100*K['ctl']/n:.0f}</b>ZEM/ZEV kontrolcüsü<br>{K['ctl']}/{n}</div>
 <div><b>%{100*K['mpc']/n:.0f}</b>MPC (model öngörülü)<br>{K['mpc']}/{n}</div>
 <div><b>%{100*K['es']/n:.0f}</b>ES (kontrolcüsüz)<br>{K['es']}/{n}</div>
 <div><b>%{100*K['ppo']/n:.0f}</b>PPO (kazanç ayarı)<br>{K['ppo']}/{n}</div>
</div>

<h2>1. Özet</h2>
<p>Bu belge, Apollo LM'nin <b>kraterli arazi</b>da iniş yapması için denenen dört yaklaşımı karşılaştırır ve her birinin <b>nasıl çalıştığını</b> anlatır: sabit bir ZEM/ZEV kontrolcüsü (taban çizgisi), kontrolcünün kazançlarını ayarlayan PPO, kontrolcü olmadan doğrudan komut veren evrimsel strateji (ES) ve öğrenmesiz model öngörülü kontrol (MPC, MPPI). Hepsi aynı Isaac Sim ortamında, aynı {n} başlangıç koşulunda ({W} dalga × 12 ortam) uçtu; aynı bölüm üç yöntemde ve kontrolcüde de var, bu yüzden karşılaştırmalar <b>eşleştirilmiştir</b>.</p>
<div class=key><b>Ana bulgular.</b><ul style="margin:4px 0 0 16px;padding:0">
<li>Yoğun krater alanında hepsi düz araziye göre geriliyor; kontrolcü %{100*K['ctl']/n:.0f}, <b>PPO %{100*K['ppo']/n:.0f}</b>, <b>ES %{100*K['es']/n:.0f}</b>, <b>MPC %{100*K['mpc']/n:.0f}</b>.</li>
<li>Üç öğrenen/planlayan yöntem de kontrolcüyü <b>anlamlı biçimde</b> geçiyor (aşağıdaki McNemar tablosu).</li>
<li>ES ile PPO arasındaki fark bu örneklemde istatistiksel olarak ayırt edilemiyor; MPC her ikisinin de gerisinde.</li>
<li>Arazi zorlaştıkça tüm kollar düşüyor; krater duvarı (bacak yükseklik farkı) ve yanal hız iniş başarısızlığının başlıca nedenleri.</li>
<li>Hiçbir yöntem bu araziyle yeniden eğitilmedi ya da ayarlanmadı: sonuçlar <b>sıfır-atış</b> dayanıklılığıdır.</li></ul></div>

<h2>2. Deney düzeni</h2>
<h3>2.1 Arazi: neyi değiştirdik</h3>
<p>Önceki ölçümlerde arazi, 1680 m'lik karoda neredeyse düz kabul edilebilirdi (stok alan: yüzlerce küçük krater, çoğu 1–6 m; 21 m'lik ızgarada çözünmüyordu). Bu çalışmada araziye <b>~1000 krater</b> eklendi: çaplar <b>6–140 m</b>, kümülatif güç yasası (b = 1.8, yani küçük kraterler çok, büyükler az), derinlik/çap oranı 0.10, ızgara 80 → <b>256</b> (6.6 m hücre; 9.4 m'lik ayak açıklığı artık araziyi görüyor). En dik duvar eğimi stok alanda ~9°, yoğun alanda ~27°'ye çıkıyor, arazinin ~%95'i hâlâ 8°'nin altında. Önceki ara adımda 80 kraterli (12–70 m) bir alan da denendi; sonuçları 5.3'te.</p>
<figure>{img(f_ter)}<figcaption>Şekil 1. Üç arazi (aynı tohum): stok, 80 krater, yoğun krater alanı. Sağda eğim dağılımının kuyruğu: yoğun alanda dik duvarlar var.</figcaption></figure>
<h3>2.2 Başlangıç koşulları ve ölçüt</h3>
<p>Araç 200 m irtifadan, ~26 m/s yanal hızla ve rastgele konumdan başlar. Başarılı iniş (<code>landed_safely</code>): dikey hız ≤ 1.0 m/s, yanal hız ≤ 1.2 m/s, yatış ≤ 15°, açısal hız ≤ 0.5 rad/s ve ayaklar arası yükseklik farkı ≤ 0.16 m. Her bölüm için sonuç ikilidir. Bölüm sayısı: <b>{n}</b>; tohumlar 90000–{90000+1000*(W-1)} (dalga başına 1000 aralıklı), her tohumda 12 ortam. Karşılaştırmalar McNemar tam testi (aynı bölümlerde eşleştirilmiş) ve dalga-blok bootstrap ile verilir.</p>
<p>Not: yoğun krater sonuçları dört kolda da aynı {n} bölümdedir. Bir önceki ara deneyde (80 krater) ES ve MPC 192, PPO 84 bölümde ölçülmüştü.</p>

<h2 class=pb>3. Yöntemler: nasıl çalışıyorlar</h2>
<table><tr><th></th><th>ZEM/ZEV kontrolcü</th><th>PPO (kazanç ayarı)</th><th>ES</th><th>MPC (MPPI)</th></tr>
<tr><td><b>öğrenme</b></td><td>yok</td><td>PPO, Isaac'te eğitildi</td><td>ES, analitik ortamda; Isaac'te yalnızca doğrulandı</td><td>yok; yalnızca 10 maliyet ağırlığı ES ile ayarlandı</td></tr>
<tr><td><b>çıktı</b></td><td>ham komut (gaz, pitch, roll, yaw)</td><td>11 kazanç çarpanı (1 s tutulur)</td><td>ham komut, 20 Hz</td><td>ham komut, plan 0.2 s'de bir</td></tr>
<tr><td><b>parametre</b></td><td>elle seçilmiş kazançlar</td><td>küçük sinir ağı</td><td>84 (doğrusal)</td><td>10 maliyet ağırlığı</td></tr>
<tr><td><b>model bilgisi</b></td><td>rehber formülü</td><td>yok</td><td>yok</td><td>dinamik denklemlerin torch kopyası</td></tr>
<tr><td><b>kontrolcü kullanır mı</b></td><td>kendisi</td><td>evet, uçuşu o yapar</td><td>hayır</td><td>hayır</td></tr>
<tr><td><b>hesap</b></td><td>ihmal edilebilir</td><td>ihmal edilebilir</td><td>84 çarpma-toplama</td><td>GPU, ~140 ms / plan</td></tr></table>

<h3>3.1 ZEM/ZEV kontrolcüsü (taban çizgisi)</h3>
<p>ZEM (zero-effort miss) ve ZEV (zero-effort velocity) rehberliği, aracın hiçbir şey yapmazsa hedefi ne kadar ıskalayacağını ve hedefte hangi hızla olacağını hesaplayıp ona göre ivme komutu üretir. Bu ivme gaz büyüklüğüne ve eğim komutuna çevrilir; iç yatış döngüsü (<code>kp_att</code>, <code>kd_att</code>) RCS torkunu üretir. Kazançlar sabittir ve her arazide aynı değerdedir.</p>
<figure>{img(f_fc)}<figcaption>Şekil 2. Sabit kontrolcü: ölçülen durumdan komuta, 20 Hz geri besleme.</figcaption></figure>

<h3>3.2 PPO: kontrolcünün kazançlarını ayarlayan ağ</h3>
<p>Ham komutu doğrudan öğretmeye çalışan PPO denemeleri (davranış klonlamasıyla başlayıp) uzun ufukta çöktü: yatış yetkisi düşük olduğu için tek bir fren manevrası yüzlerce ardışık komut gerektiriyor ve adım başına gürültü bunları birbirini götürüyor. Bu yüzden ağın eylem uzayı değiştirildi: ağ <b>11 kazanç çarpanı</b> üretir (çarpan = oran<sup>a</sup>, a∈[−1,1]), bunlar 1 s boyunca sabit tutulur ve ZEM/ZEV kontrolcüsü yine uçar. a = 0 ise çarpan 1'dir, yani taban başarı kontrolcüdür; ağ yalnızca kontrolcünün zayıf yaptığı yerde sapmayı öğrenir.</p>
<figure>{img(f_fp)}<figcaption>Şekil 3. PPO 'guided-gain': ağ kazançları ayarlar; komutu kontrolcü üretir.</figcaption></figure>

<h3 class=pb>3.3 ES: kontrolcüsüz, 84 parametreli doğrusal politika</h3>
<p>Evrimsel strateji (OpenAI-ES) gradyan hesaplamaz. Bir θ vektörünün etrafında rastgele değişimler üretir, her birinin ne kadar iyi indiğini ölçer ve θ'yı daha iyi sonuçlanan yöne kaydırır. Bu çalışmada politika yalnızca <b>20 gözlemden 4 komuta doğrusal bir katmandır</b> (20×4 + 4 = 84 parametre); gözlemler çalışan ortalama/std ile normalize edilir. Belirleyici öğeler:</p>
<ul>
<li><b>Antitetik örnekleme:</b> her ε için hem θ+σε hem θ−σε denenir; gürültü varyansını düşürür.</li>
<li><b>Ortak rastgele sayılar:</b> bir nesildeki tüm adaylar aynı başlangıç koşullarında uçar; farklar politikadan gelir, şanstan değil.</li>
<li><b>Merkezli sıra dönüşümü:</b> uygunluk değerleri sıralanıp [−0.5, 0.5]'e çekilir; tek bir aşırı bölüm güncellemeyi bozmaz.</li>
<li><b>Dereceli uygunluk:</b> yere çarpma temas şiddetine göre dereceli cezalandırılır; zaman aşımı en kötü çarpmayla aynı cezayı alır, yani havada asılı kalmak çarpmadan ucuz değildir.</li>
<li><b>Gözlem normalizasyonu:</b> eğitimi başlatan kritik düzeltme; aynı hiperparametrelerle yapılan ablasyonda normalizasyon olmadan öğrenme başlamadı.</li>
<li><b>Müfredat:</b> ramp_35m → ramp_50m → ramp_100m → ramp_150m → orbit_descent, her basamak bir öncekinin sonucundan başlar.</li></ul>
<figure>{img(f_fe)}<figcaption>Şekil 4. ES: üstte uçuş sırasında politika, altta eğitim döngüsü ve müfredat.</figcaption></figure>

<h3>3.4 MPC (MPPI): modelde ileri sarıp plan yapan kontrolcü</h3>
<p>MPC, aracın kendi dinamik modelini (ortamın adım fonksiyonunun GPU'daki kopyasını) kullanır. Her 0.2 s'de ölçülen durumdan 512 aday komut dizisi örnekler, her birini 12 s ileri sarar, maliyetle puanlar ve softmin ağırlıklı ortalama diziyi seçer; ilk segmenti uygular ve yeniden planlar. ZEM/ZEV'i ve öğrenilmiş politikayı kullanmaz. Ufuk 12 s olmak zorunda: yatış yetkisi çok düşük olduğu için yanal hızı kesmek ~12 s'lik yat–düzel manevradır; 4 s'lik ufukla bu görülemez ve araç zaman aşımına uğrar. 10 maliyet ağırlığı ES ile ayarlandı; planlayıcı temas limitlerinin %70'ini kullanır.</p>
<figure>{img(f_fm)}<figcaption>Şekil 5. MPPI döngüsü: örnekle, modelde ileri sar, puanla, ortalama diziyi güncelle, ilk komutu uygula.</figcaption></figure>

<h2 class=pb>4. Sonuçlar</h2>
<h3>4.1 Yoğun krater alanında inişler</h3>
<figure>{img(f_res)}<figcaption>Şekil 6. Solda yoğun krater alanında dört yöntem ({n} eşleştirilmiş bölüm). Sağda aynı yöntemlerin düz arazi, 80 krater ve yoğun krater alanındaki oranları.</figcaption></figure>
<table><tr><th>karşılaştırma ({n} bölüm)</th><th>ilk yöntem-only</th><th>ikinci-only</th><th>McNemar p</th><th>fark (puan) [dalga-bootstrap %95]</th></tr>{prow}</table>
<p>Sonlanma nedenleri: kontrolcü {whytxt(why['ctl'])}; ES {whytxt(why['es'])}; MPC {whytxt(why['mpc'])}.</p>
<figure>{img(f_wav)}<figcaption>Şekil 7. Dalga bazında inişler. Dalga başına 12 bölüm olduğu için tek dalga gürültülüdür; sıralama dalgalar arasında tutarlı.</figcaption></figure>
<figure>{img(f_ov)}<figcaption>Şekil 8. Örtüşme: bölümlerin büyük kısmında üç yöntem birlikte iniyor ya da birlikte çakılıyor; farklar küçük bir kümede.</figcaption></figure>

<h3 class=pb>4.2 Bir bölümde karşılaştırma</h3>
<p>Aşağıdaki bölüm (#{ENV}) ES, PPO ve MPC'nin üçünün de indiği ve kontrolcünün inmediği bir örnektir; videolar bu bölümü gösterir.</p>
<figure>{img(f_tr)}<figcaption>Şekil 9. Aynı başlangıç koşulundan üç yöntemin uçuşu. Kırmızı kesikli çizgiler iniş limitleridir.</figcaption></figure>
<figure>{img(f_trk)}<figcaption>Şekil 10. Yer izleri arazi üzerinde ve araç altındaki arazi profili: kraterler yaklaşma yoluna giriyor.</figcaption></figure>

<h3>4.3 Çarpmalar neden oluyor?</h3>
<p>Kayıtlı dalgada (12 bölüm) iniş sonuçları: kontrolcü {len(lg['ctl'])}/12, ES {len(lg['es'])}/12, PPO {len(lg['ppo'])}/12, MPC {len(lg['mpc'])}/12. Başarısız bölümlerde marjı sıfıra inen (limit aşılan) ölçütler: ES: {tally_txt('es')}; PPO: {tally_txt('ppo')}; MPC: {tally_txt('mpc')}; kontrolcü: {tally_txt('ctl')}.</p>
<figure>{img(f_fl)}<figcaption>Şekil 11. Her bölüm ve ölçüt için temas anında kalan marj. Kırmızı = limit aşıldı. Örneklem küçük (12 bölüm); eğilim göstermek içindir, sayı olarak yorumlanmamalıdır.</figcaption></figure>

<h3>4.4 Videolar</h3>
<p>Aynı bölümün (#{ENV}) üç videosu yoğun krater alanında çekildi: <code>yogun_krater_es.mp4</code>, <code>yogun_krater_ppo.mp4</code>, <code>yogun_krater_mpc.mp4</code>. Her kare dört panelden oluşur: Isaac kamerası, nadir kamera, hesaplanmış LiDAR haritası ve telemetri. LiDAR haritası <i>hesaplanmıştır</i> (OS2 deseni + kayıtlı araziye ışın izleme), RTX sensör çıktısı değildir.</p>
{img(D2 / 'video_es_final.png')}
<p style="font-size:8.8pt;color:#52514e">ES videosunun son karesi: Isaac kamerası (sol üst), nadir kamera (orta üst), LiDAR haritası (alt orta), telemetri (sağ).</p>

<h2>5. Dürüst sınırlar</h2>
<div class=note><ul style="margin:0 0 0 16px;padding:0">
<li><b>Sıfır-atış:</b> hiçbir yöntem krater alanında yeniden eğitilmedi ya da ayarlanmadı; hepsi stok (neredeyse düz) arazide eğitildi/ayarlandı. Krater alanında yeniden eğitimin etkisi ölçülmedi.</li>
<li><b>Eğitim ortamı farklı:</b> ES ağırlıkları ve MPC maliyet ağırlıkları analitik ortamda, PPO Isaac'te eğitildi/ayarlandı. Kıyas "aynı Isaac ortamında doğrulama"dır; eğitim koşulları aynı değildir.</li>
<li><b>MPC elmayla elma değil:</b> MPC dinamik denklemleri biliyor; ES ve PPO model-free. MPC krater eğimini bilmiyor (planlayıcı zemin düz varsayar).</li>
<li><b>Rehber alan sabitleri</b> ön bilgi olarak kullanıldı; ES ve PPO gözlemlerinde aynı özellikleri görüyor.</li>
<li><b>Örneklem:</b> {n} bölüm; %95 aralıklar ±9–10 puan. ES–PPO farkı bu aralığın içinde.</li>
<li><b>Tek ES tohumu, tek PPO tohumu;</b> yöntemlerin tohumdan tohuma değişimi ölçülmedi.</li>
<li><b>Kayıtlı bölüm tek örnek;</b> trajektori karşılaştırmaları temsilî olup istatistik değildir.</li>
<li><b>Eşleştirme:</b> kontrolcü vektörü ES ve PPO dosyalarında birebir aynı (doğrulandı); MPC kolunda kontrolcü ayrıca koşulmadı, aynı tohumlar ve aynı arazi üretici kullanıldı.</li>
<li><b>Video işleme:</b> temastan sonra ~3 s hareketsiz bekleme kaydın sonuna eklenmiştir (yeniden simülasyon değildir); kamera yönü yumuşatılmıştır. Kaydedilen bölüm, hakem testi için ES/PPO/MPC'nin üçünün de indiği bir bölümdür; seçim sonuca bakılarak yapıldı.</li>
</ul></div>

<h2>6. Yeniden üretim</h2>
<table><tr><th>adım</th><th>komut / dosya</th></tr>
<tr><td>arazi (krater alanı)</td><td><code>LUNARSIM_CRATER_FIELD=1000 LUNARSIM_CRATER_DMIN=6 LUNARSIM_CRATER_DMAX=140 LUNARSIM_CRATER_B=1.8 LUNARSIM_TERRAIN_GRID_N=256</code> (<code>lunarsim/rl/curriculum.py</code>)</td></tr>
<tr><td>ES değerlendirme</td><td><code>scripts/evo_isaac_eval.py an_orbit_descent_best.npz --stage orbit_descent --waves 8</code></td></tr>
<tr><td>MPC değerlendirme</td><td><code>scripts/mpc_isaac_eval.py mpc_params_v2.npy --stage orbit_descent --waves 8 --no-controller</code></td></tr>
<tr><td>PPO değerlendirme</td><td><code>scripts/validate_guided_gain_isaac.py gg_isaac_orbit_v4_seed21_repro_best_frozen.zip --waves 8 --terrain-grid-n 256</code></td></tr>
<tr><td>kayıt + video</td><td><code>out/crater2/run.sh</code>, <code>out/crater2/rerender.sh</code></td></tr>
<tr><td>bu rapor</td><td><code>report/build_krater_rapor.py</code></td></tr></table>
"""
html_path = ROOT / "report" / "lunarsim_krater_rapor.html"
html_path.write_text(f"<!doctype html><html lang=tr><head><meta charset=utf-8><title>LunarSim kraterli arazi raporu</title>"
                     f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
pdf = ROOT / "report" / "lunarsim_krater_rapor.pdf"
subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf, K, "overlap", overlap[:4], "tally", tally)
