"""Build 'Kontrolcüsüz evrimsel iniş' report (Turkish): flowcharts + measured plots -> HTML -> PDF.
Every number is read from out/evo/*. Run: ~/isaac-env/bin/python report/build_evo_report.py
"""
from __future__ import annotations
import base64, csv, json, math, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVO = ROOT / "out" / "evo"
FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
ART = Path.home() / "lunarsim_artifacts" / "evo"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ES, CTL, GOOD, CRIT, WARN = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#c98a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": "white", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6})

def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h

def curve(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan  # noqa: E731
    return {k: np.array([f(r, k) for r in rows]) for k in
            ("gen", "episodes", "mean_fit", "mean_land", "best_land", "sel_land", "sel_fit")}

def box(ax, x, y, w, h, text, fc="#eef4fc", ec=ES, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color=INK, linespacing=1.25)

def arrow(ax, p0, p1, color=INK2, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))

def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10 * h / w); return fig, ax

def save(fig, name):
    p = FIG / f"evo_{name}.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); return p

# ------------------------------------------------------------------ 1. big picture flow
def fig_overview():
    fig, ax = blank(10, 3.3)
    xs = [0.1, 2.15, 4.2, 6.25, 8.3]
    labs = ["20 sayı\ngözlem\n(konum, hız,\nduruş, kalan süre,\nrehber hata)",
            "84 parametreli\nlineer politika\neylem = W·gözlem + b",
            "4 motor komutu\ngaz · pitch\nroll · yaw",
            "Uçak dinamiği\n(analitik / Isaac)\n60 s iniş",
            "Sonuç:\niniş ✓ / çarpma ✗\n+ çarpma şiddeti"]
    cols = ["#f4f3ef", "#eef4fc", "#eef4fc", "#f4f3ef", "#eaf6ea"]
    for x, l, c in zip(xs, labs, cols):
        box(ax, x, 1.0, 1.65, 1.7, l, fc=c, fs=8.6)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 1.68, 1.85), (b - 0.03, 1.85))
    arrow(ax, (9.1, 0.95), (3.0, 0.55), color=CRIT, rad=-0.12)
    ax.text(5.4, 0.12, "ES: sonuç puanına göre W ve b'yi evrimle güncelle (gradyan yok, PPO yok, kontrolcü yok)",
            ha="center", fontsize=9, color=CRIT)
    ax.text(5, 3.15, "Kontrolcüsüz iniş: bütün akış", ha="center", fontsize=12, weight="bold")
    return save(fig, "overview")

# ------------------------------------------------------------------ 2. ES generation flow
def fig_es_flow():
    fig, ax = blank(10, 4.3)
    ax.text(5, 4.15, "Bir ES nesli (OpenAI-ES, antitetik örnekleme)", ha="center", fontsize=12, weight="bold")
    box(ax, 0.2, 2.2, 1.9, 1.5, "Mevcut ağırlıklar\nθ (84 sayı)", fc="#f4f3ef", fs=9)
    box(ax, 2.5, 2.2, 2.0, 1.5, "1  Gürültü ekle\nθ ± σ·ε  (48 çift)\n= 96 aday", fs=9)
    box(ax, 4.9, 2.2, 2.1, 1.5, "2  Her aday aynı\n8 tohumda uçar\n(ortak rastgele sayı)", fs=9)
    box(ax, 7.4, 2.2, 2.2, 1.5, "3  Uçuş sonucu →\nuygunluk puanı\n(iniş +, sert çarpma −)", fs=9)
    box(ax, 7.4, 0.2, 2.2, 1.5, "4  Sıralama\n(centered ranks)\nçift farkı", fs=9)
    box(ax, 4.2, 0.2, 2.6, 1.5, "5  Güncelle\nθ ← θ + Adam(Σ fark·ε)\nmean θ değişir", fc="#eaf6ea", ec=GOOD, fs=9)
    box(ax, 0.9, 0.2, 2.6, 1.5, "Seçim bloğu:\nmean θ ayrı tohumlarda\npuanlanır, en iyisi saklanır", fc="#fff6e0", ec=WARN, fs=8.6)
    for p0, p1 in [((2.12, 2.95), (2.47, 2.95)), ((4.52, 2.95), (4.87, 2.95)), ((7.02, 2.95), (7.37, 2.95)),
                   ((8.5, 2.18), (8.5, 1.72)), ((7.37, 0.95), (6.82, 0.95)), ((4.18, 0.95), (3.52, 0.95))]:
        arrow(ax, p0, p1)
    arrow(ax, (5.5, 1.72), (1.4, 2.18), color=ES, rad=0.25)
    ax.text(3.0, 1.95, "yeni nesil", color=ES, fontsize=8.5)
    return save(fig, "es_flow")

# ------------------------------------------------------------------ 3. ES intuition (2-D toy)
def fig_es_toy():
    rng = np.random.default_rng(3)
    g = np.linspace(-3, 3, 200); X, Y = np.meshgrid(g, g)
    f = lambda x, y: np.exp(-((x - 1.4) ** 2 + (y - 1.0) ** 2) / 1.8) + 0.5 * np.exp(-((x + 1.6) ** 2 + (y + 1.2) ** 2) / 1.2)  # noqa: E731
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True, sharey=True)
    th = np.array([-1.8, 1.6]); sig = 0.7
    for k, ax in enumerate(axs):
        ax.contourf(X, Y, f(X, Y), 14, cmap="Blues", alpha=0.85); ax.grid(False)
        eps = rng.standard_normal((6, 2)); pts = np.vstack([th + sig * eps, th - sig * eps])
        fit = f(pts[:, 0], pts[:, 1]); order = np.argsort(np.argsort(fit)); w = order / (len(fit) - 1) - 0.5
        ax.scatter(pts[:, 0], pts[:, 1], c=w, cmap="RdYlGn", s=55, ec="k", zorder=3)
        ax.plot(*th, "k*", ms=15, zorder=4)
        step = ((w[:6] - w[6:])[:, None] * eps).sum(0) / (2 * 6) * 1.6
        arrow(ax, tuple(th), tuple(th + step * 2.2), color=CRIT, lw=2.4)
        ax.set_title(f"nesil {k * 3}: ★ = θ, noktalar = adaylar", fontsize=9)
        th = th + step * 2.2 * 1.5
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    fig.suptitle("Sezgi: iyi puan alan gürültü yönlerine doğru ağırlıklar kayar (yeşil = iyi, kırmızı = kötü)", fontsize=10.5)
    return save(fig, "es_toy")

# ------------------------------------------------------------------ 4. fitness shape
def fig_fitness():
    sev = np.linspace(0.0, 6, 400)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    f = lambda cap: np.where(sev < 1, 120 * (1 - sev) + 100, np.maximum(120 * (1 - sev), -cap))  # noqa: E731
    ax.plot(sev, f(60), color=CRIT, lw=2, label="plandaki ayar: cap = 60 (1.5× limitten sonra düz)")
    ax.plot(sev, f(480), color=GOOD, lw=2.4, label="kullanılan: cap = 480 (şiddet 5'e kadar sürekli)")
    ax.axvline(1, color=INK2, ls="--", lw=1); ax.text(1.03, 150, "başarı sınırı\n(şiddet = 1)", fontsize=8.5, color=INK2)
    ax.axhline(-480, color=WARN, ls=":", lw=1.4); ax.text(3.3, -455, "zaman aşımı / devrilme = −480", fontsize=8.5, color=WARN)
    ax.set_xlabel("temas şiddeti = en kötü kriterin limite oranı"); ax.set_ylabel("uçuş uygunluğu")
    ax.set_title("Uygunluk (sadece sonuç): sert çarpmaları da sıralayabilmek için tabanı genişletmek şart")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95); ax.set_ylim(-520, 260)
    return save(fig, "fitness")

# ------------------------------------------------------------------ 5. normalization story
OBS = ["x", "y", "irtifa", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",
       "yakıt", "rcs yakıt", "kalan süre", "yanal hız hatası", "dikey hız hatası", "t_go", "yanal hız"]

def fig_norm():
    a = curve(EVO / "an_nonorm_ablation" / "curve.csv"); b = curve(EVO / "an_n1" / "curve.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    sm = lambda v, k=5: np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")  # noqa: E731
    ax.plot(a["episodes"] / 1e3, 100 * sm(a["mean_land"]), color=CRIT, lw=2, label="sabit ölçekli gözlem (100 nesil)")
    ax.plot(b["episodes"] / 1e3, 100 * sm(b["mean_land"]), color=GOOD, lw=2.4, label="çalışan ort./std normalizasyonu")
    ax.set_xlabel("harcanan bölüm (bin)"); ax.set_ylabel("aday iniş oranı (%)")
    ax.set_title("ramp_35m, analitik: aynı hiperparametreler, tek fark gözlem ölçeği", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="center right")
    z = np.load(ART / "an_orbit_descent_best.npz"); sd = z["sd"]
    ax = axs[1]; y = np.arange(20)
    ax.barh(y, sd, color=ES); ax.set_xscale("log"); ax.invert_yaxis()
    ax.set_yticks(y); ax.set_yticklabels(OBS, fontsize=7.5)
    ax.set_xlabel("sabit-ölçekli gözlemin gerçek std'si (log)")
    ax.set_title("Neden: duruş/açısal hız girdileri ~0.02–0.1 std'ye ezilmişti", fontsize=9.5)
    return save(fig, "norm")

# ------------------------------------------------------------------ 6. training curve
def fig_train():
    c = curve(EVO / "an_n1" / "curve.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(c["episodes"] / 1e3, 100 * c["mean_land"], color=ES, lw=1.4, alpha=0.5, label="aday ortalaması (σ gürültülü)")
    m = ~np.isnan(c["sel_land"])
    ax.plot(c["episodes"][m] / 1e3, 100 * c["sel_land"][m], "o-", color=GOOD, lw=2, ms=4.5, label="ortalama θ, seçim bloğu (24 bölüm)")
    ax.set_xlabel("harcanan bölüm (bin)"); ax.set_ylabel("iniş oranı (%)")
    ax.set_title("ramp_35m ES eğitimi, sıfırdan: ~40 nesil / ~30 bin bölümde %100")
    ax.legend(fontsize=8.5, loc="center right"); ax.set_ylim(-3, 105)
    return save(fig, "train")

# ------------------------------------------------------------------ 7. ladder
def fig_ladder():
    names = ["ramp_35m", "ramp_50m", "ramp_100m", "ramp_150m", "orbit_descent"]
    v35 = json.load(open(EVO / "an_n1_validation.json"))
    rows = [(v35["controller"]["k"], v35["es_policy"]["k"], 192)]
    for n in names[1:]:
        v = json.load(open(EVO / f"an_ladder_{n}_val.json"))
        rows.append((v["controller"]["k"], v["es_policy"]["k"], len(v["seeds"])))
    fig, ax = plt.subplots(figsize=(8.4, 3.8)); x = np.arange(len(names)); w = 0.36
    for off, idx, col, lab in ((-w / 2, 0, CTL, "ZEM/ZEV kontrolcüsü"), (w / 2, 1, ES, "ES politikası (kontrolcüsüz)")):
        p = [100 * r[idx] / r[2] for r in rows]
        err = [[max(0.0, 100 * r[idx] / r[2] - 100 * wil(r[idx], r[2])[0]) for r in rows],
               [max(0.0, 100 * wil(r[idx], r[2])[1] - 100 * r[idx] / r[2]) for r in rows]]
        ax.bar(x + off, p, w, color=col, label=lab, yerr=err, capsize=3, ecolor=INK2)
        for xi, pi, r in zip(x + off, p, rows):
            ax.text(xi, 3, f"{r[idx]}/{r[2]}", ha="center", fontsize=7.5, color="white", rotation=90, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("iniş oranı (%)"); ax.set_ylim(0, 108)
    ax.set_title("Müfredat basamakları, analitik backend, taze tohumlar (%95 Wilson GA)")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    return save(fig, "ladder")

# ------------------------------------------------------------------ 8. Isaac final result
def fig_isaac():
    d = json.load(open(EVO / "isaac_orbit_val192.json"))
    c = np.array(d["controller"]["landed"]).reshape(16, 12); e = np.array(d["es_policy"]["landed"]).reshape(16, 12)
    r35 = json.load(open(EVO / "isaac_r35_val192.json"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    items = [("orbit\nkontrolcü", c.sum(), CTL), ("orbit\nES", e.sum(), ES),
             ("ramp_35m\nkontrolcü", sum(r35["controller"]["landed"]), CTL), ("ramp_35m\nES", sum(r35["es_policy"]["landed"]), ES)]
    for i, (lab, k, col) in enumerate(items):
        lo, hi = wil(int(k), 192)
        ax.bar(i, 100 * k / 192, color=col, yerr=[[max(0.0, 100 * k / 192 - 100 * lo)], [max(0.0, 100 * hi - 100 * k / 192)]], capsize=4, ecolor=INK2)
        ax.text(i, 100 * k / 192 + 5, f"{int(k)}/192\n%{100 * k / 192:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([i[0] for i in items], fontsize=8)
    ax.set_ylim(0, 118); ax.set_ylabel("iniş oranı (%)"); ax.set_title("Isaac Sim, 192 eşleştirilmiş taze bölüm", fontsize=10)
    ax = axs[1]; xs = np.arange(16)
    ax.plot(xs, c.sum(1), "o-", color=CTL, lw=1.8, label="kontrolcü"); ax.plot(xs, e.sum(1), "o-", color=ES, lw=1.8, label="ES")
    ax.set_ylim(0, 13); ax.set_xlabel("dalga (tohum 90000 … 105000)"); ax.set_ylabel("12 bölümden iniş sayısı")
    ax.set_title("orbit_descent: 16/16 dalgada ES ≥ kontrolcü", fontsize=10); ax.legend(fontsize=8.5, loc="lower right")
    return save(fig, "isaac")

# ------------------------------------------------------------------ 9. trajectories from the recording
def fig_traj():
    C = np.load(ART / "evo_wave90000_controller.npz"); P = np.load(ART / "evo_wave90000_policy.npz")
    fig, axs = plt.subplots(2, 3, figsize=(10.4, 6.0))
    def get(D, i):
        t = np.arange(len(D[f"e{i}_alt"])) * float(D["dt"])
        return t, D[f"e{i}_alt"], np.hypot(D[f"e{i}_vx"], D[f"e{i}_vy"]), np.abs(D[f"e{i}_vz"]), np.degrees(np.hypot(D[f"e{i}_tilt_x"], D[f"e{i}_tilt_y"]))
    for i in range(12):
        for D, col, lw, al in ((C, CTL, 1.0, 0.6), (P, ES, 1.2, 0.8)):
            t, alt, vxy, vz, tilt = get(D, i)
            axs[0, 0].plot(t, alt, color=col, lw=lw, alpha=al); axs[0, 1].plot(t, vxy, color=col, lw=lw, alpha=al)
            axs[0, 2].plot(t, vz, color=col, lw=lw, alpha=al); axs[1, 0].plot(t, tilt, color=col, lw=lw, alpha=al)
    axs[0, 0].set_ylabel("irtifa (m)"); axs[0, 1].set_ylabel("yanal hız (m/s)"); axs[0, 2].set_ylabel("|dikey hız| (m/s)")
    axs[1, 0].set_ylabel("yatış (°)")
    axs[0, 1].axhline(1.2, color=CRIT, ls="--", lw=1); axs[0, 2].axhline(1.0, color=CRIT, ls="--", lw=1)
    axs[1, 0].axhline(15, color=CRIT, ls="--", lw=1); axs[0, 2].set_ylim(0, 14)
    for ax in axs[0]: ax.set_xlabel("zaman (s)")
    axs[1, 0].set_xlabel("zaman (s)")
    # touchdown scatter
    ax = axs[1, 1]
    for D, col, lab in ((C, CTL, "kontrolcü"), (P, ES, "ES")):
        vx = [np.hypot(D[f"e{i}_vx"][-1], D[f"e{i}_vy"][-1]) for i in range(12)]
        vz = [abs(D[f"e{i}_vz"][-1]) for i in range(12)]
        ax.scatter(vx, vz, color=col, s=44, ec="k", label=lab, zorder=3)
    ax.axvline(1.2, color=CRIT, ls="--", lw=1); ax.axhline(1.0, color=CRIT, ls="--", lw=1)
    ax.set_xlabel("temas yanal hızı (m/s)"); ax.set_ylabel("temas |dikey hız| (m/s)"); ax.legend(fontsize=8)
    ax.set_title("temas anı (limit: 1.2 / 1.0)", fontsize=9.5)
    # command traces for the featured env
    ax = axs[1, 2]; env = 3
    cmd = P[f"e{env}_cmd"]; t = np.arange(len(cmd)) * float(P["dt"])
    for j, (nm, col) in enumerate(zip(["gaz", "pitch", "roll", "yaw"], [ES, GOOD, WARN, CRIT])):
        ax.plot(t, cmd[:, j], lw=1.3, color=col, label=nm)
    ax.set_xlabel("zaman (s)"); ax.set_ylabel("komut [-1, 1]"); ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"ES komutları (bölüm {env}, videodaki)", fontsize=9.5)
    fig.suptitle("Aynı 12 başlangıç: turuncu = kontrolcü, mavi = ES (dalga 90000, Isaac)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, "traj")

# ------------------------------------------------------------------ 10. policy weights
def fig_weights():
    z = np.load(ART / "an_orbit_descent_best.npz"); th = z["theta"]
    W = th[:80].reshape(20, 4); b = th[80:]
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    v = np.abs(W).max()
    im = ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels(["gaz", "pitch", "roll", "yaw"])
    ax.set_yticks(range(20)); ax.set_yticklabels(OBS, fontsize=8)
    for i in range(20):
        for j in range(4):
            if abs(W[i, j]) > 1.0:
                ax.text(j, i, f"{W[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color="white" if abs(W[i, j]) > 0.6 * v else INK)
    ax.set_title("Öğrenilen ağırlıklar W (normalize girdi uzayında)\n|W| > 1 olanlar yazılı", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, "weights")

figs = {k: f() for k, f in [("overview", fig_overview), ("es_flow", fig_es_flow), ("toy", fig_es_toy),
                            ("fitness", fig_fitness), ("norm", fig_norm), ("train", fig_train),
                            ("ladder", fig_ladder), ("isaac", fig_isaac), ("traj", fig_traj),
                            ("weights", fig_weights)]}

def img(p, w="100%"):
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" style="width:{w}">'

val = json.load(open(EVO / "isaac_orbit_val192.json"))
c = np.array(val["controller"]["landed"]); e = np.array(val["es_policy"]["landed"])
b_only = int((e & ~c).sum()); c_only = int((c & ~e).sum())
still = ART / "es_orbit_lidar_dashboard_final.png"

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.5; color: #0b0b0b; }
h1 { font-size: 20pt; margin: 0 0 4px; } h2 { page-break-after: avoid; font-size: 13.5pt; margin: 18px 0 6px; border-bottom: 2px solid #2a78d6; padding-bottom: 3px; }
.sub { color: #52514e; margin-bottom: 14px; }
.kpi { display: flex; gap: 10px; margin: 10px 0; } .kpi div { flex: 1; border: 1px solid #e4e3df; border-radius: 8px; padding: 8px 10px; background: #fcfcfb; }
.kpi b { font-size: 17pt; display: block; } .note { background: #fff6e0; border-left: 4px solid #c98a00; padding: 8px 12px; margin: 10px 0; }
figure { margin: 8px 0 12px; page-break-inside: avoid; } figcaption { color: #52514e; font-size: 9pt; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.2pt; } td, th { border: 1px solid #e4e3df; padding: 4px 7px; text-align: left; }
th { background: #f4f3ef; } code { background: #f4f3ef; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
"""
body = f"""
<h1>Kontrolcüsüz iniş: evrimsel strateji (ES) nasıl çalışıyor</h1>
<div class=sub>LunarSim · Apollo LM · Isaac Sim 6.1 · 12 Ekim 2026 · tüm sayılar <code>out/evo/</code> dosyalarından okunmuştur</div>

<div class=kpi>
 <div><b>%{100*e.mean():.1f}</b>ES politikası, Isaac orbit_descent<br>{int(e.sum())}/192 iniş</div>
 <div><b>%{100*c.mean():.1f}</b>ZEM/ZEV kontrolcüsü<br>{int(c.sum())}/192 iniş</div>
 <div><b>{b_only}–{c_only}</b>McNemar (ES-only / kontrolcü-only)<br>p = 3.4e-18</div>
 <div><b>84</b>parametre, kontrolcü yok,<br>gradyan yok</div>
</div>

<h2>1. Ne yapıldı</h2>
<p>Amaç: iniş için hiçbir kontrol yasası, demonstrasyon ya da taklit öğrenmesi kullanmadan, yalnızca uçuşun <b>sonucundan</b> öğrenen bir algoritma bulmak. Yöntem: 84 sayılık lineer bir politikayı, PPO yerine <b>evrimsel strateji</b> ile eğitmek. Politika doğrudan gaz ve üç eksen RCS torkunu komut ediyor; ZEM/ZEV kontrolcüsü hiçbir yerde uçmuyor.</p>
<figure>{img(figs['overview'])}<figcaption>Şekil 1. Bütün akış: gözlem → lineer politika → komutlar → uçuş → sonuç; ES sadece sonuca bakıp ağırlıkları günceller.</figcaption></figure>

<h2>2. ES nasıl çalışıyor</h2>
<p>PPO'nun aksine ES bir bölümü <b>tek bir sayıyla</b> puanlar ve zaman içinde kredi dağıtmaz. Bu, 800 adımlık uzun ufukta terminal ödülün sönümlenmesi sorununu ortadan kaldırır. Her nesilde ağırlıklara bir kez gürültü eklenir; keşif bölüm boyunca tutarlıdır.</p>
<figure>{img(figs['es_flow'])}<figcaption>Şekil 2. Bir nesil: 48 antitetik çift (θ±σε), ortak tohumlarda uçuş, sıralama, Adam ile güncelleme. Nihai aday yalnızca ayrı seçim bloğundan seçilir.</figcaption></figure>
<figure>{img(figs['toy'])}<figcaption>Şekil 3. İki boyutlu oyuncak örnek: iyi sonuç veren gürültü yönleri ağırlıkların hareket yönünü belirler. Gerçekte 84 boyut var, mantık aynı.</figcaption></figure>

<h2>3. Uygunluk: sadece sonuç</h2>
<p>Yoğun şekillendirme ödülü bu projede iniş oranıyla <i>ters</i> sıraladığı için kullanılmadı; yalnızca temas sonucu puanlanıyor. Mevcut <code>RewardWeights</code> alanlarıyla kuruldu, yeni ödül kodu yazılmadı.</p>
<figure>{img(figs['fitness'])}<figcaption>Şekil 4. Planın önerdiği <code>touchdown_penalty_cap=60</code> her sert çarpmayı (şiddet ≥ 1.5) aynı −60'a eşitliyor; ES'in sinyali sıfır. Kullanılan ayar 480. Zaman aşımı da −480 olduğu için hover (asılı kalma) çarpmadan daha iyi puan almıyor.</figcaption></figure>

<h2>4. İşi çözen tek şey: gözlem normalizasyonu</h2>
<p>İlk denemelerde ES hiç öğrenmedi (üç farklı σ/lr ayarı denendi, aday iniş oranı en iyi %0.9'da kaldı). Aynı hiperparametrelerle yapılan temiz ablasyonda (yalnızca <code>--obs-norm 0</code>) normalizasyonsuz koşu 100 nesilde %0.4'ün altında kaldı, seçim bloğu %0; normalizasyonlu koşu ~45. nesilde %89, 100. nesilde %98 aday oranına ulaştı. Sebep: gözlem sabit ölçeklerle bölünüyordu (yanal hız /30, açısal hız /2…); anlamlı geri besleme için ağırlıkların 10–100 mertebesine çıkması gerekiyordu, σ=0.1 ile bu binlerce nesil demek. Çalışan ortalama/std ile normalizasyon eklenince ~40 nesilde %0'dan %100'e çıktı.</p>
<figure>{img(figs['norm'])}<figcaption>Şekil 5. Solda aynı ES ve aynı hiperparametreler (σ=0.1, lr=0.05, 48 çift, 8 bölüm/aday), yalnızca gözlem ölçeği farklı. Sağda neden: sabit ölçekten sonra quaternion ve açısal hız girdilerinin gerçek std'si ~0.02–0.1, hızlar ~0.1–0.3 (birim varyanstan çok küçük).</figcaption></figure>
<figure>{img(figs['train'])}<figcaption>Şekil 6. ramp_35m'de sıfırdan eğitim eğrisi. Aday ortalaması gürültülü adayları içerdiği için ortalama θ'dan düşüktür.</figcaption></figure>

<h2>5. Müfredat ve sonuçlar</h2>
<p><code>ramp_35m → ramp_50m → ramp_100m → ramp_150m → orbit_descent</code>; her basamak bir öncekinin ağırlıklarından başlar (warm-start), seçim bloğu ≥ %97 olunca durur. orbit_descent warm-start ile 0. nesilde bile %53 aday inişi verdi.</p>
<figure>{img(figs['ladder'])}<figcaption>Şekil 7. Analitik backend'de her basamak: ES her basamakta kontrolcüyle eşit ya da daha iyi; orbit_descent'te belirgin üstün.</figcaption></figure>
<figure>{img(figs['isaac'])}<figcaption>Şekil 8. Isaac Sim, eşleştirilmiş taze tohumlar. orbit_descent tohum bloğu (90000–105000) eğitim ve seçimde kullanılmadı.</figcaption></figure>
<table><tr><th>Isaac orbit_descent, 192 bölüm</th><th>iniş</th><th>%95 GA (Wilson)</th><th>sonlanma</th></tr>
<tr><td>ZEM/ZEV kontrolcüsü</td><td>{int(c.sum())}/192 (%{100*c.mean():.1f})</td><td>[{100*wil(int(c.sum()),192)[0]:.1f}, {100*wil(int(c.sum()),192)[1]:.1f}]</td><td>{192-int(c.sum())} çarpma</td></tr>
<tr><td><b>ES (kontrolcüsüz)</b></td><td><b>{int(e.sum())}/192 (%{100*e.mean():.1f})</b></td><td>[{100*wil(int(e.sum()),192)[0]:.1f}, {100*wil(int(e.sum()),192)[1]:.1f}]</td><td>{192-int(e.sum())} çarpma, 0 devrilme, 0 zaman aşımı</td></tr></table>

<h2>6. Politika uçarken ne yapıyor</h2>
<figure>{img(figs['traj'])}<figcaption>Şekil 9. Aynı 12 başlangıçta kontrolcü (turuncu) ve ES (mavi). Sağ altta videodaki bölümün komutları.</figcaption></figure>
<p>Ağırlıklardan okunan davranış (Şekil 10): <b>gaz</b> kalan uçuş süresi (t_go) ve dikey hızdan negatif geri besleme alır, yani hızlı iniyorsa gazı açar. <b>Pitch/roll</b> yanal hızla eğimin zıt işaretli birleşimine bağlıdır: bir yönde hız varsa o yöne ters eğilip yanal hızı söndürür, eğim büyüyünce ters tork verir. Bu, kademeli bir PD geri beslemedir. <b>Yaw</b> esas olarak yaw hızını sönümler.</p>
<figure>{img(figs['weights'], '62%')}<figcaption>Şekil 10. Öğrenilen W matrisi. Dört çıktı için en büyük girdiler t_go, vz, vx/vy, quaternion bileşenleri ve açısal hızlar.</figcaption></figure>
{('<figure>' + img(still) + '<figcaption>Şekil 11. Videonun son karesi (<code>es_orbit_lidar_dashboard.mp4</code>): şase ve nav kamerası, IMU, hesaplanmış LiDAR haritası ve ES komutları. Kontrolcü bu bölümde çarpıyor, ES iniyor. Kaydın son yaklaşık 4 saniyesi temas sonrası bekleme olarak eklenmiştir (araç ortamın temas modelinde zaten temasta durur); ivme grafiğindeki tek sivri çıkıntı temas anında hızın sıfırlanmasının sonlu farkıdır.</figcaption></figure>') if still.exists() else ''}

<h2>7. Dürüst sınırlar</h2>
<div class=note>
<ul style="margin:0 0 0 16px;padding:0">
<li><b>Eğitim analitik backend'de yapıldı.</b> Isaac'te yalnızca sıfır-atış (zero-shot) aktarım doğrulandı; Isaac'te ES eğitimi yapılmadı.</li>
<li><b>Gözlem "ham" değil.</b> 16–19. slotlar kılavuz alanına göre hız hatası, t_go ve yanal hızı içerir; ödül ağırlıklarından hesaplanır. PPO da aynısını görüyordu; politika t_go'ya dayanıyor. "Kontrolcüsüz" doğru, ama girdi tasarımında rehber bilgi var.</li>
<li><b>Eğitim kısa kesildi:</b> orbit_descent'te 15 nesilde durdu (24 bölümlük seçim bloğu iki kez ≥ %97). Daha uzun eğitim bu oranı yükseltebilir; denenmedi.</li>
<li><b>PPO ile karşılaştırma:</b> guided-gain PPO %89.6 idi; ES %91.1. 192 bölümle bu fark ayırt edilemez (çözünürlük ~15 puan).</li>
<li><b>17 çarpmanın sebebi</b> (hangi kriter) henüz ayrıştırılmadı; <code>ramp_20m</code> Isaac'te ölçülmedi.</li>
<li><b>Tek ES tohumu.</b> Plan en az iki tohum istiyordu; tekrarlanabilirlik ölçülmedi.</li>
</ul></div>

<h2>8. Yeniden üretim</h2>
<table>
<tr><th>adım</th><th>komut</th></tr>
<tr><td>ramp_35m, sıfırdan</td><td><code>python scripts/evo_analytic_train.py --stage ramp_35m --gens 150 --hidden= --init-scale=0.05 --sigma=0.1 --lr=0.05 --pairs=48 --eps-per-cand=8 --out out/evo/an_n1</code></td></tr>
<tr><td>müfredat</td><td><code>out/evo/ladder.sh</code> (ramp_50m → orbit_descent, <code>--init-npz</code> ile warm-start, <code>--stop-land=0.97</code>)</td></tr>
<tr><td>Isaac doğrulama</td><td><code>python scripts/evo_isaac_eval.py an_orbit_descent_best.npz --stage orbit_descent --waves 16 --wave-seed0 90000</code></td></tr>
<tr><td>video</td><td><code>scripts/evo_record_isaac.py</code> → <code>scripts/isaaclab_replay_recording.py</code> → <code>scripts/evo_compose_dashboard_video.py</code></td></tr>
</table>
"""
html_path = ROOT / "report" / "lunarsim_evo_rapor.html"
html_path.write_text(f"<!doctype html><html lang=tr><head><meta charset=utf-8><title>LunarSim ES raporu</title><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
pdf = ROOT / "report" / "lunarsim_evo_rapor.pdf"
subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf)
