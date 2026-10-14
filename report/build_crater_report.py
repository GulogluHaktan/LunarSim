"""3-way comparison on crater terrain: ES vs PPO(guided-gain) vs MPC (+ ZEM/ZEV controller)."""
import json, math, re, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = os.environ.get("CRATER_DIR", "out/crater")
CRATER_DESC = os.environ.get("CRATER_DESC", "• Araziye 12–70 m çaplı ~80 krater eklendi (derinlik/çap 0.10, en dik duvar ≈9°), ızgara 80→160.")
OUT = os.environ.get("CRATER_PDF", "report/lunarsim_krater_karsilastirma.pdf")
ENV = int(open(f"{C}/env_picked").read()) if os.path.exists(f"{C}/env_picked") else 3
TAG = os.environ.get("CRATER_TAG", "Kraterli arazide")
es = json.load(open(f"{C}/es.json")); mpc = json.load(open(f"{C}/mpc.json"))
ctrl = np.array(es["controller"]["landed"], bool); E = np.array(es["es_policy"]["landed"], bool)
M = np.array(mpc["mpc"]["landed"], bool)
if os.path.exists(f"{C}/ppo.json"):
    d = json.load(open(f"{C}/ppo.json")); P = np.array(d["candidate_landed"], bool); Pc = np.array(d["baseline_landed"], bool)
else:   # partial: parse per-wave log lines
    P = []; Pc = []
    for l in open(f"{C}/ppo.log"):
        m = re.search(r"controller (\d+)/12\s+policy (\d+)/12", l)
        if m: Pc.append(int(m[1])); P.append(int(m[2]))
    P = np.array(P); Pc = np.array(Pc)
    W = len(P); sel = slice(0, W * 12)
    ctrl, E, M = ctrl[sel], E[sel], M[sel]
    P = int(P.sum()); Pn = W * 12
    P = ("count", P, Pn)
n = len(E)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
def mcn(a, b):
    x = int((a & ~b).sum()); y = int((b & ~a).sum()); N = x + y
    if N == 0: return x, y, 1.0
    k = min(x, y); return x, y, min(1.0, 2 * sum(math.comb(N, i) for i in range(k + 1)) / 2 ** N)
flat = {"Kontrolcü": (101, 192), "MPC": (139, 192), "PPO": (172, 192), "ES": (175, 192)}
rows = [("Kontrolcü (ZEM/ZEV)", int(ctrl.sum()), n), ("MPC (MPPI)", int(M.sum()), n)]
if isinstance(P, tuple): rows.append(("PPO (guided-gain)", P[1], P[2]))
else: rows.append(("PPO (guided-gain)", int(P.sum()), n))
rows.append(("ES (kontrolcüsüz)", int(E.sum()), n))
col = ["#888", "#e8a33d", "#4a90d9", "#3fbf7f"]
with PdfPages(OUT) as pdf:
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    fig.text(.07, .95, f"{TAG} üç yöntem: ES · PPO · MPC", fontsize=18, weight="bold")
    fig.text(.07, .92, "Isaac Sim, orbit_descent, eğitimin görmediği tohumlar, aynı bölümler üç kolda da uçtu", fontsize=10, color="#555")
    a1 = fig.add_axes([.1, .55, .82, .32])
    for i, (nm, k, N) in enumerate(rows):
        lo, hi = wilson(k, N); a1.bar(i, 100 * k / N, color=col[i], yerr=[[100 * k / N - 100 * lo], [100 * hi - 100 * k / N]], capsize=5)
        a1.text(i, 100 * k / N + 6, f"{k}/{N}\n%{100*k/N:.1f}", ha="center", fontsize=10)
        fk, fn = list(flat.values())[[0, 1, 2, 3][i]]
        a1.plot([i - .4, i + .4], [100 * fk / fn] * 2, "k--", lw=1)
    a1.set_xticks(range(4)); a1.set_xticklabels([r[0] for r in rows], fontsize=9); a1.set_ylim(0, 112)
    a1.set_ylabel("iniş oranı (%)"); a1.set_title("Kraterli arazi (çubuk) ve düz arazi (kesik çizgi, önceki ölçüm, 192 bölüm)", fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)
    t = "Eşleştirilmiş karşılaştırmalar (McNemar, tam test):\n"
    allv = {"Kontrolcü": ctrl, "MPC": M, "ES": E}
    for a_, b_ in (("ES", "Kontrolcü"), ("MPC", "Kontrolcü"), ("ES", "MPC")):
        x, y, p = mcn(allv[a_], allv[b_]); t += f"  {a_} vs {b_}: yalnız {a_} {x} · yalnız {b_} {y} · p={p:.1e}\n"
    if not isinstance(P, tuple):
        for nm, v in (("ES", E), ("MPC", M), ("Kontrolcü", ctrl)):
            x, y, p = mcn(P, v); t += f"  PPO vs {nm}: yalnız PPO {x} · yalnız {nm} {y} · p={p:.1e}\n"
    else:
        t += f"  (PPO değerlendirmesi zaman kısıtıyla ilk {P[2]//12} dalgada ({P[2]} bölüm) kesildi; tüm kollar aynı bölümlerde kıyaslandı.)\n"
    fig.text(.08, .48, t, fontsize=10, va="top", family="monospace")
    fig.text(.08, .30, "Ne değişti?\n• Araziye 12–70 m çaplı ~80 krater eklendi (derinlik/çap 0.10, en dik duvar ≈9°), ızgara 80→160.\n"
             "• Hiçbir politika bu araziyle yeniden eğitilmedi / ayarlanmadı: sıfır-atış testi.\n"
             "• MPC dinamikleri düz zemin varsayar; krater eğimini bilmez. ES/PPO yalnız gözlemden öğrenmişti.\n"
             "• Aynı bölüm kümesi (tohum 90000+), kontrolcü vektörü tüm dosyalarda aynı.", fontsize=10, va="top")
    pdf.savefig(fig); plt.close(fig)
    # page 2: env-3 trajectories
    R = {"ES": f"{C}/rec/es_ext.npz", "PPO": f"{C}/rec/ppo_ext.npz", "MPC": f"{C}/rec/mpc_ext.npz"}
    cc = {"ES": col[3], "PPO": col[2], "MPC": col[1]}
    fig, axs = plt.subplots(5, 1, figsize=(8.27, 11.69), sharex=True)
    for nm, p in R.items():
        d = np.load(p); e = ENV; dt = float(d["dt"]); t_ = np.arange(len(d[f"e{e}_alt"])) * dt
        hv = np.hypot(d[f"e{e}_vx"], d[f"e{e}_vy"]); tilt = np.degrees(np.hypot(d[f"e{e}_tilt_x"], d[f"e{e}_tilt_y"]))
        for ax, y in zip(axs, (d[f"e{e}_alt"], hv, -d[f"e{e}_vz"], tilt, d[f"e{e}_throttle"])):
            ax.plot(t_, y, color=cc[nm], label=nm)
    for ax, lab, lim in zip(axs, ("irtifa (m)", "yanal hız (m/s)", "iniş hızı (m/s)", "yatış (°)", "gaz"), (None, 1.2, 1.0, 15, None)):
        ax.set_ylabel(lab); ax.grid(alpha=.3)
        if lim: ax.axhline(lim, color="r", ls="--", lw=.8)
    axs[0].legend(); axs[-1].set_xlabel("zaman (s)"); axs[0].set_title("Aynı krater arazisi, aynı başlangıç (bölüm " + str(ENV) + "): üç yöntem")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    # page 3: terrain
    d = np.load(R["ES"]); h = d[f"e{ENV}_tile_height"]; s = float(d[f"e{ENV}_tile_size"])
    fig, ax = plt.subplots(1, 2, figsize=(11.69, 5.5))
    im = ax[0].imshow(h, extent=[-s / 2, s / 2, -s / 2, s / 2], origin="lower", cmap="terrain"); plt.colorbar(im, ax=ax[0], label="yükseklik (m)")
    for nm, p in R.items():
        dd = np.load(p); ax[0].plot(dd[f"e{ENV}_x"], dd[f"e{ENV}_y"], color=cc[nm], label=nm)
    ax[0].legend(); ax[0].set_title("Arazi (1680 m karo) ve yer izleri"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    xs = np.concatenate([np.load(p)[f"e{ENV}_x"] for p in R.values()]); ys = np.concatenate([np.load(p)[f"e{ENV}_y"] for p in R.values()])
    ax[1].imshow(h, extent=[-s / 2, s / 2, -s / 2, s / 2], origin="lower", cmap="terrain")
    ax[1].set_xlim(xs.min() - 60, xs.max() + 60); ax[1].set_ylim(ys.min() - 60, ys.max() + 60)
    for nm, p in R.items():
        dd = np.load(p); ax[1].plot(dd[f"e{ENV}_x"], dd[f"e{ENV}_y"], color=cc[nm]); ax[1].plot(dd[f"e{ENV}_x"][-1], dd[f"e{ENV}_y"][-1], "o", color=cc[nm])
    ax[1].set_title("İniş bölgesine yakın görünüm"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
print("ok", rows)
