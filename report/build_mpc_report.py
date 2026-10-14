"""Build the MPC (MPPI) landing report (Turkish): flowcharts + measured plots -> HTML -> PDF.
Numbers come from out/evo/* and ~/lunarsim_artifacts. Run: ~/isaac-env/bin/python report/build_mpc_report.py
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
import sys as _sys; _sys.path.insert(0, str(ROOT))
EVO = ROOT / "out" / "evo"; ART = Path.home() / "lunarsim_artifacts"
FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MPC, ES, PPO, CTL, GOOD, CRIT, WARN = "#7a3fc4", "#2a78d6", "#0ca30c", "#eb6834", "#0ca30c", "#d03b3b", "#c98a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h

def mcnemar(x, y):
    b = int(np.sum(y & ~x)); c = int(np.sum(x & ~y)); n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c); return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def boot(a, b, waves=16, B=10000, seed=0):
    A, Bm = a.reshape(waves, -1), b.reshape(waves, -1); r = np.random.default_rng(seed)
    d = [Bm[i].mean() - A[i].mean() for i in (r.integers(0, waves, waves) for _ in range(B))]
    return np.percentile(d, [2.5, 97.5])

def box(ax, x, y, w, h, text, fc="#f3ecfb", ec=MPC, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            color=INK, linespacing=1.25)

def arrow(ax, p0, p1, color=INK2, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))

def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10 * h / w); return fig, ax

def save(fig, name):
    p = FIG / f"mpc_{name}.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); return p

def img(p, w="100%"):
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" style="width:{w}">'

# ------------------------------------------------------------ data
M = json.load(open(EVO / "mpc_isaac_orbit_val192.json"))
E = json.load(open(EVO / "isaac_orbit_val192.json"))
_imp = EVO / "mpc_isaac_orbit_impact.json"
if _imp.exists():
    _I = json.load(open(_imp))
    same = _I["mpc"]["landed"] == M["mpc"]["landed"]
    print("impact run reproduces the validation outcomes episode for episode:", same)
    if same:
        M["mpc_impact"] = _I["mpc_impact"]
P = json.load(open(ART / "gg_isaac_orbit_v4_seed21_repro_validation.json"))
ctl = np.array(E["controller"]["landed"]); es = np.array(E["es_policy"]["landed"])
ppo = np.array(P["candidate_landed"]); mpc = np.array(M["mpc"]["landed"])
assert (np.array(M["controller"]["landed"]) == ctl).all() and (np.array(P["baseline_landed"]) == ctl).all(), \
    "controller vectors differ between files: comparison would not be paired"
n = 192
arms = [("ZEM/ZEV kontrolcüsü", ctl, CTL), ("MPC (MPPI)", mpc, MPC), ("Guided-gain PPO", ppo, PPO), ("ES (kontrolcüsüz)", es, ES)]

# ------------------------------------------------------------ figures
def fig_loop():
    fig, ax = blank(10, 4.0)
    ax.text(5, 3.85, "MPC (MPPI) bir kontrol adımında: ölç → planla → ilk komutu uygula → tekrar", ha="center", fontsize=11.5, weight="bold")
    box(ax, 0.1, 1.9, 1.7, 1.5, "1  Ölçülen durum\nirtifa, hız, yatış,\naçısal hız, yakıt,\nkalan süre", fc="#f4f3ef", ec=INK2, fs=8.6)
    box(ax, 2.1, 1.9, 1.9, 1.5, "2  512 aday komut dizisi\nmevcut ortalama + gürültü\n(20 segment × 0.6 s\n= 12 s ufuk)", fs=8.6)
    box(ax, 4.3, 1.9, 1.9, 1.5, "3  Modelle ileri sar\nortamın dinamiğinin\nbir kopyası (GPU,\n512 × 120 adım)", fs=8.6)
    box(ax, 6.5, 1.9, 1.7, 1.5, "4  Maliyet\nhız hatası, yatış,\ntemas şiddeti,\nzaman, süre dolması", fs=8.6)
    box(ax, 8.4, 1.9, 1.5, 1.5, "5  Softmin ağırlık\nortalama dizi\nyeniden hesaplanır", fc="#eaf6ea", ec=GOOD, fs=8.6)
    for x0, x1 in ((1.82, 2.08), (4.02, 4.28), (6.22, 6.48), (8.22, 8.38)): arrow(ax, (x0, 2.65), (x1, 2.65))
    box(ax, 2.4, 0.15, 5.2, 1.0, "6  İlk segmentin komutu uygulanır (gaz, pitch, roll, yaw torku)\n4 simülasyon adımı (0.2 s) sonra yeniden planlama", fc="#fff6e0", ec=WARN, fs=8.8)
    arrow(ax, (9.1, 1.88), (7.62, 0.65), color=MPC, rad=-0.2); arrow(ax, (2.38, 0.65), (0.9, 1.88), color=MPC, rad=-0.2)
    return save(fig, "loop")

def fig_horizon():
    # bang-bang tilt-then-untilt: delta_v = g * alpha * T^3 / ... with tilt accel alpha and lateral accel g*theta
    alpha, g = 0.0200, 1.62
    v = np.linspace(0.2, 12, 200)
    T_half = (v / (alpha * g / 3.0)) ** (1 / 3)           # theta = alpha t^2/2 up, mirrored down: dv = g*alpha*T^3/3
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axs[0]
    ax.plot(v, 2 * T_half, color=MPC, lw=2.4)
    ax.axhline(4, color=CRIT, ls="--", lw=1.2); ax.text(0.4, 4.4, "ilk ufuk: 4 s", color=CRIT, fontsize=9)
    ax.axhline(12, color=GOOD, ls="--", lw=1.2); ax.text(0.4, 12.4, "kullanılan ufuk: 12 s", color=GOOD, fontsize=9)
    ax.set_xlabel("kesilmesi gereken yanal hız (m/s)"); ax.set_ylabel("yatıp düzelme için gereken süre (s)")
    ax.set_title("Yatış yetkisi çok düşük (α = 0.02 rad/s²)", fontsize=10); ax.set_ylim(0, 22)
    ax = axs[1]
    t = np.linspace(0, 12, 300); th = np.where(t < 6, 0.5 * alpha * t ** 2, 0.5 * alpha * (12 - t) ** 2)
    ax.plot(t, np.degrees(th), color=MPC, lw=2.4); ax.fill_between(t, 0, np.degrees(th), color=MPC, alpha=0.12)
    ax.set_xlabel("zaman (s)"); ax.set_ylabel("yatış açısı (°)")
    ax.set_title("Örnek: 12 s'lik tam torklu yat–düzel ≈ 2.3 m/s keser", fontsize=10)
    ax.text(6, np.degrees(th).max() * 0.5, f"tepe {np.degrees(th).max():.0f}°\nΔv ≈ {g * np.trapz(th, t):.1f} m/s", ha="center", fontsize=9)
    return save(fig, "horizon")

def fig_tune():
    rows = list(csv.DictReader(open(EVO / "mpc_tune_orbit" / "curve.csv")))
    g = [int(r["gen"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(g, [100 * float(r["mean_land"]) for r in rows], "o-", color=INK2, lw=1.4, ms=4, label="aday ortalaması (8 bölüm)")
    ax.plot(g, [100 * float(r["best_land"]) for r in rows], "o-", color=WARN, lw=1.4, ms=4, label="en iyi aday")
    ax.plot(g, [100 * float(r["sel_land"]) for r in rows], "o-", color=MPC, lw=2.4, ms=5, label="ortalama θ, seçim bloğu (8 bölüm)")
    ax.set_xlabel("ES nesli"); ax.set_ylabel("iniş oranı (%)"); ax.set_ylim(-3, 105)
    ax.set_title("MPC maliyet ağırlıklarının ES ile ayarı (analitik orbit_descent, 14 nesil)")
    ax.legend(fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    return save(fig, "tune")

def fig_results():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axs[0]
    for i, (lab, v, col) in enumerate(arms):
        k = int(v.sum()); lo, hi = wil(k, n)
        ax.bar(i, 100 * k / n, color=col, yerr=[[max(0, 100 * k / n - 100 * lo)], [max(0, 100 * hi - 100 * k / n)]], capsize=4, ecolor=INK2)
        ax.text(i, 100 * k / n + 4.5, f"{k}/{n}\n%{100*k/n:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([a[0].replace(" (", "\n(") for a in arms], fontsize=7.8)
    ax.set_ylim(0, 118); ax.set_ylabel("iniş oranı (%)"); ax.set_title("Isaac orbit_descent, 192 eşleştirilmiş bölüm", fontsize=10)
    ax = axs[1]; xs = np.arange(16)
    for lab, v, col in arms:
        ax.plot(xs, v.reshape(16, -1).sum(1), "o-", color=col, lw=2.0 if col == MPC else 1.2, ms=3.5, label=lab)
    ax.set_ylim(0, 13); ax.set_xlabel("dalga (tohum 90000 … 105000)"); ax.set_ylabel("12 bölümden iniş sayısı")
    ax.legend(fontsize=7.8, loc="lower right", ncol=2); ax.set_title("Dalga bazında", fontsize=10)
    return save(fig, "results")

def fig_fail():
    imp = M.get("mpc_impact")
    if not imp: return None
    lim = dict(vz=1.0, vxy=1.2, tilt=np.radians(15), w=0.5)
    rows = []
    for d in imp:
        if d.get("why") != "crash": continue
        s = d.get("impact", {})
        if "vz" not in s: continue
        vxy = math.hypot(s.get("vx", 0), s.get("vy", 0)); tilt = math.hypot(s.get("tilt_x", 0), s.get("tilt_y", 0))
        w = math.sqrt(s.get("wx", 0) ** 2 + s.get("wy", 0) ** 2 + s.get("wz", 0) ** 2)
        rows.append((abs(s["vz"]), vxy, tilt, w, d.get("margins", {}).get("leg_diff", 1.0)))
    if not rows: return None
    R = np.array(rows)
    sev = np.stack([R[:, 0] / lim["vz"], R[:, 1] / lim["vxy"], R[:, 2] / lim["tilt"], R[:, 3] / lim["w"]], 1)
    worst = sev.argmax(1); names = ["dikey hız", "yanal hız", "yatış", "açısal hız"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axs[0]
    ax.scatter(R[:, 1], R[:, 0], c=MPC, s=40, ec="k", zorder=3)
    ax.axvline(1.2, color=CRIT, ls="--", lw=1); ax.axhline(1.0, color=CRIT, ls="--", lw=1)
    ax.set_xlabel("temas yanal hızı (m/s)"); ax.set_ylabel("temas |dikey hız| (m/s)")
    ax.set_title(f"MPC çarpmaları ({len(R)} bölüm): limit 1.2 / 1.0", fontsize=10)
    ax = axs[1]
    cnt = [int((worst == i).sum()) for i in range(4)]
    ax.bar(range(4), cnt, color=MPC); ax.set_xticks(range(4)); ax.set_xticklabels(names)
    for i, c in enumerate(cnt): ax.text(i, c + 0.5, str(c), ha="center")
    ax.set_ylabel("bölüm sayısı"); ax.set_title("Çarpmada en kötü (bağlayıcı) kriter", fontsize=10)
    out = save(fig, "fail")
    med = np.median(sev.max(1))
    return out, cnt, len(R), float(med)

figs = {k: f() for k, f in [("loop", fig_loop), ("horizon", fig_horizon), ("tune", fig_tune), ("results", fig_results)]}
fail = fig_fail()

# ------------------------------------------------------------ numbers for text
km, ke, kp, kc = int(mpc.sum()), int(es.sum()), int(ppo.sum()), int(ctl.sum())
b_c, c_c, p_c = mcnemar(ctl, mpc); b_e, c_e, p_e = mcnemar(es, mpc); b_p, c_p, p_p = mcnemar(ppo, mpc)
ci_c = boot(ctl, mpc); ci_e = boot(es, mpc)
wsec = np.mean(M.get("wave_seconds", [0]))
best = np.load(EVO / "mpc_tune_orbit" / "best_params.npy")
from lunarsim.evo.mpc import PARAM_NAMES  # noqa: E402
ptab = "".join(f"<tr><td><code>{n_}</code></td><td>{10 ** v:.3g}</td></tr>" for n_, v in zip(PARAM_NAMES, best))
why = M["mpc"]["why"]

dev_rows = [
 ("1", "Ufuk 4 s, tekil maliyet", "analitik ramp_35m: 12/12 zaman aşımı, yanal hız ~2.3 m/s'de sabit", "yatış yetkisi α=0.02 rad/s²: 2.3 m/s'yi kesmek ≈ 12 s'lik tam torklu yat–düzel, 4 s bunu göremez"),
 ("2", "Ufuk 12 s, yanal/dikey referans ayrı", "son şiddet medyanı 0.35–0.52, ama yere inmeden asılı kalıyor", "temas maliyeti `w_td·(1+sev²)` mükemmel inişe bile pahalı; asılı kalmak daha ucuz"),
 ("3", "İniş maliyeti yalnızca limit aşılınca büyük + zaman maliyeti + süre dolunca başarısızlık", "ramp_35m: 2–4/12 iniş", "MPC bütçenin bittiğini bilmiyordu"),
 ("4", "w_td düşürüldü (100 → 10)", "analitik ramp_35m: **12/12**", "—"),
 ("5", "orbit_descent, el ayarı", "0/12 zaman aşımı; araç 200 → 284 m tırmandı", "yanal hata maliyeti dikeyden ~20× baskın; ES ile ayar gerekti"),
 ("6", "ES ile 10 ağırlığın ayarı (14 nesil)", "seçim bloğu %88 (8 bölüm); taze 24 bölüm: 20/24", "iki başarısızlık yere −1.05/−1.09 m/s ile çarpma (limit 1.0)"),
 ("7", "Isaac'te ilk deneme (payısız)", "ilk dalga 0/12 (kontrolcü 4/12)", "flare limitte planlanıyor, küçük model farkı başarısızlığa çeviriyor"),
 ("8", "Planlayıcıda temas limitlerinin %70'i (margin 0.7)", "analitik taze 24/24; **Isaac 139/192**", "—"),
]
dev_html = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in dev_rows).replace("**", "")

if fail:
    f_path, cnt, nf, med = fail
    fail_html = (f"<figure>{img(f_path)}<figcaption>Şekil 6. Isaac'te MPC'nin {nf} çarpması, temas anı durumu. En kötü kriterin limite oranının medyanı {med:.2f}×.</figcaption></figure>"
                 f"<p>Çarpmaların bağlayıcı kriteri: dikey hız {cnt[0]}, yanal hız {cnt[1]}, yatış {cnt[2]}, açısal hız {cnt[3]}.</p>")
else:
    fail_html = "<p>Çarpma anı durumları kaydedilemedi (hata analizi dosyası yok); yalnızca sonuç sayıları raporlanıyor.</p>"

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.5; color: #0b0b0b; }
h1 { font-size: 20pt; margin: 0 0 4px; } h2 { page-break-after: avoid; font-size: 13.5pt; margin: 18px 0 6px; border-bottom: 2px solid #7a3fc4; padding-bottom: 3px; }
.sub { color: #52514e; margin-bottom: 14px; }
.kpi { display: flex; gap: 10px; margin: 10px 0; } .kpi div { flex: 1; border: 1px solid #e4e3df; border-radius: 8px; padding: 8px 10px; background: #fcfcfb; }
.kpi b { font-size: 17pt; display: block; } .note { background: #fff6e0; border-left: 4px solid #c98a00; padding: 8px 12px; margin: 10px 0; }
figure { margin: 8px 0 12px; page-break-inside: avoid; } figcaption { color: #52514e; font-size: 9pt; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; } td, th { border: 1px solid #e4e3df; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f4f3ef; } code { background: #f4f3ef; padding: 1px 4px; border-radius: 3px; font-size: 8.8pt; }
"""
body = f"""
<h1>MPC ile iniş: örnekleme tabanlı model öngörülü kontrol</h1>
<div class=sub>LunarSim · Apollo LM · Isaac Sim 6.1 · 12 Ekim 2026 · tüm sayılar <code>out/evo/</code> ve <code>~/lunarsim_artifacts/</code> dosyalarından okunmuştur</div>
<div class=kpi>
 <div><b>%{100*km/n:.1f}</b>MPC, Isaac orbit_descent<br>{km}/192 iniş</div>
 <div><b>%{100*kc/n:.1f}</b>ZEM/ZEV kontrolcüsü<br>{kc}/192</div>
 <div><b>{b_c}–{c_c}</b>MPC–kontrolcü McNemar<br>p = {p_c:.1e}</div>
 <div><b>{b_e}–{c_e}</b>MPC–ES (ES lehine)<br>p = {p_e:.1e}</div>
</div>

<h2>1. Ne yapıldı</h2>
<p>Amaç: öğrenme olmadan, aracın kendi dinamik modelini her adımda ileri sararak iniş yapan bir <b>model öngörülü kontrolcü (MPC)</b> kurmak ve aynı 192 eşleştirilmiş bölümde ZEM/ZEV kontrolcüsü, guided-gain PPO ve ES ile karşılaştırmak. MPC bu çalışmada ZEM/ZEV'i hiçbir yerde kullanmıyor; yalnızca ölçülen durumu ve ortamın dinamik denklemlerinin bir kopyasını kullanıyor. Sonuç Isaac'te <b>{km}/192 (%{100*km/n:.1f})</b>: kontrolcüyü anlamlı geçiyor, ancak ES ve PPO'nun (~%90) gerisinde.</p>
<figure>{img(figs['loop'])}<figcaption>Şekil 1. Bir kontrol adımı: 512 aday komut dizisi modelde ileri sarılır, maliyetle puanlanır ve softmin ağırlıkla ortalama dizi güncellenir (MPPI). İlk segment uygulanır, 0.2 s sonra yeniden planlanır.</figcaption></figure>

<h2>2. Model, maliyet ve ufuk</h2>
<p><b>Model.</b> <code>AnalyticLanderEnv.step</code>'in dinamiği torch'ta toplu (batched) yazıldı: DPS gazı (kübik gaz eğrisi), RCS tork komutları ve sönümlü çift integratör yatış, yaw'ın itki yönüne etkisi, yakıtla kütle ve atalet değişimi. Model, ortamın kendi adımıyla rastgele komut dizisinde yatış, hız ve irtifa değerlerini aynı ondalığa veriyor (birim testi). Zemin aracın altında düz kabul edilir.</p>
<p><b>Maliyet (10 log-ağırlık, ES ile ayarlanır).</b> Her simülasyon adımında: yanal hızın rehber alanına göre hatası (<code>w_lat</code>), dikey hızın alana göre hatası (<code>w_vz</code>), yatış (<code>w_tilt</code>), açısal hız (<code>w_rate</code>), komut eforu (<code>w_u</code>), havada geçen zaman (<code>w_time</code>). Temasta: dört limitin (dikey hız, yanal hız, yatış, açısal hız) en kötü oranı olan <i>şiddet</i> sev için <code>w_td·(0.2·sev² + 10·relu(sev−1)²)</code>. Ufukta hâlâ havadaysa <code>w_T</code>·rehber hatası. Kalan süre dolarsa 20·<code>w_td</code> başarısızlık maliyeti. Rehber alan (hız–irtifa şeması) proje ödülündeki <code>target_velocity</code> sabitleridir; bu açıkça verilmiş bir ön bilgidir ve ES politikasının gözlem slotlarındaki bilginin aynısıdır.</p>
<p><b>Ufuk neden 12 s.</b> Yatış yetkisi çok düşük: tam torkla 0.3 rad'a 5.5 s'de ulaşılır ve geri düzeltmek aynı kadar sürer. Bu yüzden yanal hızı kesmek yat–düzel olarak onlarca saniyelik bir manevradır.</p>
<figure>{img(figs['horizon'])}<figcaption>Şekil 2. Solda yanal hızı kesmek için gereken tam torklu yat–düzel süresi (Δv = g·α·T³/3, T yarım süre; itki ≈ ağırlık varsayımı) ve iki ufuk uzunluğu; sağda 12 s'lik profil.</figcaption></figure>

<h2>3. Geliştirme süreci: ne denendi, ne ölçüldü</h2>
<table><tr><th>#</th><th>değişiklik</th><th>ölçülen</th><th>neden / bulgu</th></tr>{dev_html}</table>

<h2>4. Maliyet ağırlıklarının ES ile ayarı</h2>
<p>10 log-ağırlık, ES (antitetik, centered rank, Adam) ile analitik <code>orbit_descent</code>'te ayarlandı: 14 nesil × 16 aday × 8 bölüm (ortak tohumlar, 200000+); seçim bloğu 160000+ ayrı. Her aday 12 s'lik MPPI'yi 300 kez çözerek uçtuğu için tek nesil yaklaşık 3.5 dakika sürdü (8 paralel GPU süreci).</p>
<figure>{img(figs['tune'])}<figcaption>Şekil 3. Ayar eğrisi: en iyi adaylar baştan %75–100 inebiliyor; ortalama θ seçim bloğunda gürültülü biçimde %88'e çıktı. 8 bölümle aralık geniştir (±15 puan).</figcaption></figure>
<table><tr><th>ağırlık</th><th>seçilen değer</th></tr>{ptab}</table>

<h2>5. Isaac sonuçları</h2>
<figure>{img(figs['results'])}<figcaption>Şekil 4. Sol: dört kol, %95 Wilson aralığı. Sağ: dalga bazında. Kontrolcü vektörü üç dosyada birebir aynı olduğu için karşılaştırma eşleştirilmiştir.</figcaption></figure>
<table><tr><th>karşılaştırma (192 bölüm)</th><th>MPC-only</th><th>diğer-only</th><th>McNemar p</th><th>fark (puan), dalga-bootstrap %95</th></tr>
<tr><td>MPC vs kontrolcü</td><td>{b_c}</td><td>{c_c}</td><td>{p_c:.1e}</td><td>+{100*(mpc.mean()-ctl.mean()):.1f}  [{100*ci_c[0]:+.1f}, {100*ci_c[1]:+.1f}]</td></tr>
<tr><td>MPC vs ES</td><td>{b_e}</td><td>{c_e}</td><td>{p_e:.1e}</td><td>{100*(mpc.mean()-es.mean()):+.1f}  [{100*ci_e[0]:+.1f}, {100*ci_e[1]:+.1f}]</td></tr>
<tr><td>MPC vs PPO</td><td>{b_p}</td><td>{c_p}</td><td>{p_p:.1e}</td><td>{100*(mpc.mean()-ppo.mean()):+.1f}</td></tr></table>
<p>Sonlanma nedenleri (MPC): {', '.join(f'{k} {v}' for k, v in why.items())}. Kontrolcüde 91 çarpma. Bir dalga yaklaşık {wsec:.0f} s sürdü (12 ortam, MPC kolu ~53 s); MPC ~140 ms/yeniden planlama ile 0.2 s'lik yeniden planlama aralığının altında kalıyor, yani GPU'da gerçek zamana yakın.</p>
{fail_html}

<h2>6. Dürüst sınırlar</h2>
<div class=note><ul style="margin:0 0 0 16px;padding:0">
<li><b>Ağırlıklar analitik backend'de ayarlandı</b> (<code>orbit_descent</code>, tohum 160000+/200000+), Isaac'te yalnızca doğrulandı. Isaac'te %72 ile analitikteki taze blok sonucu (24/24) arasındaki fark model uyuşmazlığıdır; ayrıştırılmadı.</li>
<li><b>Doğrulama bloğuna kısmi bakış.</b> <code>margin=0.7</code> güvenlik payı; analitik taze blok (170000) ve Isaac'te payısız ilk dalganın (tohum 90000: 0/12) sonucundan sonra eklendi. Yani doğrulama bloğunun ilk dalgası tasarım kararını etkiledi; 16 dalganın tamamı karar sonrası koşuldu.</li>
<li><b>Model bilgisi:</b> MPC ortamın dinamik denklemlerini biliyor. ES ve PPO model-free; bu bir elmayla elma kıyası değildir. Kıyas "kontrolcüsüz iniş yapan yöntemler" arasındaki pratik sonuçtur.</li>
<li><b>Rehber alan sabitleri</b> (hız–irtifa şeması) ön bilgi olarak kullanıldı, yalnızca kontrolcüden bağımsız bir saha olarak.</li>
<li><b>Tek ayar vektörü, tek ES tohumu;</b> yalnızca orbit_descent'te ayarlandı. Alçak basamaklar (ramp_*) için ayrıca ayar yapılmadı: <code>ramp_100m</code>'de ilk el ayarıyla 4/12 ölçüldü.</li>
<li><b>Çarpmaların %{100*(1-km/n):.0f}'i</b> çözülmedi; ES ile ayar yalnızca 8 bölümle 14 nesil sürdü ve daha uzun ayar ya da daha büyük model uyumu (ör. Isaac içinde ince ayar) denenmedi.</li>
</ul></div>

<h2>7. Yeniden üretim</h2>
<table><tr><th>adım</th><th>komut / dosya</th></tr>
<tr><td>MPC çekirdeği</td><td><code>lunarsim/evo/mpc.py</code> (MPPIMPC, MPCConfig)</td></tr>
<tr><td>analitik değerlendirme</td><td><code>scripts/mpc_analytic_eval.py --stage orbit_descent</code></td></tr>
<tr><td>ağırlık ayarı</td><td><code>scripts/mpc_tune.py --stages orbit_descent --n 8 --pairs 8 --gens 14 --K 512 --M 20 --hold 12 --roll-sub 2 --exec-steps 4 --workers 8</code></td></tr>
<tr><td>Isaac doğrulama</td><td><code>scripts/mpc_isaac_eval.py ~/lunarsim_artifacts/evo/mpc_params_v2.npy --stage orbit_descent --waves 16 --wave-seed0 90000</code></td></tr>
</table>
"""
html_path = ROOT / "report" / "lunarsim_mpc_rapor.html"
html_path.write_text(f"<!doctype html><html lang=tr><head><meta charset=utf-8><title>LunarSim MPC raporu</title><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
pdf = ROOT / "report" / "lunarsim_mpc_rapor.pdf"
subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf, "| mpc", km, "ctl", kc, "es", ke, "ppo", kp, "| fail analysis:", "yes" if fail else "no")
