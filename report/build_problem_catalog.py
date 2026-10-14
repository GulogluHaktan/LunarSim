"""Problem catalog PDF: every problem LunarSim hit and every method tried against it.

Input: JSON lists (one per history slice) with entries
  {baslik, alan, tarih, problem, denemeler:[{yontem, sonuc, commit}], durum, not}
Output: report/lunarsim_problem_katalogu.pdf (HTML printed by headless Chrome).

Usage: python report/build_problem_catalog.py part1.json part2.json ...
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS = {  # label -> (colour, order)
    "çözüldü": ("#0ca30c", 0),
    "kısmen": ("#c98500", 1),
    "çözülmedi": ("#d03b3b", 2),
    "yanlış teşhis çıktı": ("#7a5cc4", 3),
}
AREAS = ["fizik/ortam", "arazi/veri", "sensör/render", "ödül", "eğitim/algoritma",
         "ölçüm/değerlendirme", "sessiz bayrak/konfigürasyon", "altyapı"]


def esc(s):
    return html.escape(str(s or ""))


def main(paths):
    items = []
    for p in paths:
        items += json.loads(Path(p).read_text())
    for it in items:
        if it.get("alan") not in AREAS:
            it["alan"] = "altyapı"
        if it.get("durum") not in STATUS:
            it["durum"] = "kısmen"
    items.sort(key=lambda it: (AREAS.index(it["alan"]), it.get("tarih", "")))

    st = Counter(it["durum"] for it in items)
    n_try = sum(len(it.get("denemeler", [])) for it in items)
    kpis = "".join(
        f'<div class="kpi"><b style="color:{STATUS[k][0]}">{st.get(k, 0)}</b><span>{k}</span></div>'
        for k in STATUS)

    rows = []
    for a in AREAS:
        sub = [it for it in items if it["alan"] == a]
        if sub:
            c = Counter(it["durum"] for it in sub)
            rows.append(f"<tr><td>{a}</td><td>{len(sub)}</td>"
                        + "".join(f"<td>{c.get(k, 0) or ''}</td>" for k in STATUS) + "</tr>")
    summary = ("<table class='sum'><tr><th>alan</th><th>problem</th>"
               + "".join(f"<th>{k}</th>" for k in STATUS) + "</tr>" + "".join(rows) + "</table>")

    body, n = [], 0
    for a in AREAS:
        sub = [it for it in items if it["alan"] == a]
        if not sub:
            continue
        body.append(f"<h2>{esc(a)} <small>({len(sub)})</small></h2>")
        for it in sub:
            n += 1
            col = STATUS[it["durum"]][0]
            tries = "".join(
                f"<tr><td>{esc(d.get('yontem'))}</td><td>{esc(d.get('sonuc'))}</td>"
                f"<td class='h'>{esc(d.get('commit'))}</td></tr>" for d in it.get("denemeler", []))
            note = f"<p class='note'>Not: {esc(it['not'])}</p>" if it.get("not") else ""
            body.append(
                f"<div class='card' style='border-left-color:{col}'>"
                f"<div class='hd'><span class='no'>P{n}</span> {esc(it['baslik'])}"
                f"<span class='tag' style='background:{col}'>{esc(it['durum'])}</span>"
                f"<span class='dt'>{esc(it.get('tarih'))}</span></div>"
                f"<p>{esc(it.get('problem'))}</p>"
                + (f"<table class='tr'><tr><th>denenen yöntem</th><th>sonuç</th><th>commit</th></tr>{tries}</table>"
                   if tries else "")
                + note + "</div>")

    doc = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>LunarSim Problem Kataloğu</title><style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
body {{ font-family: 'DejaVu Sans', sans-serif; color: #0b0b0b; font-size: 9.2pt; line-height: 1.42; }}
h1 {{ font-size: 19pt; margin: 0 0 2pt; }} .sub {{ color: #555; margin: 0 0 10pt; }}
h2 {{ font-size: 12.5pt; margin: 14pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt;
      break-after: avoid; }} h2 small {{ color: #777; font-weight: normal; }}
.kpis {{ display: flex; gap: 8pt; margin: 8pt 0; }}
.kpi {{ flex: 1; background: #f4f3ef; border-radius: 4pt; padding: 6pt 8pt; }}
.kpi b {{ font-size: 17pt; display: block; }} .kpi span {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; vertical-align: top; padding: 2.5pt 4pt; border-bottom: 1px solid #e6e5e0; }}
th {{ background: #f4f3ef; font-weight: bold; }}
.sum td:not(:first-child), .sum th:not(:first-child) {{ text-align: right; }}
.card {{ border-left: 4pt solid; padding: 3pt 0 4pt 8pt; margin: 0 0 8pt; break-inside: avoid; }}
.hd {{ font-weight: bold; font-size: 10pt; }} .no {{ color: #888; }}
.tag {{ color: #fff; font-size: 7.5pt; padding: 1pt 5pt; border-radius: 6pt; margin-left: 6pt; font-weight: normal; }}
.dt {{ float: right; color: #888; font-weight: normal; font-size: 8pt; }}
.card p {{ margin: 3pt 0; }} .tr {{ font-size: 8.4pt; }} .tr td:first-child {{ width: 48%; }}
.h {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; color: #666; white-space: nowrap; }}
.note {{ color: #555; font-style: italic; }}
</style></head><body>
<h1>LunarSim: problemler ve denenen yöntemler</h1>
<p class="sub">Projenin tüm geçmişi: 144 git commit'i (2026-09-21 → 2026-10-09), commit'lenmiş handover ve
2026-10-09/10 oturumundaki henüz commit'lenmemiş çalışma. {len(items)} problem, {n_try} deneme. Sayılar commit
mesajlarından ve ölçüm kayıtlarından aynen alınmıştır.</p>
<div class="kpis">{kpis}</div>
<p><b>Bugünkü durum.</b> Isaac orbit_descent'te guided-gain PPO, sadece-sonuç ödülüyle 192 eşleştirilmiş taze bölümde
%89.6 iniyor; kontrolcü %52.6 (McNemar 72–1, p=1.6e-20). En iyi sabit kazanç ayarı %71.9.
"Yanlış teşhis çıktı" etiketli kayıtlar, sonradan ölçümle çürütülen açıklamalardır (örneğin Q-SNR iddiası).</p>
{summary}
{''.join(body)}
</body></html>"""
    html_path = HERE / "problem_katalogu.html"
    pdf_path = HERE / "lunarsim_problem_katalogu.pdf"
    html_path.write_text(doc, encoding="utf-8")
    subprocess.run(["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {pdf_path} ({len(items)} problems, {n_try} attempts)")


if __name__ == "__main__":
    main(sys.argv[1:])
