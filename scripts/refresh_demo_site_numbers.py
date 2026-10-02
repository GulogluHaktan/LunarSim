"""Update the demo site's headline numbers from a capture's own outputs.

Every figure quoted on `docs/index.html` comes from a real run, so when the
landing capture is replaced the page has to be re-sourced from the new one
rather than hand-edited. This reads `eval_summary.txt`, `telemetry.csv` and
the map `.npz` the composer writes, and rewrites exactly the spans that
carry those numbers.

Run: .venv/bin/python scripts/refresh_demo_site_numbers.py out/eval_snapshots/hero_fixed
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "index.html"


def _summary(capture: Path) -> dict:
    out = {}
    for line in (capture / "eval_summary.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _telemetry(capture: Path) -> dict:
    rows = list(csv.DictReader(open(capture / "telemetry.csv")))
    t = np.array([float(r["t_s"]) for r in rows])
    pos = np.array([[float(r["pos_x_m"]), float(r["pos_y_m"])] for r in rows])
    track_m = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    return {
        "release_alt_m": float(rows[0]["alt_m"]),
        "duration_s": float(t[-1]),
        "track_m": track_m,
        "fuel_frac_end": float(rows[-1]["fuel_frac"]) if "fuel_frac" in rows[0] else None,
    }


def _lidar_stats(capture: Path) -> dict:
    """Scan count and total returns, straight from the capture's own index."""
    rows = list(csv.DictReader(open(capture / "lidar_index.csv")))
    return {
        "n_scans": len(rows),
        "n_returns": sum(int(r["hit_count"]) for r in rows),
    }


def _map_stats(capture: Path) -> dict:
    for cand in (capture.parent / f"{capture.name}_map.npz", capture / "map.npz"):
        if cand.exists():
            z = np.load(cand)
            cell = float(z["cell_m"])
            observed = int(np.isfinite(z["elevation_m"]).sum())
            return {
                "mapped_m2": observed * cell * cell,
                "landable_m2": float(z["landable"].sum()) * cell * cell,
            }
    raise SystemExit(f"no map .npz found for {capture}")


def _sub(text: str, pattern: str, repl: str, what: str) -> str:
    new, n = re.subn(pattern, repl, text)
    if n == 0:
        print(f"  WARNING: no match for {what} -- left unchanged")
    else:
        print(f"  {what}: {n} replacement(s) -> {repl}")
    return new


def main() -> None:
    capture = Path(sys.argv[1] if len(sys.argv) > 1 else "out/eval_snapshots/hero_fixed")
    if not capture.is_absolute():
        capture = REPO / capture
    s, tel, mp = _summary(capture), _telemetry(capture), _map_stats(capture)
    ld = _lidar_stats(capture)

    vz = abs(float(s["touchdown_vz_m_s"]))
    vxy = abs(float(s["touchdown_vxy_m_s"]))
    tilt = float(s["touchdown_tilt_deg"])
    fuel_pct = float(s["fuel_frac"]) * 100.0
    mapped = mp["mapped_m2"]

    print(f"from {capture.name}: vz={vz:.2f} vxy={vxy:.2f} tilt={tilt:.1f} "
          f"fuel={fuel_pct:.1f}% mapped={mapped:,.0f} m2 "
          f"release={tel['release_alt_m']:.0f} m track={tel['track_m']:.0f} m "
          f"duration={tel['duration_s']:.1f} s "
          f"scans={ld['n_scans']} returns={ld['n_returns'] / 1e6:.2f}M")
    safe = s.get("landed_safely", "").lower().startswith("true")
    if not safe and "--allow-failed-landing" not in sys.argv:
        raise SystemExit(
            f"refusing to publish: landed_safely={s.get('landed_safely')!r}.\n"
            "Pass --allow-failed-landing ONLY if the page actually presents the per-criterion\n"
            "result rather than claiming a success -- the point of this guard is that a page\n"
            "saying 'successful landing' must never be fed a run that was not one.")
    if not safe:
        print("  NOTE: publishing a run that did NOT pass every criterion, by explicit request")

    page = PAGE.read_text()
    page = _sub(page, r"<b>\d+\.\d+ <small", f"<b>{vz:.2f} <small", "hero vz")
    page = _sub(page, r"<b>\d+\.\d+°</b>", f"<b>{tilt:.1f}°</b>", "hero tilt")
    page = _sub(page, r"<b>[\d\s]+ <small style=\"display:inline;font-size:14px;color:var\(--fg-faint\)\">m²",
                f"<b>{mapped:,.0f} <small style=\"display:inline;font-size:14px;color:var(--fg-faint)\">m²".replace(",", " "),
                "hero mapped area")
    page = _sub(page, r'<span class="tr">%\d+\.\d+</span><span class="en">\d+\.\d+%</span>',
                f'<span class="tr">%{fuel_pct:.1f}</span><span class="en">{fuel_pct:.1f}%</span>',
                "hero fuel")
    page = _sub(page, r"\d+ metre yükseklikten bırakılan araç, \d+\.\d+ saniyede \d+ metrelik",
                f"{tel['release_alt_m']:.0f} metre yükseklikten bırakılan araç, "
                f"{tel['duration_s']:.1f} saniyede {tel['track_m']:.0f} metrelik",
                "TR intro")
    page = _sub(page, r"Released at \d+ m, the vehicle flew a \d+ m ground track and set down softly after\s+\d+\.\d+ s\.",
                f"Released at {tel['release_alt_m']:.0f} m, the vehicle flew a {tel['track_m']:.0f} m "
                f"ground track and set down softly after {tel['duration_s']:.1f} s.",
                "EN intro")
    page = _sub(page, r'(Dikey temas hızı</span>.*?<td class="n ok">)[\d.]+', rf"\g<1>{vz:.2f}", "table vz")
    page = _sub(page, r'(Yatay temas hızı</span>.*?<td class="n ok">)[\d.]+', rf"\g<1>{vxy:.2f}", "table vxy")
    page = _sub(page, r'(Gövde eğimi</span>.*?<td class="n ok">)[\d.]+°', rf"\g<1>{tilt:.1f}°", "table tilt")
    if "touchdown_w_rad_s" in s:
        page = _sub(page, r'(Açısal hız</span>.*?<td class="n ok">)[\d.]+',
                    rf"\g<1>{abs(float(s['touchdown_w_rad_s'])):.3f}", "table omega")
    if "touchdown_leg_height_diff_m" in s:
        page = _sub(page, r'(zemin yükseklik farkı</span>.*?<td class="n ok">)[\d.]+',
                    rf"\g<1>{float(s['touchdown_leg_height_diff_m']):.3f}", "table leg diff")
    page = _sub(page, r"\d+ saniyede \d+ tarama, [\d.]+ milyon dönüş",
                f"{tel['duration_s']:.0f} saniyede {ld['n_scans']} tarama, "
                f"{ld['n_returns'] / 1e6:.2f} milyon dönüş", "TR lidar counts")
    page = _sub(page, r"\d+ scans and [\d.]+ million returns over \d+ seconds",
                f"{ld['n_scans']} scans and {ld['n_returns'] / 1e6:.2f} million returns "
                f"over {tel['duration_s']:.0f} seconds", "EN lidar counts")

    PAGE.write_text(page)
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
