"""Long-polls Telegram and answers a FIXED set of commands about training.

Scope, deliberately: this is a dispatcher, not a shell. Incoming text is
matched against a whitelist and never interpolated into a command. Two
reasons that matters here -- the bot token has been exposed in a chat
transcript at least once, and the machine this runs on holds the whole
project.

Second layer: only messages from `TELEGRAM_CHAT_ID` are acted on. Whoever
holds a leaked token can read updates, but cannot make a message appear to
come from the owner's chat, so they cannot drive this.

Run:
    setsid nohup .venv/bin/python scripts/telegram_control.py \
        > out/telegram_control.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path.home() / ".config" / "lunarsim" / "telegram.env"
API = "https://api.telegram.org/bot{token}/{method}"
HELP = ("komutlar:\n"
        "  durum  - kacinci adim, critic/Q, tahmini bitis\n"
        "  sonuc  - son SANITY satirlari\n"
        "  log    - logun son 20 satiri\n"
        "  dur    - koşan egitimi durdurur")


def _cfg() -> dict:
    v = {}
    if CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, val = line.partition("=")
                v[k.strip()] = val.strip().strip('"').strip("'")
    return v


def _call(token: str, method: str, params: dict, timeout: int = 70) -> dict:
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # network blips must not kill the loop
        return {"ok": False, "error": str(e)}


def _active_log() -> Path | None:
    logs = sorted((ROOT / "out").glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _train_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "isaac-env/bin/python scripts/train_sac_isaac.py"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]


def _status() -> str:
    log = _active_log()
    if log is None:
        return "hic egitim logu yok"
    t = log.read_text(errors="replace")
    ts = [int(x) for x in re.findall(r"total_timesteps \| (\d+)", t)]
    el = [int(x) for x in re.findall(r"time_elapsed    \| (\d+)", t)]
    cl = [float(x) for x in re.findall(r"critic_loss     \| ([\d.e+-]+)", t)]
    al = [float(x) for x in re.findall(r"actor_loss      \| ([\d.e+-]+)", t)]
    tgt = re.search(r"steps-per-stage'?,? '?(\d+)", t)
    alive = bool(_train_pids())
    if not ts:
        return f"{log.name}: henuz metrik yok ({'calisiyor' if alive else 'durmus'})"
    lines = [f"{log.name}  [{'CALISIYOR' if alive else 'DURMUS'}]", f"adim: {ts[-1]}"]
    if cl:
        lines.append(f"critic_loss: {cl[-1]:.4g}")
    if al:
        lines.append(f"Q ~ {-al[-1]:.1f}")
    if len(ts) > 2 and len(el) > 2 and el[-1] > el[len(el) // 2]:
        rate = (ts[-1] - ts[len(ts) // 2]) / (el[-1] - el[len(el) // 2])
        goal = int(tgt.group(1)) if tgt else 500000
        if rate > 0 and goal > ts[-1]:
            lines.append(f"hiz: {rate:.0f} adim/s, kalan ~{(goal - ts[-1]) / rate / 60:.0f} dk")
    san = [l for l in t.splitlines() if "SANITY" in l]
    if san:
        lines.append("son SANITY: " + san[-1].strip())
    return "\n".join(lines)


def _result() -> str:
    log = _active_log()
    if log is None:
        return "log yok"
    san = [l.strip() for l in log.read_text(errors="replace").splitlines() if "SANITY" in l]
    return f"{log.name}\n" + ("\n".join(san[-4:]) if san else "henuz SANITY yok (stage bitmedi)")


def _log_tail() -> str:
    log = _active_log()
    if log is None:
        return "log yok"
    return f"{log.name}\n" + "\n".join(log.read_text(errors="replace").splitlines()[-20:])


def _stop() -> str:
    pids = _train_pids()
    if not pids:
        return "koşan egitim yok"
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return f"durduruldu: {', '.join(map(str, pids))}\nNOT: checkpoint sadece stage sonunda yazilir, bu koşu kaydetmeden gitti"


HANDLERS = {
    "durum": _status, "status": _status,
    "sonuc": _result, "sonuç": _result, "result": _result,
    "log": _log_tail,
    "dur": _stop, "stop": _stop,
}


def main() -> None:
    cfg = _cfg()
    token, owner = cfg.get("TELEGRAM_BOT_TOKEN", ""), str(cfg.get("TELEGRAM_CHAT_ID", ""))
    if not token or not owner:
        raise SystemExit(f"{CONFIG} icinde TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID olmali")

    offset = 0
    # drop anything already queued, so a restart does not replay old commands
    first = _call(token, "getUpdates", {"timeout": 0})
    if first.get("ok") and first.get("result"):
        offset = first["result"][-1]["update_id"] + 1
    print(f"dinleniyor (chat {owner})", flush=True)

    while True:
        res = _call(token, "getUpdates", {"timeout": 50, "offset": offset})
        if not res.get("ok"):
            time.sleep(5)
            continue
        for upd in res.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or {}
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip().lower().lstrip("/")
            if not text:
                continue
            if chat_id != owner:
                print(f"yoksayildi, yabanci chat {chat_id}", flush=True)
                continue
            handler = HANDLERS.get(text.split()[0])
            try:
                reply = handler() if handler else HELP
            except Exception as e:
                reply = f"hata: {e}"
            print(f"{text} -> {len(reply)} karakter", flush=True)
            _call(token, "sendMessage",
                  {"chat_id": owner, "text": reply[:4000]}, timeout=20)


if __name__ == "__main__":
    main()
