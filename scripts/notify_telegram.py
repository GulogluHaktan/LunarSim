"""Push a short message (and optionally a file) to Telegram.

Credentials are read from, in order:
  1. $TELEGRAM_BOT_TOKEN / $TELEGRAM_CHAT_ID
  2. ~/.config/lunarsim/telegram.env   (KEY=VALUE lines)

DELIBERATELY not from anything inside the repo: everything here except
`out/` is tracked by git, so a token living in the working tree is one
`git add -A` away from being published. The config path is outside it.

Usage:
    python scripts/notify_telegram.py --discover-chat-id    # once, after
                                                            # messaging the bot
    python scripts/notify_telegram.py "text"
    python scripts/notify_telegram.py "text" --file out/train.log --tail 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".config" / "lunarsim" / "telegram.env"
API = "https://api.telegram.org/bot{token}/{method}"


def _load_config() -> dict:
    vals = {}
    if CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"').strip("'")
    return {
        "token": os.environ.get("TELEGRAM_BOT_TOKEN") or vals.get("TELEGRAM_BOT_TOKEN", ""),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID") or vals.get("TELEGRAM_CHAT_ID", ""),
    }


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _call(token: str, method: str, params: dict) -> dict:
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--file", help="append the tail of this file")
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--discover-chat-id", action="store_true",
                    help="print the chat id of whoever last messaged the bot")
    a = ap.parse_args()

    cfg = _load_config()
    if not cfg["token"]:
        print(f"no bot token. Put it in {CONFIG} as:\n"
              f"  TELEGRAM_BOT_TOKEN=123456:ABC...\n"
              f"  TELEGRAM_CHAT_ID=<see --discover-chat-id>", file=sys.stderr)
        return 1

    if a.discover_chat_id:
        res = _call(cfg["token"], "getUpdates", {})
        ids = {str(u.get("message", {}).get("chat", {}).get("id"))
               for u in res.get("result", []) if u.get("message")}
        ids.discard("None")
        print("\n".join(sorted(ids)) if ids
              else "no messages yet -- send your bot any message, then re-run")
        return 0 if ids else 1

    if not cfg["chat_id"]:
        print("no chat id; run with --discover-chat-id", file=sys.stderr)
        return 1

    # parse_mode=HTML is only here for the <pre> wrapper on log tails, so
    # EVERYTHING that is not that wrapper has to be escaped -- including
    # --text. It was not, and a message containing "<- best" came back as
    # HTTP 400 (Telegram parsed it as an unclosed tag). This script's whole
    # job is relaying log and error output, which is full of <, > and &.
    body = _esc(a.text)
    if a.file and Path(a.file).is_file():
        tail = Path(a.file).read_text(errors="replace").splitlines()[-a.tail:]
        body += "\n\n<pre>" + "\n".join(_esc(l) for l in tail) + "</pre>"

    res = _call(cfg["token"], "sendMessage",
                {"chat_id": cfg["chat_id"], "text": body[:4096], "parse_mode": "HTML"})
    if not res.get("ok"):
        print(f"telegram refused: {res}", file=sys.stderr)
        return 1
    print("sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
