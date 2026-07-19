"""Optional Telegram notifications for success / manual-upload fallback."""
import time
import traceback as _traceback
from pathlib import Path

import requests

import config

# Telegram hard-caps a single message at 4096 chars.
_TG_LIMIT = 4096


def _enabled() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def notify_error(stage: str, exc: BaseException) -> None:
    """Send a failure alert WITH the traceback tail so you can debug from phone."""
    tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
    header = f"\u274c FAILED at: {stage}\n{type(exc).__name__}: {exc}\n\n"
    # Keep the most recent (most useful) part of the traceback within the limit.
    budget = _TG_LIMIT - len(header) - 20
    if len(tb) > budget:
        tb = "...(truncated)...\n" + tb[-budget:]
    notify(header + tb)


def notify(message: str, retries: int = 3) -> None:
    """Send a text message to Telegram (no-op if not configured), with retries."""
    if not _enabled():
        print(f"[notify] {message}")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message},
                timeout=30,
            )
            if resp.ok:
                return
            print(f"[notify] Telegram returned {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] attempt {attempt + 1} failed: {exc}")
        time.sleep(2 * (attempt + 1))
    print(f"[notify] gave up sending message after {retries} attempts")


def send_video(video_path: Path, caption: str, retries: int = 3) -> bool:
    """Send the finished MP4 to Telegram. Returns True if delivered."""
    if not _enabled():
        print(f"[notify] video ready for manual upload: {video_path}")
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendVideo"
    for attempt in range(retries):
        try:
            with open(video_path, "rb") as fh:
                resp = requests.post(
                    url,
                    data={
                        "chat_id": config.TELEGRAM_CHAT_ID,
                        "caption": caption[:1000],
                        "supports_streaming": True,
                    },
                    files={"video": fh},
                    timeout=300,
                )
            if resp.ok:
                return True
            print(f"[notify] sendVideo returned {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] send video attempt {attempt + 1} failed: {exc}")
        time.sleep(3 * (attempt + 1))
    notify(f"Video ready for manual upload but Telegram send failed:\n{video_path}")
    return False
