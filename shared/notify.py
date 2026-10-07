"""Push notifications via ntfy (https://ntfy.sh): one HTTP POST, and the phone app shows it.

Privacy: the public ntfy.sh server can read every message, and anyone who knows the topic name
can subscribe. So callers' names, numbers and message content never go in a push: the text only
says that something needs attention. Details stay in the local database.

Dry run (prints instead of sending) when NTFY_DRY_RUN=1 or no real topic is configured.
"""

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PLACEHOLDER_TOPIC = "change-me-to-something-random"  # the value in .env.example


def settings() -> dict:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    return {
        "server": os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/"),
        "topic": topic,
        "dry_run": os.getenv("NTFY_DRY_RUN", "1") == "1" or topic in ("", PLACEHOLDER_TOPIC),
    }


def send_push(title: str, message: str, priority: str = "high", tags: str = "") -> str:
    """Send one push. Returns "sent" or "dry_run". Raises on network/HTTP errors (Phase 5 retries).

    priority: min | low | default | high | urgent  (urgent = loudest alert on the phone)
    tags:     comma-separated emoji names shown in the app, e.g. "rotating_light"
    """
    cfg = settings()
    if cfg["dry_run"]:
        log.info("[ntfy dry run] priority=%s title=%r message=%r", priority, title, message)
        return "dry_run"

    response = requests.post(
        f"{cfg['server']}/{cfg['topic']}",
        data=message.encode("utf-8"),
        headers={"Title": title, "Priority": priority, "Tags": tags},
        timeout=10,  # never hang the pipeline on a slow network
    )
    response.raise_for_status()
    return "sent"
