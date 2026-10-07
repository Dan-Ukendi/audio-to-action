"""Retry a call when it fails for a reason that might fix itself (network, server busy).

    with_retries(analyze, transcript, cache_dir=...)   # up to 3 tries, waiting 2 s then 4 s

Only TRANSIENT errors are retried. A broken audio file or an invalid LLM answer fails the
same way every time, so retrying would just waste minutes before failing anyway.
"""

import logging
import time
from collections.abc import Callable

import httpx
import requests

log = logging.getLogger(__name__)


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: can't connect, timed out, server overloaded."""
    if isinstance(exc, requests.HTTPError):
        # 429 = too many requests, 5xx = server trouble. 4xx otherwise = our mistake, won't change.
        code = exc.response.status_code if exc.response is not None else 0
        return code == 429 or code >= 500
    return isinstance(exc, (
        ConnectionError,  # what the ollama client raises when Ollama isn't running
        TimeoutError,
        requests.ConnectionError, requests.Timeout,  # ntfy
        httpx.TransportError,  # ollama's HTTP library, lower level
    ))


def with_retries(fn: Callable, *args, attempts: int = 3, base_delay: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep, **kwargs):
    """Call fn(*args, **kwargs); on a transient error wait base_delay, 2x, 4x... and try again.

    Exponential backoff: waiting longer each time gives a restarting service room to come back
    instead of hammering it. `sleep` is a parameter so tests can run without real waiting.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("%s failed (%s: %s); retry %d/%d in %.0f s",
                        getattr(fn, "__name__", "call"), type(exc).__name__, exc, attempt, attempts - 1, delay)
            sleep(delay)
