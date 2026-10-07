"""Tests for shared/retry.py: which errors are retried, and the backoff waits (no real sleeping)."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for 'shared'

from shared.retry import is_transient, with_retries  # noqa: E402


class Flaky:
    """Fails with `error` the first `failures` calls, then returns 'ok'."""

    def __init__(self, failures: int, error: Exception):
        self.failures, self.error, self.calls = failures, error, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_transient_error_is_retried_with_growing_waits():
    waits = []
    fn = Flaky(2, ConnectionError("Ollama not running"))
    assert with_retries(fn, sleep=waits.append) == "ok"
    assert fn.calls == 3
    assert waits == [2.0, 4.0]  # exponential backoff


def test_gives_up_after_last_attempt():
    fn = Flaky(5, ConnectionError("still down"))
    with pytest.raises(ConnectionError):
        with_retries(fn, sleep=lambda s: None)
    assert fn.calls == 3


def test_permanent_error_fails_immediately():
    fn = Flaky(1, ValueError("not audio"))
    with pytest.raises(ValueError):
        with_retries(fn, sleep=lambda s: None)
    assert fn.calls == 1  # no pointless retries


def http_error(code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = code
    return requests.HTTPError(response=response)


def test_http_status_decides():
    assert is_transient(http_error(503))  # server busy: try again
    assert is_transient(http_error(429))  # rate limited: try again
    assert not is_transient(http_error(404))  # wrong topic/URL: won't change
