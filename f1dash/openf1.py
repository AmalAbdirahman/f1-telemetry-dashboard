"""Small REST client for the public OpenF1 API.

Handles the free tier's rate limits, retries and a disk cache of responses so
that reruns and page reloads do not spend requests. No account is needed for
historical data.
"""

import hashlib
import json
import threading
import time
from collections import deque

import requests

from f1dash import config


class OpenF1Error(Exception):
    """The API could not provide the data. The message is safe to show users."""


_lock = threading.Lock()
_recent_requests: deque[float] = deque()


def _throttle() -> None:
    """Block until another request fits inside both rate limits."""
    with _lock:
        while True:
            now = time.monotonic()
            while _recent_requests and now - _recent_requests[0] >= 60:
                _recent_requests.popleft()
            last_second = sum(now - t < 1 for t in _recent_requests)
            wait = 0.0
            if len(_recent_requests) >= config.MAX_REQUESTS_PER_MINUTE:
                wait = 60 - (now - _recent_requests[0])
            elif last_second >= config.MAX_REQUESTS_PER_SECOND:
                wait = 1 / config.MAX_REQUESTS_PER_SECOND
            if wait <= 0:
                _recent_requests.append(now)
                return
            time.sleep(wait)


def _cache_path(endpoint: str, params: dict):
    key = json.dumps([endpoint, sorted(params.items())], default=str)
    return config.cache_dir() / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _read_cache(path) -> list | None:
    try:
        if time.time() - path.stat().st_mtime > config.DISK_CACHE_TTL_S:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path, records: list) -> None:
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(records), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)  # Caching is an optimisation; never fail on it.


def _request(endpoint: str, params: dict) -> list:
    url = f"{config.OPENF1_BASE_URL}/{endpoint}"
    for attempt in range(config.MAX_RETRIES):
        _throttle()
        try:
            response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            if attempt == config.MAX_RETRIES - 1:
                raise OpenF1Error("Could not reach OpenF1. Check the connection and try again.") from exc
            time.sleep(2**attempt)
            continue

        if response.status_code == 404:
            return []  # OpenF1 answers 404 when a query matches nothing.
        if response.status_code in (401, 403):
            raise OpenF1Error(
                "OpenF1 locks its data while a session is live. "
                "It opens up again shortly after the session ends."
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < config.MAX_RETRIES - 1:
                delay = response.headers.get("Retry-After", "")
                time.sleep(min(float(delay) if delay.isdigit() else 5 * (attempt + 1), 20))
                continue
            if response.status_code == 429:
                raise OpenF1Error("OpenF1's rate limit was reached. Wait a minute and try again.")
            raise OpenF1Error("OpenF1 is having problems right now. Try again shortly.")
        if not response.ok:
            raise OpenF1Error(f"OpenF1 rejected the request (HTTP {response.status_code}).")
        try:
            return response.json()
        except ValueError as exc:
            raise OpenF1Error("OpenF1 returned a response that could not be read.") from exc
    raise OpenF1Error("Could not reach OpenF1.")


def fetch(endpoint: str, **params) -> list[dict]:
    """Return an endpoint's records as a list of dicts, using the disk cache.

    Comparison filters use keys such as ``"date>"``, passed with ``**{...}``.
    Empty results are not cached, because a recent session's data may still be
    arriving.
    """
    path = _cache_path(endpoint, params)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    records = _request(endpoint, params)
    if records:
        _write_cache(path, records)
    return records
