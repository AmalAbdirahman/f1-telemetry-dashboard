"""The REST client: error translation, retries and caching."""

import json

import pytest
import requests

from f1dash import config, openf1


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.ok = status < 400
        self.headers = {}
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(openf1, "_throttle", lambda: None)
    monkeypatch.setattr(openf1.time, "sleep", lambda _: None)


def serve(monkeypatch, *responses):
    calls = []
    queue = list(responses)

    def get(url, params=None, timeout=None):
        calls.append((url, params))
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "get", get)
    return calls


def test_404_means_no_results(monkeypatch):
    serve(monkeypatch, FakeResponse(404, {"detail": "No results found."}))
    assert openf1.fetch("starting_grid", session_key=1) == []


def test_results_are_cached_on_disk(monkeypatch):
    calls = serve(monkeypatch, FakeResponse(200, [{"a": 1}]))
    assert openf1.fetch("laps", session_key=1) == [{"a": 1}]
    assert openf1.fetch("laps", session_key=1) == [{"a": 1}]
    assert len(calls) == 1


def test_different_params_are_cached_separately(monkeypatch):
    calls = serve(monkeypatch, FakeResponse(200, [{"a": 1}]))
    openf1.fetch("laps", session_key=1)
    openf1.fetch("laps", session_key=2)
    assert len(calls) == 2


def test_rate_limit_retries_then_succeeds(monkeypatch):
    calls = serve(monkeypatch, FakeResponse(429), FakeResponse(200, [{"a": 1}]))
    assert openf1.fetch("laps", session_key=1) == [{"a": 1}]
    assert len(calls) == 2


def test_persistent_rate_limit_gives_a_readable_error(monkeypatch):
    serve(monkeypatch, FakeResponse(429))
    with pytest.raises(openf1.OpenF1Error, match="rate limit"):
        openf1.fetch("laps", session_key=1)


def test_live_session_lock_gives_a_readable_error(monkeypatch):
    serve(monkeypatch, FakeResponse(401))
    with pytest.raises(openf1.OpenF1Error, match="live"):
        openf1.fetch("laps", session_key=1)


def test_network_failure_gives_a_readable_error(monkeypatch):
    serve(monkeypatch, requests.ConnectionError("boom"))
    with pytest.raises(openf1.OpenF1Error, match="Could not reach"):
        openf1.fetch("laps", session_key=1)


def test_corrupt_cache_file_is_ignored(monkeypatch):
    serve(monkeypatch, FakeResponse(200, [{"a": 1}]))
    path = openf1._cache_path("laps", {"session_key": 1})
    path.write_text("{not json", encoding="utf-8")
    assert openf1.fetch("laps", session_key=1) == [{"a": 1}]
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]
