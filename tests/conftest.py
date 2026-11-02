import pytest

import fake_openf1
from f1dash import data, openf1


@pytest.fixture
def fake_api(monkeypatch):
    """Route all OpenF1 calls to the synthetic API."""
    monkeypatch.setattr(openf1, "fetch", fake_openf1.make_fetch())


@pytest.fixture
def race(fake_api) -> data.Session:
    return data.load_session(fake_openf1.RACE_KEY)


@pytest.fixture
def qualifying(fake_api) -> data.Session:
    return data.load_session(fake_openf1.QUALI_KEY)
