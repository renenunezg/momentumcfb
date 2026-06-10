import pytest

from backend.cfbd.client import CFBDClient, CFBDError


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def make_client(monkeypatch, responses):
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    client = CFBDClient()
    calls = iter(responses)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: next(calls))
    return client


def test_get_returns_list(monkeypatch):
    client = make_client(monkeypatch, [FakeResponse(200, [{"id": 1}])])
    assert client.get("/games", {"year": 2024}) == [{"id": 1}]


def test_get_raises_on_http_error(monkeypatch):
    client = make_client(monkeypatch, [FakeResponse(401, {"message": "bad key"})])
    with pytest.raises(CFBDError):
        client.get("/games", {"year": 2024})


def test_get_raises_on_non_list_payload(monkeypatch):
    client = make_client(monkeypatch, [FakeResponse(200, {"unexpected": "shape"})])
    with pytest.raises(CFBDError):
        client.get("/games", {"year": 2024})


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(CFBDError):
        CFBDClient()
