"""
Author: FreeTokenBuild
Date:   2026
Email:  eyeclept@pm.me

Description:
    Unit tests for FreeToken's broker client: preemptible acquire, the
    heartbeat preempt flag, and the game (unmanaged load) refusal. Broker
    HTTP is mocked; no live broker or GPU is required. Wrapper lifecycle
    tests live in test_lifecycle.py.
"""
# Imports
import logging

import httpx
import pytest

import broker_client

# Globals
logger = logging.getLogger(__name__)


class _FakeResp:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.posts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        return None

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return self._resp


@pytest.fixture(autouse=True)
def _reset_shared_client():
    broker_client.close_client()
    yield
    broker_client.close_client()

def test_acquire_sends_preemptible_and_pid(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        /acquire body must declare preemptible=True and the child PID so
        the broker can flag this holder and the watchdog can target it.
        A 200 grant returns the lease_id.
    """
    resp = _FakeResp(200, {"granted": True, "lease_id": "lease-abc"})
    fake = _FakeClient(resp)
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)

    lease_id = broker_client._broker_acquire(8382, pid=4242)

    assert lease_id == "lease-abc"
    assert len(fake.posts) == 1
    body = fake.posts[0]["json"]
    assert body["caller"] == "freetoken"
    assert body["mb"] == 8382
    assert body["preemptible"] is True
    assert body["pid"] == 4242


def test_acquire_sends_preemptible_when_pid_unknown(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        Before `ft serve` exists the wrapper still has to reserve VRAM.
        pid may be None, but preemptible must still be True so the
        holder is not structurally un-preemptible for that window.
    """
    resp = _FakeResp(200, {"granted": True, "lease_id": "lease-early"})
    fake = _FakeClient(resp)
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)

    lease_id = broker_client._broker_acquire(8382)

    assert lease_id == "lease-early"
    body = fake.posts[0]["json"]
    assert body["preemptible"] is True
    assert body["pid"] is None


def test_heartbeat_returns_preempt_flag(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        A 200 heartbeat with ok=true and preempt=true is a self-evict
        signal, not a lost lease. ok stays True so the wrapper does not
        re-acquire in place.
    """
    resp = _FakeResp(
        200,
        {"ok": True, "preempt": True, "reason": "yield to a waiting caller"},
    )
    fake = _FakeClient(resp)
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)

    ok, preempt = broker_client._broker_heartbeat("lease-abc")

    assert ok is True
    assert preempt is True


def test_heartbeat_lost_lease_is_not_preempt(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        ok=false means the sweeper already dropped the lease. That is a
        re-acquire signal, not a self-evict.
    """
    resp = _FakeResp(200, {"ok": False})
    fake = _FakeClient(resp)
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)

    ok, preempt = broker_client._broker_heartbeat("lease-abc")

    assert ok is False
    assert preempt is False


def test_heartbeat_error_is_not_preempt(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        A transport failure must not look like a preemption request.
    """
    def _boom():
        raise httpx.ConnectError("broker down")

    monkeypatch.setattr(broker_client.httpx, "Client", _boom)

    ok, preempt = broker_client._broker_heartbeat("lease-abc")

    assert ok is False
    assert preempt is False


def test_acquire_503_unmanaged_load_raises_without_retry(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        A 503 carrying the broker's unmanaged-load marker means a game is
        running. It must surface as BrokerAcquireError on the first
        attempt: no five-attempt retry burst, no fail-open, even with
        BROKER_FAIL_OPEN set.
    """
    resp = _FakeResp(503, text='{"detail":"unmanaged_gpu_load"}', headers={"Retry-After": "1800"})
    fake = _FakeClient(resp)
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)
    monkeypatch.setattr(broker_client, "BROKER_FAIL_OPEN", True)

    with pytest.raises(broker_client.BrokerAcquireError) as exc:
        broker_client._broker_acquire(8382)

    assert len(fake.posts) == 1
    assert exc.value.retry_after == 1800.0
    assert not isinstance(exc.value, broker_client.BrokerNoRoom)


def test_acquire_504_raises_retryable_no_room_with_hint(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:    A 504 is the retryable BrokerNoRoom, carrying Retry-After.
    """
    fake = _FakeClient(_FakeResp(504, text="no room", headers={"Retry-After": "10"}))
    monkeypatch.setattr(broker_client.httpx, "Client", lambda: fake)

    with pytest.raises(broker_client.BrokerNoRoom) as exc:
        broker_client._broker_acquire(8382)

    assert len(fake.posts) == 1
    assert exc.value.retry_after == 10.0


def test_consecutive_heartbeats_reuse_one_client(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        Two heartbeats share one httpx.Client. The constructor runs once.
    """
    constructions = []
    resp = _FakeResp(200, {"ok": True, "preempt": False})

    def factory():
        fake = _FakeClient(resp)
        constructions.append(fake)
        return fake

    monkeypatch.setattr(broker_client.httpx, "Client", factory)

    ok1, preempt1 = broker_client._broker_heartbeat("lease-abc")
    ok2, preempt2 = broker_client._broker_heartbeat("lease-abc")

    assert (ok1, preempt1) == (True, False)
    assert (ok2, preempt2) == (True, False)
    assert len(constructions) == 1
    assert len(constructions[0].posts) == 2
