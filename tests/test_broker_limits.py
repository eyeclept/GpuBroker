"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Regression tests for S-05, R-03, and S-22: caller and lease_id length
limits, the active-lease cap, grant-time same-caller dedup (including
when the second request no longer fits), notification TTL, and
log-injection escaping. All cases go through the HTTP API.
No live GPU or network.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_acquire_rejects_caller_longer_than_max(broker_app):
    """Test: a caller of 129 characters returns 422."""
    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "x" * 129,
        "mb": 0.0,
        "timeout_s": 1.0,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}


def test_acquire_rejects_empty_caller(broker_app):
    """Test: an empty caller returns 422."""
    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "",
        "mb": 0.0,
        "timeout_s": 1.0,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}


def test_release_rejects_lease_id_longer_than_max(broker_app):
    """Test: a lease_id of 65 characters to /release returns 422."""
    client = TestClient(broker_app.app)
    resp = client.post("/release", json={"lease_id": "y" * 65})
    assert resp.status_code == 422


def test_lease_cap_refuses_new_caller_and_still_dedups(broker_app, monkeypatch):
    """Test: a 4th distinct caller gets 503 too_many_leases; an existing caller still gets its lease."""
    monkeypatch.setattr(broker_app, "MAX_ACTIVE_LEASES", 3)
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    client = TestClient(broker_app.app)

    ids = []
    for name in ("cap_a", "cap_b", "cap_c"):
        resp = client.post("/acquire", json={
            "caller": name, "mb": 0.0, "timeout_s": 5.0,
        })
        assert resp.status_code == 200, resp.text
        ids.append(resp.json()["lease_id"])
    assert len(broker_app._leases) == 3

    refused = client.post("/acquire", json={
        "caller": "cap_d", "mb": 0.0, "timeout_s": 5.0,
    })
    assert refused.status_code == 503
    assert refused.json()["detail"] == "too_many_leases"
    assert refused.headers.get("retry-after") == "5"
    assert len(broker_app._leases) == 3
    assert "cap_d" not in broker_app._caller_to_lease

    again = client.post("/acquire", json={
        "caller": "cap_a", "mb": 0.0, "timeout_s": 5.0,
    })
    assert again.status_code == 200, again.text
    assert again.json()["lease_id"] == ids[0]
    assert len(broker_app._leases) == 3


@pytest.mark.asyncio
async def test_concurrent_same_caller_grants_one_lease(broker_app, monkeypatch):
    """Test: two concurrent first acquires for one caller grant a single lease (R-03)."""
    monkeypatch.setattr(broker_app, "ACQUIRE_POLL_MIN_SECONDS", 0.05)
    monkeypatch.setattr(broker_app, "ACQUIRE_POLL_MAX_SECONDS", 0.8)
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    broker_app._ollama_mb = 0.0
    broker_app._unmanaged_heavy = False

    transport = httpx.ASGITransport(app=broker_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        held = await client.post("/acquire", json={
            "caller": "filler", "mb": 10000.0, "timeout_s": 5.0,
        })
        assert held.status_code == 200, held.text
        filler_id = held.json()["lease_id"]

        body = {"caller": "same_caller_r03", "mb": 2000.0, "timeout_s": 60.0}
        waiter_a = asyncio.create_task(client.post("/acquire", json=body))
        waiter_b = asyncio.create_task(client.post("/acquire", json=body))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(broker_app._pending) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"pending={broker_app._pending!r} wanted >= 2")
        assert not waiter_a.done()
        assert not waiter_b.done()
        n_waiting = len(broker_app._leases)

        released = await client.post("/release", json={"lease_id": filler_id})
        assert released.status_code == 200
        resp_a, resp_b = await asyncio.gather(waiter_a, waiter_b)

    assert resp_a.status_code == 200, resp_a.text
    assert resp_b.status_code == 200, resp_b.text
    id_a = resp_a.json()["lease_id"]
    id_b = resp_b.json()["lease_id"]
    assert id_a == id_b
    # Filler is gone; the two waiters added exactly one lease.
    assert len(broker_app._leases) == n_waiting - 1 + 1
    assert id_a in broker_app._leases
    assert broker_app._caller_to_lease.get("same_caller_r03") == id_a


@pytest.mark.asyncio
async def test_concurrent_same_caller_second_does_not_fit(broker_app, monkeypatch):
    """Test: two concurrent 6000 MB acquires for one caller still share one lease when the second no longer fits (R-03)."""
    monkeypatch.setattr(broker_app, "ACQUIRE_POLL_MIN_SECONDS", 0.05)
    monkeypatch.setattr(broker_app, "ACQUIRE_POLL_MAX_SECONDS", 0.8)
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    broker_app._ollama_mb = 0.0
    broker_app._unmanaged_heavy = False

    transport = httpx.ASGITransport(app=broker_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        held = await client.post("/acquire", json={
            "caller": "filler", "mb": 10000.0, "timeout_s": 5.0,
        })
        assert held.status_code == 200, held.text
        filler_id = held.json()["lease_id"]

        body = {"caller": "same_caller_r03_nofit", "mb": 6000.0, "timeout_s": 8.0}
        waiter_a = asyncio.create_task(client.post("/acquire", json=body))
        waiter_b = asyncio.create_task(client.post("/acquire", json=body))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(broker_app._pending) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"pending={broker_app._pending!r} wanted >= 2")
        assert not waiter_a.done()
        assert not waiter_b.done()

        released = await client.post("/release", json={"lease_id": filler_id})
        assert released.status_code == 200
        t_release = time.monotonic()
        resp_a, resp_b = await asyncio.gather(waiter_a, waiter_b)
        elapsed = time.monotonic() - t_release

    assert resp_a.status_code == 200, resp_a.text
    assert resp_b.status_code == 200, resp_b.text
    id_a = resp_a.json()["lease_id"]
    id_b = resp_b.json()["lease_id"]
    assert id_a == id_b
    assert elapsed < 1.0
    assert broker_app._pending == set()
    assert id_a in broker_app._leases
    assert broker_app._caller_to_lease.get("same_caller_r03_nofit") == id_a


def test_stale_notification_is_dropped_on_grant(broker_app, monkeypatch):
    """Test: a notice older than TTL is not delivered and is removed from the map."""
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    ttl = getattr(broker_app, "NOTIFICATION_TTL_SECONDS", 3600.0)
    broker_app._pending_notifications["ttl_caller"] = (
        "stale watchdog notice",
        time.monotonic() - ttl - 1.0,
    )
    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "ttl_caller", "mb": 0.0, "timeout_s": 5.0,
    })
    assert resp.status_code == 200, resp.text
    assert "warning" not in resp.json()
    assert "ttl_caller" not in broker_app._pending_notifications


def test_acquire_logs_newline_in_caller_as_repr(broker_app, monkeypatch, caplog):
    """Test: a caller containing a newline produces one log record with escaped \\n."""
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    client = TestClient(broker_app.app)
    caller = "evil\ninjected"
    with caplog.at_level(logging.INFO, logger="server"):
        resp = client.post("/acquire", json={
            "caller": caller, "mb": 0.0, "timeout_s": 5.0,
        })
    assert resp.status_code == 200, resp.text
    msgs = [r.getMessage() for r in caplog.records if "caller=" in r.getMessage()]
    assert msgs
    for msg in msgs:
        assert "\n" not in msg
        assert "\\n" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
