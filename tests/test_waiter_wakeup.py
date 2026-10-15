"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Regression tests for P-02: a waiting /acquire must re-check when
admission-relevant state changes, not only after ACQUIRE_POLL backoff.
Exercises the HTTP API with real asyncio concurrency. No live GPU or
network.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock

import httpx
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production is 1 s / 10 s. Tests use a short cap so a missed wakeup
# still sleeps nearly POLL_MAX, while a real wakeup grants in < 0.2 s.
POLL_MIN = 0.05
POLL_MAX = 0.8
GRANT_AFTER_WAKE_S = 0.2


def _sleeps_before_cap(min_s, max_s):
    """Wall time of doubling sleeps until the next wait is the cap."""
    total = 0.0
    backoff = min_s
    while backoff < max_s - 1e-9:
        total += backoff
        backoff = min(backoff * 2, max_s)
    return total


def _prep(srv, monkeypatch, min_s=POLL_MIN, max_s=POLL_MAX):
    """Shrink poll constants, skip Proton, empty Ollama so budget math is local."""
    monkeypatch.setattr(srv, "ACQUIRE_POLL_MIN_SECONDS", min_s)
    monkeypatch.setattr(srv, "ACQUIRE_POLL_MAX_SECONDS", max_s)
    monkeypatch.setattr(srv, "_refresh_proton_state", AsyncMock())
    srv._ollama_mb = 0.0
    srv._unmanaged_heavy = False


def _client(srv):
    transport = httpx.ASGITransport(app=srv.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _lease(caller, mb, now):
    return {
        "caller": caller,
        "mb": mb,
        "acquired_at": now,
        "last_seen": now,
        "priority": "P2",
        "preemptible": False,
        "pid": None,
        "preempt_flagged_at": None,
    }


async def _wait_until_pending(srv, n, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(srv._pending) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"pending={srv._pending!r} wanted >= {n}")


async def _wait_until_capped(srv, n, min_s, max_s):
    await _wait_until_pending(srv, n)
    # Land shortly after the first capped wait starts so HEAD's
    # asyncio.sleep(backoff) would still take nearly POLL_MAX.
    await asyncio.sleep(_sleeps_before_cap(min_s, max_s) + max_s * 0.25)


@pytest.mark.asyncio
async def test_release_wakes_waiter_at_backoff_cap(broker_app, monkeypatch):
    """Test: a waiter at capped backoff is granted well under the cap after /release."""
    _prep(broker_app, monkeypatch)
    async with _client(broker_app) as client:
        held = await client.post("/acquire", json={
            "caller": "A", "mb": 10000.0, "timeout_s": 5.0,
        })
        assert held.status_code == 200
        lease_id = held.json()["lease_id"]

        waiter = asyncio.create_task(client.post("/acquire", json={
            "caller": "B", "mb": 2000.0, "timeout_s": 60.0,
        }))
        await _wait_until_capped(broker_app, 1, POLL_MIN, POLL_MAX)
        assert not waiter.done()

        t0 = time.monotonic()
        released = await client.post("/release", json={"lease_id": lease_id})
        assert released.status_code == 200
        assert released.json()["released"] is True
        resp = await asyncio.wait_for(waiter, timeout=2.0)
        latency = time.monotonic() - t0

    assert resp.status_code == 200, resp.text
    assert resp.json()["granted"] is True
    assert latency < GRANT_AFTER_WAKE_S


@pytest.mark.asyncio
async def test_new_fifo_head_wakes_when_prior_waiter_is_granted(broker_app, monkeypatch):
    """Test: after head B is granted, waiter C is granted without a full backoff step."""
    _prep(broker_app, monkeypatch)
    async with _client(broker_app) as client:
        held = await client.post("/acquire", json={
            "caller": "A", "mb": 10000.0, "timeout_s": 5.0,
        })
        assert held.status_code == 200
        lease_id = held.json()["lease_id"]

        waiter_b = asyncio.create_task(client.post("/acquire", json={
            "caller": "B", "mb": 4000.0, "timeout_s": 60.0,
        }))
        await _wait_until_pending(broker_app, 1)
        waiter_c = asyncio.create_task(client.post("/acquire", json={
            "caller": "C", "mb": 4000.0, "timeout_s": 60.0,
        }))
        await _wait_until_capped(broker_app, 2, POLL_MIN, POLL_MAX)
        assert not waiter_b.done()
        assert not waiter_c.done()

        t0 = time.monotonic()
        released = await client.post("/release", json={"lease_id": lease_id})
        assert released.status_code == 200

        done, _unfinished = await asyncio.wait(
            {waiter_b, waiter_c},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=2.0,
        )
        assert waiter_b in done
        resp_b = waiter_b.result()
        assert resp_b.status_code == 200, resp_b.text
        assert time.monotonic() - t0 < GRANT_AFTER_WAKE_S

        t_b = time.monotonic()
        resp_c = await asyncio.wait_for(waiter_c, timeout=2.0)
        latency_c = time.monotonic() - t_b

    assert resp_c.status_code == 200, resp_c.text
    assert resp_c.json()["granted"] is True
    assert latency_c < GRANT_AFTER_WAKE_S



@pytest.mark.asyncio
async def test_reclaim_broken_lease_wakes_waiter_at_backoff_cap(broker_app, monkeypatch):
    """Test: watchdog reclaim of a no-pid holder wakes a waiter at capped backoff."""
    _prep(broker_app, monkeypatch)
    now = time.monotonic()
    holder = _lease("A", 10000.0, now)
    broker_app._leases["L-hold"] = holder
    broker_app._caller_to_lease["A"] = "L-hold"

    async with _client(broker_app) as client:
        waiter = asyncio.create_task(client.post("/acquire", json={
            "caller": "B", "mb": 2000.0, "timeout_s": 60.0,
        }))
        await _wait_until_capped(broker_app, 1, POLL_MIN, POLL_MAX)
        assert not waiter.done()

        t0 = time.monotonic()
        await broker_app._reclaim_broken_lease(
            "L-hold", dict(holder), ["heartbeat_stale"],
        )
        resp = await asyncio.wait_for(waiter, timeout=2.0)
        latency = time.monotonic() - t0

    assert "L-hold" not in broker_app._leases
    assert resp.status_code == 200, resp.text
    assert resp.json()["granted"] is True
    assert latency < GRANT_AFTER_WAKE_S


@pytest.mark.asyncio
async def test_acquire_times_out_on_schedule_without_notifications(broker_app, monkeypatch):
    """Test: with no notifies, a waiter still gets 504 within one backoff of timeout_s."""
    timeout_s = 0.8
    _prep(broker_app, monkeypatch)
    async with _client(broker_app) as client:
        held = await client.post("/acquire", json={
            "caller": "A", "mb": 10000.0, "timeout_s": 5.0,
        })
        assert held.status_code == 200

        t0 = time.monotonic()
        resp = await asyncio.wait_for(
            client.post("/acquire", json={
                "caller": "B", "mb": 2000.0, "timeout_s": timeout_s,
            }),
            timeout=timeout_s + POLL_MAX + 1.0,
        )
        elapsed = time.monotonic() - t0

    assert resp.status_code == 504, resp.text
    assert elapsed < timeout_s + POLL_MAX + 0.3
    assert elapsed >= timeout_s - 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
