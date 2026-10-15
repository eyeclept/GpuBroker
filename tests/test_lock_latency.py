"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Regression tests for P-01 / S-06: /status and /acquire must not hold
_lock across nvidia-smi. /status takes a live read only when the used-MB
cache is stale, and that read happens before the lock. /acquire uses the
cache and never calls nvidia-smi. No live GPU or network.
"""

import asyncio
import logging
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _lease(now, caller="c"):
    return {
        "caller": caller,
        "mb": 1000.0,
        "acquired_at": now,
        "last_seen": now,
        "priority": "P2",
        "preemptible": False,
        "pid": None,
        "preempt_flagged_at": None,
    }


@pytest.mark.asyncio
async def test_status_slow_live_read_does_not_block_heartbeat_or_release(broker_app):
    """Test: a slow /status nvidia-smi read does not stall heartbeat or release."""
    entered = asyncio.Event()

    async def slow_used_mb():
        entered.set()
        await asyncio.sleep(1.0)
        return 2048.0

    broker_app._last_used_mb = None
    broker_app._last_used_at = None
    with patch.object(broker_app, "_live_used_mb", side_effect=slow_used_mb):
        now = time.monotonic()
        broker_app._leases["H"] = _lease(now, "hb")
        broker_app._leases["R"] = _lease(now, "rel")
        broker_app._caller_to_lease["hb"] = "H"
        broker_app._caller_to_lease["rel"] = "R"

        transport = httpx.ASGITransport(app=broker_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            status_task = asyncio.create_task(client.get("/status"))
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            t0 = time.monotonic()
            hb, rel = await asyncio.wait_for(
                asyncio.gather(
                    client.post("/heartbeat", json={"lease_id": "H"}),
                    client.post("/release", json={"lease_id": "R"}),
                ),
                timeout=0.3,
            )
            elapsed = time.monotonic() - t0
            status_resp = await status_task

    assert elapsed < 0.3
    assert hb.status_code == 200
    assert rel.status_code == 200
    assert status_resp.status_code == 200
    assert status_resp.json()["nvidia_smi_used_mb"] == pytest.approx(2048.0)


def test_acquire_grants_when_nvidia_smi_raises(broker_app):
    """Test: /acquire grants without invoking nvidia-smi, even if the query raises."""
    client = TestClient(broker_app.app)

    def boom(field):
        raise RuntimeError("nvidia-smi should not run on acquire")

    with patch.object(broker_app, "_nvidia_smi_query", side_effect=boom):
        resp = client.post("/acquire", json={
            "caller": "no_smi",
            "mb": 2000.0,
            "timeout_s": 5.0,
        })

    assert resp.status_code == 200
    assert resp.json()["granted"] is True
    assert "lease_id" in resp.json()


def test_status_fresh_cache_skips_nvidia_smi(broker_app):
    """Test: /status with a fresh cache reports the cached value and does not call nvidia-smi."""
    client = TestClient(broker_app.app)
    broker_app._last_used_mb = 3210.0
    broker_app._last_used_at = time.monotonic()

    with patch.object(
        broker_app, "_nvidia_smi_query", side_effect=RuntimeError("should not run"),
    ) as query:
        resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json()["nvidia_smi_used_mb"] == pytest.approx(3210.0)
    query.assert_not_called()


def test_status_stale_cache_failed_query_returns_null(broker_app):
    """Test: /status with a stale cache and a failing query returns 200 and null used MB."""
    client = TestClient(broker_app.app)
    broker_app._last_used_mb = 3210.0
    broker_app._last_used_at = (
        time.monotonic() - (2 * broker_app.UNMANAGED_POLL_INTERVAL_SECONDS + 0.1)
    )

    with patch.object(
        broker_app, "_nvidia_smi_query", side_effect=RuntimeError("smi down"),
    ):
        resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json()["nvidia_smi_used_mb"] is None
