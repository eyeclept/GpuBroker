"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for the unmanaged-heavy-load (game) handling in server.py: the
baseline margin must ignore heavy samples, the hysteresis state machine,
the container-aware overhead metric, preemptible-lease flagging, the 503
refusal of every new acquire, and the zero-size FIFO bypass. No live GPU
or network.
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lease(caller, preemptible, flagged=None, mb=8000.0):
    return {
        "caller": caller, "mb": mb, "acquired_at": 0.0, "last_seen": 0.0,
        "priority": "P2", "preemptible": preemptible, "pid": None,
        "preempt_flagged_at": flagged,
    }


CONTAINER_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/user.slice/"
    "libpod-d1ecaf800b2d.scope/container\n"
)
HOST_CGROUP = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-gnome.scope\n"


def _gpu(monkeypatch, srv, used_mb, procs, cgroups):
    """
    Patch live GPU reads. procs: list of (pid, used_mb). cgroups: pid ->
    /proc/<pid>/cgroup text, or None for an unreadable file.
    """
    monkeypatch.setattr(srv, "_live_used_mb", AsyncMock(return_value=used_mb))
    monkeypatch.setattr(srv, "_list_gpu_compute_processes", lambda: [
        {"pid": pid, "process_name": "p", "used_mb": mb} for pid, mb in procs
    ])

    def fake_open(path, *args, **kwargs):
        text = cgroups.get(int(path.split("/")[2]))
        if text is None:
            raise FileNotFoundError(path)
        return io.StringIO(text)
    monkeypatch.setattr(srv, "open", fake_open, raising=False)


async def _one_poll(srv):
    with patch.object(srv.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await srv._poll_unmanaged_load()


def _samples(srv, baseline_n, heavy_n, baseline_mb=1500.0, heavy_mb=10000.0):
    srv._non_lease_samples[:] = (
        [{"ts": 0.0, "mb": baseline_mb, "locked": "unlocked"}] * baseline_n
        + [{"ts": 0.0, "mb": heavy_mb, "locked": "unlocked"}] * heavy_n
    )


# Tests: baseline margin ignores games

def test_margin_percentile_excludes_heavy_samples(broker_app):
    """Test: 15 per cent game samples must not drag p95 up to the game's footprint."""
    _samples(broker_app, baseline_n=170, heavy_n=30)
    broker_app._recompute_margin_locked()
    assert broker_app._computed_margin_mb == pytest.approx(1500.0)


def test_margin_falls_back_when_only_heavy_samples_exist(broker_app):
    """Test: with no baseline samples left, the margin is the configured fallback, not the game's usage."""
    _samples(broker_app, baseline_n=0, heavy_n=200)
    broker_app._recompute_margin_locked()
    assert broker_app._computed_margin_mb is None
    assert broker_app._safety_margin_mb() == float(broker_app.SAFETY_MARGIN_MB)


def test_margin_sample_exactly_at_threshold_is_heavy(broker_app):
    """Test: the threshold is inclusive on the heavy side, matching _update_unmanaged_state."""
    threshold = broker_app.UNMANAGED_YIELD_OVERHEAD_MB
    _samples(broker_app, baseline_n=100, heavy_n=100, heavy_mb=threshold)
    broker_app._recompute_margin_locked()
    assert broker_app._computed_margin_mb == pytest.approx(1500.0)

def test_margin_excludes_light_proton_game_samples(broker_app):
    """Test: Proton samples below the VRAM trigger do not raise p95 after the game ends."""
    _samples(broker_app, baseline_n=100, heavy_n=0)
    broker_app._non_lease_samples.extend([
        {"ts": 0.0, "mb": 3800.0, "locked": "unlocked", "game": True}
        for _ in range(20)
    ])
    broker_app._recompute_margin_locked()
    assert broker_app._computed_margin_mb == pytest.approx(1500.0)

def test_light_proton_game_samples_stay_excluded_after_restart(broker_app, monkeypatch, tmp_path):
    """Test: `_load_non_lease_samples` keeps the game tag and the baseline p95.

    This test writes a sample file and loads it. It does not restart the process.
    """
    path = tmp_path / "usage.json"
    now = time.time()
    samples = (
        [{"ts": now, "mb": 1500.0, "locked": "unlocked"} for _ in range(100)]
        + [{"ts": now, "mb": 3800.0, "locked": "unlocked", "game": True} for _ in range(20)]
    )
    path.write_text(json.dumps({"samples": samples}))
    monkeypatch.setattr(broker_app, "NON_LEASE_USAGE_PATH", str(path))
    broker_app._load_non_lease_samples()
    assert len(broker_app._non_lease_samples) == 120
    assert sum(s["game"] for s in broker_app._non_lease_samples) == 20
    assert broker_app._computed_margin_mb == pytest.approx(1500.0)


# Tests: hysteresis state machine

def test_state_needs_consecutive_over_threshold_polls(broker_app):
    """Test: a single spike does not declare heavy load; a dip resets the count."""
    over = broker_app.UNMANAGED_YIELD_OVERHEAD_MB + 1.0
    assert broker_app._update_unmanaged_state(over, 0.0) is False
    assert broker_app._update_unmanaged_state(1000.0, 5.0) is False
    assert broker_app._update_unmanaged_state(over, 10.0) is False
    assert broker_app._update_unmanaged_state(over, 15.0) is True


def test_state_clears_only_after_continuous_quiet_period(broker_app):
    """Test: heavy load ends only after UNMANAGED_CLEAR_SECONDS below threshold; a spike restarts the clock."""
    over = broker_app.UNMANAGED_YIELD_OVERHEAD_MB + 1.0
    clear = broker_app.UNMANAGED_CLEAR_SECONDS
    broker_app._update_unmanaged_state(over, 0.0)
    assert broker_app._update_unmanaged_state(over, 5.0) is True

    assert broker_app._update_unmanaged_state(500.0, 10.0) is True
    assert broker_app._update_unmanaged_state(500.0, 10.0 + clear - 1.0) is True
    # Spike inside the quiet period restarts the clock.
    assert broker_app._update_unmanaged_state(over, 10.0 + clear) is True
    assert broker_app._update_unmanaged_state(500.0, 20.0 + clear) is True
    assert broker_app._update_unmanaged_state(500.0, 20.0 + clear + clear - 1.0) is True
    assert broker_app._update_unmanaged_state(500.0, 20.0 + clear + clear) is False


# Tests: flagging

def test_flag_preemptible_leases_skips_non_preemptible_and_keeps_existing_flag(broker_app):
    """Test: only unflagged preemptible leases are flagged; an existing flag time is not reset."""
    broker_app._leases["A"] = _lease("freetoken", preemptible=True)
    broker_app._leases["B"] = _lease("whisper", preemptible=False)
    broker_app._leases["C"] = _lease("other", preemptible=True, flagged=7.0)

    newly = broker_app._flag_preemptible_leases_locked(99.0, "unmanaged")

    assert newly == ["A"]
    assert broker_app._leases["A"]["preempt_flagged_at"] == 99.0
    assert broker_app._leases["B"]["preempt_flagged_at"] is None
    assert broker_app._leases["C"]["preempt_flagged_at"] == 7.0


# Tests: container-aware overhead (poller)

@pytest.mark.asyncio
async def test_poller_flags_preemptible_lease_when_game_detected(broker_app, monkeypatch):
    """Test: one poll tick with a game-sized host load flags the preemptible holder's heartbeat."""
    monkeypatch.setattr(broker_app, "UNMANAGED_ENTER_SAMPLES", 1)
    broker_app._leases["A"] = _lease("freetoken", preemptible=True, mb=8000.0)
    broker_app._caller_to_lease["freetoken"] = "A"
    # FreeToken container uses 7000; the game is a pure graphics host process.
    _gpu(monkeypatch, broker_app, 17000.0, [(100, 7000.0)], {100: CONTAINER_CGROUP})

    await _one_poll(broker_app)

    assert broker_app._unmanaged_heavy is True
    hb = await broker_app.heartbeat(broker_app.HeartbeatRequest(lease_id="A"))
    assert hb["preempt"] is True


@pytest.mark.asyncio
async def test_overhead_is_used_minus_container_processes_not_leases(broker_app, monkeypatch):
    """Test: overhead subtracts container process memory, ignores lease size, counts host processes."""
    broker_app._leases["A"] = _lease("freetoken", preemptible=True, mb=8382.0)
    _gpu(
        monkeypatch, broker_app, 9000.0,
        [(100, 7000.0), (200, 300.0)],
        {100: CONTAINER_CGROUP, 200: HOST_CGROUP},
    )

    await _one_poll(broker_app)

    assert broker_app._unmanaged_overhead_mb == pytest.approx(2000.0)
    assert broker_app._unmanaged_heavy is False
    status = await broker_app.status()
    assert status["unmanaged_overhead_mb"] == pytest.approx(2000.0)


@pytest.mark.asyncio
async def test_game_behind_oversized_lease_is_detected(broker_app, monkeypatch):
    """Test: a model using less than its lease no longer masks a 5000 MB host game."""
    monkeypatch.setattr(broker_app, "UNMANAGED_ENTER_SAMPLES", 1)
    broker_app._ollama_mb = 0.0
    broker_app._leases["A"] = _lease("freetoken", preemptible=True, mb=8382.0)
    used = 7104.0 + 5000.0
    # The old lease-based formula stays under the threshold.
    assert used - 8382.0 < broker_app.UNMANAGED_YIELD_OVERHEAD_MB
    _gpu(monkeypatch, broker_app, used, [(100, 7104.0)], {100: CONTAINER_CGROUP})

    await _one_poll(broker_app)

    assert broker_app._unmanaged_overhead_mb == pytest.approx(5000.0)
    assert broker_app._unmanaged_heavy is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unreadable_cgroup", "nvidia_smi_failed"])
async def test_scan_failure_falls_back_to_lease_formula(broker_app, monkeypatch, caplog, failure):
    """Test: an unknown container process is not counted as a game; the poll uses used - leases - Ollama."""
    monkeypatch.setattr(broker_app, "UNMANAGED_ENTER_SAMPLES", 1)
    broker_app._ollama_mb = 0.0
    broker_app._leases["A"] = _lease("freetoken", preemptible=True, mb=8000.0)
    _gpu(monkeypatch, broker_app, 9500.0, [(100, 7000.0)], {100: None})
    if failure == "nvidia_smi_failed":
        def boom():
            raise subprocess.CalledProcessError(9, "nvidia-smi")
        monkeypatch.setattr(broker_app, "_list_gpu_compute_processes", boom)

    with caplog.at_level(logging.WARNING, logger="server"):
        await _one_poll(broker_app)

    assert broker_app._unmanaged_overhead_mb == pytest.approx(1500.0)
    assert broker_app._unmanaged_heavy is False
    assert broker_app._leases["A"]["preempt_flagged_at"] is None
    assert "lease-based overhead" in caplog.text


@pytest.mark.asyncio
async def test_unmanaged_poller_updates_used_mb_cache(broker_app, monkeypatch):
    """Test: one unmanaged-load poll tick stores memory.used in the module cache."""
    assert broker_app._last_used_mb is None
    assert broker_app._last_used_at is None
    _gpu(monkeypatch, broker_app, 4321.0, [], {})
    before = time.monotonic()
    await _one_poll(broker_app)
    after = time.monotonic()
    assert broker_app._last_used_mb == pytest.approx(4321.0)
    assert broker_app._last_used_at is not None
    assert before <= broker_app._last_used_at <= after


# Tests: acquire gate

def _enter_heavy(srv):
    over = srv.UNMANAGED_YIELD_OVERHEAD_MB + 1.0
    for t in range(srv.UNMANAGED_ENTER_SAMPLES):
        srv._update_unmanaged_state(over, float(t))
    assert srv._unmanaged_heavy is True


@pytest.mark.parametrize("body", [
    {"caller": "freetoken", "mb": 100.0, "preemptible": True},
    {"caller": "whisper", "mb": 100.0},
    {"caller": "oaie-proxy:ollama:abc", "mb": 0},
])
def test_every_acquire_refused_with_503_while_heavy_then_granted_after_clear(broker_app, body):
    """Test: during a game any new acquire gets 503 + Retry-After and is not queued; after clearing it is granted."""
    from fastapi.testclient import TestClient

    _enter_heavy(broker_app)
    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=1500.0):
        resp = client.post("/acquire", json={**body, "timeout_s": 5.0})

        assert resp.status_code == 503
        assert resp.json()["detail"] == broker_app.UNMANAGED_BUSY_DETAIL
        assert resp.headers["retry-after"] == str(int(broker_app.UNMANAGED_CLEAR_SECONDS))
        assert not broker_app._pending
        assert not broker_app._leases

        broker_app._update_unmanaged_state(500.0, 100.0)
        broker_app._update_unmanaged_state(500.0, 100.0 + broker_app.UNMANAGED_CLEAR_SECONDS)
        assert broker_app._unmanaged_heavy is False
        resp = client.post("/acquire", json={**body, "timeout_s": 5.0})

    assert resp.status_code == 200
    assert resp.json()["granted"] is True


def test_dedup_acquire_while_heavy_returns_existing_lease(broker_app):
    """Test: a caller that already holds a lease gets it back during a game, with no new lease."""
    from fastapi.testclient import TestClient

    broker_app._leases["A"] = _lease("whisper", preemptible=False, mb=2000.0)
    broker_app._caller_to_lease["whisper"] = "A"
    _enter_heavy(broker_app)
    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=1500.0):
        resp = client.post("/acquire", json={"caller": "whisper", "mb": 2000.0, "timeout_s": 5.0})

    assert resp.status_code == 200
    assert resp.json() == {"granted": True, "lease_id": "A"}
    assert list(broker_app._leases) == ["A"]


# Tests: zero-size FIFO bypass

@pytest.mark.asyncio
async def test_zero_size_acquire_granted_while_unfittable_waiter_queued(broker_app, monkeypatch):
    """Test: an mb=0 request is not starved by an earlier non-zero waiter that cannot fit."""
    monkeypatch.setattr(broker_app, "_live_used_mb", AsyncMock(return_value=6000.0))
    monkeypatch.setattr(broker_app, "_refresh_proton_state", AsyncMock())
    # budget 12288 - 1536 = 10752; Ollama 4096 + 9000 does not fit.
    big = asyncio.create_task(broker_app.acquire(
        broker_app.AcquireRequest(caller="flux", mb=9000.0, timeout_s=30.0),
    ))
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(broker_app._pending) == 1

    try:
        result = await asyncio.wait_for(
            broker_app.acquire(broker_app.AcquireRequest(caller="oaie-proxy:ollama:abc", mb=0, timeout_s=5.0)),
            timeout=2.0,
        )
        assert result["granted"] is True
        assert len(broker_app._pending) == 1
        assert not big.done()
    finally:
        big.cancel()
        with pytest.raises(asyncio.CancelledError):
            await big
