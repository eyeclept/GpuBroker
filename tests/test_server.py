"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for GpuBroker server.py. Covers admission, idempotent acquire,
admin clear, input validation, and Host header checks. No live GPU or
network. Sweeper and ledger-restore cases in this file simulate the
ledger in memory. They do not run `_sweep_expired_leases` or `startup`.
"""

import json
import logging
import os
import sys
import time
from unittest.mock import AsyncMock, Mock, patch
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# mock_env and broker_app fixtures live in tests/conftest.py, shared with
# test_priority_watchdog.py.

# Tests: Admission Formula

def test_admission_formula_accept(broker_app):
    """Test: Admission formula grants lease when budget allows."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "test_client",
            "mb": 2000.0,
            "timeout_s": 5.0,
        })
        
        assert resp.status_code == 200
        assert resp.json()["granted"] is True
        assert "lease_id" in resp.json()


def test_admission_formula_deny_insufficient_budget(broker_app):
    """Test: Admission formula denies lease when budget exhausted."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "test_client_big",
            "mb": 10000.0,
            "timeout_s": 1.0,
        })
        
        assert resp.status_code == 504


def test_admission_ignores_live_usage_spike_when_ledger_has_room(broker_app):
    """Test: live nvidia-smi usage well above tracked leases does not block a request the ledger can fit.

    Live usage is not part of the gate. The gate is ledger-only:
    reserved_total + requested <= budget. A high nvidia-smi reading
    (for example 8000 MB from desktop or other processes) does not
    block a request the ledger says fits. Live usage is logged as a
    warning if it exceeds SAFETY_MARGIN + reserved_total. It never
    blocks admission.
    """
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=8000.0):
        resp = client.post("/acquire", json={
            "caller": "test_max",
            "mb": 2000.0,
            "timeout_s": 5.0,
        })
        
        assert resp.status_code == 200
        assert resp.json()["granted"] is True


def test_double_counted_margin_regression_grants_physically_fitting_request(broker_app):
    """Test: a 5120 MB request that physically fits is granted (was wrongly refused).

    Regression for the double-counted-margin bug
    (Assist/ISSUE_double_counted_margin.md). Numbers from the incident:
    gpu_total_mb=12282, safety_margin_mb=4864, budget=7418.
    nvidia-smi shows 3222 MB in use (desktop baseline). Empty ledger
    (reserved=0, ollama=0). Request 5120 MB (float16 whisper model).

    Old formula: max(3222, 0) + 5120 = 8342 > 7418, wrongly refused (504).
    Fixed formula: 0 + 5120 = 5120 <= 7418, granted (200).

    Live usage is not part of the gate. Gate is ledger-only:
    reserved_total + req <= budget_mb, where budget_mb = total - SAFETY_MARGIN_MB.
    SAFETY_MARGIN_MB is baseline idle usage already inside every nvidia-smi
    reading, so the gate uses only the ledger (what we reserved).
    """
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    # Override defaults with exact bug-report numbers
    broker_app._gpu_total_mb = 12282.0
    broker_app._ollama_mb = 0.0
    broker_app.SAFETY_MARGIN_MB = 4864
    
    # Mock nvidia-smi to return the measured live usage from the incident
    with patch.object(broker_app, "_nvidia_smi_query", return_value=3222.0):
        resp = client.post("/acquire", json={
            "caller": "whisper_regression",
            "mb": 5120.0,
            "timeout_s": 5.0,
        })
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.json()}"
        assert resp.json()["granted"] is True


def test_double_counted_margin_regression_still_denies_oversized_request(broker_app):
    """Test: a genuinely oversized request is still denied (the fix does not grant everything).

    Same setup as the regression grant test, but request 9000 MB, which
    does not fit: reserved_total(0) + 9000 = 9000 > budget(7418), so it
    must be refused. Proves the fix is not "always grant".
    """
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    # Override defaults with exact bug-report numbers
    broker_app._gpu_total_mb = 12282.0
    broker_app._ollama_mb = 0.0
    broker_app.SAFETY_MARGIN_MB = 4864
    
    # Mock nvidia-smi to return the measured live usage from the incident
    with patch.object(broker_app, "_nvidia_smi_query", return_value=3222.0):
        resp = client.post("/acquire", json={
            "caller": "oversized_request",
            "mb": 9000.0,
            "timeout_s": 1.0,  # short timeout so the poll does not hang
        })
        
        assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.json()}"


# Tests: Idempotent Acquire

def test_acquire_idempotent_dedup_by_caller(broker_app):
    """Test: Same caller acquires twice -> returns existing lease_id."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp1 = client.post("/acquire", json={
            "caller": "dedup_test",
            "mb": 1000.0,
            "timeout_s": 5.0,
        })
        
        assert resp1.status_code == 200
        lease_id_1 = resp1.json()["lease_id"]
        
        resp2 = client.post("/acquire", json={
            "caller": "dedup_test",
            "mb": 1000.0,
            "timeout_s": 5.0,
        })
        
        assert resp2.status_code == 200
        lease_id_2 = resp2.json()["lease_id"]
        
        assert lease_id_1 == lease_id_2


# Tests: Queueing and Timeout

def test_queue_timeout_when_no_room(broker_app):
    """Test: acquire times out when the ledger budget has no room."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    # Gate is ledger-only: reserved_total + req <= budget_mb.
    # Default: _gpu_total_mb=12288, SAFETY_MARGIN_MB=1536, _ollama_mb=4096.
    # Budget = 12288 - 1536 = 10752. Reserved = 4096.
    # Available = 10752 - 4096 = 6656.
    # Request 7000 MB > available 6656 MB, so timeout/504.
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "test_timeout",
            "mb": 7000.0,
            "timeout_s": 1.0,
        })
        
        assert resp.status_code == 504


# Tests: Lease Sweeper

def test_sweeper_reclaims_stale_lease(broker_app):
    """Test: a lease with last_seen older than TTL is removed from the in-memory maps.

    This test ages last_seen and deletes matching entries itself.
    It does not call `_sweep_expired_leases`.
    """
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "stale_caller",
            "mb": 1000.0,
            "timeout_s": 5.0,
        })
        
        assert resp.status_code == 200
        lease_id = resp.json()["lease_id"]
        
        broker_app._leases[lease_id]["last_seen"] = time.monotonic() - 200
        
        now = time.monotonic()
        expired = [
            (lid, l) for lid, l in broker_app._leases.items()
            if now - l["last_seen"] > broker_app.LEASE_TTL_SECONDS
        ]
        
        for lid, lease in expired:
            del broker_app._leases[lid]
            if broker_app._caller_to_lease.get(lease["caller"]) == lid:
                del broker_app._caller_to_lease[lease["caller"]]
        
        assert lease_id not in broker_app._leases
        assert "stale_caller" not in broker_app._caller_to_lease


# Tests: Ledger Persistence (in-memory simulation)

def test_ledger_restore_resets_last_seen(broker_app):
    """Test: copying a lease dict and setting last_seen to now updates the in-memory maps.

    This test does not call `startup` or read `leases.json`.
    """
    lease_id = "test-lease-123"
    now = time.monotonic()
    
    broker_app._leases[lease_id] = {
        "caller": "restore_caller",
        "mb": 1000.0,
        "acquired_at": now - 100,
        "last_seen": now - 100,
    }
    broker_app._caller_to_lease["restore_caller"] = lease_id
    
    saved_leases = broker_app._leases.copy()
    broker_app._leases.clear()
    broker_app._caller_to_lease.clear()
    
    now_new = time.monotonic()
    for lid, lease in saved_leases.items():
        lease["last_seen"] = now_new
    broker_app._leases.update(saved_leases)
    for lid, lease in broker_app._leases.items():
        broker_app._caller_to_lease[lease["caller"]] = lid
    
    restored_last_seen = broker_app._leases[lease_id]["last_seen"]
    assert restored_last_seen >= now_new - 0.1


# Tests: Error Handling and Edge Cases

def test_release_unknown_lease(broker_app):
    """Test: release of an unknown lease returns False."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    resp = client.post("/release", json={
        "lease_id": "nonexistent_lease_id",
    })
    
    assert resp.status_code == 200
    assert resp.json()["released"] is False


def test_heartbeat_unknown_lease(broker_app):
    """Test: heartbeat on an unknown lease returns False."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    resp = client.post("/heartbeat", json={
        "lease_id": "nonexistent_lease_id",
    })
    
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_health_endpoint(broker_app):
    """Test: Health endpoint always returns OK."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    resp = client.get("/health")
    
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_status_endpoint(broker_app):
    """Test: Status endpoint returns full ledger and budget info."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.get("/status")
        
        assert resp.status_code == 200
        data = resp.json()
        assert "budget_mb" in data
        assert "gpu_total_mb" in data
        assert "safety_margin_mb" in data
        assert "reserved_mb" in data
        assert "ollama_mb" in data
        assert "nvidia_smi_used_mb" in data
        assert "leases" in data
        assert isinstance(data["leases"], dict)


def test_docs_and_openapi_are_disabled(broker_app):
    """Test: FastAPI /docs, /redoc, and /openapi.json return 404."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    host = {"Host": "localhost:8792"}
    assert client.get("/docs", headers=host).status_code == 404
    assert client.get("/redoc", headers=host).status_code == 404
    assert client.get("/openapi.json", headers=host).status_code == 404


# Tests: Admin Clear

def test_admin_clear_disabled_without_token(broker_app):
    """Test: /admin/clear returns 501 when GPU_BROKER_ADMIN_TOKEN is unset."""
    from fastapi.testclient import TestClient

    assert broker_app.ADMIN_TOKEN == ""
    client = TestClient(broker_app.app)

    resp = client.post("/admin/clear", json={"admin_token": "anything"})

    assert resp.status_code == 501


def test_admin_clear_rejects_wrong_token(broker_app):
    """Test: /admin/clear returns 403 on a token mismatch, and touches nothing."""
    from fastapi.testclient import TestClient

    broker_app.ADMIN_TOKEN = "correct-secret"
    broker_app._leases["existing"] = {"caller": "x", "mb": 100.0, "acquired_at": 0.0, "last_seen": 0.0}
    client = TestClient(broker_app.app)

    with patch.object(broker_app, "_list_gpu_compute_processes") as mock_list:
        resp = client.post("/admin/clear", json={"admin_token": "wrong-secret"})

        assert resp.status_code == 403
        mock_list.assert_not_called()
    assert "existing" in broker_app._leases  # ledger untouched on rejection


def test_admin_clear_dry_run_lists_without_killing(broker_app):
    """Test: dry_run=true reports targets without signaling or clearing the ledger."""
    from fastapi.testclient import TestClient

    broker_app.ADMIN_TOKEN = "correct-secret"
    broker_app._leases["existing"] = {"caller": "x", "mb": 100.0, "acquired_at": 0.0, "last_seen": 0.0}
    fake_processes = [{"pid": 4242, "process_name": "fake_proc", "used_mb": 512.0}]
    client = TestClient(broker_app.app)

    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0), \
         patch.object(broker_app, "_list_gpu_compute_processes", return_value=fake_processes), \
         patch.object(broker_app, "_kill_process") as mock_kill:
        resp = client.post("/admin/clear", json={"admin_token": "correct-secret", "dry_run": True})

        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["would_kill"] == fake_processes
        mock_kill.assert_not_called()
    assert "existing" in broker_app._leases  # dry run never touches the ledger


def test_admin_clear_real_run_kills_and_resets_ledger(broker_app):
    """Test: a real (non-dry-run) call signals every target and clears the ledger."""
    from fastapi.testclient import TestClient

    broker_app.ADMIN_TOKEN = "correct-secret"
    broker_app._leases["existing"] = {"caller": "x", "mb": 100.0, "acquired_at": 0.0, "last_seen": 0.0}
    broker_app._caller_to_lease["x"] = "existing"
    fake_processes = [
        {"pid": 4242, "process_name": "fake_proc_a", "used_mb": 512.0},
        {"pid": 4243, "process_name": "fake_proc_b", "used_mb": 256.0},
    ]
    client = TestClient(broker_app.app)

    with patch.object(broker_app, "_nvidia_smi_query", return_value=1024.0), \
         patch.object(broker_app, "_list_gpu_compute_processes", return_value=fake_processes), \
         patch.object(broker_app, "_kill_process", return_value="terminated") as mock_kill, \
         patch.object(broker_app, "_process_alive", return_value=False):
        resp = client.post("/admin/clear", json={"admin_token": "correct-secret", "dry_run": False})

        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is False
        assert data["leases_cleared"] == 1
        assert {p["pid"] for p in data["killed"]} == {4242, 4243}
        assert all(p["signal_result"] == "terminated" for p in data["killed"])
        assert mock_kill.call_count == 2
    assert broker_app._leases == {}
    assert broker_app._caller_to_lease == {}


def test_admin_clear_excludes_own_pid(broker_app):
    """Test: the broker's own process never appears in the kill target list."""
    from fastapi.testclient import TestClient

    broker_app.ADMIN_TOKEN = "correct-secret"
    my_pid = broker_app.os.getpid()
    fake_processes = [{"pid": my_pid, "process_name": "python3", "used_mb": 10.0}]
    client = TestClient(broker_app.app)

    with patch.object(broker_app, "_nvidia_smi_query", return_value=1024.0), \
         patch.object(broker_app, "_list_gpu_compute_processes", return_value=fake_processes):
        resp = client.post("/admin/clear", json={"admin_token": "correct-secret", "dry_run": True})

        assert resp.status_code == 200
        assert resp.json()["would_kill"] == []


# Tests: Input validation (S-01, S-04)

def test_acquire_rejects_negative_mb(broker_app):
    """Test: negative mb returns 422 and does not create a lease or change reserved_mb."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    reserved_before = broker_app._reserved_mb()
    resp = client.post("/acquire", json={
        "caller": "neg_mb",
        "mb": -20000.0,
        "timeout_s": 1.0,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}
    assert broker_app._reserved_mb() == reserved_before


@pytest.mark.parametrize("raw_mb", ["NaN", "Infinity", "-Infinity"])
def test_acquire_rejects_nonfinite_mb(broker_app, raw_mb):
    """Test: NaN/Inf mb returns 422 and does not create a lease."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    reserved_before = broker_app._reserved_mb()
    resp = client.post(
        "/acquire",
        content='{"caller":"nonfinite_mb","mb":%s,"timeout_s":1.0}' % raw_mb,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert broker_app._leases == {}
    assert broker_app._reserved_mb() == reserved_before


@pytest.mark.parametrize("raw_timeout", ["NaN", "Infinity", "-Infinity"])
def test_acquire_rejects_nonfinite_timeout_s(broker_app, raw_timeout):
    """Test: NaN/Inf timeout_s returns 422."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.post(
        "/acquire",
        content='{"caller":"nonfinite_timeout","mb":100.0,"timeout_s":%s}' % raw_timeout,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert broker_app._leases == {}


def test_acquire_rejects_negative_timeout_s(broker_app):
    """Test: negative timeout_s returns 422."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "neg_timeout",
        "mb": 100.0,
        "timeout_s": -1.0,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}


def test_acquire_rejects_timeout_s_above_cap(broker_app):
    """Test: timeout_s above 600 s returns 422."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "over_cap_timeout",
        "mb": 100.0,
        "timeout_s": broker_app.MAX_ACQUIRE_TIMEOUT_SECONDS + 0.1,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}


@pytest.mark.parametrize("pid", [0, -5])
def test_acquire_rejects_non_positive_pid(broker_app, pid):
    """Test: pid 0 and negative pid return 422."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={
        "caller": "bad_pid",
        "mb": 100.0,
        "timeout_s": 1.0,
        "pid": pid,
    })
    assert resp.status_code == 422
    assert broker_app._leases == {}


def test_acquire_never_fit_rejects_immediately(broker_app):
    """Test: mb larger than gpu_total minus the margin floor returns 422 without waiting."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    max_grantable = broker_app._gpu_total_mb - broker_app.SAFETY_MARGIN_MIN_MB
    too_big = max_grantable + 1.0
    t0 = time.monotonic()
    with patch.object(broker_app, "_refresh_proton_state", new_callable=AsyncMock):
        resp = client.post("/acquire", json={
            "caller": "never_fit",
            "mb": too_big,
            "timeout_s": 20.0,
        })
    elapsed = time.monotonic() - t0
    assert resp.status_code == 422
    assert elapsed < 5.0
    assert broker_app._pending == set()
    assert broker_app._leases == {}
    detail = str(resp.json()["detail"])
    assert f"{too_big:.0f}" in detail
    assert f"{max_grantable:.0f}" in detail


def test_acquire_fitting_request_grants(broker_app):
    """Test: a fitting request queued behind nothing is still granted."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "fits_ok",
            "mb": 2000.0,
            "timeout_s": 5.0,
        })
    assert resp.status_code == 200
    assert resp.json()["granted"] is True
    assert "lease_id" in resp.json()


def test_acquire_zero_mb_grants(broker_app):
    """Test: mb=0 is still a valid zero-size admission token and is granted."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "zero_token",
            "mb": 0.0,
            "timeout_s": 5.0,
        })
    assert resp.status_code == 200
    assert resp.json()["granted"] is True
    lease_id = resp.json()["lease_id"]
    assert broker_app._leases[lease_id]["mb"] == 0.0


def test_acquire_never_fit_still_returns_existing_lease(broker_app):
    """Test: a never-fit repeat acquire still returns the caller's existing lease."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        first = client.post("/acquire", json={
            "caller": "holder",
            "mb": 1000.0,
            "timeout_s": 5.0,
        })
    assert first.status_code == 200
    lease_id = first.json()["lease_id"]
    max_grantable = broker_app._gpu_total_mb - broker_app.SAFETY_MARGIN_MIN_MB
    with patch.object(broker_app, "_refresh_proton_state", new_callable=AsyncMock):
        second = client.post("/acquire", json={
            "caller": "holder",
            "mb": max_grantable + 1.0,
            "timeout_s": 20.0,
        })
    assert second.status_code == 200
    assert second.json()["lease_id"] == lease_id
    assert broker_app._pending == set()


@pytest.mark.asyncio
async def test_startup_skips_invalid_restored_mb(broker_app, mock_env, monkeypatch):
    """Test: startup skips restored leases whose mb is negative or non-finite."""
    ledger = {
        "leases": {
            "bad-neg": {
                "caller": "poison",
                "mb": -20000.0,
                "acquired_at": 0.0,
                "last_seen": 0.0,
            },
            "bad-nan": {
                "caller": "nan_caller",
                "mb": float("nan"),
                "acquired_at": 0.0,
                "last_seen": 0.0,
            },
            "good": {
                "caller": "ok_caller",
                "mb": 1000.0,
                "acquired_at": 0.0,
                "last_seen": 0.0,
            },
        },
        "proton_game_active": False,
        "proton_game_cleared_at": None,
    }
    with open(mock_env["ledger_path"], "w") as f:
        json.dump(ledger, f)
    monkeypatch.setattr(broker_app, "_nvidia_smi_query", lambda field: 12288.0)
    monkeypatch.setattr(broker_app, "_load_non_lease_samples", lambda: None)
    monkeypatch.setattr(broker_app, "_steam_proton_running", lambda: False)
    monkeypatch.setattr(broker_app.asyncio, "create_task", lambda task: task.close())
    await broker_app.startup()
    assert "bad-neg" not in broker_app._leases
    assert "bad-nan" not in broker_app._leases
    assert "good" in broker_app._leases
    assert broker_app._leases["good"]["mb"] == 1000.0
    assert broker_app._reserved_mb() == 1000.0
    assert broker_app._caller_to_lease.get("ok_caller") == "good"
    assert "poison" not in broker_app._caller_to_lease


# Tests: Host header allowlist (S-16)

_ALLOWED_HOST_HEADERS = (
    "localhost:8792",
    "127.0.0.1:8792",
    "host.containers.internal:8792",
    "[::1]:8792",
)


@pytest.mark.parametrize("host", _ALLOWED_HOST_HEADERS)
@pytest.mark.parametrize("path", ["/health", "/status"])
def test_trusted_host_allows_loopback_and_container_dns(broker_app, host, path):
    """Test: loopback and host.containers.internal Host values return 200."""
    from fastapi.testclient import TestClient

    broker_app._last_used_mb = 1000.0
    broker_app._last_used_at = time.monotonic()
    client = TestClient(broker_app.app)
    resp = client.get(path, headers={"Host": host})
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "host,status",
    [
        ("LocalHost:8792", 200),
        ("HOST.CONTAINERS.INTERNAL:8792", 200),
        ("localhost.:8792", 200),
        ("localhost.", 200),
        ("LocalHost.", 200),
        ("evil.example", 400),
        ("EVIL.example", 400),
    ],
)
def test_trusted_host_case_and_trailing_dot(broker_app, host, status):
    """Test: mixed-case and trailing-dot Host values match; evil stays 400."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.get("/health", headers={"Host": host})
    assert resp.status_code == status


def test_trusted_host_rejects_evil_example_on_status_and_acquire(broker_app):
    """Test: Host evil.example returns 400 on GET /status and POST /acquire."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    status = client.get("/status", headers={"Host": "evil.example"})
    assert status.status_code == 400
    acquire = client.post(
        "/acquire",
        json={"caller": "x", "mb": 0.0, "timeout_s": 1.0},
        headers={"Host": "evil.example"},
    )
    assert acquire.status_code == 400


def test_trusted_host_env_extra_is_accepted(mock_env, monkeypatch):
    """Test: GPU_BROKER_ALLOWED_HOSTS appends a host that then gets 200."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GPU_BROKER_ALLOWED_HOSTS", "Extra.Example")
    if "server" in sys.modules:
        del sys.modules["server"]
    import server as srv

    srv._leases.clear()
    srv._caller_to_lease.clear()
    srv._pending.clear()
    srv._pending_notifications.clear()
    srv._gpu_total_mb = 12288.0
    srv._ollama_mb = 4096.0
    srv._persist_ledger_sync = Mock()
    srv._persist_ledger = AsyncMock()

    client = TestClient(srv.app)
    resp = client.get("/health", headers={"Host": "extra.example"})
    assert resp.status_code == 200
    mixed = client.get("/health", headers={"Host": "EXTRA.EXAMPLE"})
    assert mixed.status_code == 200
    denied = client.get("/health", headers={"Host": "evil.example"})
    assert denied.status_code == 400


def test_trusted_host_testserver_is_not_a_production_default(mock_env, monkeypatch):
    """Test: TestClient's testserver Host is 400 unless added via env."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GPU_BROKER_ALLOWED_HOSTS", "")
    if "server" in sys.modules:
        del sys.modules["server"]
    import server as srv

    client = TestClient(srv.app)
    assert client.get("/health").status_code == 400
    assert client.get("/health", headers={"Host": "localhost:8792"}).status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
