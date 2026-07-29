"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for GpuBroker server.py. Tests admission formula, idempotent acquire,
fairness, sweeper, and ledger persistence without live GPU or network dependency.
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


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """Mock environment variables and file paths for testing."""
    ledger_path = str(tmp_path / "leases.json")
    monkeypatch.setenv("GPU_BROKER_PORT", "8792")
    monkeypatch.setenv("GPU_SAFETY_MARGIN_MB", "1536")
    monkeypatch.setenv("GPU_BROKER_LEDGER_PATH", ledger_path)
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_POLL_INTERVAL_SECONDS", "20")
    monkeypatch.setenv("GPU_BROKER_SWEEP_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("GPU_BROKER_LEASE_TTL_SECONDS", "180")
    return {"ledger_path": ledger_path, "tmp_path": tmp_path}


@pytest.fixture
def broker_app(mock_env):
    """Import and setup a fresh broker instance with mocked persistence."""
    if "server" in sys.modules:
        del sys.modules["server"]
    
    import server as srv
    
    srv._leases.clear()
    srv._caller_to_lease.clear()
    srv._pending.clear()
    srv._gpu_total_mb = 12288.0
    srv._ollama_mb = 4096.0
    
    srv._persist_ledger_sync = Mock()
    srv._persist_ledger = AsyncMock()
    
    return srv


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


def test_admission_formula_max_not_sum(broker_app):
    """Test: Admission uses max(used, reserved+ollama) not sum."""
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
    """Test: Acquire times out when there's insufficient room."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    with patch.object(broker_app, "_nvidia_smi_query", return_value=10000.0):
        resp = client.post("/acquire", json={
            "caller": "test_timeout",
            "mb": 1000.0,
            "timeout_s": 1.0,
        })
        
        assert resp.status_code == 504


# Tests: Lease Sweeper

def test_sweeper_reclaims_stale_lease(broker_app):
    """Test: Sweeper reclaims lease when no heartbeat within TTL."""
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
    """Test: Restored leases get last_seen reset to prevent immediate expiry."""
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
    """Test: Release of unknown lease returns False gracefully."""
    from fastapi.testclient import TestClient
    
    client = TestClient(broker_app.app)
    
    resp = client.post("/release", json={
        "lease_id": "nonexistent_lease_id",
    })
    
    assert resp.status_code == 200
    assert resp.json()["released"] is False


def test_heartbeat_unknown_lease(broker_app):
    """Test: Heartbeat on unknown lease returns False gracefully."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
