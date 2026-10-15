"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Shared pytest fixtures for GpuBroker's server.py test suite
(test_server.py, test_priority_watchdog.py). Provides a fresh, isolated
server module import per test with mocked persistence and no live GPU or
network -- see broker_app's own docstring.
"""

import logging
import os
import sys
from unittest.mock import AsyncMock, Mock

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
    # TestClient default Host is testserver. httpx ASGI tests use
    # base_url http://test. Neither is a production Host. Append them
    # so unit tests pass without opening the production allowlist.
    monkeypatch.setenv("GPU_BROKER_ALLOWED_HOSTS", "testserver,test")
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
    srv._pending_notifications.clear()
    srv._gpu_total_mb = 12288.0
    srv._ollama_mb = 4096.0

    srv._persist_ledger_sync = Mock()
    srv._persist_ledger = AsyncMock()

    return srv
