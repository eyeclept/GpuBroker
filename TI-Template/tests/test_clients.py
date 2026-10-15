"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Unit tests for the TI-Template async and sync broker clients.
    HTTP is faked with httpx.MockTransport. No live broker.
"""
# Imports
import asyncio
import inspect
import json
import logging

import httpx
import pytest

import async_broker_client as async_mod
import sync_broker_client as sync_mod

# Globals
logger = logging.getLogger(__name__)

MODULES = [async_mod, sync_mod]
MODULE_IDS = ["async", "sync"]


def invoke(fn, *args, **kwargs):
    """
    Input:      fn -- _broker_* function, plus its args
    Output:     return value, with coroutines driven to completion
    Details:    Lets the same test body cover async and sync variants.
    """
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def install_transport(monkeypatch, mod, handler):
    """
    Input:      monkeypatch, client module, MockTransport handler
    Output:     None
    Details:    Injects transport into both Client types and silences
                backoff sleeps so retry tests do not wait.
    """
    transport = httpx.MockTransport(handler)
    real_sync = httpx.Client
    real_async = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_sync(*args, **kwargs)

    def aclient(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "Client", client)
    monkeypatch.setattr(mod.httpx, "AsyncClient", aclient)
    if hasattr(mod, "time"):
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    if hasattr(mod, "asyncio"):
        async def _sleep(_s):
            return None
        monkeypatch.setattr(mod.asyncio, "sleep", _sleep)


@pytest.fixture(params=MODULES, ids=MODULE_IDS)
def mod(request):
    return request.param


def test_no_module_lease_global():
    """Test: neither variant keeps a module-level lease_id."""
    assert "lease_id" not in async_mod.__dict__
    assert "lease_id" not in sync_mod.__dict__


def test_shared_names_and_signatures():
    """Test: async and sync export the same names and parameter lists."""
    names = [
        "AcquireResult",
        "HeartbeatResult",
        "BrokerError",
        "BrokerNoRoom",
        "BrokerBusy",
        "BrokerRejected",
        "BrokerUnavailable",
        "_broker_acquire",
        "_broker_release",
        "_broker_heartbeat",
    ]
    for name in names:
        assert hasattr(async_mod, name)
        assert hasattr(sync_mod, name)
    for fn_name in ("_broker_acquire", "_broker_release", "_broker_heartbeat"):
        async_params = list(inspect.signature(getattr(async_mod, fn_name)).parameters)
        sync_params = list(inspect.signature(getattr(sync_mod, fn_name)).parameters)
        assert async_params == sync_params
    assert list(inspect.signature(async_mod._broker_release).parameters) == ["lease_id"]
    assert list(inspect.signature(async_mod._broker_heartbeat).parameters) == ["lease_id"]


def test_fail_open_source_default_is_false():
    """Test: both templates default BROKER_FAIL_OPEN to false."""
    marker = 'os.environ.get("BROKER_FAIL_OPEN", "false")'
    assert marker in inspect.getsource(async_mod)
    assert marker in inspect.getsource(sync_mod)


def test_no_fastapi_dependency():
    """Test: templates do not import FastAPI."""
    assert "fastapi" not in inspect.getsource(async_mod)
    assert "fastapi" not in inspect.getsource(sync_mod)


def test_acquire_200_without_warning(mod, monkeypatch):
    """Test: 200 with lease_id and no warning returns AcquireResult."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/acquire")
        body = json.loads(request.content)
        assert body["mb"] == 5120
        assert "caller" in body
        return httpx.Response(200, json={"granted": True, "lease_id": "abc123"})

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_acquire, 5120)
    assert result == mod.AcquireResult(lease_id="abc123", warning=None)


def test_acquire_200_with_warning(mod, monkeypatch, caplog):
    """Test: 200 with warning logs it at WARNING and returns it."""
    notice = "previous lease was force-terminated by the watchdog"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"granted": True, "lease_id": "lease-w", "warning": notice},
        )

    install_transport(monkeypatch, mod, handler)
    with caplog.at_level(logging.WARNING):
        result = invoke(mod._broker_acquire, 1024)
    assert result.lease_id == "lease-w"
    assert result.warning == notice
    assert notice in caplog.text


def test_acquire_504_raises_no_room(mod, monkeypatch):
    """Test: 504 raises BrokerNoRoom with Retry-After and does not retry."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            504,
            json={"detail": "timeout waiting for GPU headroom"},
            headers={"Retry-After": "10"},
        )

    install_transport(monkeypatch, mod, handler)
    with pytest.raises(mod.BrokerNoRoom) as exc:
        invoke(mod._broker_acquire, 2048)
    assert exc.value.retry_after == 10.0
    assert hits["n"] == 1


@pytest.mark.parametrize(
    "detail,retry",
    [("unmanaged_gpu_load", "120"), ("too_many_leases", "5")],
)
def test_acquire_503_raises_busy_no_retry_no_fail_open(
    mod, monkeypatch, detail, retry,
):
    """Test: 503 raises BrokerBusy even when BROKER_FAIL_OPEN is true."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            503,
            json={"detail": detail},
            headers={"Retry-After": retry},
        )

    install_transport(monkeypatch, mod, handler)
    monkeypatch.setattr(mod, "BROKER_FAIL_OPEN", True)
    with pytest.raises(mod.BrokerBusy) as exc:
        invoke(mod._broker_acquire, 4096)
    assert not isinstance(exc.value, mod.BrokerNoRoom)
    assert exc.value.detail == detail
    assert exc.value.retry_after == float(retry)
    assert hits["n"] == 1


def test_acquire_422_raises_rejected(mod, monkeypatch):
    """Test: 422 raises BrokerRejected and does not retry."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            422,
            json={"detail": "request larger than the GPU can ever grant"},
        )

    install_transport(monkeypatch, mod, handler)
    with pytest.raises(mod.BrokerRejected) as exc:
        invoke(mod._broker_acquire, 999999)
    assert exc.value.status_code == 422
    assert hits["n"] == 1


def test_acquire_400_raises_rejected(mod, monkeypatch):
    """Test: 400 (bad priority) raises BrokerRejected."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "unknown priority"})

    install_transport(monkeypatch, mod, handler)
    with pytest.raises(mod.BrokerRejected) as exc:
        invoke(mod._broker_acquire, 512)
    assert exc.value.status_code == 400


def test_acquire_connect_error_fail_closed(mod, monkeypatch):
    """Test: connect errors retry then raise BrokerUnavailable."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    install_transport(monkeypatch, mod, handler)
    monkeypatch.setattr(mod, "BROKER_FAIL_OPEN", False)
    with pytest.raises(mod.BrokerUnavailable):
        invoke(mod._broker_acquire, 512)
    assert hits["n"] == 5


def test_acquire_connect_error_fail_open(mod, monkeypatch):
    """Test: connect errors retry then return None when fail-open."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    install_transport(monkeypatch, mod, handler)
    monkeypatch.setattr(mod, "BROKER_FAIL_OPEN", True)
    result = invoke(mod._broker_acquire, 512)
    assert result is None
    assert hits["n"] == 5


def test_heartbeat_ok(mod, monkeypatch):
    """Test: live lease maps to ok=True, preempt=False."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"lease_id": "live-1"}
        return httpx.Response(
            200, json={"ok": True, "preempt": False, "reason": ""},
        )

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_heartbeat, "live-1")
    assert result == mod.HeartbeatResult(ok=True, preempt=False, reason="")


def test_heartbeat_preempt(mod, monkeypatch):
    """Test: preempt=True is returned with the broker reason."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": True, "preempt": True, "reason": "fairness aging"},
        )

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_heartbeat, "held-1")
    assert result.ok is True
    assert result.preempt is True
    assert result.reason == "fairness aging"


def test_heartbeat_lost(mod, monkeypatch):
    """Test: ok=false maps to a lost lease, not a transport miss."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False})

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_heartbeat, "dead-1")
    assert result.ok is False
    assert result.preempt is False


def test_heartbeat_transport_error(mod, monkeypatch):
    """Test: transport error maps to ok=None, not ok=False."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_heartbeat, "held-2")
    assert result.ok is None
    assert result.preempt is False


def test_heartbeat_skips_missing_lease(mod, monkeypatch):
    """Test: None lease_id does not call the broker."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("heartbeat must not POST without a lease_id")

    install_transport(monkeypatch, mod, handler)
    result = invoke(mod._broker_heartbeat, None)
    assert result == mod.HeartbeatResult(ok=True, preempt=False, reason="")


def test_release_posts_explicit_lease_id(mod, monkeypatch):
    """Test: release sends the caller-supplied lease_id."""
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/release")
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"released": True})

    install_transport(monkeypatch, mod, handler)
    invoke(mod._broker_release, "explicit-lease")
    assert posted == [{"lease_id": "explicit-lease"}]


def test_release_skips_missing_lease(mod, monkeypatch):
    """Test: None lease_id does not call the broker."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("release must not POST without a lease_id")

    install_transport(monkeypatch, mod, handler)
    invoke(mod._broker_release, None)
