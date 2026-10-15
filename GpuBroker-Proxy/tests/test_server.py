"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for GpuBroker-Proxy server.py. Tests registry resolution, model
extraction, header stripping, and the acquire/forward/release request flow
against a mocked broker and mocked backend -- no live GPU, broker, or
Ollama dependency.
"""

import asyncio
import json
import logging
import os
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """Mock environment variables and registry path for testing."""
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "backends": [
            {"name": "ollama", "base_url": "http://ollama:11434", "models": [], "default": True},
        ],
    }))
    monkeypatch.setenv("GPUBROKER_PROXY_PORT", "11434")
    monkeypatch.setenv("BROKER_URL", "http://broker.invalid:8792")
    monkeypatch.setenv("BROKER_FAIL_OPEN", "true")
    monkeypatch.setenv("BROKER_ACQUIRE_TIMEOUT_S", "10.0")
    monkeypatch.setenv("PROXY_RETRY_AFTER_SECONDS", "3.0")
    monkeypatch.setenv("PROXY_REGISTRY_PATH", str(registry_path))
    return {"registry_path": str(registry_path), "tmp_path": tmp_path}


@pytest.fixture
def proxy_app(mock_env):
    """Import and set up a fresh proxy instance."""
    if "server" in sys.modules:
        del sys.modules["server"]

    import server as srv

    srv._registry = srv._load_registry(mock_env["registry_path"])
    return srv


def _make_mock_broker_client(acquire_response=None, forward_response=None, forward_body=b'{"ok":true}'):
    """
    Build an AsyncMock standing in for the shared httpx.AsyncClient.
    Wires both the broker acquire/release calls and the backend forward call.
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    if acquire_response is not None:
        client.post.return_value = acquire_response
    client.build_request = Mock(return_value=Mock())
    if forward_response is not None:
        client.send.return_value = forward_response
    return client


def _mock_response(status_code=200, json_body=None, headers=None):
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json = Mock(return_value=json_body or {})
    resp.raise_for_status = Mock()
    return resp


# Tests: registry resolution

def test_resolve_backend_default_used_when_no_model(proxy_app):
    """Test: with model=None, resolution falls to the default-flagged backend."""
    backend = proxy_app._resolve_backend(None)
    assert backend is not None
    assert backend["name"] == "ollama"


def test_resolve_backend_exact_model_match(proxy_app):
    """Test: a backend with a matching declared model wins over the default."""
    proxy_app._registry = [
        {"name": "ollama", "base_url": "http://ollama:11434", "models": [], "default": True},
        {"name": "whisper", "base_url": "http://whisper:9000", "models": ["whisper-1"], "default": False},
    ]
    backend = proxy_app._resolve_backend("whisper-1")
    assert backend["name"] == "whisper"


def test_resolve_backend_no_match_no_default_returns_none(proxy_app):
    """Test: an unmatched model with no default-flagged backend resolves to None."""
    proxy_app._registry = [
        {"name": "whisper", "base_url": "http://whisper:9000", "models": ["whisper-1"], "default": False},
    ]
    assert proxy_app._resolve_backend("some-other-model") is None


def test_resolve_backend_empty_registry_returns_none(proxy_app):
    """Test: an empty registry resolves nothing, regardless of model."""
    proxy_app._registry = []
    assert proxy_app._resolve_backend("anything") is None
    assert proxy_app._resolve_backend(None) is None


# Tests: model extraction

def test_extract_model_from_json_body(proxy_app):
    """Test: model field is pulled out of a well-formed JSON body."""
    body = json.dumps({"model": "qwen2.5:3b", "messages": []}).encode()
    assert proxy_app._extract_model(body) == "qwen2.5:3b"


def test_extract_model_empty_body_returns_none(proxy_app):
    """Test: an empty body (for example GET /v1/models) returns None with no parse."""
    assert proxy_app._extract_model(b"") is None


def test_extract_model_malformed_body_returns_none(proxy_app):
    """Test: a non-JSON body (for example a multipart upload) returns None, not a raise."""
    assert proxy_app._extract_model(b"not json at all") is None


def test_extract_model_missing_field_returns_none(proxy_app):
    """Test: valid JSON without a model field returns None."""
    body = json.dumps({"messages": []}).encode()
    assert proxy_app._extract_model(body) is None


# Tests: header stripping

def test_strip_hop_by_hop_removes_framing_headers(proxy_app):
    """Test: Connection, Transfer-Encoding, Host, Content-Length are stripped."""
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
        "Host": "localhost",
        "Content-Length": "42",
        "Authorization": "Bearer x",
    }
    stripped = proxy_app._strip_hop_by_hop(headers)
    assert stripped == {"Content-Type": "application/json", "Authorization": "Bearer x"}


# Tests: end-to-end proxied request flow

def test_proxy_forwards_and_acquires_release_with_mb_zero(proxy_app):
    """Test: a successful request acquires with mb=0, forwards body verbatim, releases after."""
    from fastapi.testclient import TestClient

    acquire_resp = _mock_response(200, {"granted": True, "lease_id": "lease-abc"})
    release_resp = _mock_response(200, {"released": True})
    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"choices":[]}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = [acquire_resp, release_resp]
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 200
    acquire_call = mock_client.post.call_args_list[0]
    assert acquire_call.kwargs["json"]["mb"] == 0
    assert acquire_call.kwargs["json"]["caller"].startswith("oaie-proxy:ollama:")
    release_call = mock_client.post.call_args_list[1]
    assert release_call.kwargs["json"] == {"lease_id": "lease-abc"}


def test_proxy_returns_429_with_retry_after_on_broker_denial(proxy_app):
    """Test: broker 504 (no room) becomes a 429 with Retry-After to the OAIE caller."""
    from fastapi.testclient import TestClient

    denial_resp = _mock_response(504, {})
    mock_client = _make_mock_broker_client()
    mock_client.post.return_value = denial_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "3.0"
    # Forwarding must never have been attempted once admission was denied.
    mock_client.send.assert_not_called()


def test_proxy_maps_broker_503_to_429_with_broker_retry_after(proxy_app):
    """Test: broker 503 (game running) is a denial: 429 with the broker's Retry-After, never fail open."""
    from fastapi.testclient import TestClient

    assert proxy_app.BROKER_FAIL_OPEN is True
    busy_resp = _mock_response(503, {"detail": "unmanaged_gpu_load"}, headers={"Retry-After": "120"})
    busy_resp.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
        "503", request=Mock(), response=busy_resp,
    ))
    mock_client = _make_mock_broker_client()
    mock_client.post.return_value = busy_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "120"
    assert resp.json()["detail"] == "GPU reserved for unmanaged load, retry later"
    mock_client.send.assert_not_called()
    assert mock_client.post.call_count == 1


def test_proxy_fails_open_on_broker_connection_error(proxy_app):
    """Test: broker unreachable + BROKER_FAIL_OPEN=true forwards unmetered (no lease)."""
    from fastapi.testclient import TestClient

    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b"{}"
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = httpx.ConnectError("connection refused")
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 200
    mock_client.send.assert_called_once()
    # Fail-open means no lease was granted, so only the failed acquire
    # attempts hit /acquire. Never /release.
    for call in mock_client.post.call_args_list:
        assert "/release" not in call.args[0]


def test_proxy_404_when_no_backend_resolves(proxy_app):
    """Test: a request whose model matches nothing and no default backend exists gets 404."""
    from fastapi.testclient import TestClient

    mock_client = _make_mock_broker_client()

    with TestClient(proxy_app.app) as client:
        proxy_app._registry = [
            {"name": "whisper", "base_url": "http://whisper:9000", "models": ["whisper-1"], "default": False},
        ]
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "unregistered-model"})

    assert resp.status_code == 404
    mock_client.post.assert_not_called()
    mock_client.send.assert_not_called()


def test_proxy_forwards_native_ollama_endpoint_ungated(proxy_app):
    """
    Test: a native Ollama endpoint with no "model" in the body (for example
    GET /api/ps, which Headroom calls directly) is forwarded to /api/ps on
    the backend without touching the broker. Regression for the proxy
    silently 404ing non-/v1/* paths and breaking Headroom and GpuBroker's
    own passive polling.
    """
    from fastapi.testclient import TestClient

    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"models":[]}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    mock_client = _make_mock_broker_client()
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.get("/api/ps")

    assert resp.status_code == 200
    mock_client.post.assert_not_called()
    upstream_call = mock_client.build_request.call_args
    assert upstream_call.args[1] == "http://ollama:11434/api/ps"


def test_health_endpoint(proxy_app):
    """Test: health endpoint always returns ok."""
    from fastapi.testclient import TestClient

    with TestClient(proxy_app.app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_release_still_happens_when_aclose_raises(proxy_app):
    """Test: broker release still runs if upstream resp.aclose() raises."""
    from fastapi.testclient import TestClient

    acquire_resp = _mock_response(200, {"granted": True, "lease_id": "lease-abc"})
    release_resp = _mock_response(200, {"released": True})
    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"choices":[]}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock(side_effect=RuntimeError("aclose failed"))

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = [acquire_resp, release_resp]
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 200
    release_calls = [c for c in mock_client.post.call_args_list if c.args[0].endswith("/release")]
    assert len(release_calls) == 1
    assert release_calls[0].kwargs["json"] == {"lease_id": "lease-abc"}


def test_stream_heartbeats_lease_until_complete(proxy_app, monkeypatch):
    """Test: a long stream heartbeats the lease, then stops after it ends."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(proxy_app, "PROXY_HEARTBEAT_INTERVAL_S", 0.05)

    acquire_resp = _mock_response(200, {"granted": True, "lease_id": "lease-abc"})
    heartbeat_resp = _mock_response(200, {"ok": True, "preempt": True})
    release_resp = _mock_response(200, {"released": True})
    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"choices":['
        await asyncio.sleep(0.20)
        yield b']}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    heartbeats = []

    def post_side_effect(url, **kwargs):
        if url.endswith("/acquire"):
            return acquire_resp
        if url.endswith("/heartbeat"):
            heartbeats.append(kwargs.get("json"))
            return heartbeat_resp
        if url.endswith("/release"):
            return release_resp
        raise AssertionError(f"unexpected post url: {url}")

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = post_side_effect
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 200
    hb_count = len(heartbeats)
    assert hb_count >= 2
    assert all(h == {"lease_id": "lease-abc"} for h in heartbeats)
    time.sleep(0.15)
    assert len(heartbeats) == hb_count


def test_client_disconnect_mid_stream_releases_lease(proxy_app):
    """Test: a client disconnect mid-stream still releases the broker lease once."""
    body = json.dumps({"model": "qwen2.5:3b", "prompt": "hi"}).encode()
    acquire_resp = _mock_response(200, {"granted": True, "lease_id": "lease-disconnect"})
    heartbeat_resp = _mock_response(200, {"ok": True, "preempt": False})
    release_resp = _mock_response(200, {"released": True})
    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    chunk_count = 40

    async def fake_aiter_bytes():
        for i in range(chunk_count):
            yield f"chunk-{i}\n".encode()
            await asyncio.sleep(0.05)
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    releases = []

    async def post_side_effect(url, **kwargs):
        # Checkpoint so a cancelled anyio scope cannot count a /release
        # that never actually completed.
        await asyncio.sleep(0)
        if url.endswith("/acquire"):
            return acquire_resp
        if url.endswith("/heartbeat"):
            return heartbeat_resp
        if url.endswith("/release"):
            releases.append(kwargs.get("json"))
            return release_resp
        raise AssertionError(f"unexpected post url: {url}")

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = post_side_effect
    mock_client.send.return_value = forward_resp
    proxy_app._broker_client = mock_client

    first_chunk_sent = asyncio.Event()
    messages = []
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/generate",
        "raw_path": b"/api/generate",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 123),
        "server": ("testserver", 80),
    }

    async def _run():
        try:
            await asyncio.wait_for(proxy_app.app(scope, receive, send), timeout=2.0)
        except asyncio.CancelledError:
            logger.warning("ASGI app cancelled after client disconnect")
        except TimeoutError:
            logger.exception("ASGI app did not return after client disconnect")
            raise
        await asyncio.sleep(0.2)
        live_hb = [
            t for t in asyncio.all_tasks()
            if (not t.done())
            and getattr(t.get_coro(), "__name__", "") == "_heartbeat_loop"
        ]
        body_messages = [m for m in messages if m["type"] == "http.response.body"]
        assert first_chunk_sent.is_set()
        assert body_messages, "expected at least one body chunk before disconnect"
        assert len(body_messages) < chunk_count
        assert releases == [{"lease_id": "lease-disconnect"}]
        assert live_hb == []

    asyncio.run(_run())


def test_content_length_over_limit_returns_413_without_broker(proxy_app, monkeypatch):
    """Test: Content-Length above the body cap is 413 and never calls the broker."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(proxy_app, "MAX_REQUEST_BODY_BYTES", 64)
    mock_client = _make_mock_broker_client()

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5:3b", "pad": "x" * 200},
        )

    assert resp.status_code == 413
    mock_client.post.assert_not_called()
    mock_client.send.assert_not_called()


def test_chunked_body_over_limit_returns_413_without_broker(proxy_app, monkeypatch):
    """Test: a chunked body over the cap is 413 and never calls the broker."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(proxy_app, "MAX_REQUEST_BODY_BYTES", 32)
    mock_client = _make_mock_broker_client()

    def body_iter():
        yield b'{"model":"qwen2.5:3b","pad":"'
        yield b"x" * 100
        yield b'"}'

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post(
            "/v1/chat/completions",
            content=body_iter(),
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 413
    mock_client.post.assert_not_called()
    mock_client.send.assert_not_called()


def test_body_under_limit_still_forwards(proxy_app, monkeypatch):
    """Test: a body under the cap still acquires, forwards, and releases."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(proxy_app, "MAX_REQUEST_BODY_BYTES", 1024)
    acquire_resp = _mock_response(200, {"granted": True, "lease_id": "lease-abc"})
    release_resp = _mock_response(200, {"released": True})
    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"choices":[]}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    mock_client = _make_mock_broker_client()
    mock_client.post.side_effect = [acquire_resp, release_resp]
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 200
    assert mock_client.send.call_count == 1
    release_calls = [c for c in mock_client.post.call_args_list if c.args[0].endswith("/release")]
    assert len(release_calls) == 1


def test_proxy_maps_too_many_leases_503_to_429(proxy_app):
    """Test: broker 503 too_many_leases is 429 with the lease-limit message and Retry-After."""
    from fastapi.testclient import TestClient

    busy_resp = _mock_response(
        503, {"detail": "too_many_leases"}, headers={"Retry-After": "5"},
    )
    mock_client = _make_mock_broker_client()
    mock_client.post.return_value = busy_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/v1/chat/completions", json={"model": "qwen2.5:3b", "messages": []})

    assert resp.status_code == 429
    assert resp.json()["detail"] == "broker lease limit reached, retry later"
    assert resp.headers["retry-after"] == "5"
    mock_client.send.assert_not_called()
    assert mock_client.post.call_count == 1


def test_api_show_forwards_without_broker(proxy_app):
    """Test: POST /api/show with a model is routed by model and skips acquire/release."""
    from fastapi.testclient import TestClient

    forward_resp = Mock(spec=httpx.Response)
    forward_resp.status_code = 200
    forward_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"modelfile":""}'
    forward_resp.aiter_bytes = fake_aiter_bytes
    forward_resp.aclose = AsyncMock()

    mock_client = _make_mock_broker_client()
    mock_client.send.return_value = forward_resp

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        resp = client.post("/api/show", json={"model": "qwen2.5:3b"})

    assert resp.status_code == 200
    mock_client.post.assert_not_called()
    mock_client.send.assert_called_once()
    upstream_call = mock_client.build_request.call_args
    assert upstream_call.args[1] == "http://ollama:11434/api/show"


def test_docs_and_openapi_are_disabled(proxy_app):
    """Test: FastAPI /docs and /openapi.json are 404, not proxied."""
    from fastapi.testclient import TestClient

    mock_client = _make_mock_broker_client()

    with TestClient(proxy_app.app) as client:
        proxy_app._broker_client = mock_client
        docs = client.get("/docs")
        spec = client.get("/openapi.json")

    assert docs.status_code == 404
    assert spec.status_code == 404
    mock_client.post.assert_not_called()
    mock_client.send.assert_not_called()
