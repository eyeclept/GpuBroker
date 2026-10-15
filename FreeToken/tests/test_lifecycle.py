"""
Author: FreeTokenBuild
Date:   2026
Email:  eyeclept@pm.me

Description:
    Lifecycle tests for serve_wrapper's lazy-load front proxy: load on
    first request, idle unload, preemption, broker denial, and
    single-flight loading. The broker functions, `subprocess.Popen`, and
    the `ft serve` HTTP API (httpx.MockTransport) are faked. The clock is
    injected, and the idle check and heartbeat are driven directly.
"""
# Imports
import asyncio
import json
import logging

import httpx
import pytest

import serve_wrapper
from broker_client import BrokerAcquireError, BrokerNoRoom

# Globals
logger = logging.getLogger(__name__)

MB = serve_wrapper.ACQUIRE_MB
IDLE = 60.0


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeChild:
    """Stands in for the `ft serve` Popen handle."""

    def __init__(self, harness, pid):
        self._h = harness
        self.pid = pid
        self.returncode = 1 if harness.die_on_start else None

    def poll(self):
        return self.returncode

    def terminate(self):
        self._h.events.append(("terminate", self.pid))
        self.returncode = -15
        # A stopped child ends any stream it was serving.
        self._h.loop.call_soon_threadsafe(self._h.stream_gate.set)

    def kill(self):
        self._h.events.append(("kill", self.pid))
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class Harness:
    """Fake broker, fake Popen, and fake ft serve HTTP API."""

    def __init__(self, monkeypatch):
        self.events = []
        self.acquires = []
        self.children = []
        self.loading_polls = 2
        self.health_calls = 0
        self.die_on_start = False
        self.acquire_error = None
        self.heartbeat_result = (True, False)
        self.reacquire_error = None
        self.block_stream = False
        self.stream_gate = asyncio.Event()
        self.first_chunk_sent = asyncio.Event()
        self.child_requests = []
        self.loop = asyncio.get_running_loop()
        self._next_lease = 0
        self._lease = None
        self.clock = FakeClock()
        self.monkeypatch = monkeypatch
        self.upstream_closes = 0
        self.aclose_error = None

        monkeypatch.setattr(serve_wrapper, "IDLE_SECONDS", IDLE)
        monkeypatch.setattr(serve_wrapper, "LOAD_POLL_SECONDS", 0.001)
        monkeypatch.setattr(serve_wrapper, "HEARTBEAT_INTERVAL_SECONDS", 3600)
        monkeypatch.setattr(serve_wrapper, "IDLE_CHECK_SECONDS", 3600)
        monkeypatch.setattr(serve_wrapper, "_broker_acquire", self.acquire)
        monkeypatch.setattr(serve_wrapper, "_broker_release", self.release)
        monkeypatch.setattr(serve_wrapper, "_broker_heartbeat", self.heartbeat)
        monkeypatch.setattr(serve_wrapper.subprocess, "Popen", self.popen)
        monkeypatch.setattr(serve_wrapper, "_evict_ollama_models", self.evict_ollama)

        self.sup = serve_wrapper.ModelSupervisor(
            clock=self.clock, transport=httpx.MockTransport(self.child_api),
        )
        self.app = serve_wrapper.create_app(self.sup)

    # broker fakes (run in worker threads via asyncio.to_thread)
    def acquire(self, mb, pid=None):
        self.acquires.append((mb, pid))
        if pid is None:
            if self.acquire_error is not None:
                raise self.acquire_error
            self._next_lease += 1
            self._lease = f"lease-{self._next_lease}"
            self.events.append(("acquire", self._lease))
            return self._lease
        if self.reacquire_error is not None:
            raise self.reacquire_error
        # Same-caller dedup: the broker returns the held lease.
        return self._lease

    def release(self, lease_id):
        self.events.append(("release", lease_id))

    def heartbeat(self, lease_id):
        return self.heartbeat_result

    async def evict_ollama(self):
        self.events.append(("evict_ollama", None))

    def popen(self, cmd):
        child = FakeChild(self, 4000 + len(self.children))
        self.children.append(child)
        self.events.append(("popen", child.pid))
        return child

    # ft serve fake
    async def child_api(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            if self.health_calls <= self.loading_polls:
                return httpx.Response(200, json={"status": "loading", "phase": "weights"})
            return httpx.Response(200, json={"status": "ok", "maintenance": "serving"})
        self.child_requests.append(request)

        async def chunks():
            yield b'data: {"n": 1}\n\n'
            self.first_chunk_sent.set()
            if self.block_stream:
                await self.stream_gate.wait()
                if self.children[-1].returncode is not None:
                    raise httpx.RemoteProtocolError("peer closed connection")
            yield b'data: {"n": 2}\n\n'
            yield b"data: [DONE]\n\n"

        resp = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=chunks(),
        )
        orig_aclose = resp.aclose

        async def tracked_aclose():
            # httpx aiter_raw() acloses when the iterator ends. Raise only
            # on the later close-hook call so this matches the production bug.
            if self.aclose_error is not None and self.upstream_closes:
                raise self.aclose_error
            await orig_aclose()
            self.upstream_closes += 1

        resp.aclose = tracked_aclose
        return resp

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://wrapper",
        )

    def start_stream(self, path="/v1/chat/completions", body=b'{"stream": true}'):
        """Run one request against the ASGI app; returns (task, sent messages queue)."""
        sent: asyncio.Queue = asyncio.Queue()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path,
            "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"host", b"wrapper"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 50000), "server": ("wrapper", 1919),
        }
        task = asyncio.create_task(self.app(scope, receive, sent.put))
        return task, sent


def scenario(fn):
    """Run an async test body with a started supervisor and a Harness."""

    def wrapper(monkeypatch):
        async def main():
            h = Harness(monkeypatch)
            await h.sup.start()
            try:
                await fn(h)
            finally:
                await h.sup.shutdown()

        asyncio.run(main())

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


async def body_chunks(sent: asyncio.Queue) -> tuple[int, list[bytes]]:
    """Drain one ASGI response; returns (status, non-empty body chunks)."""
    start = await asyncio.wait_for(sent.get(), 5)
    assert start["type"] == "http.response.start"
    chunks = []
    while True:
        msg = await asyncio.wait_for(sent.get(), 5)
        if msg.get("body"):
            chunks.append(msg["body"])
        if not msg.get("more_body", False):
            return start["status"], chunks


async def load_and_finish(h: Harness) -> None:
    async with h.client() as c:
        resp = await c.post("/v1/chat/completions", json={"model": "m"})
    assert resp.status_code == 200


@scenario
async def test_startup_holds_no_lease_and_health_does_not_load(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (1) The wrapper starts with no lease and no child. /health answers
        200 with model_state and never triggers a load.
    """
    async with h.client() as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_state": "unloaded"}
    assert h.acquires == []
    assert h.children == []
    assert h.health_calls == 0


@scenario
async def test_first_request_loads_and_streams(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (2) First request: acquire (no pid), Popen, pid registration
        through the deduped acquire, wait until ft serve reports
        status=ok + maintenance=serving, then forward and stream back
        chunk by chunk.
    """
    task, sent = h.start_stream(path="/v1/chat/completions")
    status, chunks = await body_chunks(sent)
    await task

    assert status == 200
    assert len(chunks) == 3
    assert b"".join(chunks).endswith(b"data: [DONE]\n\n")
    assert h.acquires == [(MB, None), (MB, 4000)]
    # Ollama yields to FreeToken before the acquire and again before Popen.
    assert [e[0] for e in h.events] == ["evict_ollama", "acquire", "evict_ollama", "popen"]
    assert h.health_calls == h.loading_polls + 1
    forwarded = h.child_requests[0]
    assert forwarded.method == "POST"
    assert forwarded.url.path == "/v1/chat/completions"
    assert json.loads(forwarded.content) == {"stream": True}
    assert h.sup.state == "ready"
    assert h.sup.in_flight == 0


@scenario
async def test_idle_unload_stops_child_before_release(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (3) Idle unload waits IDLE_SECONDS after the last response
        finished, then stops the child and only then releases the lease.
    """
    await load_and_finish(h)

    h.clock.now += IDLE - 1
    await h.sup.idle_check()
    assert h.children[0].returncode is None
    assert ("release", "lease-1") not in h.events

    h.clock.now += 1
    await h.sup.idle_check()
    assert h.events[-2:] == [("terminate", 4000), ("release", "lease-1")]
    assert h.sup.state == "unloaded"
    assert h.sup.lease_id is None


@scenario
async def test_in_flight_stream_blocks_idle_unload(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (4) A stream still open past IDLE_SECONDS keeps the model loaded.
        The idle timer starts when that stream finishes.
    """
    h.block_stream = True
    task, sent = h.start_stream()
    await asyncio.wait_for(h.first_chunk_sent.wait(), 5)
    assert h.sup.in_flight == 1

    h.clock.now += IDLE * 10
    await h.sup.idle_check()
    assert h.children[0].returncode is None
    assert h.sup.state == "ready"

    h.stream_gate.set()
    status, chunks = await body_chunks(sent)
    await task
    assert status == 200 and len(chunks) == 3
    assert h.sup.in_flight == 0

    h.clock.now += IDLE - 1
    await h.sup.idle_check()
    assert h.sup.state == "ready"
    h.clock.now += 1
    await h.sup.idle_check()
    assert h.sup.state == "unloaded"
    assert h.events[-2:] == [("terminate", 4000), ("release", "lease-1")]


@scenario
async def test_request_after_unload_reloads(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:    (5) A request after an idle unload acquires a new lease and starts a new child.
    """
    await load_and_finish(h)
    h.clock.now += IDLE
    await h.sup.idle_check()
    assert h.sup.state == "unloaded"

    await load_and_finish(h)
    assert len(h.children) == 2
    assert h.sup.lease_id == "lease-2"
    assert h.sup.state == "ready"
    assert h.events[-3:] == [("acquire", "lease-2"), ("evict_ollama", None), ("popen", 4001)]


def test_game_denial_returns_503_at_once_with_broker_retry_after(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        (6) A game denial (broker 503) is not retried: the client gets a
        503 carrying the broker's Retry-After, and ft serve never starts.
    """

    async def main():
        h = Harness(monkeypatch)
        h.acquire_error = BrokerAcquireError("yielding to unmanaged GPU load", 1800.0)
        await h.sup.start()
        try:
            async with h.client() as c:
                resp = await c.post("/v1/chat/completions", json={"model": "m"})
            assert resp.status_code == 503
            assert resp.headers["retry-after"] == "1800"
            assert "unmanaged GPU load" in resp.json()["error"]["message"]
            assert h.acquires == [(MB, None)]
            assert h.children == []
            assert h.sup.state == "unloaded"
            assert h.sup.in_flight == 0
        finally:
            await h.sup.shutdown()

    asyncio.run(main())


def test_no_room_backs_off_and_loads_when_room_frees(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        A broker 504 is retried while the client waits; once the holder
        yields, the lease is granted and the request is served.
    """

    async def main():
        h = Harness(monkeypatch)
        monkeypatch.setattr(serve_wrapper, "ACQUIRE_BACKOFF_MIN_SECONDS", 0.001)
        denials = [BrokerNoRoom("GPU memory full", None)] * 2

        def acquire(mb, pid=None):
            if pid is None and denials:
                h.acquires.append((mb, pid))
                raise denials.pop()
            return h.acquire(mb, pid)

        monkeypatch.setattr(serve_wrapper, "_broker_acquire", acquire)
        await h.sup.start()
        try:
            async with h.client() as c:
                resp = await c.post("/v1/chat/completions", json={"model": "m"})
            assert resp.status_code == 200
            assert h.acquires == [(MB, None)] * 3 + [(MB, 4000)]
            assert len(h.children) == 1
        finally:
            await h.sup.shutdown()

    asyncio.run(main())


def test_no_room_gives_up_after_wait_budget(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:
        A holder that never yields ends the wait after
        ACQUIRE_WAIT_SECONDS with a 503 carrying the broker's Retry-After.
    """

    async def main():
        h = Harness(monkeypatch)
        monkeypatch.setattr(serve_wrapper, "ACQUIRE_BACKOFF_MIN_SECONDS", 0.001)
        monkeypatch.setattr(serve_wrapper, "ACQUIRE_WAIT_SECONDS", 0.0035)
        h.acquire_error = BrokerNoRoom("GPU memory full", None)
        await h.sup.start()
        try:
            async with h.client() as c:
                resp = await c.post("/v1/chat/completions", json={"model": "m"})
            assert resp.status_code == 503
            # Sleeps 0.001 and 0.002; the next 0.004 would pass the budget.
            assert h.acquires == [(MB, None)] * 3
            assert h.children == []
            assert h.sup.state == "unloaded"
        finally:
            await h.sup.shutdown()

    asyncio.run(main())


@scenario
async def test_preempt_heartbeat_unloads_mid_stream(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (7) preempt=true stops the child at once, even with a stream in
        flight, then releases the lease. The open stream ends.
    """
    h.block_stream = True
    task, sent = h.start_stream()
    await asyncio.wait_for(h.first_chunk_sent.wait(), 5)

    h.heartbeat_result = (True, True)
    await h.sup.heartbeat_once()

    assert h.events[-2:] == [("terminate", 4000), ("release", "lease-1")]
    assert h.sup.state == "unloaded"
    assert h.sup.lease_id is None
    status, chunks = await body_chunks(sent)
    await task
    assert status == 200
    assert chunks == [b'data: {"n": 1}\n\n']
    assert h.sup.in_flight == 0


@scenario
async def test_lost_lease_without_reacquire_stops_child(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        ok=false and a refused re-acquire stop the child. The lost lease
        is not released (the broker already dropped it).
    """
    await load_and_finish(h)
    h.heartbeat_result = (False, False)
    h.reacquire_error = BrokerAcquireError("yielding to unmanaged GPU load")

    await h.sup.heartbeat_once()

    assert h.events[-1] == ("terminate", 4000)
    assert ("release", "lease-1") not in h.events
    assert h.sup.state == "unloaded"


@scenario
async def test_concurrent_requests_share_one_load(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        (8) Two requests that arrive during a load wait for the same load:
        one new acquire (plus its pid registration) and one Popen.
    """
    h.loading_polls = 20
    async with h.client() as c:
        first, second = await asyncio.gather(
            c.post("/v1/chat/completions", json={"n": 1}),
            c.post("/v1/chat/completions", json={"n": 2}),
        )
    assert first.status_code == 200 and second.status_code == 200
    assert h.acquires == [(MB, None), (MB, 4000)]
    assert len(h.children) == 1
    assert len(h.child_requests) == 2


@scenario
async def test_child_exit_during_load_releases_and_503(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        ft serve dying before it is ready ends the load: lease released,
        state unloaded, client gets 503.
    """
    h.die_on_start = True
    async with h.client() as c:
        resp = await c.post("/v1/chat/completions", json={"model": "m"})
    assert resp.status_code == 503
    assert "retry-after" in resp.headers
    assert h.events[-1] == ("release", "lease-1")
    assert h.sup.state == "unloaded"


@scenario
async def test_shutdown_stops_child_then_releases(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:    Lifespan shutdown stops the child before releasing the lease.
    """
    await load_and_finish(h)
    await h.sup.shutdown()
    assert h.events[-2:] == [("terminate", 4000), ("release", "lease-1")]
    assert h.sup.state == "unloaded"

@scenario
async def test_content_length_over_limit_is_413_without_load(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        Content-Length above MAX_REQUEST_BODY_BYTES is 413 with an
        OpenAI-style error body. ensure_loaded never runs, so there is
        no acquire and no child. The in-flight slot is released.
    """
    h.monkeypatch.setattr(serve_wrapper, "MAX_REQUEST_BODY_BYTES", 64)
    async with h.client() as c:
        resp = await c.post(
            "/v1/chat/completions",
            json={"model": "m", "pad": "x" * 200},
        )
    assert resp.status_code == 413
    assert resp.json() == {
        "error": {"message": "request body too large", "type": "payload_too_large"},
    }
    assert h.acquires == []
    assert h.children == []
    assert h.sup.in_flight == 0


@scenario
async def test_chunked_body_over_limit_is_413_without_load(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        A chunked body with no Content-Length that exceeds the cap is
        413. No acquire, no child, in_flight returns to 0.
    """
    h.monkeypatch.setattr(serve_wrapper, "MAX_REQUEST_BODY_BYTES", 32)
    pieces = [b'{"pad":"', b"x" * 40, b'"}']
    idx = 0
    sent: asyncio.Queue = asyncio.Queue()

    async def receive():
        nonlocal idx
        if idx < len(pieces):
            body = pieces[idx]
            idx += 1
            return {"type": "http.request", "body": body, "more_body": idx < len(pieces)}
        await asyncio.Event().wait()

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
        "headers": [(b"host", b"wrapper"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000), "server": ("wrapper", 1919),
    }
    task = asyncio.create_task(h.app(scope, receive, sent.put))
    status, chunks = await body_chunks(sent)
    await task
    assert status == 413
    assert json.loads(b"".join(chunks)) == {
        "error": {"message": "request body too large", "type": "payload_too_large"},
    }
    assert h.acquires == []
    assert h.children == []
    assert h.sup.in_flight == 0


@scenario
async def test_body_under_limit_still_loads_and_streams(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        A body under the cap still acquires, loads the child, and streams.
    """
    h.monkeypatch.setattr(serve_wrapper, "MAX_REQUEST_BODY_BYTES", 1024)
    async with h.client() as c:
        resp = await c.post("/v1/chat/completions", json={"model": "m"})
    assert resp.status_code == 200
    assert h.acquires == [(MB, None), (MB, 4000)]
    assert len(h.children) == 1
    assert h.sup.state == "ready"
    assert h.sup.in_flight == 0


@scenario
async def test_aclose_error_still_decrements_in_flight(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        If upstream aclose raises, finish still runs. in_flight returns
        to 0 and idle unload can stop the child and release the lease.
    """
    h.aclose_error = RuntimeError("aclose failed")
    await load_and_finish(h)
    assert h.sup.in_flight == 0
    h.clock.now += IDLE
    await h.sup.idle_check()
    assert h.sup.state == "unloaded"
    assert h.events[-2:] == [("terminate", 4000), ("release", "lease-1")]


@scenario
async def test_client_disconnect_mid_stream_closes_upstream(h):
    """
    Input:      h -- Harness
    Output:     None
    Details:
        After the first body chunk, receive returns http.disconnect.
        The ASGI call returns with in_flight 0 and the upstream closed.
    """
    h.block_stream = True
    body = b'{"stream": true}'
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await h.first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    sent: asyncio.Queue = asyncio.Queue()

    async def send(message):
        await sent.put(message)
        if message["type"] == "http.response.body" and message.get("body"):
            h.first_chunk_sent.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
        "headers": [(b"host", b"wrapper"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000), "server": ("wrapper", 1919),
    }
    try:
        await asyncio.wait_for(h.app(scope, receive, send), 5)
    except asyncio.CancelledError:
        logger.warning("ASGI app cancelled after client disconnect")
    assert h.sup.in_flight == 0
    assert h.upstream_closes == 1



def _patch_ollama(monkeypatch, handler):
    """Route serve_wrapper's Ollama client through an httpx.MockTransport."""
    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(serve_wrapper.httpx, "AsyncClient", client)


def test_evict_ollama_unloads_every_resident_model(monkeypatch):
    """
    Input:      monkeypatch
    Output:     None
    Details:    Each model in /api/ps gets /api/generate with keep_alive 0.
    """
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "a:1"}, {"name": "b:2"}]})
        return httpx.Response(200, json={"done": True})

    _patch_ollama(monkeypatch, handler)
    asyncio.run(serve_wrapper._evict_ollama_models())
    assert calls[0][:2] == ("GET", "/api/ps")
    assert [json.loads(c[2]) for c in calls[1:]] == [
        {"model": "a:1", "keep_alive": 0, "stream": False},
        {"model": "b:2", "keep_alive": 0, "stream": False},
    ]


def test_evict_ollama_failure_is_logged_not_raised(monkeypatch, caplog):
    """
    Input:      monkeypatch, caplog
    Output:     None
    Details:    An unreachable Ollama must not block a FreeToken load.
    """
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_ollama(monkeypatch, handler)
    asyncio.run(serve_wrapper._evict_ollama_models())
    assert "could not unload Ollama models" in caplog.text
