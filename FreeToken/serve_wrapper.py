"""
Author: FreeTokenBuild
Date:   2026
Email:  eyeclept@pm.me

Description:
    GpuBroker-integrated front proxy for `ft serve`. FreeToken itself is an
    unmodified third-party CLI (github.com/FlashML-org/FreeToken) with no
    knowledge of GpuBroker. This process wraps it rather than patching its
    source.

    The wrapper serves HTTP on FREETOKEN_HOST:FREETOKEN_PORT and owns the
    model lifecycle, following the Whisper/Flux2Klein9B lazy-load shape:
      - Startup holds no lease and starts no child.
      - The first request acquires a lease, starts `ft serve` on
        127.0.0.1:FREETOKEN_INTERNAL_PORT, and waits until the engine
        reports ready. Concurrent requests wait for the same load.
      - Requests are stream-forwarded to the child.
      - After FREETOKEN_IDLE_SECONDS with no request in flight, the child
        is stopped and only then is the lease released.
      - A preempt=true heartbeat (game or fairness) stops the child at
        once, even mid-request, then releases the lease.
      - Shutdown stops the child, then releases the lease.
    GET /health answers from the wrapper and never loads the model.

    BROKER_FAIL_OPEN=false (see README.md): a broker denial or a down
    broker returns 503 to the client and never starts `ft serve`. The
    card is shared with Ollama, Flux2Klein9B, Talkie, Needle, and games,
    so an uncoordinated load risks an OOM here or in a sibling.
"""
# Imports
import asyncio
import contextlib
import logging
import os
import subprocess
import time
from typing import Callable

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from broker_client import (
    BROKER_FAIL_OPEN,
    CALLER_NAME,
    BrokerAcquireError,
    BrokerNoRoom,
    _broker_acquire,
    _broker_heartbeat,
    _broker_release,
    close_client,
)

# Globals
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s wrapper: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = os.environ.get("FREETOKEN_MODEL", "primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8")
SERVED_MODEL_NAME = os.environ.get("FREETOKEN_SERVED_MODEL_NAME", "ornith-1.5-35b-a3b")
HOST = os.environ.get("FREETOKEN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FREETOKEN_PORT", "1919"))
# `ft serve` binds here, loopback only. Clients reach it through the wrapper.
INTERNAL_PORT = int(os.environ.get("FREETOKEN_INTERNAL_PORT", "1920"))
MOE_BACKEND = os.environ.get("FREETOKEN_MOE_BACKEND", "offload")
EXTRA_ARGS = os.environ.get("FREETOKEN_EXTRA_ARGS", "")

# See ../examples/sizing_a_request.md: this must be a measured nvidia-smi
# peak (weights + MoE cache + KV + working memory), never the checkpoint's
# on-disk size. Documented and justified in README.md.
ACQUIRE_MB = int(os.environ.get("FREETOKEN_ACQUIRE_MB", "6656"))

# Broker sweeper reclaims a lease after GPU_BROKER_LEASE_TTL_SECONDS (180s
# default) without a heartbeat. 5s is short on purpose: a preempt flag set
# because a game started is only seen on the next beat.
HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("FREETOKEN_HEARTBEAT_INTERVAL_SECONDS", "5"))

# Seconds with no request in flight before the model is unloaded.
IDLE_SECONDS = float(os.environ.get("FREETOKEN_IDLE_SECONDS", "60"))

# Upper bound on one model load. First boot downloads a ~20 GB checkpoint.
LOAD_TIMEOUT_SECONDS = float(os.environ.get("FREETOKEN_LOAD_TIMEOUT_SECONDS", "1800"))

# How often a load polls the child's /health, and how often the idle
# check runs. Idle unload therefore fires up to IDLE_CHECK_SECONDS late.
LOAD_POLL_SECONDS = 1.0
IDLE_CHECK_SECONDS = 5.0

# A broker 504 (no room yet) is retried with exponential backoff while the
# client request waits: 2 s doubling to 60 s, never shorter than the
# broker's Retry-After. Each attempt is itself a 30 s broker long poll. The
# broker keeps this caller's fairness age across retries, so a long wait
# ends by preempting an idle preemptible holder (Laya, Flux, Whisper).
# Bounded so a client is not held forever; 300 s stays under JobCollector's
# 420 s request timeout.
ACQUIRE_WAIT_SECONDS = float(os.environ.get("FREETOKEN_ACQUIRE_WAIT_SECONDS", "300"))
ACQUIRE_BACKOFF_MIN_SECONDS = 2.0
ACQUIRE_BACKOFF_MAX_SECONDS = 60.0

# Retry-After on a 503. Matches the broker's UNMANAGED_CLEAR_SECONDS
# default: a game must be gone this long before the broker admits again.
RETRY_AFTER_SECONDS = 120

# FreeToken has priority over Ollama. ft serve sizes its cache from free
# VRAM about 6 s after start, so a resident Ollama model can fail the
# load. Reached through GpuBroker-Proxy; /api/ps has no model field and
# passes ungated to Ollama.
OLLAMA_URL = os.environ.get("FREETOKEN_OLLAMA_URL", "http://host.containers.internal:11434")

# uvicorn waits this long for open streams on SIGTERM, then cancels them.
# Leaves room inside compose's 90s stop_grace_period for the 30s child
# stop (+10s after SIGKILL) and the lease release.
GRACEFUL_SHUTDOWN_SECONDS = 30

# Vision payloads carry base64 images.
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024

# RFC 9110 hop-by-hop headers, plus headers httpx/uvicorn recompute.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
})
_REQUEST_DROP = _HOP_BY_HOP | {"host", "content-length"}


class LoadFailed(RuntimeError):
    """Model could not be loaded. The client gets a 503 with Retry-After;
    retry_after is the broker's hint in seconds, if it sent one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PayloadTooLarge(Exception):
    """Request body exceeds MAX_REQUEST_BODY_BYTES."""


# Functions
async def _evict_ollama_models() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Unloads every model Ollama reports in /api/ps with keep_alive=0.
        A failure is logged and ignored: broker admission still protects
        the card, and the load then fails cleanly if memory is short.
    """
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=10.0) as client:
            resp = await client.get("/api/ps")
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            for name in names:
                unload = await client.post(
                    "/api/generate", json={"model": name, "keep_alive": 0, "stream": False},
                )
                unload.raise_for_status()
                logger.info("unloaded Ollama model %s before loading ft serve", name)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("could not unload Ollama models before load: %s", exc)


def _build_command() -> list[str]:
    """
    Input:      None
    Output:     argv for the `ft serve` child
    Details:
        Child binds loopback on INTERNAL_PORT. The wrapper owns the
        public port.
    """
    cmd = [
        "ft", "serve",
        "--model", MODEL,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(INTERNAL_PORT),
        "--moe-backend", MOE_BACKEND,
    ]
    if EXTRA_ARGS:
        cmd.extend(EXTRA_ARGS.split())
    return cmd


def _stop_child(child: subprocess.Popen) -> None:
    """
    Input:      child -- the `ft serve` process
    Output:     None
    Details:
        SIGTERM the child, escalate to SIGKILL after 30s. No-op if it
        already exited. Blocking; call through asyncio.to_thread. Does
        not release the broker lease; caller does that after VRAM is gone.
    """
    if child.poll() is None:
        logger.info("sending SIGTERM to ft serve (pid %d)", child.pid)
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("ft serve did not exit within 30s, killing")
            child.kill()
            child.wait(timeout=10)


def _ready_from_health(doc: dict) -> bool:
    """
    Input:      doc -- JSON body of the child's GET /health
    Output:     True when the engine admits generation requests
    Details:
        ft serve's control_api.build_health always answers 200. The body
        is {"status": "loading"} until the backend acks ready, then
        {"status": "ok", "maintenance": <maintenance_state>}. The OpenAI
        gate (_maintenance_gate) admits work only when maintenance_state
        is "serving", so both fields are required.
        Raises LoadFailed on {"status": "error"} (dead backend).
    """
    status = doc.get("status")
    if status == "error":
        raise LoadFailed(f"ft serve reported a fatal error: {doc.get('message')}")
    return status == "ok" and doc.get("maintenance") == "serving"


def _unavailable(message: str, retry_after: float | None = None) -> JSONResponse:
    """
    Input:      message -- reason shown to the client
                retry_after -- broker's hint in seconds, if it sent one
    Output:     503 response with Retry-After
    Details:    OpenAI-style error body so SDK clients surface the reason.
    """
    return JSONResponse(
        status_code=503,
        content={"error": {"message": message, "type": "service_unavailable"}},
        headers={"Retry-After": str(int(retry_after or RETRY_AFTER_SECONDS))},
    )


def _payload_too_large() -> JSONResponse:
    """
    Input:      None
    Output:     413 response
    Details:    OpenAI-style error body so SDK clients surface the reason.
    """
    return JSONResponse(
        status_code=413,
        content={"error": {"message": "request body too large", "type": "payload_too_large"}},
    )


async def _read_bounded_body(request: Request) -> bytes:
    """
    Input:      request -- incoming client request
    Output:     body bytes, at most MAX_REQUEST_BODY_BYTES
    Details:
        Rejects with PayloadTooLarge when Content-Length exceeds the cap,
        without reading the body. Also counts bytes from request.stream()
        so a chunked body without Content-Length is bounded too.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            logger.warning("invalid Content-Length header: %s", content_length)
        else:
            if declared > MAX_REQUEST_BODY_BYTES:
                raise PayloadTooLarge()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_REQUEST_BODY_BYTES:
            raise PayloadTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


class _TrackedStream(StreamingResponse):
    """StreamingResponse that runs a close hook however the stream ends:
    finished, client disconnect, or cancellation."""

    def __init__(self, *args, on_close: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


class ModelSupervisor:
    """Owns the `ft serve` child, its broker lease, and the request count.

    Every load and unload runs under `lock`, so a request that races an
    idle unload waits for it and then reloads."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """
        Input:      clock -- monotonic seconds source (tests inject one)
                    transport -- httpx transport to the child (tests mock it)
        Output:     None
        Details:    Creates no lease and no child.
        """
        self._clock = clock
        self._transport = transport
        self.lock = asyncio.Lock()
        self.state = "unloaded"
        self.child: subprocess.Popen | None = None
        self.lease_id: str | None = None
        self.in_flight = 0
        self.last_activity = clock()
        self._client: httpx.AsyncClient | None = None
        self._tasks: list[asyncio.Task] = []
        # Set by a preempt, a lost lease, or shutdown; aborts a load in progress.
        self._evict_reason: str | None = None
        # Counts finished loads. A caller that saw a load finish (and fail)
        # while it waited on the lock reports that failure, not a new load.
        self._loads_finished = 0
        self._last_load_error: LoadFailed | None = None

    async def start(self) -> None:
        """
        Input:      None
        Output:     None
        Details:    Opens the child client and starts the heartbeat and idle loops.
        """
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{INTERNAL_PORT}",
            transport=self._transport,
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None),
        )
        self._tasks = [
            asyncio.create_task(self._loop(HEARTBEAT_INTERVAL_SECONDS, self.heartbeat_once)),
            asyncio.create_task(self._loop(IDLE_CHECK_SECONDS, self.idle_check)),
        ]

    async def shutdown(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Stops the loops, aborts any load in progress, stops the child,
            then releases the lease.
        """
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        self._evict_reason = "shutting down"
        async with self.lock:
            await self._unload_locked("shutdown")
        if self._client is not None:
            await self._client.aclose()
        close_client()

    async def _loop(self, interval: float, step: Callable) -> None:
        """
        Input:      interval -- seconds between steps
                    step -- coroutine function run each tick
        Output:     None (runs until cancelled)
        Details:    One failing tick is logged; the loop keeps running.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                await step()
            except Exception:
                logger.exception("background step %s failed", step.__name__)

    def _child_alive(self) -> bool:
        return self.child is not None and self.child.poll() is None

    def _child_pid(self) -> int | None:
        return self.child.pid if self._child_alive() else None

    async def ensure_loaded(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Returns once the model is ready. Raises LoadFailed if it
            cannot be loaded. The load runs in its own task, so a client
            that disconnects mid-load does not leave a half-started child.
        """
        if self.state == "ready" and self._child_alive():
            return
        task = asyncio.ensure_future(self._locked_load(self._loads_finished))
        task.add_done_callback(_consume_task_error)
        await asyncio.shield(task)

    async def _locked_load(self, seen_loads: int) -> None:
        """
        Input:      seen_loads -- finished-load counter when the caller arrived
        Output:     None
        Details:
            Single-flight: a caller that waited on the lock while another
            load failed gets that failure instead of a second acquire.
        """
        async with self.lock:
            if self.state == "ready" and self._child_alive():
                return
            if self._loads_finished != seen_loads and self._last_load_error is not None:
                raise LoadFailed(str(self._last_load_error))
            if self.state == "ready":
                logger.warning("ft serve exited on its own; reloading")
                await self._unload_locked("ft serve exited")
            try:
                await self._load_locked()
            except LoadFailed as exc:
                self._last_load_error = exc
                raise
            else:
                self._last_load_error = None
            finally:
                self._loads_finished += 1

    async def _load_locked(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Acquire lease, start child, register its pid, wait for ready.
            Caller holds `lock`. On any failure the child is stopped and
            the lease released before LoadFailed is raised.
        """
        self._evict_reason = None
        self.state = "loading"
        logger.info(
            "loading model: acquiring %d MB lease for caller=%s (fail_open=%s)",
            ACQUIRE_MB, CALLER_NAME, BROKER_FAIL_OPEN,
        )
        try:
            self.lease_id = await self._acquire_with_backoff()
        except LoadFailed:
            self.state = "unloaded"
            raise
        if self.lease_id is None:
            logger.warning("proceeding WITHOUT a broker lease (fail-open path)")
        else:
            logger.info("lease held: %s", self.lease_id)

        await _evict_ollama_models()
        cmd = _build_command()
        logger.info("starting: %s", " ".join(cmd))
        try:
            self.child = subprocess.Popen(cmd)
        except OSError as exc:
            logger.exception("could not start ft serve")
            await self._unload_locked("ft serve did not start")
            raise LoadFailed(f"could not start ft serve: {exc}") from exc
        try:
            if self.lease_id is not None:
                await self._register_child_pid()
            await self._wait_ready()
        except LoadFailed as exc:
            logger.warning("model load failed: %s", exc)
            await self._unload_locked("load failed")
            raise
        self.state = "ready"
        self.last_activity = self._clock()
        logger.info("model ready (pid %d)", self.child.pid)

    async def _acquire_with_backoff(self) -> str | None:
        """
        Input:      None
        Output:     lease_id, or None on the fail-open path
        Details:
            Unloads Ollama, then asks the broker for the lease. A 504 (no
            room yet) sleeps max(backoff, Retry-After) and retries, with
            backoff 2 s doubling to 60 s, until ACQUIRE_WAIT_SECONDS. The
            broker keeps this caller's wait age across retries, so the
            wait can preempt an idle preemptible holder. A game denial
            (503) or any other error fails at once with the broker's
            Retry-After. A preempt or shutdown during the wait aborts it.
        """
        deadline = self._clock() + ACQUIRE_WAIT_SECONDS
        delay = ACQUIRE_BACKOFF_MIN_SECONDS
        while True:
            # Every attempt: Ollama may have reloaded during the wait.
            await _evict_ollama_models()
            try:
                return await asyncio.to_thread(_broker_acquire, ACQUIRE_MB)
            except BrokerNoRoom as exc:
                sleep_for = max(delay, exc.retry_after or 0.0)
                if self._evict_reason is not None or self._clock() + sleep_for >= deadline:
                    logger.warning("no GPU lease after waiting %.0fs; not starting ft serve", ACQUIRE_WAIT_SECONDS)
                    raise LoadFailed(f"GPU unavailable: {exc}", exc.retry_after) from exc
                logger.info("no GPU room yet; retrying the lease in %.0fs", sleep_for)
                await asyncio.sleep(sleep_for)
                delay = min(delay * 2, ACQUIRE_BACKOFF_MAX_SECONDS)
            except BrokerAcquireError as exc:
                logger.warning("no GPU lease (%s); not starting ft serve", exc)
                raise LoadFailed(f"GPU unavailable: {exc}", exc.retry_after) from exc

    async def _register_child_pid(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            The first acquire precedes Popen, so the live child pid is
            registered through the broker's same-caller dedup path.
        """
        try:
            lease_id = await asyncio.to_thread(_broker_acquire, ACQUIRE_MB, pid=self.child.pid)
        except BrokerAcquireError as exc:
            logger.warning("could not register ft serve pid with the broker: %s", exc)
            raise LoadFailed(f"GPU unavailable: {exc}") from exc
        if lease_id is None:
            logger.warning("pid registration failed open; keeping lease %s", self.lease_id)
            return
        self.lease_id = lease_id

    async def _wait_ready(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Polls the child's GET /health until _ready_from_health is True.
            Raises LoadFailed if the child exits, reports a fatal error,
            is evicted, or LOAD_TIMEOUT_SECONDS passes.
        """
        deadline = self._clock() + LOAD_TIMEOUT_SECONDS
        while True:
            if self._evict_reason is not None:
                raise LoadFailed(f"load aborted: {self._evict_reason}")
            code = self.child.poll()
            if code is not None:
                raise LoadFailed(f"ft serve exited during load with code {code}")
            try:
                resp = await self._client.get("/health", timeout=5.0)
                if _ready_from_health(resp.json()):
                    return
            except (httpx.HTTPError, ValueError) as exc:
                # Only in the first moments of a load: the child's uvicorn
                # binds before the engine loads, then always answers JSON.
                logger.warning("ft serve /health not answering yet: %s", exc)
            if self._clock() >= deadline:
                raise LoadFailed(f"ft serve not ready after {LOAD_TIMEOUT_SECONDS:.0f}s")
            await asyncio.sleep(LOAD_POLL_SECONDS)

    async def _unload_locked(self, reason: str) -> None:
        """
        Input:      reason -- logged cause
        Output:     None
        Details:
            Caller holds `lock`. Stops the child first; releases the lease
            only after the child has exited, so VRAM is free before the
            broker hands it out again.
        """
        if self.child is None and self.lease_id is None:
            self.state = "unloaded"
            return
        logger.info("unloading model: %s", reason)
        if self.child is not None:
            try:
                await asyncio.to_thread(_stop_child, self.child)
            except subprocess.TimeoutExpired:
                logger.exception("ft serve survived SIGKILL; keeping lease %s", self.lease_id)
                return
            self.child = None
        lease_id, self.lease_id = self.lease_id, None
        if lease_id is not None:
            await asyncio.to_thread(_broker_release, lease_id)
        self.state = "unloaded"

    async def idle_check(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Unloads when the model is ready, nothing is in flight, and the
            last request finished IDLE_SECONDS ago. Also unloads if the
            child died, so a dead child does not keep a lease.
        """
        if self.state != "ready":
            return
        if not self._child_alive():
            async with self.lock:
                if self.state == "ready" and not self._child_alive():
                    logger.warning("ft serve exited on its own")
                    await self._unload_locked("ft serve exited")
            return
        if self.in_flight > 0 or self._clock() - self.last_activity < IDLE_SECONDS:
            return
        async with self.lock:
            idle_for = self._clock() - self.last_activity
            if self.state == "ready" and self.in_flight == 0 and idle_for >= IDLE_SECONDS:
                await self._unload_locked(f"idle for {idle_for:.0f}s")

    async def heartbeat_once(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            No-op without a lease. preempt=true stops the child at once,
            even mid-request or mid-load, then releases. ok=false
            re-acquires with the child pid; failure or a fail-open None
            unloads, so the child never runs without a lease. Neither path
            waits on the lock before flagging _evict_reason, so a long load
            aborts within one poll instead of blocking the heartbeat.
        """
        lease_id = self.lease_id
        if lease_id is None:
            return
        ok, preempt = await asyncio.to_thread(_broker_heartbeat, lease_id)
        if self.lease_id != lease_id:
            # Unloaded or re-acquired while the beat was in flight.
            return
        if preempt:
            logger.warning("broker preempted lease %s; stopping ft serve", lease_id)
            self._evict_reason = "preempted by the broker"
            async with self.lock:
                if self.lease_id == lease_id:
                    await self._unload_locked("preempted by the broker")
            return
        if ok:
            return
        logger.warning("heartbeat failed for lease %s; re-acquiring", lease_id)
        try:
            new_lease = await asyncio.to_thread(_broker_acquire, ACQUIRE_MB, pid=self._child_pid())
        except BrokerAcquireError:
            logger.exception("re-acquire after heartbeat failure was refused")
            new_lease = None
        if new_lease is not None:
            if self.lease_id == lease_id:
                self.lease_id = new_lease
                logger.info("re-acquired broker lease after heartbeat failure: %s", new_lease)
            else:
                # Model was unloaded during the re-acquire; return the new lease.
                await asyncio.to_thread(_broker_release, new_lease)
            return
        self._evict_reason = "lost GPU lease"
        async with self.lock:
            if self.lease_id == lease_id:
                # The old lease is gone on the broker; do not release it.
                self.lease_id = None
                await self._unload_locked("lost GPU lease")

    async def forward(self, request: Request):
        """
        Input:      request -- any client request except GET /health
        Output:     the child's response, streamed; or 503/502
        Details:
            Counts the request as in flight from arrival until its
            response finishes streaming, then stamps last_activity.
        """
        self.in_flight += 1
        finished = False

        async def finish() -> None:
            nonlocal finished
            if not finished:
                finished = True
                self.in_flight -= 1
                self.last_activity = self._clock()

        handed_off = False
        try:
            body = await _read_bounded_body(request)
            await self.ensure_loaded()
            headers = [
                (k, v) for k, v in request.headers.items() if k.lower() not in _REQUEST_DROP
            ]
            # raw_path keeps the client's percent-encoding intact. Some ASGI
            # clients (httpx.ASGITransport) leave the query on it.
            raw_path = (request.scope.get("raw_path") or b"").partition(b"?")[0]
            target = raw_path.decode("latin-1") or request.url.path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            upstream_req = self._client.build_request(
                request.method, target, headers=headers, content=body,
            )
            upstream = await self._client.send(upstream_req, stream=True)
            handed_off = True
        except PayloadTooLarge:
            logger.warning(
                "rejecting %s %s: request body too large",
                request.method, request.url.path,
            )
            return _payload_too_large()
        except LoadFailed as exc:
            logger.warning("answering 503 to %s %s: %s", request.method, request.url.path, exc)
            return _unavailable(str(exc), exc.retry_after)
        except httpx.HTTPError as exc:
            logger.warning("ft serve unreachable for %s %s: %s", request.method, request.url.path, exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "ft serve unreachable", "type": "bad_gateway"}},
            )
        finally:
            # Until the stream exists, this request's in-flight slot ends here.
            if not handed_off:
                await finish()

        async def close() -> None:
            try:
                await upstream.aclose()
            except Exception as exc:
                logger.warning("upstream response close failed: %s", exc)
            finally:
                await finish()

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                # Child stopped mid-stream (preempt or crash); end the stream.
                logger.warning("stream from ft serve ended early: %s", exc)

        resp_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        return _TrackedStream(
            relay(), status_code=upstream.status_code, headers=resp_headers, on_close=close,
        )


def _consume_task_error(task: asyncio.Task) -> None:
    """
    Input:      task -- a finished load task
    Output:     None
    Details:
        Marks the exception retrieved when the waiting client already
        left, so asyncio does not log "exception was never retrieved".
        The load itself already logged the cause.
    """
    if not task.cancelled():
        task.exception()


def create_app(supervisor: ModelSupervisor | None = None) -> FastAPI:
    """
    Input:      supervisor -- injected by tests; a fresh one otherwise
    Output:     FastAPI app
    Details:    GET /health is local; every other route is forwarded.
    """
    sup = supervisor or ModelSupervisor()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        await sup.start()
        try:
            yield
        finally:
            await sup.shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.supervisor = sup

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_state": sup.state}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str):
        return await sup.forward(request)

    return app


def main() -> None:
    """
    Input:      None
    Output:     None
    Details:    Serves the wrapper. uvicorn's SIGTERM runs the lifespan shutdown.
    """
    logger.info(
        "wrapper listening on %s:%d; ft serve will bind 127.0.0.1:%d on first request",
        HOST, PORT, INTERNAL_PORT,
    )
    uvicorn.run(
        create_app(),
        host=HOST,
        port=PORT,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


if __name__ == "__main__":
    main()
