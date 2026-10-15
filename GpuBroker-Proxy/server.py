"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

GpuBroker-Proxy: a stateless reverse proxy for Ollama's full API
(native /api/* and OpenAI-compatible /v1/*). It routes each request
to a backend from registry.json and gates generation work through
GpuBroker /acquire and /release before it forwards.

This proxy does not edit request or response bodies. It peeks at the
JSON "model" field only to pick a backend. It then forwards the original
bytes, including streaming responses.

registry.json currently lists Talkie (model talkie-1930-13b), FreeToken
(model ornith-1.5-35b-a3b), and Ollama (default, empty models list).
Ollama is reached at http://ollama:11434 on the ollama_default podman
network. Ollama no longer publishes that port to the host, so this
proxy is the host path to it. A new OpenAI-compatible backend is a new
registry.json entry. No proxy code change is required.

Admission accounting: every gated request calls /acquire with mb=0.
Ollama VRAM is already tracked by the broker via /api/ps (_ollama_mb).
Charging a size here would double-count. The mb=0 acquire gives each
gated request a lease. The broker does not queue mb=0 acquires in its
FIFO, so a large waiter that cannot fit does not block them. They still
wait while the ledger is over budget and are refused while a game is
running.
There is no pre-load size to gate on. Ollama /api/ps reports size_vram
only after a model finishes loading.

Admission signaling: on denial this proxy returns HTTP 429 with
Retry-After. That matches the OpenAI rate-limit convention. Two broker
responses are denials: 504 (acquire timed out waiting for room) and 503
(unmanaged GPU load such as a game, or the lease cap; the broker's
Retry-After is passed through). 503 detail selects the caller message:
unmanaged_gpu_load, too_many_leases, or a generic refusal. On
broker-unreachable (connection failure, not a denial) this proxy fails
open by default (BROKER_FAIL_OPEN). Ollama's footprint has always been
observed passively, never gated with hard exclusivity.

Endpoints:
    GET, POST, DELETE /{path} -> proxied to the resolved backend /{path}.
                     Gated through the broker when the request declares
                     a "model", except UNGATED_MODEL_PATHS. POST /api/show
                     is Ollama model metadata and does not load weights,
                     so it is routed by model but not acquire/release.
                     Other metadata and management calls with no model
                     field pass through unmetered.
    GET /health   -> {"status": "ok"}
"""

# Imports

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import uuid
from typing import AsyncIterator

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Globals

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("GPUBROKER_PROXY_PORT", "11434"))
BROKER_URL = os.environ.get("BROKER_URL", "http://host.containers.internal:8792")
BROKER_FAIL_OPEN = os.environ.get("BROKER_FAIL_OPEN", "true").lower() in ("true", "1", "yes")
BROKER_ACQUIRE_TIMEOUT_S = float(os.environ.get("BROKER_ACQUIRE_TIMEOUT_S", "10.0"))
PROXY_RETRY_AFTER_SECONDS = float(os.environ.get("PROXY_RETRY_AFTER_SECONDS", "3.0"))
REGISTRY_PATH = os.environ.get("PROXY_REGISTRY_PATH", "/app/registry.json")
# One third of the broker lease TTL (180 s).
PROXY_HEARTBEAT_INTERVAL_S = 60.0
# Vision requests carry base64 images; 64 MiB is generous.
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
# Only endpoints verified not to load weights belong here.
UNGATED_MODEL_PATHS = {"api/show"}


# Connection-failure retry budget for the proxy's own /acquire and
# /release calls. Distinct from GpuBroker's internal admission wait,
# which is bounded by BROKER_ACQUIRE_TIMEOUT_S above. Shorter than
# Flux2Klein9B's matching constants. OAIE callers are interactive and
# more latency-sensitive than an occasional model-load acquire.
_ACQUIRE_MAX_ATTEMPTS = 3
_ACQUIRE_BACKOFF_MIN_S = 0.5
_ACQUIRE_BACKOFF_MAX_S = 2.0

_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})

_registry: list[dict] = []
_broker_client: httpx.AsyncClient | None = None


# Functions

def _load_registry(path: str) -> list[dict]:
    """
    Input:      path - filesystem path to the JSON registry file
    Output:     list[dict] - backend entries, each {name, base_url, models, default}
    Details:
        Loads the model -> backend registry once at startup. Warns (does not
        raise) on an empty or ambiguous (multiple "default": true entries)
        registry -- the proxy still starts, it just resolves nothing until
        registry.json is fixed and the container restarted.
    """
    with open(path, "r") as f:
        data = json.load(f)
    backends = data.get("backends", [])
    if not backends:
        logger.warning("registry at %s has no backends configured", path)
    default_count = sum(1 for b in backends if b.get("default"))
    if default_count > 1:
        logger.warning(
            "registry at %s has %d backends flagged default; first match wins",
            path, default_count,
        )
    logger.info("loaded %d backend(s) from registry: %s", len(backends), path)
    return backends


def _resolve_backend(model: str | None) -> dict | None:
    """
    Input:      model - the "model" field from the request body, or None
    Output:     dict - the matching backend entry, or None if none resolves
    Details:
        Exact-match model against each backend's "models" list first.
        Falls back to the backend flagged "default": true. Talkie matches
        talkie-1930-13b. Unmatched models, including Ollama names, fall
        through to Ollama (default, empty models list).
    """
    if model is not None:
        for backend in _registry:
            if model in backend.get("models", []):
                return backend
    for backend in _registry:
        if backend.get("default"):
            return backend
    return None


def _extract_model(body: bytes) -> str | None:
    """
    Input:      body - raw request body bytes
    Output:     str | None - the "model" field if present and a string
    Details:
        Empty bodies (for example GET /v1/models) are not an error.
        Skip parse. Non-empty bodies that fail as JSON (for example a
        multipart /v1/audio/transcriptions upload, not supported by any
        registered backend today) are logged and treated as model=None.
        That falls through to the registry default instead of failing
        the request.
    """
    if not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("could not parse request body as JSON to extract model: %s", exc)
        return None
    if isinstance(data, dict):
        model = data.get("model")
        if isinstance(model, str):
            return model
    return None


def _strip_hop_by_hop(headers) -> dict[str, str]:
    """
    Input:      headers - mapping of header name -> value
    Output:     dict[str, str] - headers with hop-by-hop and framing headers removed
    Details:
        Strips headers that are specific to one leg of the connection
        (Connection, Transfer-Encoding, etc.) plus Host and Content-Length,
        which must be regenerated for the new leg rather than forwarded
        verbatim.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


async def _broker_acquire(caller: str) -> str | None:
    """
    Input:      caller - unique per-request caller name for this lease
    Output:     str | None - lease_id if acquired, None if fail-open (no lease)
    Details:
        Calls GpuBroker /acquire with mb=0 (see module docstring).
        On denial (broker long-poll timed out, no GPU room) raises
        HTTPException(429) with Retry-After. The caller should retry.
        On broker 503 also raises HTTPException(429), with the broker's
        Retry-After when present. 503 detail unmanaged_gpu_load,
        too_many_leases, or anything else each get a distinct caller
        message. A 503 is a denial, never a reason to fail open.
        On connection failure: retries with backoff, then fails open
        (returns None) or closed (raises 503) per BROKER_FAIL_OPEN.
    """
    backoff = _ACQUIRE_BACKOFF_MIN_S
    attempts = 0
    while attempts < _ACQUIRE_MAX_ATTEMPTS:
        attempts += 1
        try:
            resp = await _broker_client.post(
                f"{BROKER_URL}/acquire",
                json={"caller": caller, "mb": 0, "timeout_s": BROKER_ACQUIRE_TIMEOUT_S},
                timeout=BROKER_ACQUIRE_TIMEOUT_S + 5.0,
            )
            if resp.status_code == 504:
                logger.warning("broker denied admission (no GPU room) for caller=%s", caller)
                raise HTTPException(
                    status_code=429,
                    detail="GPU busy, retry later",
                    headers={"Retry-After": resp.headers.get("Retry-After") or str(PROXY_RETRY_AFTER_SECONDS)},
                )
            if resp.status_code == 503:
                retry_after = resp.headers.get("Retry-After") or str(PROXY_RETRY_AFTER_SECONDS)
                detail = None
                try:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        detail = payload.get("detail")
                except ValueError:
                    logger.warning("broker 503 body was not JSON for caller=%s", caller)
                if detail == "unmanaged_gpu_load":
                    message = "GPU reserved for unmanaged load, retry later"
                elif detail == "too_many_leases":
                    message = "broker lease limit reached, retry later"
                else:
                    message = "GPU broker refused admission, retry later"
                logger.warning(
                    "broker refused admission (detail=%s) for caller=%s, retry_after=%s",
                    detail, caller, retry_after,
                )
                raise HTTPException(
                    status_code=429,
                    detail=message,
                    headers={"Retry-After": retry_after},
                )
            resp.raise_for_status()
            lease_id = resp.json().get("lease_id")
            logger.info("proxy acquired broker lease=%s caller=%s", lease_id, caller)
            return lease_id
        except HTTPException:
            raise
        except httpx.ConnectError:
            logger.exception("broker unreachable (attempt %d/%d)", attempts, _ACQUIRE_MAX_ATTEMPTS)
            if attempts < _ACQUIRE_MAX_ATTEMPTS:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _ACQUIRE_BACKOFF_MAX_S)
                continue
            if BROKER_FAIL_OPEN:
                logger.warning("broker unreachable after %d attempts, failing open (unmetered)", attempts)
                return None
            logger.error("broker unreachable after %d attempts, failing closed", attempts)
            raise HTTPException(status_code=503, detail="broker unreachable")
        except httpx.HTTPError:
            logger.exception("broker request failed for caller=%s", caller)
            if BROKER_FAIL_OPEN:
                logger.warning("broker error, failing open (unmetered)")
                return None
            raise HTTPException(status_code=502, detail="broker error")
    # Should not reach here. The loop always returns or raises.
    logger.warning("broker acquire: unexpected end of retry loop for caller=%s", caller)
    if BROKER_FAIL_OPEN:
        return None
    raise HTTPException(status_code=503, detail="broker acquire exhausted retries")


async def _broker_release(lease_id: str | None, caller: str) -> None:
    """
    Input:      lease_id - lease to release, or None to skip (fail-open case)
                caller - caller name, for logging only
    Output:     None
    Details:
        Best-effort release. Logs and continues on failure. GpuBroker's
        sweeper reclaims an un-released lease after LEASE_TTL_SECONDS, so
        a failed release here is not a permanent leak.
    """
    if lease_id is None:
        return
    try:
        resp = await _broker_client.post(
            f"{BROKER_URL}/release", json={"lease_id": lease_id}, timeout=5.0,
        )
        resp.raise_for_status()
        logger.info("proxy released broker lease=%s caller=%s", lease_id, caller)
    except httpx.HTTPError:
        logger.warning("broker release failed for lease=%s caller=%s (continuing)", lease_id, caller)


async def _read_bounded_body(request: Request) -> bytes:
    """
    Input:      request - the incoming HTTP request
    Output:     bytes - the full request body, at most MAX_REQUEST_BODY_BYTES
    Details:
        Rejects with 413 when Content-Length exceeds the cap, without
        reading the body. Also counts bytes from request.stream() so a
        chunked body without Content-Length is bounded too.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            logger.warning("invalid Content-Length header: %s", content_length)
        else:
            if declared > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request body too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_REQUEST_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _heartbeat_loop(lease_id: str, caller: str) -> None:
    """
    Input:      lease_id - live zero-size lease to keep alive
                caller - caller name, for logging only
    Output:     None
    Details:
        POST /heartbeat every PROXY_HEARTBEAT_INTERVAL_S while a leased
        response streams. A heartbeat HTTP failure logs a warning and
        keeps the loop running. Ignore preempt: a zero-size lease holds
        no VRAM. If the broker answers ok: false, log a warning once
        and stop heartbeating.
    """
    while True:
        await asyncio.sleep(PROXY_HEARTBEAT_INTERVAL_S)
        try:
            resp = await _broker_client.post(
                f"{BROKER_URL}/heartbeat",
                json={"lease_id": lease_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning(
                "broker heartbeat failed for lease=%s caller=%s (continuing)",
                lease_id, caller,
            )
            continue
        if not data.get("ok", False):
            logger.warning(
                "broker heartbeat returned ok=false for lease=%s caller=%s; stopping heartbeats",
                lease_id, caller,
            )
            return
        # Ignore preempt: zero-size lease, holds no VRAM.


async def _stream_backend_response(
    resp: httpx.Response, lease_id: str | None, caller: str,
) -> AsyncIterator[bytes]:
    """
    Input:      resp - streaming httpx.Response from the backend
                lease_id, caller - passed through to release the lease once done
    Output:     AsyncIterator[bytes] - the response body, chunk by chunk
    Details:
        Releases the broker lease only after the full response has drained
        (success) or the generator is torn down early (client disconnect,
        backend error mid-stream). Never right after headers. The backend
        may still be resident and generating for the whole stream.
        While the stream is open, a background task heartbeats the lease.
        The heartbeat is cancelled before release. resp.aclose() failure
        is logged and does not skip release.
        Starlette cancels this generator in an anyio cancel scope on
        http.disconnect. Shield cleanup so gather, aclose, and release
        still finish; CancelledError still propagates after this block.
    """
    hb_task = None
    if lease_id is not None:
        hb_task = asyncio.create_task(_heartbeat_loop(lease_id, caller))
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        # Shield, not a detached create_task: cleanup stays in this
        # generator so release still runs exactly once and callers
        # observe it before the ASGI request returns. anyio is already
        # a Starlette dependency. Use sync `with` so the shield is
        # armed before any await (async with would checkpoint first).
        with anyio.CancelScope(shield=True):
            try:
                if hb_task is not None:
                    hb_task.cancel()
                    await asyncio.gather(hb_task, return_exceptions=True)
            finally:
                try:
                    await resp.aclose()
                except Exception:
                    logger.warning("upstream response close failed for caller=%s", caller)
                finally:
                    await _broker_release(lease_id, caller)


async def proxy_any(path: str, request: Request):
    """
    Input:      path - the full request path after the leading slash;
                request - the incoming HTTP request
    Output:     StreamingResponse - the backend's response, forwarded verbatim
    Details:
        Forwards GET, POST, and DELETE paths to the resolved backend, not
        only /v1/*. Headroom's headroom-ollama calls Ollama GET /api/ps
        directly. Restricting routing to /v1/* would 404 those callers.

        Admission-gates requests that declare a "model" in the body,
        except UNGATED_MODEL_PATHS. POST /api/show includes a model field
        but does not load weights, so it is routed by model and skipped
        for acquire/release. GET /api/ps, /api/tags, and /v1/models have
        no model field, so they pass through unmetered.
    """
    body = await _read_bounded_body(request)
    model = _extract_model(body)
    backend = _resolve_backend(model)
    if backend is None:
        logger.warning("no backend registered for model=%r path=%s", model, path)
        raise HTTPException(status_code=404, detail=f"no backend registered for model {model!r}")

    lease_id = None
    caller = f"oaie-proxy:{backend['name']}:{uuid.uuid4().hex}"
    if model is not None and path not in UNGATED_MODEL_PATHS:
        lease_id = await _broker_acquire(caller)

    upstream_url = f"{backend['base_url']}/{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"
    headers = _strip_hop_by_hop(dict(request.headers))

    try:
        upstream_req = _broker_client.build_request(
            request.method, upstream_url, headers=headers, content=body,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        )
        upstream_resp = await _broker_client.send(upstream_req, stream=True)
    except httpx.HTTPError:
        logger.exception("upstream request to backend=%s failed", backend["name"])
        await _broker_release(lease_id, caller)
        raise HTTPException(status_code=502, detail="upstream backend unreachable")

    response_headers = _strip_hop_by_hop(dict(upstream_resp.headers))
    return StreamingResponse(
        _stream_backend_response(upstream_resp, lease_id, caller),
        status_code=upstream_resp.status_code,
        headers=response_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


async def health():
    """
    Input:      None
    Output:     dict - {"status": "ok"}
    Details:    Liveness endpoint for the container healthcheck.
    """
    return {"status": "ok"}


async def _disabled_docs():
    """
    Input:      None
    Output:     never returns
    Details:    FastAPI docs are disabled. Explicit 404 so the catch-all
                proxy route does not forward /docs or /openapi.json.
    """
    raise HTTPException(status_code=404, detail="Not Found")


# Wire up endpoints and startup via lifespan

@asynccontextmanager
async def lifespan(app_ctx: FastAPI):
    """
    Input:      app_ctx - the FastAPI app instance
    Output:     None (async context manager)
    Details:
        Startup: loads the backend registry and opens one shared httpx
        client reused across every proxied request and broker call.
        A proxy forwards sustained streaming traffic. A persistent
        client avoids reconnecting per request.
        Shutdown: closes the shared client.
    """
    global _broker_client, _registry
    _registry = _load_registry(REGISTRY_PATH)
    _broker_client = httpx.AsyncClient()
    logger.info("GpuBroker-Proxy starting: %d backend(s) registered", len(_registry))
    yield
    await _broker_client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/docs", _disabled_docs, methods=["GET"])
app.add_api_route("/redoc", _disabled_docs, methods=["GET"])
app.add_api_route("/openapi.json", _disabled_docs, methods=["GET"])
app.add_api_route("/{path:path}", proxy_any, methods=["GET", "POST", "DELETE"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
