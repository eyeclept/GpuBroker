"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

GpuBroker-Proxy: a stateless reverse proxy speaking Ollama's full API
surface -- both its native /api/* endpoints and the OpenAI-compatible
/v1/* subset -- routing requests to a configured model -> upstream-backend
registry and gating real generation work through GpuBroker's existing
/acquire and /release endpoints before forwarding.

Traffic light, not editor: this proxy never mutates request or response
bodies. It peeks at the "model" field of a JSON request body only to pick a
backend, then forwards the original bytes verbatim (including streaming
responses) to that backend and streams the reply straight back.

Ollama is the only registered backend today (see registry.json), reached at
http://ollama:11434 over the ollama_default podman network -- Ollama's own
docker-compose.yml no longer publishes that port to the host, so this proxy
is the only path to it. A future OpenAI-compatible backend (e.g. a
Whisper/audio service) is added purely as a new registry.json entry; no
proxy code changes.

Admission accounting: every proxied request calls GpuBroker's /acquire with
mb=0, never a real size estimate. Ollama's actual VRAM footprint is already
tracked passively by the broker itself via its own /api/ps poll (_ollama_mb
in server.py) -- charging mb here too would double-count against the shared
budget. The mb=0 acquire call exists purely to give each proxied request a
lease and a turn in the broker's fairness queue, so a burst of Ollama/OAIE
traffic queues behind Flux2Klein9B/Needle's real VRAM holds under contention
instead of firing blind into an already-saturated GPU. There is no
pre-load size estimate to gate on regardless -- confirmed live, Ollama's
/api/ps only reports size_vram after a model finishes loading.

Admission signaling: on denial (broker's /acquire itself times out waiting
for room) this proxy returns HTTP 429 with a Retry-After header, matching
the OpenAI API's own rate-limit convention and relying on OAIE SDKs' builtin
429/Retry-After retry logic. On broker-unreachable (connection failure, not
a denial) this proxy fails open by default (BROKER_FAIL_OPEN) -- consistent
with the existing accepted risk that Ollama's footprint has always been
observed passively, never gated with hard exclusivity.

Endpoints:
    ANY /{path}   -> proxied to the resolved backend's /{path} (gated
                     through the broker only when the request declares a
                     "model"; metadata/management calls pass straight
                     through unmetered)
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

# Connection-failure retry budget for the broker's own /acquire and /release
# calls (distinct from GpuBroker's internal admission long-poll, which is
# bounded by BROKER_ACQUIRE_TIMEOUT_S above). Shorter than Flux2Klein9B's
# matching constants -- OAIE callers are usually interactive and more
# latency-sensitive than an occasional model-load acquire.
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
        Exact-match model against each backend's declared "models" list
        first; falls back to whichever backend is flagged "default": true.
        With today's single-backend registry (Ollama, flagged default, empty
        models list) every request resolves to it regardless of the model
        name -- this generalizes cleanly once a second, model-specific
        backend is registered.
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
        Empty bodies (e.g. GET /v1/models) are not an error -- short-circuit
        before attempting to parse. Non-empty bodies that fail to parse as
        JSON (e.g. a multipart /v1/audio/transcriptions upload -- not
        supported by any backend registered today) are logged and treated
        as model=None, which falls through to the registry's default
        backend rather than failing the request outright.
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
        Calls GpuBroker's /acquire with mb=0 (see module docstring for why).
        On denial (broker's own long-poll timed out, no GPU room) raises
        HTTPException(429) with a Retry-After header -- caller should retry.
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
                    headers={"Retry-After": str(PROXY_RETRY_AFTER_SECONDS)},
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
    # Should not reach here -- loop always returns or raises.
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
        Best-effort release. Logs and continues on failure -- GpuBroker's own
        sweeper reclaims an un-released lease after LEASE_TTL_SECONDS, so a
        failed release here is not a permanent leak.
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
        backend error mid-stream) -- never right after headers, since the
        backend may still be resident/generating for the whole stream.
    """
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()
        await _broker_release(lease_id, caller)


async def proxy_any(path: str, request: Request):
    """
    Input:      path - the full request path after the leading slash;
                request - the incoming HTTP request
    Output:     StreamingResponse - the backend's response, forwarded verbatim
    Details:
        Forwards ANY path (not just /v1/*) to the resolved backend --
        confirmed live that Headroom's headroom-ollama calls Ollama's native
        GET /api/ps directly, not just the OpenAI-compatible /v1/* surface,
        so restricting routing to /v1/* silently 404s existing consumers.
        "Identical behavior to before the reroute, no config change needed"
        (Phase 5's own Step 5 contract) requires forwarding the full
        namespace Ollama itself answered on port 11434.

        Admission-gates (broker acquire/release) only requests that declare
        a "model" in the body -- the signal that this is real generation
        work, not a metadata/management call (GET /api/ps, /api/tags,
        /v1/models etc. never carry a model field and so pass straight
        through unmetered, exactly as Ollama itself behaved for those calls
        before this proxy existed). Known imperfection: a handful of
        Ollama's own metadata endpoints (e.g. POST /api/show) do include a
        "model" field despite not consuming GPU generation resources, and so
        get gated too -- accepted as a minor, non-breaking inefficiency
        rather than hardcoding a full per-endpoint classification table.
    """
    body = await request.body()
    model = _extract_model(body)
    backend = _resolve_backend(model)
    if backend is None:
        logger.warning("no backend registered for model=%r path=%s", model, path)
        raise HTTPException(status_code=404, detail=f"no backend registered for model {model!r}")

    lease_id = None
    caller = f"oaie-proxy:{backend['name']}:{uuid.uuid4().hex}"
    if model is not None:
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


# Wire up endpoints and startup via lifespan

@asynccontextmanager
async def lifespan(app_ctx: FastAPI):
    """
    Input:      app_ctx - the FastAPI app instance
    Output:     None (async context manager)
    Details:
        Startup: loads the backend registry and opens one shared httpx
        client reused across every proxied request and broker call (a
        proxy forwards sustained/streaming traffic, unlike GpuBroker's
        clients which only make occasional short round-trips -- a
        persistent client avoids reconnecting per request).
        Shutdown: closes the shared client.
    """
    global _broker_client, _registry
    _registry = _load_registry(REGISTRY_PATH)
    _broker_client = httpx.AsyncClient()
    logger.info("GpuBroker-Proxy starting: %d backend(s) registered", len(_registry))
    yield
    await _broker_client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/{path:path}", proxy_any, methods=["GET", "POST", "DELETE"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
