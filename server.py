"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

GpuBroker: a small cross-process VRAM-budget lock for this host's single
RTX 4070 shared by Flux2Klein9B, Needle, and Ollama.

Not a proxy in the GPU data path -- callers still talk to CUDA/XLA directly.
This is a gatekeeper: managed clients (Flux2Klein9B, Needle) call POST
/acquire before actually loading weights onto the GPU and POST /release once
evicted, so the broker's ledger always reflects what's about to be resident,
not just what nvidia-smi has already observed. Ollama is not modified to
cooperate (its Go source is out of scope) -- its footprint is accounted for
passively via its own GET /api/ps endpoint.

Endpoints:
  POST /acquire   {caller, mb, timeout_s} -> {granted: true, lease_id} or 504
  POST /release    {lease_id}             -> {released: true}
  POST /heartbeat  {lease_id}             -> {ok: true}
  GET  /status                           -> ledger + live budget numbers
  GET  /health                           -> {"status": "ok"}
"""

# Imports

import asyncio
from contextlib import asynccontextmanager
import itertools
import json
import logging
import os
import subprocess
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Globals

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("GPU_BROKER_PORT", "8792"))
SAFETY_MARGIN_MB = int(os.environ.get("GPU_SAFETY_MARGIN_MB", "1536"))
LEDGER_PATH = os.environ.get("GPU_BROKER_LEDGER_PATH", "/data/leases.json")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.containers.internal:11434")
OLLAMA_POLL_INTERVAL_SECONDS = int(os.environ.get("OLLAMA_POLL_INTERVAL_SECONDS", "20"))
SWEEP_INTERVAL_SECONDS = int(os.environ.get("GPU_BROKER_SWEEP_INTERVAL_SECONDS", "30"))
LEASE_TTL_SECONDS = int(os.environ.get("GPU_BROKER_LEASE_TTL_SECONDS", "180"))
ACQUIRE_POLL_MIN_SECONDS = 1.0
ACQUIRE_POLL_MAX_SECONDS = 10.0

_lock = asyncio.Lock()
_leases: dict[str, dict] = {}          # lease_id -> {caller, mb, acquired_at, last_seen}
_caller_to_lease: dict[str, str] = {}   # caller -> lease_id
_pending_seq = itertools.count()
_pending: set[int] = set()             # currently-waiting acquire sequence numbers

_gpu_total_mb: float = 0.0
_ollama_mb: float = 0.0


class AcquireRequest(BaseModel):
    """
    Input: JSON body with caller (str), mb (float), timeout_s (float, default 300.0)
    Output: None (Pydantic model)
    Details: Request model for /acquire endpoint
    """
    caller: str
    mb: float
    timeout_s: float = 300.0


class ReleaseRequest(BaseModel):
    """
    Input: JSON body with lease_id (str)
    Output: None (Pydantic model)
    Details: Request model for /release endpoint
    """
    lease_id: str


class HeartbeatRequest(BaseModel):
    """
    Input: JSON body with lease_id (str)
    Output: None (Pydantic model)
    Details: Request model for /heartbeat endpoint
    """
    lease_id: str


# Functions

def _nvidia_smi_query(field: str) -> float:
    """
    Input:      field - nvidia-smi query field (e.g., "memory.used", "memory.total")
    Output:     float - queried value in MB
    Details:
        Synchronous subprocess call to nvidia-smi. Queries the first GPU's memory
        field and returns it as a float. Raises on timeout or failure.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return float(out.stdout.strip().splitlines()[0])


async def _live_used_mb() -> float:
    """
    Input:      None
    Output:     float - current GPU memory used in MB
    Details:
        Async wrapper around nvidia-smi memory.used query. Runs synchronously
        in a thread to avoid blocking event loop. Logs exception and re-raises
        on failure to fail-close admission checks.
    """
    try:
        return await asyncio.to_thread(_nvidia_smi_query, "memory.used")
    except Exception:
        logger.exception("nvidia-smi memory.used query failed -- failing closed on admission")
        raise


def _persist_ledger_sync():
    """
    Input:      None
    Output:     None
    Details:
        Synchronous ledger persistence to disk. Writes to temp file then
        atomically replaces LEDGER_PATH to avoid corruption on write failure.
    """
    tmp_path = LEDGER_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"leases": _leases}, f)
    os.replace(tmp_path, LEDGER_PATH)


async def _persist_ledger():
    """
    Input:      None
    Output:     None
    Details:
        Async wrapper around _persist_ledger_sync. Runs synchronously in a
        thread. Logs exception and re-raises on OSError to catch missing
        /data mount or permission issues.
    """
    try:
        await asyncio.to_thread(_persist_ledger_sync)
    except OSError:
        logger.exception("failed to persist GPU broker ledger to %s", LEDGER_PATH)
        raise


def _reserved_mb() -> float:
    """
    Input:      None
    Output:     float - total MB reserved by all active leases
    Details:
        Sums the mb field across all active leases in the in-memory ledger.
        Must be called within _lock to avoid TOCTOU.
    """
    return sum(lease["mb"] for lease in _leases.values())


async def _poll_ollama():
    """
    Input:      None
    Output:     None
    Details:
        Infinite loop polling Ollama's GET /api/ps endpoint every
        OLLAMA_POLL_INTERVAL_SECONDS to read model footprints. Updates global
        _ollama_mb. Logs on each model list change. Catches exceptions and
        holds last-known-good value on transient failures.
    """
    global _ollama_mb
    last_value = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                resp = await client.get(f"{OLLAMA_URL}/api/ps")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                total = sum(m.get("size_vram", 0) for m in models) / (1024 * 1024)
                _ollama_mb = total
                if total != last_value:
                    logger.info("ollama /api/ps: %.0f MB across %d model(s)", total, len(models))
                    last_value = total
            except Exception as exc:
                logger.warning("ollama /api/ps poll failed, holding last-known-good value (%.0f MB): %s",
                                _ollama_mb, exc)
            await asyncio.sleep(OLLAMA_POLL_INTERVAL_SECONDS)


async def _sweep_expired_leases():
    """
    Input:      None
    Output:     None
    Details:
        Infinite loop that runs every SWEEP_INTERVAL_SECONDS. Reclaims any
        lease whose last_seen is older than LEASE_TTL_SECONDS. Logs reclaim
        events at WARNING level. Persists ledger after each sweep.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        async with _lock:
            now = time.monotonic()
            expired = [
                (lease_id, lease) for lease_id, lease in _leases.items()
                if now - lease["last_seen"] > LEASE_TTL_SECONDS
            ]
            for lease_id, lease in expired:
                del _leases[lease_id]
                if _caller_to_lease.get(lease["caller"]) == lease_id:
                    del _caller_to_lease[lease["caller"]]
                logger.warning(
                    "reclaiming leaked lease %s (caller=%s, mb=%.0f) -- no heartbeat for %ds",
                    lease_id, lease["caller"], lease["mb"], LEASE_TTL_SECONDS,
                )
            if expired:
                await _persist_ledger()


async def startup():
    """
    Input:      None
    Output:     None
    Details:
        FastAPI startup hook. Queries GPU total memory, restores ledger from
        disk (resetting last_seen timestamps to prevent immediate expiry),
        and launches background tasks for Ollama polling and lease sweeping.
        Logs exceptions and re-raises to fail startup if GPU query fails.
    """
    global _gpu_total_mb
    try:
        _gpu_total_mb = await asyncio.to_thread(_nvidia_smi_query, "memory.total")
    except Exception:
        logger.exception("failed to read GPU total memory at startup")
        raise

    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH) as f:
                data = json.load(f)
            now = time.monotonic()
            restored = data.get("leases", {})
            for lease in restored.values():
                lease["last_seen"] = now
            _leases.update(restored)
            for lease_id, lease in _leases.items():
                _caller_to_lease[lease["caller"]] = lease_id
            logger.info("ledger loaded at startup: %d lease(s) restored", len(_leases))
        except (OSError, ValueError):
            logger.exception("ledger file at %s corrupt/unreadable, starting from empty ledger", LEDGER_PATH)
            _leases.clear()
            _caller_to_lease.clear()
    else:
        logger.info("no ledger file found at %s, fresh start", LEDGER_PATH)

    asyncio.create_task(_poll_ollama())
    asyncio.create_task(_sweep_expired_leases())


async def health():
    """
    Input:      None
    Output:     dict - {"status": "ok"}
    Details:
        GET /health endpoint for liveness checks. Always returns OK.
    """
    return {"status": "ok"}


async def status():
    """
    Input:      None
    Output:     dict - ledger snapshot with budget/reservation/ollama details
    Details:
        GET /status endpoint. Returns current GPU budget, total capacity,
        safety margin, reserved MB (all leases), Ollama footprint, live
        nvidia-smi used MB, and full lease ledger under lock.
    """
    async with _lock:
        used_mb = await _live_used_mb()
        return {
            "budget_mb": _gpu_total_mb - SAFETY_MARGIN_MB,
            "gpu_total_mb": _gpu_total_mb,
            "safety_margin_mb": SAFETY_MARGIN_MB,
            "reserved_mb": _reserved_mb(),
            "ollama_mb": _ollama_mb,
            "nvidia_smi_used_mb": used_mb,
            "leases": _leases,
        }


async def acquire(req: AcquireRequest):
    """
    Input:      req - AcquireRequest with caller, mb, timeout_s
    Output:     dict - {"granted": True, "lease_id": str} or raises HTTPException(504)
    Details:
        POST /acquire endpoint. Implements admission formula:
        reserved_mb + ollama_mb + requested_mb <= budget_mb, where budget_mb =
        gpu_total_mb - SAFETY_MARGIN_MB. SAFETY_MARGIN_MB measures baseline
        idle GPU usage (desktop compositor, driver reserve, stray usage --
        see .env.example), which is exactly what nvidia-smi reports with an
        empty ledger. Live usage is therefore not added to the gate: doing so
        charged that same baseline twice, once via the margin and once via
        the live reading, and refused requests that physically fit (see
        Assist/ISSUE_double_counted_margin.md). Live nvidia-smi usage is
        still read and compared against margin + reserved, purely to log a
        warning when something is consuming VRAM the ledger doesn't know
        about; it never gates admission.
        Deduplicates by caller name (one load per caller). Enforces fairness via
        monotonic sequence numbers on pending requests (minimum waits). Polls with
        exponential backoff (1s→10s) until timeout or granted. Persists ledger on
        grant. Raises 504 on timeout.
    """
    async with _lock:
        existing_id = _caller_to_lease.get(req.caller)
        if existing_id is not None and existing_id in _leases:
            logger.info("acquire deduped for caller=%s -> existing lease %s", req.caller, existing_id)
            return {"granted": True, "lease_id": existing_id}

    seq = next(_pending_seq)
    async with _lock:
        _pending.add(seq)

    logger.info("acquire received: caller=%s mb=%.0f timeout_s=%.0f", req.caller, req.mb, req.timeout_s)
    start = time.monotonic()
    deadline = start + req.timeout_s
    backoff = ACQUIRE_POLL_MIN_SECONDS
    budget_mb = _gpu_total_mb - SAFETY_MARGIN_MB

    try:
        while True:
            async with _lock:
                is_my_turn = not _pending or seq == min(_pending)
                if is_my_turn:
                    reserved_total = _reserved_mb() + _ollama_mb
                    used_mb = await _live_used_mb()
                    expected_mb = SAFETY_MARGIN_MB + reserved_total
                    if used_mb > expected_mb:
                        logger.warning(
                            "live nvidia-smi usage (%.0f MB) exceeds baseline + tracked "
                            "leases (%.0f MB = %.0f MB margin + %.0f MB reserved); "
                            "something is consuming VRAM the ledger doesn't know about",
                            used_mb, expected_mb, SAFETY_MARGIN_MB, reserved_total,
                        )
                    if reserved_total + req.mb <= budget_mb:
                        lease_id = str(uuid.uuid4())
                        now = time.monotonic()
                        _leases[lease_id] = {
                            "caller": req.caller, "mb": req.mb,
                            "acquired_at": now, "last_seen": now,
                        }
                        _caller_to_lease[req.caller] = lease_id
                        await _persist_ledger()
                        _pending.discard(seq)
                        waited = now - start
                        logger.info(
                            "acquire granted: caller=%s lease=%s mb=%.0f waited=%.1fs",
                            req.caller, lease_id, req.mb, waited,
                        )
                        return {"granted": True, "lease_id": lease_id}

            if time.monotonic() >= deadline:
                async with _lock:
                    _pending.discard(seq)
                logger.warning(
                    "acquire timed out: caller=%s mb=%.0f after %.0fs, no room freed",
                    req.caller, req.mb, req.timeout_s,
                )
                raise HTTPException(status_code=504, detail="timeout waiting for GPU headroom")

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, ACQUIRE_POLL_MAX_SECONDS)
    except HTTPException:
        # Already logged as a WARNING above (expected timeout, not a bug) --
        # let it propagate without also logging it as an unhandled error.
        async with _lock:
            _pending.discard(seq)
        raise
    except BaseException:
        logger.exception("acquire: unexpected error for caller=%s", req.caller)
        async with _lock:
            _pending.discard(seq)
        raise


async def release(req: ReleaseRequest):
    """
    Input:      req - ReleaseRequest with lease_id
    Output:     dict - {"released": True} or False if lease not found
    Details:
        POST /release endpoint. Removes lease from ledger and caller index.
        Logs warnings on unknown/already-released lease_id. Persists ledger
        on successful removal. Returns False gracefully on not-found (client
        may retry on transient broker loss, but should not error).
    """
    async with _lock:
        lease = _leases.pop(req.lease_id, None)
        if lease is None:
            logger.warning("release for unknown/already-released lease_id=%s", req.lease_id)
            return {"released": False}
        if _caller_to_lease.get(lease["caller"]) == req.lease_id:
            del _caller_to_lease[lease["caller"]]
        await _persist_ledger()
        logger.info("release: caller=%s lease=%s mb=%.0f", lease["caller"], req.lease_id, lease["mb"])
        return {"released": True}


async def heartbeat(req: HeartbeatRequest):
    """
    Input:      req - HeartbeatRequest with lease_id
    Output:     dict - {"ok": True} or False if lease not found
    Details:
        POST /heartbeat endpoint. Updates last_seen timestamp for a lease to
        prevent expiry. Called periodically by clients during GPU hold. Logs
        debug on every heartbeat. Returns False gracefully on unknown lease_id.
    """
    async with _lock:
        lease = _leases.get(req.lease_id)
        if lease is None:
            logger.warning("heartbeat for unknown lease_id=%s", req.lease_id)
            return {"ok": False}
        lease["last_seen"] = time.monotonic()
        logger.debug("heartbeat: caller=%s lease=%s", lease["caller"], req.lease_id)
        return {"ok": True}


# Wire up endpoints to FastAPI app

# Wire up endpoints and startup via lifespan
@asynccontextmanager
async def lifespan(app_ctx: FastAPI):
    """
    Input:      app_ctx - FastAPI application instance
    Output:     None
    Details:
        Lifespan context manager for FastAPI startup/shutdown.
        Runs startup tasks before yielding, and cleanup on exit.
    """
    # Startup
    await startup()
    yield
    # Shutdown (none needed)


app = FastAPI(lifespan=lifespan)
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/status", status, methods=["GET"])
app.add_api_route("/acquire", acquire, methods=["POST"])
app.add_api_route("/release", release, methods=["POST"])
app.add_api_route("/heartbeat", heartbeat, methods=["POST"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
