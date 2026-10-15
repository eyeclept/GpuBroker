"""
Author: FreeTokenBuild
Date:   2026
Email:  eyeclept@pm.me

Description:
    Sync broker client for FreeToken's GpuBroker integration. Adapted from
    ../TI-Template/sync_broker_client.py.

    Deviation from the template: the template's `_broker_acquire` raises
    FastAPI's `HTTPException` because its reference callers (Needle,
    Flux2Klein9B) call it straight from request handlers. FreeToken's
    wrapper (serve_wrapper.py) calls it from a single-flight model load
    shared by several waiting requests, through asyncio.to_thread. It
    raises `BrokerAcquireError` instead; the wrapper turns that into a 503
    with Retry-After for every waiting request and does not start
    `ft serve`. Retry counts, backoff timings, and the fail-open/fail-closed
    decision points are otherwise unchanged from the template.
"""
# Imports
import logging
import os
import threading
import time

import httpx

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://host.containers.internal:8792")
BROKER_FAIL_OPEN = os.environ.get("BROKER_FAIL_OPEN", "false").lower() in ("true", "1", "yes")

CALLER_NAME = "freetoken"

# Must match server.py's UNMANAGED_BUSY_DETAIL.
UNMANAGED_BUSY_DETAIL = "unmanaged_gpu_load"

_client: httpx.Client | None = None
_client_lock = threading.Lock()


class BrokerAcquireError(RuntimeError):
    """Raised by _broker_acquire on denial or unrecoverable error when
    BROKER_FAIL_OPEN is false. The wrapper answers 503 and loads nothing.
    retry_after carries the broker's Retry-After header in seconds, if any."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BrokerNoRoom(BrokerAcquireError):
    """Broker 504: no headroom within timeout_s. The caller may back off
    and retry; the broker keeps its fairness wait age across retries."""


# Functions
def _retry_after(resp: httpx.Response) -> float | None:
    """
    Input:      resp -- broker response
    Output:     Retry-After in seconds, or None if absent or not a number
    Details:    The broker only sends delta-seconds, never an HTTP date.
    """
    value = resp.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        logger.warning("ignoring non-numeric Retry-After from broker: %r", value)
        return None


def _get_client() -> httpx.Client:
    """
    Input:      None
    Output:     the process-wide broker HTTP client
    Details:
        Creates the client on first use. httpx.Client is thread-safe for
        concurrent requests. Callers pass per-call timeouts.
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client()
        return _client


def close_client() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Closes the shared broker HTTP client. Idempotent. The next
        broker call creates a new client.
    """
    global _client
    with _client_lock:
        client, _client = _client, None
    if client is None:
        return
    try:
        client.close()
    except Exception as exc:
        logger.warning("broker HTTP client close failed: %s", exc)


def _broker_acquire(mb: int, pid: int | None = None) -> str | None:
    """
    Input:      mb -- VRAM to request in MB
                pid -- host PID of the GPU process holding this lease
                    (`ft serve` child), or None if it is not started yet
    Output:     lease_id (str) if acquired, None if fail-open
    Details:
        Acquire VRAM lease from broker. Always declares preemptible=True
        so the broker's fairness preemption can flag this holder. Retries
        on connection error with backoff (0.5s -> double -> cap 5s, max 5
        attempts). On broker unreachable: logs WARNING, returns None if
        BROKER_FAIL_OPEN else raises BrokerAcquireError. On broker denial
        (504): raises BrokerNoRoom immediately with the broker's
        Retry-After; the caller decides whether to back off and retry.
        On a 503 carrying UNMANAGED_BUSY_DETAIL (game running): raises
        BrokerAcquireError with the broker's Retry-After. Neither path
        fails open.
    """
    backoff = 0.5
    max_backoff = 5.0
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_client()
            resp = client.post(
                f"{BROKER_URL}/acquire",
                json={
                    "caller": CALLER_NAME,
                    "mb": mb,
                    "timeout_s": 30.0,
                    "preemptible": True,
                    "pid": pid,
                },
                timeout=45.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                lease_id = data.get("lease_id")
                logger.info("broker lease acquired: %s", lease_id)
                return lease_id
            elif resp.status_code == 504:
                logger.warning("broker denied (no room): %s", resp.text)
                raise BrokerNoRoom("GPU memory full", _retry_after(resp))
            elif resp.status_code == 503 and UNMANAGED_BUSY_DETAIL in resp.text:
                logger.warning("broker refused: unmanaged GPU load (game) is active")
                raise BrokerAcquireError("yielding to unmanaged GPU load", _retry_after(resp))
            else:
                logger.warning(
                    "broker replied %d (attempt %d/%d): %s",
                    resp.status_code, attempt, max_attempts, resp.text,
                )
                if attempt < max_attempts:
                    time.sleep(min(backoff, max_backoff))
                    backoff *= 2
                    continue
                if BROKER_FAIL_OPEN:
                    logger.warning("broker failed after %d attempts, failing open", max_attempts)
                    return None
                logger.error("broker failed after %d attempts, failing closed", max_attempts)
                raise BrokerAcquireError("broker request failed")
        except httpx.ConnectError as exc:
            logger.warning("broker unreachable (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(min(backoff, max_backoff))
                backoff *= 2
                continue
            if BROKER_FAIL_OPEN:
                logger.warning("broker unreachable after %d attempts, failing open", max_attempts)
                return None
            logger.error("broker unreachable after %d attempts, failing closed", max_attempts)
            raise BrokerAcquireError("broker unreachable")
        except httpx.TimeoutException as exc:
            logger.warning("broker request timed out waiting for a response (mb=%d): %s", mb, exc)
            if BROKER_FAIL_OPEN:
                logger.warning("broker timed out, failing open")
                return None
            logger.error("broker timed out, failing closed")
            raise BrokerAcquireError("broker request timed out")
        except BrokerAcquireError:
            raise
        except Exception:
            logger.exception("unexpected broker error (attempt %d/%d)", attempt, max_attempts)
            if attempt < max_attempts:
                time.sleep(min(backoff, max_backoff))
                backoff *= 2
                continue
            if BROKER_FAIL_OPEN:
                return None
            raise BrokerAcquireError("broker error")
    logger.warning("broker acquire: unexpected end of retry loop")
    if BROKER_FAIL_OPEN:
        return None
    raise BrokerAcquireError("broker acquire timeout")


def _broker_release(lease_id: str | None) -> None:
    """
    Input:      lease_id -- lease to release, or None to skip
    Output:     None
    Details:    Release a held VRAM lease. Idempotent; logs but never raises.
    """
    if not lease_id:
        return
    try:
        client = _get_client()
        resp = client.post(f"{BROKER_URL}/release", json={"lease_id": lease_id}, timeout=5.0)
        if resp.status_code == 200:
            logger.info("broker lease released: %s", lease_id)
        else:
            logger.warning("broker release returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("broker release error (lease %s): %s", lease_id, exc)


def _broker_heartbeat(lease_id: str | None) -> tuple[bool, bool]:
    """
    Input:      lease_id -- lease to heartbeat, or None to skip
    Output:     (ok, preempt) -- ok is True if the broker confirmed this
                lease is still alive (or there was nothing to heartbeat).
                preempt is True when the broker has flagged this lease
                for fairness preemption; the caller must self-evict and
                must not re-acquire in place.
    Details:
        Send a heartbeat to keep a lease alive. Logs on error, never
        raises. Callers must treat ok=False as a signal to call
        _broker_acquire again and start heartbeating the new lease_id --
        the broker's sweeper has already reclaimed the old one.
    """
    if not lease_id:
        return True, False
    try:
        client = _get_client()
        resp = client.post(
            f"{BROKER_URL}/heartbeat",
            json={"lease_id": lease_id},
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning("broker heartbeat returned %d: %s", resp.status_code, resp.text)
            return False, False
        data = resp.json()
        if not data.get("ok", False):
            logger.warning("broker heartbeat returned ok=false (lease may be lost): %s", lease_id)
            return False, False
        preempt = bool(data.get("preempt", False))
        if preempt:
            logger.warning(
                "broker requested preemption for lease %s: %s",
                lease_id, data.get("reason", ""),
            )
        return True, preempt
    except Exception as exc:
        logger.warning("broker heartbeat error (lease %s): %s", lease_id, exc)
        return False, False
