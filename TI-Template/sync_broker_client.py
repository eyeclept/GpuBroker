"""
Author: [Your Name]
Date:   [Date]
Email:  [Your Email]

Description:
    Sync broker client for a tightly-integrated GPU service.
    Provides acquire, release, and heartbeat functions that coordinate
    VRAM leases with GpuBroker.

    This file is a copy-paste template. Edit CALLER_NAME below and
    wire the three broker functions into your load, evict, and idle
    lifecycle. The caller stores lease_id and passes it to release
    and heartbeat. This module holds no lease global.

    Use this variant when your service's request-handling code is
    already synchronous. Do not mix this sync client into an otherwise
    fully async request path.

    Behavior matches async_broker_client.py except for sync I/O.

"""
# Imports
import logging
import os
import time
from typing import NamedTuple

import httpx

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://host.containers.internal:8792")
BROKER_FAIL_OPEN = os.environ.get("BROKER_FAIL_OPEN", "false").lower() in ("true", "1", "yes")

CALLER_NAME = "REPLACE_ME"  # TODO(integrator): replace with your service's unique name


class AcquireResult(NamedTuple):
    """Granted lease. warning is the one-shot watchdog notice, or None."""

    lease_id: str
    warning: str | None = None


class HeartbeatResult(NamedTuple):
    """Heartbeat outcome. ok is True, False (lease gone), or None (unknown)."""

    ok: bool | None
    preempt: bool = False
    reason: str = ""


class BrokerError(Exception):
    """Base class for broker client errors."""


class BrokerNoRoom(BrokerError):
    """Broker 504: no headroom within timeout_s."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BrokerBusy(BrokerError):
    """Broker 503: unmanaged_gpu_load or too_many_leases. Never fail open."""

    def __init__(
        self,
        message: str,
        detail: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.detail = detail
        self.retry_after = retry_after


class BrokerRejected(BrokerError):
    """Broker 400 or 422: invalid request. Do not retry."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class BrokerUnavailable(BrokerError):
    """Transport failure after retries, fail closed."""


# Functions
def _retry_after(resp: httpx.Response) -> float | None:
    """
    Input:      resp -- broker HTTP response
    Output:     Retry-After in seconds, or None if absent or not a number
    Details:    The broker sends delta-seconds, never an HTTP-date.
    """
    value = resp.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        logger.warning("ignoring non-numeric Retry-After from broker: %r", value)
        return None


def _response_detail(resp: httpx.Response) -> str:
    """
    Input:      resp -- broker HTTP response
    Output:     FastAPI detail string, or empty string if missing
    Details:    400/422/503 bodies are {"detail": "..."}.
    """
    try:
        payload = resp.json()
    except Exception:
        logger.warning("broker error body was not JSON: %s", resp.text)
        return ""
    if isinstance(payload, dict):
        detail = payload.get("detail", "")
        if isinstance(detail, str):
            return detail
        return str(detail)
    return ""


def _broker_acquire(mb: int) -> AcquireResult | None:
    """
    Input:      mb -- VRAM to request in MB
    Output:     AcquireResult if granted, None if fail-open after
                transport retries
    Details:
        Acquire a VRAM lease from the broker. Retries transport errors
        with backoff (0.5s, then double, cap 5s, 5 attempts). Explicit
        broker answers do not retry and never fail open: 504 raises
        BrokerNoRoom, 503 raises BrokerBusy, 400/422 raise
        BrokerRejected. After transport retries: returns None if
        BROKER_FAIL_OPEN else raises BrokerUnavailable.
    """
    backoff_seconds = 0.5
    max_backoff = 5.0
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(
                    f"{BROKER_URL}/acquire",
                    json={"caller": CALLER_NAME, "mb": mb, "timeout_s": 30.0},
                )
        except httpx.TransportError as exc:
            logger.warning(
                "broker transport error (attempt %d/%d): %s",
                attempt, max_attempts, exc,
            )
            if attempt < max_attempts:
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)
                continue
            if BROKER_FAIL_OPEN:
                logger.warning(
                    "broker unreachable after %d attempts, failing open",
                    max_attempts,
                )
                return None
            logger.error(
                "broker unreachable after %d attempts, failing closed",
                max_attempts,
            )
            raise BrokerUnavailable("broker unreachable") from exc
        except Exception:
            logger.exception("unexpected error in broker acquire")
            raise BrokerUnavailable("broker error")

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                logger.exception("broker acquire 200 body was not JSON")
                raise BrokerUnavailable("broker acquire returned invalid JSON")
            if not isinstance(data, dict):
                logger.error("broker acquire 200 body was not an object: %s", resp.text)
                raise BrokerUnavailable("broker acquire returned invalid JSON")
            lease_id = data.get("lease_id")
            if not lease_id:
                logger.error("broker acquire 200 missing lease_id: %s", resp.text)
                raise BrokerUnavailable("broker acquire returned no lease_id")
            warning = data.get("warning") or None
            if warning:
                logger.warning("broker acquire warning: %s", warning)
            logger.info("broker lease acquired: %s (mb=%d)", lease_id, mb)
            return AcquireResult(lease_id=str(lease_id), warning=warning)

        if resp.status_code == 504:
            retry_after = _retry_after(resp)
            logger.error("broker denied (no room): %s", resp.text)
            raise BrokerNoRoom("GPU memory full", retry_after)

        if resp.status_code == 503:
            detail = _response_detail(resp)
            retry_after = _retry_after(resp)
            logger.warning(
                "broker busy detail=%s retry_after=%s: %s",
                detail, retry_after, resp.text,
            )
            raise BrokerBusy(
                f"broker busy: {detail or resp.text}",
                detail=detail,
                retry_after=retry_after,
            )

        if resp.status_code in (400, 422):
            logger.error(
                "broker rejected request (%d): %s",
                resp.status_code, resp.text,
            )
            raise BrokerRejected(
                f"broker rejected request ({resp.status_code}): {resp.text}",
                status_code=resp.status_code,
            )

        logger.warning(
            "broker replied %d (attempt %d/%d): %s",
            resp.status_code, attempt, max_attempts, resp.text,
        )
        if attempt < max_attempts:
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, max_backoff)
            continue
        logger.error("broker failed after %d attempts, failing closed", max_attempts)
        raise BrokerUnavailable("broker request failed")

    logger.warning("broker acquire: unexpected end of retry loop")
    raise BrokerUnavailable("broker acquire timeout")


def _broker_release(lease_id: str | None) -> None:
    """
    Input:      lease_id -- lease to release, or None to skip
    Output:     None
    Details:
        Release a held VRAM lease. Call AFTER GPU memory is actually
        freed (e.g. after torch.cuda.empty_cache() / equivalent), not
        merely once eviction is decided. Idempotent. Logs result but
        never raises.
    """
    if not lease_id:
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{BROKER_URL}/release", json={"lease_id": lease_id},
            )
        if resp.status_code == 200:
            logger.info("broker lease released: %s", lease_id)
        else:
            logger.warning(
                "broker release returned %d: %s", resp.status_code, resp.text,
            )
    except Exception:
        logger.exception("broker release failed (continuing anyway)")


def _broker_heartbeat(lease_id: str | None) -> HeartbeatResult:
    """
    Input:      lease_id -- lease to heartbeat, or None to skip
    Output:     HeartbeatResult
    Details:
        Send a heartbeat to keep the lease alive. Call on a periodic
        tick (e.g. every 20-30s) whenever a lease is held. The endpoint
        always answers 200. A lost lease is {"ok": false}. A live lease
        is {"ok": true, "preempt": bool, "reason": str}. Transport
        errors return ok=None (unknown); do not treat that as lost.
        preempt=True means unload and release before the 30s grace.
        Never raises.
    """
    if not lease_id:
        return HeartbeatResult(ok=True)
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{BROKER_URL}/heartbeat", json={"lease_id": lease_id},
            )
        if resp.status_code != 200:
            logger.warning(
                "broker heartbeat returned %d: %s", resp.status_code, resp.text,
            )
            return HeartbeatResult(ok=None)
        try:
            data = resp.json()
        except Exception:
            logger.warning(
                "broker heartbeat body was not JSON (lease %s): %s",
                lease_id, resp.text,
            )
            return HeartbeatResult(ok=None)
        if not isinstance(data, dict):
            logger.warning(
                "broker heartbeat body was not an object (lease %s): %s",
                lease_id, resp.text,
            )
            return HeartbeatResult(ok=None)
        if not data.get("ok", False):
            logger.warning(
                "broker heartbeat returned ok=false (lease is gone): %s",
                lease_id,
            )
            return HeartbeatResult(
                ok=False,
                preempt=bool(data.get("preempt", False)),
                reason=data.get("reason") or "",
            )
        preempt = bool(data.get("preempt", False))
        reason = data.get("reason") or ""
        if preempt:
            logger.warning(
                "broker requested preemption for lease %s: %s",
                lease_id, reason,
            )
        return HeartbeatResult(ok=True, preempt=preempt, reason=reason)
    except Exception as exc:
        logger.warning(
            "broker heartbeat failed (lease %s): %s (will retry next tick)",
            lease_id, exc,
        )
        return HeartbeatResult(ok=None)
