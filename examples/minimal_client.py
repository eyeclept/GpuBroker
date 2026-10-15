"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Minimal synchronous client integration with GpuBroker.

    This is the smallest correct pattern: acquire a lease, do work in a
    try/finally block so the lease is released even if the work raises.
    GPU work runs only with a granted lease. A refused or failed acquire
    logs the reason and exits non-zero. A client that dies holding a
    lease is the most common failure mode in production.

    For lazy loading, background heartbeat, and idle eviction, see
    long_running_service.py in the same folder.
"""
# Imports
import logging
import os
import sys
import time

import httpx

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://localhost:8792")
CALLER_NAME = "minimal_example"
ACQUIRE_MB = 2048


# Functions
def _retry_after(resp: httpx.Response) -> str | None:
    """
    Input:      resp -- broker HTTP response
    Output:     Retry-After header value, or None if absent
    Details:    The broker sends delta-seconds, never an HTTP-date.
    """
    return resp.headers.get("Retry-After")


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


def acquire_lease(mb: int) -> str | None:
    """
    Input:      mb -- VRAM to request in MB
    Output:     lease_id string if granted, None on refusal or error
    Details:
        Request a VRAM lease from the broker. On 200, returns the
        lease_id and logs the one-shot warning if present. On 504, 503,
        422, or any other failure, logs the reason and returns None.
        The caller must not start GPU work without a lease.
    """
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{BROKER_URL}/acquire",
                json={"caller": CALLER_NAME, "mb": mb, "timeout_s": 30.0},
                timeout=45.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            lease_id = data.get("lease_id")
            if not lease_id:
                logger.warning("broker acquire 200 missing lease_id: %s", resp.text)
                return None
            warning = data.get("warning")
            if warning:
                logger.warning("broker acquire warning: %s", warning)
            logger.info("Lease acquired: %s for %d MB", lease_id, mb)
            return lease_id
        if resp.status_code == 504:
            logger.warning(
                "broker denied (504, no room) Retry-After=%s: %s",
                _retry_after(resp), resp.text,
            )
            return None
        if resp.status_code == 503:
            logger.warning(
                "broker busy (503) detail=%s Retry-After=%s: %s",
                _response_detail(resp), _retry_after(resp), resp.text,
            )
            return None
        if resp.status_code == 422:
            logger.warning("broker rejected request (422): %s", resp.text)
            return None
        logger.warning("broker returned %d: %s", resp.status_code, resp.text)
        return None
    except Exception:
        logger.exception("Failed to acquire lease")
        return None


def release_lease(lease_id: str | None) -> None:
    """
    Input:      lease_id -- lease to release, or None to skip
    Output:     None
    Details:
        Release a held VRAM lease. Idempotent; logs errors but never raises.
        Safe to call from a finally block or exception handler unconditionally.
    """
    if not lease_id:
        return
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{BROKER_URL}/release",
                json={"lease_id": lease_id},
                timeout=5.0,
            )
        if resp.status_code == 200:
            logger.info("Lease released: %s", lease_id)
        else:
            logger.warning("Release returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("Failed to release lease: %s", exc)


def main() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Main entry point. Acquires a lease, does work only if granted,
        then releases in finally. Exits non-zero when acquire fails.
    """
    logging.basicConfig(level=logging.INFO)

    lease_id = acquire_lease(ACQUIRE_MB)
    if not lease_id:
        logger.error("no lease; not starting GPU work")
        sys.exit(1)

    try:
        # === DO THE GPU WORK HERE ===
        # In a real service, this is where you would load a model and
        # run inference. This placeholder just sleeps.
        logger.info("Work started with lease %s", lease_id)
        time.sleep(1)
        logger.info("Work completed")
        # === END GPU WORK ===
    finally:
        # Release always runs, whether work succeeded or raised.
        # This is the pattern that avoids leaking leases.
        release_lease(lease_id)


if __name__ == "__main__":
    main()
