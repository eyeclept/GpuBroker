"""
Author: [Your Name]
Date:   [Date]
Email:  [Your Email]

Description:
    Sync broker client for tightly-integrated GPU service.
    Provides acquire/release/heartbeat functions to coordinate VRAM
    leases with GpuBroker's VRAM lease API.
    
    This file is a copy-paste template: edit CALLER_NAME below and
    wire the three broker functions into your own load/evict/idle
    lifecycle (see TODO comments for wiring points).
    
    Use this variant when your service's request-handling code is
    already synchronous/blocking. Do not mix this sync client into
    an otherwise fully async request path.

"""
# Imports
import logging
import os
import time

import httpx
from fastapi import HTTPException

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://host.containers.internal:8792")
BROKER_FAIL_OPEN = os.environ.get("BROKER_FAIL_OPEN", "false").lower() in ("true", "1", "yes")

CALLER_NAME = "REPLACE_ME"  # TODO(integrator): replace with your service's unique name


# Functions
def _broker_acquire(mb: int) -> str | None:
    """
    Input:      mb -- VRAM to request in MB
    Output:     lease_id (str) if acquired, None if fail-open
    Details:
        Acquire VRAM lease from broker. Retries on connection error with
        backoff (0.5s -> double -> cap 5s, max 5 attempts). On broker
        unreachable: logs WARNING, returns None if BROKER_FAIL_OPEN else
        raises. On broker denial (504): raises HTTPException(503, ...)
        immediately -- does not retry, does not fail open.
    """
    backoff = 0.5
    max_backoff = 5.0
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
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
                logger.info("broker lease acquired: %s", lease_id)
                return lease_id
            elif resp.status_code == 504:
                logger.error("broker denied (no room): %s", resp.text)
                raise HTTPException(status_code=503, detail="GPU memory full")
            else:
                logger.warning("broker replied %d (attempt %d/%d): %s", resp.status_code, attempt, max_attempts, resp.text)
                if attempt < max_attempts:
                    time.sleep(min(backoff, max_backoff))
                    backoff *= 2
                    continue
                if BROKER_FAIL_OPEN:
                    logger.warning("broker failed after %d attempts, failing open", max_attempts)
                    return None
                logger.error("broker failed after %d attempts, failing closed", max_attempts)
                raise HTTPException(status_code=503, detail="broker request failed")
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
            raise HTTPException(status_code=503, detail="broker unreachable")
        except httpx.TimeoutException as exc:
            logger.warning("broker request timed out waiting for a response (mb=%d): %s", mb, exc)
            if BROKER_FAIL_OPEN:
                logger.warning("broker timed out, failing open")
                return None
            logger.error("broker timed out, failing closed")
            raise HTTPException(status_code=503, detail="broker request timed out")
        except HTTPException:
            raise
        except Exception:
            logger.exception("unexpected broker error (attempt %d/%d)", attempt, max_attempts)
            if attempt < max_attempts:
                time.sleep(min(backoff, max_backoff))
                backoff *= 2
                continue
            if BROKER_FAIL_OPEN:
                return None
            raise HTTPException(status_code=503, detail="broker error")
    logger.warning("broker acquire: unexpected end of retry loop")
    if BROKER_FAIL_OPEN:
        return None
    raise HTTPException(status_code=503, detail="broker acquire timeout")


def _broker_release(lease_id: str | None) -> None:
    """
    Input:      lease_id -- lease to release, or None to skip
    Output:     None
    Details:    Release a held VRAM lease. Idempotent; logs but never raises.
    """
    if not lease_id:
        return
    try:
        client = httpx.Client()
        resp = client.post(f"{BROKER_URL}/release", json={"lease_id": lease_id}, timeout=5.0)
        if resp.status_code == 200:
            logger.info("broker lease released: %s", lease_id)
        else:
            logger.warning("broker release returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("broker release error (lease %s): %s", lease_id, exc)


def _broker_heartbeat(lease_id: str | None) -> None:
    """
    Input:      lease_id -- lease to heartbeat, or None to skip
    Output:     None
    Details:    Send a heartbeat to keep a lease alive. Logs on error, never raises.
    """
    if not lease_id:
        return
    try:
        client = httpx.Client()
        resp = client.post(f"{BROKER_URL}/heartbeat", json={"lease_id": lease_id}, timeout=5.0)
        if resp.status_code != 200:
            logger.warning("broker heartbeat returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("broker heartbeat error (lease %s): %s", lease_id, exc)
