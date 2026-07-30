"""
Author: [Your Name]
Date:   [Date]
Email:  [Your Email]

Description:
    Async broker client for tightly-integrated GPU service.
    Provides acquire/release/heartbeat functions to coordinate VRAM
    leases with GpuBroker's VRAM lease API.
    
    This file is a copy-paste template: edit CALLER_NAME below and
    wire the three broker functions into your own load/evict/idle
    lifecycle (see TODO comments for wiring points).

"""
# Imports
import asyncio
import logging
import os

import httpx
from fastapi import HTTPException

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://host.containers.internal:8792")
BROKER_FAIL_OPEN = os.environ.get("BROKER_FAIL_OPEN", "true").lower() in ("true", "1", "yes")

CALLER_NAME = "REPLACE_ME"  # TODO(integrator): replace with your service's unique name

lease_id = None  # holds the current broker lease ID if acquired


# Functions
async def _broker_acquire(mb: int) -> str | None:
    """
    Input:       mb -- VRAM to request in MB
    Output:      lease_id string if acquired, None if fail-open
    Details:
        Acquire VRAM lease from broker. Retries on connection error with
        backoff (0.5s -> double -> cap 5s, 3-5 attempts). On broker
        unreachable: logs WARNING, returns None if BROKER_FAIL_OPEN else
        raises. On broker denial (504): raises HTTPException(503, ...)
        immediately -- does not retry, does not fail open.
    """
    backoff_seconds = 0.5
    max_backoff = 5.0
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        attempts += 1
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{BROKER_URL}/acquire",
                    json={"caller": CALLER_NAME, "mb": mb, "timeout_s": 30.0},
                )
                if resp.status_code == 504:
                    logger.error("broker denied VRAM lease (no room): %s", resp.text)
                    raise HTTPException(status_code=503, detail="GPU memory full")
                resp.raise_for_status()
                data = resp.json()
                lease_id = data.get("lease_id")
                logger.info("acquired broker lease: %s (mb=%d)", lease_id, mb)
                return lease_id
        except httpx.ConnectError:
            logger.exception("broker unreachable")
            if attempts < max_attempts:
                logger.warning("broker connection failed (attempt %d/%d), retrying in %.1fs",
                             attempts, max_attempts, backoff_seconds)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)
                continue
            if BROKER_FAIL_OPEN:
                logger.warning("broker unreachable after %d attempts, failing open", max_attempts)
                return None
            logger.error("broker unreachable after %d attempts, failing closed", max_attempts)
            raise HTTPException(status_code=503, detail="broker unreachable")
        except httpx.TimeoutException as exc:
            # The broker's own /acquire already long-polls internally for up to
            # its timeout_s (default 300s) while queueing for GPU room -- a
            # client-side timeout here means that full wait elapsed without a
            # response, not a fast/transient failure. Resolve once, immediately.
            logger.warning("broker request timed out waiting for a response (mb=%d): %s", mb, exc)
            if BROKER_FAIL_OPEN:
                logger.warning("broker timed out, failing open")
                return None
            logger.error("broker timed out, failing closed")
            raise HTTPException(status_code=503, detail="broker request timed out")
        except httpx.HTTPError:
            logger.exception("broker request failed")
            if attempts < max_attempts:
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)
                continue
            if BROKER_FAIL_OPEN:
                logger.warning("broker failed after %d attempts, failing open", max_attempts)
                return None
            logger.error("broker failed after %d attempts, failing closed", max_attempts)
            raise HTTPException(status_code=503, detail="broker request failed")
        except HTTPException:
            raise
        except Exception:
            logger.exception("unexpected error in broker acquire")
            raise HTTPException(status_code=503, detail="broker error")
    logger.warning("broker acquire: unexpected end of retry loop")
    if BROKER_FAIL_OPEN:
        return None
    raise HTTPException(status_code=503, detail="broker acquire timeout")


async def _broker_release() -> None:
    """
    Input:       None (uses module-level lease_id)
    Output:      None
    Details:
        Release current broker lease. Call AFTER GPU memory is actually
        freed (e.g. after torch.cuda.empty_cache() / equivalent), not
        merely once eviction is decided. Logs result but never raises.
    """
    global lease_id
    if lease_id is None:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{BROKER_URL}/release", json={"lease_id": lease_id})
            resp.raise_for_status()
            logger.info("released broker lease: %s", lease_id)
    except Exception:
        logger.exception("broker release failed (continuing anyway)")
    finally:
        lease_id = None


async def _broker_heartbeat() -> None:
    """
    Input:       None (uses module-level lease_id)
    Output:      None
    Details:
        Send heartbeat to keep the lease alive. Call on a periodic tick
        (e.g. every 20-30s) whenever a lease is held. Logs failures but
        never raises -- a missed heartbeat is not fatal, only risks the
        broker's sweeper reclaiming the lease if it keeps failing.
    """
    if lease_id is None:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{BROKER_URL}/heartbeat", json={"lease_id": lease_id})
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("broker heartbeat failed: %s (will retry next tick)", exc)
