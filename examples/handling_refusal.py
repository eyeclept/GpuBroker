"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Shows how a client should respond when /acquire returns 422, 503,
    or 504. A 422 whose detail says the request exceeds capacity is
    permanent. Raise NeverFitsError. A 503 with detail
    unmanaged_gpu_load or too_many_leases is transient: sleep
    max(Retry-After, current backoff) inside the retry budget. A 504
    means the broker already waited up to timeout_s. Honor Retry-After
    (sleep at least that). Check /status to classify a 504: request MB
    exceeds budget_mb (never fits) or the request would fit if current
    leases and Ollama unloaded (transient). Retry transient refusals
    with exponential backoff for up to six hours. A preemptible holder
    such as FreeToken can take minutes to hours to be evicted or idle
    out. Log the one-shot acquire warning if present.
"""
# Imports
import logging
import os
import time

import httpx

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://localhost:8792")
RETRY_INITIAL_DELAY_S = 5.0
RETRY_MAX_DELAY_S = 900.0
TRANSIENT_503_DETAILS = ("unmanaged_gpu_load", "too_many_leases")


# Custom exception for refusal scenarios
class BrokerRefusalError(Exception):
    """
    Input:      message (str)
    Output:     None (exception class)
    Details:
        Raised when the broker refuses an /acquire request and retry
        will not succeed (the request exceeds budget, or the bounded
        retry limit is exhausted).
    """
    pass


class NeverFitsError(BrokerRefusalError):
    """
    Input:      message (str)
    Output:     None (exception class)
    Details:
        Raised when the broker returns 422 because the request exceeds
        capacity, or when /status shows the requested MB exceeds
        budget_mb after a 504. The request will never fit under the
        current broker configuration.
    """
    pass


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


def _get_broker_status() -> dict:
    """
    Input:      None
    Output:     dict with keys budget_mb, reserved_mb, ollama_mb, gpu_total_mb
    Details:
        Fetch current GPU budget and allocation state from the broker.
        Used to determine whether a 504 refusal is permanent or transient.
    """
    try:
        resp = httpx.get(f"{BROKER_URL}/status", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("failed to fetch broker status")
        raise


def _retry_wait(
    backoff: float,
    retry_after: float | None,
    remaining: float,
    cap_backoff: bool,
) -> float:
    """
    Input:      backoff -- current exponential backoff
                retry_after -- broker Retry-After, or None
                remaining -- seconds left in the retry budget
                cap_backoff -- if True, cap backoff at RETRY_MAX_DELAY_S
    Output:     seconds to sleep, clipped to remaining
    Details:
        Return max(Retry-After, current backoff), clipped to remaining.
        503 honors a long Retry-After (up to 1800) inside the budget.
        504 still caps the backoff half at RETRY_MAX_DELAY_S, then
        sleeps at least Retry-After.
    """
    delay = min(backoff, RETRY_MAX_DELAY_S) if cap_backoff else backoff
    wait = max(delay, retry_after or 0.0)
    return min(wait, remaining)


def acquire_with_retry(mb: int, retry_budget_s: float = 21600.0) -> str:
    """
    Input:      mb (int) - requested VRAM in MB
                retry_budget_s (float) - total retry budget in seconds
    Output:     lease_id (str) - the acquired lease ID
    Details:
        Attempt to acquire a lease from the broker. A 422 whose detail
        says the request exceeds capacity raises NeverFitsError. A 503
        with unmanaged_gpu_load or too_many_leases is transient: sleep
        max(Retry-After, backoff) inside retry_budget_s. If /acquire
        returns 504, check /status to classify the failure:

        - Never fits: request MB exceeds budget_mb. Raise NeverFitsError.
        - Transient: request would fit under budget but not now due to
          current reserved_mb and ollama_mb. Retry with exponential backoff
          for up to retry_budget_s (default six hours). Sleep at least
          the 504 Retry-After value.

        A transient refusal can clear when a preemptible holder such as
        FreeToken is evicted or idles out, which can take minutes to hours.
        Use a long retry budget for that wait. The broker already waits
        internally for up to timeout_s before returning 504. Retry after
        a pause because another lease may be released or Ollama unloaded.
        Log the one-shot warning on a 200 grant.
    """
    started_at = time.monotonic()
    deadline = started_at + retry_budget_s
    failure_count = 0
    backoff = RETRY_INITIAL_DELAY_S

    while True:
        if failure_count and time.monotonic() >= deadline:
            elapsed = time.monotonic() - started_at
            msg = (
                "transient refusal: gave up after "
                f"{failure_count} attempts and {elapsed:.1f}s"
            )
            logger.error(msg)
            raise BrokerRefusalError(msg)
        try:
            # Issue acquire. The broker waits up to timeout_s (this call
            # sets 30s; the broker default is 300s) before returning 504
            # if space is not available.
            resp = httpx.post(
                f"{BROKER_URL}/acquire",
                json={"caller": "refusal_example", "mb": mb, "timeout_s": 30.0},
                timeout=45.0,
            )

            if resp.status_code == 200:
                data = resp.json()
                lease_id = data.get("lease_id")
                if not lease_id:
                    logger.error("broker acquire 200 missing lease_id: %s", resp.text)
                    raise BrokerRefusalError("broker acquire returned no lease_id")
                warning = data.get("warning")
                if warning:
                    logger.warning("broker acquire warning: %s", warning)
                logger.info("acquired lease: %s (mb=%d)", lease_id, mb)
                return lease_id

            elif resp.status_code == 422:
                try:
                    detail = resp.json().get("detail", "")
                except Exception as exc:
                    logger.exception("could not parse 422 body")
                    raise BrokerRefusalError("broker error: 422") from exc
                if isinstance(detail, list):
                    detail_text = " ".join(
                        str(item.get("msg", item) if isinstance(item, dict) else item)
                        for item in detail
                    )
                else:
                    detail_text = str(detail)
                if "exceeds" in detail_text.lower() and "capacity" in detail_text.lower():
                    msg = f"request ({mb} MB) exceeds capacity: {detail_text}"
                    logger.error(msg)
                    raise NeverFitsError(msg)
                logger.error("broker returned 422: %s", resp.text)
                raise BrokerRefusalError("broker error: 422")

            elif resp.status_code == 503:
                detail = _response_detail(resp)
                retry_after = _retry_after(resp)
                if detail not in TRANSIENT_503_DETAILS:
                    logger.error("broker returned 503: %s", resp.text)
                    raise BrokerRefusalError(f"broker error: 503 {detail}")
                failure_count += 1
                elapsed = time.monotonic() - started_at
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = (
                        "transient refusal: gave up after "
                        f"{failure_count} attempts and {elapsed:.1f}s"
                    )
                    logger.error(msg)
                    raise BrokerRefusalError(msg)
                wait = _retry_wait(
                    backoff, retry_after, remaining, cap_backoff=False,
                )
                logger.warning(
                    "transient 503 detail=%s Retry-After=%s; retry attempt %d "
                    "after %.1fs (elapsed %.1fs)",
                    detail, retry_after, failure_count, wait, elapsed,
                )
                if wait > 0:
                    time.sleep(wait)
                backoff = min(backoff * 2, RETRY_MAX_DELAY_S)
                continue

            elif resp.status_code == 504:
                # The broker returned 504: no room at the moment. Do not
                # retry immediately. The broker already waited.
                retry_after = _retry_after(resp)

                # Fetch status to classify the failure
                try:
                    status = _get_broker_status()
                except Exception as exc:
                    logger.exception("could not fetch status to classify 504")
                    raise BrokerRefusalError(
                        "broker denied but status fetch failed"
                    ) from exc

                budget_mb = status.get("budget_mb")
                reserved_mb = status.get("reserved_mb", 0)
                ollama_mb = status.get("ollama_mb", 0)

                # Test 1: Does the request exceed the total budget?
                if mb > budget_mb:
                    msg = (
                        f"request ({mb} MB) exceeds budget ({budget_mb} MB); "
                        f"will never fit"
                    )
                    logger.error(msg)
                    raise NeverFitsError(msg)

                # Test 2: Would it fit if current allocations were released?
                if mb <= budget_mb:
                    # The request fits under budget, but not right now due to
                    # reserved leases and/or Ollama. This is transient. Retry
                    # with backoff. Sleep at least Retry-After.
                    failure_count += 1
                    elapsed = time.monotonic() - started_at
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        msg = (
                            "transient refusal: gave up after "
                            f"{failure_count} attempts and {elapsed:.1f}s"
                        )
                        logger.error(msg)
                        raise BrokerRefusalError(msg)

                    wait = _retry_wait(
                        backoff, retry_after, remaining, cap_backoff=True,
                    )
                    logger.warning(
                        "transient refusal (reserved=%.0f ollama=%.0f); retry "
                        "attempt %d after %.1fs (elapsed %.1fs, Retry-After=%s)",
                        reserved_mb, ollama_mb, failure_count, wait, elapsed,
                        retry_after,
                    )
                    if wait > 0:
                        time.sleep(wait)
                    backoff = min(backoff * 2, RETRY_MAX_DELAY_S)
                    continue

            else:
                # Unexpected status code from broker
                logger.error("broker returned %d: %s", resp.status_code, resp.text)
                raise BrokerRefusalError(f"broker error: {resp.status_code}")

        except BrokerRefusalError:
            raise
        except httpx.ConnectError as exc:
            logger.exception("broker unreachable")
            raise BrokerRefusalError("broker unreachable") from exc
        except httpx.TimeoutException as exc:
            logger.exception("broker request timed out")
            raise BrokerRefusalError("broker request timed out") from exc
        except Exception as exc:
            logger.exception("unexpected error acquiring lease")
            raise BrokerRefusalError("unexpected error") from exc

    msg = "acquire retry loop exhausted without decision"
    logger.error(msg)
    raise BrokerRefusalError(msg)


def main() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Demonstrate the handling pattern. This example attempts to acquire
        12000 MB. On a 12 GB card with the default 1536 MB margin that
        exceeds maximum grantable capacity, so the broker returns 422.
        Real clients would wrap this in their workload lifecycle (load
        model, acquire, do work, release).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("attempting to acquire 12000 MB lease")

    try:
        lease_id = acquire_with_retry(mb=12000, retry_budget_s=21600.0)
        logger.info("success: acquired lease %s", lease_id)

        # In a real client, hold the lease, do work, heartbeat periodically,
        # then release. For this example, just release immediately.
        try:
            resp = httpx.post(
                f"{BROKER_URL}/release",
                json={"lease_id": lease_id},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("released"):
                    logger.info("released lease %s", lease_id)
                else:
                    logger.warning("release returned false (lease already expired?)")
        except Exception:
            logger.exception("failed to release lease")

    except NeverFitsError as exc:
        logger.warning("request will never fit: %s", exc)
    except BrokerRefusalError as exc:
        logger.warning("broker refused and retry exhausted: %s", exc)
    except Exception:
        logger.exception("unexpected error in main")


if __name__ == "__main__":
    main()
