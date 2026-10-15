"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    What a client should do when CUDA raises out-of-memory (or any GPU
    allocation failure) while holding a lease GpuBroker validly granted.

    This is a reproduced failure, not a hypothetical. See
    Assist/HANDOVER_post_admission_oom.md. A held lease is a promise
    about what the broker will not hand to someone else who asks. It is
    not a guarantee about physical memory, because not every GPU consumer
    asks. Ollama loaded a 4996 MB model without calling /acquire.
    GpuBroker only found out via its background poll of Ollama's API,
    up to OLLAMA_POLL_INTERVAL_SECONDS (default 20s) later. Whisper's
    held lease was still listed as valid while the next transcription
    hit CUDA "out of memory".

    The policy this file encodes, in order:

    1. Release the lease immediately, even though the lease is still
       valid. The failed process's GPU state may be partial or
       corrupted. Holding the lease does not recover the allocation,
       and it blocks the ledger from reflecting reality for others.
    2. Do not immediately re-acquire and retry the same work. GpuBroker's
       view of unmanaged consumers is only as fresh as its last poll.
       An immediate re-acquire can be granted against the same stale
       numbers and OOM again. compileYT hit this 3 times in a row
       before the fix (see the handover "What I changed on the client
       side" section).
    3. Raise a distinct typed exception (PostAdmissionOOM), not a
       generic one, so the caller can tell GPU pressure from bad input
       or a down service. Swallowing this into a generic exception hid
       the original compileYT bug for 570 downloads.
    4. Count consecutive post-admission failures. Stop offering new
       work to this GPU dependency after a small fixed threshold
       (2 here, matching compileYT) within a single run. Check the
       module counter on entry so a tripped breaker stays tripped for
       the rest of the run.
    5. Before the one bounded retry, wait at least one broker poll
       interval and re-check GET /status. Skip the retry if a game
       block is active (unmanaged_heavy or
       proton_game_retry_after_seconds > 0) or if mb > budget_mb.
       Retrying before the poll has elapsed is retrying against numbers
       you already know are stale.

    This is client-side damage limitation, matching the design GpuBroker
    has today. Broker-side revocation and a pressure endpoint are future
    work, not implemented here. See the handover "design question"
    section. A client that ignores this pattern will reinvent a worse
    version of it under a real OOM.
"""
# Imports

import logging
import os
import time

import httpx

# Globals

logger = logging.getLogger(__name__)

# Host-side default (this file is runnable directly, not only a template).
# A client inside a container uses http://host.containers.internal:8792,
# matching every real service in ~/Documents/Code/Docker/.
BROKER_URL = os.environ.get("BROKER_URL", "http://localhost:8792")
CALLER_NAME = "example-client"

# GpuBroker's default OLLAMA_POLL_INTERVAL_SECONDS (see .env.example).
# The one bounded retry below waits at least this long before trying
# again, so it retries against a /status reading GpuBroker has had a
# chance to refresh, not the same stale numbers that just OOM'd.
BROKER_POLL_INTERVAL_SECONDS = 20

# Stop offering new work to this dependency after this many consecutive
# post-admission failures within a single run. Matches the compileYT
# precedent in Assist/HANDOVER_post_admission_oom.md.
MAX_CONSECUTIVE_OOM_FAILURES = 2

_consecutive_oom_failures = 0


class PostAdmissionOOM(Exception):
    """
    Raised when a GPU allocation fails despite a currently-held, valid
    lease. Distinct from a generic exception on purpose: the caller needs
    to tell "GPU memory pressure from an unmanaged consumer" apart from
    every other failure. The correct response is to back off and
    eventually give up on this dependency, not retry the same request
    immediately or treat it as a bad input.
    """


class DependencyUnavailable(Exception):
    """
    Raised once MAX_CONSECUTIVE_OOM_FAILURES consecutive PostAdmissionOOM
    failures have been observed within a single run. The caller should
    stop offering new work to this dependency for the rest of the run
    (matching compileYT's WhisperUnavailable), not keep calling in and
    paying the cost of a doomed attempt for every remaining item.
    """


def _is_gpu_oom(exc: Exception) -> bool:
    """
    Input:      exc - an exception raised by the GPU work step
    Output:     bool - True if this looks like a GPU memory allocation
                failure rather than some other error
    Details:
        Best-effort text match. Real services in this repo hit this under
        two messages for the same cause (see Assist/HANDOVER_post_admission_oom.md):
        an explicit "CUDA failed with error out of memory", and, under the
        same pressure, a misleading "cudaErrorInvalidDevice: invalid device
        ordinal" on a single-GPU host with a valid device. Both count.
        Adjust this list if your runtime's OOM error text differs. This
        is a text match, not a typed exception, because CUDA OOM surfaces
        through whatever library wraps it (here, faster-whisper/CTranslate2)
        with no shared base exception type across frameworks.
    """
    text = str(exc).lower()
    return "out of memory" in text or "invaliddevice" in text or "invalid device ordinal" in text


def _retry_allowed_by_status(mb: int) -> bool:
    """
    Input:      mb - VRAM that the retry would request
    Output:     True if GET /status says a retry is worth attempting
    Details:
        Skip the retry on a game block (unmanaged_heavy or
        proton_game_retry_after_seconds > 0) or when mb > budget_mb.
        A failed status fetch also skips the retry.
    """
    try:
        resp = httpx.get(f"{BROKER_URL}/status", timeout=5.0)
        resp.raise_for_status()
        status = resp.json()
    except Exception:
        logger.exception("could not fetch /status before OOM retry; skipping retry")
        return False
    if status.get("unmanaged_heavy"):
        logger.warning("skipping OOM retry: unmanaged_heavy is true")
        return False
    proton_retry = status.get("proton_game_retry_after_seconds") or 0
    try:
        proton_retry = float(proton_retry)
    except (TypeError, ValueError):
        logger.warning(
            "ignoring non-numeric proton_game_retry_after_seconds: %r",
            proton_retry,
        )
        proton_retry = 0
    if proton_retry > 0:
        logger.warning(
            "skipping OOM retry: proton_game_retry_after_seconds=%s",
            proton_retry,
        )
        return False
    budget_mb = status.get("budget_mb")
    if budget_mb is None:
        logger.warning("skipping OOM retry: /status omitted budget_mb")
        return False
    if mb > budget_mb:
        logger.warning(
            "skipping OOM retry: mb=%d exceeds budget_mb=%s",
            mb, budget_mb,
        )
        return False
    return True


def _release_lease(lease_id: str) -> None:
    """
    Input:      lease_id - lease to release
    Output:     None
    Details:
        Best-effort /release. Logs a non-200 status or released=false.
        Never raises; the sweeper reclaims an unreleased lease.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{BROKER_URL}/release", json={"lease_id": lease_id},
            )
        if resp.status_code != 200:
            logger.warning(
                "release returned %d for lease %s: %s",
                resp.status_code, lease_id, resp.text,
            )
            return
        try:
            payload = resp.json()
        except Exception:
            logger.warning(
                "release body was not JSON for lease %s: %s",
                lease_id, resp.text,
            )
            return
        if not payload.get("released"):
            logger.warning(
                "release returned released=false for lease %s",
                lease_id,
            )
    except httpx.HTTPError as exc:
        logger.warning(
            "release failed for lease %s (broker's sweeper will reclaim it): %s",
            lease_id, exc,
        )


def do_gpu_work_with_oom_policy(work_fn, mb: int, timeout_s: float = 30.0) -> object:
    """
    Input:      work_fn - zero-argument callable that does the actual GPU
                work and either returns a result or raises on failure
                mb - VRAM to request from the broker
                timeout_s - acquire timeout, passed straight to /acquire
    Output:     whatever work_fn returns
    Details:
        Acquires a lease, runs work_fn, and on a GPU OOM specifically
        (not any other failure) applies the release-then-raise-distinct
        policy from the module docstring, points 1 through 3. Point 4
        (the consecutive-failure counter and the give-up threshold) lives
        in run_with_dependency_breaker below, since it spans multiple
        calls to this function, not one.
    """
    lease_id = None
    # Acquire phase: a 504 here (or a connection failure) is a plain
    # acquire denial, not a post-admission OOM. The broker never
    # promised anything, so there is nothing to release and nothing this
    # file's OOM-recovery policy applies to. See handling_refusal.py for
    # how to handle a denial at this stage (never-fits vs transient,
    # bounded retry). Not caught as PostAdmissionOOM.
    with httpx.Client(timeout=timeout_s + 15.0) as client:
        resp = client.post(
            f"{BROKER_URL}/acquire",
            json={"caller": CALLER_NAME, "mb": mb, "timeout_s": timeout_s},
        )
    if resp.status_code == 504:
        logger.warning("acquire refused (no room): %s", resp.text)
        raise RuntimeError(
            f"broker refused a {mb} MB lease before any GPU work ran -- see "
            "handling_refusal.py, not this file, for how to respond to that"
        )
    resp.raise_for_status()
    lease_id = resp.json().get("lease_id")

    # Work phase: the lease is valid. Only a failure here, specifically
    # one that looks like a GPU OOM, is what this file's policy covers.
    try:
        return work_fn()

    except Exception as exc:
        if _is_gpu_oom(exc):
            logger.warning(
                "GPU OOM while holding lease %s (mb=%d): %s -- releasing and raising "
                "PostAdmissionOOM, not retrying in place",
                lease_id, mb, exc,
            )
            raise PostAdmissionOOM(f"GPU OOM despite a valid {mb} MB lease: {exc}") from exc
        logger.exception("GPU work failed for a reason other than OOM (lease %s)", lease_id)
        raise

    finally:
        if lease_id is not None:
            _release_lease(lease_id)


def run_with_dependency_breaker(work_fn, mb: int) -> object:
    """
    Input:      work_fn - as above; mb - as above
    Output:     whatever work_fn returns
    Details:
        Point 4 and point 5 from the module docstring. Tracks consecutive
        PostAdmissionOOM failures across calls (module-level state, since
        a real client calls this once per work item in a loop). Checks
        the counter on entry so a tripped breaker stays tripped for the
        rest of the run. After MAX_CONSECUTIVE_OOM_FAILURES in a row,
        raises DependencyUnavailable. A success resets the counter.

        The one retry waits BROKER_POLL_INTERVAL_SECONDS, then GET
        /status. Skip the retry if unmanaged_heavy is true,
        proton_game_retry_after_seconds > 0, or mb > budget_mb.
    """
    global _consecutive_oom_failures
    if _consecutive_oom_failures >= MAX_CONSECUTIVE_OOM_FAILURES:
        raise DependencyUnavailable(
            f"{MAX_CONSECUTIVE_OOM_FAILURES} consecutive post-admission OOM failures; "
            "not offering this dependency further work this run"
        )
    for attempt in range(1, MAX_CONSECUTIVE_OOM_FAILURES + 1):
        try:
            result = do_gpu_work_with_oom_policy(work_fn, mb)
            _consecutive_oom_failures = 0
            return result
        except PostAdmissionOOM as exc:
            _consecutive_oom_failures += 1
            logger.warning(
                "post-admission OOM %d/%d for this dependency: %s",
                _consecutive_oom_failures, MAX_CONSECUTIVE_OOM_FAILURES, exc,
            )
            if _consecutive_oom_failures >= MAX_CONSECUTIVE_OOM_FAILURES:
                raise DependencyUnavailable(
                    f"{MAX_CONSECUTIVE_OOM_FAILURES} consecutive post-admission OOM failures; "
                    "not offering this dependency further work this run"
                ) from exc
            if attempt < MAX_CONSECUTIVE_OOM_FAILURES:
                logger.info(
                    "waiting %ds before the one retry (broker poll interval)",
                    BROKER_POLL_INTERVAL_SECONDS,
                )
                time.sleep(BROKER_POLL_INTERVAL_SECONDS)
                if not _retry_allowed_by_status(mb):
                    raise


if __name__ == "__main__":
    # Illustrative only. run_with_dependency_breaker expects a real
    # GpuBroker and a real GPU workload behind work_fn. This just shows
    # the call shape a real client (like compileYT's Whisper calls) would
    # use around each unit of work.
    logging.basicConfig(level=logging.INFO)

    def example_work():
        raise RuntimeError("CUDA failed with error out of memory")  # simulated

    try:
        run_with_dependency_breaker(example_work, mb=2048)
    except DependencyUnavailable as exc:
        logger.warning("giving up on this dependency for the rest of the run: %s", exc)
    except PostAdmissionOOM as exc:
        logger.warning("post-admission OOM without a retry: %s", exc)
