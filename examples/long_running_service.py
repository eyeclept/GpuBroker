"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Long-running service integration with GpuBroker. Shows lazy loading,
    background heartbeat, and idle eviction.

    This is the fuller "Whisper shape": the model is acquired only on first
    use. A background thread sends periodic heartbeats to keep the lease
    alive. An idle timeout evicts the model after N seconds of inactivity.
    Acquire sends preemptible: true because this service can unload on
    demand. The broker may then ask it to unload via heartbeat preempt=true.
    Heartbeat outcomes: preempt=true unloads and releases now; ok=false
    means the lease is gone so unload and clear it; a transport error
    keeps the lease and retries on the next tick.

    See minimal_client.py for the smallest correct pattern (acquire, work,
    release). See the Whisper server and Flux2Klein9B server in
    ~/Documents/Code/Docker/ for production implementations.
"""
# Imports
import argparse
import gc
import logging
import os
import threading
import time
from typing import NamedTuple

import httpx

# Globals
logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("BROKER_URL", "http://localhost:8792")
CALLER_NAME = "long_running_example"
ACQUIRE_MB = 2048
IDLE_TIMEOUT_SECONDS = 15
HEARTBEAT_INTERVAL_SECONDS = 5

# Broker internal constant: if no heartbeat is sent within this time,
# the broker's sweeper reclaims the lease. The client does not receive a
# notification when this happens. /acquire returns 200
# {"granted": true, "lease_id": ..., "warning"?} on grant; 504 when there
# is no room (Retry-After 10); 503 unmanaged_gpu_load (Retry-After up to
# 1800) or too_many_leases (Retry-After 5); 422 for invalid input or a
# never-fit request; 400 for a bad priority. A reclaimed lease is
# invisible until the client's next heartbeat ({"ok": false}) or the
# next acquire for a fresh lease. A bug in the heartbeat loop that stops
# sending heartbeats will silently lose the lease. A preempt-flagged
# holder must release within 30 s or the watchdog reclaims it.
GPU_BROKER_LEASE_TTL_SECONDS = 180


class HeartbeatResult(NamedTuple):
    """Heartbeat outcome. ok is True, False (lease gone), or None (unknown)."""

    ok: bool | None
    preempt: bool = False
    reason: str = ""


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
    Output:     lease_id string if granted, None if timeout or error
    Details:
        Request a VRAM lease from the broker. Returns lease_id on success
        (200) and logs the one-shot warning if present. Logs 504, 503
        (detail and Retry-After), and 422 as distinct refusals. Sends
        preemptible: true so the broker can flag this holder. Omits pid.
    """
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{BROKER_URL}/acquire",
                json={
                    "caller": CALLER_NAME,
                    "mb": mb,
                    "timeout_s": 30.0,
                    "preemptible": True,
                    # Do not send pid: a container pid would not match host pids.
                },
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
        Release a held VRAM lease. Idempotent; logs but never raises.
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
            logger.warning("Release returned %d", resp.status_code)
    except Exception as exc:
        logger.warning("Failed to release lease: %s", exc)


def send_heartbeat(lease_id: str | None) -> HeartbeatResult:
    """
    Input:      lease_id -- lease to heartbeat, or None to skip
    Output:     HeartbeatResult
    Details:
        Send a heartbeat to keep the lease alive. Called periodically by
        the background heartbeat thread. The endpoint answers 200.
        ok=False means the lease is gone. ok=True with preempt=True means
        unload and release before the 30 s grace. ok=None is a transport
        or non-200 failure; keep the lease and retry on the next tick.
        Never raises.
    """
    if not lease_id:
        return HeartbeatResult(ok=True)
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{BROKER_URL}/heartbeat",
                json={"lease_id": lease_id},
                timeout=5.0,
            )
        if resp.status_code != 200:
            logger.warning("Heartbeat returned %d", resp.status_code)
            return HeartbeatResult(ok=None)
        try:
            data = resp.json()
        except Exception:
            logger.warning(
                "heartbeat body was not JSON (lease %s): %s",
                lease_id, resp.text,
            )
            return HeartbeatResult(ok=None)
        if not isinstance(data, dict):
            logger.warning(
                "heartbeat body was not an object (lease %s): %s",
                lease_id, resp.text,
            )
            return HeartbeatResult(ok=None)
        if not data.get("ok", False):
            logger.warning("Heartbeat returned ok=false (lease is gone): %s", lease_id)
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
        logger.warning("Heartbeat failed: %s (will retry next tick)", exc)
        return HeartbeatResult(ok=None)


class ServiceWithBroker:
    """
    Simulates a long-running GPU service that coordinates with GpuBroker.

    Shows:
      - Lazy loading: model acquired on first request, not at startup
      - Background heartbeat: a thread that sends heartbeats so the
        broker sweeper does not reclaim the lease after 180s
      - Idle eviction: if no request arrives for IDLE_TIMEOUT_SECONDS,
        the model is unloaded and the lease is released
      - Preempt: unload and release as soon as heartbeat says preempt
      - Lost lease: ok=false unloads and clears lease_id

    Acquire declares preemptible=True because this service can unload
    on demand. The broker may then set preempt on a later heartbeat.
    """

    def __init__(self, idle_timeout_seconds: float | None = IDLE_TIMEOUT_SECONDS):
        self.model = None
        self.lease_id = None
        self.last_activity_time = time.monotonic()
        self.heartbeat_thread = None
        self.stop_heartbeat = False
        self._lock = threading.Lock()
        self.idle_timeout_seconds = idle_timeout_seconds

    def _load_model(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Synchronously load the model. In a real service, this is where
            you would load a Whisper model, Flux2Klein9B weights, etc.
            This placeholder just sets a flag.
        """
        logger.info("Loading model (placeholder)...")
        time.sleep(0.1)  # Pretend model load
        self.model = {"name": "example-model", "loaded_at": time.monotonic()}
        logger.info("Model loaded")

    def _unload_model(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Unload the model from GPU memory. In a real service, this is where
            you would drop references to the model, call gc.collect(), etc.
        """
        logger.info("Unloading model...")
        self.model = None
        gc.collect()
        logger.info("Model unloaded")

    def _ensure_loaded(self) -> bool:
        """
        Input:      None
        Output:     True if model is loaded and lease is valid, False otherwise
        Details:
            Ensure the model is loaded and we hold a valid lease. Acquires the
            lease on first load. Returns False if acquire failed, True if
            load succeeded.
        """
        with self._lock:
            if self.model is not None:
                return True

            # Not loaded. Try to acquire a lease first.
            self.lease_id = acquire_lease(ACQUIRE_MB)
            if not self.lease_id:
                logger.error("Could not acquire lease; cannot load model")
                return False

            # Lease acquired. Now load the model.
            try:
                self._load_model()
            except Exception:
                logger.exception("Failed to load model")
                release_lease(self.lease_id)
                self.lease_id = None
                return False

            # Model loaded. Start the heartbeat thread.
            self._start_heartbeat_thread()
            return True

    def _start_heartbeat_thread(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Start the background heartbeat thread if it is not already
            running. The thread sends heartbeats so the lease stays alive.
            If the thread stops sending, the broker sweeper reclaims the
            lease after GPU_BROKER_LEASE_TTL_SECONDS (180s default). The
            client is not notified. It discovers the loss on the next
            heartbeat or release.
        """
        if self.heartbeat_thread is not None and self.heartbeat_thread.is_alive():
            return

        self.stop_heartbeat = False
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        logger.info("Heartbeat thread started")

    def _heartbeat_loop(self) -> None:
        """
        Input:      None (runs in a background thread)
        Output:     None (runs until stop_heartbeat is set)
        Details:
            Background heartbeat loop. Wakes every HEARTBEAT_INTERVAL_SECONDS
            to send a heartbeat and check for idle timeout. preempt=true
            unloads and releases now. ok=false unloads and clears the
            lease. A transport error keeps the lease. Idle timeout evicts
            the model and stops the loop.
        """
        while not self.stop_heartbeat:
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)

            lease_id = self.lease_id
            if lease_id:
                hb = send_heartbeat(lease_id)
                if hb.ok is False:
                    logger.warning(
                        "lease %s is gone; unloading and clearing lease",
                        lease_id,
                    )
                    with self._lock:
                        if self.lease_id == lease_id:
                            self._unload_model()
                            self.lease_id = None
                    break
                if hb.ok is True and hb.preempt:
                    logger.warning(
                        "preempt for lease %s (%s); unloading and releasing",
                        lease_id, hb.reason,
                    )
                    with self._lock:
                        if self.lease_id == lease_id:
                            self._unload_model()
                            release_lease(self.lease_id)
                            self.lease_id = None
                    break
                # hb.ok is None: transport error; keep the lease.

            # Check idle timeout. None means hold until shutdown or preempt.
            idle_timeout = self.idle_timeout_seconds
            if idle_timeout is not None:
                time_since_activity = time.monotonic() - self.last_activity_time
                if (
                    self.model is not None
                    and time_since_activity >= idle_timeout
                ):
                    logger.info(
                        "Idle timeout reached (%ds), evicting model",
                        idle_timeout,
                    )
                    with self._lock:
                        self._unload_model()
                        release_lease(self.lease_id)
                        self.lease_id = None
                    break

    def process_request(self, work_id: str) -> str:
        """
        Input:      work_id -- a request identifier
        Output:     result string
        Details:
            Process a work request (simulated). Ensures the model is loaded,
            does placeholder work, and returns a result. Updates last_activity
            so the idle timeout is reset.
        """
        if not self._ensure_loaded():
            raise RuntimeError("Could not load model")

        with self._lock:
            if self.model is None:
                raise RuntimeError("Model unloaded (possible race with idle eviction)")

            # === DO WORK HERE ===
            logger.info("Processing request %s", work_id)
            time.sleep(0.1)  # Placeholder work
            result = f"Completed {work_id}"
            # === END WORK ===

            # Update activity so the idle timer resets
            self.last_activity_time = time.monotonic()
            return result

    def shutdown(self) -> None:
        """
        Input:      None
        Output:     None
        Details:
            Gracefully shut down the service. Stops the heartbeat thread,
            unloads the model, and releases the lease.
        """
        logger.info("Shutting down...")
        self.stop_heartbeat = True

        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=5)

        with self._lock:
            if self.model is not None:
                self._unload_model()
            release_lease(self.lease_id)
            self.lease_id = None

        logger.info("Shutdown complete")


def main() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Main entry point. Demonstrates the long-running service pattern
        with lazy loading, heartbeat, and idle eviction. --hold-seconds
        N (or HOLD_SECONDS) keeps the lease that long with idle eviction
        off, so a P1 waiter can preempt this holder.
    """
    parser = argparse.ArgumentParser(
        description="Long-running GpuBroker service example.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=float(os.environ.get("HOLD_SECONDS", "0") or "0"),
        help=(
            "After the first request, keep the lease this many seconds "
            "without idle eviction. 0 runs the default idle-timeout demo."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    idle_timeout = None if args.hold_seconds > 0 else IDLE_TIMEOUT_SECONDS
    service = ServiceWithBroker(idle_timeout_seconds=idle_timeout)

    try:
        # Simulate a few requests
        result1 = service.process_request("work-1")
        logger.info("Result: %s", result1)

        if args.hold_seconds > 0:
            logger.info(
                "Holding lease for %.0fs (idle eviction off)",
                args.hold_seconds,
            )
            time.sleep(args.hold_seconds)
            return

        time.sleep(2)

        result2 = service.process_request("work-2")
        logger.info("Result: %s", result2)

        # Let the idle timer run (period of no requests)
        logger.info("Waiting for idle timeout...")
        time.sleep(IDLE_TIMEOUT_SECONDS + 5)

        # The model should now be evicted and the lease released.
        # Another request shows it re-acquires.
        result3 = service.process_request("work-3")
        logger.info("Result: %s", result3)


    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
