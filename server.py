"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

GpuBroker: a small cross-process VRAM-budget lock for this host's single
RTX 4070 shared by Flux2Klein9B, Needle, and Ollama.

This is not a proxy in the GPU data path. Callers still talk to CUDA/XLA
directly. This is a gatekeeper. Managed clients (Flux2Klein9B, Needle)
call POST /acquire before they load weights onto the GPU. They call
POST /release once they evict those weights. The ledger then records
what is about to be resident, not only what nvidia-smi already saw.
Ollama is not modified to cooperate. Its Go source is out of scope.
The broker accounts for Ollama passively via GET /api/ps.

Fairness has two independent mechanisms. Preemption (cooperative,
contention-driven, `preemptible=True` holders only): a queued caller's
priority tier (P0-P3) ages the longer it waits, and once that aged value
exceeds the weakest current preemptible holder's, the broker flags that
holder's heartbeat response for it to self-evict. See
_effective_priority, _weakest_preemptible_holder. The watchdog reclaims a
lease whose heartbeat is stale, whose non-preemptible or stale lease
exceeds GPU_BROKER_MAX_LEASE_AGE_SECONDS, or whose preemption grace
period expired. Fresh preemptible leases remain available for fairness
preemption. The watchdog resolves a registered pid to a unique container
GPU compute process (host pid or innermost NSpid), re-checks the lease,
then SIGTERMs that host pid. Host and ambiguous matches stay ledger-only.
See _reclaim_broken_lease.

Endpoints:
  POST /acquire   {caller, mb, timeout_s, priority, preemptible, pid}
                  -> {granted: true, lease_id, warning?} or 504/400/422, or
                     503 + Retry-After while a game holds the GPU or 64
                     leases are already active (detail too_many_leases)
  POST /release    {lease_id}             -> {released: true}
  POST /heartbeat  {lease_id}             -> {ok: true, preempt, reason}
  POST /admin/clear {admin_token, dry_run} -> kill list + ledger reset, or 403/501
  GET  /status                           -> ledger + live budget numbers
  GET  /health                           -> {"status": "ok"}
"""

# Imports

import asyncio
from contextlib import asynccontextmanager
import hmac
import itertools
import json
import logging
import math
import os
import signal
import subprocess
import tempfile
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Globals

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("GPU_BROKER_PORT", "8792"))
SAFETY_MARGIN_MB = int(os.environ.get("GPU_SAFETY_MARGIN_MB", "1536"))
# Floor on the computed p95. Defaults to GPU_SAFETY_MARGIN_MB so a
# cold-start broker and a quiet-desktop p95 cannot admit more
# aggressively than the last operator-set fallback.
SAFETY_MARGIN_MIN_MB = int(os.environ.get("GPU_SAFETY_MARGIN_MIN_MB", str(SAFETY_MARGIN_MB)))
LEDGER_PATH = os.environ.get("GPU_BROKER_LEDGER_PATH", "/data/leases.json")
NON_LEASE_USAGE_PATH = os.environ.get(
    "NON_LEASE_USAGE_PATH",
    os.path.join(os.path.dirname(LEDGER_PATH) or ".", "non_lease_usage.json"),
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.containers.internal:11434")
OLLAMA_POLL_INTERVAL_SECONDS = int(os.environ.get("OLLAMA_POLL_INTERVAL_SECONDS", "20"))
NON_LEASE_POLL_INTERVAL_SECONDS = int(os.environ.get("NON_LEASE_POLL_INTERVAL_SECONDS", "30"))
# 48h covers two weekday work cycles (Steam/Firefox evenings) without
# keeping a weekend game session in the p95 for half a week. 24h misses
# a quiet morning after a heavy night. 72h keeps stale desktop changes.
NON_LEASE_WINDOW_HOURS = float(os.environ.get("NON_LEASE_WINDOW_HOURS", "48"))
# 60 samples at the 30s default is 30 minutes. p95 of fewer points is
# one of the top two values and is too noisy to replace the fallback.
NON_LEASE_MIN_SAMPLES = int(os.environ.get("NON_LEASE_MIN_SAMPLES", "60"))
# p95 of unlocked samples: high enough to cover a heavy compositor/
# browser/Steam session, not max() which a single overlay spike would
# pin for the rest of the window.
NON_LEASE_PERCENTILE = 95.0
# Persist the sample file every 10 samples (10 x 30 s = 5 min). Losing
# up to 5 min of a 48 h window on a crash is negligible.
NON_LEASE_PERSIST_EVERY = 10

# Unmanaged heavy load (a game, a render, anything big the ledger does not
# track). Unmanaged overhead at or above this many MB means "someone else
# needs the GPU now". Four consequences: (1) such samples are excluded
# from the baseline percentile, so a 2-hour game session cannot inflate
# the safety margin and lock every lease out for the next 48h; (2) every
# preemptible lease is flagged so its holder self-evicts; (3) every new
# acquire, preemptible or not and of any size, is refused with 503 and
# Retry-After until the load has been gone for UNMANAGED_CLEAR_SECONDS;
# (4) a caller that already holds a lease still gets it back on a repeat
# acquire. Detection measures nvidia-smi used minus the GPU memory of
# processes running inside a container (see _poll_unmanaged_load), not
# used minus leases: leases are sized above real use and would mask a game.
# Desktop baseline on this host is 1.0-1.7 GB; a game was observed at ~10 GB.
UNMANAGED_YIELD_OVERHEAD_MB = float(os.environ.get("UNMANAGED_YIELD_OVERHEAD_MB", "4096"))
# Fast poll: yielding must beat the 30s sample poll or a game stutters first.
UNMANAGED_POLL_INTERVAL_SECONDS = float(os.environ.get("UNMANAGED_POLL_INTERVAL_SECONDS", "5"))
# Consecutive over-threshold polls before declaring heavy load, so one
# transient spike (shader compile, alt-tab) does not evict a model.
UNMANAGED_ENTER_SAMPLES = int(os.environ.get("UNMANAGED_ENTER_SAMPLES", "2"))
# Continuous under-threshold time before heavy load is declared over, so a
# loading screen or menu does not let the model reload mid-session.
UNMANAGED_CLEAR_SECONDS = float(os.environ.get("UNMANAGED_CLEAR_SECONDS", "120"))
# Detail string a refused acquire carries; clients match on it.
UNMANAGED_BUSY_DETAIL = "unmanaged_gpu_load"
TOO_MANY_LEASES_DETAIL = "too_many_leases"
# cgroup path markers of a container process (podman, docker). The broker
# runs with `pid: host`, so /proc/<pid>/cgroup is the host process's.
CONTAINER_CGROUP_MARKERS = ("libpod-", "docker-")

# Steam's launch wrapper stays alive for a Proton game. Scan its host PID
# before each acquire and once per second to preempt holders even if the
# game uses little VRAM. Keep the block for 30 minutes after it exits.
PROTON_POLL_INTERVAL_SECONDS = 1.0
PROTON_CLEAR_SECONDS = 1800
# Tests may point this at a fake tree. Production reads the host procfs.
_PROC_ROOT = "/proc"

SWEEP_INTERVAL_SECONDS = int(os.environ.get("GPU_BROKER_SWEEP_INTERVAL_SECONDS", "30"))
LEASE_TTL_SECONDS = int(os.environ.get("GPU_BROKER_LEASE_TTL_SECONDS", "180"))

# Watchdog safety net (Mechanism 2). Reclaims stale heartbeats for every
# lease and reclaims old non-preemptible leases. A fresh preemptible lease
# is excluded from age-based reclaim because fairness preemption can
# reach its live holder without discarding its ledger entry. The age limit
# remains a backstop for non-preemptible leases and preemptible leases
# whose heartbeat has gone stale.
# As measured here, legitimate long-running tasks top out around 3
# minutes. The 30-minute default is a backstop, not a normal-operation
# limit.
MAX_LEASE_AGE_SECONDS = float(os.environ.get("GPU_BROKER_MAX_LEASE_AGE_SECONDS", "1800"))

# Fairness preemption (Mechanism 1 -- contention-driven, cooperative,
# preemptible=True leases only). See _effective_priority for the aging
# formula and _weakest_preemptible_holder for the comparison. Default
# window matches this project's other "how long is too long to go
# unnoticed" reference point (LEASE_TTL_SECONDS's default), kept as a
# separate knob since it controls a different concern (priority aging,
# not heartbeat staleness).
PRIORITY_AGING_WINDOW_SECONDS = float(os.environ.get("GPU_BROKER_PRIORITY_AGING_WINDOW_SECONDS", "180"))

# See the INCUMBENCY_BONUS assertion in startup() for the correctness
# requirement this value must satisfy (strictly less than TIER_STEP).
# Default is half a tier-step: preemption requires substantial aging
# before it can even mathematically trigger, while still allowing it
# before a waiter fully ages out.
INCUMBENCY_BONUS = float(os.environ.get("GPU_BROKER_INCUMBENCY_BONUS", "5"))

# How long a lease flagged for preemption (see acquire()) gets to
# self-evict via its own idle-watcher loop before falling through to the
# watchdog kill path. Reasoned, not measured: a genuine evict/restore-to-
# CPU cycle at a backend's own safe/step boundary is expected to take at
# most a handful of seconds, so 30s is generous headroom without leaving
# a waiting caller stuck for long behind an unresponsive holder.
PREEMPT_GRACE_SECONDS = float(os.environ.get("GPU_BROKER_PREEMPT_GRACE_SECONDS", "30"))

ACQUIRE_POLL_MIN_SECONDS = 1.0
ACQUIRE_POLL_MAX_SECONDS = 10.0
# Cap on /acquire timeout_s. A waiter occupies the FIFO head until
# this deadline. Infinity and NaN never expire. 600 s is above the
# 300 s default and below the Proton cooldown. Not an env var: this
# is a safety bound, not a tuning knob.
MAX_ACQUIRE_TIMEOUT_SECONDS = 600.0
# Max caller name length. Not an env var: this is a safety bound
# against unbounded ledger keys and /status payload growth.
MAX_CALLER_LEN = 128
# uuid4 strings are 36 characters. Cap well above that so a
# well-formed id always fits, but a multi-kilobyte probe does not.
MAX_LEASE_ID_LEN = 64
# A single-GPU host has a handful of clients; the proxy creates one
# zero-size lease per in-flight request, so 64 leaves headroom.
MAX_ACTIVE_LEASES = 64
# Drop one-shot watchdog notices that no caller has come back to
# collect. Pruned at reclaim time and at grant time.
NOTIFICATION_TTL_SECONDS = 3600.0


# A caller that got a 504 and retries within this window keeps its
# fairness wait age. Must exceed a consumer's longest backoff sleep
# (FreeToken caps at 60 s).
WAITER_MEMORY_SECONDS = 120.0

# Priority tiers, highest to lowest, matching this repo's own P0-P3
# vocabulary (P0 = highest priority/most urgent, P3 = lowest). TIER_STEP
# is the numeric gap between adjacent tiers; the anti-starvation aging
# formula (_effective_priority) and the incumbency bonus above are both
# defined relative to it.
TIER_BASE_PRIORITY = {"P0": 30.0, "P1": 20.0, "P2": 10.0, "P3": 0.0}
TIER_STEP = 10.0
DEFAULT_PRIORITY = "P2"

# Fixed, generic text: this is what a heartbeat response's "reason" field
# carries once a lease is flagged for preemption (see heartbeat()). It
# never names the waiting caller -- callers only need to know to evict,
# not who is waiting. Source-specific strings cover Proton and unmanaged
# load; a restored lease without preempt_reason uses the fairness text.
PREEMPT_REASON_TEXT = (
    "held past the fairness aging threshold; another queued request needs the GPU"
)
PREEMPT_PROTON_REASON_TEXT = (
    "a Steam Proton game is using the GPU"
)
PREEMPT_UNMANAGED_REASON_TEXT = (
    "unmanaged GPU load is using the GPU"
)
_PREEMPT_REASON_TEXT_BY_SOURCE = {
    "fairness": PREEMPT_REASON_TEXT,
    "proton": PREEMPT_PROTON_REASON_TEXT,
    "unmanaged": PREEMPT_UNMANAGED_REASON_TEXT,
}

# Caller-facing fragments for the one-shot watchdog notice (see
# _reclaim_broken_lease). Joined with "and" when more than one reason
# applied to the same lease.
_RECLAIM_REASON_TEXT = {
    "heartbeat_stale": "exceeding the heartbeat staleness limit",
    "lease_age": f"exceeding the {int(MAX_LEASE_AGE_SECONDS // 60)}-minute total lease-age limit",
    "preempt_grace_expired": "not self-evicting within the fairness-preemption grace period",
}
# Unset (the default) disables POST /admin/clear. The endpoint then
# returns 501 instead of running without a token. This endpoint kills
# host GPU processes. The broker listens on the docker network, not
# only loopback, because other containers must reach /acquire. Without
# a shared secret, any container on that network could kill host
# processes owned by this user. See docker-compose.yml `pid: host` for
# why this process can reach host PIDs. See Assist/ for the incident
# that motivated this endpoint.
ADMIN_TOKEN = os.environ.get("GPU_BROKER_ADMIN_TOKEN", "")
ADMIN_KILL_GRACE_SECONDS = 3.0
ADMIN_KILL_POLL_SECONDS = 0.1
# Host header allowlist (S-16 DNS rebinding). Extra names from
# GPU_BROKER_ALLOWED_HOSTS are appended. Defaults cover loopback,
# the in-container healthcheck (localhost), and client containers
# (host.containers.internal).
DEFAULT_ALLOWED_HOSTS = (
    "localhost",
    "127.0.0.1",
    "::1",
    "host.containers.internal",
)

_lock = asyncio.Lock()
# Broadcast Event. Waiters capture the current object under `_lock`
# after the admission check, then wait outside the lock. Notify sets
# that object and replaces this global with a fresh Event, so a notify
# between capture and wait is not lost. See acquire().
_admission_event = asyncio.Event()

_leases: dict[str, dict] = {}          # lease_id -> {caller, mb, acquired_at, last_seen,
                                        #              priority, preemptible, pid,
                                        #              preempt_flagged_at, preempt_reason}
_caller_to_lease: dict[str, str] = {}   # caller -> lease_id
_pending_seq = itertools.count()
_pending: set[int] = set()             # currently-waiting acquire sequence numbers
_pending_notifications: dict[str, tuple[str, float]] = {}  # caller -> (text, created_monotonic)
# caller -> (wait start, last request end), monotonic. Keeps a caller's
# fairness age across a 504 and a backoff retry; see WAITER_MEMORY_SECONDS.
_waiting_since: dict[str, tuple[float, float]] = {}

_gpu_total_mb: float = 0.0
_ollama_mb: float = 0.0
_non_lease_samples: list[dict] = []  # {ts, mb, locked} rolling window
_computed_margin_mb: float | None = None  # None -> SAFETY_MARGIN_MB fallback
_unmanaged_heavy = False                 # see UNMANAGED_YIELD_OVERHEAD_MB
_unmanaged_over_count = 0                # consecutive over-threshold polls
_unmanaged_clear_since: float | None = None  # monotonic start of the under-threshold run
_unmanaged_overhead_mb: float | None = None  # last overhead the state machine saw
_last_used_mb: float | None = None  # last nvidia-smi memory.used; pollers write
_last_used_at: float | None = None  # monotonic time of _last_used_mb
_proton_game_active = False
_proton_game_uncertain = False  # ledger said active before a broker restart
_proton_game_cleared_at: float | None = None  # wall clock; persisted in ledger
# Pids of foreign-uid Proton wrappers already logged. Pruned to live
# pids after each scan so a reused pid can warn again.
_foreign_proton_warned_pids: set[int] = set()

_jeepney_import_logged = False


class AcquireRequest(BaseModel):
    """
    Input: JSON body with caller (str, 1 to MAX_CALLER_LEN chars), mb
        (float, finite, >= 0), timeout_s (float, default 300.0, finite,
        0 to MAX_ACQUIRE_TIMEOUT_SECONDS), priority (str, default
        "P2" -- must be one of TIER_BASE_PRIORITY's keys), preemptible
        (bool, default False -- this holder can genuinely pause/evict
        and so may be flagged for fairness preemption), pid (int > 0,
        optional -- this caller's host pid or its own container-namespace
        pid. The watchdog signals only a matching container GPU process;
        see _reclaim_broken_lease)
    Output: None (Pydantic model)
    Details: Request model for /acquire endpoint. FastAPI returns 422
        when a field fails these bounds.
    """
    caller: str = Field(min_length=1, max_length=MAX_CALLER_LEN)
    mb: float = Field(ge=0, allow_inf_nan=False)
    timeout_s: float = Field(
        default=300.0,
        ge=0,
        le=MAX_ACQUIRE_TIMEOUT_SECONDS,
        allow_inf_nan=False,
    )
    priority: str = DEFAULT_PRIORITY
    preemptible: bool = False
    pid: int | None = Field(default=None, gt=0)


class ReleaseRequest(BaseModel):
    """
    Input: JSON body with lease_id (str, max MAX_LEASE_ID_LEN chars)
    Output: None (Pydantic model)
    Details: Request model for /release endpoint
    """
    lease_id: str = Field(max_length=MAX_LEASE_ID_LEN)


class HeartbeatRequest(BaseModel):
    """
    Input: JSON body with lease_id (str, max MAX_LEASE_ID_LEN chars)
    Output: None (Pydantic model)
    Details: Request model for /heartbeat endpoint
    """
    lease_id: str = Field(max_length=MAX_LEASE_ID_LEN)


class AdminClearRequest(BaseModel):
    """
    Input: JSON body with admin_token (str), dry_run (bool, default False)
    Output: None (Pydantic model)
    Details: Request model for /admin/clear. dry_run=true reports what would
        be killed without killing anything or touching the ledger.
    """
    admin_token: str
    dry_run: bool = False


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


def _list_gpu_compute_processes() -> list[dict]:
    """
    Input:      None
    Output:     list of {"pid": int, "process_name": str, "used_mb": float},
                one per process nvidia-smi reports as holding GPU memory
    Details:
        Synchronous nvidia-smi compute-apps query. PIDs are only
        meaningful (and killable) when this process shares the host PID
        namespace. See docker-compose.yml `pid: host`. Returns an empty
        list, not an error, when nvidia-smi reports nothing or a line
        cannot be parsed. An admin clear on an empty GPU is a normal case.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    processes = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            logger.warning("unparseable nvidia-smi compute-apps line, skipping: %r", line)
            continue
        try:
            processes.append({
                "pid": int(parts[0]),
                "process_name": parts[1],
                "used_mb": float(parts[2]),
            })
        except ValueError:
            logger.warning("unparseable nvidia-smi compute-apps line, skipping: %r", line)
    return processes


def _pid_in_container(pid: int) -> bool:
    """
    Input:      pid - host PID
    Output:     bool - True if the process runs inside a podman or docker
                container
    Details:
        Reads _PROC_ROOT/<pid>/cgroup. A container process has a cgroup
        path with a CONTAINER_CGROUP_MARKERS entry (libpod-<id>.scope
        for podman). Host desktop processes sit under app.slice. Raises
        OSError when the file cannot be read (pid exited, no access).
        The caller must not treat that as "not a container".
    """
    with open(os.path.join(_PROC_ROOT, str(pid), "cgroup")) as f:
        cgroup = f.read()
    return any(marker in cgroup for marker in CONTAINER_CGROUP_MARKERS)


def _innermost_nspid(pid: int) -> int | None:
    """
    Input:      pid - host PID
    Output:     int innermost NSpid, or None if missing or unreadable
    Details:
        Reads _PROC_ROOT/<pid>/status. The NSpid line lists the pid in
        each nested namespace, host first, innermost last. FreeToken
        registers the innermost value. Returns None on a missing line,
        an unreadable file, or a value that is not an int. A failed
        read is not a match.
    """
    path = os.path.join(_PROC_ROOT, str(pid), "status")
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("NSpid:"):
                    parts = line.split()[1:]
                    if not parts:
                        return None
                    try:
                        return int(parts[-1])
                    except ValueError:
                        logger.warning("unparseable NSpid for pid %d: %r", pid, line)
                        return None
    except OSError:
        logger.warning("cannot read NSpid for pid %d", pid)
        return None
    return None


def _proc_start_time(pid: int) -> int | None:
    """
    Input:      pid - host PID
    Output:     int starttime ticks from /proc/<pid>/stat field 22, or
                None if unreadable or unparseable
    Details:
        comm (field 2) sits in parentheses and can contain spaces and
        parentheses, so fields after comm start after the last ')'.
        Field 22 is index 19 of that list. Used to refuse SIGKILL when
        the pid number has been reused. Does not raise.
    """
    path = os.path.join(_PROC_ROOT, str(pid), "stat")
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        logger.warning("cannot read start time for pid %d", pid)
        return None
    rparen = text.rfind(")")
    if rparen < 0:
        logger.warning("cannot parse start time for pid %d: no comm close paren", pid)
        return None
    fields = text[rparen + 1:].split()
    if len(fields) < 20:
        logger.warning(
            "cannot parse start time for pid %d: only %d fields after comm",
            pid, len(fields),
        )
        return None
    try:
        return int(fields[19])
    except ValueError:
        logger.warning("cannot parse start time for pid %d: %r", pid, fields[19])
        return None


def _gpu_proc_matches_registered(host_pid: int, registered_pid: int) -> bool:
    """
    Input:      host_pid - nvidia-smi compute-apps pid (host namespace);
                registered_pid - pid stored on the lease
    Output:     True if this GPU process is the registered process
    Details:
        Matches a host pid (clients that share the host pid namespace)
        or the innermost NSpid (clients such as FreeToken that register
        a container-namespace pid).
    """
    if host_pid == registered_pid:
        return True
    nspid = _innermost_nspid(host_pid)
    return nspid is not None and nspid == registered_pid


def _resolve_container_gpu_target(
    registered_pid: int, live_targets: list[dict],
) -> tuple[dict | None, str]:
    """
    Input:      registered_pid - pid stored on the lease;
                live_targets - fresh compute-apps list
    Output:     (target, skip_reason). target is the unique container
                GPU process to signal, or None. skip_reason is empty
                when target is set, else a log phrase
    Details:
        Candidates are GPU compute processes that run in a container
        and whose host pid or innermost NSpid equals registered_pid.
        Exactly one candidate is the target. Zero candidates log
        "no container GPU match", or "refusing to signal host process"
        when a matching GPU pid is not in a container. Two or more
        log "ambiguous". A cgroup read error does not count as host
        or container.
    """
    candidates = []
    host_gpu_match = False
    for proc in live_targets:
        host_pid = proc["pid"]
        if not _gpu_proc_matches_registered(host_pid, registered_pid):
            continue
        try:
            in_container = _pid_in_container(host_pid)
        except OSError:
            logger.warning(
                "watchdog: cannot read cgroup for gpu pid %d; not a kill candidate",
                host_pid,
            )
            continue
        if not in_container:
            host_gpu_match = True
            logger.warning("refusing to signal host process pid %d", host_pid)
            continue
        candidates.append(proc)
    if len(candidates) == 1:
        return candidates[0], ""
    if len(candidates) > 1:
        return None, "ambiguous"
    if host_gpu_match:
        return None, "refusing to signal host process"
    return None, "no container GPU match"


def _pid_is_own_steam_proton_wrapper(pid_name: str) -> bool:
    """
    Input:      pid_name - directory name under _PROC_ROOT
    Output:     True if this pid is a Steam Proton wrapper owned by
                this process's euid
    Details:
        Raises OSError when the pid cannot be read. Caller skips that
        pid. The broker runs in a rootless user namespace with pid: host.
        The desktop user's processes map to the container's euid
        (verified: host user procs show uid 0 inside the container, all
        other users show 65534). Other local users therefore cannot
        trigger the block.
    """
    proc_dir = os.path.join(_PROC_ROOT, pid_name)
    with open(os.path.join(proc_dir, "comm"), "rb") as f:
        if f.read().strip() != b"steam-launch-wr":
            return False
    with open(os.path.join(proc_dir, "cmdline"), "rb") as f:
        argv = f.read().split(b"\0")
    if not argv or not argv[0]:
        return False
    if not (
        os.path.basename(argv[0]) == b"steam-launch-wrapper"
        and b"SteamLaunch" in argv
        and any(arg.startswith(b"AppId=") and arg[6:].isdigit() for arg in argv)
        and any(os.path.basename(arg) == b"proton" for arg in argv)
    ):
        return False
    st = os.stat(proc_dir)
    if st.st_uid != os.geteuid():
        pid = int(pid_name)
        if pid not in _foreign_proton_warned_pids:
            logger.warning(
                "ignoring Steam Proton wrapper pid %s: owner uid %s is not this process uid %s",
                pid_name, st.st_uid, os.geteuid(),
            )
            _foreign_proton_warned_pids.add(pid)
        return False
    return True


def _steam_proton_running() -> bool:
    """
    Input:      None
    Output:     True if a Steam Proton game launch wrapper is running
    Details:
        Steam keeps `steam-launch-wrapper SteamLaunch AppId=N -- .../proton`
        alive until the game exits (observed in gameprocess_log.txt).
        Read /proc on the host PID namespace. Do not match Steam itself
        or steamwebhelper. A process that exits during the scan is normal.
        Per-pid OSError skips that pid. Failure to list _PROC_ROOT still
        raises OSError so _refresh_proton_state can mark the scan uncertain.
    """
    live_pids: set[int] = set()
    found = False
    with os.scandir(_PROC_ROOT) as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            live_pids.add(pid)
            try:
                if _pid_is_own_steam_proton_wrapper(entry.name):
                    found = True
            except OSError as exc:
                logger.warning(
                    "Steam process %s skipped during Proton scan: %s",
                    entry.name, exc,
                )
    _foreign_proton_warned_pids.intersection_update(live_pids)
    return found


def _container_gpu_mb() -> float:
    """
    Input:      None
    Output:     float - total GPU memory in MB held by compute processes
                that run inside a container
    Details:
        Synchronous; run it in a thread. Every managed service (FreeToken,
        Whisper, Flux, the Ollama runner) runs in a container. Games and
        the desktop are host processes. So nvidia-smi used minus this
        value is desktop plus game, independent of lease sizes. Raises
        if nvidia-smi fails or any listed pid's cgroup cannot be read.
        The caller then falls back to the lease-based formula.
    """
    return sum(
        proc["used_mb"]
        for proc in _list_gpu_compute_processes()
        if _pid_in_container(proc["pid"])
    )


def _kill_process(pid: int, process_name: str, sig: int = signal.SIGTERM) -> str:
    """
    Input:      pid - host PID to signal; process_name - for logging only;
                sig - SIGTERM (default) or SIGKILL
    Output:     str - one of "terminated", "killed", "already_exited",
                "permission_denied"
    Details:
        Sends sig. _kill_targets_with_grace sends SIGTERM first, then
        SIGKILL to survivors whose start time is unchanged. Never
        raises. Already gone and no-permission are normal bulk-kill
        outcomes. The return value reports each one.
    """
    try:
        sig_name = signal.Signals(sig).name
    except ValueError:
        logger.warning("unknown signal %s for pid %d (%s)", sig, pid, process_name)
        sig_name = str(sig)
    try:
        os.kill(pid, sig)
        if sig == signal.SIGKILL:
            return "killed"
        return "terminated"
    except ProcessLookupError:
        logger.info("pid %d (%s) already exited before %s", pid, process_name, sig_name)
        return "already_exited"
    except PermissionError:
        logger.warning(
            "no permission to signal pid %d (%s) with %s -- owned by a different user "
            "than this container's rootless UID mapping", pid, process_name, sig_name,
        )
        return "permission_denied"


def _process_alive(pid: int) -> bool:
    """
    Input:      pid - host PID to check
    Output:     bool - True if the process still exists and is signalable
    Details:
        Uses signal 0 (no-op signal) to check liveness without sending a
        real signal. That is the standard os.kill idiom for this. A
        PermissionError still means the process exists (not ours to
        touch), so it counts as alive for the SIGKILL sweep in admin_clear.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        logger.warning("pid %d exited before liveness check", pid)
        return False
    except PermissionError:
        logger.warning("no permission to check liveness of pid %d", pid)
        return True


async def _live_used_mb() -> float:
    """
    Input:      None
    Output:     float - current GPU memory used in MB
    Details:
        Async wrapper around nvidia-smi memory.used. Runs in a thread so
        it does not block the event loop. Logs and re-raises on failure.
        Live usage is not a gate. /acquire does not call this.
    """
    try:
        return await asyncio.to_thread(_nvidia_smi_query, "memory.used")
    except Exception:
        logger.exception("nvidia-smi memory.used query failed")
        raise


def _record_used_mb(used_mb: float) -> None:
    """
    Input:      used_mb - nvidia-smi memory.used in MB
    Output:     None
    Details:
        Writes _last_used_mb and _last_used_at (monotonic). The
        unmanaged and non-lease pollers call this after a successful
        memory.used read.
    """
    global _last_used_mb, _last_used_at
    _last_used_mb = used_mb
    _last_used_at = time.monotonic()


def _fresh_cached_used_mb(now: float | None = None) -> float | None:
    """
    Input:      now - time.monotonic(); defaults to time.monotonic()
    Output:     cached used MB, or None if missing or older than
                2 * UNMANAGED_POLL_INTERVAL_SECONDS
    Details:
        /status uses this to skip a live nvidia-smi call when a
        poller reading is still fresh. Acquire uses the same age
        limit for the over-usage warning.
    """
    if _last_used_mb is None or _last_used_at is None:
        return None
    if now is None:
        now = time.monotonic()
    if now - _last_used_at <= 2 * UNMANAGED_POLL_INTERVAL_SECONDS:
        return _last_used_mb
    return None


def _atomic_write_json(path: str, payload) -> None:
    """
    Input:      path - destination file; payload - JSON-serializable object
    Output:     None
    Details:
        Writes compact JSON through a unique temp file in the same
        directory (tempfile.mkstemp, O_EXCL, mode 0600), flush+fsync,
        then os.replace onto path. On any exception, removes the temp
        file (logs a cleanup failure) and re-raises. Callers keep their
        own error handling.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("atomic JSON write to %s failed", path)
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.exception("failed to remove temp file %s after write error", tmp_path)
        raise


def _persist_ledger_sync():
    """
    Input:      None
    Output:     None
    Details:
        Synchronous ledger persistence to disk. Delegates to
        _atomic_write_json so the write is unique-temp + fsync +
        os.replace onto LEDGER_PATH.
    """
    _atomic_write_json(LEDGER_PATH, {
        "leases": _leases,
        "proton_game_active": _proton_game_active or _proton_game_uncertain,
        "proton_game_cleared_at": _proton_game_cleared_at,
    })


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


def _persist_non_lease_sync(samples: list[dict]) -> None:
    """
    Input:      samples - snapshot of {ts, mb, locked} dicts to write
    Output:     None
    Details:
        Delegates to _atomic_write_json so a crash mid-write cannot
        leave a truncated JSON file at NON_LEASE_USAGE_PATH. Caller
        runs this in a thread.
    """
    _atomic_write_json(NON_LEASE_USAGE_PATH, {"samples": samples})


def _percentile(values: list[float], p: float) -> float:
    """
    Input:      values - non-empty list of sample MB; p - percentile in
                [0, 100]
    Output:     float - linearly interpolated percentile
    Details:
        Sorts a copy. Rank is p/100 * (n-1). Interpolates between adjacent
        ranks so p95 of a short list is not forced to max().
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _safety_margin_mb() -> float:
    """
    Input:      None
    Output:     float - live admission safety margin in MB
    Details:
        Returns the computed unlocked-window percentile when enough
        samples exist, otherwise GPU_SAFETY_MARGIN_MB. Always at least
        GPU_SAFETY_MARGIN_MIN_MB. Replaces every former read of the
        SAFETY_MARGIN_MB constant in admission and /status.
    """
    computed = _computed_margin_mb
    base = float(SAFETY_MARGIN_MB) if computed is None else float(computed)
    return max(base, float(SAFETY_MARGIN_MIN_MB))


def _dbus_variant_value(item):
    """
    Input:      item - jeepney Properties.Get body element
    Output:     unwrapped Python value
    Details:
        jeepney returns D-Bus variants as (signature, value) tuples, for
        example ('b', False). Plain values pass through unchanged.
    """
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[1]
    return item


def _host_euid() -> int:
    """
    Input:      None
    Output:     int - UID dbus AUTH EXTERNAL must send
    Details:
        dbus-daemon compares AUTH EXTERNAL's hex UID with SO_PEERCRED.
        Rootless podman maps container euid 0 to the launching host uid
        (uid_map "0 1000 1" on this host). os.geteuid() is 0 inside the
        container and is REJECTED. This returns the mapped host uid.
    """
    euid = os.geteuid()
    try:
        with open("/proc/self/uid_map") as f:
            first = f.readline().split()
        if len(first) != 3:
            return euid
        inner, outer, length = int(first[0]), int(first[1]), int(first[2])
        if inner <= euid < inner + length:
            return outer + (euid - inner)
    except (OSError, ValueError) as exc:
        logger.warning("failed to map container euid for dbus AUTH EXTERNAL: %s", exc)
    return euid


def _open_system_bus(timeout: float = 2.0):
    """
    Input:      timeout - socket timeout in seconds for connect and auth
    Output:     jeepney DBusConnection (caller must close)
    Details:
        jeepney's open_dbus_connection() uses os.geteuid() for AUTH
        EXTERNAL. Rootless podman makes that 0 while SO_PEERCRED is the
        mapped host uid, so the bus replies REJECTED EXTERNAL. This
        handshake sends the uid_map-translated uid instead. Confirmed
        live inside gpubroker-gpubroker-1: AUTH with 1000 returns OK.
    """
    import socket
    from binascii import hexlify
    from jeepney.auth import BEGIN, AuthenticationError
    from jeepney.bus import get_bus
    from jeepney.io.blocking import DBusConnection

    addr = get_bus("SYSTEM")
    sock = socket.socket(family=socket.AF_UNIX)
    sock.settimeout(timeout)
    try:
        sock.connect(addr)
        sock.send(b"\0")
        uid_hex = hexlify(str(_host_euid()).encode("ascii"))
        sock.sendall(b"AUTH EXTERNAL %b\r\n" % uid_hex)
        data = sock.recv(1024)
        if not data.startswith(b"OK "):
            raise AuthenticationError(data)
        sock.sendall(BEGIN)
        sock.settimeout(None)
        return DBusConnection(sock)
    except Exception:
        logger.exception("system bus AUTH EXTERNAL handshake failed")
        sock.close()
        raise



def _read_lock_state() -> str:
    """
    Input:      None
    Output:     str - "locked", "unlocked", or "unknown"
    Details:
        Queries org.freedesktop.login1 Session.LockedHint on the seated
        graphical session over the system bus. systemd 258 session_save()
        never writes LOCKED_HINT to /run/systemd/sessions/<id> (verified
        against v258.7 source and against this host's live session files),
        so the file-parse path cannot see lock state here. jeepney is a
        pure-Python D-Bus client and keeps the slim image off libdbus.
        Rootless podman AUTH EXTERNAL uses _open_system_bus (uid_map),
        not jeepney's default geteuid handshake. Any failure tags the
        sample unknown so the poller still records overhead.
    """
    global _jeepney_import_logged
    try:
        from jeepney import DBusAddress, new_method_call, Properties
        from jeepney.wrappers import unwrap_msg

    except ImportError as exc:
        if not _jeepney_import_logged:
            logger.warning("jeepney not installed; tagging lock-state unknown: %s", exc)
            _jeepney_import_logged = True
        return "unknown"

    try:
        with _open_system_bus() as conn:
            manager = DBusAddress(
                "/org/freedesktop/login1",
                bus_name="org.freedesktop.login1",
                interface="org.freedesktop.login1.Manager",
            )
            reply = conn.send_and_get_reply(
                new_method_call(manager, "ListSessions"), timeout=2.0
            )
            sessions = unwrap_msg(reply)[0]
            seated = [entry for entry in sessions if entry[3]]
            if not seated:
                logger.warning("login1 ListSessions returned no seated session; lock-state unknown")
                return "unknown"
            chosen_path = seated[0][4]
            for _sid, _uid, _user, _seat, path in seated:
                sess = DBusAddress(
                    path,
                    bus_name="org.freedesktop.login1",
                    interface="org.freedesktop.login1.Session",
                )
                type_reply = conn.send_and_get_reply(
                    Properties(sess).get("Type"), timeout=2.0
                )
                stype = _dbus_variant_value(unwrap_msg(type_reply)[0])
                if stype in ("wayland", "x11", "mir"):
                    chosen_path = path
                    break
            sess = DBusAddress(
                chosen_path,
                bus_name="org.freedesktop.login1",
                interface="org.freedesktop.login1.Session",
            )
            locked_reply = conn.send_and_get_reply(
                Properties(sess).get("LockedHint"), timeout=2.0
            )
            locked = _dbus_variant_value(unwrap_msg(locked_reply)[0])
            if locked is True:
                return "locked"
            if locked is False:
                return "unlocked"
            logger.warning(
                "login1 LockedHint returned unexpected value %r; lock-state unknown",
                locked,
            )
            return "unknown"
    except Exception as exc:
        logger.warning("login1 LockedHint query failed; tagging lock-state unknown: %s", exc)
        return "unknown"


def _recompute_margin_locked() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Sets _computed_margin_mb from the rolling store. Prefers p95 of
        unlocked samples once NON_LEASE_MIN_SAMPLES exist, because
        admission safety matters while the user is at the desktop. If
        every sample is lock-unknown (dbus unavailable) and the count
        still meets the minimum, uses all samples rather than gating
        forever on a missing signal. Otherwise leaves None so
        _safety_margin_mb falls back to GPU_SAFETY_MARGIN_MB.

        Samples at or above UNMANAGED_YIELD_OVERHEAD_MB never enter the
        percentile. A Steam Proton game below that threshold is also
        excluded by its `game` sample tag. Otherwise a long game session
        would raise p95 for 48 hours after the game ends. Caller holds
        _lock, or runs at startup before background tasks start.
    """
    global _computed_margin_mb
    before = _safety_margin_mb()
    baseline = [
        s for s in _non_lease_samples
        if s["mb"] < UNMANAGED_YIELD_OVERHEAD_MB and not s.get("game", False)
    ]
    unlocked = [s["mb"] for s in baseline if s.get("locked") == "unlocked"]
    if len(unlocked) >= NON_LEASE_MIN_SAMPLES:
        _computed_margin_mb = _percentile(unlocked, NON_LEASE_PERCENTILE)
    else:
        known = [s for s in _non_lease_samples if s.get("locked") in ("locked", "unlocked")]
        if not known and len(baseline) >= NON_LEASE_MIN_SAMPLES:
            _computed_margin_mb = _percentile([s["mb"] for s in baseline], NON_LEASE_PERCENTILE)
        else:
            _computed_margin_mb = None
    # A lower effective margin raises the admission budget.
    if _safety_margin_mb() < before:
        _notify_waiters()



def _load_non_lease_samples() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Restores the rolling store from NON_LEASE_USAGE_PATH and drops
        samples older than NON_LEASE_WINDOW_HOURS. Recomputes the live
        margin so a restarted broker does not sit on the fallback when
        enough history already exists. Logs and starts empty on a
        missing or corrupt file.
    """
    global _computed_margin_mb
    if not os.path.exists(NON_LEASE_USAGE_PATH):
        logger.info(
            "no non-lease usage file at %s, using safety-margin fallback %.0f MB",
            NON_LEASE_USAGE_PATH, SAFETY_MARGIN_MB,
        )
        return
    try:
        with open(NON_LEASE_USAGE_PATH) as f:
            data = json.load(f)
        now = time.time()
        cutoff = now - NON_LEASE_WINDOW_HOURS * 3600.0
        loaded = []
        for raw in data.get("samples", []):
            try:
                ts = float(raw["ts"])
                mb = float(raw["mb"])
                locked = raw.get("locked", "unknown")
                if locked not in ("locked", "unlocked", "unknown"):
                    locked = "unknown"
                if ts >= cutoff:
                    loaded.append({
                        "ts": ts, "mb": mb, "locked": locked,
                        "game": raw.get("game") is True,
                    })
            except (TypeError, ValueError, KeyError):
                logger.warning("skipping malformed non-lease sample: %r", raw)
                continue
        _non_lease_samples.clear()
        _non_lease_samples.extend(loaded)
        _recompute_margin_locked()
        logger.info(
            "non-lease usage loaded: %d sample(s), margin=%.0f MB (%s)",
            len(_non_lease_samples),
            _safety_margin_mb(),
            "percentile" if _computed_margin_mb is not None else "fallback",
        )
    except (OSError, ValueError, TypeError):
        logger.exception(
            "non-lease usage file at %s corrupt/unreadable, starting empty",
            NON_LEASE_USAGE_PATH,
        )
        _non_lease_samples.clear()
        _computed_margin_mb = None


def _reserved_mb() -> float:
    """
    Input:      None
    Output:     float - total MB reserved by all active leases
    Details:
        Sums the mb field across all active leases in the in-memory ledger.
        Must be called within _lock to avoid TOCTOU.
    """
    return sum(lease["mb"] for lease in _leases.values())


def _notify_waiters() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Wake every acquire waiter so it re-checks admission. Callable
        with or without `_lock`. Sets the current Event, then replaces
        it so later waiters capture a fresh unset Event. Broadcast is
        required: wait-queue order is not FIFO sequence order, so a
        single-waker notify could wake a non-head waiter.
    """
    global _admission_event
    _admission_event.set()
    _admission_event = asyncio.Event()


def _discard_pending_locked(seq: int) -> None:
    """
    Input:      seq - acquire sequence number
    Output:     None
    Details:
        Drop seq from `_pending`. If it was queued, wake waiters so
        the new FIFO head re-checks at once. Caller holds `_lock`.
    """
    was_queued = seq in _pending
    _pending.discard(seq)
    if was_queued:
        _notify_waiters()


def _prune_pending_notifications_locked(now: float | None = None) -> None:
    """
    Input:      now - optional monotonic timestamp; default time.monotonic()
    Output:     None
    Details:
        Drop _pending_notifications entries whose created time is older
        than NOTIFICATION_TTL_SECONDS. Caller holds `_lock`.
    """
    if now is None:
        now = time.monotonic()
    expired = [
        caller
        for caller, (_text, created) in _pending_notifications.items()
        if now - created > NOTIFICATION_TTL_SECONDS
    ]
    for caller in expired:
        del _pending_notifications[caller]



def _effective_priority(base_priority: float, waited_seconds: float) -> float:
    """
    Input:      base_priority - a tier's TIER_BASE_PRIORITY value;
                waited_seconds - how long this request has been queued
    Output:     float - base_priority plus an anti-starvation aging term,
                capped at exactly base_priority + TIER_STEP (parity with
                the next tier up, never exceeding it)
    Details:
        Aging climbs linearly from 0 to TIER_STEP over
        PRIORITY_AGING_WINDOW_SECONDS of continuous waiting, then holds at
        the cap. This guarantees a same-tier (or lower-tier-holder) waiter
        eventually out-ranks an incumbent holder's (tier + INCUMBENCY_BONUS)
        -- see the INCUMBENCY_BONUS assertion in startup() for why that
        needs bonus < TIER_STEP. It does not, and must not, let a waiter
        eventually out-rank a holder a full tier or more above it: that
        would let low priority starve out high priority given enough
        waiting, defeating the point of having tiers.
    """
    aged_fraction = min(waited_seconds / PRIORITY_AGING_WINDOW_SECONDS, 1.0)
    return base_priority + aged_fraction * TIER_STEP


def _weakest_preemptible_holder(leases: dict) -> tuple[str, dict] | None:
    """
    Input:      leases - the lease ledger dict (lease_id -> lease dict)
    Output:     (lease_id, lease) for the lowest-priority preemptible
                holder, or None if no active lease has preemptible=True
    Details:
        Only preemptible=True holders are ever candidates -- the default
        (preemptible=False) means "cannot actually pause," so it is never
        a target, full stop. Ties on tier break on the oldest acquired_at
        (the longest-resident equal-tier holder is preempted first); the
        exact tiebreak is an arbitrary but deterministic choice, not a
        load-bearing part of the design.
    """
    candidates = [
        (lease_id, lease) for lease_id, lease in leases.items()
        if lease.get("preemptible", False)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (TIER_BASE_PRIORITY[item[1]["priority"]], item[1]["acquired_at"]),
    )


def _maybe_flag_for_preemption(waiter_priority: str, waited_seconds: float, leases: dict) -> tuple[str, dict] | None:
    """
    Input:      waiter_priority - the queued caller's priority tier;
                waited_seconds - how long that caller has been queued;
                leases - the lease ledger dict, mutated in place when a
                target is flagged
    Output:     (lease_id, lease) of the holder just flagged, or None if
                there is no preemptible holder, the weakest one is
                already flagged, or the waiter's aged priority does not
                yet exceed it
    Details:
        Computes the waiter's aged effective priority
        (_effective_priority) and compares it against the weakest current
        preemptible holder's (tier + INCUMBENCY_BONUS,
        _weakest_preemptible_holder). Flags that holder
        (lease["preempt_flagged_at"] = time.monotonic(),
        lease["preempt_reason"] = "fairness") exactly once --
        an already-flagged holder is left alone so its grace-period timer
        never resets just because another poll cycle re-evaluates the
        same comparison. Must be called while holding _lock (mutates the
        shared ledger).
    """
    target = _weakest_preemptible_holder(leases)
    if target is None:
        return None
    target_id, target_lease = target
    if target_lease.get("preempt_flagged_at") is not None:
        return None
    my_effective_priority = _effective_priority(TIER_BASE_PRIORITY[waiter_priority], waited_seconds)
    holder_value = TIER_BASE_PRIORITY[target_lease["priority"]] + INCUMBENCY_BONUS
    if my_effective_priority <= holder_value:
        return None
    target_lease["preempt_flagged_at"] = time.monotonic()
    target_lease["preempt_reason"] = "fairness"
    return target_id, target_lease


def _classify_reclaim_reasons(lease: dict, now: float) -> list[str]:
    """
    Input:      lease - a lease dict (or snapshot copy); now - a
                time.monotonic() reading to compare against
    Output:     list[str] - zero or more of "heartbeat_stale", "lease_age",
                "preempt_grace_expired"; empty means the lease is fine
    Details:
        A lease can trigger more than one reason at once (e.g. it aged out
        and has also gone stale) -- every reason that applies is returned,
        not just the first one checked, so _reclaim_broken_lease's
        caller-facing notice can name all of them.
    """
    reasons = []
    stale = now - lease["last_seen"] > LEASE_TTL_SECONDS
    if stale:
        reasons.append("heartbeat_stale")
    if (
        now - lease["acquired_at"] > MAX_LEASE_AGE_SECONDS
        and (not lease.get("preemptible", False) or stale)
    ):
        reasons.append("lease_age")
    flagged_at = lease.get("preempt_flagged_at")
    if flagged_at is not None and now - flagged_at > PREEMPT_GRACE_SECONDS:
        reasons.append("preempt_grace_expired")
    return reasons


async def _kill_targets_with_grace(targets: list[dict]) -> list[dict]:
    """
    Input:      targets - list of {"pid": int, "process_name": str, ...};
                optional "start_time" is the /proc/<pid>/stat field 22
                captured at target resolution. Other keys are preserved.
    Output:     list[dict] - the same list, each with "signal_result" set
                to "terminated", "already_exited", "permission_denied", or
                "killed"
    Details:
        Shared SIGTERM-then-SIGKILL sequence used by admin_clear() and
        the watchdog kill path in _reclaim_broken_lease(). After SIGTERM,
        poll liveness every ADMIN_KILL_POLL_SECONDS and return as soon
        as every signaled target has exited, up to
        ADMIN_KILL_GRACE_SECONDS. Before SIGKILL, re-read start time.
        Skip SIGKILL if it changed, the process is gone, or no start
        time was captured.
    """
    for target in targets:
        target["signal_result"] = await asyncio.to_thread(
            _kill_process, target["pid"], target["process_name"],
        )

    pending = [t for t in targets if t["signal_result"] == "terminated"]
    if not pending:
        return targets

    deadline = time.monotonic() + ADMIN_KILL_GRACE_SECONDS
    while pending:
        still = []
        for target in pending:
            if await asyncio.to_thread(_process_alive, target["pid"]):
                still.append(target)
            else:
                target["signal_result"] = "terminated"
        pending = still
        if not pending:
            return targets
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(ADMIN_KILL_POLL_SECONDS, remaining))

    for target in pending:
        expected = target.get("start_time")
        current = _proc_start_time(target["pid"])
        if expected is None or current is None or current != expected:
            logger.warning(
                "skipping SIGKILL for pid %d (%s): start time changed or process gone "
                "(expected %r, now %r)",
                target["pid"], target["process_name"], expected, current,
            )
            continue
        result = await asyncio.to_thread(
            _kill_process, target["pid"], target["process_name"], signal.SIGKILL,
        )
        target["signal_result"] = result
        if result == "killed":
            logger.warning(
                "SIGKILL sent to pid %d (%s), SIGTERM ignored",
                target["pid"], target["process_name"],
            )
        elif result == "already_exited":
            target["signal_result"] = "terminated"
    return targets


async def _reclaim_broken_lease(lease_id: str, lease: dict, reasons: list[str]) -> None:
    """
    Input:      lease_id - str; lease - a snapshot dict of the lease at the
                moment it was flagged (caller, mb, acquired_at, last_seen,
                priority, preemptible, pid, preempt_flagged_at); reasons -
                non-empty list[str] from _classify_reclaim_reasons
    Output:     None
    Details:
        Shared watchdog reclaim path for every trigger in
        _sweep_expired_leases. Runs outside _lock except for the
        pre-signal recheck and the final ledger mutation.

        No pid: ledger-only delete. Otherwise list GPU compute
        processes and resolve a unique container candidate whose host
        pid or innermost NSpid equals the registered pid. Zero
        candidates or more than one stay ledger-only (no container GPU
        match, refusing to signal host process, or ambiguous). A unique
        match reads start time, then under _lock confirms the lease
        still exists, still has the same registered pid, and still has
        a reclaim reason. If not, abort with no ledger change. Release
        the lock, then SIGTERM/SIGKILL via _kill_targets_with_grace.
        Verifies the host pid is gone afterward. The check is
        tri-state: cleared (WARNING confirmed cleared), still present
        (ERROR), or unverified when the process query raises
        (WARNING could not verify). Removes the ledger entry either way.

        Queues a one-shot notice for a reclaimed GPU lease. A zero-size
        admission token holds no GPU memory, so its reclaim needs no
        caller-facing GPU warning.
    """
    reason_text = " and ".join(_RECLAIM_REASON_TEXT[r] for r in reasons)
    pid = lease.get("pid")

    if pid is None:
        logger.warning(
            "watchdog reclaiming lease %r (caller=%r, mb=%.0f) via ledger-only delete -- %s, "
            "and no pid was reported at acquire time so no real kill is possible",
            lease_id, lease["caller"], lease["mb"], reason_text,
        )
    else:
        try:
            live_targets = await asyncio.to_thread(_list_gpu_compute_processes)
        except Exception:
            logger.exception(
                "watchdog: failed to list GPU compute processes while reclaiming lease %r",
                lease_id,
            )
            live_targets = []
        match, skip_reason = _resolve_container_gpu_target(pid, live_targets)

        if match is None:
            logger.warning(
                "watchdog reclaiming lease %r (caller=%r, mb=%.0f) via ledger-only delete -- %s, "
                "%s (registered pid %d)",
                lease_id, lease["caller"], lease["mb"], reason_text, skip_reason, pid,
            )
        else:
            host_pid = match["pid"]
            start_time = _proc_start_time(host_pid)
            async with _lock:
                current = _leases.get(lease_id)
                if current is None:
                    logger.warning(
                        "watchdog: aborting kill for lease %r; lease was released",
                        lease_id,
                    )
                    return
                if current.get("pid") != pid:
                    logger.warning(
                        "watchdog: aborting kill for lease %r; registered pid changed",
                        lease_id,
                    )
                    return
                if not _classify_reclaim_reasons(current, time.monotonic()):
                    logger.warning(
                        "watchdog: aborting kill for lease %r; lease is no longer reclaimable",
                        lease_id,
                    )
                    return
            logger.warning(
                "watchdog killing pid %d (%s) for lease %r (caller=%r, mb=%.0f): %s",
                host_pid, match["process_name"], lease_id, lease["caller"],
                lease["mb"], reason_text,
            )
            killed = await _kill_targets_with_grace([{
                "pid": host_pid,
                "process_name": match["process_name"],
                "start_time": start_time,
            }])
            verification = "unverified"
            try:
                after_targets = await asyncio.to_thread(_list_gpu_compute_processes)
                if any(t["pid"] == host_pid for t in after_targets):
                    verification = "still present"
                else:
                    verification = "cleared"
            except Exception:
                logger.exception(
                    "watchdog: failed to verify pid %d cleared after kill for lease %r",
                    host_pid, lease_id,
                )
            if verification == "still present":
                logger.error(
                    "watchdog: pid %d (%s) for lease %r (caller=%r) still holds GPU memory after "
                    "the kill sequence (%s) -- removing the ledger entry anyway, real GPU state and "
                    "the ledger now disagree",
                    host_pid, match["process_name"], lease_id, lease["caller"],
                    killed[0]["signal_result"],
                )
            elif verification == "cleared":
                logger.warning(
                    "watchdog: pid %d (%s) for lease %r (caller=%r) confirmed cleared after kill (%s)",
                    host_pid, match["process_name"], lease_id, lease["caller"],
                    killed[0]["signal_result"],
                )
            else:
                logger.warning(
                    "watchdog: pid %d (%s) for lease %r (caller=%r) could not verify after kill (%s)",
                    host_pid, match["process_name"], lease_id, lease["caller"],
                    killed[0]["signal_result"],
                )

    async with _lock:
        current = _leases.pop(lease_id, None)
        if current is not None:
            if _caller_to_lease.get(current["caller"]) == lease_id:
                del _caller_to_lease[current["caller"]]
            _prune_pending_notifications_locked()
            if current["mb"] > 0:
                _pending_notifications[current["caller"]] = (
                    f"your previous lease was force-terminated by the watchdog after {reason_text}; "
                    "if you hold state across runs, verify its integrity before resuming.",
                    time.monotonic(),
                )
            await _persist_ledger()
            _notify_waiters()



async def _poll_ollama():
    """
    Input:      None
    Output:     None
    Details:
        Infinite loop polling Ollama GET /api/ps every
        OLLAMA_POLL_INTERVAL_SECONDS. Updates global _ollama_mb. Logs
        when the total footprint changes. Holds last-known-good value
        on transient failures.
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
                async with _lock:
                    decreased = total < _ollama_mb
                    _ollama_mb = total
                    if decreased:
                        _notify_waiters()
                if total != last_value:
                    logger.info("ollama /api/ps: %.0f MB across %d model(s)", total, len(models))
                    last_value = total

            except Exception as exc:
                logger.warning("ollama /api/ps poll failed, holding last-known-good value (%.0f MB): %s",
                                _ollama_mb, exc)
            await asyncio.sleep(OLLAMA_POLL_INTERVAL_SECONDS)



async def _poll_non_lease_usage():
    """
    Input:      None
    Output:     None
    Details:
        Every NON_LEASE_POLL_INTERVAL_SECONDS, read live nvidia-smi
        used MB and subtract leases plus Ollama. Tag the sample with
        lock state and Steam Proton game state. Append it to the
        rolling store, drop old samples, and recompute the margin.
        Persist the store every NON_LEASE_PERSIST_EVERY samples.
        Record memory.used in _last_used_mb. Hold the last-known-good
        margin on failure.
    """
    samples_since_persist = 0
    while True:
        try:
            used_mb = await _live_used_mb()
            _record_used_mb(used_mb)
            locked = await asyncio.to_thread(_read_lock_state)
            now = time.time()
            async with _lock:
                tracked = _reserved_mb() + _ollama_mb
                overhead = used_mb - tracked
                if overhead < 0.0:
                    overhead = 0.0
                _non_lease_samples.append({
                    "ts": now, "mb": overhead, "locked": locked,
                    "game": _proton_game_active or _proton_game_uncertain,
                })
                cutoff = now - NON_LEASE_WINDOW_HOURS * 3600.0
                _non_lease_samples[:] = [
                    s for s in _non_lease_samples if s["ts"] >= cutoff
                ]
                _recompute_margin_locked()
                snapshot = list(_non_lease_samples)
                margin = _safety_margin_mb()
                source = (
                    "percentile" if _computed_margin_mb is not None else "fallback"
                )
            logger.info(
                "non-lease poll: used=%.0f tracked=%.0f overhead=%.0f locked=%s "
                "margin=%.0f source=%s samples=%d",
                used_mb, tracked, overhead, locked, margin, source, len(snapshot),
            )
            samples_since_persist += 1
            if samples_since_persist >= NON_LEASE_PERSIST_EVERY:
                try:
                    await asyncio.to_thread(_persist_non_lease_sync, snapshot)
                    samples_since_persist = 0
                except OSError as exc:
                    logger.warning(
                        "failed to persist non-lease usage to %s: %s",
                        NON_LEASE_USAGE_PATH, exc,
                    )
        except Exception as exc:
            logger.warning(
                "non-lease usage poll failed, holding last-known-good margin (%.0f MB): %s",
                _safety_margin_mb(), exc,
            )
        await asyncio.sleep(NON_LEASE_POLL_INTERVAL_SECONDS)



def _update_unmanaged_state(overhead_mb: float, now: float) -> bool:
    """
    Input:      overhead_mb - unmanaged usage right now (live nvidia-smi
                used minus container GPU processes, see
                _poll_unmanaged_load); now - time.monotonic()
    Output:     bool - the new _unmanaged_heavy value
    Details:
        Hysteresis state machine. Enters heavy after
        UNMANAGED_ENTER_SAMPLES consecutive polls at or above
        UNMANAGED_YIELD_OVERHEAD_MB. Leaves only after the overhead has
        stayed below it for UNMANAGED_CLEAR_SECONDS; any over-threshold
        poll restarts that clock. Pure state, no I/O. Caller holds _lock.
    """
    global _unmanaged_heavy, _unmanaged_over_count, _unmanaged_clear_since
    if overhead_mb >= UNMANAGED_YIELD_OVERHEAD_MB:
        _unmanaged_over_count += 1
        _unmanaged_clear_since = None
        if not _unmanaged_heavy and _unmanaged_over_count >= UNMANAGED_ENTER_SAMPLES:
            _unmanaged_heavy = True
            logger.warning(
                "unmanaged GPU load detected (%.0f MB unmanaged, threshold %.0f MB); "
                "flagging preemptible leases and refusing all new acquires",
                overhead_mb, UNMANAGED_YIELD_OVERHEAD_MB,
            )
    else:
        _unmanaged_over_count = 0
        if _unmanaged_heavy:
            if _unmanaged_clear_since is None:
                _unmanaged_clear_since = now
            elif now - _unmanaged_clear_since >= UNMANAGED_CLEAR_SECONDS:
                _unmanaged_heavy = False
                _unmanaged_clear_since = None
                logger.info(
                    "unmanaged GPU load gone for %.0fs (%.0f MB unmanaged); "
                    "acquires allowed again",
                    UNMANAGED_CLEAR_SECONDS, overhead_mb,
                )
    return _unmanaged_heavy


def _flag_preemptible_leases_locked(now: float, reason: str) -> list[str]:
    """
    Input:      now - time.monotonic(); reason - 'proton' or 'unmanaged'
    Output:     list[str] - lease ids newly flagged on this call
    Details:
        Sets preempt_flagged_at and preempt_reason on every preemptible
        lease not already flagged, so each holder's next heartbeat
        carries preempt=True with source-matching text. An already-flagged
        lease is left alone so its PREEMPT_GRACE_SECONDS timer never
        resets and its original reason is kept. Same flag the fairness
        path uses; the grace expiry in _sweep_expired_leases is the
        backstop. Caller holds _lock.
    """
    newly = []
    for lease_id, lease in _leases.items():
        if lease.get("preemptible", False) and lease.get("preempt_flagged_at") is None:
            lease["preempt_flagged_at"] = now
            lease["preempt_reason"] = reason
            newly.append(lease_id)
    return newly


def _proton_retry_after_seconds() -> int:
    """
    Input:      None
    Output:     seconds to wait for Proton's block; zero if clear
    Details:
        Caller holds _lock. An active game stays blocked. A game that
        ended stays blocked for at least PROTON_CLEAR_SECONDS.
    """
    if _proton_game_uncertain:
        return 5
    if _proton_game_active:
        return PROTON_CLEAR_SECONDS
    if _proton_game_cleared_at is None:
        return 0
    return max(0, math.ceil(_proton_game_cleared_at + PROTON_CLEAR_SECONDS - time.time()))


async def _refresh_proton_state() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Scan before each acquire and on a one-second tick. Persist the
        start and end transitions so a broker restart cannot erase the
        30-minute cooldown. A failed scan blocks admission until it works.
    """
    global _proton_game_active, _proton_game_uncertain, _proton_game_cleared_at
    try:
        active = await asyncio.to_thread(_steam_proton_running)
    except OSError:
        logger.exception("Steam Proton process scan failed; refusing new GPU work")
        async with _lock:
            if not _proton_game_uncertain:
                _proton_game_uncertain = True
                await _persist_ledger()
            _flag_preemptible_leases_locked(time.monotonic(), "proton")
        return
    async with _lock:
        changed = False
        if active:
            if not _proton_game_active or _proton_game_uncertain:
                logger.warning("Steam Proton game detected; refusing GPU work")
                changed = True
            _proton_game_active = True
            _proton_game_uncertain = False
            _proton_game_cleared_at = None
        elif _proton_game_active or _proton_game_uncertain:
            _proton_game_active = False
            _proton_game_uncertain = False
            _proton_game_cleared_at = time.time()
            logger.info("Steam Proton game exited; blocking new GPU work for %ds", PROTON_CLEAR_SECONDS)
            changed = True
        if changed:
            await _persist_ledger()
        if _proton_retry_after_seconds():
            for lease_id in _flag_preemptible_leases_locked(time.monotonic(), "proton"):
                logger.warning("flagged lease=%s for Steam Proton game", lease_id)


async def _poll_proton_games() -> None:
    """
    Input:      None
    Output:     None
    Details:    Scan once per second so a new game preempts active holders.
    """
    while True:
        try:
            await _refresh_proton_state()
        except Exception:
            logger.exception("Steam Proton poll failed; retaining the previous state")
        await asyncio.sleep(PROTON_POLL_INTERVAL_SECONDS)


async def _poll_unmanaged_load():
    """
    Input:      None
    Output:     None
    Details:
        Infinite loop, every UNMANAGED_POLL_INTERVAL_SECONDS. Reads live
        nvidia-smi usage and subtracts the GPU memory of compute processes
        that run inside a container (_container_gpu_mb). What is left is
        desktop plus game. Leases are sized above real use, so the
        lease-based formula of _poll_non_lease_usage can hide a game
        behind a loaded model. If the process list or any cgroup cannot
        be read, this poll falls back to that lease-based formula instead
        of counting the unknown process as unmanaged. Advances the
        heavy-load state machine and, while heavy, flags every
        preemptible lease. Separate from the 30s sampling loop so
        detection latency is seconds, not half a minute, without
        rewriting the persisted sample file that often. Records
        memory.used in _last_used_mb. Holds the previous state on a
        failed poll.
    """
    global _unmanaged_overhead_mb
    while True:
        try:
            used_mb = await _live_used_mb()
            _record_used_mb(used_mb)
            try:
                container_mb = await asyncio.to_thread(_container_gpu_mb)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning(
                    "container GPU process scan failed, using lease-based overhead for this poll: %s", exc,
                )
                container_mb = None
            now = time.monotonic()
            async with _lock:
                if container_mb is None:
                    overhead = max(used_mb - (_reserved_mb() + _ollama_mb), 0.0)
                else:
                    overhead = max(used_mb - container_mb, 0.0)
                _unmanaged_overhead_mb = overhead
                heavy = _update_unmanaged_state(overhead, now)
                if heavy:
                    flagged = _flag_preemptible_leases_locked(now, "unmanaged")
                    for lease_id in flagged:
                        logger.warning(
                            "flagged lease=%s (caller=%s) for preemption: unmanaged GPU load",
                            lease_id, _leases[lease_id]["caller"],
                        )
        except Exception as exc:
            logger.warning("unmanaged-load poll failed, holding state (heavy=%s): %s", _unmanaged_heavy, exc)
        await asyncio.sleep(UNMANAGED_POLL_INTERVAL_SECONDS)


async def _sweep_expired_leases():
    """
    Input:      None
    Output:     None
    Details:
        Infinite loop that runs every SWEEP_INTERVAL_SECONDS. Three
        independent triggers feed the same reclaim path (see
        _classify_reclaim_reasons and _reclaim_broken_lease):
          - heartbeat staleness: no heartbeat within LEASE_TTL_SECONDS
          - total lease age: acquired_at exceeds
            GPU_BROKER_MAX_LEASE_AGE_SECONDS for a non-preemptible lease
            or a stale preemptible lease. Fresh preemptible leases remain
            available for fairness preemption.
          - preemption grace expiry: a lease flagged for fairness
            preemption (see acquire()) that did not self-evict within
            GPU_BROKER_PREEMPT_GRACE_SECONDS.

        Candidates are snapshotted under _lock, then the kill-and-verify
        sequence in _reclaim_broken_lease runs OUTSIDE the lock -- it can
        block for seconds (a subprocess call plus ADMIN_KILL_GRACE_SECONDS),
        and holding the broker's single global lock for that long would
        stall every other /acquire, /release, and /heartbeat call in the
        meantime. Each candidate's own ledger removal then happens under a
        fresh, short _lock acquisition inside _reclaim_broken_lease.

        A classify exception for one lease is logged with that lease id;
        that lease is skipped and remaining leases in the same pass still
        become candidates. Snapshot exceptions outside per-lease work
        (for example acquiring the lock) are logged with logger.exception;
        the loop then sleeps and continues. A reclaim exception for one
        lease is logged with that lease id; remaining candidates in the
        same pass still run. asyncio.CancelledError is not caught, so
        shutdown still stops the task.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            now = time.monotonic()
            async with _lock:
                candidates = []
                for lease_id, lease in _leases.items():
                    try:
                        reasons = _classify_reclaim_reasons(lease, now)
                    except Exception:
                        logger.exception("watchdog sweep: classify failed for lease %s", lease_id)
                        continue
                    if reasons:
                        candidates.append((lease_id, dict(lease), reasons))
        except Exception:
            logger.exception("watchdog sweep: snapshot failed")
            continue
        for lease_id, lease_snapshot, reasons in candidates:
            try:
                await _reclaim_broken_lease(lease_id, lease_snapshot, reasons)
            except Exception:
                logger.exception("watchdog sweep: reclaim failed for lease %s", lease_id)


def _validate_priority_config() -> None:
    """
    Input:      None
    Output:     None
    Details:
        Raises AssertionError if INCUMBENCY_BONUS is not in [0, TIER_STEP).
        Correctness requirement, not a tunable to relax casually: the
        aging formula caps a waiting request's effective priority at
        parity with the next tier up (see _effective_priority). For that
        cap to guarantee a same-tier (or lower-tier-holder) waiter
        eventually exceeds an incumbent's (tier + INCUMBENCY_BONUS) value
        -- the anti-starvation property this whole mechanism exists for
        -- the bonus must be strictly smaller than one tier-step. A bonus
        >= TIER_STEP would let a holder's incumbency alone out-rank a
        fully aged same-tier waiter forever, i.e. real starvation. Called
        from startup() so the server fails to start rather than silently
        run with a config that can starve a caller. Factored out as its
        own function so it can be unit-tested without touching nvidia-smi,
        the ledger, or background tasks.
    """
    if not (0 <= INCUMBENCY_BONUS < TIER_STEP):
        raise AssertionError(
            f"GPU_BROKER_INCUMBENCY_BONUS ({INCUMBENCY_BONUS}) must be >= 0 and strictly less "
            f"than TIER_STEP ({TIER_STEP}), or aging can never let a waiter preempt an "
            "equal-or-lower-tier incumbent -- see _validate_priority_config's docstring"
        )


async def startup():
    """
    Input:      None
    Output:     None
    Details:
        FastAPI startup hook. Asserts the priority-preemption config is
        sane (see _validate_priority_config), queries GPU total memory,
        restores ledger from disk (resetting last_seen timestamps to
        prevent immediate expiry, skipping any lease that is not a dict,
        lacks a string caller, or whose mb is not a finite number >= 0,
        and defaulting any priority/preemption fields missing from a
        ledger written before this feature existed). A non-dict top
        level or a non-dict leases map logs a warning and starts empty.
        Restored proton_game_cleared_at must be a finite number or None.
        Launches background tasks for Ollama polling, non-lease overhead
        sampling, unmanaged-load detection, and lease sweeping. Logs
        exceptions and re-raises to fail startup if the config assertion
        or the GPU query fails.
    """
    _validate_priority_config()

    global _gpu_total_mb, _proton_game_uncertain, _proton_game_cleared_at
    try:
        _gpu_total_mb = await asyncio.to_thread(_nvidia_smi_query, "memory.total")
    except Exception:
        logger.exception("failed to read GPU total memory at startup")
        raise

    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH) as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get("leases", {}), dict):
                logger.warning(
                    "ledger file at %s has invalid shape, starting from empty ledger",
                    LEDGER_PATH,
                )
                _leases.clear()
                _caller_to_lease.clear()
            else:
                now = time.monotonic()
                _proton_game_uncertain = bool(data.get("proton_game_active", False))
                cleared_at = data.get("proton_game_cleared_at")
                if cleared_at is not None and (
                    not isinstance(cleared_at, (int, float))
                    or isinstance(cleared_at, bool)
                    or not math.isfinite(cleared_at)
                ):
                    logger.warning(
                        "ignoring invalid proton_game_cleared_at=%r in ledger",
                        cleared_at,
                    )
                    cleared_at = None
                _proton_game_cleared_at = cleared_at
                restored = data.get("leases", {})
                kept = {}
                for lease_id, lease in restored.items():
                    if not isinstance(lease, dict):
                        logger.warning(
                            "skipping restored lease %s: not a dict (%r)",
                            lease_id, lease,
                        )
                        continue
                    caller = lease.get("caller")
                    if not isinstance(caller, str):
                        logger.warning(
                            "skipping restored lease %s: missing string caller",
                            lease_id,
                        )
                        continue
                    mb = lease.get("mb")
                    if (
                        not isinstance(mb, (int, float))
                        or isinstance(mb, bool)
                        or not math.isfinite(mb)
                        or mb < 0
                    ):
                        logger.warning(
                            "skipping restored lease %s with invalid mb=%r",
                            lease_id, mb,
                        )
                        continue
                    lease["last_seen"] = now
                    # Ledger written before this feature existed: default the
                    # new fields rather than KeyError the first time acquire()
                    # or the sweeper reads them.
                    lease.setdefault("priority", DEFAULT_PRIORITY)
                    lease.setdefault("preemptible", False)
                    lease.setdefault("pid", None)
                    lease.setdefault("preempt_flagged_at", None)
                    lease.setdefault("preempt_reason", None)
                    kept[lease_id] = lease
                _leases.update(kept)
                for lease_id, lease in _leases.items():
                    _caller_to_lease[lease["caller"]] = lease_id
                logger.info("ledger loaded at startup: %d lease(s) restored", len(_leases))
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            logger.exception("ledger file at %s corrupt/unreadable, starting from empty ledger", LEDGER_PATH)
            _leases.clear()
            _caller_to_lease.clear()
    else:
        logger.info("no ledger file found at %s, fresh start", LEDGER_PATH)

    _load_non_lease_samples()
    await _refresh_proton_state()


    asyncio.create_task(_poll_ollama())
    asyncio.create_task(_poll_non_lease_usage())
    asyncio.create_task(_poll_unmanaged_load())
    asyncio.create_task(_poll_proton_games())

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
        live safety margin (computed p95 or fallback), reserved MB (all
        leases), Ollama footprint, nvidia-smi used MB, non-lease
        sample counts, unmanaged overhead, Steam Proton game state and
        cooldown seconds, and the full lease ledger. Uses _last_used_mb
        when that reading is no older than
        2 * UNMANAGED_POLL_INTERVAL_SECONDS. Otherwise takes one live
        reading outside _lock. A failed live read logs a warning and
        reports nvidia_smi_used_mb as null. The lock is held only to
        snapshot ledger and poller state.
    """
    used_mb = _fresh_cached_used_mb()
    if used_mb is None:
        try:
            used_mb = await _live_used_mb()
        except Exception:
            logger.warning(
                "status: nvidia-smi memory.used unavailable; reporting nvidia_smi_used_mb=null",
            )
            used_mb = None
    async with _lock:
        margin_mb = _safety_margin_mb()
        last_sample = _non_lease_samples[-1] if _non_lease_samples else None
        return {
            "budget_mb": _gpu_total_mb - margin_mb,
            "gpu_total_mb": _gpu_total_mb,
            "safety_margin_mb": margin_mb,
            "safety_margin_source": (
                "percentile" if _computed_margin_mb is not None else "fallback"
            ),
            "reserved_mb": _reserved_mb(),
            "ollama_mb": _ollama_mb,
            "nvidia_smi_used_mb": used_mb,
            "non_lease_overhead_mb": None if last_sample is None else last_sample["mb"],
            "non_lease_locked": None if last_sample is None else last_sample["locked"],
            "non_lease_sample_count": len(_non_lease_samples),
            "non_lease_unlocked_sample_count": sum(
                1 for s in _non_lease_samples if s.get("locked") == "unlocked"
            ),
            "unmanaged_heavy": _unmanaged_heavy,
            "unmanaged_yield_overhead_mb": UNMANAGED_YIELD_OVERHEAD_MB,
            "unmanaged_overhead_mb": _unmanaged_overhead_mb,
            "proton_game_active": _proton_game_active,
            "proton_game_scan_uncertain": _proton_game_uncertain,
            "proton_game_retry_after_seconds": _proton_retry_after_seconds(),
            "leases": _leases,
        }


async def _dedup_existing_lease_locked(req: AcquireRequest) -> dict | None:
    """
    Input:      req - AcquireRequest; caller holds `_lock`
    Output:     {"granted": True, "lease_id": str} if this caller already
                holds a live lease, else None
    Details:
        Shared by the pre-wait dedup path and the wait-loop re-check.
        Two concurrent first acquires can both pass the pre-wait check.
        A new pid replaces the stored pid and persists the ledger.
    """
    existing_id = _caller_to_lease.get(req.caller)
    if existing_id is None or existing_id not in _leases:
        return None
    existing_lease = _leases[existing_id]
    if req.pid is not None and existing_lease.get("pid") != req.pid:
        existing_lease["pid"] = req.pid
        await _persist_ledger()
        logger.info(
            "acquire deduped for caller=%r -> existing lease %r, registered pid=%d",
            req.caller, existing_id, req.pid,
        )
    else:
        logger.info(
            "acquire deduped for caller=%r -> existing lease %r",
            req.caller, existing_id,
        )
    return {"granted": True, "lease_id": existing_id}


async def acquire(req: AcquireRequest):
    """
    Input:      req - AcquireRequest with caller, mb, timeout_s, priority
                (default "P2"), preemptible (default False), pid (optional)
    Output:     dict - {"granted": True, "lease_id": str, "warning": str
                (optional, one-shot)} or raises HTTPException(504/503/400/422)
    Details:
        POST /acquire. Grants when reserved_mb + ollama_mb
        <= budget_mb, where budget_mb = gpu_total_mb - _safety_margin_mb().
        The margin is the p95 of unlocked non-lease samples once enough
        exist, else GPU_SAFETY_MARGIN_MB, always floored at
        GPU_SAFETY_MARGIN_MIN_MB. That baseline is idle GPU use (desktop
        compositor, driver reserve, stray use). See .env.example. Live
        usage is not added to the gate. Adding it charged the same
        baseline twice and refused requests that physically fit. See
        Assist/ISSUE_double_counted_margin.md. Acquire does not spawn
        nvidia-smi. It logs the cached used reading from the pollers
        (or None). If that reading is no older than
        2 * UNMANAGED_POLL_INTERVAL_SECONDS and exceeds margin plus
        reserved, a warning is logged. A missing or stale cache does
        not fail acquire and does not emit that warning. The cache
        never gates admission. Deduplicates by caller name
        (one load per caller). Fairness uses monotonic sequence numbers
        on pending requests: only the earliest-queued pending request
        evaluates budget/preemption each poll; later-queued requests wait
        their turn. A zero-size request (mb <= 0, the proxy's per-request
        lease) never joins that queue and checks the budget on every
        poll, so a large waiter that cannot fit does not starve it.
        Polls with exponential backoff (1s to 10s) until timeout or
        grant. `_notify_waiters` wakes a waiter when a lease is
        released, reclaimed, or cleared, when a sequence leaves
        `_pending`, when `_ollama_mb` falls, or when the effective
        margin falls. The backoff is the fallback so deadline checks,
        priority aging, and fairness flagging still run on that
        schedule. Persists the ledger on grant. Raises 504
        on timeout, 400 on an unknown priority tier, 422 when caller
        length is outside 1..MAX_CALLER_LEN or mb exceeds the largest
        budget the GPU can ever grant (gpu_total minus
        SAFETY_MARGIN_MIN_MB), and 503 with detail
        UNMANAGED_BUSY_DETAIL plus a Retry-After header while heavy
        unmanaged load or a Steam Proton game/cooldown blocks the GPU.
        That 503 applies to every new request regardless of preemptible
        or mb, including one already waiting. A repeat acquire by a
        caller that already holds a lease still returns that lease,
        including during a game block. Each wait-loop pass re-checks
        _caller_to_lease under _lock so two concurrent first acquires
        cannot create two leases, even when the second request no
        longer fits after the first is granted. If
        len(_leases) >= MAX_ACTIVE_LEASES, a new grant is refused with
        503 TOO_MANY_LEASES_DETAIL and Retry-After: 5. An existing
        holder is not affected by that cap.

        Fairness preemption (see _effective_priority,
        _weakest_preemptible_holder): when this request's turn comes and
        it does not fit, and its own aged effective priority exceeds the
        weakest current preemptible holder's (tier + INCUMBENCY_BONUS),
        that holder is flagged (lease["preempt_flagged_at"]) so its next
        heartbeat response carries preempt=True. This does not grant
        anything immediately -- the waiter still polls normally until the
        flagged holder actually releases (self-evict) or the watchdog
        kill path in _sweep_expired_leases() reclaims it after
        GPU_BROKER_PREEMPT_GRACE_SECONDS. A non-preemptible holder
        (preemptible=False, the default) is never a preemption candidate.

        If a previous lease for this caller was force-terminated by the
        watchdog (see _reclaim_broken_lease), the next successful grant
        for that caller includes a one-shot "warning" field, then clears
        it -- never surfaced again after this one response. Notices
        older than NOTIFICATION_TTL_SECONDS are dropped at reclaim and
        at grant.
    """
    if req.priority not in TIER_BASE_PRIORITY:
        raise HTTPException(
            status_code=400,
            detail=f"invalid priority tier {req.priority!r}; expected one of {sorted(TIER_BASE_PRIORITY)}",
        )

    # A fresh /proc scan closes the one-second gap between poller ticks.
    # Deduped acquires remain valid for a process that already holds VRAM.
    await _refresh_proton_state()

    async with _lock:
        existing = await _dedup_existing_lease_locked(req)
        if existing is not None:
            return existing


    # Never-fit early reject, after caller dedup and before the FIFO.
    # gpu_total minus SAFETY_MARGIN_MIN_MB is the largest budget the
    # broker can ever offer: empty ledger, no Ollama, margin at its
    # floor. Do not use _safety_margin_mb() here. The computed p95 can
    # drop later, so a request that misses today's p95 might fit then.
    # A never-fit request at the FIFO head would block every later
    # positive-size waiter until timeout_s. _gpu_total_mb is 0 until
    # startup; do not treat that as a zero-capacity GPU. mb=0 is a
    # token and is never too large.
    max_grantable_mb = _gpu_total_mb - SAFETY_MARGIN_MIN_MB
    if _gpu_total_mb > 0 and req.mb > 0 and req.mb > max_grantable_mb:
        logger.warning(
            "acquire rejected: caller=%r mb=%.0f exceeds max grantable %.0f MB",
            req.caller, req.mb, max_grantable_mb,
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"request {req.mb:.0f} MB exceeds maximum grantable "
                f"capacity of {max_grantable_mb:.0f} MB"
            ),
        )

    zero_size = req.mb <= 0
    seq = next(_pending_seq)
    if not zero_size:
        async with _lock:
            _pending.add(seq)

    logger.info(
        "acquire received: caller=%r mb=%.0f timeout_s=%.0f priority=%r preemptible=%s",
        req.caller, req.mb, req.timeout_s, req.priority, req.preemptible,
    )
    start = time.monotonic()
    deadline = start + req.timeout_s
    backoff = ACQUIRE_POLL_MIN_SECONDS
    # Fairness age survives a 504 when the same caller retries within
    # WAITER_MEMORY_SECONDS. Without this, a consumer that backs off and
    # retries restarts at zero age and can never preempt a holder.
    aging_start = start
    if not zero_size:
        async with _lock:
            for caller, (_first, last) in list(_waiting_since.items()):
                if start - last > WAITER_MEMORY_SECONDS:
                    del _waiting_since[caller]
            aging_start = _waiting_since.get(req.caller, (start, start))[0]
            _waiting_since[req.caller] = (aging_start, start)
    # Recompute margin each poll, matching used_mb. A mid-wait p95 rise
    # must not grant against a stale lower budget. A mid-wait drop must
    # be visible to this waiter rather than waiting out the timeout.

    try:
        while True:
            async with _lock:
                existing = await _dedup_existing_lease_locked(req)
                if existing is not None:
                    _discard_pending_locked(seq)
                    _waiting_since.pop(req.caller, None)
                    return existing
                proton_retry = _proton_retry_after_seconds()
                if _unmanaged_heavy or proton_retry:
                    # Refuse without waiting behind the FIFO. A grant
                    # here could load VRAM while a game is starting.
                    _discard_pending_locked(seq)

                    logger.warning(
                        "acquire refused: caller=%r mb=%.0f, game block active "
                        "(proton_retry=%ds, unmanaged_heavy=%s)",
                        req.caller, req.mb, proton_retry, _unmanaged_heavy,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=UNMANAGED_BUSY_DETAIL,
                        headers={"Retry-After": str(max(
                            proton_retry, int(UNMANAGED_CLEAR_SECONDS) if _unmanaged_heavy else 0,
                        ))},
                    )
                # Zero-size requests skip the FIFO; see the docstring.
                is_my_turn = zero_size or not _pending or seq == min(_pending)
                if not is_my_turn:
                    logger.info(
                        "acquire poll: caller=%r mb=%.0f my_turn=False pending=%d "
                        "my_seq=%d min_pending_seq=%s (waiting for fair-ordering turn, not budget)",
                        req.caller, req.mb, len(_pending), seq,
                        min(_pending) if _pending else "n/a",
                    )
                if is_my_turn:
                    margin_mb = _safety_margin_mb()
                    budget_mb = _gpu_total_mb - margin_mb
                    reserved_total = _reserved_mb() + _ollama_mb
                    used_mb = _last_used_mb
                    expected_mb = margin_mb + reserved_total
                    fits = reserved_total + req.mb <= budget_mb
                    # Logged every time it is this request's turn, not
                    # only on grant or timeout. Distinguishes budget
                    # exhaustion from waiting on another caller's turn.
                    logger.info(
                        "acquire poll: caller=%r mb=%.0f my_turn=True pending=%d "
                        "reserved_total=%.0f budget_mb=%.0f fits=%s nvidia_smi_used_mb=%s",
                        req.caller, req.mb, len(_pending), reserved_total, budget_mb, fits,
                        None if used_mb is None else f"{used_mb:.0f}",
                    )
                    cache_fresh = _fresh_cached_used_mb() is not None
                    if cache_fresh and used_mb is not None and used_mb > expected_mb:
                        logger.warning(
                            "live nvidia-smi usage (%.0f MB) exceeds baseline + tracked "
                            "leases (%.0f MB = %.0f MB margin + %.0f MB reserved); "
                            "something is consuming VRAM the ledger doesn't know about",
                            used_mb, expected_mb, margin_mb, reserved_total,
                        )

                    if fits:
                        now = time.monotonic()
                        _prune_pending_notifications_locked(now)

                        if len(_leases) >= MAX_ACTIVE_LEASES:
                            _discard_pending_locked(seq)
                            logger.warning(
                                "acquire refused: caller=%r mb=%.0f, too many leases (%d)",
                                req.caller, req.mb, len(_leases),
                            )
                            raise HTTPException(
                                status_code=503,
                                detail=TOO_MANY_LEASES_DETAIL,
                                headers={"Retry-After": "5"},
                            )

                        lease_id = str(uuid.uuid4())
                        _leases[lease_id] = {
                            "caller": req.caller, "mb": req.mb,
                            "acquired_at": now, "last_seen": now,
                            "priority": req.priority, "preemptible": req.preemptible,
                            "pid": req.pid, "preempt_flagged_at": None,
                            "preempt_reason": None,
                        }
                        _caller_to_lease[req.caller] = lease_id
                        await _persist_ledger()
                        _discard_pending_locked(seq)

                        _waiting_since.pop(req.caller, None)
                        waited = now - start
                        logger.info(
                            "acquire granted: caller=%r lease=%r mb=%.0f waited=%.1fs",
                            req.caller, lease_id, req.mb, waited,
                        )
                        result = {"granted": True, "lease_id": lease_id}
                        watchdog_notice = _pending_notifications.pop(req.caller, None)
                        if watchdog_notice is not None:
                            result["warning"] = watchdog_notice[0]
                            logger.info(
                                "acquire: delivering one-shot watchdog notice to caller=%r",
                                req.caller,
                            )
                        return result

                    # Fairness preemption: only evaluated on this request's
                    # own turn, same scope as the fits check above -- see
                    # this function's docstring.
                    waited = time.monotonic() - aging_start
                    flagged = _maybe_flag_for_preemption(req.priority, waited, _leases)
                    if flagged is not None:
                        target_id, target_lease = flagged
                        logger.warning(
                            "acquire: caller=%r (priority=%r) flagged lease=%r (caller=%r, "
                            "priority=%r) for preemption -- not killed yet, waiting for "
                            "self-eviction or the watchdog grace period",
                            req.caller, req.priority, target_id,
                            target_lease["caller"], target_lease["priority"],
                        )

                now = time.monotonic()
                if now >= deadline:
                    _discard_pending_locked(seq)
                    if not zero_size:
                        _waiting_since[req.caller] = (aging_start, now)
                    logger.warning(
                        "acquire timed out: caller=%r mb=%.0f after %.0fs (waiting %.0fs total), no room freed",
                        req.caller, req.mb, req.timeout_s, now - aging_start,
                    )
                    # Retry-After is the shortest useful backoff: the broker
                    # re-checks budget at most this often while a caller waits.
                    raise HTTPException(
                        status_code=504,
                        detail="timeout waiting for GPU headroom",
                        headers={"Retry-After": str(int(ACQUIRE_POLL_MAX_SECONDS))},
                    )

                wait_s = min(backoff, deadline - now)
                # Capture after the admission check, still under `_lock`.
                # Notify sets this Event and swaps in a new one, so a
                # wakeup that lands after capture cannot be lost.
                ev = _admission_event
            task = asyncio.ensure_future(ev.wait())
            try:
                await asyncio.wait({task}, timeout=wait_s)
            finally:
                if not task.done():
                    task.cancel()
            backoff = min(backoff * 2, ACQUIRE_POLL_MAX_SECONDS)
    except HTTPException as exc:
        logger.warning("acquire: refusing caller=%r with HTTP %d", req.caller, exc.status_code)
        async with _lock:
            _discard_pending_locked(seq)
        raise
    except BaseException:
        logger.exception("acquire: unexpected error for caller=%r", req.caller)
        async with _lock:
            _discard_pending_locked(seq)
        raise



async def release(req: ReleaseRequest):
    """
    Input:      req - ReleaseRequest with lease_id
    Output:     dict - {"released": True} or False if lease not found
    Details:
        POST /release. Removes the lease from the ledger and caller index.
        Logs a warning on an unknown or already-released lease_id. Persists
        the ledger on successful removal. Returns False when not found.
        The client may retry after a transient broker loss.
    """
    async with _lock:
        lease = _leases.pop(req.lease_id, None)
        if lease is None:
            logger.warning("release for unknown/already-released lease_id=%r", req.lease_id)
            return {"released": False}
        if _caller_to_lease.get(lease["caller"]) == req.lease_id:
            del _caller_to_lease[lease["caller"]]
        await _persist_ledger()
        _notify_waiters()
        logger.info("release: caller=%r lease=%r mb=%.0f", lease["caller"], req.lease_id, lease["mb"])

        return {"released": True}


async def heartbeat(req: HeartbeatRequest):
    """
    Input:      req - HeartbeatRequest with lease_id
    Output:     dict - {"ok": True, "preempt": bool, "reason": str} or
                {"ok": False} if lease not found
    Details:
        POST /heartbeat. Updates last_seen for a lease so it does not
        expire. Clients call this while they hold GPU. Logs at debug.
        Returns False on an unknown lease_id.

        preempt is True once this lease has been flagged as the
        preemption target for a higher (aged) effective-priority waiter
        (see acquire()'s fairness-preemption check and
        _weakest_preemptible_holder()), a Proton block, or unmanaged
        GPU load. It stays True on every subsequent
        heartbeat for this lease -- there is no un-flagging -- so a
        preemptible client's own idle-watcher loop can check it alongside
        its existing idle-timeout check and self-evict (call /release) at
        its own next safe/step boundary. reason is a fixed, generic
        string for that flag source: it never names the waiting caller.
        A restored lease without preempt_reason uses the fairness text.
        A flagged client that
        does not self-evict within GPU_BROKER_PREEMPT_GRACE_SECONDS falls
        through to the watchdog kill path in _sweep_expired_leases()
        (see _classify_reclaim_reasons "preempt_grace_expired").
    """
    async with _lock:
        lease = _leases.get(req.lease_id)
        if lease is None:
            logger.warning("heartbeat for unknown lease_id=%r", req.lease_id)
            return {"ok": False}
        lease["last_seen"] = time.monotonic()
        logger.debug("heartbeat: caller=%r lease=%r", lease["caller"], req.lease_id)
        preempt = lease.get("preempt_flagged_at") is not None
        if not preempt:
            reason = ""
        else:
            source = lease.get("preempt_reason") or "fairness"
            reason = _PREEMPT_REASON_TEXT_BY_SOURCE.get(source, PREEMPT_REASON_TEXT)
        return {
            "ok": True,
            "preempt": preempt,
            "reason": reason,
        }


async def admin_clear(req: AdminClearRequest):
    """
    Input:      req - AdminClearRequest with admin_token, dry_run
    Output:     dict - kill results + ledger reset summary, or dry-run preview
    Details:
        POST /admin/clear. Force-reclaims the GPU. Signals every process
        nvidia-smi reports as holding GPU memory (SIGTERM, then SIGKILL
        after ADMIN_KILL_GRACE_SECONDS for survivors whose start time
        is unchanged). Does this even if the process holds no broker
        lease. Then clears the lease ledger. This is a manual admin
        action, not automatic enforcement. See
        Assist/HANDOVER_post_admission_oom.md. It kills every GPU compute
        process nvidia-smi lists except the broker itself, including
        processes no lease owns (games, desktop renders, host Talkie),
        then resets all leases. There is no allowlist.

        Requires GPU_BROKER_ADMIN_TOKEN to be set (501 if unset).
        req.admin_token must match it via a constant-time comparison
        (403 on mismatch, logged as WARNING). dry_run=true reports the
        targets (pid, name, mb) without sending a signal or touching
        the ledger. Call it with dry_run=true first and review would_kill.
    """
    if not ADMIN_TOKEN:
        logger.error("POST /admin/clear called but GPU_BROKER_ADMIN_TOKEN is not set; refusing")
        raise HTTPException(status_code=501, detail="admin clear is disabled: set GPU_BROKER_ADMIN_TOKEN to enable it")
    if not hmac.compare_digest(req.admin_token, ADMIN_TOKEN):
        logger.warning("POST /admin/clear called with an invalid admin_token")
        raise HTTPException(status_code=403, detail="invalid admin_token")

    try:
        before_mb = await asyncio.to_thread(_nvidia_smi_query, "memory.used")
    except Exception:
        logger.exception("admin clear: failed to read memory.used before clearing")
        before_mb = None

    try:
        targets = await asyncio.to_thread(_list_gpu_compute_processes)
    except Exception:
        logger.exception("admin clear: failed to list GPU compute processes")
        raise HTTPException(status_code=502, detail="nvidia-smi compute-apps query failed")

    my_pid = os.getpid()
    targets = [t for t in targets if t["pid"] != my_pid]

    if req.dry_run:
        logger.info("admin clear dry-run: would target %d process(es)", len(targets))
        return {"dry_run": True, "would_kill": targets, "nvidia_smi_used_mb": before_mb}

    logger.warning(
        "admin clear: killing %d GPU process(es) and resetting the lease ledger (%d active lease(s))",
        len(targets), len(_leases),
    )

    for target in targets:
        target["start_time"] = _proc_start_time(target["pid"])
    targets = await _kill_targets_with_grace(targets)
    for target in targets:
        target.pop("start_time", None)

    async with _lock:
        cleared_leases = len(_leases)
        _leases.clear()
        _caller_to_lease.clear()
        await _persist_ledger()
        _notify_waiters()


    try:
        after_mb = await asyncio.to_thread(_nvidia_smi_query, "memory.used")
    except Exception:
        logger.exception("admin clear: failed to read memory.used after clearing")
        after_mb = None

    logger.warning(
        "admin clear complete: %d process(es) signaled, %d lease(s) cleared, "
        "nvidia_smi_used_mb %.0f -> %s",
        len(targets), cleared_leases, before_mb or -1,
        f"{after_mb:.0f}" if after_mb is not None else "unknown",
    )
    return {
        "dry_run": False,
        "killed": targets,
        "leases_cleared": cleared_leases,
        "nvidia_smi_used_mb_before": before_mb,
        "nvidia_smi_used_mb_after": after_mb,
    }


# Wire up endpoints and startup via lifespan
@asynccontextmanager
async def lifespan(app_ctx: FastAPI):
    """
    Input:      app_ctx - FastAPI application instance
    Output:     None
    Details:
        FastAPI lifespan. Runs startup() before serving. After yield,
        persists the in-memory non-lease sample list. Logs a failure
        and does not raise.
    """
    # Startup
    await startup()
    yield
    try:
        async with _lock:
            snapshot = list(_non_lease_samples)
        await asyncio.to_thread(_persist_non_lease_sync, snapshot)
    except Exception:
        logger.exception("failed to persist non-lease usage on shutdown")


def _host_header_hostname(value: str) -> str:
    """
    Input:      value - raw Host header
    Output:     hostname without port; IPv6 unbracketed; lowercased
    Details:    Starlette TrustedHostMiddleware splits Host on the
                first colon, so [::1]:8792 becomes '[' and ::1
                becomes ''. Parse RFC 3986 Host instead. Hostnames
                are case-insensitive (RFC 9110 / RFC 3986). A
                trailing dot is the DNS FQDN marker and is stripped.
    """
    host = value.strip()
    if not host:
        return ""
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            host = host[1:end]
    elif host.count(":") <= 1:
        colon = host.find(":")
        if colon != -1:
            host = host[:colon]
    host = host.lower()
    if host.endswith("."):
        host = host[:-1]
    return host


def _parse_allowed_hosts(raw: str | None = None) -> list[str]:
    """
    Input:      raw - comma-separated extra hosts, or None to read
                GPU_BROKER_ALLOWED_HOSTS
    Output:     allowlist starting with DEFAULT_ALLOWED_HOSTS
    Details:    Strips whitespace, drops empties, unbrackets IPv6
                literals, lowercases every entry, and skips
                duplicates. Extra names are appended; they do not
                replace the defaults. A lone '*' is ignored so the
                env var cannot disable the check.
    """
    if raw is None:
        raw = os.environ.get("GPU_BROKER_ALLOWED_HOSTS", "")
    hosts: list[str] = []
    seen: set[str] = set()
    for name in list(DEFAULT_ALLOWED_HOSTS) + [p.strip() for p in raw.split(",")]:
        if not name or name == "*":
            continue
        if not name.startswith("*"):
            name = _host_header_hostname(name)
        name = name.lower()
        if not name or name in seen:
            continue
        seen.add(name)
        hosts.append(name)
    return hosts


class _TrustedHostMiddleware(TrustedHostMiddleware):
    """
    Input:      same as starlette.middleware.trustedhost.TrustedHostMiddleware
    Output:     400 PlainTextResponse on a disallowed Host; otherwise
                the wrapped app
    Details:    Overrides Host parsing so IPv6 literals (::1 and
                [::1]:port) match DEFAULT_ALLOWED_HOSTS. Starlette's
                split-on-colon check cannot. Host comparison is
                case-insensitive; a trailing FQDN dot is ignored.
                Rejected Host is 400, same as Starlette's default body.
    """

    async def __call__(self, scope, receive, send) -> None:
        if self.allow_any or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = _host_header_hostname(headers.get("host", ""))
        is_valid_host = False
        for pattern in self.allowed_hosts:
            if host == pattern or (
                pattern.startswith("*") and pattern != "*" and host.endswith(pattern[1:])
            ):
                is_valid_host = True
                break
        if is_valid_host:
            await self.app(scope, receive, send)
            return
        response = PlainTextResponse("Invalid host header", status_code=400)
        await response(scope, receive, send)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def _json_safe_errors(value):
    """
    Input:      value - a Pydantic/FastAPI error structure
    Output:     the same structure with non-finite floats replaced by str
    Details:
        FastAPI's default 422 JSON encoder uses allow_nan=False. A
        rejected NaN/Inf field is echoed in the error's input and that
        dump raises. Replace non-finite floats so the 422 returns.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe_errors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_errors(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(request: Request, exc: RequestValidationError):
    """
    Input:      request - FastAPI Request; exc - RequestValidationError
    Output:     JSONResponse with status 422
    Details:
        Same body shape as FastAPI's default handler ({"detail": errors}),
        with non-finite floats made JSON-safe. See _json_safe_errors.
    """
    return JSONResponse(status_code=422, content={"detail": _json_safe_errors(exc.errors())})


# CORS for read-only browser access to the ledger. A dashboard on another
# origin can poll GET /status. allow_methods is GET only. The browser
# then refuses preflight for POST /acquire, /release, /heartbeat, and
# /admin/clear. That is the drive-by threat model. CORS is not a
# substitute for ADMIN_TOKEN, which still guards /admin/clear against
# any non-browser client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Content-Type"],
)

app.add_middleware(
    _TrustedHostMiddleware,
    allowed_hosts=_parse_allowed_hosts(),
    www_redirect=False,
)

app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/status", status, methods=["GET"])
app.add_api_route("/acquire", acquire, methods=["POST"])
app.add_api_route("/release", release, methods=["POST"])
app.add_api_route("/heartbeat", heartbeat, methods=["POST"])
app.add_api_route("/admin/clear", admin_clear, methods=["POST"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
