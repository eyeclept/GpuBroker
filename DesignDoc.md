# GpuBroker — cross-process VRAM-budget lock

Distilled implementation plan for this folder. Full context/history (why
`EXCLUSIVE_PROCESS` was abandoned, Part A Ollama containerization, Part C Steam
GPU-yield script) lives in `/home/eyeclept/.claude/plans/smooth-fluttering-dragon.md`.
This file scopes just the broker + its two client integrations.

**Status:** `server.py` in this folder is a first-draft working implementation of the
broker itself (untested). `Dockerfile`, `docker-compose.yml`, and both client-side
integrations (Flux2Klein9B, Needle) are not yet written.

## What this is

A standalone broker container ("similar to HashiCorp Vault: a service tacked on via
an extra container that does the listening and coordination, and a small runner added
to the gpu containers that talk to it" — the user's own framing) coordinating VRAM
between Flux2Klein9B and Needle, with Ollama accounted for passively (its Go source is
not modified) via its own `GET /api/ps` endpoint.

## Remaining work

### 1. `Dockerfile`
Plain `python:3.12-slim` base — **not** a CUDA image. Confirmed live on this host:
Podman's CDI device passthrough (`devices: [nvidia.com/gpu=all]`,
`security_opt: [label=disable]`) injects a working `/usr/sbin/nvidia-smi` binary into
even a minimal non-CUDA container via CDI hooks, so no CUDA toolkit is needed — the
broker only shells out to `nvidia-smi`, never touches CUDA itself. Install:
`fastapi`, `uvicorn[standard]`, `httpx`. `COPY server.py /app/server.py`.

### 2. `docker-compose.yml`
Mirror the shape of `~/Documents/Code/Docker/Flux2Klein9B/docker-compose.yml`: port `8792:8792`,
`devices: [nvidia.com/gpu=all]`, `security_opt: [label=disable]`, bind mount
`./data:/data` (ledger persistence — `leases.json`), `restart: unless-stopped`,
healthcheck against `/health`. Env vars (see `server.py` top for full list and
defaults): `GPU_SAFETY_MARGIN_MB`, `OLLAMA_URL` (default
`http://host.containers.internal:11434` — correct whether Ollama is still the native
systemd service or already cut over to a container, since the port stays 11434 either
way), `OLLAMA_POLL_INTERVAL_SECONDS`, `GPU_BROKER_LEASE_TTL_SECONDS`.

**Before first run:** get `GPU_SAFETY_MARGIN_MB` empirically, not by guessing —
with Flux2Klein9B/Needle/Ollama all stopped and no game running:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```
That baseline (desktop compositor + driver reserve + any stray usage) plus a small
buffer is the margin. 1536 is a placeholder default in `server.py`, not a measured value.

### 3. Client runner — Flux2Klein9B (`~/Documents/Code/Docker/Flux2Klein9B/server.py`), async

Remove outright: `GPU_CONTENTION_RETRIES`/`GPU_CONTENTION_BACKOFF_SECONDS` constants
and comment, `_is_gpu_contention()`, and the `for attempt in range(...)` retry loop in
`_ensure_loaded()` — all written for the abandoned `EXCLUSIVE_PROCESS` mechanism.
Replace with an `httpx.AsyncClient`-based acquire/release/heartbeat, matching this
file's existing async style:
- **Acquire**: inside `async with load_lock:`, after the `state = "loading"` log,
  before `pipe = await asyncio.to_thread(_build_pipeline)`.
  `mb = round(2560 * (width*height)/(1024*1024))`, floored at 2560, capped at 8192
  (scales with resolution since `GenerateRequest` has no server-enforced cap and
  sequential-offload peak VRAM scales with it).
- **Release**: inside `_evict()`, **after** `torch.cuda.empty_cache()` — release only
  once memory is actually freed, not merely once eviction is decided.
- **Heartbeat**: inside `_idle_watcher()`'s loop body, every tick, if a lease is held.
- **Broker unreachable** (connection error, not a denial): fail open — log WARNING,
  proceed without a lease, gated by `BROKER_FAIL_OPEN=true` env flag.
- **Broker reachable but denies** (504, no room): do **not** fail open — raise
  `HTTPException(503, ...)`, matching this file's existing failed-load pattern.
- Connection retry (broker unreachable, distinct from "no room"): 0.5s → double →
  cap 5s, 3–5 attempts, before falling back per `BROKER_FAIL_OPEN`.
- New dependency: add `httpx` to `Flux2Klein9B/Dockerfile`.

### 4. Client runner — Needle (`~/Documents/Code/Docker/Needle/server.py`), **sync** (material distinction)

`_ensure_loaded()`/`_evict()` are plain `def`s under `threading.RLock`, not
`async def` — this file already blocks the event loop during synchronous JAX
generation (`classify()`), so use a **synchronous `httpx.Client`** here, not async.
- **Acquire (first load)**: before `_load_models()` itself, inside the
  `if _load_state == "idle":` branch — `_load_models()`'s own `jnp.array()` placement
  is what puts params on GPU (per this file's existing comment), there's no separate
  "load, then move to GPU" step for this path.
- **Acquire (restore path)**: before `_restore_to_gpu()`.
- **Release**: inside `_evict()`, after both `jax.device_put(..., _cpu_device)` calls,
  before `_on_gpu = False`.
- **Heartbeat**: fired via the same sync client from within `async def _idle_watcher()`'s
  loop (brief blocking call, same class already accepted elsewhere in this file).
- `mb = 768` flat — no resolution-scaling concern like Flux2Klein9B.
- Same fail-open/fail-closed/`BROKER_FAIL_OPEN`/backoff rules as Flux2Klein9B,
  duplicated in this file (not shared as a package — matches the existing convention
  that these two files intentionally mirror each other's shape rather than share code).
- New dependency: add `httpx` to `Needle/Dockerfile`.

## Design reference (already implemented in `server.py`)

- **Admission formula** (superseded live-usage gate): an earlier draft
  admitted `requested_mb` iff
  `max(nvidia_smi_used_mb, reserved_mb + ollama_mb) + requested_mb <= BUDGET_MB`.
  That formula is not in `server.py`. Live usage only feeds a log warning
  from the poller cache.
  Current gate: `reserved_mb + ollama_mb + requested_mb <= budget_mb`,
  where `budget_mb = gpu_total_mb - _safety_margin_mb()`.
  `_safety_margin_mb()` is the larger of the p95 of unlocked non-lease
  samples (or `GPU_SAFETY_MARGIN_MB` as fallback) and
  `GPU_SAFETY_MARGIN_MIN_MB`.
  `reserved_mb` is the broker ledger.
- **Idempotency**: dedup key is the caller's own name (not a generated UUID) — both
  clients only ever have one load attempt in flight at a time already, guarded by
  their own existing lock.
- **Persistence**: `./data/leases.json`, atomic write (temp file + `os.replace()`).
  On startup, restored leases get `last_seen` reset to "now" so a healthy
  pre-restart lease isn't immediately swept.
- **Fairness**: monotonic arrival sequence number per pending `/acquire`; only the
  minimum currently-pending sequence number may be granted, avoiding starvation
  between the two callers.
- **Sweeper**: reclaims any lease not heartbeated within 180s (crash-without-release
  safety net).
- Full logging-level table (every event categorized INFO/WARNING/ERROR) is in the
  master plan file and mirrored in `server.py`'s log calls directly.

## Verification (once wired up)

1. `podman compose up -d`; `GET :8792/status` shows empty ledger, correct `budget_mb`,
   live `nvidia_smi_used_mb`, `ollama_mb` populated from a real `/api/ps` poll.
2. Restart the broker container while Flux2Klein9B holds a lease; confirm `/status`
   shows it restored (persistence).
3. `podman kill` (not `stop`) Flux2Klein9B mid-generation without releasing; confirm
   the sweeper reclaims the lease within ~180s.
4. Drive Flux2Klein9B and Needle to load concurrently while Ollama holds a model;
   confirm one queues (logged wait duration) instead of either OOMing.
5. Stop the broker entirely; confirm both clients still serve requests (fail-open,
   WARNING logged), then restart the broker and confirm normal operation resumes
   without restarting the clients.
