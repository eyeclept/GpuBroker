# GpuBroker client examples

Working, runnable reference code for integrating a client with GpuBroker. This folder exists
because writing the first real integration (Whisper, via compileYT) meant reading `server.py`
and guessing at the contract, and two of those guesses were wrong in ways that only surfaced
in production: see `Assist/ISSUE_double_counted_margin.md` and
`Assist/HANDOVER_post_admission_oom.md`. Every file here is written from those two incidents,
not from the API alone.

`TI-Template/` (repo root) has the copy-paste acquire/release/heartbeat functions to drop into
your own service. This folder is a level up: worked demonstrations of the patterns around those
functions -- when to call them, what to do when they fail, and how to size the numbers you pass.
Every `.py` file here runs standalone (`python3 examples/<file>.py`) against a live broker at
`http://localhost:8792` (override with the `BROKER_URL` env var); none of them are stubs.

## Files, in the order a new integrator should read them

1. **`minimal_client.py`** -- acquire, do work only with a lease, release in a
   `finally` block. Logs 504, 503 (detail and Retry-After), 422, and the one-shot
   `warning`. Exits non-zero on refusal. Start here.
2. **`long_running_service.py`** -- the fuller shape used by every real service in
   `~/Documents/Code/Docker/` (Whisper, Flux2Klein9B, Needle): lazy load on first request, a
   background heartbeat, idle eviction. `/acquire` sends `preemptible: true` because
   this service can unload on demand. The broker may then ask it to unload via
   heartbeat `preempt=true`. `preempt=true` unloads and releases.
   `ok=false` unloads and clears the lease. A transport error keeps the lease.
   `--hold-seconds N` (or `HOLD_SECONDS`) keeps the lease that long without idle
   eviction.
3. **`handling_refusal.py`** -- 422 never-fit raises `NeverFitsError`. 503
   `unmanaged_gpu_load` / `too_many_leases` retries with `max(Retry-After, backoff)`.
   504 is classified against `/status.budget_mb` and also honors Retry-After.
4. **`post_admission_oom.py`** -- what to do when CUDA OOMs despite a *valid, currently-held*
   lease. This is a different failure from #3: the broker granted the request in good faith, and
   something it does not control (see "Unmanaged consumers" below) took the memory anyway. Read
   `Assist/HANDOVER_post_admission_oom.md` first. The retry re-checks GET `/status` and skips
   on a game block or when `mb` exceeds `budget_mb`. The breaker stays tripped for the run.
5. **`sizing_a_request.md`** -- how to fill in your own `*_ACQUIRE_MB` value. Sizing from a
   model's documented weight size, not measured peak usage, is exactly how Whisper's lease ended
   up smaller than its real demand.
6. **`raw_protocol.sh`** -- GET `/health`, GET `/status`, POST `/acquire`, POST `/heartbeat`,
   and POST `/release` as `curl` calls. It does not call POST `/admin/clear`. A trap
   releases the lease on exit.

## Outcome handling

`POST /acquire` answers:

- 200 `{granted, lease_id, warning?}`: do the GPU work. Log `warning` if present.
  That field is a one-shot watchdog notice.
- 504: no room. Honor `Retry-After` (10). Classify against GET `/status` before a retry.
- 503 `unmanaged_gpu_load` or `too_many_leases`: transient. Sleep
  `max(Retry-After, backoff)` inside the retry budget.
- 422: invalid input, or a request that exceeds maximum grantable capacity.
  Do not retry the never-fit case.
- 400: bad priority. Do not retry.

`POST /heartbeat` answers `{ok: false}` for an unknown lease, else
`{ok: true, preempt, reason}`. A preempt-flagged holder must unload and release
within 30 s. A transport error is not a lost lease. Retry on the next tick.

`POST /release` answers `{released: bool}`. Check the HTTP status.

## `BROKER_FAIL_OPEN`

None of the examples in this folder read `BROKER_FAIL_OPEN`.
They fail closed: GPU work runs only with a granted lease.
A refused or failed acquire logs the reason and does not start work.

`TI-Template/` at the repo root is the copy-paste client that reads that env var.
Set it there for a service that has a safe fallback when the broker is unreachable.
Do not fail open on 503 or 504. Those are explicit refusals.

- **`true`**: proceed after a transport failure only, logging a warning.
  Flux2Klein9B uses this because it can still generate an image without coordination.
- **`false`**: refuse the request on transport failure too.
  Needle uses this because uncoordinated GPU use can crash the service.

State which value your service uses, and why, in your own README.

## Unmanaged consumers, and the design question this folder answers

`Assist/HANDOVER_post_admission_oom.md` reproduced a real failure: Ollama loaded a 4996 MB model
without ever calling `/acquire`, GpuBroker only learned about it from its own background poll
(`_poll_ollama`, up to `OLLAMA_POLL_INTERVAL_SECONDS` stale), and Whisper's next transcription hit
a genuine CUDA OOM despite holding a lease GpuBroker had correctly granted. That report asked
GpuBroker to pick one of three shapes for handling consumers that never ask (gate Ollama directly,
reserve a bigger floor instead of just counting it, or add a signal clients can check) and could
not decide from outside. The answer, implemented across this repo:

**Reject before consuming, on the client side, and say why.** Concretely:

- `~/Documents/Code/Scripts/sync_to_public.py` -- the exact script that caused the incident
  documented above (its changelog-drafting call to Ollama is what loaded the colliding model --
  Ollama itself was started by unrelated work on the host, but this script's own call is what
  this repo controls) -- now calls GpuBroker's `/status` and Ollama's `/api/ps` /`/api/tags`
  before every Ollama call, computes whether the model's VRAM footprint fits under
  `budget_mb - reserved_mb - ollama_mb`, and refuses to run (`sys.exit(1)`, with the exact
  numbers and which lease is holding the room) if it does not. `--skip-headroom-check` overrides
  it; an unreachable broker fails open (a documentation sync should not block on broker uptime).
  This is "gate Ollama" applied at the one call site this repo actually owns, rather than at
  GpuBroker itself gating a process it does not control.
- `post_admission_oom.py` in this folder is the client-side complement for the case an
  unmanaged consumer collides *after* your own lease was already granted: release immediately,
  do not retry in place, raise a distinct exception, and stop offering work to the dependency
  after a small number of consecutive failures (see that file's module docstring for the full
  policy and why each step is there).

This was chosen over broker-side enforcement (GpuBroker forcibly unloading Ollama models, or
sizing `GPU_SAFETY_MARGIN_MB` large enough to absorb any unmanaged consumer) for the reason
`HANDOVER_post_admission_oom.md` already gives: Ollama alone (4996 MB) exceeds the entire
configured margin (4864 MB), so no margin on a 12 GB card both admits a 5 GB Whisper lease and
absorbs a 5 GB Ollama load; the margin lever cannot fix this. Broker-side revocation or a
pressure endpoint remain real options and are not implemented -- they need every client to
cooperate to be worth building, which is exactly what this folder is for if that work happens
later.
