# State of the program

GpuBroker is a working VRAM admission service. The broker and the
proxy run as live compose containers on this host. Tightly-integrated
clients in Flux2Klein9B and Needle already call the lease API.

This file is a status snapshot for a reader who has not seen the
code. Claims below were checked against `server.py`,
`GpuBroker-Proxy/server.py`, `Assist/TODO.md`, git history, unit
tests, and the running containers on 2026-08-19.

## Current stage

complete-dev-with-todo. Phases 0 through 5 in `Assist/TODO.md` are
marked done. The remaining work is known bugs, a multi-tenant design
question, and later features. The program is not a prototype.

Live observation on 2026-08-19 (containers were not restarted):

- `gpubroker-gpubroker-1`: Up 11 days, healthy, `0.0.0.0:8792`
- `gpubroker-proxy-gpubroker-proxy-1`: Up 12 days, healthy, `0.0.0.0:11434`
- `GET http://localhost:8792/health` returned `{"status":"ok"}`
- `GET http://localhost:11434/health` returned `{"status":"ok"}`
- `GET http://localhost:8792/status` returned
  `budget_mb=7418`, `gpu_total_mb=12282`, `safety_margin_mb=4864`,
  `reserved_mb=0`, `ollama_mb=0`, `nvidia_smi_used_mb=1656`, empty
  `leases`

`GPU_SAFETY_MARGIN_MB=4864` in `.env` is a measured value, not the
1536 placeholder in `.env.example`.

## What is done

### Broker (`server.py`)

The FastAPI broker admits leases with:

```
reserved_mb + ollama_mb + requested_mb <= budget_mb
```

Live `nvidia-smi` use is read for a warning only. It does not gate
admission. Commit `d9c736d` removed the old
`max(nvidia_smi_used_mb, ...)` gate after it double-counted the
safety margin.

Also shipped:

- Idempotent acquire by caller name
- FIFO fairness by pending sequence number
- Atomic ledger persist to `GPU_BROKER_LEDGER_PATH`
- Sweeper that reclaims leases with no heartbeat within 180s
- Passive Ollama poll on `/api/ps`
- `POST /admin/clear` (token-gated, needs `pid: host`)
- GET-only CORS for browser dashboards polling `/status`

`docker-compose.yml` publishes `8792:8792` on all interfaces, not
loopback. Other containers reach the broker through
`host.containers.internal`. The service joins `ollama_default` so the
passive poll can hit `http://ollama:11434` without the proxy.

### Proxy (`GpuBroker-Proxy/`)

A separate compose project. It publishes host port 11434. Ollama no
longer publishes that port. The proxy routes by `registry.json`:

- `talkie` -> `http://host.containers.internal:8101` for
  `talkie-1930-13b`
- `freetoken` -> `http://host.containers.internal:1919` for
  `ornith-1.5-35b-a3b`
- `ollama` -> `http://ollama:11434` (default)

Gated requests (`model` in the body, except `POST /api/show`) take an
`mb=0` lease, then forward the body unchanged. Broker denial returns 429 with
`Retry-After`. Broker unreachable fails open.

Headroom needed no config change. It still targets `:11434`.

### Tightly-integrated clients (other repos)

Flux2Klein9B (async `httpx`) and Needle (sync `httpx`) call acquire,
release, and heartbeat. `EXCLUSIVE_PROCESS` retry code in
Flux2Klein9B is gone. Cross-repo verification in `Assist/TODO.md`
phase 4 is marked done, including persistence across a host reboot.

FreeToken loads on demand. Its wrapper serves :1919, acquires a lease
and starts `ft serve` on the first request, and unloads then releases
after 60 s idle (`FREETOKEN_IDLE_SECONDS`). The broker scans for
Steam Proton games before each acquire and every second. It refuses new
leases while one runs and for 30 minutes after it exits. Heavy
unmanaged VRAM use also causes a 503 with Retry-After. Preemptible
holders get `preempt: true` on heartbeat with a per-source reason.
The proxy and Flux2Klein9B treat that 503 as a denial.

Flux keeps P2 priority and now waits up to 150 s for a broker lease.
Needle is slated for deletion and is no longer used.

`TI-Template/` holds copy-paste async and sync clients.
`examples/` holds runnable patterns, including post-admission OOM.

### Tests

Broker, proxy, FreeToken, and TI-Template tests live under `tests/`,
`GpuBroker-Proxy/tests/`, `FreeToken/tests/`, and `TI-Template/tests/`.
`python3 -m pytest -q tests/` : 190 passed (2026-10-11).
They mock `nvidia-smi` and HTTP. They do not need a live GPU.

Broker tests cover admission, the double-counted-margin regression,
idempotency, queue timeout, input bounds, Host allowlist, watchdog
reclaim, Proton and unmanaged gates, waiter wakeup, and `/admin/clear`.
Proxy tests cover registry resolution, mb=0 acquire, 429, fail-open,
stream heartbeat, disconnect release, and ungated metadata passthrough.

`feature_list.json` in both folders marks shipped features
`"passes": true`.

## What is left

Open items in `Assist/TODO.md` (do not treat this as a plan to
implement here):

1. **Multi-tenant isolation (item 7).** Design question only. Do not
   implement yet. GpuBroker is a cooperative admission controller. It
   is not an isolation boundary. It cannot stop a process that never
   calls `/acquire`. A short written threat-model note is the done
   criterion. That note is not in `docs/` yet. This file states the
   limit in "Known issues" below.
2. **Later features (item 8).** Multi-GPU. Priority preemption with
   pause/resume. AMD (`rocm-smi`) and Intel (`xpu-smi`) backends.
   No AMD or Intel GPU is on this workstation to test against.

`CORS_ALLOW_ORIGINS` is read by `server.py` and is not listed in
`.env.example`.

## Known issues and limits

**Ledger overcommit has limited revocation.** If an unmanaged consumer
(Ollama, a game, a browser) takes memory a held lease relied on, the
ledger can go over `budget_mb`. Heavy unmanaged load and Steam Proton
games flag every preemptible holder. The next heartbeat returns
`preempt: true` with a per-source reason (`unmanaged` or `proton`).
The watchdog reclaims a flagged lease after a 30 s grace.
Non-preemptible holders are not flagged. They find out from CUDA.
Reproduced in
`Assist/HANDOVER_post_admission_oom.md`. Client-side mitigation
exists (`examples/post_admission_oom.py`, and a headroom check in
`~/Documents/Code/Scripts/sync_to_public.py`). Raising
`GPU_SAFETY_MARGIN_MB` does not fix this. A single Ollama load
(4996 MB) already exceeds the entire configured margin (4864 MB) on
this 12 GB card.

**A granted lease does not bound real usage.** Whisper sized
`WHISPER_ACQUIRE_MB=5120` from static weights. Measured peak was at
least 5874 MB. That is a client sizing bug. See
`examples/sizing_a_request.md`.

**`GET /v1/models` through the proxy returns Ollama's list only.** A
GET has no body, so routing cannot see Talkie. Aggregating lists
would merge response bodies and break the byte-forward invariant.
Clients that need Talkie's list should query `:8101`.

**`POST /api/show` is routed by model and is not gated.** It is in
`UNGATED_MODEL_PATHS`. It does not load weights. Other POSTs that
include `model` still take an `mb=0` lease.

**In-container healthcheck can lie.** The Dockerfile probes
`localhost:8792` inside the container. A dead host port mapping still
reads healthy. Confirmed once: port 8792 refused connections while
`podman ps` showed healthy. `BROKER_FAIL_OPEN=true` then let every
consumer run unmetered. Probe the published port from the host.

**CORS is GET-only, on purpose.** It does not protect `curl`.
`GPU_BROKER_ADMIN_TOKEN` is the control that matters for non-browser
clients.

**Single NVIDIA GPU.** Admission shells out to `nvidia-smi`. Compose
requests `devices: [nvidia.com/gpu=all]`. One RTX 4070 is assumed.

## Repo notes

This working tree is a local-only git repo (no network remote). A
public copy lives at `Projects/Public/GpuBroker-public` and is the
GitHub-facing sync target. `Assist/sync_to_public.sh` is the
sync wrapper. That split is intended.

GpuBroker-Proxy is a subdirectory of this repo. It shares this git
history. It is not a separate project. Flux2Klein9B and Needle are
separate local repos under `~/Documents/Code/Docker/`.

Recent commits (newest first):

- 2026-10-11 audit follow-up: `32101a0` through `435e337` (never-fit 422,
  sweep stays alive, nvidia-smi off `_lock`, waiter wakeup, LlamaSwap
  removed, Proton preempt reasons, same-caller race, proxy heartbeat
  and disconnect release, dashboard `/status` only, TI-Template and
  example contract, hashed image pins, container-only watchdog, Host
  allowlist). See `git log 15c0864..HEAD`.
- `ab745d7` Record the multi-tenant memory-isolation question
- `2b8c9aa` Note the stale admission formula in the README
- `5f288e1` Document the CORS and Talkie-backend changes
- `a94fe45` Allow read-only browser access to the lease ledger
- `5c173ae` Register the Talkie OpenAI service as a proxy backend
