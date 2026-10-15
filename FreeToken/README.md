# FreeToken

FreeToken (github.com/FlashML-org/FreeToken) serving Ornith-1.5-35B-A3B as a
tightly-integrated (TI) GPU service. A wrapper on port 1919 loads the model on
the first request and unloads it after 60 s idle. The wrapper holds a GpuBroker
VRAM lease only while the model is loaded. Reachable through GpuBroker-Proxy
on port 11434.

## Why this checkpoint

`primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8` was the planned checkpoint
(smallest weight footprint = most KV budget), but it does not load on FreeToken
0.1.2 at all: the compressed-tensors walker passes a raw `Float8_e4m3fn` tensor
into a bf16 fuse group and `torch.cat` dies with "Promotion for Float8 Types is
not supported" during weight loading. This is upstream issue #238 (open). The
proposed fix PR #294 was fetched and tested (`pip install` of the PR head
`e6e2b261`): it does NOT fix this checkpoint — same traceback, shifted line
numbers, proving the patched code ran. Substitution was therefore unavoidable;
the Ornith org's official `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (~20 GB, same
35B-A3B MoE, modelopt MIXED_PRECISION recipe: FP8 attention/GDN linears + NVFP4
rest) loads and serves on unmodified 0.1.2 and keeps the original intent
(minimal weight footprint for maximum KV budget). Revert to the mixed
checkpoint when upstream fixes #238.

One extra wart: FreeToken's own downloader never fetches
`model.safetensors.index.json` into the HF snapshot, then its expert-bank
loader fails with FileNotFoundError on first load of the NVFP4 repo. The file
is fetched once into `./data/cache` (already done here); a fresh cache volume
needs:

```
podman run --rm -v ./data/cache:/root/.cache freetoken-freetoken python3 -c \
  "from huggingface_hub import hf_hub_download; \
   hf_hub_download('ornith-ai/Ornith-1.5-35B-A3B-NVFP4', 'model.safetensors.index.json')"
```

## Model lifecycle

`serve_wrapper.py` is a FastAPI front proxy on `FREETOKEN_PORT` (1919). It
runs `ft serve` as a child on `127.0.0.1:FREETOKEN_INTERNAL_PORT` (1920) and
controls when the child runs:

1. At container start the wrapper holds no lease and runs no child.
2. The first request unloads every resident Ollama model and asks the
   broker for a lease. If the broker answers 504 (no room yet), the
   wrapper backs off (2 s doubling to 60 s, at least the broker's
   `Retry-After`) and retries while the request waits, up to
   `FREETOKEN_ACQUIRE_WAIT_SECONDS` (300). A game denial (503) fails at
   once with the broker's `Retry-After`. After the grant it unloads
   Ollama models again, starts `ft serve`, and waits until the engine is
   ready. Requests that arrive during the load wait for the same load.
3. The wrapper streams each request to `ft serve` and streams the response
   back.
4. After `FREETOKEN_IDLE_SECONDS` (60) with no request in flight, the wrapper
   stops `ft serve`. It releases the lease only after the child exits.
5. The next request starts again at step 2.

Cost: the first request after an unload waits for a full model load. A warm
load on this host takes about 50 s. A first boot also downloads the ~20 GB
checkpoint. `FREETOKEN_LOAD_TIMEOUT_SECONDS` (1800) limits one load. A load
that runs longer is stopped, its lease is released, and the waiting requests
get a 503.

FreeToken has priority over Ollama. `ft serve` sizes its cache from free
VRAM about 6 s after start. A resident Ollama model (for example the
494 MB `nomic-embed-text`) made that check fail with `cache budget too
small`. The wrapper sends `keep_alive: 0` for each model in `/api/ps`
through `FREETOKEN_OLLAMA_URL` (GpuBroker-Proxy). Ollama can load models
again after the load. A failed unload is logged, and the load continues.

Readiness comes from the child's `GET /health` JSON body, not its status code.
`ft serve` always answers 200 there. The body says `"status": "loading"` until
the backend is up. The wrapper waits for `"status": "ok"` together with
`"maintenance": "serving"`. That is the same state that FreeToken's OpenAI
API gate needs before it accepts work. `"status": "error"` stops the load.

The wrapper's own `GET /health` always answers 200 with
`{"status": "ok", "model_state": "unloaded" | "loading" | "ready"}`. It never
starts a load, so the container healthcheck needs only a 30 s start period.

## BROKER_FAIL_OPEN: false (fail closed)

Same reasoning as Needle, not Flux2Klein9B. A loaded FreeToken puts ~22 GB of
expert banks into host RAM and holds a resident GPU working set. There is no
degraded mode. Without broker coordination a load either OOMs this process or
over-commits the card against Ollama, Flux2Klein9B, Talkie, and Needle.

On a broker denial the wrapper does not start `ft serve`. It answers the
waiting requests with HTTP 503 and `Retry-After: 120`. Denials are a 504
(budget full), a 503 `unmanaged_gpu_load` (a game runs), or a broker that
stays unreachable after 5 attempts. Clients retry later. The wrapper does
not hold a queue of retries.

## Yielding to games

The broker treats non-lease VRAM at or above `UNMANAGED_YIELD_OVERHEAD_MB`
(default 4096) as a game. While a game runs, the broker refuses every new
acquire, so a request that needs a load gets a 503. If the model is already
loaded, the broker flags the preemptible lease. The wrapper sees the flag on
its next heartbeat (5 s). It stops `ft serve` at once, even with requests in
flight, and then releases the lease. Those open streams end early. The game
wins. The next request after the game has been gone for 120 s loads the model
again. No manual `podman stop` is needed. See `../README.md` and
`../Assist/TODO.md` for limits (detection latency, sub-threshold games).

If a heartbeat reports a lost lease, the wrapper asks for the lease again with
the child PID. If the broker refuses, the wrapper stops `ft serve`. It never
keeps the child running without a lease.

## Lease sizing (measured, not guessed)

Per `../examples/sizing_a_request.md`, the lease is sized from nvidia-smi
peaks observed during real load+inference cycles (0.5-1 s sampling,
`--query-compute-apps`), NOT from the checkpoint's on-disk size (~20 GB):

- Engine resident after load (hybrid, memory-ratio 0.63): 6944-6964 MB,
  measured across 10+ clean boots on 2026-09-01.
- `FREETOKEN_ACQUIRE_MB` was originally `7348` = 7092 + 256 margin, from the
  pre-fix measurement (peak during a working request under the old
  config). **Found undersized 2026-09-01:** with the request-hang defect
  fixed (Known Upstream Defects #2 below), a peak-during-a-large-request
  measurement was finally possible and found FreeToken's real footprint
  under large-context requests is a stable **7906 MB** (isolated via
  `nvidia-smi --query-compute-apps`; held even after the request
  completes, not a transient spike) -- 558 MB over the then-current
  lease. Initially left unbumped: `7906` also exceeded this host's
  `budget_mb` at the time (7418 MB = 12282 total - 4864
  `GPU_SAFETY_MARGIN_MB`), and raising `FREETOKEN_ACQUIRE_MB` past
  `budget_mb` would crash-loop this service forever at boot
  (`BROKER_FAIL_OPEN=false` here, by design) -- strictly worse than the
  merely-undersized lease, which at least booted and served correctly.
  That was the broker-wide `GPU_SAFETY_MARGIN_MB` gap tracked in
  `GpuBroker/Assist/TODO.md` (previously confirmed against llama-swap
  only, then confirmed against FreeToken directly too). See
  `/home/eyeclept/Documents/WebLLMChats/BUGS.md` for the full incident.
  **Bumped 2026-09-01, same session:** `GPU_SAFETY_MARGIN_MB` was
  re-measured and lowered (4864 -> 3000, per `GpuBroker/Assist/TODO.md`),
  raising `budget_mb` to 9282 MB and clearing the blocker.
  `FREETOKEN_ACQUIRE_MB` is now **8162** = 7906 + 256 margin, the same
  buffer convention as the original derivation, leaving 1120 MB of
  `budget_mb` headroom for other broker-coordinated consumers (Ollama,
  llama-swap) rather than leasing right up to the ceiling.
  When another consumer holds VRAM (Ollama, or a llama-swap model loaded
  via the same `GPU_SAFETY_MARGIN_MB` gap -- see
  `GpuBroker/Assist/TODO.md`), the broker denies the lease and the
  wrapper answers 503 until that changes -- that is the broker doing its
  job.

**Raised further, 2026-09-02:** llama-swap was permanently retired
(Ornith-1.5-35B-A3B via FreeToken replaces the dense/MoE chat models it
served), removing the VRAM-contention concern that forced the 2026-09-01
`--kv-reserve-tokens` backoff from 65536 to 57344. Re-tested 65536 with
contention gone: still missed the cache-budget assertion by a small
margin (~7-46 MB short across two boot attempts), closed by bumping
`--memory-ratio` 0.63 -> 0.65. Verified across 3 consecutive clean boots
(65572-65611 KV tokens each, no collapse) plus real large-context
requests up to 64200 prompt tokens (HTTP 200). New verified context
ceiling: **65611 tokens** (prompt + generation), confirmed via a live
overflow test ("68700 tokens > 65611 maximum"), up from 58665. Real peak
VRAM under the larger KV pool is a stable **8126 MB** (up 220 MB from
7906, consistent with the bigger fixed-size KV allocation: 1.25 GiB vs
1.10 GiB), confirmed identical across two independent large-context
requests. `FREETOKEN_ACQUIRE_MB` is now **8382** = 8126 + 256 margin,
leaving 900 MB of `budget_mb` headroom for Ollama (the only other
broker-coordinated consumer left; llama-swap is retired).

### KV-cache bug (fixed 2026-09-01)

The config shipped through most of 2026-08 (`--memory-ratio 0.62
--kv-reserve-tokens 65536 --moe-cache-size 512`) claimed "~64.8k tokens
of KV capacity" and *did* produce that on a clean boot (60058-64585
tokens measured) — but on a boot where another GPU consumer was already
resident (a llama-swap model, loaded via the `GPU_SAFETY_MARGIN_MB` gap
above), the real allocation silently collapsed to 3547 tokens, and once
observed even lower, to 814. Root cause: `ft serve --help` documents
`--kv-reserve-tokens` as a floor that only applies "before
`--moe-cache-auto` fills experts". With an *explicit* `--moe-cache-size`
(512, chosen as the minimum allowed with prefill overlap), that floor is
silently ignored — real KV cache is just whatever's left of a
`--memory-ratio`-scaled budget (itself computed from *actual free VRAM
at boot*, not the broker-guaranteed lease) after weights and the fixed
512-slot expert cache. Shrink the free-VRAM-at-boot number (contention)
and KV shrinks with it, silently, all the way toward zero.

Fix: `--moe-cache-auto` makes `--kv-reserve-tokens` a real, enforced
floor — MoE cache is sized dynamically from whatever's left *after* the
KV floor and weights, and the engine fails loudly at boot
(`AssertionError: cache budget too small...`) if the floor genuinely
can't fit, instead of silently shrinking below it. Config at the time of
this fix (superseded 2026-09-02 -- see "Lease sizing" above for the
current `--kv-reserve-tokens 65536 --memory-ratio 0.65`):
`--memory-ratio 0.63 --moe-cache-auto --kv-reserve-tokens 57344`.
memory-ratio needed a 0.62 -> 0.63 bump because 0.62 can't fit (MoE
minimum 512 slots + a 65536-token floor) with any margin on this host's
free VRAM at boot (~10 GB clean); reaching the full 65536 target would
have needed pushing ratio to within ~30 MB of the exact ceiling, which
is fragile against ordinary boot-time VRAM jitter — not just the
llama-swap contention case above. 57344 was chosen instead, for real
margin. Measured result across 10+ clean boots: a consistent
57369-57415 KV tokens, no collapse observed. `--moe-cache-auto`'s own
help text says "Not supported for owned-KV models" and this model *is*
GDN-hybrid (owned-KV: `cache_type` resolves to `hybrid_radix`, and the
scheduler logs a separate `mamba-slot`/`mamba usage` counter) — but in
practice it works: it does not error, and it does not hang, when given
an adequate floor. (An earlier apparent "moe-cache-auto hangs every
request, unconditionally" finding during this fix turned out to be a
false positive: repeated `AssertionError`-triggered crash loops from an
earlier, too-large floor value (65536, see above) force-killed `ft
serve` mid-CUDA-teardown multiple times in a row ("did not exit within
30s, killing"), which left the GPU driver state degraded for the next
boot. A clean stop, a rest period, and a fresh restart cleared it —
the same `--moe-cache-auto` config then served requests normally.)
`--num-tokens` (a direct KV-token-count override, independent of
`--moe-cache-auto`) was also tried and works for sizing, but was
dropped in favor of `--kv-reserve-tokens` + `--moe-cache-auto` since
that pairing is the documented, purpose-built mechanism for this.

## Backend choice: hybrid

`ft bench bw` (run once on this host, profile persisted in
`./data/cache/freetoken/benchbw.json`): CPU STREAM read 55.1 GB/s vs PCIe
13.2 GB/s = 3.9x for NVFP4, above the 2.0x threshold → `hybrid` (25.2% of
decode-step expert misses fetched over PCIe, rest computed on the CPU pool).
Offload was also tested and works; hybrid decodes faster.

## Known upstream defects (FreeToken 0.1.2, observed on this host)

All of these were reproduced, not guessed:

1. CUDA graph capture with the default multi-size list (`[1, 2, 4]`) wedges
   intermittently: scheduler spins at 100% CPU + 100% GPU util, memory climbs,
   the progress bar freezes mid-capture, and `/health` stays `loading`
   forever. Workaround baked in: `--cuda-graph-max-bs 1` (single bs=1 graph).
   Capture usually completes in ~1-5 s at boot, but the wedge still recurred
   repeatedly even with this workaround during 2026-09-01 testing (roughly
   half of ~10 restarts wedged at the *same* `bs = 1` capture step and never
   recovered — only a container restart cleared it; longer waits, up to
   several minutes, did not help). `--cuda-graph-max-bs 1` reduces the
   frequency but does not eliminate this defect. With the lazy-load
   wrapper, a wedged load holds its lease until
   `FREETOKEN_LOAD_TIMEOUT_SECONDS` (1800) ends it, or a broker preempt
   or a container restart stops it first.
2. **Fixed 2026-09-01.** Request-time hangs were real and reproduced
   reliably (see the pre-fix history retained below for context). Root
   cause: upstream `FlashML-org/FreeToken` issue #123, an exact match for
   this hardware (checkpoint `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` /
   `nvidia/Qwen3.6-35B-A3B-NVFP4` on `sm_89`; our RTX 4070 reproduces
   theirs on RTX 4090). PyTorch's row-wise FP8 `_scaled_mm` kernel ignores
   the current CUDA stream (`pytorch/pytorch#177651`, fixed upstream in
   torch 2.12) and races/stalls forever inside FreeToken's fused Q/K/V and
   GDN projections on `sm_89` — a genuine stuck CUDA kernel, confirmed via
   `py-spy` (main scheduler thread parked forever inside
   `flashinfer.prefill.paged_run`, GPU pinned at 100% util, no progress).
   Fixed upstream by merged PR #243 (`fix(kernel): avoid row-wise
   _scaled_mm stall on sm_89 with torch<2.12`, commit `58f4b9e`,
   2026-08-29) — routes the fused projections through a per-part
   tensor-wise GEMM instead on affected hardware. Only on GitHub `main`,
   not yet in a tagged PyPI release (`freetoken` is still pinned at
   0.1.2). `Dockerfile` now installs
   `freetoken[accel] @ git+https://github.com/FlashML-org/FreeToken.git@58f4b9e`
   instead of the PyPI wheel. Verified for real, cold-booted on the fixed
   image: 20, 100, 300, 700, and a 9900-prompt-token request (previously
   hung on every attempt at ~2900) all completed normally (`HTTP 200`,
   correct `completion_tokens`, no stuck kernel, `Scheduler is idle,
   waiting for new reqs` between batches). No hang across any tested
   size.

   Pre-fix history (retained for anyone diagnosing a *recurrence*, e.g.
   after an upstream regression or a revert to the PyPI package): the
   hang was not a clean function of prompt size — trivial ~11-token
   requests were reliably safe, results in the 100-700 range were
   inconsistent across sessions, and ~500-700 and ~2900-token prompts
   hung on every attempt. No hard threshold existed; treat any recurrence
   as the same class of intermittent CUDA-kernel-launch race, and check
   first whether the container is still on the git-main-pinned commit
   above before re-diagnosing from scratch.
3. **Resolved by fix (2).** Long-context behavior is no longer blocked by
   the hang: a 9900-prompt-token request (see above) completed cleanly
   post-fix. The overflow check remains verified and clean: an oversized
   prompt returns HTTP 400 `invalid_request_error` /
   `context_length_exceeded` with an explicit count. Ceiling raised
   2026-09-02 (see "Lease sizing" above) from 58665 to **65611** tokens
   (prompt + generation); current example: "prompt is too long: 68700
   tokens > 65611 maximum (prompt + generation); shorten the prompt or
   increase the KV cache budget". No silent truncation, no Ollama-style
   context shift.

## Run

```
cp .env.example .env          # defaults are already tuned for this host
podman compose up -d --build
podman compose logs -f        # wrapper is up when: "Uvicorn running on"
```

The container starts with no model loaded. The first request triggers the
load. The log shows "API server is ready to serve" from `ft serve`, then
"wrapper: model ready". First load downloads ~20 GB into `./data/cache` and
JIT-compiles CUDA kernels (install.md requires a CUDA 13 toolkit with nvcc.
The image uses the `nvidia/cuda:13.2.1-cudnn-devel` base for that). Later
loads take about 50 s.

Stop: `podman compose stop`. `stop_grace_period: 90s` matters. On SIGTERM the
wrapper lets open streams finish for up to 30 s. Then it stops `ft serve`
(SIGTERM, SIGKILL after 30 s) and releases the lease. The 10 s default grace
killed the container before the release and leaked a lease. Only the broker's
180 s TTL sweeper reclaims a leaked lease.

Logs: `podman compose logs -f freetoken` (wrapper lines are prefixed
`wrapper:`).

Direct (bypasses proxy): `curl http://127.0.0.1:1919/v1/models`. This request
also loads the model.

Model state without a load: `curl http://127.0.0.1:1919/health`

## Through the proxy

`GpuBroker-Proxy/registry.json` routes `ornith-1.5-35b-a3b` to
`http://host.containers.internal:1919` (same convention as Talkie at :8101).
The registry is baked into the proxy image at build time:

```
cd ../GpuBroker-Proxy && podman compose up -d --build
```

Then:

```
curl http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-1.5-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":100}'
```

Verify the lease lifecycle on the broker: `curl http://localhost:8792/status`.
Caller `freetoken` shows a lease of 8382 MB while the model is loaded. The
lease goes away about 60-65 s after the last response finishes. It also goes
away when the container stops. The proxy adds its own separate mb=0 fairness
lease around each forwarded named-model request, per its README.
