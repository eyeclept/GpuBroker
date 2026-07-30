# TI-Template: Tightly-Integrated GPU Service Broker Client

## What is a Tightly-Integrated (TI) GPU Service?

A tightly-integrated GPU service coordinates its own GPU load and evict lifecycle with GpuBroker's VRAM lease API. This differs from a passive backend service, which the broker only observes without direct coordination.

Passive backends expose an API but do not call the broker. The broker queries them to check their current VRAM use. Active services call the broker to announce when they load models onto GPU and when they free that memory.

A tightly-integrated service uses acquire, release, and heartbeat calls around its own load and evict operations. This coordination prevents two services from loading models that together exceed available VRAM.

## The Acquire-Release-Heartbeat Contract

Three operations coordinate GPU resource use:

**Acquire** requests VRAM before loading a model onto GPU. The broker allocates VRAM from the pool and issues a lease ID. If the broker has insufficient free VRAM, it queues the request and waits for room to open up. The acquire call blocks until the broker either grants the lease or times out (configurable timeout on the broker side, default 300 seconds).

**Release** returns VRAM after the model is removed from GPU. Call this function only after GPU memory is freed (for example, after `torch.cuda.empty_cache()` or equivalent). Do not call release merely when eviction is decided.

**Heartbeat** keeps a lease alive during active use. Call heartbeat periodically on a background tick (for example, every 20 to 30 seconds) only when a lease is held. The broker expires leases that miss heartbeats for longer than the broker's lease TTL (default 180 seconds).

## Integration Checklist

### Step 1: Choose Your Client

Two templates exist: `async_broker_client.py` for async request handlers and `sync_broker_client.py` for synchronous request handlers.

Use `async_broker_client.py` if your service already processes requests asynchronously. Use `sync_broker_client.py` if your service already blocks synchronously during request handling. Do not mix a sync client into an otherwise async request path.

### Step 2: Adapt the Template

Copy the chosen template into your service repository as `broker_client.py` or similar. Edit the top of the file:

- Replace `"REPLACE_ME"` in the `CALLER_NAME` constant with a unique name for your service (for example, `"my-inference-service"`). This name must be unique across all services using this broker instance. Use a fixed string if your service has exactly one concurrent load slot, matching the existing examples.
- Set `BROKER_URL` in your `.env` file to point to the broker (default `http://host.containers.internal:8792` works when the broker runs as a container).
- Decide whether `BROKER_FAIL_OPEN` should be `true` or `false` for your specific service (see below).

### Step 3: Wire Acquire Into Your Load Path

Inside your service's code, wherever you load a model onto GPU, do this:

1. Call `_broker_acquire(mb)` with the VRAM size the model needs, in MB. The function blocks until the broker grants or denies the lease, or the timeout expires.
2. Check the returned `lease_id`. If it is `None` and `BROKER_FAIL_OPEN` is enabled, proceed with the load. If it is `None` and `BROKER_FAIL_OPEN` is disabled, reject the request (the function raises `HTTPException`).
3. Store the lease ID (the sync client variant takes it as a parameter to release and heartbeat).

The acquire call must complete before the load actually places data on GPU.

### Step 4: Wire Release Into Your Evict Path

Wherever you remove a model from GPU and free VRAM, do this:

1. Call the VRAM-freeing operation (for example, `torch.cuda.empty_cache()`).
2. After that call returns, call `_broker_release()` (async variant) or `_broker_release(lease_id)` (sync variant).
3. Clear any stored lease ID state.

Call release only after VRAM is actually freed, not merely when eviction is decided.

### Step 5: Wire Heartbeat Into Your Idle Watcher

If your service runs a periodic idle-watcher or health-check background task, call `_broker_heartbeat()` (async) or `_broker_heartbeat(lease_id)` (sync) on each tick, only when a lease is held.

An acceptable brief blocking call inside an async loop is fine, similar to Needle's own pattern in the existing codebase.

### Step 6: Set Environment Variables

Add these to your `.env` or container environment:

```
BROKER_URL=http://host.containers.internal:8792
BROKER_FAIL_OPEN=true  # or false, depending on your decision (see below)
```

### Step 7: Verify Integration

Once wired, verify the integration:

1. Start your service and trigger a load operation.
2. Query the broker's `/status` endpoint while your service is loading: `curl http://broker:8792/status`.
3. In the response, find the `leases` object. Confirm your caller name shows up with a lease entry.
4. Trigger an evict operation. Query `/status` again and confirm the lease entry has disappeared.

## Broker API Endpoints

The broker exposes three endpoints for acquire, release, and heartbeat.

### POST /acquire

Request:
```json
{
  "caller": "my-service",
  "mb": 5120,
  "timeout_s": 30.0
}
```

Response on grant (HTTP 200):
```json
{
  "granted": true,
  "lease_id": "abc123"
}
```

Response on denial (HTTP 504): The broker blocks internally up to `timeout_s` and returns HTTP 504 if never admitted. Treat 504 as a permanent denial and reject the request immediately. Do not retry.

### POST /release

Request:
```json
{
  "lease_id": "abc123"
}
```

Response (HTTP 200):
```json
{
  "released": bool
}
```

The endpoint always returns 200, even for an unknown lease ID. The operation is idempotent.

### POST /heartbeat

Request:
```json
{
  "lease_id": "abc123"
}
```

Response (HTTP 200):
```json
{
  "ok": bool
}
```

The endpoint always returns 200. A lease that receives no heartbeat for longer than the broker's lease TTL (default 180 seconds) is reclaimed by the broker's background sweeper.

## Choosing BROKER_FAIL_OPEN

The `BROKER_FAIL_OPEN` environment variable controls what happens if the broker is unreachable or denies a lease.

Set `BROKER_FAIL_OPEN=true` if your service has a safe fallback when the broker is unavailable. For example, Flux2Klein9B sets this to `true` because it can still generate images without broker coordination, risking OOM only if resources collide with other services. A missing broker is an availability reduction but not a service failure.

Set `BROKER_FAIL_OPEN=false` if your service must reject requests it cannot coordinate with the broker. For example, Needle sets this to `false` because uncoordinated inference risks crashing the entire runtime. A missing broker is a service failure, not a graceful degradation.

This is a deliberate per-service decision, not a universal default that applies to all services. Document your choice in your own README.

## What Each Client Template Provides

Both templates provide the same three functions and globals:

- `BROKER_URL`, `BROKER_FAIL_OPEN`, `CALLER_NAME` (module-level globals you edit)
- `lease_id` (module-level state for the async variant). The sync variant takes it as a parameter.
- `_broker_acquire(mb: int)` returns `str | None`
- `_broker_release()` (async) or `_broker_release(lease_id: str | None)` (sync), returns `None`
- `_broker_heartbeat()` (async) or `_broker_heartbeat(lease_id: str | None)` (sync), returns `None`

The async and sync variants use identical retry logic and backoff timings. The async variant uses `httpx.AsyncClient` and `asyncio.sleep`, while the sync variant uses `httpx.Client` and `time.sleep`.

## Error Handling

All three functions log their outcomes but handle retries internally.

- `_broker_acquire()` raises `HTTPException(503, ...)` on broker denial or unrecoverable error (unless `BROKER_FAIL_OPEN=true`).
- `_broker_release()` and `_broker_heartbeat()` never raise. They log warnings and continue.

Every exception is logged at the appropriate level (info, warning, or error) before the function returns or raises.

## Notes on the Reference Implementations

Flux2Klein9B (async) defaults `BROKER_FAIL_OPEN=true` and uses async httpx for non-blocking I/O.

Needle (sync) defaults `BROKER_FAIL_OPEN=false` and uses sync httpx because its own request handler blocks during inference anyway.

Both are production integrations. Study both if you are unsure which pattern fits your service.
