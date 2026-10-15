# TI-Template: Tightly-Integrated GPU Service Broker Client

## What is a tightly-integrated GPU service

A tightly-integrated GPU service coordinates its load and evict cycle with the GpuBroker VRAM lease API. A passive backend does not call the broker. The broker only observes that backend.

An active service calls the broker when it loads a model onto the GPU and when it frees that memory. The three calls are acquire, release, and heartbeat. That sequence stops two services from loading models that together exceed free VRAM.

## Lease state

The caller stores `lease_id`. Pass that value into `_broker_release` and `_broker_heartbeat`. The modules keep no module-level lease global.

The async and sync files share names and behavior. They differ only in I/O. The async file uses `httpx.AsyncClient` and `asyncio.sleep`. The sync file uses `httpx.Client` and `time.sleep`.

## Acquire, release, and heartbeat

**Acquire** asks for VRAM before any load. The broker grants a lease id or refuses. This template sends `timeout_s` of 30 seconds. The broker blocks up to that time.

A 200 returns `AcquireResult(lease_id, warning)`. If the response includes `warning`, the client logs it at WARNING. That field is a one-shot watchdog notice from a previous reclaim.

**Release** returns VRAM after the model is off the GPU. Call it after the free, for example after `torch.cuda.empty_cache()`. Do not call it when you only decide to evict.

**Heartbeat** keeps a live lease alive. Call it on a background tick every 20 to 30 seconds while you hold a lease. The broker sweeper reclaims a lease that misses heartbeats for longer than the lease TTL (default 180 seconds). The default preemption grace is 30 seconds.

`_broker_heartbeat` returns `HeartbeatResult(ok, preempt, reason)`:

| `ok` | `preempt` | Meaning | Action |
|---|---|---|---|
| `True` | `False` | Lease is alive | Keep the model loaded |
| `True` | `True` | Broker wants this lease gone | Unload, then release, before the 30 s grace ends |
| `False` | any | Lease is gone | Re-acquire before any GPU use |
| `None` | any | Transport error | Retry on the next tick. Do not treat the lease as lost |

## Integration checklist

### Step 1: Choose the client

Copy `async_broker_client.py` if your request path is async. Copy `sync_broker_client.py` if your request path already blocks. Do not mix the sync client into an async request path.

### Step 2: Adapt the template

Copy the file into your service as `broker_client.py`. Then:

1. Set `CALLER_NAME` to a unique name such as `"my-inference-service"`.
2. Set `BROKER_URL` in `.env` (default `http://host.containers.internal:8792`).
3. Leave `BROKER_FAIL_OPEN` at `false` unless your service can run with no broker.

Use a fixed caller string if the service has one load slot.

### Step 3: Wire acquire, heartbeat, and release

Acquire must finish before the load writes VRAM. Release must run after you free VRAM. Store `lease_id` in your own state.

```python
lease_id = None

async def load_model(mb: int) -> None:
    global lease_id
    result = await _broker_acquire(mb)
    if result is None:
        # BROKER_FAIL_OPEN: load without a lease
        load_weights()
        return
    lease_id = result.lease_id
    try:
        load_weights()
    except Exception:
        await _broker_release(lease_id)
        lease_id = None
        raise

async def heartbeat_tick() -> None:
    global lease_id
    if lease_id is None:
        return
    hb = await _broker_heartbeat(lease_id)
    if hb.preempt:
        unload_weights()
        await _broker_release(lease_id)
        lease_id = None
        return
    if hb.ok is False:
        await _broker_release(lease_id)
        result = await _broker_acquire(MB)
        lease_id = None if result is None else result.lease_id
        return
    # hb.ok is True or None: keep going, or retry next tick

async def unload_model() -> None:
    global lease_id
    unload_weights()
    await _broker_release(lease_id)
    lease_id = None
```

Drop `await` for the sync client. Map the plain exceptions to HTTP in your service. See FastAPI mapping below.

### Step 4: Environment variables

```
BROKER_URL=http://host.containers.internal:8792
BROKER_FAIL_OPEN=false
```

### Step 5: Verify

1. Start your service and trigger a load.
2. Call `curl http://broker:8792/status`.
3. Confirm your caller name appears under `leases`.
4. Trigger an evict. Confirm the lease is gone from `/status`.

## Broker API endpoints

### POST /acquire

Request:

```json
{"caller": "my-service", "mb": 5120, "timeout_s": 30.0}
```

| Status | Body / header | Client result |
|---|---|---|
| 200 | `{granted, lease_id, warning?}` | `AcquireResult` |
| 504 | `Retry-After` (broker sends 10) | `BrokerNoRoom` |
| 503 | `detail` is `unmanaged_gpu_load` or `too_many_leases`, plus `Retry-After` | `BrokerBusy` |
| 422 or 400 | invalid input, size that can never fit, or bad priority | `BrokerRejected` |

The broker already waited up to `timeout_s` before a 504. The template does not retry 504, 503, 422, or 400. Your service may retry 504 later. The broker keeps wait age for 120 seconds after a 504.

### POST /release

Request: `{"lease_id": "abc123"}`. Response: `{released: bool}`. Always HTTP 200. Idempotent.

### POST /heartbeat

Request: `{"lease_id": "abc123"}`. Response is `{ok: false}` for an unknown lease. A live lease returns `{ok: true, preempt: bool, reason: str}`. Always HTTP 200.

## BROKER_FAIL_OPEN

Default is `false` (fail closed) in both templates.

A tightly-integrated service that loads weights without a lease can OOM the GPU and take down other holders. Fail closed unless your service has a fallback that does not need the GPU.

`BROKER_FAIL_OPEN` applies only after transport retries fail (connect, timeout, and similar). It does not apply to 504, 503, 422, or 400. Those are explicit broker answers. A 503 `unmanaged_gpu_load` during a game must not load weights.

Set `true` only when a missing broker should not fail the service. Flux2Klein9B uses `true` because it can still generate, with OOM risk on collision. Needle uses `false` because uncoordinated inference can crash the runtime.

## Names the templates export

- `BROKER_URL`, `BROKER_FAIL_OPEN`, `CALLER_NAME`
- `AcquireResult`, `HeartbeatResult`
- `BrokerNoRoom`, `BrokerBusy`, `BrokerRejected`, `BrokerUnavailable`
- `_broker_acquire(mb) -> AcquireResult | None`
- `_broker_release(lease_id) -> None`
- `_broker_heartbeat(lease_id) -> HeartbeatResult`

Each call opens an `httpx` client in a context manager and closes it before return.

Retry on transport errors: 5 attempts, backoff 0.5 s, then double, cap 5 s. Explicit denials do not retry.

## FastAPI mapping

The templates raise plain exceptions so they work outside FastAPI. A FastAPI service can map them like this:

```python
from fastapi import HTTPException

def raise_http(exc: BrokerError) -> None:
    if isinstance(exc, BrokerNoRoom):
        retry = str(int(exc.retry_after or 10))
        raise HTTPException(
            status_code=503,
            detail="GPU memory full",
            headers={"Retry-After": retry},
        )
    if isinstance(exc, BrokerBusy):
        retry = str(int(exc.retry_after or 5))
        raise HTTPException(
            status_code=503,
            detail=exc.detail,
            headers={"Retry-After": retry},
        )
    if isinstance(exc, BrokerRejected):
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    raise HTTPException(status_code=503, detail="broker unreachable")
```

## Error handling

- `_broker_acquire` raises `BrokerNoRoom`, `BrokerBusy`, `BrokerRejected`, or `BrokerUnavailable`. It returns `None` only when `BROKER_FAIL_OPEN` is true and the last transport retry fails.
- `_broker_release` never raises. It logs a warning and continues.
- `_broker_heartbeat` never raises. Read `ok` and `preempt` on every tick.

## Reference implementations

Flux2Klein9B is async and sets `BROKER_FAIL_OPEN=true`. Needle is sync and sets `BROKER_FAIL_OPEN=false`. Both are production integrations. Study both if the choice is unclear.
