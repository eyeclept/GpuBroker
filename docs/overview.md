# GpuBroker Project Overview

## Project Structure

```
.
├── server.py              # Main broker service (FastAPI application)
├── Dockerfile             # Container image build spec
├── docker-compose.yml     # Container orchestration
├── .env                   # Environment variables (generated from .env.example)
├── .env.example           # Template for environment variables with defaults
├── Assist/
│   ├── base.py            # Canonical formatting reference
│   ├── init.sh            # Baseline smoke test script
│   └── TODO.md            # Task tracking and requirements
├── data/                  # Runtime ledger persistence (bind mount, .gitignored)
├── docs/
│   └── overview.md        # This file
└── tests/
    └── test_server.py     # Unit tests for broker logic
```

## Server Module Map

`server.py` implements a FastAPI-based VRAM budget lock with five HTTP endpoints:

### Endpoints

- **`POST /acquire`** — Request GPU VRAM lease
  - Input: `{caller: str, mb: float, timeout_s: float}`
  - Output: `{granted: true, lease_id: str}` or `HTTPException(504)`
  - Logic: Admission formula checks if `max(nvidia_smi_used_mb, reserved_mb + ollama_mb) + requested_mb <= budget_mb`
  - Features: Idempotent by caller name, fair ordering via sequence numbers, exponential backoff polling

- **`POST /release`** — Return GPU VRAM lease
  - Input: `{lease_id: str}`
  - Output: `{released: true/false}`
  - Logic: Removes lease from ledger, persists to disk

- **`POST /heartbeat`** — Keep a lease alive
  - Input: `{lease_id: str}`
  - Output: `{ok: true/false}`
  - Logic: Updates `last_seen` timestamp to prevent expiry

- **`GET /status`** — Query broker state
  - Output: JSON with `budget_mb`, `gpu_total_mb`, `safety_margin_mb`, `reserved_mb`, `ollama_mb`, `nvidia_smi_used_mb`, `leases` ledger

- **`GET /health`** — Liveness check
  - Output: `{status: "ok"}`

### Key Functions

- `_nvidia_smi_query(field)` — Subprocess call to nvidia-smi for live GPU memory queries
- `_live_used_mb()` — Async wrapper for nvidia-smi memory.used (fails closed on error)
- `_persist_ledger_sync()` / `_persist_ledger()` — Atomic JSON ledger write to `/data/leases.json`
- `_reserved_mb()` — Sum of all active lease MB
- `_poll_ollama()` — Background task polling Ollama's `/api/ps` every 20s
- `_sweep_expired_leases()` — Background task reclaiming leases not heartbeated within 180s
- `startup()` — Lifespan startup hook: reads GPU total memory, restores ledger, launches background tasks

## Install and Run

### Prerequisites

- Docker/Podman with NVIDIA GPU support (CDI passthrough)
- `.env` file with environment variables (template at `.env.example`)

### Quick Start

```bash
# Copy template and customize (at minimum, measure GPU_SAFETY_MARGIN_MB)
cp .env.example .env

# Start broker in background
podman compose up -d

# Check health
curl http://localhost:8792/health

# View broker status and ledger
curl http://localhost:8792/status

# Stop broker
podman compose down
```

### Environment Variables

See `.env.example` for full list. Key variables:

- `GPU_SAFETY_MARGIN_MB` — Baseline idle GPU memory (measure empirically)
- `OLLAMA_URL` — Ollama API endpoint (default: `http://host.containers.internal:11434`)
- `GPU_BROKER_LEASE_TTL_SECONDS` — Lease heartbeat timeout (default: 180s)
- `GPU_BROKER_PORT` — Broker API port (default: 8792)

## Running Assist/init.sh

The baseline smoke test verifies the broker can start without a live GPU:

```bash
bash Assist/init.sh
```

This:
1. Checks Python syntax and imports (`python3 -c "import server"`)
2. Builds the Docker image (`podman build -t gpubroker:test .`)
3. Exits 0 if both succeed

Use this during development to catch syntax errors and build failures without needing a running broker.

## Testing

Run unit tests (requires pytest):

```bash
python3 -m pytest tests/test_server.py -v
```

Tests cover:
- Admission formula logic
- Idempotent acquire (dedup by caller)
- Fairness via sequence numbers
- Lease sweeper reclaiming stale leases
- Ledger persistence round-trip

Tests mock `nvidia-smi` subprocess calls and Ollama HTTP polls — no live GPU or network dependency.

## Verification Checklist

Once wired to Flux2Klein9B and Needle clients:

1. `podman compose up -d` and confirm `/status` shows empty ledger with correct budget
2. Restart broker while a client holds a lease; confirm ledger is restored
3. Kill a client without release; confirm sweeper reclaims the lease within 180s
4. Run two clients concurrently; confirm one queues and neither OOMs
5. Stop broker entirely; confirm both clients proceed (fail-open), then restart and confirm normal operation

See `plan.md` Verification section for full details.
