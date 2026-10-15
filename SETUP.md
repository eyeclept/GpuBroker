# GpuBroker Setup

This guide sets up the broker, the two tightly-integrated clients, and
the optional proxy. Set up the parts in this order. Each part depends
on the one before it.

1. GpuBroker (the broker)
2. Tightly-integrated clients (Flux2Klein9B, Needle)
3. GpuBroker-Proxy (optional, only for Ollama traffic)

## Prerequisites

Confirm these before you start.

1. Confirm `podman` and the `compose` plugin are installed.
2. Run `podman compose version`.
3. Confirm the host has one NVIDIA GPU with CDI passthrough.
4. Run this command:
   ```bash
   podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu:24.04 nvidia-smi
   ```
5. Confirm the command prints a GPU table, not an error.
6. Stop every GPU client on the host, including Ollama and any game.

## Part 1: GpuBroker

### Step 1: Measure the safety margin

The safety margin covers desktop compositor and driver memory. That
memory shows in `nvidia-smi` as used. No client can free it.

1. Confirm every GPU client is stopped.
2. Run this command:
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
   ```
3. Add a small buffer, for example 256 MB, to the reported value.
4. Write down this number. You need it in the next step.

The `.env.example` default of 1536 is a placeholder. The live host
uses 4864 after measurement.

### Step 2: Configure the broker

1. Change directory to the GpuBroker repository root.
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` in an editor.
4. Set `GPU_SAFETY_MARGIN_MB` to the number from Step 1.
5. Set `OLLAMA_URL` to `http://host.containers.internal:11434` if you
   do not plan to deploy GpuBroker-Proxy.
6. Skip the previous item if you do plan to deploy the proxy. Part 3
   sets the correct value later.
7. Leave every other value at its default unless you have a reason to
   change it. Each variable has a comment in `.env.example`.

Broker environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `GPU_BROKER_PORT` | `8792` | Broker listen port. |
| `GPU_SAFETY_MARGIN_MB` | `1536` | Idle GPU memory. Measure it. Do not guess. |
| `GPU_BROKER_LEDGER_PATH` | `/data/leases.json` | Lease ledger path inside the container. |
| `OLLAMA_URL` | code: `http://host.containers.internal:11434` | Ollama `/api/ps` poll target. `.env.example` sets `http://ollama:11434`. |
| `OLLAMA_POLL_INTERVAL_SECONDS` | `20` | Ollama footprint poll interval. |
| `GPU_BROKER_SWEEP_INTERVAL_SECONDS` | `30` | Stale-lease sweeper interval. |
| `GPU_BROKER_LEASE_TTL_SECONDS` | `180` | Heartbeat timeout. A silent lease is reclaimed after this. |
| `GPU_BROKER_ADMIN_TOKEN` | empty | Shared secret for `POST /admin/clear`. Empty disables the endpoint (returns 501). Set a secret with `openssl rand -hex 32` before you use that route. The route kills every GPU compute process except the broker, including processes no lease owns. Call with `dry_run: true` first and review `would_kill`. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated origins for GET CORS. Not in `.env.example`. `server.py` reads it. CORS allows GET only. |
| `GPU_BROKER_ALLOWED_HOSTS` | empty | Extra Host names, comma-separated, appended to `localhost`, `127.0.0.1`, `::1`, and `host.containers.internal`. A disallowed Host returns 400. |

The code default for `OLLAMA_URL` is `http://host.containers.internal:11434`.
`.env.example` sets `http://ollama:11434` because compose joins `ollama_default`.
That external network must exist even if you skip the proxy.
`podman compose up` fails if the network is missing.

Do not open host port 8792 in the firewall.
Containers reach the broker through `host.containers.internal`.
LAN access is closed by design.

The stale-lease sweeper accepts a host pid or a container-namespace pid.
It signals only a GPU compute process that runs in a container.
A host match or an ambiguous match removes the lease and sends no signal.

`POST /admin/clear` also needs `pid: host` on the `gpubroker` service.
`docker-compose.yml` already sets that. The endpoint kills every GPU
compute process `nvidia-smi` lists except the broker itself. That includes
processes no lease owns: games, desktop renders, and host Talkie.
It then resets all leases with no allowlist. Call it with `dry_run: true`
first and review `would_kill`.

### Step 3: Start the broker

1. Confirm network `ollama_default` exists (`podman network ls`).
   Create it, or start Ollama first, if the network is missing.
   Compose declares it as `external: true`.
2. Run:
   ```bash
   podman compose up -d
   ```
3. Wait 10 seconds for the health check to pass.
4. Run:
   ```bash
   curl http://localhost:8792/health
   ```
5. Confirm the response is `{"status":"ok"}`.
6. If it is not, run `podman logs gpubroker-gpubroker-1` and read the
   error before you continue.
7. Run:
   ```bash
   curl http://localhost:8792/status
   ```
8. Confirm `budget_mb` is a positive number and `leases` is an empty
   object.

The image healthcheck probes `localhost:8792` inside the container. It
does not probe the published host port. A dead port mapping can still
read healthy. Confirm with `curl` from the host, as above.

## Part 2: Tightly-integrated clients (Flux2Klein9B, Needle)

Both Flux2Klein9B and Needle already call `/acquire`, `/release`, and
`/heartbeat`. This part only sets the connection values and rebuilds
each image.

For a new service, copy `TI-Template/` and follow
`TI-Template/README.md`. Size `mb` from measured peak use. See
`examples/sizing_a_request.md`.

### Step 1: Configure Flux2Klein9B

1. Change directory to the Flux2Klein9B repository root.
2. Open `.env`. If `.env` does not exist, copy `.env.example` to `.env`.
3. Confirm `BROKER_URL` is `http://host.containers.internal:8792`.
4. Confirm `BROKER_FAIL_OPEN` is `true`. This lets Flux2Klein9B serve
   requests without a lease if the broker is down.

### Step 2: Configure Needle

1. Change directory to the Needle repository root.
2. Open `.env`. If `.env` does not exist, copy `.env.example` to `.env`.
3. Confirm `BROKER_URL` is `http://host.containers.internal:8792`.
4. Confirm `BROKER_FAIL_OPEN` is `false`. Needle fails closed by
   default. A broker outage then blocks a new model load.

### Step 3: Rebuild and restart both clients

1. In the Flux2Klein9B repository root, run:
   ```bash
   podman compose up -d --build
   ```
2. In the Needle repository root, run:
   ```bash
   podman compose up -d --build
   ```

### Step 4: Verify lease coordination

1. Send one generation request to Flux2Klein9B.
2. While the request is in progress, run:
   ```bash
   curl http://localhost:8792/status
   ```
3. Confirm the `leases` object has one entry with `caller` equal to
   `flux2klein9b`.
4. Wait for the request to finish.
5. Run the same `curl` command again.
6. Confirm the `leases` object is empty. This confirms the client
   released the lease after eviction.

## Part 3: GpuBroker-Proxy (optional)

Deploy this part only if you want Ollama traffic gated by the same
broker. Skip this part if you do not run Ollama on this host.

### Step 1: Confirm Ollama's network name

1. Run:
   ```bash
   podman network ls
   ```
2. Confirm a network named `ollama_default` exists.

### Step 2: Move Ollama off the host port

1. Change directory to Ollama's repository root.
2. Open `docker-compose.yml`.
3. Remove the `ports:` entry that publishes `11434:11434` to the host.
4. Leave every other line unchanged. Ollama still listens on `11434`
   inside its own container network.
5. Run:
   ```bash
   podman compose up -d
   ```
6. Confirm port 11434 is no longer bound on the host:
   ```bash
   curl --max-time 3 http://localhost:11434/api/version
   ```
7. This command must fail with a connection error. A successful
   response means the port publish is still present.

### Step 3: Join the broker to Ollama's network

1. Change directory to the GpuBroker repository root.
2. Open `docker-compose.yml`.
3. Confirm the `gpubroker` service has a `networks:` list with
   `default` and `ollama_default`.
4. Confirm the file has a top-level `networks:` block that declares
   `ollama_default` as `external: true`.
5. Open `.env`.
6. Set `OLLAMA_URL` to `http://ollama:11434`. This reaches Ollama by
   service name. The poll is read-only telemetry. It must not go
   through the proxy.
7. Run:
   ```bash
   podman compose up -d --build
   ```

### Step 4: Configure and start the proxy

1. Change directory to `GpuBroker-Proxy/`.
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Confirm `BROKER_URL` is `http://host.containers.internal:8792`.
4. Confirm `registry.json` has a default Ollama entry with `base_url`
   set to `http://ollama:11434`.
5. Run:
   ```bash
   podman compose up -d --build
   ```

Use `--build`, not `restart`. `registry.json` is copied into the image
at build time. A restart reloads the stale baked-in copy. The startup
log line `loaded N backend(s)` reports how many entries the process
actually loaded.

Proxy environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `GPUBROKER_PROXY_PORT` | `11434` | Proxy listen port. Takes over Ollama's host port. |
| `BROKER_URL` | `http://host.containers.internal:8792` | Broker URL for `/acquire` and `/release` calls. |
| `BROKER_FAIL_OPEN` | `true` | Forward unmetered if the broker is unreachable. |
| `BROKER_ACQUIRE_TIMEOUT_S` | `10.0` | How long a gated request waits for admission. |
| `PROXY_RETRY_AFTER_SECONDS` | `3.0` | `Retry-After` value on HTTP 429. |
| `PROXY_REGISTRY_PATH` | `/app/registry.json` | Path to the baked-in registry. |

### Step 5: Verify the proxy

1. Run:
   ```bash
   curl http://localhost:11434/health
   ```
2. Confirm the response is `{"status":"ok"}`.
3. Run:
   ```bash
   curl http://localhost:11434/api/ps
   ```
4. Confirm this returns Ollama's own model list format, not a 404.
5. Run:
   ```bash
   curl http://localhost:8792/status
   ```
6. Confirm `ollama_mb` is a number. This confirms the broker's
   passive poll reaches Ollama on `ollama_default`.

### Step 6: Verify existing Ollama consumers

Any service that pointed at `host.containers.internal:11434` before
this setup needs zero configuration changes. Headroom is one such
service.

1. Send one request to that service through its normal path.
2. Confirm the response is the same as before this setup.
3. A genuine Ollama-side error, such as out of memory, is not a proxy
   failure.
4. If the request fails with a connection error, recheck Step 2 and
   Step 3.

## Troubleshooting

- **Broker returns 504 on every `/acquire` call.** The GPU already
  exceeds the configured budget. Run `curl http://localhost:8792/status`
  and compare `reserved_mb + ollama_mb` against `budget_mb`. Free GPU
  memory. Increase `GPU_SAFETY_MARGIN_MB` only if the current value is
  wrong, not as a workaround for genuine contention.
- **A tightly-integrated client proceeds without a lease.** Check
  `BROKER_FAIL_OPEN` in that client's `.env`. A value of `true` allows
  this. This is expected when the broker is unreachable, not a bug.
- **The proxy returns 502 for every request.** The registry
  `base_url` does not resolve. Run
  `podman exec <proxy-container> python3 -c "import socket; print(socket.gethostbyname('ollama'))"`
  and confirm it returns an address. If it fails, confirm both the
  proxy and Ollama are on `ollama_default`.
- **A registry change has no effect.** Rebuild the proxy image. Do
  not only restart the container.
- **`podman ps` shows the broker healthy, but port 8792 refuses
  connections.** The in-container healthcheck did not see the dead
  publish. Probe `http://localhost:8792/health` from the host.

## Updating pinned dependencies

Broker and proxy images install from `requirements.txt` with
`--require-hashes`. FreeToken pins `pip==26.2.1` and the wrapper
packages by version. It does not pin hashes for `freetoken[accel]` or the
CUDA and torch stack. A git URL install cannot use `--require-hashes`.

### Broker and proxy

The base digest is in `Dockerfile`. The proxy file uses the same digest.

1. Create a scratch directory.
2. Resolve linux/amd64 wheels inside the pinned image.
3. Drop `jeepney` when you refresh `GpuBroker-Proxy/requirements.txt`.
4. Convert the JSON report to `==` pins with `--hash=sha256:` lines.
5. Copy that file next to the Dockerfile and rebuild.

```bash
mkdir -p /tmp/pin-scratch
podman run --rm --arch amd64 \
  -v /tmp/pin-scratch:/out:Z \
  python:3.12-slim@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b \
  python3 -m pip install --no-cache-dir --dry-run --ignore-installed \
    --report /out/report.json \
    fastapi "uvicorn[standard]" httpx jeepney
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("/tmp/pin-scratch/report.json").read_text())
rows = []
for item in report["install"]:
    meta = item["metadata"]
    archive = (item.get("download_info") or {}).get("archive_info") or {}
    hashes = dict(archive.get("hashes") or {})
    sha = hashes.get("sha256")
    raw = archive.get("hash") or ""
    if not sha and raw.startswith("sha256:"):
        sha = raw.split(":", 1)[1]
    if not sha and raw.startswith("sha256="):
        sha = raw.split("=", 1)[1]
    if not sha:
        raise SystemExit(f"no sha256 for {meta['name']}=={meta['version']}")
    rows.append((meta["name"], meta["version"], sha))
rows.sort(key=lambda r: r[0].lower())
lines = []
for name, version, sha in rows:
    lines.append(f"{name}=={version} \\")
    lines.append(f"    --hash=sha256:{sha}")
Path("/tmp/pin-scratch/requirements.txt").write_text("\n".join(lines) + "\n")
PY
```

To bump the base image, pull `python:3.12-slim`. Copy a registry
`RepoDigests` value into both Dockerfiles.

### FreeToken

1. Keep `FROM` on the CUDA tag plus digest in `FreeToken/Dockerfile`.
2. Pin `pip==26.2.1` on the upgrade line, or a newer pip the image accepts.
3. In a throwaway container from that image, install `python3` and `python3-pip`.
4. Run `python3 -m pip install --dry-run --ignore-installed --report` for `httpx fastapi uvicorn`.
5. Write every resolved `name==version` into `FreeToken/requirements.txt`.
6. Do not add `--require-hashes` on that install.

The `freetoken[accel]` git commit stays on its own pin in the Dockerfile.
Do not add hashes for torch or the CUDA wheels that `freetoken[accel]` pulls.

