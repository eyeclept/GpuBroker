# GpuBroker Setup Guide

This guide sets up all three parts of the system: the broker, the proxy,
and the two tightly integrated (TI) clients, Flux2Klein9B and Needle.

Set up the parts in this order. Each part depends on the one before it.

1. GpuBroker (the broker)
2. TI client integration (Flux2Klein9B, Needle)
3. GpuBroker-Proxy (optional, only for Ollama traffic)

## Prerequisites

Confirm these before you start.

1. Confirm `podman` and the `compose` plugin are installed. Run `podman
   compose version`.
2. Confirm the host has one NVIDIA GPU with CDI device passthrough set up.
   Run `podman run --rm --device nvidia.com/gpu=all
   docker.io/library/ubuntu:24.04 nvidia-smi`. This must print a GPU table,
   not an error.
3. Stop every GPU client on the host, including Ollama and any game. This
   is required for step 2 below.

## Part 1: GpuBroker

### Step 1: Measure the safety margin

The safety margin covers desktop compositor and driver memory that
`nvidia-smi` reports as used, but that no client can free.

1. Confirm every GPU client is stopped (see Prerequisites, item 3).
2. Run this command:
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
   ```
3. Add a small buffer, for example 256 MB, to the reported value.
4. Write down this number. You need it in the next step.

### Step 2: Configure the broker

1. Change directory to the GpuBroker repository root.
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` in an editor.
4. Set `GPU_SAFETY_MARGIN_MB` to the number from Step 1.
5. Set `OLLAMA_URL` to `http://host.containers.internal:11434` if you do
   not plan to deploy GpuBroker-Proxy. Skip this if you do plan to deploy
   it. Part 3 below sets the correct value at that point.
6. Leave every other value at its default unless you have a specific
   reason to change it. Each variable has a comment in `.env.example`.

### Step 3: Start the broker

1. Run:
   ```bash
   podman compose up -d
   ```
2. Wait 10 seconds for the health check to pass.
3. Run:
   ```bash
   curl http://localhost:8792/health
   ```
4. Confirm the response is `{"status":"ok"}`. If it is not, run `podman
   logs gpubroker-gpubroker-1` and read the error before you continue.
5. Run:
   ```bash
   curl http://localhost:8792/status
   ```
6. Confirm `budget_mb` is a positive number and `leases` is an empty
   object. This confirms the broker started with a clean ledger.

## Part 2: TI client integration (Flux2Klein9B, Needle)

Both Flux2Klein9B and Needle already call the broker's `/acquire`,
`/release`, and `/heartbeat` endpoints. This part only sets the connection
values and rebuilds each image.

### Step 1: Configure Flux2Klein9B

1. Change directory to the Flux2Klein9B repository root.
2. Open `.env` (or `.env.example` if `.env` does not exist yet, then copy
   it to `.env`).
3. Confirm `BROKER_URL` is `http://host.containers.internal:8792`.
4. Confirm `BROKER_FAIL_OPEN` is `true`. This lets Flux2Klein9B serve
   requests without a lease if the broker is down.

### Step 2: Configure Needle

1. Change directory to the Needle repository root.
2. Open `.env` (or `.env.example` if `.env` does not exist yet, then copy
   it to `.env`).
3. Confirm `BROKER_URL` is `http://host.containers.internal:8792`.
4. Confirm `BROKER_FAIL_OPEN` is `false`. Needle fails closed by default,
   so a broker outage blocks a new model load instead of risking an
   uncoordinated one.

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
6. Confirm the `leases` object is empty. This confirms the client released
   the lease after eviction, not just after the response returned.

## Part 3: GpuBroker-Proxy (optional)

Deploy this part only if you want Ollama traffic gated by the same broker
that gates Flux2Klein9B and Needle. Skip this part if you do not run
Ollama on this host.

### Step 1: Confirm Ollama's network name

1. Run:
   ```bash
   podman network ls
   ```
2. Confirm a network named `ollama_default` exists. This is the network
   Ollama's own compose project creates.

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
   This command must fail with a connection error. A successful response
   means Step 3 did not remove the port publish correctly.

### Step 3: Join the broker to Ollama's network

1. Change directory to the GpuBroker repository root.
2. Open `docker-compose.yml`.
3. Confirm the `gpubroker` service has a `networks:` list with `default`
   and `ollama_default`.
4. Confirm the file has a top-level `networks:` block that declares
   `ollama_default` as `external: true`.
5. Open `.env`.
6. Set `OLLAMA_URL` to `http://ollama:11434`. This reaches Ollama directly
   by its service name, not through the proxy, since this poll is
   read-only telemetry and does not need admission gating.
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
4. Confirm `registry.json` has one entry for Ollama with `base_url` set to
   `http://ollama:11434` and `default` set to `true`.
5. Run:
   ```bash
   podman compose up -d
   ```

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
4. Confirm this returns Ollama's own model list format, not a 404 error. A
   404 here means the proxy did not start correctly, or Ollama is not
   reachable on `ollama_default`.
5. Run:
   ```bash
   curl http://localhost:8792/status
   ```
6. Confirm `ollama_mb` is a number, not stuck at its previous value. This
   confirms the broker's passive poll reaches Ollama through the network
   joined in Part 3, Step 3.

### Step 6: Verify existing Ollama consumers

Any service that pointed at `host.containers.internal:11434` before this
setup, such as Headroom, needs zero configuration changes. Confirm this
directly.

1. Send one request to that service through its normal path.
2. Confirm the response is the same as before this setup, allowing for the
   response containing genuine Ollama-side errors, such as an
   out-of-memory error from a separate GPU consumer.
3. If the request fails with a connection error, not a generation error,
   recheck Step 2 (host port removal) and Step 3 (network join) above.

## Troubleshooting

- **Broker returns 504 on every `/acquire` call.** The GPU already exceeds
  the configured budget. Run `curl http://localhost:8792/status` and
  compare `nvidia_smi_used_mb` against `budget_mb`. Free GPU memory, or
  increase `GPU_SAFETY_MARGIN_MB` only if the current value is wrong, not
  as a workaround for genuine contention.
- **A TI client proceeds without a lease.** Check `BROKER_FAIL_OPEN` in
  that client's `.env`. A value of `true` allows this. This is expected
  behavior when the broker is unreachable, not a bug.
- **The proxy returns 502 for every request.** The registry's `base_url`
  does not resolve. Run `podman exec <proxy-container> python3 -c "import
  socket; print(socket.gethostbyname('ollama'))"` and confirm it returns
  an address. If it fails, confirm both the proxy and Ollama are on
  `ollama_default`.
