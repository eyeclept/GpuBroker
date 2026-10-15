#!/usr/bin/env bash
set -euo pipefail

# This script shows GpuBroker's public lease API using curl: GET /health,
# GET /status, POST /acquire, POST /heartbeat, POST /release.
# It does not call POST /admin/clear (token-gated; omitted on purpose).
# It expects a live GpuBroker instance at $BROKER_URL and will create
# and release a real lease when executed.
# Usage: ./raw_protocol.sh or BROKER_URL=<url> ./raw_protocol.sh

# Default broker URL (host-side scripts). Containers should use http://host.containers.internal:8792
BROKER_URL="${BROKER_URL:-http://localhost:8792}"
LEASE_ID=""

_release_lease() {
  if [ -n "${LEASE_ID:-}" ]; then
    echo "Releasing lease $LEASE_ID"
    curl -s -X POST "$BROKER_URL/release" \
      -H "Content-Type: application/json" \
      -d "{\"lease_id\": \"$LEASE_ID\"}" \
      -w "HTTP %{http_code}\n" || true
    LEASE_ID=""
  fi
}
trap _release_lease EXIT

echo "Using BROKER_URL: $BROKER_URL"
echo ""

# (1) GET /health -- Verify broker is alive
# Response: 200 with {"status": "ok"}
echo "=== (1) GET /health ==="
echo "Verify broker is alive."
curl -s -w "HTTP %{http_code}\n" "$BROKER_URL/health"
echo ""

# (2) GET /status -- Inspect current broker state
# Response: 200 with {
#   "budget_mb": float,
#   "gpu_total_mb": float,
#   "safety_margin_mb": int,
#   "reserved_mb": float,
#   "ollama_mb": float,
#   "nvidia_smi_used_mb": float,
#   "unmanaged_heavy": bool,
#   "proton_game_retry_after_seconds": int,
#   "leases": { lease_id: {"caller": str, "mb": float, "acquired_at": float, "last_seen": float} }
# }
echo "=== (2) GET /status ==="
echo "Check budget, total GPU VRAM, and active leases (keyed by lease_id)."
curl -s -w "HTTP %{http_code}\n" "$BROKER_URL/status" | head -20
echo ""

# (3) POST /acquire -- Request a GPU VRAM lease
# Request body: {"caller": str, "mb": int, "timeout_s": float, "priority": "P2"}
# Response 200: {"granted": true, "lease_id": "<uuid>", "warning": "<optional one-shot>"}
# Response 504: {"detail": "timeout waiting for GPU headroom"} Retry-After: 10
# Response 503: {"detail": "unmanaged_gpu_load"} Retry-After: up to 1800
#            or {"detail": "too_many_leases"} Retry-After: 5
# Response 422: invalid input, or never-fit:
#   {"detail": "request N MB exceeds maximum grantable capacity of M MB"}
# Response 400: {"detail": "invalid priority tier ..."}
# The broker already waited up to timeout_s on 504. Do not retry immediately. See handling_refusal.py.
echo "=== (3) POST /acquire ==="
echo "Request a 2048 MB lease (single call; captures body and HTTP code together)."
ACQUIRE_RESPONSE=$(curl -s -X POST "$BROKER_URL/acquire" \
  -H "Content-Type: application/json" \
  -d '{"caller": "example-client", "mb": 2048, "timeout_s": 30.0}' \
  -w "\n%{http_code}")
HTTP_CODE=$(echo "$ACQUIRE_RESPONSE" | tail -1)
ACQUIRE_BODY=$(echo "$ACQUIRE_RESPONSE" | sed '$d')
LEASE_ID=$(echo "$ACQUIRE_BODY" | jq -r '.lease_id // empty')

if [ -z "$LEASE_ID" ]; then
  echo "Acquire failed with HTTP $HTTP_CODE: $ACQUIRE_BODY"
  echo "Broker denied or timed out. Not proceeding with heartbeat/release."
  exit 1
fi

echo "Granted. Lease ID: $LEASE_ID (HTTP $HTTP_CODE)"
echo ""

# (4) POST /heartbeat -- Keep the lease alive
# Request body: {"lease_id": str}
# Response: 200 with {"ok": false} for an unknown lease, else
#   {"ok": true, "preempt": bool, "reason": str}
# A preempt-flagged holder must release within 30 s or the watchdog reclaims it.
# Call periodically (every 20-30 seconds is typical) while holding the lease,
# or the broker's sweeper will reclaim it after GPU_BROKER_LEASE_TTL_SECONDS (default 180s).
echo "=== (4) POST /heartbeat ==="
echo "Send heartbeat to keep lease $LEASE_ID alive."
curl -s -X POST "$BROKER_URL/heartbeat" \
  -H "Content-Type: application/json" \
  -d "{\"lease_id\": \"$LEASE_ID\"}" \
  -w "HTTP %{http_code}\n"
echo ""

# (5) POST /release -- Return the lease early
# Request body: {"lease_id": str}
# Response: 200 with {"released": bool} (always 200, idempotent, safe to call from finally blocks)
# If lease_id is unknown or already expired, returns {"released": false}, not an error.
echo "=== (5) POST /release ==="
echo "Release lease $LEASE_ID."
RELEASED_ID="$LEASE_ID"
_release_lease
echo ""

# (6) GET /status (final) -- Confirm lease is gone
# Response: 200 with the updated status; the released lease should no longer appear in "leases".
echo "=== (6) GET /status (final) ==="
echo "Confirm lease $RELEASED_ID is gone from the ledger."
curl -s -w "HTTP %{http_code}\n" "$BROKER_URL/status" | grep -q "$RELEASED_ID" && echo "WARNING: Lease still in ledger" || echo "Confirmed: Lease removed."
echo ""

echo "=== Script complete ==="
