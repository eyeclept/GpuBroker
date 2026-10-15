#!/bin/sh
# Mock nvidia-smi for docker-compose.test.yml's gpubroker-test service.
#
# That service intentionally has no `devices: [nvidia.com/gpu=all]` (see the
# compose file's comments) and so has no real nvidia-smi binary at all --
# server.py's startup() calls _nvidia_smi_query("memory.total") unconditionally
# and fails closed (raises) if that fails, which would prevent the test
# instance from starting. This script is bind-mounted over /usr/local/bin/
# nvidia-smi inside that container only (see docker-compose.test.yml volumes)
# so the test broker can boot and its admission formula can be exercised
# without ever touching the real GPU. It never runs against or is mounted
# into the live gpubroker-gpubroker-1 container.
#
# Mirrors the exact argv shapes server.py invokes:
#   nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
#   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
#   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits

case "$1" in
    --query-gpu=memory.total)
        echo "12288"
        ;;
    --query-gpu=memory.used)
        echo "1536"
        ;;
    --query-compute-apps=*)
        # Empty stdout == "no compute processes", a normal nvidia-smi state.
        # _list_gpu_compute_processes() treats this as an empty list, not an
        # error. Only reachable via /admin/clear, which stays disabled (501)
        # on the test instance since GPU_BROKER_ADMIN_TOKEN is unset there.
        ;;
    *)
        echo "mock nvidia-smi: unsupported query '$1'" >&2
        exit 1
        ;;
esac
