FROM docker.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu22.04@sha256:29f08e1d33ee43759e1882a820f65ecffaead7a5b08994caeac177fe6041b6bb

# install.md: "CUDA kernels are JIT-compiled on first use, need a CUDA 13
# toolkit with nvcc on PATH." The -devel variant (not Flux2Klein9B's
# -runtime, which has no compiler) ships nvcc under /usr/local/cuda/bin,
# already on PATH in this base image. Same 13.2.1 point release already
# validated against this host's r580+ driver by Flux2Klein9B's container.

# ENV values are standalone image fallbacks. Compose `.env` (env_file)
# overrides them at runtime. They are not required to match `.env.example`.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    FREETOKEN_MODEL=ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
    FREETOKEN_SERVED_MODEL_NAME=ornith-1.5-35b-a3b \
    FREETOKEN_HOST=0.0.0.0 \
    FREETOKEN_PORT=1919 \
    FREETOKEN_MOE_BACKEND=hybrid \
    FREETOKEN_EXTRA_ARGS="--memory-ratio 0.62 --kv-reserve-tokens 65536 --moe-cache-size 512 --cuda-graph-max-bs 1"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        build-essential \
        ninja-build \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip==26.2.1

# freetoken[accel] pulls its own CUDA-matched torch; installed directly
# (not via uv) since the container is already an isolated environment and
# does not need uv's venv management on top of it.
#
# Pinned to a git commit, NOT the PyPI release (0.1.2, uploaded 2026-08-19):
# PyPI is stale relative to this pin. Commit 58f4b9e (merged PR #243,
# 2026-08-29) fixes a real, reproduced-on-this-exact-hardware bug --
# FlashML-org/FreeToken#123: on sm_89 (this host's RTX 4070 included;
# PR #243 was tested by its author on an RTX 4070 SUPER, also sm_89) with
# torch < 2.12, PyTorch's row-wise FP8 `_scaled_mm` kernel launch ignores
# the current CUDA stream and stalls forever in FreeToken's fused Q/K/V
# and GDN projections at prompt sizes >= ~256 tokens -- exactly the
# "silent, zero-log hang, GPU pinned at 100% util, frontend poisoned"
# defect this project's own README documents. #123 was reproduced by its
# reporter on this exact checkpoint, ornith-ai/Ornith-1.5-35B-A3B-NVFP4.
# Bump this commit forward whenever a newer tagged PyPI release ships
# freetoken>0.1.2 and includes this fix -- check the fixed commit is an
# ancestor of the release tag before reverting to a PyPI pin.
RUN python3 -m pip install --no-cache-dir \
        "freetoken[accel] @ git+https://github.com/FlashML-org/FreeToken.git@58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6"

# Runtime dependencies of broker_client.py (httpx) and serve_wrapper.py
# (fastapi, uvicorn: the wrapper is the front proxy on :1919). Version
# pins live in requirements.txt. No --require-hashes: the git install
# above cannot supply hashes, and this layer does not lock CUDA/torch.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Lifecycle defaults for the front proxy. A separate ENV after the heavy
# layers: changing it in the first ENV block would invalidate the cached
# multi-GB freetoken[accel] install. Same standalone-fallback rule as above.
ENV FREETOKEN_INTERNAL_PORT=1920 \
    FREETOKEN_IDLE_SECONDS=60 \
    FREETOKEN_LOAD_TIMEOUT_SECONDS=1800

COPY broker_client.py /app/broker_client.py
COPY serve_wrapper.py /app/serve_wrapper.py
WORKDIR /app

# Only the wrapper's port. ft serve binds 127.0.0.1:FREETOKEN_INTERNAL_PORT.
EXPOSE 1919

# Hits the wrapper's own GET /health, which answers 200 in seconds and
# never loads the model. Mirrors the httpx-based healthcheck convention
# used by GpuBroker-Proxy and Flux2Klein9B.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:1919/health', timeout=5.0).raise_for_status()" || exit 1

CMD ["python3", "/app/serve_wrapper.py"]
