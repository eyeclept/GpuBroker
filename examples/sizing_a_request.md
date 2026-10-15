# How to Size the MB Parameter for /acquire

## The Problem: Sizing from Model Weights is Not Enough

When you request a lease via `POST /acquire`, the `mb` parameter must reflect your service's
true peak VRAM demand. Many integrators size this from a model's documented static weights and
then hit CUDA OOM later despite holding a valid lease from the broker.

The Whisper service in this repo learned this the hard way. Its lease size (`WHISPER_ACQUIRE_MB=5120`)
was based on the large-v3 model weights (float16: 4.5 to 5 GB), as documented in its config.
In practice, when transcribing a 35-minute audio file, the service:

- Held 4012 MB resident
- Had only 1862 MB physically free
- Failed to allocate more memory, raising `RuntimeError: CUDA failed with error out of memory`

The real peak demand was at least `4012 + 1862 = 5874 MB`, a full 754 MB more than the lease
it requested. Static model weights do not account for working memory.

## Why the Gap Exists

VRAM usage includes more than model weights:

- **Batch/context buffers**: inputs and outputs for each inference
- **KV cache**: for transformer models, grows with sequence length
- **Decode buffers**: temporary allocations during forward/backward passes
- **Grad accumulation**: if training or fine-tuning (rare in inference services)

Together, these can easily exceed 500 MB on top of resident weights. Your service's actual peak
demand is model weights plus working memory, and that difference varies by workload characteristics
(audio length, batch size, input shape) and model architecture (KV cache depth/width).

## How to Measure Peak Usage

Do not guess. Run your service against a representative workload and measure the live peak.

### Recipe: nvidia-smi Sampling

1. Start your service in its normal operating mode.

2. Have a separate terminal ready to poll GPU memory while work runs.

3. Begin the representative workload. This should be:
   - Realistic input size and duration (not a trivial one-second test)
   - Representative of typical load in production (typical audio length, typical batch size, etc.)

4. In the polling terminal, run this loop every 1-2 seconds:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
   ```
   Or if isolating a single-consumer process is impractical, poll total GPU memory:
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
   ```

5. Let the workload run to completion. Record every sample.

6. Find the maximum observed value across the entire run. This is your peak usage.

7. Add a margin of your choice (suggest 256-512 MB as a buffer for variance across inputs)
   and use that as your `*_ACQUIRE_MB` env var.

### Example

If you observe a peak of 5874 MB and choose a 256 MB margin, set `WHISPER_ACQUIRE_MB=6130`.

## Important Distinctions

### Your Margin vs. The Broker's Margin

The broker has its own `GPU_SAFETY_MARGIN_MB` (typically 4-5 GB), which accounts for baseline
idle GPU usage: desktop compositor, driver reserves, stray system processes. Do not reuse that
value for your client-side request sizing. They serve different purposes:

- **Broker margin**: "how much GPU memory must stay free for system stability"
- **Your margin**: "how much buffer do I add to my measured peak to account for variance"

### The Lease Does Not Guarantee Allocation

A granted lease means the broker will not hand that much VRAM to another client. It does NOT
guarantee your process can allocate that much. If you under-size your lease and your actual
working memory grows, CUDA will still refuse to allocate, even if the lease is valid. This is
why accurate measurement matters: your lease size should match what you actually use, not what
you think you use.

### Multi-Model Scenarios

If your service can load different models at different times (or multiple models concurrently),
measure the peak for each scenario in isolation, then request the maximum of all peaks. Do not
try to average or estimate; a single peak is the scenario that matters most.

## When Measurement is Not Practical

If you cannot run the service standalone with deterministic workloads (for example, if your
service only runs inside a larger container orchestration and is fed random live traffic),
start conservatively:

- Begin with static weights plus an optimistic estimate (e.g., +20% for working memory)
- Monitor production for CUDA OOM errors
- If OOM occurs despite a valid lease, measure one instance in a test environment, increase
  the lease size, and push the change

Real production deployments often refine this over weeks as you see the actual tail of workloads.
Measurement in a controlled environment beats guessing every time.
