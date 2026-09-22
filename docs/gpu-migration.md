# Reproducible GPU container migration procedure

This is an executable migration rehearsal for a new lab deployment, not evidence
that an employer's bare-metal server was migrated. The target container can be
tested before an operator changes external traffic; this repository does not
stop, copy or modify an unknown bare-metal service.

## 1. Capture the source contract and rollback point

Before migration record the existing endpoint contract, model/tokenizer revision,
image or dependency versions, CUDA/driver, dtype, context/concurrency limits,
auth configuration and a small approved canary set. Use fictional prompts here.
Keep the old service and its configuration recoverable; back up the owned data
using the database platform's normal procedure. Never copy employer credentials,
personal data or model weights with unclear distribution rights into this repo.

## 2. Preflight the target

```sh
python scripts/gpu_preflight.py --output results/gpu-preflight.json
# After explicitly pulling the selected official compatible image:
python scripts/gpu_preflight.py --container-image YOUR_CACHED_VLLM_IMAGE
```

The script checks the GPU's name, driver, compute capability, total/free memory,
Linux Docker engine, and (only when requested) GPU visibility inside that cached
image. It uses `--pull=never`, disables networking for the device probe, and does
not download a model. Missing probes fail closed. A 4 GiB free-memory floor is
an initial admission heuristic, **not** a measured Qwen memory requirement.
Container-visible `nvidia-smi` does not prove CUDA kernels or model generation.

As of the official [vLLM GPU requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
checked 2026-09-23, NVIDIA support starts at compute capability 7.5 and Linux is
required (Windows uses WSL). The locally observed RTX 2060 is compute 7.5 with
6 GiB. The selected vLLM/CUDA image, attention kernels and memory budget still
need an actual inference trial; eligibility alone is not compatibility evidence.

## 3. Deploy on separate loopback ports and run real inference

Build the delivery image as in the README. Set **different** `API_KEY` and
`UPSTREAM_KEY`, `VLLM_IMAGE` to an explicitly cached compatible image,
`UPSTREAM_MODEL` (default public `Qwen/Qwen3-0.6B`), and unused `BENTO_PORT` /
`RETRIEVAL_PORT`. Keep keys in the environment, never in recorded shell history.

```sh
python scripts/deploy.py --project gpu-canary --profile gpu --image model-serving-lab:delivery
```

This reuses the existing pinned vLLM service from `compose.gpu.yaml` and the
shared gateway. The delivery overlay selects FP16, context 1024, two model
sequences, and gateway concurrency one as a cautious starting configuration.
FP16 avoids assuming native BF16 support on Turing. These settings are not
claimed to be optimal. Initial model download requires internet and cache space.
For a fully offline canary, preload permitted weights into the model-cache volume.

`deploy.py` requires an actual container GPU probe, distinct credentials and a
cached app/vLLM image. It starts the configured vLLM container, waits for backend
readiness, and requires both non-empty normal generation and a correctly ended
SSE generation with public model alias `qwen`. Live failure never selects the
deterministic backend. Deployment evidence records successful calls, not answer
quality, token throughput or prefill/decode latency.

An initial model download may outlast the 600-second deployment wait. In that
case the attempt fails safely; inspect model cache/download progress and retry.
No inference throughput or GPU migration success is claimed until this step passes.

## 4. Compare before changing traffic

Run the same approved canaries against the old and new endpoints. Compare output
shape, authentication, generation completion, maximum accepted context, errors,
disconnect cleanup and agreed latency/memory budgets. Confirm retrieval IDs and
stored rows after database restart. The current stack smoke test covers protocol
and persistence only; add domain answer evaluation before handling real queries.

With a single 6 GiB GPU, parallel old/new model servers may exceed memory. Arrange
a maintenance window and drain active requests rather than assume blue/green
capacity. Do not terminate the source process automatically. Only after explicit
acceptance should the owner point its reverse proxy or client configuration at
the verified new endpoint.

## 5. Roll back and preserve evidence

For a subsequent release managed by `deploy.py`, a failed canary triggers the
last verified application/vLLM image IDs and repeats readiness/inference/search
checks. The prior model and ports are restored; current environment credentials
remain in effect. The named database and model cache are retained. A failed
rollback is reported explicitly. CI exercises this mechanism with an actual
broken CPU container; that drill is not proof of successful GPU rollback.

For the first bare-metal-to-container switch, rollback is owned by the source
system: restore its previous traffic target and start its preserved service if
needed, then rerun its canaries. The lab cannot infer or operate that unknown
source. Save preflight, deployment, canary and rollback results separately and
label each as actual measurement or an unexecuted plan.
