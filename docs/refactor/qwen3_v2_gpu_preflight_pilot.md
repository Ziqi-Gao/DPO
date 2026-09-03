# Qwen3-v2 GPU preflight pilot

This deployment slice proposes one bounded four-GPU environment and training-path
preflight. It does not authorize G0, seed-42 training, the full three-seed
factorial, a registration change, a service change, or a scheduler submission.
The complete proposed OPD registration remains `enabled = false` in
`deployments/qwen3_v2_gpu_preflight/registration-proposal-v2.toml`.

## Scientific and execution contract

- Task: `qwen3_v2_gpu_preflight`.
- Request parameters: `workflow_id`, `plan_sha256`, and `unit_id` only.
- Models: `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Prompt protocol: `qwen3_non_thinking_v1`, with `enable_thinking=false`.
- Required result: one `gpu_preflight.json` plus a scheduler-neutral
  `ScientificCompletion` whose handler-owned gates all pass.
- Required checks: four visible logical CUDA devices, NCCL all-reduce, pinned
  offline student and teacher loading, a finite student/teacher forward pass,
  finite soft-teacher forward/backward and gradients, a nonzero parameter
  update, four distinct rank prompt shards, rank-zero-only teacher loading,
  FSDP save/resume, finite cgroup memory evidence, and measured CPU/GPU peak
  memory evidence.
- Retry behavior: every central attempt receives isolated staging. Temporary
  FSDP resume shards belong below `/scr/del6500/OPD/tmp` and must be removed
  before successful completion; a valid published completion is reusable.

The handler fixes all scientific configuration and storage roots. Its child
environment must contain only reviewed values such as:

```text
HF_HOME=/scr/del6500/OPD/cache/huggingface
HF_HUB_CACHE=/scr/del6500/OPD/cache/huggingface/hub
HF_HUB_OFFLINE=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET,COLL
NCCL_P2P_DISABLE=1
PYTHONDONTWRITEBYTECODE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
TMPDIR=/scr/del6500/OPD/tmp
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
```

`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` are not fixed by OPD. They must
be preserved exactly from ServerScheduler, and each rank may select only its
logical device inside that visibility envelope.

The worker uses Gloo as its explicit control plane and an explicit NCCL group
for CUDA tensors and FSDP. Its first operation on the NCCL group is a scalar
all-reduce with a 120-second process-group and work timeout. The report records
the per-rank elapsed time, observed sum, logical CUDA identity, and PCI bus ID.
`NCCL_P2P_DISABLE=1` is a fixed profile workaround for the reproduced first
all-reduce hang on this server's dual-NUMA four-GPU Blackwell topology; it is
not a request parameter and does not change scheduler-provided GPU visibility.
Rank-zero-only teacher loading and inference are synchronized over Gloo before
the remaining ranks enter the next NCCL collective.

No repository snapshot digest is an input. The deployment contract binds the
fixed runtime and implementation bundle, while the workflow binds the exact
scientific configuration and preregistration content needed by this task.

## Proposed resource envelope

| Resource | Proposed value |
| --- | ---: |
| Processes | 4 |
| CPU cores | 16 fixed |
| Host memory | 196,608 MiB |
| GPUs | 4 |
| Per-GPU memory reservation | 81,920 MiB |
| Per-GPU utilization reservation | 95% |
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU exclusivity | required |
| Initial runtime estimate | 1,800 seconds |
| CPU scaling efficiency | 0.0 |

These are conservative admission values for the first bounded run, not measured
workload consumption. The successful pilot must report peak GPU allocation and
reservation, process MaxRSS, cgroup peak, runtime, and device identities. Those
measurements may support a later separately reviewed profile change.

## Three external readiness gates

The proposal must remain disabled until all three external conditions are
verified outside a charged GPU allocation:

1. **Fixed CUDA environment.** A non-editable project runtime exists at the
   reviewed path, its Python executable is a regular non-symlink inode, the
   direct GPU dependency contract is fixed, and the pinned CUDA-enabled PyTorch
   build imports successfully. Prepare it once from the login node with:

   ```bash
   /usr/bin/python3.12 scripts/server_scheduler/prepare-qwen3-v2-runtime.py --execute
   ```

   The preparation script is not a scheduler entrypoint and must never be run
   inside a GPU allocation.
2. **Pinned offline cache.** Both exact Qwen snapshots, tokenizers, and chat
   template bytes are present below the OPD cache. Loading both revisions with
   network access disabled must succeed before a GPU request is prepared.
3. **Central memory cgroup.** ServerScheduler must launch the job in an
   allocation-specific cgroup with a finite 196,608-MiB memory limit and expose
   readable current/peak counters. Admission bookkeeping alone does not satisfy
   the preregistered finite-limit and headroom gate.

## Independent approval sequence

The following remain separate decisions; none authorizes the next:

1. Complete the handler, deployment bytes, focused tests, and clean immutable
   project commit.
2. Have the central operator review and install the complete registration while
   preserving `enabled = false`.
3. Obtain explicit approval to enable the two-task OPD registration.
4. Prepare a fresh project-owned protocol-v2 outbox request.
5. Obtain explicit approval for central validation and submission of that exact
   GPU preflight request.
6. Validate the terminal `ScientificCompletion` and measured resource evidence.
7. Only after a passing preflight, separately review and authorize the G0
   workflow. G0 success is itself required before any seed-42 training request.

Installation is not enabling, enabling is not submission, and successful GPU
preflight completion is not permission to run G0 or a formal experiment.

After the runtime/cache and central cgroup gates are satisfied, and only after
the final project commit is clean, prepare (but do not submit) a fresh request:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 \
  -m posttrain_circuits.scheduler_adapter.gpu_preflight_request
```
