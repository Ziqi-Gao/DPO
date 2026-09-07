# Qwen3-v2 GPU preflight pilot

This deployment slice proposes one scheduler-managed environment and
training-path preflight with four reviewed execution shapes: 1, 2, 3, or 4
GPUs. It does not create four scientific tasks. ServerScheduler chooses one
concrete world size at claim time; that attempt is never resized. One attempt
proves only the world size it actually received.

This document does not authorize G0, seed-42 training, the full three-seed
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
- Allocation: exactly one scheduler-managed profile, with the concrete count
  read from the running manifest and `SERVER_SCHEDULER_GPU_COUNT`; only counts
  1, 2, 3, and 4 are accepted.
- Required checks: the allocated visible logical CUDA devices, NCCL all-reduce,
  pinned offline student and teacher loading, a finite student/teacher forward pass,
  finite canonical-SFT forward/backward and gradients over the exact
  production response mask, a nonzero AdamW parameter update, distinct rank
  prompt shards, rank-zero-only teacher loading, production-shaped FSDP
  save/resume, finite cgroup memory evidence, and measured CPU/GPU peak memory
  evidence.
- Shape contract: one exact 64-sequence global optimizer window, physical
  microbatches no larger than four, and model inputs of exactly 1,536 tokens.
  Rank-local sequence totals are `64`, `32/32`, `22/21/21`, and
  `16/16/16/16`; the three-rank microbatch tail is exactly `2/1/1`.
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
retained for every reviewed shape, is not a request parameter, and does not
change scheduler-provided GPU visibility.
Rank-zero-only teacher loading and inference are synchronized over Gloo before
the remaining ranks enter the next NCCL collective.

The fully trainable student requests the production Accelerate topology:
`FULL_SHARD`, `FULL_STATE_DICT`, transformer-based wrapping of every
`Qwen3DecoderLayer`, and `use_orig_params=false`. PyTorch 2.8 makes the effective
single-rank strategy `NO_SHARD` (equivalent when there is no peer); effective
world sizes 2, 3, and 4 remain `FULL_SHARD`. Reports and checkpoint manifests
bind both values and the observed wrapper count. The optimizer consumes FSDP's
nonempty flat shards without exposing Qwen3's tied embedding/output weights as
invalid original-parameter views. Single-rank full-state save uses
`offload_to_cpu=false, rank0_only=false`; multi-rank save uses rank-zero CPU
offload. Save/resume then restores the concrete same-world execution state.
The handler logs explicit FSDP, student-forward, backward, optimizer-step,
save, and restore phase boundaries so a later failure can be assigned to the
exact training stage.

Student activation checkpointing remains enabled and explicitly uses PyTorch's
non-reentrant implementation. The Transformers 4.56.2 default is reentrant;
with one root FSDP unit it re-entered Qwen3 layers during backward through
parameter views whose full flat-parameter storage had already been released
after forward. The non-reentrant implementation preserves the checkpointed
training-path check without retaining that stale-view failure mode.

No repository snapshot digest is an input. The execution-class descriptor
binds the fixed runtime, its dependency locks, the explicit safety subject,
and only the 52-file execution kernel listed in
`docs/refactor/qwen3_v2_execution_class_certification.md`. Each experiment's
science protocol separately binds the scientific configuration it needs.

The training probe follows the same allocation-neutral batch planner used by
G0. It reserves the exact cross-rank non-padding input-token count before any
backward, applies the same compensation for framework accumulation and rank
averaging, and checks a full-state model/optimizer save and restore. This is a
bounded one-update safety probe. Its evidence qualifies one fingerprinted
elastic execution class; it is not a separate task variant and it is not an
experiment-specific artifact that must be regenerated for every new seed or
job.

## Proposed resource envelope

| Resource | Proposed value |
| --- | ---: |
| Processes | scheduler-chosen 1, 2, 3, or 4 |
| CPU cores | 24 fixed allocation; 24/12/8/6 threads per rank |
| Host memory | 196,608 MiB |
| GPUs | scheduler-managed; `gpu_count` omitted |
| Per-GPU memory reservation | 81,920 MiB |
| Per-GPU utilization reservation | 95% |
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU exclusivity | required |
| Initial runtime estimate | 7,200-second conservative one-GPU cold-start estimate |
| CPU scaling efficiency | 0.0 |

These are conservative admission values, not measured workload consumption.
Every real pilot used as class evidence must report peak GPU allocation and
reservation, process MaxRSS, cgroup peak, runtime, rank thread counts, and
device identities. Every rank's measured allocated and reserved CUDA peak must
remain at or below 81,920 MiB. Central per-count observations, rather than a
request-side hint, provide the timing evidence used for later scheduling
decisions.

## Readiness and evidence boundaries

This checked-in proposal stays disabled. A central operator reviewing whether
to install or enable it must first verify conditions 1-3 outside a charged GPU
allocation. Condition 4 is a later G0 evidence decision, not a preflight
enablement prerequisite:

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
4. **G0 evidence policy.** Candidate E v1's already accepted
   amendment requires distinct real pilots covering world sizes 1, 2, 3, and
   4 before its first G0 request. That is an OPD-specific gate, not a generic
   ServerScheduler rule. The normal request cannot select a count, and OPD
   must not encode one in a parameter, job ID, filename, or configuration.
   The proposed successor instead records the existing accepted W=1/W=2 real
   evidence plus fail-closed W=1/2/3/4 static evidence against one exact
   execution-safety fingerprint. It explicitly retains the unobserved W=3
   uneven-tail and W=4 topology risks and cannot operate until independently
   accepted. Once accepted, later experiments with the identical fingerprint
   reuse that certificate rather than repeating pilots per experiment.

## Independent approval sequence

The following remain separate decisions; none authorizes the next:

1. Complete the handler, deployment bytes, focused tests, and clean immutable
   project commit.
2. Have the central operator review and install the complete registration while
   preserving `enabled = false`.
3. Obtain explicit approval to enable the OPD registration.
4. Prepare a fresh project-owned protocol-v2 outbox request.
5. Obtain explicit approval for central validation and submission of that exact
   GPU preflight request.
6. Validate the terminal `ScientificCompletion` and measured resource evidence
   for the world size actually assigned.
7. Under the accepted Candidate E v1 amendment, complete its distinct W=1/2/3/4
   real-pilot gate. Alternatively, independently review and jointly accept a
   new successor amendment and certificate that explicitly justifies a
   different initial evidence threshold; never reinterpret v1 in place.
8. Treat an accepted execution-class certificate as reusable topology evidence
   only. Separately review and authorize each G0 or later scientific workflow,
   including its config, artifacts, and completion criteria. G0 success remains
   required before any larger seed-42 experiment request.

Installation is not enabling, enabling is not submission, and successful GPU
preflight completion is not permission to run G0 or a formal experiment.

After the runtime/cache and central cgroup gates are satisfied, and only after
the final project commit is clean, prepare (but do not submit) a fresh request:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 \
  -m posttrain_circuits.scheduler_adapter.gpu_preflight_request
```
