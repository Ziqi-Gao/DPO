# Accepted adapted teacher → student calibration

The accepted teacher from fit `54494742`, qualification `54496291` and independent
CPU audit `54497294` now has a separate student consumer. The original stopped
v3 flows remain historical; do not restart them or substitute this teacher into
their original prerequisites.

`prereg/amendments/qwen3_adapted_student_calibration_v1.json` requires a genuine
implementation commit and a distinct subsequent review-only acceptance commit.
It preserves all 47 accepted producer files and binds the named student
implementation. Acceptance of this protocol is not acceptance of GPU results.

The learned teacher identity is dense checkpoint SHA
`6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7`.
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` remains its base
ancestry. The original nine teacher input files are physically pinned and
preserved. The in-memory SFT adapter only materializes the explicit learned
teacher identity columns required by the existing deterministic demo source;
all 2048 candidates, tokens, log probabilities, masks, seeds and order remain
unchanged. Its 256-token teacher acceptance does not assert historical
128-token teacher readiness. Student evaluation stays at 128 tokens.

## Execution

1. Check existing `.sdsc/submissions/` and supervision states, then the shared
   SSH master through `tools/sdsc check` on the current Quest host.
2. Review/commit the implementation and protocol acceptance, then perform a
   matching `sync --dry-run` and immutable `sync`. Export genuine provenance
   using `tools/sdsc_provenance.py create` with the actual implementation and
   acceptance commits and upload it after the upload dry-run. The bounded local
   history limit is 32 commits; public refs are never moved to bypass the bound.
3. Submit `qwen3-v2-adapted-preflight` with `--teacher-job-id 54496291` and
   explicit resources below. It restores reviewed science, verifies node-local
   and persistent mounts, stages the real selected teacher, and exercises a
   synthetic global-64 FSDP update, full-state restore and adapted teacher
   forward. It produces `adapted-preflight.json` and durable checkpoint evidence.
4. Only after accounting shows every related step COMPLETED/0:0 and published
   results validate, create a fresh release for
   `qwen3-v2-adapted-calibration`, passing that actual `--preflight-job-id` and
   `--teacher-job-id 54496291`. The upstream must match the same protocol,
   scientific source, teacher, runtime and cache. No automatic retry is allowed.
5. Calibration executes the original full-parameter canonical-SFT entrypoint:
   seed 42, 256 unique prompts × 8 accepted demonstrations, global batch 64,
   maximum microbatch 4, 1536-token model input, 120-step/2M-token ceilings,
   validation/checkpoint every 20 steps. Its original strict artifact validator
   must accept before the wrapper can publish success.

Both tasks use account `nwu181`, partition `nairr-gpu-shared`, QoS
`nairr-gpu-shared-normal`, **2 H100, 24 CPUs, 192 GiB**. Preflight is bounded at
`01:00:00`; initial calibration uses `02:00:00`. Use the existing fixed runtime:
`/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12`
and HF cache `/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface`.
All Slurm calls execute remotely over the already authenticated SSH master.

The wrapper checks 256 GiB free node-local disk before copying inputs. It stages
21,350,818 bytes of original teacher evidence and, for calibration, the original
1,097,575,722-byte data family whose bytes were independently regenerated during
teacher isolation. Reusing this dataset does not accept failed teacher job
54345715. Both pinned Hub snapshots remain staged for the unchanged runtime
check; preflight additionally stages the actual learned dense checkpoint.

Outputs are read-back hashed on persistent project storage before the final
receipt. Calibration retains the nine teacher input files under
`artifacts/teacher_demos/`, the initial checkpoint, training checkpoints,
Accelerate state, metrics and reports. No large checkpoint is automatically
fetched to Quest. Status verifies small published bytes and all published file
sizes against the receipt; the allocation performs full large-file read-back.
A future checkpoint consumer must verify large contents again when staging.

Calibration success is **not complete G0, multistep resume equivalence, pilot,
factorial, or a Blackwell execution-class certificate**. The old G0/pilot adapters
still bind the historical teacher; further adaptation and independent review are
required before those stages can consume this learned teacher. Keep every
stage/job/run identity distinct, and reconcile a missing receipt before another
submission.

## CPU review evidence

Before GPU execution, the combined affected CLI/remote/qualification,
new student protocol/adapter/workers and original factorial-completion suite
passed 260 tests. Separate genuine-history export/restore tests passed, including
the 32-commit bound and rejection at 33. Thirteen wrapper regression tests cover
real read-back, tamper rejection, preservation of already published attempts,
nonpublication after semantic failure, and bounded draining of a TERM handler
that emits more than a pipe buffer. Ruff, shell syntax and Git whitespace checks
pass. These tests do not claim any new GPU execution success.
