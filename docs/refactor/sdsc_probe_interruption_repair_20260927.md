# Teacher diagnostic interruption repair

## Evidence and scope

The user explicitly authorized fixing the failure and submitting a fresh run.
Diagnostic 54472139 remains FAILED / 1:0 after 28m37s. Its wrapper received
SIGTERM after source/model staging, GPU-node storage verification and metadata
validation. It did not publish a completed candidate measurement. Coverage is
unknown; this is not evidence that prompt-v7 failed the scientific quality gate.
The `B:TERM@60` request permits an early warning before the 30-minute limit;
accounting has `Reason=None`, so the signal sender is not independently proven.

The wrapper, runtime/allocation guards and model-loading code were unchanged
from successful v5/v6 diagnostics; the worker retained the reviewed prompt and
observation changes. Three hypotheses were examined: Torch/native initialization, dataset hashing,
and tokenizer initialization. Hidden-GPU checks on the SDSC login node reproduced
slow Torch imports at both one and 24 threads. The 35-second stacks moved through
Python import metadata lookups; an earlier 40-second stack was inside native
extension loading. These observations support slow startup, not a demonstrated
thread deadlock, and do not prove where the old GPU worker spent its allocation.
The longer bounded test completed Torch import after **162.15 seconds with
2.62 CPU seconds**, then returned from the requested set-thread operation;
the combined Torch/Transformers check reached its 180-second deadline during
later imports. This is consistent with substantial filesystem/startup wait;
the test did not establish full runtime or GPU readiness.
No CPU-thread, dependency, CUDA, model or environment change follows from them.
Raw observations are under `.sdsc/diagnostics/prompt-v7-recovery/`.

## Reproduced software defects and repair

Real child-process tests reproduced two independent evidence-loss defects:

- The worker's default SIGTERM action bypassed `finally`. Termination during
  model loading left no report; termination after two candidates left only the
  ledger, without the interrupted stage or partial-result qualification.
- The wrapper stopped reading its stdout pipe before waiting for a terminated
  worker. Shutdown diagnostics remained in that pipe and were not published.

The worker now saves atomic, flushed stage/timing snapshots before expensive
operations, and fsyncs each completed candidate before updating its counter.
SIGTERM/SIGINT trigger a C-level faulthandler trace and controlled Python cleanup
when the interpreter can execute it. A periodic stack dump also covers native
calls. Interruption reports stay failed; partial counts are not quality estimates
and never substitute for a complete baseline/candidate reduction. Temporary
progress files are removed on interrupted replacement so failure reporting does
not mask the original signal.

The wrapper records staging and worker timings, drains shutdown output within
fixed bounds, and preserves the final tail after terminating its own child.
The top-level failure report includes the small identity-checked progress record;
the full ledger, timeline and stacks remain on persistent storage under the
job's hashed artifact publication. Bounded fetch still excludes raw ledgers and
large artifacts. SIGKILL cannot run Python cleanup: only already written evidence
is guaranteed recoverable, and no successful completion is inferred from it.

## Fresh run and continuation boundary

The reviewed prompt diagnostic limit is now **one H100, 24 CPUs, 192 GiB and at
most one hour**, account `nwu181`, partition `nairr-gpu-shared`, QoS
`nairr-gpu-shared-normal`. Both the Quest and remote validators enforce it.
The capability diagnostic retains its separate 30-minute bound. Historical v5/v6
runs spent roughly 8–10 minutes outside their measured generation loops, and
the 30-minute v7 run was interrupted. The extra margin and durable stage evidence
make one fresh bounded recovery run useful; they do not establish that the
underlying startup slowdown has been fixed. There is no automatic rerun.

The v7 instruction, original first 32 training prompts, exact v5 baseline,
32+256 serial generations, model revisions, 24-thread execution, candidate RNG,
sampling, parser, verifier and every scientific threshold remain unchanged.
Multiple GPUs would change this serial diagnostic's execution without evidence
of benefit, so this recovery still requests one.

Use a fresh source release, matching sync/submit dry-runs, a new submission
intent and the actual `sbatch --parsable` receipt. Preserve job 54472139 and its
finished observer/stopped automatic flow. A new observer and separately reviewed
automatic-continuation binding must target the actual new job/run/source hash;
never re-arm an old flow or erase its claims.

Only accounting COMPLETED / 0:0, verified persistent publication and unchanged
256-candidate coverage of all **32/32** prompts may trigger the already
authorized Quest continuation. Supplemental validation, original readiness,
independent implementation/acceptance, a complete fresh teacher store and matching
preflight still precede student training. The current handoff and real submission
and supervision records are authoritative for whether the fresh run is active.
