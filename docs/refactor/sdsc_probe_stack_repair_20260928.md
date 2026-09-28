# Diagnostic observer crash repair

Job **54485969** failed with accounting **FAILED / 1:0** after **11m33s**
on exp-19-13. Its worker exited **-11 (SIGSEGV)** after 600.307789 seconds.
This is distinct from job 54472139's earlier SIGTERM and startup delay; it was
not an allocation timeout or a measured failure of prompt-v7 quality.

The wrapper successfully published and hash-verified its failure report,
progress, ledger and logs. The baseline completed **28/32** attempts and the
candidate arm completed **0/256**. The last in-flight prompt was baseline
`pgpair-35642e952a781cbd71e4-neg`. No v7 coverage estimate exists.
The report SHA is
`97cc8834738b5c5b24949f3216b5bcd75f1c7599b94aa2763939d87c1a72731a`.
Bounded verified evidence is under `.sdsc/fetched/54485969/diagnostic-verified/`.

## Reproduction and causal evidence

The second 300-second asynchronous faulthandler dump stopped in the middle of
printing a frame, immediately after Torch `Linear.forward` / `Module._call_impl`.
Its file ends with an incomplete `File` line. The timer text `Timeout (0:05:00)`
describes the observation interval, not Slurm walltime or a scientific timeout.
The preceding 300-second dump showed imports in progress. Durable stage times
record 455.85 seconds in scientific imports, 49.20 seconds in model loading,
and 2.85 seconds transferring the model to its device.

A bounded Quest CPU A/B test reproduced the observer-induced failure using
real Torch 2.8.0+cu128, hidden CUDA, one CPU thread, and a tiny Linear/GELU model:

- With `dump_traceback_later(0.005, repeat=True)`, the child exited **-11**, and
  its stack file likewise ended mid-frame in a Torch call chain.
- With that timer disabled, the same workload exited **0**, completing **61,131**
  finite forwards in five seconds.

The harness is `.sdsc/diagnostics/segv-54485969/faulthandler_cpu_repro.py`;
original logs and report are under `.sdsc/diagnostics/segv-54485969/first/`.
Harness SHA:
`712e4fce3272c4984cb01bb1cf242a9084f86ff1e6ea28a330a63da9b45bad0c`;
report SHA:
`7c8d9309a79485127e57288bc76e6f41952bf872d47d8dc5932045e7d976242b`.
This reproduces the mechanism, not every remote condition: local Python is
3.12.14 versus remote 3.12.13, and the CPU stress interval is shorter than the
GPU job's 300 seconds. A fresh complete GPU diagnostic remains necessary.
The upstream [CPython report](https://github.com/python/cpython/issues/116008)
also describes unsafe cross-thread faulthandler traversal with mid-frame
segmentation faults; it is supporting context, not proof of an identical build
or a claim that any particular remote Python patch is installed.

## Minimal repair and preserved contracts

Remove periodic `dump_traceback_later` and signal `faulthandler.register`
callbacks. Preserve ordinary Python signal handling, synchronous exception
tracebacks, atomic stage/counter snapshots, each candidate's ledger fsync, and
the wrapper's bounded TERM/KILL/drain and persistent publication. A hard native
hang or SIGKILL cannot guarantee a traceback or final worker report; only
previously saved progress and committed ledger records survive reliably.

Do not alter Python/Torch dependencies, CPU-thread allocation, model revisions,
prompt bytes, sampling/RNG, parser/verifier or thresholds. Execute the complete
original **32 v5 baseline + 256 v7 candidate** loop, without reusing partial
responses. The approved diagnostic envelope remains one H100 / 24 CPU /
192 GiB / at most 60 minutes. A tiny CPU test is not GPU/scientific acceptance.

Verification: 31 worker tests passed, including real CPU forward with the
production observer and unchanged Python/Torch RNG, actual TERM/KILL handling,
interrupted atomic progress and the complete 288-generation fixture. Independent
wrapper/boundary checks passed 21 tests; retargeted observer/continuation checks
passed 50. Independent review found all scientific top-level function ASTs and
52 execution-safety hashes unchanged. Ruff, compilation and diff checks passed.

## Recovery and continuation

Both flows for 54485969 are terminal: observation finished at 21:49:08Z and
automatic continuation stopped at 21:54:06Z on September 27, with
`child_started=false`. No successor training job or continuation claim exists.
Their plans, receipts, failure results and earlier stopped flows are preserved.

Following review and tests, use a new content-hashed release, matching sync and
submission dry-runs, a fresh intent and the actual returned job ID. Bind new
finite observation and automatic-continuation plans to that exact receipt.
Never restart an old flow, retry an unknown submission or change scientific
gates. Only complete execution/publication and **32/32** candidate coverage can
trigger the authorized continuation; supplemental validation, original formal
readiness, independent acceptance, a full teacher store and matching preflight
still precede student training. Actual new job/flow state belongs in the
canonical handoff and submission records.
