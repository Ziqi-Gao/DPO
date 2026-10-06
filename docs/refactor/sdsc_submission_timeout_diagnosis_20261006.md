# SDSC submission acknowledgement timeout diagnosis

The preparation intent `19665dfbd26d7b0024a46d9b7b811ce4` remains unknown.
Its single actual submission started at 2026-10-06T03:22:36.979479Z and the
client timeout was recorded at 03:23:22.045304Z. No actual JobID has been
recovered. Preserve both claims and all original submission evidence.

SSH remained connected. The timeout was imposed by the project's remote
Python controller: `sdsc_student_name_invariant.run` aliases
`sdsc_student_lr.run`, whose default is 45 seconds. The outer SSH operation
has a separate 240-second limit. The observed exception is the former.

The original exception handler loses `TimeoutExpired.stdout` and `.stderr`:
it retains only the exception string. Therefore an empty controller JSON
output does not establish that `sbatch` printed nothing. Five local tests
using actual child processes reproduce this loss, and verify inherited stdin
EOF and removal of `SBATCH_`, `SQUEUE_` and `SACCT_` overrides. Importing the
actual controller leaves the caller's environment and working directory
unchanged. Its admission helpers do not mutate those parent-process values.

Reconciliation has two additional compatibility problems. A UTC calendar date
was interpreted as a future local date by `sacct`; an explicit remote UTC query
corrected that observation. Also this site reports `AccountingStoreFlags=null`
and historical accounting records have an empty Comment, while the frozen
reconciler requires a matching Comment. These defects do not establish the
original submission's outcome, and the old receipt must not be invented.

## Bounded remote diagnosis

All evidence is under
`.sdsc/diagnostics/student-name-invariant-preparation-v1/`.

- At 04:10:07Z, SSH, `squeue`, `sacct` and controller ping succeeded; user queue
  and accounting were empty in the relevant interval.
- The actual binary is the Slurm 23.02.7 ELF. `CliFilterPlugins` and
  `PlugStackConfig` are null. Script, plan and configuration reads at 04:14:45Z
  each took less than one millisecond.
- One explicitly non-submitting `sbatch --test-only -vv` request used the same
  parameters and exact script. At 04:15:43Z it completed in 80.018 milliseconds.
  Its bounded syscall trace records rapid configuration, script, MUNGE and TCP
  operations; the longest observed wait was 29.607 milliseconds for controller
  input. Both claims and the original plan retain identical hashes.
- `54690409` is exclusively that WILL_RUN prediction's temporary ID. It must
  never be used to reconcile the original submission or claimed as a running
  preflight. At 04:17:23Z queue/accounting contained no corresponding job and
  `scontrol show job 54690409` reported an invalid ID.
- Controller thread count was 256 at 04:18:37Z and 3 at 04:19:48Z. In upstream
  23.02.7 this is a dynamic count and 256 is the default ceiling. This supports
  transient pressure at those observation times, without determining the cause
  of the original 03:22 submission timeout. No site binary build limit was
  independently established. Empty DBD/outgoing-RPC queues are not an original
  request rejection certificate.

Slurm's [versioned sbatch source](https://github.com/SchedMD/slurm/blob/slurm-23-02-7-1/src/sbatch/sbatch.c#L257-L298)
separates WILL_RUN from actual submission. WILL_RUN invokes server-side
validation, including job-submit plugins, and can consume a temporary ID;
it neither runs the script nor allocates GPUs. Its success is not a GPU
preflight result or proof that the original submission was rejected.
The same source contains retry/backoff for certain submission errors, so
`--no-requeue` does not mean that the client performs no such processing.

The [controller source](https://github.com/SchedMD/slurm/blob/slurm-23-02-7-1/src/slurmctld/controller.c#L1408-L1581)
explains the thread counter. [Batch submission handling](https://github.com/SchedMD/slurm/blob/slurm-23-02-7-1/src/slurmctld/proc_req.c)
can save and schedule a created job despite a failed acknowledgement send.
Consequently client timeout does not cancel a possibly accepted request.

## Additive diagnostic repair

`tools/sdsc_observed_command.py` captures one caller-authorized command with
bounded stdout/stderr, retaining raw bytes, hashes, timestamps, return status,
timeout and child-reaping evidence. It supplies stdin EOF, filters inherited
Slurm CLI option overrides and permits an explicit UTC override for enumerated
read queries. It never retries or turns a numeric stdout line into a JobID.
Output-limit and timeout observations remain incomplete even if cleanup later
reaches EOF. Cleanup concerns only its own direct child; no scheduler job or
service is cancelled.

This component is additive and is not connected to the accepted submission
controller. All 213 distinct paths covered by the current plan's 203 scientific
files and 29 controls remain byte-identical. The component prevents evidence
loss when explicitly used; it cannot recover the original discarded output or
authorize another submission.

Author verification passes 14 real local-child cases. Non-author review repeats
all14 and adds four real-process boundaries plus five argument rejections, all
passing. A real SDSC login-node fixture at04:27:53Z verifies that a timed-out
child is reaped and both partial streams survive. Subsequent UTC queue and
explicit-cluster/duplicate accounting queries complete with empty outputs.
The initial remote test harness failed an assertion before printing its fixture
result; preserve that record alongside the corrected raw-first v2 harness.
The reviewed helper bytes are identical throughout. Key records are `submission-client-path-independent-review.json`,
`submission-controller-import-env-review.json`,
`submission-test-only-independent-review.json` and
`observed-command-author-validation.json`,
`observed-command-independent-review.json` and
`observed-command-sdsc-rehearsal-v2.json`.

An authoritative server-side outcome would recover the original actual JobID
or establish definitive rejection with no pending request. Ordinary-user queries
and the later WILL_RUN result do not supply that outcome, and the controller log
is unavailable on the login node. This remains the forensic uncertainty; it is
not proof that administrator intervention is the only way to prevent duplicate
scientific execution.

## Proposed recovery without administrator intervention

The exact frozen node
`tools/sdsc_student_name_invariant_job.py`, SHA
`4d9b45d6a5ea6cba717982d32aebc434121c16735a768d8f196e5c050b1ad31f`,
executes `result_root.mkdir(mode=0o700)` without `exist_ok` at line598.
This happens before its main try/finally, runtime staging, CUDA probe, model
loading and optimizer work. The node has no path that deletes this directory,
changes the destination or retries an existing-directory error. Its own missing
receipt/live-binding is not an execution gate and must not be used as one.

A separately authorized fencing operation could exclusively create this exact
old output directory and retain it permanently:

```
/expanse/lustre/projects/nwu181/zgao12/OPD/student-name-invariant-v1/19665dfbd26d7b0024a46d9b7b811ce4
```

This is an atomic competition with the old node, not reliance on an earlier
empty-directory check. Only a successful exclusive creation by that operation
can establish the fence. If the node already created the directory, or any
symlink, existing file, permission error or uncertain result appears, stop;
never insert a marker into an existing directory or adopt it as a fence.
If creation succeeds but proof persistence fails, leave the directory in place
and reconcile the partial operation instead of deleting it and trying again.

The proof must bind the original plan, node and claim hashes, the successful
exclusive creation, timestamp and directory identity, with durable readback.
Use a clearly named retirement/fence record, never a fabricated job receipt,
scientific result or scheduler cancellation. Preserve the original plan,
claims, deployed source, unknown record and all prior evidence. Removing the
fence would invalidate the guarantee and is forbidden while the old outcome
remains unresolved.

Two independent static reviews agree on this gate. Three bounded local CPU
fixtures against the original main function also pass: an existing directory
raises FileExistsError before any downstream work; an absent directory is
created and reaches the immediately following sentinel; two real processes
competing for the same directory yield exactly one success. External pre-gate
Slurm, mount, parent and source validation are fixture substitutions; the mkdir
and filesystem competition are real. These tests are not a remote installed
fence or a successful GPU allocation. See
`submission-fence-static-independent-review.json`,
`unknown-intent-engineering-fence-independent-review.json`,
`submission-fence-main-rehearsal.py` and
`submission-fence-main-rehearsal-result.json` in the diagnostic directory.

Read-only checks at04:37:10UTC still show empty user queue/accounting and an
absent old result directory. Original plan/claims/unknown bytes match.
At04:38:03UTC the deployed node under the release's **source/** directory
matches the frozen hash and the persistent parent chain has no symlinks.
The first observation looked for the node without that source/ component;
its absent-path record is preserved, and the separate correct-path observation
supplies the actual source evidence. Neither observation wins the atomic gate.

The frozen submitter deliberately rejects the consumed scientific claim.
A future execution-only successor therefore needs an explicit reviewed
single-attempt recovery contract tied to the retained fence, a fresh identity,
durable full submission observations and its own receipt. It must preserve all
accepted science, inputs, runtime snapshot, thresholds and prerequisite gates;
removing claims or changing an ID in the old submit path is not recovery.
The new preflight would keep2H100/24CPU/384GiB/1h, account nwu181,
nairr-gpu-shared / nairr-gpu-shared-normal.

Fencing prevents duplicate model work but does not cancel an in-flight scheduler
request. A late old allocation can still consume2GPUs while prolog/import/source
checks run before the gate. Conservatively reserve those2GPUs against the4GPU
ceiling throughout successor admission. Its original1h allocation envelope,
plus any site termination/accounting overhead, remains a possible cost; do not
promise immediate release or zero charges. Cancelling an exact name once would
not by itself prevent an even later request from creating a job.

No fence, cancellation, new submission or claim reset has occurred. Permanently
blocking the old intent is conservatively treated as retiring that task, so the
AGENTS.md separate cancellation authorization applies by interpretation; it
must not be presented as an already authorized ordinary status query. The
alternative is technically supported but its successor controller has not yet
been implemented or accepted. An administrator remains useful for resolving
the historical scheduler outcome, rather than being the sole recovery option.
