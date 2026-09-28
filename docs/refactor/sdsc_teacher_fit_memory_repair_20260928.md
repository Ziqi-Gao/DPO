# Four-H100 teacher preflight: allocation versus workload memory

Job `54493777` completed with FAILED/1:0 after 3m36s on exp-19-08.
Source/model staging, node-local ext4 and input/output Lustre checks passed.
The metadata-only worker failed before model imports or training because the
inherited shared-node guard required an actual cgroup limit of exactly192 GiB.

The submitted request was4 H100,24 CPUs,192 GiB,one hour. Real `scontrol` and
`sacct` distinguish `ReqTRES=cpu=24,mem=192G,gres/gpu=4` from
`AllocTRES=cpu=72,node=1,gres/gpu=4`; the batch step reports memory0.
The `nairr-gpu` partition is exclusive and reports `SelectTypeParameters=CR_CORE`.
Evidence is `.sdsc/diagnostics/teacher-adaptation-v1/four-gpu-memory-failure-accounting.json`.
Failure outputs are `.sdsc/fetched/54493777/fetch-0ua7wqtr/`.

The old combined guard did not save its raw cgroup levels. Consequently, the
record establishes an allocation/request discrepancy but does not establish the
precise kernel limit or exclude inconsistent counters. Slurm documents separate
requested memory and exclusive allocation behavior; this site's actual counters
must be measured rather than inferred from a command argument or generic manual.
See [Slurm configuration documentation](https://slurm.schedmd.com/slurm.conf.html).

The repair is a separate teacher-fit execution contract. It preserves the
requested192-GiB workload budget and requires peak usage no greater than that
budget minus max(32GiB,ceil20%×192GiB), i.e.38.4GiB headroom. It reports the
actual finite effective kernel limit separately and requires it to be at least
the budget. A larger kernel cap is explicitly labeled software-budget-only;
it does not increase the acceptable training peak. Usage comes from this exact
Slurm job and batch step, never a user's historical or whole-node peak.
Unattributable, missing, inconsistent or inadequate evidence fails closed.
Raw hierarchy evidence is persisted even on metadata-only failure.

Actual allocated CPU count is recorded separately from the24 requested CPUs.
The fixed computation policy remains6 threads on each of four ranks. No system
cgroup, scheduler configuration, service or CPU placement is changed. Original
shared-node and central scheduler guards remain untouched. The new19-surface
execution plan invalidates the failed preflight's identity and the unused
reserved full-fit release; new immutable releases and a fresh real preflight
are required. Teacher quality thresholds, data, optimizer and selection rules
are unchanged by this resource-contract repair.

The new preflight `54494477` completed with COMPLETED/0:0 in11m24s on
exp-19-08. All13 execution checks passed, including real four-rank updates,
same-world restore and dense export/reload. The own-job peak was69.75GiB;
the actual finite own-job/batch kernel cap was898.44GiB. Thus the separate
192-GiB workload budget remains conservative and is accurately distinguished
from the kernel cap. Stable64-sequence updates took about4.42seconds.
Actual evidence is `.sdsc/fetched/54494477/fetch-5mpzfgb2/`; report SHA
`a04ea2b34f2c380cb40d7a68c068cf722a142cdd51c239cdc8d25fc124e81ffb`.
This is execution validation only; its development quality gates did not all
pass. Full fit `54494742` was submitted once from the separately staged matching
release after this execution success, with a bounded four-hour limit.
