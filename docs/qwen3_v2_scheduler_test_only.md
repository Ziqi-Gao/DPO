# Qwen3-v2 Quest scheduler calibration (test-only)

Recorded 2026-08-20 on account `p32737`, partition `gengpu`. Every comparison kept the script,
GPU count, CPU count, and 192-GiB memory request identical and changed only walltime. These were
`sbatch --test-only` queries; they did not submit jobs.

| Stage | Walltimes compared | Scheduler estimate for every candidate | Test-only IDs |
| --- | --- | --- | --- |
| four-GPU preflight | 00:20, 00:30, 00:45 | 2026-09-02 11:53:49 on qgpu2012 | 9973745–9973747 |
| four-GPU G0 | 08:00, 10:00, 12:00 | 2026-09-02 11:53:49 on qgpu2012 | 9973750–9973752 |
| four-GPU pilot training array | 08:00, 10:00, 12:00 | 2026-09-02 11:53:49 on qgpu2012 | 9973753–9973755 |
| four-GPU resume validation | 01:00, 02:00, 03:00 | 2026-09-02 11:53:49 on qgpu2012 | 9973756–9973758 |

The estimates did not distinguish the candidates. Until measured Qwen3-v2 runtime exists, retain
00:30 for preflight, 12:00 for G0 and each pilot-training array task, and 02:00 for the two-resume
validation. Twenty minutes does not have a defensible first-run margin for two pinned model loads,
FSDP save/resume, and cleanup; G0/pilot have no comparable completed run from which to defend a
shorter request. The first successful run becomes the calibration evidence for shortening later
requests.

Post-query `squeue` contained only protected Qwen2.5 preflight 9717819. None of the test-only IDs
was queued. Qwen3-v1 preflight 9949357 remained cancelled with zero elapsed allocation.
