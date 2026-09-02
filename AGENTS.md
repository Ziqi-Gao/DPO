# OPD Agent Guide

## Mandatory ServerScheduler boundaries

Before changing or running project code, read all of the following files in full:

- `/home/del6500/projects/ServerScheduler/docs/project-integration.md`
- `/home/del6500/projects/ServerScheduler/docs/job-contract.md`
- `/home/del6500/projects/ServerScheduler/docs/project-migration-instructions.md`
- `/home/del6500/projects/ServerScheduler/schemas/job-request-v2.schema.json`

These rules take precedence over operational guidance elsewhere in this file:

- The only writable project roots are `/home/del6500/projects/OPD`, `/data/del6500/OPD`, and `/scr/del6500/OPD`. Do not request or add another writable root; place temporary files under `/scr/del6500/OPD/tmp`. Do not edit ServerScheduler source, configuration, registrations, queues, leases, audit records, performance data, schemas, or service files.
- Do not submit ServerScheduler, Slurm, or other compute jobs. Preparing or validating project-owned configuration and request examples is not permission to submit them. Job submission requires a separate, explicit user-approved action outside the current project task.
- Do not start, stop, enable, disable, restart, or edit system or user services. Report external service dependencies without managing them.
- Do not implement or retain project-local CPU/GPU scheduling. In particular, do not select physical GPUs, poll `nvidia-smi` for placement, create resource locks or leases, manage CPU capacity, fan out work through Screen/tmux, launch detached workers, or build a duplicate queue/retry/status system.
- Do not modify `CUDA_VISIBLE_DEVICES` to choose host GPUs. A future approved foreground entrypoint must consume the scheduler-provided allocation, preserve assigned GPU visibility, remain in the foreground, validate protocol-v2 manifests and allowlisted parameters, and confine outputs to approved OPD-owned paths.
- Treat registration edits, request submission, pilot execution, service changes, legacy-scheduler cutover, and cleanup as separate approval gates. Never infer one approval from another, and never delete scientific outputs or validated completion markers.

The Slurm material below documents safety constraints for a future, separately authorized operation. It does not override the no-submission boundary above and does not itself authorize any job or service action.

## Quest Slurm submission policy

- Request the shortest walltime that has a defensible completion margin. A shorter request can improve backfill opportunities and pending priority; never shorten a job below the time needed to preserve the registered scientific workflow.
- Before an expensive GPU submission, compare plausible walltimes with `sbatch --test-only` while keeping account, partition, GPU count, CPU count, and script identical. Record the estimates in the task log.
- Use the shortest candidate supported by prior runtime evidence. If there is no comparable completed run, include model download/startup, checkpointing, evaluation, and failure-cleanup margin; treat the first successful run as calibration evidence for later requests.
- After submission, inspect `squeue --start`/`squeue` and `scontrol show job`. If the predicted start is poor, reconsider walltime only when a shorter value remains scientifically and operationally safe. Do not submit duplicate jobs merely to probe the scheduler.
- For OPD G0, the initial no-history request is 12 hours on four GPUs. Override with `SLURM_G0_TIME` only when evidence supports a different bound. The production script must export explicit `PROJECT_ROOT`, `PYTHON_BIN`, `ACCELERATE_BIN`, and `OUTPUT_ROOT` values before `sbatch`.
- Before the long G0 job, run the short four-GPU preflight. It must verify a CUDA-enabled project Python, four visible devices, NCCL all-reduce, pinned offline Qwen loading, and a finite real-model forward pass. A CPU-only wheel or missing pinned cache is a submission blocker.
- Keep pinned Hugging Face snapshots in the project cache and set `HF_HOME` plus `HF_HUB_OFFLINE=1` for production jobs. Model downloads belong in CPU/login preparation, not inside a charged GPU allocation.
- A queued job is not a completed experiment. Monitor `squeue`, `sacct`, and logs through terminal state; diagnose deterministic failures before retrying. Submit the seed-42 pilot only when the hash-valid G0 artifact says `passed: true`.
- Never launch the full three-seed factorial or Gemma replication unless the user separately authorizes it.
- Keep this task at no more than four concurrently allocated GPUs. Four-GPU training arrays run one task at a time; one-GPU circuit arrays may run at most four tasks concurrently.
- Qwen3-v2 four-GPU preflight, G0, pilot training, and resume jobs request `--mem=192G`. The preflight must record the finite cgroup-v1 or cgroup-v2 memory limit, peak/MaxRSS, and at least 32 GiB plus 20% headroom; do not infer node memory from CPU count or Slurm defaults.
- Source `scripts/production/slurm_supervision.sh` in new supervisors. Bounded scheduler/accounting retries are required, empty or `UNKNOWN` accounting is never success, and a four-GPU submission must stop if another pending or running OPD GPU job could allocate concurrently.
