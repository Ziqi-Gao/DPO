# OPD Agent Guide

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
