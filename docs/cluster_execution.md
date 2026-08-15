# Cluster execution

Copy `.env.example` to an untracked `cluster.local.env`, fill site-specific values, and source it
before submitting. Templates never embed an account, partition, email, username, environment name,
or scratch path. Arrays map a checked-in ordered cell list and seed list to one run per task.

Every production CLI should first be invoked with `--dry-run`. The report prints resolved config,
model, dataset, prompt/token estimates, output directory, and production classification. Model size,
token budget, or step count beyond smoke thresholds requires `--confirm-production`. Passing that
flag confirms scale only; readiness must still be `GO`.

Training templates request a pre-timeout signal and the trainer catches `SIGTERM` to write an atomic
checkpoint. Resume from the newest fully written checkpoint. Do not place source, secrets, or final
artifacts in node-local scratch; use scratch only for a staged cache and copy validated artifacts to
the configured output root.

Suggested sequence: generate immutable splits, calibrate behavior success, build the shared bank,
score teacher prefixes, run factorial arrays, run canonical baselines, produce compatibility reports,
discover on the frozen circuit split, exact-patch on validation, run local forks, and aggregate.

