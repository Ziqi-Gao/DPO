# Post-Training Circuits

Research code for asking what rewires an instruction-tuned language model: visited states,
supervision information, or the policy-gradient update rule. The controlled study is a 2 x 3 grid:

| Cell | State source | Supervision |
|---|---|---|
| `offline_hard` | shared frozen rollout bank | teacher top-1 at each visited prefix |
| `online_hard` | current policy | teacher top-1 at each visited prefix |
| `offline_soft` | shared frozen rollout bank | teacher top-k forward KL |
| `online_soft_opd` | current policy | teacher top-k forward KL, no reward/PG term |
| `offline_verified_replay` | shared frozen rollout bank | exact-verifier-gated sequence NLL |
| `online_verified_replay` | current policy | exact-verifier-gated sequence NLL |

Canonical verified-demonstration SFT and official-TRL GRPO connect those causal contrasts to common
post-training algorithms. Random-matched and format-only rewards are GRPO controls.

## Quick start

```bash
/usr/bin/python3.12 -m venv /scr/del6500/OPD/envs/opd
/scr/del6500/OPD/envs/opd/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
/scr/del6500/OPD/envs/opd/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock
make test
make validate-configs
make smoke-factorial
```

The tests instantiate a tiny random causal LM from configuration and build a tiny local tokenizer;
they never download production weights. Generated outputs live below
`/data/del6500/OPD/outputs`; caches, environments, logs, and temporary files live below
`/scr/del6500/OPD`.

## Server storage layout

OPD uses three project-owned roots with distinct roles:

| Role | Root |
|---|---|
| Version-controlled code and configuration | `/home/del6500/projects/OPD` |
| Durable datasets, artifacts, results, and checkpoints | `/data/del6500/OPD` |
| Caches, environments, logs, runtime state, and temporary files | `/scr/del6500/OPD` |

Copy `.env.example` to the ignored `project.local.env` when a shell workflow needs the canonical
paths. Keep temporary project files below `/scr/del6500/OPD/tmp`; no additional writable root is
part of the OPD server contract.

## ServerScheduler execution boundary

OPD does not submit, poll, retry, or place compute work itself. Project code may prepare a
content-addressed protocol-v2 request for later, separately approved ServerScheduler submission.
The registered project entrypoint validates the running manifest and allocation, stays in the
foreground while its fixed handler runs, and preserves the scheduler-provided CPU/GPU visibility.
Project configuration must not set `SERVER_SCHEDULER_*` or rewrite `CUDA_VISIBLE_DEVICES`.

The active boundary is `scripts/server_scheduler/opd-entrypoint` plus
`posttrain_circuits.scheduler_adapter`. Production registration, request submission, and pilot
execution are independent approval gates; the repository quick start authorizes none of them.

## Tiny CPU commands

```bash
make validate-configs
make smoke-factorial
make smoke-sft
make smoke-grpo
make smoke-local-fork
make smoke-resume
make smoke-circuits
make smoke-repaired-g0
/scr/del6500/OPD/envs/opd/bin/python -m posttrain_circuits.cli.build_anchor_pilots \
  --output-dir /data/del6500/OPD/outputs/anchor-pilots --seed 42 \
  --discovery-per-task 4 --validation-per-task 4
```

`smoke-grpo` uses the official `trl.GRPOTrainer`, executes one optimizer step on CPU, and refuses to
pass unless the tiny model's parameter-update norm is nonzero. `smoke-circuits` writes both a genuine
activation-space EAP-IG artifact and a held-out exact-patching artifact. `smoke-repaired-g0` runs the
complete three-stage CPU vertical slice and labels its result `passed_cpu_smoke_only`; it is not a
real G0 and does not download Qwen or Gemma.

Production-scale commands print their resolved model, dataset, token estimate, output directory, and
scale classification under `--dry-run`; they require `--confirm-production` to execute. Nothing in
the quick start downloads Qwen or Gemma.



## Frozen scientific-design gates

The active version-controlled design is [`prereg/core_v2.yaml`](prereg/core_v2.yaml), with execution order
in [`docs/core_execution_plan.md`](docs/core_execution_plan.md). Production run manifests record
the frozen preregistration Git commit and SHA-256; a dirty or uncommitted preregistration is refused.
`core_v0.yaml` and `core_v1.yaml` are retained only as scientific history; their datasets, circuit
probes, and run artifacts are rejected by formal core-v2 loaders.

Core-v2 ProofGraph is paired signed entailment. Each pair has the same query, rules, topology,
counts, and ordering, while critical support entails either `Q` or `NOT Q`. Both labels require a
nonempty exact proof, so the query surface cannot determine the answer. Circuit evidence is
stage-specific (`first_rule_selection`, `intermediate_conclusion`, `final_answer`): EAP-IG proposes
candidates and held-out exact activation/path patching supplies the causal validation.

Before preparing a Qwen factorial request, evaluate the initial checkpoint's semantics-preserving
anti-shortcut suite and freeze the `base_capable` and `challenge` discovery/validation probe
manifests. The project preflight and each production factorial training entry point validate the
evidence hashes, model revision, and configured `shortcut_gap` threshold. ServerScheduler remains
the sole owner of allocation and execution state.

```bash
# Read-only command preview; this does not load a production model.
/scr/del6500/OPD/envs/opd/bin/python -m posttrain_circuits.cli.evaluate_anti_shortcut \
  model=qwen25_1p5b task=proofgraph_main --dry-run

# After separate initial-student scoring and learnability-pilot artifacts exist.
/scr/del6500/OPD/envs/opd/bin/python -m posttrain_circuits.cli.build_probe_cohorts \
  --splits-root /data/del6500/OPD/outputs/datasets/proofgraph \
  --scores /data/del6500/OPD/outputs/probes/initial_student_scores.json \
  --initial-checkpoint-hash <resolved-model-commit> \
  --learnability-evidence-hash <frozen-pilot-manifest-hash> \
  --output /data/del6500/OPD/outputs/probes/proofgraph
```

Local forks use matched `KL(output_new || output_fork)` as the primary axis and retain update count
and parameter-update norm as secondary axes. Circuit dynamics subtract same-checkpoint bootstrap
noise as `excess_churn` and require full-score stability, cross-checkpoint mask transfer, and
held-out exact-patching evidence alongside thresholded-mask diagnostics.

The core non-Qwen mini-replication is the six-cell/anchor Gemma scientific plan in
[`configs/replication/gemma2_2b_core.yaml`](configs/replication/gemma2_2b_core.yaml). Any future
execution must be represented by reviewed ServerScheduler workflow units; no active project entry
uses a repository-local scheduler. The plan is not a complete factorial and does not make OLMo or
Edge Pruning a dependency.

## Final CPU acceptance

Run the same gates used by CI, followed by all executable scientific smoke paths:

```bash
make lint
make typecheck
make test-scientific-design
make test
make validate-configs
make smoke-factorial
make smoke-sft
make smoke-grpo
make smoke-local-fork
make smoke-resume
make smoke-circuits
make smoke-repaired-g0
```

## Outputs

Each run contains `resolved_config.yaml`, `config_binding.json`, `manifest.json`, `metrics.jsonl`,
`environment.json`, `git_diff.patch`, `checkpoints/`, and `evaluations/`. Dataset and rollout-bank
outputs add immutable manifests and content hashes. Weights, banks, datasets, checkpoints, W&B
caches, and secrets are excluded by `.gitignore`.

Production manifests also record the Git commit and SHA-256 of the frozen preregistration. A missing,
uncommitted, or modified `prereg/core_v2.yaml` is a hard production refusal.

## Status

The Phase-0 task, controlled trainer, canonical baselines, local-fork workflow, and tiny-model circuit
vertical slice are CPU-testable. Production configurations and ServerScheduler project contracts
are interfaces, not evidence that GPU-scale experiments or the scientific hypothesis have been
validated. See `docs/known_risks.md` and the readiness command before requesting production compute.
