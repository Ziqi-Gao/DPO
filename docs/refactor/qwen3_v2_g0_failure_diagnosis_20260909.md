# G0 teacher-demo failure diagnosis

## Confirmed outcome

Job `opd-2947686c51c3e93ad1b18e5a3b7d6b22` ran once on the
`qwen3-v2-g0-elastic` profile with one scheduler-assigned GPU. Central audit
records show a ready runtime scope at 2026-09-09 07:04:25 UTC and terminal
failure at 09:40:33 UTC, exit code 2, with no automatic retry scheduled.

`build_splits` and `export_initial_checkpoint` completed. `build_teacher_demos`
generated all 2,048 candidates (256 prompts, eight fixed-seed candidates each),
then failed the required complete-prompt-coverage gate. The exception lists
243 prompts with no exact-verifier success: 119 positive and 124 negative
prompts. Only 13/256 prompts had any accepted candidate. This is a prompt
coverage count, not an individual-candidate accuracy estimate. Training did
not start.

The scheduler's `gpu_unknown` label classifies the nonzero exit; it does not
establish a hardware fault. The application error explicitly identifies the
teacher-demo scientific gate.

## Diagnostic evidence was deleted

`datasets/teacher_demos/store.py::write_teacher_demo_store` writes the full
attempt ledger, accepted view, and manifest before raising on zero-success
prompts. `scripts/server_scheduler/qwen3-v2-g0-handler.py::_supervise` only
publishes the scientific bundle after all stages succeed, but unconditionally
removes its temporary workspace in `finally`.

Consequently `/scr/del6500/OPD/tmp/qwen3-v2-g0-c0sywxc7` is absent and the
job's durable output staging directory is empty. The retained scheduler logs
contain prompt IDs, but no generated response text or per-candidate verifier
errors. A search of OPD data/scratch found only three fixture teacher-demo
ledgers, not this production ledger.

## CPU/offline checks

The exact first 256 training examples were reconstructed with the pinned
offline tokenizer, without loading a model or using CUDA. All 243 failed IDs
matched the reconstructed population. Every canonical target passes the exact
verifier. Canonical target lengths range from 53 to 162 tokens, all within the
256-token response bound; model-facing prompt prefixes range from 368 to 1,246
tokens, with the maximum exactly at the configured prefix bound.

The executable reproduction and per-example JSON evidence are respectively
`/scr/del6500/OPD/tmp/g0-failure-diagnosis-20260909/reproduce_cpu_diagnosis.py`
and `cpu_diagnosis.json` in that directory. Reproduction command:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/del6500/projects/OPD/src \
TMPDIR=/scr/del6500/OPD/tmp TOKENIZERS_PARALLELISM=false \
/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python \
/scr/del6500/OPD/tmp/g0-failure-diagnosis-20260909/reproduce_cpu_diagnosis.py
```

Thus the evidence does not support malformed canonical proofs or a response
budget intrinsically too short to encode a valid answer. It does not exclude
actual generations wasting their budget on prose or being truncated.

The prompt supplies one placeholder proof line, while the verifier requires
ordered step IDs, exact antecedent citations, correct rule conclusions, and a
nonempty derivation of the appropriate query polarity. Missing instructions
are a plausible contributor, but the deleted responses prevent attribution
among response formatting, reasoning, citation errors, and truncation.
Appending instructions also risks exceeding the current 1,246-token prefix
bound. No prompt or verifier change is justified as an observed fix yet.

## Safe retry preparation

An observability-only repair in `cli/build_teacher_demos.py` preserves failed
ledger/view/manifest files under a separate fixed OPD scratch diagnostics root
and logs aggregate verifier/termination counts. It retains the original
scientific exception and creates no completion marker. Candidate generation,
sampling seeds, prompt population, token limits, and acceptance remain the
same. A retry with this repair is diagnostic; it is not evidence that teacher
coverage has been repaired.

That CLI is outside the execution descriptor's named safety files, so this
repair can reuse the unchanged execution class and deployment. It still needs
a new scientific implementation review. The proposed successor is
`prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_diagnostics_v2.yaml`.
The original accepted science protocol, amendment, descriptor, and certificate
remain unchanged.

Required sequence: agent-created implementation commit, independent review of
that exact commit, review-only science acceptance commit, and generic request
builder with the new protocol. Under the updated central intake contract,
publish the fresh file to the armed outbox and follow the central receipt;
ordinary requests within the approved scope need no manual per-request
authorization or digest handoff. The old accepted job ID must not be reused.
No job was submitted by this diagnosis session.

## Verification of the candidate

The four focused suites cover 25 tests: failed-ledger retention, original-error
preservation, unchanged success behavior, stale-store exclusion, symlink
rejection, ledger contracts, teacher-demo/SFT fixture integration, and CLI
configuration smoke checks. The combined run passed 24 tests and identified
one test assertion that incorrectly rejected this scratch filesystem's
inherited setgid bit. After restricting that assertion to access-permission
bits, the exact affected test passed 1/1. All 25 tests therefore have passing
evidence for the final candidate; production code did not change after the
combined run.

The fixed G0 runtime has no pytest. Test tooling uses an isolated copy of
pytest 8.4.2 and its pure-Python dependencies from the installed Anaconda
environment under the OPD scratch directory below. No fixed runtime was
modified; test discovery disables plugin autoload. The working test command is:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/scr/del6500/OPD/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/home/del6500/projects/OPD/src:/scr/del6500/OPD/tmp/g0-diagnostics-test-tools-20260909 \
/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python -B -m pytest -q \
-p no:cacheprovider \
--basetemp=/scr/del6500/OPD/tmp/g0-diagnostics-tests-20260909 \
tests/unit/test_teacher_demo_failure_diagnostics.py \
tests/unit/test_teacher_demo_ledger_contracts.py \
tests/integration/test_teacher_demo_store.py \
tests/unit/test_config_cli_smoke.py
```

The exact recheck uses the same environment with
`--basetemp=/scr/del6500/OPD/tmp/g0-diagnostics-tests-recheck-20260909` and only
`tests/unit/test_teacher_demo_failure_diagnostics.py::test_failed_store_survives_source_cleanup_with_exact_private_bytes`.

Descriptor recomputation and accepted certificate validation passed with
unchanged fingerprint
`ca27527ea4aa345114bb58859ae39078887a084bc078204c32f8782103381108`.
Original accepted artifacts and the handler are byte-identical to HEAD. The
new science protocol passes strict shape/config validation and correctly fails
the accepted-status gate. Independent candidate code review found no blocker;
this is not the required post-commit scientific acceptance review. All runtime
testing was CPU/fixture-only; no new GPU execution or submission occurred.

After the two required agent-created commits and independent acceptance, the
existing generic builder command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/del6500/projects/OPD/src \
/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python \
-m posttrain_circuits.scheduler_adapter.g0_request \
--execution-science-protocol prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_diagnostics_v2.yaml
```

Run it from `/home/del6500/projects/OPD`. It must remain blocked while the
checkout is dirty or the new protocol is proposed. The user has authorized the
retry and clarified that task submission does not need another approval.
The central service restarted at 2026-09-09 13:05:24 CDT; read-only
intake-status subsequently observed a live scan at 18:10:47 UTC, with the
global gate enabled, OPD armed, no blocked projects, and 14 old files excluded.
No OPD receipt was present. The generic builder was actually invoked and
rejected the still-dirty checkout before creating a plan or outbox. HEAD
remained bc6ee8e...; the user's documentation was staged but not committed.
Verify actual acceptance through central records after publication. This updated
transport workflow does not remove the clean-checkout and accepted-science
checks enforced by the OPD code. The user subsequently gave standing
authorization for agents to stage and commit authorized OPD work; AGENTS.md
now assigns implementation, acceptance, and handoff commits to the agent.

## Sources

- Durable job:
  `/data/del6500/ServerScheduler/state/jobs/opd-2947686c51c3e93ad1b18e5a3b7d6b22.json`
- Retry record: the corresponding filename under
  `/data/del6500/ServerScheduler/state/retries/`.
- Central audit: `/data/del6500/ServerScheduler/audit/events.jsonl`.
- Logs:
  `/scr/del6500/ServerScheduler/logs/opd-2947686c51c3e93ad1b18e5a3b7d6b22.attempt-001.stdout.log`
  and the corresponding `.stderr.log`.
