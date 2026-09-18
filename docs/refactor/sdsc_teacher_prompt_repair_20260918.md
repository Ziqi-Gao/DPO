# SDSC teacher failure and prompt-v4 candidate

## Actual failure

Expanse job `54345715` is `FAILED`, ExitCode `1:0`, elapsed `02:21:48`.
Fresh accounting was obtained on Quest `quser34` after manual SSH authentication.
The published attempt ledger has SHA-256
`28233d89872739b2d9d34d2dba4704ba5e857ca737005166c5776249fd8229ab`.
All 2048 candidates (256 prompts × 8) were generated. There were eight accepted
candidates covering only four prompts. The original complete-coverage gate
correctly prevented student training.

| Outcome | Candidates |
| --- | ---: |
| Antecedent mismatch | 1099 |
| Response syntax | 829 |
| Step syntax | 111 |
| Conclusion mismatch | 1 |
| Accepted | 8 |

The literal `TRUE X` occurs in 1875 responses; 1468 copy the exact fake schema
line `S01: R01(F01,F02) -> TRUE X`. The copied line often precedes an irrelevant
rule enumeration. All 829 response-syntax failures also hit the 256-token limit.
This strongly supports schema copying as a cause; it does not prove every
remaining reasoning error will disappear after a prompt change. There was no
accounting OOM or timeout, and the retained memory checks passed.

The hardcoded diagnostic destination `/scr/del6500/OPD/diagnostics/teacher_demos`
also caused a permission error. It did not cause the scientific rejection:
the SDSC wrapper preserved the original ledger, accepted view, manifest and
complete dataset on persistent project storage, with read-back hashes.

## Repair and scientific scope

`datasets/proofgraph/rendering.py` now describes an abstract rule-application
syntax, actual rule and citation IDs, actual consequents, and stopping at the
proved query polarity. It supplies no invented rule/atom line to copy.
Facts, rules, identifiers, order, labels, canonical targets, parser, verifier,
model revision, non-thinking chat protocol, candidate seeds, sampling parameters,
eight-candidate limit, 256-token completion limit and coverage gate are unchanged.

The shared instruction constant affects ordinary task prompts and paraphrased
anti-shortcut prompts as well as teacher generation. This is an explicit
scientific prompt change. Accepted prompt-v3 remains immutable. The new
`prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_prompt_v4.yaml` remains
**proposed**; it needs independent review of an implementation commit followed
by a distinct review-only acceptance commit before formal use. The historical
Blackwell certificate is not evidence of H100 execution.

`build_teacher_demos.main(..., diagnostics_with_output=True)` lets the SDSC
caller preserve exact failed-store files in the caller's output tree. Central
server defaults remain unchanged. Bounded current-attempt coverage/error/finish
counts are now printed even when that extra copy fails; the original exception
and stale-store protection remain intact.

## Verification and limits

The pinned offline Qwen tokenizer measured all original 256 prompts: baseline
402–1246 tokens, repaired 395–1239. The chosen instruction is 80 tokens instead
of 87. An earlier 113-token draft was rejected before GPU submission because
its maximum was 1272; the deployed candidate is the shorter measured version.

CPU replay of 18 retained raw samples exactly matches their original verifier
traces, prompt identities and candidate seeds. All 256 unchanged canonical
proofs pass the original verifier. Independent review confirms that all 52
named execution-safety files still match the existing descriptor. These checks
establish preservation and rejection behavior, not teacher quality improvement.

Local evidence is under `.sdsc/diagnostics/teacher-54345715/`: `analysis.json`,
`population.json`, `variant-counts.json`, `sample-replay.json`, and the reusable
`replay_samples.py`. `regression-tests.txt` records the consolidated run: 483 tests passed, with
15 dependency warnings. Ruff and shell syntax checks passed. The new proposed
protocol validates structurally and is correctly rejected for formal use.

## Real diagnostic job and recovery

One fresh exploratory job, **54351516**, was submitted at
`2026-09-18T16:57:16Z`, intent `08139fa0cfa24f09b3838d0cad192ded`.
It uses one H100, 24 CPUs, 192 GiB and at most 30 minutes, account `nwu181`,
partition `nairr-gpu-shared`, QoS `nairr-gpu-shared-normal`.

- Run: `20260918T165336Z-2424d4393021-90ec9634`.
- Snapshot SHA: `2424d4393021365aea5b67be5db87413395f0dec051a4aa86bcd81e50f8519f2`.
- Independent snapshot: 493 eligible files, 4,962,478 bytes; no source overwrite.
- Initial state: `PENDING (ReqNodeNotAvail, Reserved for maintenance)`.
- Scheduler test-only estimate: September 20. Shorter limits and a smaller
  4-CPU/64-GiB estimate did not advance that date. Estimates are not guarantees.

The probe uses the original train population with its verified file SHA. It
checks all 256 candidate prompt lengths, then generates baseline candidate zero
and repaired candidates zero through seven on the fixed first 32 prompts.
Compare `paired_candidate_zero` to the baseline for a like-for-like comparison;
32×8 candidate coverage is a separate diagnostic measure. Raw responses,
token IDs, logprobs and verifier traces remain unmodified and are published
before allocation exit. The job is independent of the old stopped flows.

A finite read-only watcher is running on Quest quser34, PID 2037247, under
`.sdsc/supervision/probe54351516-readonly-v1/`. It checks every five minutes for
at most fourteen days and fetches bounded results after terminal accounting.
Its 19 CPU tests passed. It stops on SSH loss, changed controls or unknown state;
it neither retries jobs nor advances training. Automatic fetch requires the
existing shared SSH connection to remain available. Source existence alone is
not proof that the watcher is running; inspect its state/launch evidence.

A probe's `passed=true` means diagnostic completion and persistence only.
`accepted_science`, `full_teacher_ready`, `g0_passed` and
`execution_class_certified` remain false. No calibration or G0 prerequisite
accepts a probe as a formal teacher store. Do not blindly resubmit job 54351516.

Next inspect its real accounting and verified result, then review prompt-v4 and
regenerate the full 256×8 teacher store under fresh identities. Old preflight
54345604 cannot simply be combined with a changed scientific inventory: current
calibration admission requires matching source. Use a new matching preflight
or a separately reviewed compatibility change. The old v3-pinned calibration
and full pipeline plans remain stopped and must not be re-armed. Full successor
migration must consistently bind the new accepted science HEAD and protocol;
this diagnostic does not establish that migration or student training started.
