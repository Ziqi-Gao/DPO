# Scientific repair audit

## A. Scientific design restatement

The controlled comparison independently varies the source of visited states (one frozen common
behavior bank versus current policy) and the signal applied at those exact states (hard teacher,
forward-KL soft teacher, or exact-verifier replay). Canonical SFT and GRPO remain separate anchors.
Shared-state local forks isolate immediate update-rule effects at matched
`KL(output_new || output_fork)`. EAP-IG only discovers candidates; held-out exact activation/path
patching supplies causal evidence on separately frozen base-capable and challenge probes.

## B. Pre-repair verdict

`NOT READY — SCIENTIFIC BLOCKERS`. The production generation call failed on Transformers 4.56.2;
formal training evaluated smoke examples; production local forks bundled smoke states; MIB ignored
checkpoint weights; circuit validation regenerated probes; and the trainer accepted two partial
gates instead of the full readiness decision.

## C. Scientific fidelity matrix

| Component | Before | Repair/evidence | Status |
| --- | --- | --- | --- |
| Fixed state source | Unknown prompts fell back to unrelated records | Unknown IDs now fail; store hash is in every run | repaired |
| Current-policy source | Generator version was overwritten | Exact requested/returned version equality is enforced | repaired |
| Hard/soft/replay objectives | Unified trainer already present | Architecture preserved; no objective merge | retained |
| Canonical SFT/GRPO | Separate configs/entry points | Kept distinct in G0/pilot scripts | retained |
| Local fork | Production built smoke examples | Requires checkpoint, prompt/store/target/reward/logprob/probe bytes; mismatch exits nonzero | repaired |
| Anti-shortcut | Gap-only gate allowed zero/zero | Absolute and per-transform floors plus provenance bindings | repaired |
| Probe cohorts | IDs/hashes only | Exact example bytes and nested hashes are persisted/verified | repaired |
| Circuit discovery | Checkpoint was a label | Checkpoint file is strict-loaded before HF→TL parity and byte-hashed | repaired |
| Exact patching | Regenerated validation pairs | Uses same manifest's held-out validation bytes/checkpoint hash | repaired |
| Noise floor | Metrics library only | Bootstrap indices/vectors/raw hashes retained; formal dynamics CLI added | repaired |
| Distributed resume | Shared files written by all ranks | main-only metadata, barriers, Accelerate save/load and full export | target-stack validation required in G0 |
| Preregistration | v0 had a direction-dependent gate | v0 retained; active v1 removes direction as a gate | repaired |
| Qwen3 prompt protocol | Model paths could format raw prompts independently | One hash-bound non-thinking formatter is consumed by rollout, teacher, training, validation and circuits | CPU-validated; real-tokenizer gate pending GPU |
| Qwen3 circuit adapter | Qwen2-only architecture assumptions | Native TransformerLens Qwen3 conversion, config `head_dim`, GQA/QK-norm semantics and exact pre-norm hooks | tiny HF↔TL parity passed; real parity required in G0 |
| Teacher readiness | Coverage/recovery were metrics but not gates; causal shift was structural only | Top-k coverage, corrupted-prefix recovery and clean/corrupt target-logprob shift are independent fail-closed gates | repaired and adversarially tested |
| Four-rank schedules | Each rank could replay the complete prompt order | Deterministic disjoint rank shards and rank-bound resume state | repaired and adversarially tested |
| Pilot finalization | One positive circuit effect could suffice | All hash-bound artifacts, noise-floor fields, transfer and held-out exact-patching protocol are required | repaired and tamper-tested |
| Formal Git state | Only prereg dirtiness blocked formal runs | Any source-tree dirtiness blocks formal execution | repaired and adversarially tested |
| Three-seed inference | Cluster covariance was allowed with three clusters | Three-seed output is descriptive seed-level aggregation; cluster inference starts at five seeds | repaired |

## D–F. Severity findings

- P0: unsupported generation API and padding boundary semantics; fixed in `rollout/generation.py`.
- P0: smoke validation/local-fork/circuit probes in production; replaced by hash-verified artifacts.
- P0: checkpoint label without loaded weights; strict checkpoint load and parity now precede discovery.
- P0: partial readiness bypass; production trainer/submission now require full readiness GO.
- P1: zero-capability anti-shortcut pass, missing bootstrap provenance, overwritten multi-seed
  aggregation, direction-dependent escalation, multi-rank duplication, incomplete teacher gates,
  partial pilot finalization, dirty-source acceptance, and invalid three-cluster inference; repaired.
- Remaining environment-dependent validation is explicitly performed by G0, never inferred from CPU
  tests.

## G–K. Traces, tests, reproducibility, and next safe run

The audit traced the offline/current generation path through teacher scoring and supervisor loss,
the exact shared fork bundle restoration, and checkpoint→HF→TL→EAP-IG→held-out patching. Adversarial
tests cover padding bytes, stale versions, unknown prompts, zero/zero anti-shortcut, checkpoint
weight changes, probe byte tampering, local-fork mismatch, production config resolution, and
multi-seed preservation. Formal artifacts record model/tokenizer/checkpoint/dataset/bank/probe,
Git and prereg hashes. The only safe GPU action is G0 via `scripts/production/run_g0.sh`; pilot is
cryptographically gated on G0 PASS, and the three-seed factorial remains forbidden here.

## L. Confidence and unresolved environment checks

CPU behavior and static production routing are directly tested. HF↔TransformerLens parity,
four-GPU next-step resume identity, Qwen capability thresholds, and causal separation are empirical
G0 results; they must not be claimed until the corresponding production artifacts pass.

For the `qwen3_v1` track, CPU acceptance additionally fixes the exact student/teacher revisions,
chat-template SHA-256, tokenizer fingerprint, non-thinking prompt bytes, explicit sampling fields,
native Qwen3 head dimension/GQA/QK-normalization behavior, rank sharding, cross-model artifact
rejection, and complete prereg/launch provenance. Full snapshots, simultaneous-load memory,
four-rank FSDP resume, and real-model HF↔TransformerLens parity remain empirical gates; no waiver is
permitted.

## M. Qwen3-v2 independent-review repair

`qwen3_v1` remains frozen historical evidence. The repair track is the new
`prereg/qwen3_v2.yaml` / `outputs/qwen3-v2` namespace; no v1 or Qwen2.5 scientific artifact is an
input. The repair closes the following review findings:

- Every four-GPU Qwen3 job requests 192 GiB explicitly. The preflight reads the finite Slurm
  cgroup-v1 or cgroup-v2 memory limit and peak, requires at least 32 GiB and 20% headroom, records process MaxRSS, and
  fails when the registered request is not enforced. Students use low-CPU-memory loading before
  FSDP; the 8B teacher is loaded and scores on rank zero only.
- Formal provenance is resolved from the composed config. Teacher readiness and every G0 decision
  input bind the exact protocol, namespace, model/tokenizer revisions, prompt fingerprint,
  dataset/prefix probes, code, and preregistration. Cross-prereg and modified artifacts exit
  nonzero.
- The token budget is global non-padding model-input tokens. Factorial/SFT reserve the exact
  cross-rank optimizer window; GRPO uses a preregistered conservative admission bound and reduces
  actual per-step rank deltas. Budget state is checkpointed, cannot reset on resume, and stops only
  at optimizer boundaries. `max_steps` remains an independent safety ceiling.
- A completed cell manifest is not trusted until it binds its metrics, resolved config, final
  checkpoint, budget usage, and successful Slurm array task. The pilot report reconstructs that
  chain and hashes every accuracy, output-KL, circuit, dynamics, resume, and terminal-state input.
- Circuit feasibility covers all eight cells, both cohorts, and both process/final stages: four
  initial and 32 final matrix rows. The seed-42 report can establish execution and artifact-chain
  feasibility only. It cannot establish a confirmatory primary endpoint or population inference,
  and effect direction is never a pass condition.
- Supervisor polling uses bounded retry/backoff for scheduler errors and accounting lag. Empty or
  `UNKNOWN` state is never success, and every four-GPU submission refuses a potentially concurrent
  OPD GPU job.

No Qwen3-v2 GPU result is claimed by this repair. Real cgroup headroom, measured MaxRSS, Qwen
forward/backward, FSDP resume, and HF↔TransformerLens parity remain independent-review GPU gates.
