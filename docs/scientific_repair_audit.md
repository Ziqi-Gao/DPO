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

## D–F. Severity findings

- P0: unsupported generation API and padding boundary semantics; fixed in `rollout/generation.py`.
- P0: smoke validation/local-fork/circuit probes in production; replaced by hash-verified artifacts.
- P0: checkpoint label without loaded weights; strict checkpoint load and parity now precede discovery.
- P0: partial readiness bypass; production trainer/submission now require full readiness GO.
- P1: zero-capability anti-shortcut pass, missing bootstrap provenance, overwritten multi-seed
  aggregation, direction-dependent escalation, and multi-rank writes; repaired.
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
