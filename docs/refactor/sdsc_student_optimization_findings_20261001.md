# Student optimization diagnosis: paired learning-rate experiment

Status: actual diagnostic evidence, not student or G0 acceptance. Scientific
successors below are proposals; no accepted protocol or initial checkpoint is
changed by this document.

## Observed experiment

SDSC job **54560292** completed in18m18s with job, batch and extern all
`COMPLETED / 0:0`. It used2 H100,24 CPU and384 GiB; its Slurm wall-time limit was1 hour.
The independent audit replayed all128 generated responses,32 original prompts
and16 rank-local batch records. Exact plan and source identities are in
`docs/refactor/current_handoff.md` and the immutable submission records.

Both arms start from the same original initial checkpoint, RNG, fresh AdamW
state, accepted teacher store and original instruction. Each processes the same
four successive global64-example windows:256 samples and237,754 nonpadding
input tokens. The only treatment is the constant learning rate:5e-4 versus5e-5.
All observations in this diagnostic use training data; no validation/test
response was used to select the lower rate.

The fixed first64 teacher-target sequence-mean cross entropy is:

| Optimizer updates | LR5e-4 | LR5e-5 |
| --- | ---: | ---: |
| 0 | 1.055534 | 1.055534 |
| 1 | 4.957614 | 0.518697 |
| 2 | 4.280615 | 0.304089 |
| 3 | 11.990564 | 0.109616 |
| 4 | 6.939187 | 0.072471 |

After update4, standalone generation on the same32 training prompts gives:

| Original metric | LR5e-4 | LR5e-5 |
| --- | ---: | ---: |
| Answer-correct /32 | 0 | 5 |
| Complete correct proof /32 | 0 | 5 |
| Valid format /32 | 0 | 31 |
| EOS termination /32 | 0 | 32 |

All32 high-rate final responses are malformed and exhaust the256-token cap.
The low-rate arm still has27 unsuccessful proofs:17 antecedent mismatches,
8 invalid citations,1 wrong conclusion and1 step-syntax error. It has improved
on training examples, not demonstrated generalization or passed acceptance.
Both initial arms have0/32 answer/proof successes.

## What the diagnosis establishes

The actual first-step AdamW observations havezero coordinate residual-bound
violations. Both ranks execute one optimizer call after exactly8 microsteps;
there is no evidence of an accidental optimizer call at every microstep. The
first update has a negative gradient/update inner product. Its global L2 norm
is18.75284 at5e-4 and1.87529 at5e-5. Four root-versus-export logit comparisons
per arm are bitwise equal. Together with the paired loss/generation outcomes,
this supports excessive step size in this configuration; it does not identify
an optimizer formula, label-shift or FSDP scaling implementation defect.

The CPU auditor independently reconstructs token encodings, decodings, verifier
traces, batch/token arithmetic and report consistency. GPU loss, gradient,
AdamW-coordinate and logit measurements remain hash-bound producer evidence:
the auditor did not download large weights or recompute GPU logits. The single
seed/four-step comparison does not establish an optimal learning rate or the
outcome of a complete training run.

Peak host memory was65,098,821,632 bytes; the observed own-job failcnt waszero.
Both final8,127,108,761-byte model-only checkpoints were persisted and read-back
hashed on project Lustre. Neither is a full optimizer-resume checkpoint or a
replacement for the original initial weights.

Evidence:

- Final small results: `.sdsc/fetched/54560292/fetch-5e1mw600/`.
- Publication SHA-256:
  `8625d8264f3d9b7fee3081cd63cff013d6cb10ab515d21756699cfe583afeaf7`.
- Independently reviewed auditor result:
  `.sdsc/diagnostics/student-lr-v2/actual-independent-audit-54560292.json`, SHA-256
  `c9c5c0c1c3a2ff4e1fec85c41dc603ee639941c7f1270142b6628a4840ca7a45`.

## Next scientific decision

The separately reviewed original-native-initial check54560398 has now completed
and passed independent raw-response replay:0/128 `answer_correct` results, below
the required13 on the unscreened first128 validation examples. Complete proof
accuracy is also0/128; valid format is13/128 and is not the acceptance metric.
The original initial therefore fails a necessary G0 criterion. Lowering a later
training learning rate cannot change this fixed initial score.

A minimal optimization successor would fix constant LR5e-5 before another
training run, preserve every other scientific parameter and threshold, and
restart from the original initial with fresh optimizer/RNG/cursors. It must not
resume the four-step diagnostic or treat its checkpoint as initial. Preserve the
2,000,000-token and120-step ceilings;33 historical updates are an observed budget
outcome, not a newly selected ceiling. Create a new proposed student amendment
and student-owned config, independent implementation/review acceptance commits,
and matching preflight/checkpoint/resume evidence. Frozen teacher-source files
and accepted v6 bytes remain historical authority.

An LR-only successor therefore cannot pass original G0. A different model or
independently prepared common baseline would change scientific initial identity
and needs an explicit direction choice, successor design and independent review
before execution or reuse. The reviewable options are in
[the recovery decision](sdsc_student_acceptance_recovery_options_20261001.md).
Existing trained checkpoints must
not be relabeled, and the original failure must remain visible. The complete
anti-shortcut, paired circuit, calibration-improvement and other G0 gates still
apply; neither diagnostic completion nor a passing single base metric replaces
them.
