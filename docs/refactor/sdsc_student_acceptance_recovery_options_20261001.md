# Student acceptance: conditional recovery decision

Status: a reviewable decision proposal, not an accepted scientific amendment or
authorization for a new student experiment. Native initial job `54560398` completed
with job/batch/extern0:0 in12m38s. Independent replay confirms **0/128 original
answer-correct results**, below the required13. Preserve the original experiment
and every failed observation; no new student track has been selected or submitted.

The actual audit is
`.sdsc/diagnostics/student-initial-native-v1/actual-independent-audit-54560398.json`,
SHA `55b59e7946e39f67e83f9cbfba7157832d5e9f5ed5e602a1e0a2828e70eeba34`.
It also finds0/128 full proofs and13/128 valid format. All128 prompts and responses
were replayed; all memory checks pass. Complete G0 was not executed: this failed
necessary initial criterion already prevents its acceptance.

## What the completed diagnosis establishes

The original study separates visited states, supervision information and the
update rule in a two-by-three design, with SFT/GRPO anchors. Its circuit analysis
needs both existing capability and a learnable challenge population. See
[README](../../README.md) and [frozen Qwen3 protocol](../../prereg/qwen3_v2.yaml).

LR diagnostic `54560292` completed with all Slurm exits zero. The independent audit
`.sdsc/diagnostics/student-lr-v2/actual-independent-audit-54560292.json` has SHA
`c9c5c0c1c3a2ff4e1fec85c41dc603ee639941c7f1270142b6628a4840ca7a45`.
With identical initial weights, data order, optimizer state and RNG, changing only
learning rate from `5e-4` to `5e-5` changed four-step train32 proof accuracy from
0/32 to 5/32. Fixed64 teacher sequence-mean CE changed from a first-step increase
1.0555→4.9576 to a decrease 1.0555→0.5187, reaching0.07247 after four steps.
The raw response metrics were independently replayed. GPU numerical observations
remain hash-bound producer evidence; neither arm establishes validation quality.
This supports a prospectively fixed lower-LR successor, not rewriting the old
experiment or claiming an AdamW implementation bug.

A lower-LR calibration successor must start from the original initial checkpoint
with fresh optimizer/RNG/cursors. Preserve all other scientific parameters,
2,000,000-token/120-step limits, evaluation definitions and acceptance thresholds.
It needs its own configuration, amendment, implementation/independent acceptance,
execution evidence and complete results. The four-step diagnostic checkpoint is
neither a formal calibration continuation nor a replacement initial checkpoint.

## The required decision after the native check

The native check uses the original scorer, initial checkpoint, unscreened first128
validation examples, original instruction and256-token generation cap, without
the extra autocast used by an earlier diagnosis. The original base criterion is
at least13 `VerificationResult.answer_correct` results; proof reward and answer-tag
accuracy are different metrics. See
[scorer](../../src/posttrain_circuits/cli/score_probe_candidates.py) and
[G0 finalizer](../../src/posttrain_circuits/cli/finalize_g0.py).

If independently replayed native results reach13, continue the reviewed LR
successor and the remaining original gates. If they do not, the fixed original
initial cannot satisfy G0. Training later weights or adding GPUs does not alter
that initial result. Keep the failed original track; do not replace its initial,
filter the128 examples, change thresholds, or promote the failed instruction
candidate. A new student track then requires a scientific choice:

| Direction | Concrete candidate and scientific interpretation | Required work before execution |
| --- | --- | --- |
| Larger native student | First reviewable candidate: unadapted `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, whose original cache already belongs to the project. This preserves the native-capability interpretation more closely, while changing model scale and the experimental subject. Never use the adapted teacher checkpoint as this initial. This candidate is not selected or shown capable merely by its availability. | Review existing development evidence; freeze one model/revision, prompt, selection/stop rules and explicit successor scope before new evaluation. Verify raw weights/tokenizer, model-shape/runtime/memory, HF/TransformerLens parity and distributed checkpoint/resume. Re-establish every original base, anti-shortcut, teacher, paired-cohort and circuit gate for the new subject. A failed candidate remains a failed candidate; no validation-driven model search. |
| Prepared 1.7B baseline | A separately trained and qualified Qwen3-1.7B baseline shared by every later method. The comparison then concerns capability after task preparation, rather than native instruction-tuned capability. Preparation may itself influence later method effects and must be reported. | Before training, freeze genuinely disjoint preparation train/dev data, budget, optimizer and development-only checkpoint-selection rule, with overlap checks against evaluation/circuit populations. Produce new weights and independent provenance; never rename existing step4/20/33 weights as initial. Requalify all original gates and both paired cohorts; preparation that eliminates the challenge cohort fails rather than changing its definition. |

The larger-native direction is the first discussion choice because it preserves
the original interpretation more closely. It is not a claim that8B will pass.
The previous raw8B v7 training probe (job54489646) accepted42/256 sampled
candidates and covered7/32 training prompts; its complete-coverage teacher gate
failed. These are different populations and sampling criteria, not the proposed
student's greedy validation128 score. They are relevant uncertainty, not proof
that the separate student base criterion would pass or fail. Prepared baseline remains a distinct, explicit
research alternative rather than an invisible repair.

## Authorization and stopping conditions

Existing authorization covers current diagnostics, actual implementation fixes,
resource selection and preparation of reviewable successor work. It does not
make different scientific subjects interchangeable. A decision to replace the
student or introduce task preparation must be explicit before that new experiment.
This is a scientific-direction choice, not renewed approval for the already
authorized SDSC connection, resource envelope or diagnostic submission.

The [teacher adaptation precedent](sdsc_teacher_adaptation_20260928.md) explicitly
preserves the student's1.7B initialization and excludes downstream student
migration. It supplies a pattern for isolated data, prospective selection and
independent acceptance, not permission to borrow its acceptance for a new student.
The original model-scale escalation clause also requires established stable causal
evidence; the current calibration does not supply that prerequisite.

Neither option permits threshold relaxation, evaluator changes, selective
reporting, unlimited candidate search or automatic factorial/Gemma execution.
Scientific failure stops that track. Preserve the original initial result and
report which new question any accepted successor actually answers.
