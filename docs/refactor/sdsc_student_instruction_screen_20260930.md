# Proposed single student instruction screen

Status: preparation only; no submission or scientific acceptance. This bounded
screen needs independent implementation and actual-deployment review before
submission. Its fixed inputs and initial-capability question are independent of
checkpoint diagnosis 54558773's trained-model question, so the two diagnostics
may run concurrently after those reviews, using at most two combined H100 GPUs
and remaining below the standing four-GPU concurrency limit. Freeze each plan
and preserve each separate job/receipt; neither result may modify the other's
running controls or predeclared design. Combine both complete, independently
audited results before deciding any formal successor or new training.

It is a training-side screen of one scientific prompt candidate, not a software
bug fix already shown to work. It does not authorize replacing original G0
criteria, using trained weights as the initial model, or automatic progression.

## Evidence and hypothesis fixed before inference

Initial-only failed diagnosis 54557365 retained all original responses. Its
training first 32 / 256-token arm has zero strict successes: 21 length stops,
nine step-syntax errors, one invalid citation and one incorrect conclusion.
Six of the syntax errors omit the step colon; two use a fact ID in the rule
position. Three responses spell a conclusion as TRUE(...). The old v7
instructions are not contradictory. The hypothesis is only that explicit rule
applicability, F/S identifiers and copying the selected consequent may improve
adherence to the existing task. The model may still lack the required ability.

The candidate was constructed using those fixed training records. The validation
population has already been exposed during prior debugging; it is not a new
holdout. This screen must neither generate validation responses nor select
between multiple candidates based on them. No few-shot example, actual answer,
reference proof, per-example solution, assistant prefill or constrained decoder
is added. Rules and facts are not filtered, simplified or reordered.

## Frozen comparison

- Parent: completed calibration 54548846, report SHA
  `91794e528ec83ff0c7f35bd3734d36022d60db2794c3e41c7686beb97fbff4f1`.
- Model: original initial checkpoint only, SHA
  `85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4`.
  Before loading/overwriting any parameters, compare every original pinned HF
  BF16 tensor with the checkpoint: keys, shapes, dtypes and exact values.
  Record embedding/lm_head tying observations. A mismatch stops the screen;
  existing export provenance is strong but did not include this direct check.
- Runtime: the existing discovered Python 3.12.13 / Torch 2.8.0+cu128 /
  Transformers 4.56.2 environment and pinned Qwen3-1.7B revision. Offline only.
- Population: original dataset family's first 32 training examples in original
  order; the same 32 for both arms. No validation/test/circuit generation.
- Arms: original v7 instruction, then the single constant below. Only that
  instruction changes. Preserve graph, query, labels, tokenizer and original
  non-thinking chat wrapper. Labels/IDs/metadata/reference proofs must never
  enter instruction construction or candidate selection.
- Generation: greedy, cache disabled, 256 new tokens, unchanged 1536 total-input
  envelope. Check every prefix plus all 256 before any generation and reject
  overflow rather than truncate. Original student training budget remains 128.
- Precision: native BF16 parameters without an explicit autocast context,
  matching the original standalone G0 scorer's context. This differs from the
  prior diagnostic's explicit BF16 autocast and is disclosed. Do not claim
  bitwise-equivalent inference across different arithmetic/distributed paths.
- Resources: one H100, eight CPU cores, 64 GiB, at most 30 minutes. No optimizer,
  training, checkpoint mutation or model download. Separate task/run/intent/job.
- Evidence: all 64 raw responses, exact prompt/response token IDs and text,
  original full parser/verifier traces, strict and separately named answer-tag
  metrics, EOS/length stops, and canonical-target NLL/token accuracy including
  EOS. Canonical targets are used only for diagnostic scoring, never prompting.
- Storage: node-local work, actual-node input/output mount and free-space checks,
  read-back-hashed persistent publication before exit, bounded small fetch under
  `.sdsc/fetched/<job-id>/`. No weights fetched to Quest; no HOME training I/O.

The original instruction SHA is
`8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6`.
The only candidate SHA is
`8c44dc8a1bc86167e3787cdcd84e20537db69585e3d7ee30b920447072f03c1e`:

```text
Prove QUERY (1) or its negation (0); omit unrelated rules.
Output only:
<proof>
Snn: Rnn(citations) -> literal
</proof>
<answer>0 or 1</answer>
Replace placeholders; number S01,S02,... . Apply a rule only when all its premises are established. Cite those premises by comma-separated Fnn or earlier Snn IDs, not literals. Copy its consequent: positive TRUE <atom>, negative NOT <atom>; no literal parentheses. Stop at the target.
```

The actual tokenizer measures 87 versus 112 instruction tokens. All training 32
prefixes increase by 25 tokens; candidate prefix plus 256 is at most 1325. These
measurements establish only this screen's population. Prior full 256 maximum
shape plus 25 projects to 1527, leaving only 9 tokens; that projection is not a
new full-population check. Do not infer anti-shortcut/OOD/circuit safety.

## Interpretation and progression boundary

`passed` may mean only that this fixed diagnostic executed and published
verifiable evidence. All student, teacher-under-candidate, G0, pilot and
factorial acceptance flags stay false, regardless of metrics. Preserve the old
response bytes, scoring, failed job and original protocols without relabeling.

A predeclared provisional viability screen requires the candidate to improve
both original strict answer and full-proof accuracy over the paired baseline,
with at least 4/32 in each. This is a training-side futility criterion, not a
replacement acceptance threshold, checkpoint selection or formal confirmation.
Failure stops promotion of this candidate; do not iterate prompt wording on
validation until it passes. Passing only permits considering a separately
reviewed formal prompt successor; it does not automatically submit that path.

Formal use would require its own scientific implementation and independent
acceptance commits, complete sequence-shape checks, newly qualified teacher
readiness under all eight unchanged gates, complete newly bound teacher stores,
top-k/rollout evidence, unchanged initial base and anti-shortcut gates, complete
paired circuit cohorts, token alignment, HF/TL parity and all original G0 circuit
and checkpoint/resume checks. Old teacher or execution-class PASS cannot be
borrowed for changed scientific inputs. Extra confirmation populations require
an exposure/isolation audit and never replace the original gates.

The first 32 training examples are small and already inspected. Improvement here
cannot establish generalization; failure may reflect genuine model/benchmark
incompatibility rather than a correctable implementation error. Preserve that
outcome instead of forcing a success claim.
