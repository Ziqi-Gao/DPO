# Qwen3-v2 v7 teacher quality diagnosis

Job **54489646** completed normally (`COMPLETED / 0:0`, 29m03s) after the
asynchronous stack-observer repair. Its **256 v7 candidates cover only 7/32
training prompts**, with 42 accepted responses. This is a measured failure of
the necessary training-probe gate. The formal readiness evaluator has not run
for v7; neither diagnostic execution success nor this analysis accepts v7.
Both observation and automatic continuation are terminal, and no student
training or successor experiment was submitted.

## Evidence and reproducible feedback loop

The complete 2,462,548-byte ledger was fetched once through the existing Quest
SSH master and verified against the published receipt:

- Ledger: `.sdsc/fetched/54489646/raw-verified/attempts.jsonl`, SHA
  `ca97a4cef49e2884dea31a2332b6e6aaa310b3174f898966d65772188ca09fc6`.
- Published report: `.sdsc/fetched/54489646/fetch-0ol5j72o/teacher-prompt-probe.json`,
  SHA `db2e2c0ff4da20694196a1308f65b75dee8c3f47b777413e922e6bca1c29b3ea`.
- Exact example population:
  `.sdsc/diagnostics/teacher-54345715/population.json`; source train-file SHA
  `377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b`.

The CPU feedback loop deserializes the original examples, calls the unchanged
production parser/verifier on every saved response, and compares the complete
verification dataclass with the recorded trace. **All 288 traces match exactly**
(32 v5 baseline plus 256 v7 candidates). Responses, seeds and verifier results
are not modified. No GPU, model inference or new validation cohort is used.

The ranked hypotheses were: (1) incorrect serialization of proof dependencies,
(2) incorrect rule/premise reasoning, (3) prompt/template/tokenization corruption,
and (4) erroneous rejection of valid proofs. The tests below distinguish these
instead of treating every failure as another runtime bug.

## What failed

The production verifier's first-failure categories are:

| v7 result | Count |
| --- | ---: |
| Accepted | 42 |
| Unknown or not-yet-established citation | 96 |
| Premises do not match the selected rule | 61 |
| Invalid proof-step syntax | 36 |
| Invalid overall response syntax | 21 |

There were 235 EOS completions and 21 responses reaching 256 output tokens.
Those are a separate classification, not additional errors. The 21 truncated
responses cannot explain the other **193 invalid EOS completions**.

For `pgpair-095b26e1aebe0f9eba88-pos`, candidate 1, the model writes:

```text
S01: R09(S02, S03) -> SYM_088
S02: R11(F01) -> SYM_080_R
S03: R18(F01) -> SYM_080_L
```

The first line cites steps that have not been established. The correct
dependencies are present in this particular response, but their order violates
the serial proof contract. The original verifier correctly rejects it.

The negative sibling's candidate 0 has a different failure. Its first two
valid steps establish `SYM_021_L` and `SYM_021_R`, but the next step selects
`R09`, whose antecedents are `SYM_080_L` and `SYM_080_R`. The available negative
rule is `R10`, concluding `NOT SYM_088`. The response instead mixes the positive
and negative branches and answers 1. This is a concrete reasoning error;
whitespace, step renumbering or adding TRUE does not repair it.

## Order-only counterfactual, for attribution only

An isolated CPU analysis topologically sorts each fully parsed response's
original citation graph, keeps **every original rule, literal, citation edge
and answer**, consistently renumbers steps, and runs the original verifier.
It rejects missing references, duplicate IDs and cycles. It neither writes
altered teacher responses nor changes official acceptance.

This adds only **14** verifier-valid responses in the offline analysis, for
56/256 and 9/32 covered prompts. The official result remains **42/256 and 7/32**.
The counterfactual breakdown is 56 valid, 57 parse-invalid, 78 premise mismatch,
42 cyclic dependencies, 10 duplicate step IDs and 13 absent references.
Counts use the analysis's precedence and must not be added to original error
categories. Stable ordering is a bounded diagnostic, not proof that every
possible reordering was exhaustively tested.

The independent verifier audit further resolves the original 96 citation errors:
83 self/forward step references, eight uses of a rule ID as evidence, and five
references to undefined earlier steps such as S00. All 61 first premise-mismatch
errors cite the right number of objects but the wrong literal multiset; this is
not an order-sensitive comparison bug.

A separate attribution check ignores citation names while retaining original
step order, rule identity, rule antecedent availability, conclusion and answer.
Only 74/256 responses across 11/32 prompts have a valid rule path under that
relaxation. Its counts overlap the order-only analysis and must not be summed.
Neither counterfactual is an admissible teacher artifact.

Thus some failures are ordering errors, but an order-only correction does not
come close to satisfying complete coverage. This is not a proposal to reorder
responses in the teacher store or relax the verifier.

## Input and generation path checks

Every saved v7 raw prompt matches the current renderer, including the exact
instruction SHA
`8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6`.
All 288 input-token sequences encode/decode exactly to their saved model-facing
text; all response-token sequences decode to the original responses. The
single-user chat template and empty-think non-thinking assistant prefix are
present. There is no observed prompt overwrite, truncation or decode corruption.
The ledger independently encodes inputs immediately before calling the production
generator; it does not hook the generator's internal tensor. Deployed source
confirms the same renderer/template/tokenizer path. The job did not serialize
the complete effective Transformers generation configuration, so this audit
does not claim every inherited runtime option was independently observed.

The 32 selected prefixes contain 456–1044 tokens. Their unchanged canonical
proofs need only 54–163 output tokens including EOS; all fit the 256-token
generation budget. The largest prefix plus canonical target, including EOS,
is 1123 tokens, below the production model-input bound. Increasing output length
is not a demonstrated remedy for these samples.

All 256 candidate seeds, including their low 32 bits, are unique. The 32 v5
baseline input IDs, seeds, response IDs and token log-probabilities exactly match
the corresponding old v5 candidate-zero evidence. Paired v5/v7 seeds agree,
all 32 inputs differ, and 26/32 response-token sequences differ. This supports
an actual prompt intervention with reproducible controls, rather than accidental
reuse of baseline outputs.

The model still produces **zero TRUE-marked positive conclusions out of 655
parsed positive conclusions**, including when special tokens are retained during
decoding. However, the existing parser accepts both bare positive atoms and
TRUE-prefixed atoms. All 42 accepted v7 responses remain accepted with their
original bare spelling. Missing TRUE therefore does not cause the coverage
failure; it remains a separate mismatch with canonical prefix-scoring targets.

Independent checks also reproduce all 256 original graphs from their original
seeds and original range-based task configuration, verify their signed
closure/labels and canonical proofs,
and round-trip their rendered graph text. Bare positive and TRUE-prefixed
canonical proofs verify equivalently. No examined response is rejected solely
for the bare positive spelling.

## Diversity and answer correctness

The 256 v7 candidates contain only **83 distinct token sequences when counted
within each prompt**. For eight prompts, all eight candidates are identical;
six of those prompts fail in every attempt. Candidate zero already covers six
prompts; the remaining 224 generations add only one newly covered prompt.
The equivalent historical v5 counts were 74 sequences and ten prompts. Seeds
are distinct, so duplication alone is not evidence of broken seed handling.
The observed sampled-token log-probabilities describe the filtered generation
distribution, not the unfiltered model's entropy or a calibration measurement.

An independent answer-tag audit does not use `verification_trace.answer_correct`,
because the verifier stops at the first invalid proof step. Among 235 complete
v7 outputs, **162 answer tags are correct and 73 are wrong**; all 21 truncated
outputs lack a complete tag. This is 162/235 (68.94%) on complete outputs or
162/256 (63.28%) on all attempts. Of the 162 correct answers, only 42 have valid
proofs; the remaining 120 contain citation, premise or proof-syntax errors.
Even an answer-only interpretation would therefore retain substantial errors.

Coverage by polarity is 5/16 positive and 2/16 negative prompts. The sample is
32 fixed training prompts in 16 pairs; correlated candidates and these strata
are not independent experimental replications or evidence of a causal polarity
effect. The paired candidate-zero comparison is v7 6/32 versus v5 3/32;
comparing v7's eight candidates with v5's one-candidate baseline would be unfair.

## Consequences for the experiment

The evidence does not identify a new submission, parser or prompt-transport
bug to fix. It establishes that the frozen Qwen3-8B non-thinking generation
configuration does not reliably produce this task's required serial proofs.
This is a statement about this exact configuration and cohort, not a claim
that Qwen3-8B can never solve the task.

Do not launch the full store, supplemental cohort, calibration or G0 from this
result. More GPUs speed execution but do not fix the demonstrated rule/reference
errors. Repeating the same prompt/seeds or increasing candidates is not an
authorized recovery and does not establish the original coverage contract.

The next scientific decision should address teacher proof competence explicitly:
a separately reviewed change to the teacher's reasoning/generation protocol or
an independently specified teacher-adaptation experiment. A single change should
be frozen and tested on training evidence before exposing supplemental validation.
Thinking mode, demonstrations, constrained decoding, model replacement or teacher
training are scientific interventions, not transparent software fixes. None is
selected, implemented or submitted by this diagnostic session. The original
readiness checks and independent acceptance remain mandatory.

The raw ledger, CPU audit scripts and detailed JSON outputs are retained under
`.sdsc/fetched/54489646/` and `.sdsc/diagnostics/v7-quality/`. All analysis is
offline; no weights, generated text, scientific configuration, parser or threshold
was altered, and no failed flow was restarted.

Reproduce the retained CPU analyses from the actual Quest repository, with GPUs
hidden and one CPU thread per math library:

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src .venv/bin/python -B .sdsc/diagnostics/v7-quality/verifier-audit.py
PYTHONPATH=src .venv/bin/python -B .sdsc/diagnostics/v7-quality/topology_audit.py
PYTHONPATH=src .venv/bin/python -B .sdsc/diagnostics/v7-quality/diversity.py
PYTHONPATH=src .venv/bin/python -B .sdsc/diagnostics/v7-quality/inference_audit.py
```

These are diagnosis artifacts, not deployment entrypoints. They consume the
retained local evidence and write only diagnostic reports. There is no production
code fix or new regression-test claim: the reproduced failure is the unchanged
model's scientific output, and the feedback loop confirms the original strict
verifier's judgments. Independent agents audited tokenization/generation,
graph/parser/verifier correctness, diversity and the order-only counterfactual.
