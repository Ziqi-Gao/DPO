# Signed-literal output-contract repair

The user continued the recommended route on September 27: keep the original
Qwen3-8B teacher and repair the output contract before considering teacher
adaptation. No teacher training, model replacement, threshold change or response
postprocessing is part of this candidate.

## Evidence and scope

Completed diagnostic 54368737 measured v5 with the unchanged frozen protocol.
Its 128 intermediate-conclusion targets all begin with ` TRUE`, but the prompt
asks for the rule consequent, whose positive form is a bare symbol. None of the
128 free generations emitted the TRUE convention. Independent replay of all
384 probe sides found no token-index error. The observed mismatch motivates
an explicit signed-literal instruction.

The original parser already normalizes a bare positive symbol to the same
literal as `TRUE` followed by that symbol. All 24 verifier-valid generations
therefore remain valid, including the 23 exact canonical proofs. This repair
does not explain or erase the separate illegal citations, wrong antecedents,
wrong conclusions and truncated responses. No improvement is claimed before
a new observation.

The v7 renderer explicitly describes positive `TRUE <atom>` and negative
`NOT <atom>` consequents, using abstract placeholders and actual rule/citation
IDs. It preserves the graph, reference targets, query, stopping polarity and
sequential step numbering. The same instruction is used by anti-shortcut
prompts. The frozen model revisions, non-thinking chat protocol, seeds,
candidate count, generation limits, parser/verifier and success thresholds
remain unchanged. Historical proposed v4/v5/v6 protocols and failed evidence
are retained. The v7 execution-science protocol is also **proposed**.

## Plan fixed before new GPU outputs

Run one `qwen3-v2-teacher-prompt-probe` on the original ordered first 32 training
prompts. Compare exact v5 candidate zero with v7 candidates zero through seven,
using the production generator and original per-candidate seeds. Report the
paired candidate-zero result separately from eight-candidate coverage. Save
unaltered prompts, input/response IDs, log probabilities, verification traces,
and observational TRUE-spelling counts. These new counts never determine
acceptance and never rewrite responses.

Resources: one H100, 24 CPU cores, 192 GiB, at most 30 minutes; account nwu181,
partition nairr-gpu-shared, QoS nairr-gpu-shared-normal. Use the existing fixed
SDSC Python/HF cache and verified per-job node-local staging and persistent
publication. Submission uses a new source snapshot, intent and Slurm job ID.
The observer may query and fetch only; it starts no successor.

If fewer than 32/32 prompts have a valid candidate, stop larger progression:
those same prompts and seeds would fail the full store's coverage contract.
No automatic v8, extra candidates, changed seed or repeated submission is
allowed by this diagnostic plan. Even 32/32 is only a necessary pilot result;
it is not evidence of complete 256-prompt coverage or teacher readiness.

## Validation exposure and independent confirmation

The first 128 validation examples informed the diagnosis above. Any repeated
evaluation on those examples is explicitly exposed evidence; it must not be
described as independent confirmation. The original first-128 readiness gate
and its thresholds remain mandatory and unchanged.

Before any new GPU output, a supplemental confirmation cohort is fixed as
zero-based ordered validation rows `[128:256]` from the preserved validation
file, SHA-256
`8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3`.
The canonical ordered example hash is
`532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073`.
It contains 128 unique examples in 64 complete sibling pairs. Local identity
checks establish disjoint example IDs, pair groups and graph/query identities
from the exposed first 128 validation examples and the 32-prompt train pilot.
Selection evidence is `.sdsc/diagnostics/prompt-v7/fresh-confirmation-selection.json`.
No teacher outputs for this cohort have been inspected.

Only after the fixed training pilot covers all 32 prompts may the frozen
candidate receive one supplemental diagnostic using this cohort, with the
original greedy 128-token generation, prefix construction/scoring and eight
metric gates. This is additional evidence, not a replacement cohort in the
formal evaluator or a formal readiness artifact. A scientific failure stops
progression; further adaptation cannot reuse it as an independent holdout.

Formal use still requires an independently reviewed implementation and a
distinct review-only acceptance commit, a fresh complete teacher store, the
original readiness gate and a matching new preflight. Existing stopped v3
flows are not re-armed. No Blackwell execution-class certificate is claimed
for H100 execution.

## Current verification

The instruction SHA-256 is
`8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6`.
The pinned tokenizer measures 87 tokens, exactly the v5 length. All 256 original
train-prefix token counts remain 402–1246; adding 256 completion tokens stays
at or below 1502. All graph sections and canonical targets are byte-identical;
all 256 targets verify. The target-list SHA is
`a0aea1eaacea4044a809591fda4222cbed788475f9ebbbacf584ee7c9984362c`.
Both normal and anti-shortcut renderers retain hidden-label/reference
independence. The new signed-convention regression fails against v5 and passes
against the proposed repair.

The scientific/configuration suite passed 64 tests; the worker/observer suite
passed 42. Independent review found two observational-helper edge cases
(embedded closing tags and arrows inside invalid citations), which are fixed
without changing the parser. Regressions keep those responses rejected and
verify preservation of a complete 288-record ledger. Ruff and diff checks pass.
Independent reviewer `/root/v7_science_review` found no remaining blocker for
exactly the bounded diagnostic above, after 23/23 post-fix worker tests and
recomputation of all 52 unchanged safety-file hashes and the unchanged science
configuration SHA. This operational review is not protocol acceptance.

On Quest quser42, the runtime-computed shared master was checked successfully;
SDSC identity, account/QoS and an empty user queue were verified. The routine
15-second Python metadata probe timed out. A separate bounded lightweight
check then verified Python 3.12.13, its expected executable hash, and installed
Torch 2.8.0+cu128, Transformers 4.56.2, Accelerate 1.10.1 and tokenizers 0.22.0
metadata in 1.69 seconds, without loading a model or querying a GPU. Login-node
storage visibility does not replace GPU-node mount verification.

The six ServerScheduler contract paths are absent on this Quest host. This
work uses the explicit Quest/SDSC exception in AGENTS.md; no central scheduler
code, registration, service or submission is touched.
