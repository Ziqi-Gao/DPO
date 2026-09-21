# Teacher grammar and dependency-order diagnosis

## Reproduced failure

Probe `54351516` completed on September 18 with Slurm `COMPLETED / 0:0`,
but its prompt-v4 candidate accepted zero of 256 responses and covered zero
of 32 prompts. Diagnostic completion was not scientific success.

The 2,433,937-byte raw ledger was fetched from its recorded persistent result
directory and matched receipt SHA-256
`3f4f342764a95fa9eeb71d72075ba16f809b08e2900b7cf032ca58398415ccad`.
CPU replay reproduced all 288 original traces and prompt identities. Evidence
and the replay script are under `.sdsc/diagnostics/probe-54351516/`; the
unchanged raw ledger is under `.sdsc/fetched/54351516/raw-verified/`.

Three hypotheses were tested against those preserved responses:

1. The parser rejects a space between a rule ID and its opening parenthesis.
   Removing only that space/tab in a diagnostic copy recovers five responses,
   covering one prompt. It does not resolve the teacher-store failure.
2. Most responses use the wrong proof grammar. After that diagnostic whitespace
   substitution, 161 first-invalid lines put literals before the rule ID,
   23 put rule text/citations in the wrong positions, and three mix formulas
   or prose into the citation field. Another 16 fail unknown-citation checks
   and 20 fail antecedent checks. These must not become accepted by relaxing
   the verifier.
3. Length truncation explains the other 28 responses: all have finish reason
   `length`. The remaining 228 terminate at EOS, so increasing the completion
   budget is not established as the repair.

The production parser, verifier version, verifier logic and original ledger
remain unchanged. The diagnostic substitution is not a response postprocessor
and is never used to construct a teacher store.

## Candidate and preserved scientific contract

The renderer now gives the explicit abstract grammar
`Snn: Rnn(citations) -> consequent`, requires the opening parenthesis to attach
to the actual rule ID, and specifies comma-separated fact/earlier-step IDs
matching every antecedent. It retains sequential `S01,S02,...` numbering,
actual rule consequents and stopping at the query polarity.

The instruction contains no concrete fake rule application or atom to copy.
The renderer does not inspect the label, reference proof, example ID or
metadata. Graphs, canonical targets, sampling seeds, candidate count,
non-thinking chat protocol, model revisions and all acceptance thresholds
remain unchanged. The shared instruction also affects anti-shortcut prompts;
this is an explicit scientific prompt change, not merely an execution fix.

Instruction SHA-256 is
`55edb00c4d57197224c49dddfcb36ad1152640c8f8a19e2b223c1f3e968c504f`.
The pinned Qwen tokenizer measured 87 instruction tokens and 402–1246 tokens
for all 256 formatted production prompts, within the existing 1246 bound.
Formatting uses the exact common chat prefix/suffix recovered from every
retained probe record; the GPU worker independently rechecks the production
tokenizer/chat path. All 256 original canonical proofs still verify.

The new `qwen3_v2_g0_candidate_e_seed42_prompt_v5.yaml` protocol remains
**proposed**. Historical accepted v3 and proposed, unsuccessful v4 are retained.
The frozen scientific configuration hash and all 52 named execution-safety
file hashes are unchanged. The historical Blackwell certificate is not H100
execution evidence.

## New GPU diagnostic

Job **54368250** was submitted once at `2026-09-20T22:45:35Z` and completed
on `exp-19-08` with Slurm `COMPLETED / 0:0` in 20m36s. It used one H100,
24 CPUs, 192 GiB and at most 30
minutes, account `nwu181`, partition `nairr-gpu-shared`, QoS
`nairr-gpu-shared-normal`.

- Intent: `9b859df2cdee4228b82f8dcf5f9d7924`.
- Run: `20260920T224412Z-332009900a23-7374520e`.
- Snapshot: `332009900a23a705924dae65ce93b3536a0f52c5134bfcb02d9563f552833a9b`.
- Reviewed dry-run: 499 eligible files, 5,021,219 bytes.
- Baseline: exact frozen prompt-v4 bytes, explicitly identified in the report.
- Population: the same fixed first 32 train prompts; baseline candidate zero
  versus v5 candidates zero through seven, with unchanged per-candidate RNG.

Compare the paired candidate-zero summaries for equal sampling effort. Eight-
candidate coverage is a separate diagnostic measure. Raw prompts, response
IDs, response text, token logprobs and verification traces are preserved.
Results are staged on node-local scratch and hash-verified after publication
to the existing project Lustre path. No original release is overwritten.

The finished read-only watcher used
`.sdsc/supervision/probe54368250-readonly-v1/`, with immutable plan SHA
`cf8a7c41157fe2cd89eb5c6b3c8e4fab7bfecac22c18a5b70e28a1fc2cd4fbc6`.
It observes every five minutes and fetches bounded terminal reports/logs; it
never submits, retries, cancels or promotes diagnostic results into training.
Its final state was recorded at `2026-09-20T23:07:18Z`. Both old formal flows
remain stopped.

The v5 candidate accepted **24/256** responses, covering **4/32** prompts.
The paired candidate-zero comparison is v4 **0/32** versus v5 **3/32**.
The report SHA is
`f961e7076f601b5ffc44d2abe80d07b80306d69e3741d1148180edb41bc18864`;
the fetched 2,350,160-byte raw ledger matches receipt SHA
`720d76b2634d135b1a5496227ce2176e1d686f4873ff14ecb9dd162fa4441d67`.
All 288 verification traces replay exactly. The baseline's 32 response-token
sequences, logprobs and seeds exactly reproduce the previous v4 candidate zero.
Narrow spacing and ID-padding hypotheses recover no additional candidates.

Independent first-failure classification found 64 fact-restatement lines,
58 references to later/self/missing steps, 32 literal/formula citations,
22 antecedent mismatches, 20 truncated responses, 11 literal-first lines,
10 rule-ID citations, seven empty citation lists and eight other syntax errors.
The generic `unknown_citation` category therefore conceals a dependency-order
problem, not just an ID-format problem. No parser relaxation is justified.
Only 16 of the 24 valid proofs equal the canonical step sequence.
Answer-tag correctness fell from v4's 196/256 to v5's 136/256, so the grammar
improvement must not be described as overall reasoning improvement.
v5 remains proposed and must not be promoted on these results.

## Bounded forward-derivation candidate

The proposed v6 prompt keeps the abstract grammar, attached parentheses,
comma-separated IDs and actual rule consequents. It explicitly directs forward
derivation from FACTS, forbids restating facts, and allows fact IDs or earlier
step IDs as premises. Its SHA is
`d7196f08386c4231415eaf3bdc973f8e5dcb493d5e815f308594548f839637a2`.
The instruction remains 87 pinned-tokenizer tokens, with all 256 formatted
prefixes at 402–1246 tokens. Hash-verified tokenizer metadata was copied once
to Quest; local tokenization reproduces all 256 previous remote v5 lengths.
Independent static review
confirmed unchanged graph/target bytes, all 256 canonical proofs, and absence
of hidden-label/reference/metadata/ID leakage in both prompt renderers.

The probe baseline is now the exact v5 instruction with its explicit v5
protocol identity. Candidate selection, population, generation and acceptance
stay unchanged; the v6 protocol is proposed. This is one cause-specific
diagnostic, not an automatic series of prompt-tuning jobs. If coverage remains
poor, inspect teacher capability and the independent frozen-validation
readiness contract before any larger training submission.

The v6 implementation is commit
`6c12ab9a278686a30b141be02e0313be38481f21`; independent review found no static
blocker for a bounded diagnostic, while the proposed protocol still fails
formal admission. Job **54368464** was submitted once at
`2026-09-20T23:24:32Z` and completed on `exp-19-01` with accounting
COMPLETED / 0:0 in 27m34s:

- Run `20260920T232307Z-41ee08e5252f-77a20595`;
  snapshot `41ee08e5252f1984f59e7a77b6522971645d631781742e04eb9297bad4155204`.
- Intent `1e3d211e6c7e43a8b63c6aefe9d083a6`; matching preview/upload contains
  501 files, 5,036,550 bytes.
- One H100 / 24 CPUs / 192 GiB / at most 30 minutes, same account/shared QoS.
- Finite read-only observer on Quest quser43, PID 1505692, under
  `.sdsc/supervision/probe54368464-readonly-v1/`, plan SHA
  `e03b29c62ce891fe528131c2d38f551432d1a5362e18c75f4f6853a29cbd0807`.

The read-only watcher finished at 2026-09-20T23:55:10Z and verified/fetched
publication. Report SHA is
`3e695297bdf038f1eddcd05ecefe821c5540f015119c60d369e57a30cd99cb3a`;
raw ledger SHA is
`6660106e52ef6d504adc1a8193a6ad86da76b635a286d5a56fccdb811913bece`.
All 288 traces replay exactly and all 32 v5 baseline outputs match the earlier
job down to token IDs/logprobs. v6 has 0/256 valid candidates and 0/32 coverage.
Every candidate begins its first proof step with Fxx instead of a rule call;
161 complete responses fail syntax and 95 are length-terminated. Of completed
responses, 143/161 have correct answer tags. This localizes a systematic
prompt-induced schema/verbosity regression, not general reasoning incapability.
Exact replay evidence is `.sdsc/diagnostics/probe-54368464/replay-diagnosis.json`.
No successor was started by the watcher.

The existing readiness evaluation uses one greedy response on each of 128
validation examples, with a 128-token completion limit and an exact canonical-
proof comparison (not merely verifier success). Its answer/proof thresholds
remain 0.90/0.85. Full training-store coverage does not establish this gate.
Syntax-constrained decoding or teacher calibration would change the scientific
generation policy/model and require a separately reviewed proposal; neither
may silently substitute solver-generated proofs for teacher responses.

The first 128 validation rows were reconstructed locally and matched the
actual preserved dataset (file SHA
`8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3`).
Pinned-tokenizer target lengths are 53–162; six seven-step proofs exceed 128
tokens. Bare positive literals and compact valid syntax still leave the same
six too long. All variants parse to identical canonical steps and verify.
Thus the completion limit does not by itself make the 0.85 proof threshold
unreachable: 122/128 standard targets fit. This is a CPU feasibility check,
not a teacher-readiness measurement or a reason to change the frozen limit.

## Separate capability diagnostic

The inference audit reproduced every seed in both retained ledgers (576 rows)
and all 32 cross-job baseline response IDs/logprobs exactly. All seeds are
distinct within each candidate arm. Ten v5 prompts repeat the same response
eight times; nine follow probability-one paths after the frozen sampling
filters. Across v5 candidates, 98.68% of processed token logprobs are zero.
This supports low-entropy wrong outputs under this specific policy, not a
random-seed bug or a general claim that the model cannot reason under another
protocol. Cache, attention masks, chat formatting, model revision and token
alignment were also checked. The audit is reproducible with
`.sdsc/diagnostics/probe-54368250/inference_audit.py`.

The new `qwen3-v2-teacher-capability-probe` measures the fixed first 128
validation examples using the existing readiness CLI's exact greedy generation,
counterfactual prefix construction, tokenizer alignment, first-rule/intermediate
scoring and thresholds. It records raw outputs and prefix scores. Its pure
metric reduction is compared directly against the original scientific function
in CPU fixtures; it does not invent formal Git bindings or emit
`teacher_readiness.json`. Greedy transition logprobs retain the original
generator's meaning; they are not probabilities of a stochastic greedy policy.

Its `passed` field means diagnostic execution and publication completed.
`metrics_passed` reports the unchanged metric conjunction separately.
`readiness`, `accepted_science`, `full_teacher_ready`, `g0_passed`,
`training_started` and `readiness_artifact_produced` must stay false, even if
all metrics pass. The diagnostic receipt's task identity cannot satisfy a
formal teacher prerequisite. It uses one H100, 24 CPUs, 192 GiB and at most
30 minutes; source staging, node-local execution, persistent read-back and
duplicate-submission protection follow the existing diagnostic boundary.

Before any GPU validation measurement, restore exact v5 instruction bytes
(SHA `55edb00c4d57197224c49dddfcb36ad1152640c8f8a19e2b223c1f3e968c504f`).
This selects the better training-probe candidate and rolls back v6 without
using validation outcomes for prompt selection. Both protocols remain proposed.
The worker records the actual renderer instruction SHA in its report.

All 128 independently checked v5 validation prefixes fit 404–1194 tokens; adding the complete
128-token output allowance reaches at most 1322, below 1536. Original graph
identities, parser/verifier, model weights and all 52 safety files are unchanged.
The capability worker has 27 passing CPU tests; independent worker/wrapper
review passed 30. The control registration was first exercised in isolated
fixture copies (112 CLI/remote/boundary tests), preserving active observer
hashes until the v6 observation became terminal. The actual controls then passed
196 combined tests; final focused verification covers the restored renderer and
metadata addition. The observer permits only a 1e-6 float32 rounding envelope
above one for raw top-k mass, without clamping or changing any scientific
threshold. These tests do not establish GPU capability or authorize promotion.

## Validation and progression

The capability implementation is commit `78661c2`. Final affected tests passed
72/72, including independent fixtures and both observer tasks; Ruff, shell
syntax and diff checks passed. Independent review confirmed the restored
renderer is byte-identical to v5 and the worker's only final delta records its
actual instruction SHA.

Diagnostic **54368737** was submitted once at 2026-09-21T00:05:39Z and completed
on exp-19-01 with COMPLETED / 0:0 in 5m51s, using the resource envelope above. Intent
`841f146c4b394884bbf504cdda644893`, run
`20260921T000425Z-24c06afe7afa-8d07dad1`, source SHA
`24c06afe7afa3df0cd72d503a7e9dcab6136708f1d196175b10047a4b73b6d3a`;
the preview/upload contains 506 files / 5,130,786 bytes. Its separate observer
is `.sdsc/supervision/capability54368737-readonly-v1/`, Quest quser43 PID
1701453, plan SHA
`b3c101d8686b14c07643f2ce52d065646c14ac1b0318693410a85f2c1946e23f`.
No formal store, calibration, G0 or pilot has been restarted.

The observer finished at 2026-09-21T00:16:17Z. The report's SHA is
`70ac8ccfe11034b61711697047d8e344902dd28f6de8cbbdefbeb2262281a51c`;
publication receipt SHA is
`ff6287bce86df48676641a820168d9be107ba484493075d88d29691845d70ce5`.
Reports/logs were fetched both on demand and by the observer. Four raw evidence
files (8,138,235 bytes) were independently receipt-hash-verified and fetched
into `.sdsc/fetched/54368737/raw-verified/`. The local
`.sdsc/diagnostics/capability-v5/replay_capability.py` reproduced 128 tokenized
prompts, seeds, output decodings and verification traces, then reproduced the
original scientific reduction of all 256 prefix scores. It executes only the
trusted scientific function's pure metric statements, never formal bindings
or artifact publication. The diagnostic result is:

| Metric | Observed | Required |
| --- | --- | --- |
| Parser-gated answer accuracy | 53/128 = 41.41% | 90% |
| Exact canonical proof accuracy | 23/128 = 17.97% | 85% |
| First-rule top1 | 21/64 = 32.81% | 80% |
| Intermediate-conclusion top1 | 0/64 | 80% |
| Minimum retained top-128 mass | 0.9999970794 | 0.90 |
| Target coverage | 128/256 = 50% | 90% |
| Corrupted-prefix recovery | 32/128 = 25% | 70% |
| Minimum causal shift | -6.7364874 | nonnegative, with every valid shift positive |

`metrics_passed=false`; only retained mass passes. Format validity is 72/128.
There are 24 verifier-valid generations, of which 23 equal the canonical proof.
Other first errors are antecedent mismatch 22, conclusion mismatch 2,
response syntax 32, step syntax 24 and unknown citation 24.
All 32 response-syntax failures reached the 128-token limit; 28 of those
examples have canonical proofs that fit within that limit. Among 96 complete
outer answers, 65 are correct, before the unchanged parser gate. The one
verifier-valid, noncanonical proof merely swapped independent branch order.

Independent replay of all 192 tokenized probes, both sides (384 total), found
no causal-index or token-boundary mismatch. However, all 128 intermediate target
sequences start with ` TRUE` (token 8214), while the prompt says to use the rule
consequent, which is rendered as a bare positive symbol. None of the 128 free
responses emits `-> TRUE `; all 24 accepted proofs use bare positive literals,
which the parser legitimately accepts. First-rule target coverage is 128/128;
intermediate target coverage is 0/128. Every intermediate causal shift is
positive despite failing exact target coverage. The representation mismatch
is a supported explanation for part of this pattern, but aggregate scores do
not identify the exact failing token or establish general teacher incapability.
Because bare positive literals already parse to identical canonical proof
objects, omitting TRUE does not explain the free-generation failures. The
step/citation and genuine antecedent/conclusion errors require separate
attention even if a reviewed prompt clarification resolves the prefix mismatch.

Adding an explicit positive-literal convention now would be validation-informed
protocol adaptation. Preserve this failed evidence, disclose exposure of the
first 128 validation examples, and never call a rerun on them independent
held-out confirmation. No new prompt candidate has been generated or submitted.
The pending research choice is to preserve the original teacher and review an
output-contract repair first, or introduce an independently specified teacher
adaptation stage. The latter changes the teacher checkpoint/scientific identity;
the existing two-H100 calibration trains the 1.7B student and cannot implement
that repair. Both alternatives retain the original success thresholds.

The targeted CPU suite passed 89 tests. Independent additional scientific
validation passed 58 tests, including stage-4, Qwen3 prompt protocol, execution-
science review, teacher store, method binding and anti-shortcut checks.
These results establish preservation and fail-closed admission. v5 improved
grammar but failed coverage. The v6 revision again passed the same 89 targeted
tests; its GPU result failed every candidate despite successful infrastructure.

Formal progression requires independent review of the actual implementation
commit and observed candidate results, followed by a separate review-only
acceptance commit. A new complete 256-by-eight teacher store must pass its
original full-coverage gate, and a matching new two-H100 preflight must pass
before calibration. Old teacher/preflight jobs cannot be mixed with new
scientific source. Calibration, G0, four-H100 preflight and seed-42 pilot retain
their existing gates; full factorial and Gemma remain outside this scope.

The provenance exporter now supports at most 16 genuine linear unpublished
commits before and including an explicitly named final implementation/acceptance
pair. It audits every new historical tree on export and restore, leaves the
public ref unchanged, and does not perform scientific acceptance. Independent
review and 40 provenance/upload tests passed. This permits retaining previous
repair and handoff commits without rewriting history or forcing publication.
