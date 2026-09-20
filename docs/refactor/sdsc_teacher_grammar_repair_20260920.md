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

The existing readiness evaluation uses one greedy response on each of 128
validation examples, with a 128-token completion limit and an exact canonical-
proof comparison (not merely verifier success). Its answer/proof thresholds
remain 0.90/0.85. Full training-store coverage does not establish this gate.
Syntax-constrained decoding or teacher calibration would change the scientific
generation policy/model and require a separately reviewed proposal; neither
may silently substitute solver-generated proofs for teacher responses.

## Validation and progression

The targeted CPU suite passed 89 tests. Independent additional scientific
validation passed 58 tests, including stage-4, Qwen3 prompt protocol, execution-
science review, teacher store, method binding and anti-shortcut checks.
These results establish preservation and fail-closed admission. v5 improved
grammar but failed coverage. The v6 revision again passed the same 89 targeted
tests; its GPU quality remains to be measured separately.

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
