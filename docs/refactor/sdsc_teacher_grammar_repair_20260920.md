# Teacher output grammar diagnosis and prompt-v5 candidate

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

Job **54368250** was submitted once at `2026-09-20T22:45:35Z` and observed
running on `exp-19-08`. It uses one H100, 24 CPUs, 192 GiB and at most 30
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

The finite read-only watcher uses
`.sdsc/supervision/probe54368250-readonly-v1/`, with immutable plan SHA
`cf8a7c41157fe2cd89eb5c6b3c8e4fab7bfecac22c18a5b70e28a1fc2cd4fbc6`.
It observes every five minutes and fetches bounded terminal reports/logs; it
never submits, retries, cancels or promotes diagnostic results into training.
Inspect its actual state before continuing, and do not edit its pinned controls
while active. Both old formal flows remain stopped.

## Validation and progression

The targeted CPU suite passed 89 tests. Independent additional scientific
validation passed 58 tests, including stage-4, Qwen3 prompt protocol, execution-
science review, teacher store, method binding and anti-shortcut checks.
These results establish preservation and fail-closed admission, not improved
teacher generation. GPU quality remains to be measured by the new diagnostic.

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
