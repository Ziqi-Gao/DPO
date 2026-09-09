# Teacher output protocol repair

## Observed failure

Diagnostic G0 job `opd-15db6153a4b750c67fc4706b0c0aceb6` ran from
2026-09-09T18:35:32Z to 21:41:05Z on three scheduler-assigned GPUs. It failed
with exit 2 at `build_teacher_demos`, before student training. All 2,048
candidates were generated; 45 passed the exact verifier, covering only 13 of
256 prompts. The complete-coverage gate correctly rejected the store.

The diagnostic repair worked: the actual ledger, manifest and accepted view
survive in `/scr/del6500/OPD/diagnostics/teacher_demos/failure-g9wqngmn/`.
Direct inspection and independent analysis found:

| Verifier result | Candidates | Observed behavior |
| --- | ---: | --- |
| response_syntax | 881 | 880 reached the 256-token limit; one EOS response added external prose |
| step_syntax | 872 | 828 contain illegal `Sxx: F...` fact restatements; 871 ended with EOS |
| antecedent_mismatch | 188 | Formatted rule applications cite the wrong supporting premises |
| unknown_citation | 62 | Responses use symbols instead of existing fact or step identifiers |
| accepted | 45 | Exact rule-application proofs, no prose, 53–119 response tokens |

Parsing failures account for 1,753/2,003 rejected candidates (87.5%). All 881
length-terminated outputs contain exactly 256 tokens. Of those, 600 already
start with `<proof>`; requiring that prefix alone would not address fact
restatement and explanations inside the proof. All attempts retain actual
behavior logprobs.

For example, `pgpair-095b26e1aebe0f9eba88-pos:candidate-0000` describes a
fact activating multiple rules on one line. The accepted format requires one
rule application per line. In
`pgpair-a1d03f5e2f991d6d61af-pos:candidate-0000`, the final rule cites F01
instead of deriving its actual antecedent. The latter is a real proof error,
so fixing output instructions cannot establish full teacher coverage in advance.

## Scientific change

Replace the ambiguous one-line placeholder with explicit output and citation
instructions. Compact fact/rule punctuation to fit those instructions inside
the existing prefix limit. Preserve every fact, rule, identifier, symbol,
polarity and input order. Share the output contract with the anti-shortcut
surface paraphrase, so that transformation does not independently change the
required answer syntax.

The prompt does not consume labels or canonical proofs. The parser and exact
verifier remain strict. There is no output cleanup, reference-proof injection,
candidate replenishment, prompt filtering, token-limit increase, or coverage
waiver. Teacher model, sampling parameters/seeds, eight candidates per prompt,
256-prompt population, training semantics and accepted-view selection stay fixed.
Rendered prompt bytes and their recorded hashes change; this is a new reviewed
scientific implementation, not a claim that the previous experiment succeeded.

The successor science protocol is
`prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_prompt_v3.yaml`.
It retains the same configuration and reusable execution class. The modified
renderers are outside the descriptor's named safety files; actual tokenizer
checks must confirm the unchanged 1,246-token prefix and 1,536-token model
input envelope before accepting reuse. Original accepted artifacts are immutable.

## Scheduler update and retry

The latest central handoff and verified maintenance record confirm that the
new daemon started at 2026-09-09T21:41:31Z. Ordinary unknown application exits
now become terminal `application_unknown`; after confirmed process/scope
cleanup, devices follow cooldown and health verification. Historical quarantine
records are retained. The failed diagnostic job was classified by the old
daemon, so its `gpu_unknown` label is not hardware-failure evidence.

At 22:14 UTC, automatic intake was enabled, OPD armed, and no intake blocker
was present. Three historical GPU quarantines and foreign processes remained
central admission concerns. OPD does not acknowledge GPUs or operate services.
Publish one fresh builder-generated request within the existing authorized
scope after implementation and independent science-acceptance commits. Stage
the request outside production intake, commit its handoff, then atomically
publish its unchanged bytes. Preserve the launch HEAD through terminal state.

## Validation

The existing ProofGraph/stage-4/teacher-ledger/store suites pass 28 tests;
the existing anti-shortcut selections pass 3 tests (14 unrelated deselected).
The new 14-case suite passes and covers graph reconstruction, absence of hidden-label or
reference-proof input, signed/conjunctive syntax, and rejection of observed
failure forms. Exact commands use the isolated pytest runtime documented in
the earlier diagnostic report, with CUDA hidden, plugin/cache writes disabled,
and all temporary files under OPD scratch.

The pinned offline tokenizer verifies all 256 teacher prefixes are 402–1246
tokens, with zero over the unchanged 1246 cap. All 256 canonical targets are
byte-identical to the baseline and still verify, with lengths 53–162 tokens.
The final renderer SHA-256 is
`2a8fd4cb4f1507fc39fbebc58c0a9fef47547ef649913f1898229bc2d3e2108a`.
Evidence and its executable are in
`/scr/del6500/OPD/tmp/g0-prompt-repair-20260909/teacher_prompt_envelope.json`
and `verify_teacher_prompt_envelope.py` in the same directory.

Additional actual 128-example populations were measured: validation maximum
1194, IID 1172, circuit discovery 1200, circuit validation 1166 tokens. The
distributed validation prefix plus its completion remains below 1536. Across
all 640 anti-shortcut cases, the longest prefix decreases from 2020 to 1988;
paraphrase decreases from 1377 to 1364. This auxiliary serial inference path
already had a larger envelope than the distributed training limit; its
256-token completion ceiling is unchanged, and its maximum total decreases
from 2276 to 2244. Do not misreport these auxiliary prompts as below 1536.
Reproduction: `/scr/del6500/OPD/tmp/check_prompt_auxiliary_envelope_20260909.py`;
results: `/scr/del6500/OPD/tmp/prompt_auxiliary_envelope_20260909.json`.

All 52 named safety-file identities remain unchanged. Descriptor/certificate
validation, recomposed science-config identity, AST/import checks and
`git diff --check` pass. Science config SHA-256 remains
`6c3942f4d6cf87329e1c70ba32b0b8d4cdbd526a3bbcfdcb1afd3e1674662657`.
Independent acceptance must bind the exact implementation commit. CPU
fixtures establish preservation and rejection behavior; only the next central
GPU run can demonstrate improved teacher coverage or scientific completion.
