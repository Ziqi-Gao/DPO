# Name-cue diagnostic: complete outputs, failed execution acceptance

Job `54687025` ended `FAILED 1:0` after 2252 seconds on two H100, 24 CPUs and
384 GiB. Both ranks produced their 512 responses, then the final runtime-origin
check rejected a PyTorch-generated temporary import. The original science and
execution publications remain failed. These results support descriptive failure
analysis and planning a separately reviewed preparation only; they are not an
accepted scientific run, model qualification, G0 result or checkpoint promotion.

The latest reconciled status at `2026-10-06T02:32:10Z` still contains
`54687025|FAILED` in the queue response. Accounting records job/batch `FAILED
1:0` and extern `COMPLETED 0:0`; do not describe the queue as empty.
The bounded fetch is `.sdsc/fetched/54687025/fetch-53wdbbl1/`, 34,470,691 bytes.
All 18 science-publication files and 17 execution-publication files match their
recorded sizes and hashes. Publication identities are:

| Publication | SHA-256 |
| --- | --- |
| Original science, failed | `22fdd458b74c895f709cdc596042c6e2c69bce652d8bde5e98662e3e6c3bc35c` |
| Recovery execution, failed | `2876568c3b9ebdbc09831880cdb7db8ece4f5ee837b9121f4fd3ce3ac34bfb51` |

The unchanged original auditor's pure replay reconstructed all 512 prompts with
the pinned tokenizer, checked all 1024 response token sequences, original parser
and verifier results, checkpoint/map/condition/rank bindings, and reproduced the
reported readouts. Each model has exactly one response for every ordinal 0–511.
The normal publication validator still rejects `publication scope/binding
differs: passed`. The worker's earlier generation-complete report and nested
replay-complete flag do not override that rejection. Historical isolation
evidence was checked for consistency, not recomputed; GPU arithmetic was not
independently replayed.

The fixed inputs are step32 diagnostic weights from control `54673886` (LR
`5e-5`) and treatment `54673887` (`2.5e-5`). The 64 fresh bases comprise 48 depth4
branches and 16 depth4 chains, each rendered under two maps and four conditions.
No optimizer update occurred and no new checkpoint was produced or selected.

Each branch cell below has 48 responses. Entries are **valid proofs / valid
format / responses reaching the 256-token cap**. Original answer-correct counts
equal proof-correct counts in these cells.

| Model / map | Preserve | Break pairs | Break paths | Break both |
| --- | --- | --- | --- | --- |
| Control / 0 | 31 / 40 / 8 | 19 / 41 / 7 | 15 / 41 / 7 | 12 / 40 / 9 |
| Control / 1 | 31 / 41 / 7 | 19 / 40 / 8 | 20 / 37 / 11 | 11 / 40 / 8 |
| Treatment / 0 | 34 / 39 / 9 | 16 / 38 / 10 | 22 / 39 / 10 | 14 / 39 / 11 |
| Treatment / 1 | 29 / 37 / 11 | 18 / 34 / 14 | 21 / 38 / 10 | 12 / 31 / 18 |

Each chain cell has 16 responses. In the same condition order, control map0
proof counts are `15,14,14,15`, map1 `14,14,15,14`; treatment map0
`16,15,14,15`, map1 `15,15,16,14`. Thus control ranges 14–15/16 and treatment
14–16/16. Chains are generic naming-permutation controls, not branch pair/path
interventions. Some chain prompts repeat across maps; signed pairs, maps and
conditions are not independent repetitions and must not be pooled as such.

The branch losses are not explained solely by formatting or truncation. For
control map0, breaking pairs loses 16 previously correct cases and gains four;
15 losses remain parse-valid and only one reaches the cap. Breaking paths loses
17 and gains one; 16 losses remain parse-valid and only one reaches the cap.
Treatment map0 loses 23 versus five gains when pairs break, including 14
parse-valid losses; path breaking loses 16 versus four gains, including 11
parse-valid losses. Canonical targets including EOS require at most 212 tokens;
all eight variants of each base match prompt and target token lengths, and
prefix plus the full 256-token allowance is at most 1606 tokens.

Within these fixed failed-run outputs, both models and maps show sensitivity to
same-level naming pairs and path-wide suffix continuity. This does not identify
an internal mechanism or establish names as the sole cause of prior failures.
Both models also fail cue-preserving cases. Conditional effects vary: treatment
map0's path contrast with pairing already broken is only +2/48, with ten wins
and eight reversals. Near-ceiling chain results cannot establish invariance.
Posthoc prefix diagnosis may use explicitly marked synthetic closing tags for
truncated proofs; it never repairs their original verifier scores.

The execution failure exposes two origin-auditor coverage omissions:

1. Pinned Torch `distributed/nn/jit/instantiator.py` creates a temporary directory,
   adds it to `sys.path`, and generates `_remote_module_non_scriptable.py` when
   `distributed/nn/api/remote_module.py` is imported. Each rank's actual generated
   file is 2355 bytes, UID 543540, mode 0600, SHA-256
   `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`.
   Independent pure string reconstruction from template sources whose bytes
   match the runtime manifest produces that exact hash. The current allowlist
   rejects its work-directory search path and would also reject its file origin.
2. After that first rejection, 15 ordinary imported scientific dependencies
   would fail the narrower 168-file lookup. Every one matches both genuine
   original science acceptance `f1674f27b0b488ef1690fb5b67de639d64d3107f` and
   deployed HEAD `16f5febcf1990be8b7721b48b04fc5428ed45029`, using Git reads with
   replacement objects disabled. All 53 imported scientific files also match
   the already-bound full provenance manifest and deployment manifest.

The 15 explicit dependency paths are:

```text
src/posttrain_circuits/__init__.py
src/posttrain_circuits/artifacts/__init__.py
src/posttrain_circuits/core/__init__.py
src/posttrain_circuits/datasets/__init__.py
src/posttrain_circuits/datasets/proofgraph/__init__.py
src/posttrain_circuits/experiments/__init__.py
src/posttrain_circuits/experiments/protocols/__init__.py
src/posttrain_circuits/experiments/protocols/local_fork.py
src/posttrain_circuits/learning/__init__.py
src/posttrain_circuits/learning/teacher/__init__.py
src/posttrain_circuits/methods/__init__.py
src/posttrain_circuits/methods/controls.py
src/posttrain_circuits/methods/opd.py
src/posttrain_circuits/methods/rl.py
src/posttrain_circuits/models/__init__.py
```

Each rank's final inventory contains 3242 runtime files, 14 deployed control
files, 53 scientific files, 11 host native libraries and the one deterministic
generated file: 3321 total, covering 3430 modules and 260 native mappings. The
exhaustive comparison found no additional unexplained recorded origin. This
supports the diagnosis of incomplete origin accounting; it does not rewrite
the failed execution contract or turn an inventory snapshot into a complete
execution trace. All 65 memory samples and 195 own-job ancestor records pass
the original allocation-failure checks, with peak 80.451 GiB, minimum headroom
303.549 GiB and no observed OOM. This was not a memory-capacity failure.

The proposed scientific repair is a training-only, role-independent bijection
of the original tagged name pool **before constructing all eight views**, so
literal pairing and `_L`/`_R` continuity cease to predict proof roles in every
training view. Permuting only opaque renamed views would leave the original
shortcut available. A new independently reviewed preparation should start from
the original native 1.7B initialization, with constant LR `2.5e-5` motivated by
the [prior LR diagnostic](sdsc_student_focus_lr_results_20261005.md), fresh
independent data, and unchanged validation and common-initial qualification
gates. Neither the LR nor the new naming intervention is established as optimal.
The separate implementation is in progress and remains proposed; no new experiment
has been accepted or submitted.

Preserve the accepted v1 controls, runtime snapshot, both failed publications,
all original flags and the consumed one-recovery claim. No retry, checkpoint
promotion or threshold change follows from this analysis. Future execution can
use a separately reviewed exact generated-origin/dependency validator; a broad
scratch-directory or arbitrary-source exemption would not address the evidence
boundary. This document only records the failure and its descriptive results.

Local evidence is under `.sdsc/diagnostics/student-name-cue-probe-v1/`:

| Artifact | SHA-256 |
| --- | --- |
| `failed-54687025-local-raw-replay.json` | `af79caf132b18446ec67220df3cc6a3c7e9fa702cfbc481ebcbe04e31cbbe19f` |
| `failed-54687025-descriptive-readouts.json` | `ef7ab971001e09014838240542a6573fe196e3faba7e3a31f83f9cbdeacd5fd6` |
| `recovery-54687025-failure-independent-review.json` | `d2cba9d113c8d0a4278c808cb56c87f965b2c6d1df28b2ad781a1b0d4bf5c10f` |
| `recovery-54687025-failure-review-addendum.json` | `775ce425822659c17c2619ace632d8a9540f935fc5b04e8a24abd43b98ebd071` |
