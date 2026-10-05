# Paired learning-rate diagnostic: complete, residual renaming failure

Both GPU jobs completed the accepted diagnostic. Reducing constant AdamW LR
from5e-5 to2.5e-5 greatly reduced early capped, invalid generation and improved
several final development views. It did not resolve symbol-renaming invariance,
and it did not uniformly improve intermediate checkpoints. Neither checkpoint
is selected or qualified as a common initial model. Formal OPD/RL remains gated.

## Execution and provenance

Control54673886 and treatment54673887 each ran two H100,24 CPUs,384GiB with a
3h envelope. Fresh2026-10-05T15:14:56Z accounting reports job/batch/extern
COMPLETED0:0 for both,7704/6307s elapsed and empty own-job queues. Each completed
32 full-parameter updates,2048 ordered rows,1,660,712 input tokens,seven fixed
checkpoints2/4/6/8/12/16/32 and5376 raw development responses. Their data, order,
token masks, optimizer settings and per-candidate RNG agree except the accepted
LR intervention. All before/after optimizer LR arrays match their assigned arm.

Implementation04a8832bb0eff69d7b7a26823e7759d35b7f150e and distinct acceptance
ec349703d379400addd4ec244dcb79ee1b2b8059 bind protocol artifact
28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f.
Both use release20261005T062954Z-6b4d114e538c-5d4201ed and shared dataset
84227f0858cad084ed7e4ef72317d02c1e0ab18875628b4065c938bc4a510669.

Independent execution review replayed exact plans, actual live bindings,
accounting, all29 small artifacts per arm, raw update/token coverage, startup,
runtime, checkpoint reload, parity, finite context probes and actual W2 restore.
All242/196 memory samples pass, with respective peaks157.876/157.879GiB and
zero own-job OOM/failure counters. Publication took249.82/238.99s. Fourteen large
artifacts per arm are bound to node/publication readback evidence; remote model
weights were not downloaded or numerically recomputed by the CPU reviewers.

## Fixed readouts

Each entry below is control/treatment valid proofs out of128. These are all seven
prespecified same-step comparisons, not a selected favorable checkpoint.

| Step | IID | Rename | Fact order | Rule order | Paraphrase | Distractors |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 4 | 23/22 | 23/15 | 25/16 | 14/18 | 25/25 | 18/19 |
| 6 | 50/78 | 44/60 | 41/67 | 39/58 | 69/70 | 37/76 |
| 8 | 81/93 | 76/68 | 66/81 | 67/76 | 89/80 | 63/89 |
| 12 | 101/98 | 99/78 | 92/91 | 95/88 | 92/96 | 84/97 |
| 16 | 100/105 | 95/94 | 102/107 | 103/105 | 92/107 | 85/101 |
| 32 | 118/120 | 107/108 | 117/123 | 117/120 | 120/122 | 113/119 |

At step2, all768 control responses hit256tokens and fail syntax. Treatment has
18 capped responses and497 format-valid responses, but still zero valid proofs.
This is evidence of improved early generation behavior under the measured LR
intervention, not proof that2.5e-5 is optimal or that format alone solves reasoning.
Exact response duplication alone is unsuitable as a collapse measure: treatment
IID duplicate excess rises from4 to44 despite eliminating that view's caps.

At step32 the renamed view gains only one proof:7 paired treatment wins versus
6 losses. The IID-to-rename proof gap is11/128(8.594pp) for control and12/128
(9.375pp) for treatment. Treatment renamed caps increase from3 to10. Therefore
the final small overall gains do not establish renaming robustness. Step12
renaming is substantially worse under lowerLR,78 versus99 proofs.

The frozen additional nominal LR-times-step comparisons are control4/treatment8,
control6/treatment12 and control8/treatment16. Proof vectors in the table's view
order are respectively23/23/25/14/25/18 versus93/68/81/76/80/89;
50/44/41/39/69/37 versus98/78/91/88/96/97; and81/76/66/67/89/63 versus
105/94/107/105/107/101. These comparisons have different data exposure and do
not equate parameter displacement. The fixed comparison retains each view,
all three structural strata, paired wins/losses, answer/format counts, termination
and error histograms, exact duplicates and signed gap decompositions separately.
There is no cross-view primary average, significance or noninferiority claim.

## Remaining failure

An independent original-parser/verifier replay matches all10752 raw responses
and every reported view count. The same tokenizer verifies all768 canonical
development responses includingEOS fit the unchanged256-token allowance:
maximum203 for renaming and163 for the other views. The ten final treatment
rename-capped examples have valid target lengths119–203. Every one already
contains an invalid premise/citation in a completed stepS02–S09 before truncation;
all are depth4, comprising eight branches, one chain and one converging DAG.
More output tokens cannot repair those invalid prefixes. Prefix replay used an
explicit synthetic closing wrapper only for error diagnosis; original scores
remain failed and no truncated output is accepted as a proof.

Current evidence points to dependency/premise tracking under opaque symbols,
rather than an impossible generation budget. Source inspection identifies a
testable shortcut: native branch/DAG atom names contain paired stems with `_L`
and `_R`, while the renaming transform removes that structure. A whole L/R swap
alone preserves branch grouping and can yield a valid reversed traversal, so
it cannot isolate dependence on these cues. A future diagnostic must separate
structural name hints from token length/copying, preserve the original verifier
and budgets, and receive its own review before execution. This mechanism is
currently a hypothesis, not a confirmed cause or an implemented repair.

The development population comprises128 bases in64 signed pairs repeated across
views and checkpoints. This is exploratory after multiple adaptations. Previously
exposed formal896 responses were not used here. Neither arm can be promoted from
this diagnostic, and no model/G0/pilot/factorial/execution-class gate is waived.

## Evidence locations

Bounded fetches are `.sdsc/fetched/54673886/fetch-rbozfxw5/`(46,479,770B) and
`.sdsc/fetched/54673887/fetch-uelo8apy/`(45,118,665B). Local analysis below is under
`.sdsc/diagnostics/student-focus-lr-probe-v1/`:

| Artifact | SHA-256 |
| --- | --- |
| Control publication | `431d2cff78c620caf652982bf9363071f4d81b1d5ccf5d97b8349a60f1d9ade9` |
| Treatment publication | `88345bb575423c0eb1276aeef25317bd086d2da0ec2c8ed828f6ab909cd8de8a` |
| `control-raw-audit.json` | `2a962a358ceddbea15456b39d5294c14a279cbed179112e8474f31cab3303078` |
| `treatment-raw-audit.json` | `ad9aa84691572af3268d7f25c3ba9128a2d39c0596f135cb4bbd2611b663743b` |
| `paired-comparison.json` | `742a6254a2931289acefcf3e9156ec79eddf51d1d9f6b42cf9edbac95c8db2e5` |
| `paired-execution-independent-review.json` | `5903fd3ee9608accdac62b40053f03ee86ec410f894d73ef595305f8b865d011` |
| `independent-canonical-response-feasibility.json` | `a1db321f8ff65557830f96b4b66d539f6f8226be6d91312f55e640eaba1f739d` |

Observer28744 stopped at07:09:31Z on a failed queue query. Its replacement54658
stopped at08:30:48Z after154 polls on an explicit45s owner-squeue timeout.
Both stopped states and original raw observations are retained; neither was
silently re-armed. Later exact-job terminal accounting and actual live bindings
reconcile both completed GPU jobs. No submission was repeated or cancelled.
