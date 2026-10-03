# V3 common-student preparation: complete quality rejection

Job `54615110` completed the prescribed 32 updates and all 12 development
checkpoints, but no checkpoint passed the frozen selection rule. The unchanged
parser and verifier replay all 18,432 responses identically. This is a failed
scientific result with complete evidence, not a memory or publication failure.

## Outcome

All counts below are correct proofs out of 256 per view. The IID answer band
is 26 through 153; an individual transformed proof rate may trail IID by at most
0.05. The mean transformed gap and each structure floor also remain required.

| Step | IID | Renamed | Facts permuted | Rules permuted | Paraphrased | Distractors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 34 | 37 | 11 | 28 | 27 | 33 |
| 4 | 73 | 65 | 36 | 40 | 60 | 38 |
| 5 | 63 | 76 | 32 | 42 | 63 | 41 |
| 6 | 79 | 86 | 38 | 63 | 76 | 49 |
| 7 | 103 | 104 | 53 | 83 | 103 | 74 |
| 8 | 133 | 134 | 69 | 107 | 136 | 107 |
| 12 | 150 | 153 | 94 | 121 | 157 | 142 |
| 16 | 162 | 167 | 140 | 145 | 161 | 145 |
| 24 | 196 | 193 | 169 | 192 | 206 | 171 |
| 32 | 200 | 192 | 178 | 179 | 212 | 182 |

Steps 1 and 2 have zero correct proofs in every view. At steps 8 and 12 the
IID structure counts are respectively branch18/110, chain73/86, DAG42/60 and
branch31/110, chain81/86, DAG38/60. These pass the structure floors, but the
permutation gaps and transformed mean still reject both checkpoints. Later
checkpoints exceed the IID answer ceiling and still fail permutation gates.
The historical checkpoints cannot be reselected or qualified.

## Supported diagnosis and proposed repair

V3 supplies eight training views per base. Three permute facts, three permute
rules, and all these permutations also rename symbols. There are zero views
that preserve the original names while changing only fact or rule order.
Development contains exactly those transformations. This is an observable
coverage mismatch; it does not establish a unique causal explanation for every
wrong response.

At step8, 66 IID-correct examples fail after fact permutation; at step12, 62 do.
Their first generated step is invalid in 50 and 44 cases. Of the paired losses,
17 and 14 cite exactly the first displayed fact in a wrong first step. Only
6 and 9 are otherwise the exact canonical first rule/conclusion. Other errors
include wrong rule/conclusion choices, incomplete premises and invalid syntax.

For source `pgpair-e8cbc82ec682eba1dfb7-neg`, shuffled ordinal8 starts its facts
with `F15 NOT DST_160000042_013`; the needed `F01 SYM_048` is fifth. Both step8
and step12 incorrectly output `S01: R17(F15) -> TRUE SYM_064`, whereas the same
model's IID response correctly cites F01. The rule requires SYM_048. This shows
one positional copying error without any change to the graph or valid target.

The prospective V4 candidate replaces redundant renameB and jointB with pure
fact-order and pure rule-order views. It retains identity, paraphrase, renameA,
rename-plus-facts, rename-plus-rules and jointA. There are still eight views,
with renamed coverage explicitly changing from75% to50%; fact/rule permutation
coverage stays37.5% each. New independent seeds and full exclusion of every
previous base and transformed population are mandatory. Keep the native initial,
FP32/W2/global64 optimizer semantics,5e-5 LR and every selection threshold.
Implementation, independent review and a distinct acceptance are required
before new preflight and fit submissions. This proposal is not GPU evidence.

## Execution and provenance

Fresh status at 2026-10-03T16:02:09Z reports empty queue; job/batch FAILED1:0,
extern COMPLETED0:0, elapsed21142s. The worker intentionally returned2 for null
selection; the node then published preserved artifacts and returned1. A later
launcher teardown message is not the primary failure.

Independent execution review verifies32 global updates/1,652,820 input tokens,
64 finite nonzero rank updates,12 dense checkpoints with exact311-key FP32 reload
on both ranks,48 native-BF16 parity checks and actual step4 full-state restoration.
All683 own-job memory samples pass, peak168.229/384GiB, noOOM/failcnt.
Publication read-back hashed119,417,324,544 large bytes in390.16s. Review checks
publication evidence; it does not independently download weights or rerun GPU
arithmetic. Only139,356,887 bounded small-result bytes were fetched.

- Plan: `.sdsc/student-branch/44d1ca080d7f0cb075a8d35f8934fb55/plan.json`.
- Fetch: `.sdsc/fetched/54615110/fetch-19byha9l/`.
- Publication SHA: `5aa13b37658d807e1beede498bf22feba66b1ba75c80e55f3f246c281d9f4706`.
- Report SHA: `1ba54dda54a865fcee0c66b74df4da8b9296d5cae0175bd7bbc511a2511ac7cb`.
- Independent raw audit SHA: `be9964301c0a6a09957e63c2ba84b923e4237fd0be243ff37df5a5c61b223b9c`.
- Independent execution review SHA: `9bbd337ab3c0bf1768d34977030717878a88f3bb4d21fc3f96fada2e3c4025ca`.
- Post-audit full diagnosis SHA: `0dc72ae476fdac2d0b45e6b5b944bdb0d5fc68a4d9de8d939e2cfea6dd12a144`.
- Paired first-fact diagnosis SHA: `0d161ed01efe05d4e5e00280f0cd4c06db56b4a45023eb6ae3584537b5670450`.

Audits and diagnostics live under `.sdsc/diagnostics/student-branch-v3/`.
The one-minute observer stopped normally on terminal failure at15:53:44Z after
353 polls. Do not restart it or resubmit its known intent. All formal model,
G0, pilot, factorial and execution-class acceptance flags remain false.
