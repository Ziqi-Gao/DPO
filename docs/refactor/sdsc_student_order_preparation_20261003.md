# Order-coverage common student preparation v4

V3 fit job54615110 completed its32 updates,1,652,820 input tokens and all18432
development responses, but no checkpoint passed the unchanged selection gates.
The formal independent raw audit replayed the entire population and reproduced
`selected_checkpoint=null`. Preserve this failed experiment and every earlier
protocol/checkpoint; none is a qualified common initial model.

The audited step8/12 IID proof counts were133/150 of256. Renaming scored134/153
and paraphrase136/157, but plain fact permutation fell to69/94 and plain rule
permutation to107/121. At step12, only the signed mean, fact-order and rule-order
gap gates failed. From step16 onward, IID answers also exceeded153/256. Selecting
a later checkpoint or weakening a gap would change the scientific question.

## Evidence and bounded interpretation

Paired comparisons retain the same base examples. At steps8/12, fact permutation
turned66/62 IID successes into failures;50/44 already failed on the first generated
proof step. Their original verifier reasons were antecedent mismatch30/37,
conclusion mismatch20/18, syntax10/6 and unknown citation6/1. The latter syntax
failures also hit the256-token generation cap; the other56/56 ended at EOS.
This is not solely a context-bound or parser failure.

A narrow trace check found only3/3 failures where a wrong F-number equalled the
displayed position of a required fact. Separately, an independent check found17/14
failures citing the first displayed fact, including6/9 whose first rule and
conclusion still exactly matched the canonical first step. These are different,
partly overlapping observations, not an exhaustive causal classification.
For rule permutations,43/42 IID successes became failures;18/14 failed on the
first generated step. Missing prerequisites, wrong available citations and
incorrect conclusions also occur. Do not characterize every failure as copying
an identifier incorrectly or as omitting a terminal branch premise.

Source inspection identifies a concrete coverage gap: V3's fact and rule
permutations occur only after entity renaming. Its original-symbol training
views contain identity and paraphrase, with no plain fact/rule permutation.
All2048 fit and1536 development canonical targets independently verify. The
execution evidence has finite nonzero updates, complete windows and exact
saved-master reload checks. These observations support testing a focused data
coverage change; they do not prove a unique model-internal cause or guarantee
that this successor will pass. V2 and V3 development populations differ, so
their aggregate accuracy difference is not a controlled causal comparison.

## Frozen prospective treatment

The new protocol is `prereg/amendments/qwen3_student_order_preparation_v4.json`,
implemented by `experiments/protocols/student_order_preparation.py`. Begin from
the same pinned native Qwen3-1.7B weights with fresh optimizer, RNG and cursors.
Replace the duplicate pure-renaming and duplicate joint-renaming views with
plain fact and plain rule permutations. The eight views are:

1. identity;
2. the unchanged surface paraphrase;
3. plain fact permutation;
4. plain rule permutation;
5. entity renaming;
6. renamed fact permutation;
7. renamed rule permutation;
8. renamed joint fact/rule permutation.

Renamed sequences change explicitly from75% to50%. Plain fact and rule
permutations each become12.5%; total fact- and rule-permuted fractions each
remain37.5%, and paraphrase remains12.5%. Citation IDs are preserved. The plain
and renamed versions use identical permutations of IDs; all symbolic canonical
complete-proof targets are checked with the original verifier and include EOS.
Teacher-generated standard proofs are not required.

Use independent fit pair seeds190000042+pair_index for exactly128 pairs;
original build_split development uses the200000042 namespace. Fit transform
case seeds are210000042+base_index*10000: rename+1, fact+3, rule+4; joint applies
both latter permutations after renaming. The original five-view development
suite uses220000042 with32 OOD distractors. Seeds were chosen before token
measurement; no row, seed or structure is filtered or replaced after measurement.

Preserve V3's256 fit bases,2048 sequences,32 complete global64 windows, with two
signed branch pairs, one signed chain pair and one signed DAG pair per window.
Depth is2+(window+slot)%3 and fit distractors4..8. Negative siblings rotate the
fixed view order left by one. Under the actual ordinal-modulo-two training
partition, each rank receives four of every view and structure counts16 branch,
8 chain,8 DAG in every window, with16 positive and16 negative sequences.

Keep W2 FULL_SHARD, microbatch4/rank, accumulation8, full-parameter FP32 masters,
response/EOS sequence-mean loss and constant AdamW5e-5, with all other optimizer
settings unchanged. The complete training population consumes1,614,932 tokens;
maxprefix998, maxresponse204 including EOS, maxmodelinput1161. It fits the
unchanged1536 input shape and2,000,000 global nonpadding-input-token budget.
All windows must fit before the first update; no truncation or target repair.

Isolation checks explicitly cover original144000-family rows, teacher8192/512,
V1 student2048/512, V2 bases2048/256 and views8192/1536, plus V3 bases256/256 and
views2048/1536. New bases and transformed views are checked for semantic, ID,
pair-group and pair-seed overlap. CPU regeneration checks complement the runtime
check, which hashes the actual persisted original family files before reading.

## Unchanged development and execution gates

Use256 new development bases from the original difficulty distribution and all
six original views:1536 records at each of12 fixed checkpoints, steps
1,2,3,4,5,6,7,8,12,16,24,32. The new development structure counts are88 chain,
70 branch and98 DAG. Maxprefix2194 plus256 generation tokens is2450, below the
unchanged preparation-only2454 bound. Formal qualification remains2244.

Complete all32 updates and all18432 responses before selecting the earliest
eligible checkpoint. Preserve IID answers26..153/256, IID proof>=.10,
mean transformed proof>=.08, each transformed proof>=.05, signed IID-minus-mean
and IID-minus-each-view gaps<=.05, and each IID structure proof>=.10. Use each
new structure's actual denominator. A null selection is scientific failure;
no extension, automatic retry, threshold relaxation or retrospective reselection.
The original template, greedy native-BF16 inference, parser/verifier, candidate
seed and complete six-view base-block rank partition remain unchanged.

All125 parent science files and four parent protocols remain byte-for-byte
frozen at accepted V3 parentd54749a89836a6b04e18121d561232dddfc0d284. Append only
the new protocol module and four `tools/sdsc_student_order*.py` controls to make
130 science paths. The new artifact starts proposed. A clean implementation
commit, independent non-author review, then distinct review-only acceptance
are required before runtime admission. CPU tests are not real-GPU evidence.

Keep2 H100,24 CPUs,384GiB, the discovered shared account/partition/QoS,1h
preflight and8h fit. Preflight performs only four updates, step4 full-state
restore, exact FP32/native-BF16 checks, the2454-token finite forward and eight
training-only responses; it never selects on development. Full fit restarts
native weights only after its matching new preflight and independent raw audit.
Keep publication reserves300/600 seconds,192GiB node-local free space,128GiB
large artifacts,32MiB prompts/16MiB other small files and224MiB small results.
Require actual accounting, verified persistent publication and full independent
raw replay. The finite read-only observers never submit, retry, cancel or fetch.

The already exposed original896 formal responses remain disclosed; they are
not a fresh holdout. This successor changes preparation only. A future eligible
candidate still needs its separately reviewed qualification adapter and every
original capability, anti-shortcut, circuit and calibration gate. All OPD/RL
methods must share the same eventually accepted initial bytes. Qualification,
G0, pilot, factorial and execution-class acceptance remain false here.

Local evidence is under `.sdsc/diagnostics/student-branch-v3/` (actual formal
audit, full12-checkpoint diagnostic and paired first-error traces) and
`.sdsc/diagnostics/student-order-v4/` (new full-population CPU token/target,
isolation and review evidence). These ignored explanations do not replace the
formal audit or the versioned proposed protocol.
