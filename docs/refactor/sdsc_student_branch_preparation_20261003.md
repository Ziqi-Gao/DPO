# Branch-balanced common student preparation v3

The user authorized keeping Qwen3-1.7B, preparing one common initial model for
all later methods, and improving new failures while monitoring execution.
Preparation v2 job54613837 completed all128 updates and9216 development
responses, but selected no checkpoint. Independent raw replay reproduced the
rejection. Step4 had50/256 IID answers, but branch1/58, DAG4/98 and paraphrase8/256
failed. Later checkpoints exceeded the unchanged153/256 answer ceiling and still
failed the per-transformation renaming gap. At step64, IID branch proof accuracy
was57/58 and renamed branch accuracy37/58. Twenty-one of22 renamed failures
omitted a required branch premise; generated symbols were present in the input
and the outputs parsed correctly. This is evidence for improving proof coverage,
not evidence of a symbol-copying or optimizer implementation defect.

The execution audit found finite nonzero updates, exact saved-master reloads,
complete raw partitions and persistent artifact readback, with no own-job OOM.
The worker intentionally rejected the model. Preserve the full failed V2 result,
all V1/V2 protocols, original task/verifier and all numerical gates. No historical
checkpoint is reselected. These observations motivate a new prospective
preparation treatment; they do not establish that it will succeed or isolate a
unique causal mechanism.

## Prospective training distribution

The new protocol is `prereg/amendments/qwen3_student_branch_preparation_v3.json`,
implemented by `experiments/protocols/student_branch_preparation.py`. Start from
the same pinned native1.7B weights with fresh optimizer/RNG/cursors. Every global
64-sequence window contains four complete signed pairs: two branch pairs, one
chain pair and one converging-DAG pair. Depth2/3/4 follows a fixed rotation.
Explicit fit difficulty uses4..8 distractors, keeping full proofs and all tokens;
no example is removed or shortened after tokenization. The smaller fit contexts
are a disclosed new training distribution, not an evaluation change.

Each base example has eight views: identity, surface paraphrase, two independent
pure renamings, renaming with fact permutation, renaming with rule permutation,
and two independently renamed joint fact/rule permutations. Thus75% of sequences
use renamed symbols, up from25% in V2. Negative siblings rotate this fixed view
order left by one, balancing all eight views across both ranks under the actual
ordinal-modulo-two training partition. Symbolic canonical targets are verified
for every view; teacher-generated standard proofs are not needed.

The fixed population has256 bases and2048 training sequences:32 global64 updates,
one pass. Exact pinned-tokenizer measurement gives1,652,820 input tokens,
maxprefix1000, maxresponse203 including EOS and maxtraininginput1173. Keep the
1536 training shape and a2,000,000 global nonpadding-input-token budget. Token
admission must pass for the complete population before any update. All training
and transformation seeds are prospective and independent of previous populations.
Isolation checks include the original144000-family rows, teacher8192/512,
preparation-v1 student2048/512 and preparation-v2 student2048/256, including views.

Keep full-parameter FP32 masters, W2 FULL_SHARD, microbatch4/rank, accumulation8,
response/EOS sequence-mean loss, and constant AdamW5e-5 with the existing betas,
epsilon and zero weight decay. The observed failure does not justify changing
this already stable optimizer simultaneously. Original downstream training
budgets and OPD/RL method semantics remain unchanged.

## Prospective development selection

Generate256 new development bases independently, using the unchanged original
structure/depth/distractor distribution, and evaluate each in all six original
views. Their measured structure counts are86 chain,110 branch and60 DAG. The
largest prefix2196 plus256 generated tokens fits the unchanged preparation-only
2454 context; formal qualification remains2244. Greedy native-BF16 inference,
original template/tokenizer/parser/verifier, per-candidate RNG and complete
six-view base-block rank partition remain unchanged.

Save and evaluate at steps1,2,3,4,5,6,7,8,12,16,24,32. This observes the early
learning interval missed between V2 steps4 and8. Complete all32 updates and all
18432 raw responses before selecting the earliest eligible checkpoint. Preserve
all V2 selection criteria exactly: IID answers26..153/256, IID proof>=.10,
mean transformed proof>=.08, each transformed proof>=.05, signed IID-minus-mean
and IID-minus-each-view gaps<=.05, and each IID structure's proof accuracy>=.10.
Use each structure's actual denominator, not V2's population counts. No eligible
checkpoint means failure, with no extension, threshold relaxation or post-hoc
checkpoint reselection.

The changed training distribution and early checkpoint schedule are a new
preparation experiment, not a formal OPD/RL comparison. The prior qualification
exposure to all896 formal responses is disclosed. The new development examples
are independent; the later formal population is not claimed to be an unexposed
holdout. Successful development selection still needs an additive qualification
adapter and all original formal capability, anti-shortcut, cohort, circuit and
calibration gates. All compared methods must eventually share identical accepted
initial checkpoint bytes.

## Execution and acceptance

New `tools/sdsc_student_branch*.py` controls and a separate intent namespace
preserve every historical producer. A clean implementation commit and distinct
non-author review-only acceptance precede new execution. No protocol may accept
its own implementation commit identity. CPU checks establish contracts, not GPU
execution or model success.

Use2 H100,24 CPUs,384GiB, accountnwu181 and the discovered shared partition/QoS.
The preflight allowance is1h; it runs four real updates, saves onlystep4,
restores actual full state, verifies exact FP32 reload/native-BF16 logit parity,
performs the2454-token finite forward on both ranks and retains eight
training-only responses. It never evaluates development or selects an initial.
Full fit restarts native weights and requires this new matching preflight,
complete accounting, persistent publication and an independent raw audit.

Fit's upper bound is8h, supported by the historical128-update/9216-response
execution time2h48m21 and the doubled development evaluation count. Actual new
preflight performance must support admission. Twelve dense checkpoints plus
step4 full-state artifacts are estimated at119.4GB, below128GiB. Require192GiB
free node-local space,384GiB own-job memory with the original headroom rules,
bounded32MiB prompt files/16MiB other small files and224MiB total small results.
Reserve600 seconds of the fit walltime for publication (preflight300 seconds).
Persist and readback-hash all required artifacts before allocation exit; fetch
no weights. Keep at most four concurrently allocatable GPUs. Existing-master
SSH, matching sync/submit previews, immutable claims and no submission retry
remain mandatory. All Slurm operations occur on Expanse through SSH.

Monitoring is a finite foreground read-only process, with one-minute observations
of the receipted job and existing memory samples. It does not submit, cancel,
retry, relax scientific gates or provide autonomous model wakeups after the
conversation/SSH session ends. A captured failure leads to independent evidence
review and a separately identified prospective repair, not resubmission of the
failed intent. Model, formal-initial, G0, pilot, factorial and execution-class
acceptance flags remain false throughout preparation.
