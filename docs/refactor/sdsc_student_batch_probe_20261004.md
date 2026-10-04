# Controlled batch grouping and presentation-order diagnostic

The completed anchor experiment54655732/54655735 improves renamed-symbol proof
accuracy by13.542pp but lowers fact/rule permutation by8.594/6.510pp at the three
prespecified primary checkpoints. Both executions and all raw responses passed
independent review; neither model is accepted. See
`sdsc_student_anchor_probe_20261004.md` for complete results and evidence.

A more specific failure remains after naturally correct generation of the first
branch: control/treatment34/33 responses then fail at the first step of the other
branch. In the same primary window, treatment scores135/144 on IID chains but
11/90 on IID branches. The current global64 batch contains only four independent
signed pairs, each repeated in eight views. By steps6/7/8 the optimizer has seen
only24/28/32 distinct pairs. This motivates testing earlier exposure to more
independent graphs and lower within-window repetition. It does not establish a
unique internal cause or prove that the current learning rate is optimal.

The historical5e-5 choice came from a four-step comparison against5e-4, which
avoided the earlier fixed-batch loss explosion. Current training losses are
finite and fall to about0.017; this is not evidence of the same old overshoot.
The next experiment holds5e-5 fixed to isolate the overall batch-order policy.
It does not compare learning rates or change any scientific qualification gate.

## Frozen prospective intervention

New identities are `student_batch_probe` and
`prereg/amendments/qwen3_student_batch_probe_v1.json`. Both arms construct one
shared population of256 signed base examples/128 pairs and2048 views, using the
preceding two-anchor policy: original order for identity and rename-only slots,
independent joint permutations for the other six,50% renamed and12.5% paraphrased.
All row fields, IDs, metadata, prompts, canonical targets, token IDs and masks are
identical across arms. Arm identity belongs to execution and ordered manifests.
No row is regenerated or transformed according to its new position.

Control keeps all16 rows of a signed pair together, hence four pairs per global64
window. Treatment takes32 pairs with one view per pair per window. Let cycle
`c=0..7`, cohort`g=0..3`, local block`b=0..7` and structure slot`s=0..3`.
Pair index is`32*g+4*b+s`, with slots branch/branch/chain/DAG; optimizer step is
`4*c+g+1`, and declared view index is`(b+c)%8`. Within a window, order by view
`v=0..7` then`s`, using`b=(v-c)%8`. Put the label1 row first for even views and
label0 first for odd views, followed by its sibling. This preserves the old
rank ownership of every scientific row under global-slot parity.

Every global window remains64 sequences,32 branch/16 chain/16 DAG rows. Each
rank sees32 rows,16 positive/16 negative and four rows of each declared view,
with the same rank-by-view-by-label cells as control. The two even-index anchors
retain the previous rank exposure: rank0 positive anchors/rank1 negative anchors;
this experiment does not silently rebalance that subset. The first four treatment
updates expose all128 independent pairs. After32 updates, each arm has consumed
exactly the same2048 rows once, including all eight views of every pair.

Early checkpoints therefore differ in the independent graphs, depth combinations
and tokens already consumed. Those differences are part of the order policy and
must be reported. The experiment cannot separate earlier coverage from reduced
within-window correlation. At step32, the full data exposure is equal, but the
AdamW trajectory still depends on order; that is the intended comparison.

Fresh seed namespaces350000042/360000042 generate fit/development, and
370000042/380000042 determine transforms. They were checked unused and frozen
before one complete population was generated. No filtering, replacement or
seed search is allowed. All2048 training rows total1,587,844 input tokens per
arm, maximum1170 within1536; maximum response203. The new128-base development
panel contains42 branch/48 chain/38 DAG cases, each in six original views.
Maximum prefix2198 plus256 generation equals the unchanged2454 preparation
context limit. Original formal anti-shortcut context remains2244.

Full CPU isolation reconstructs the original144000 family and every historical
teacher/preparation population, including both preceding order and anchor
probe arms and development views. The original family is used only for exclusion
keys, not formal generated responses or scores. Complete isolation passed;
SHA692daffc6abd313e77033097837c0040951460358ee32d8254711ea7c6b81be1.
All142 prior scientific files, seven accepted protocols and eleven fixed helpers
remain byte-for-byte unchanged. The new implementation adds six scientific
files for a total of148.

## Training, reporting and completion

Each arm restarts native1.7B artifact54548846 with fresh optimizer/RNG/cursors.
Keep full-parameter FP32 masters, W2 FULL_SHARD, global64/microbatch4/accumulation8,
sequence-mean response/EOS loss, constant AdamW5e-5, betas0.9/0.95, epsilon1e-8,
zero weight decay/warmup/clipping,1536 training input and2M token budget. Complete
all32 optimizer updates. Evaluate exactly steps4/6/7/8/12/32:128 bases×6 views×6
checkpoints=4608 raw responses per arm. Step4 still performs actual full-state
restoration. Every checkpoint requires both ranks' exact311-key FP32 reload,
native-BF16 parity and finite2454-token forward. Greedy generation, tokenizer,
prompt, parser/verifier, no cache/autocast/truncation and256-token cap are fixed.

The three primary descriptive contrasts remain the separate treatment-minus-
control absolute proof-rate changes for renaming, fact order and rule order,
averaged over6/7/8. Never combine them into one primary score. Report all six
steps, six views, answer/proof/format rates, structures and paired wins/losses.
Step32 is a fixed companion after equal full-row exposure. For each view,
changed IID-minus-view gap is deltaIID minus deltaView; IID loss alone is not
improvement. Repeated checkpoint observations are not independent base examples.

Additional explanatory metrics report branch depths2/3/4 and, for depths3/4,
responses whose parsed first depth-minus-one steps match every field of the
canonical first branch. Report exact next canonical step, correct next rule ID,
immediately invalid next step, and missing next step. These overlapping
numerators are descriptive, not new thresholds. Parser failures have no steps
under the frozen parser and cannot enter that denominator; zero denominators
produce no conditional rate. Conditional denominators can differ between arms,
so they do not define a same-case causal mechanism contrast. The companion
helper must be reviewed and frozen
before results are observed.

No checkpoint is selected and every model/preparation/formal/G0/pilot/factorial/
execution-class acceptance flag remains false. Diagnostic completion does not
depend on correctness. A later common initial still needs a fresh complete
preparation, original earliest-eligible selection and896-response formal
qualification with unchanged scientific criteria.

## Execution and verification

Each arm requests2H100/24CPU/384GiB/02:30:00 under accountnwu181,
partitionnairr-gpu-shared/QoSnairr-gpu-shared-normal; at most four concurrently
allocatable GPUs. Model shapes and the2048-row population size are unchanged;
the allocation allows the extra20 training updates and sixth evaluation over
observed79.7/85.6-minute12-update predecessors. Actual runtime is not yet known.
Preserve300s early startup,12 threads,192GiB node-local free space,384GiB memory
checks,600s publication reserve,128GiB large and224MiB small artifact bounds.
Six dense checkpoints plus the original step4 full state require13 large files;
only bounded reports/logs are fetched. No retry, cancellation or blind unknown
submission recovery is introduced.

Author verification passes74 core,121 transport/node/startup,56 worker and88
raw-auditor tests (339 unique tests before the final identity-string correction).
Independent review found that the real worker report still used the prior anchor
source identity. Both actual-execute arm tests reproduced that rejection; the
two-string identity/progress correction then passed four targeted real-execute/
controller-identity/tiny-generation regressions. The worker suite includes actual two-process
CPU Gloo32-window global-mean comparisons and bitwise step4 restore/next-update
replay, actual shared token/mask/rank exposure, checkpoint32 and4608-response
producer-to-auditor interfaces. The raw auditor independently reconstructs both
orders, exact row equality, all2048 token encodings and every one of32 published
rank/slot/window/budget records. Root additionally reviewed normalized transport
deltas and ran33 focused non-author fixtures. CPU evidence is not GPU success.

Evidence is under `.sdsc/diagnostics/student-batch-probe-v1/`. Protocol core is
7523e1a43e23c4d98464c0307ebd474bacfaa39caffbb929755a9a7ac6365cc5;
proposed artifact85cf798f00c3ceed563b1d98a0a7e6870bde28f45a8bd41a78ce05fd13e2eab8.
Non-author reviews cover the core, raw auditor, controller/node, startup/worker
and ignored observation/analysis helpers. The observer/live reader passes58
fixtures, comparator30 and branch helper27; their reviewed bytes are frozen
before GPU results. Root static Ruff/format/AST/JSON checks pass. An exact-commit
review and distinct review-only acceptance must finish before deployment and
submission. No batch-probe job is submitted by implementation or CPU checks.
