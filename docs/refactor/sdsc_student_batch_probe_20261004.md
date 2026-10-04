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
observed79.7/85.6-minute12-update predecessors. Actual completed runtimes are5746/5714s.
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
before GPU results. Root static Ruff/format/AST/JSON checks pass. Exact-commit
science and corrected-worker reviews pass for implementation
fa1b8b5693db644a842a7bbefe72def353da0d48. Separate acceptance
07d5813b9319ac401f09ce9847d564f841f80a50 changes only review metadata and handoff;
accepted artifact is5c5f4bd45e0902083151eb966904db0e7588e20bedff3766ac00a9acfbb5594d.
Neither implementation acceptance nor CPU tests establish GPU/model success.

## Actual deployment and submission

Release20261004T184239Z-ac70f9af5121-555462c2 binds792 files/11,699,048B, code SHA
ac70f9af5121520ce5d22bc1f18eff6c7ff5fee2db260025563d1afdabf16410.
Genuine accepted-HEAD provenance is
7dca28abe649f0a03b742d485833120d8edcdd7c3e4bbc1f34422ca01588ac3c.
Independent actual deployment/two-plan review SHA
06851c5a2d1558c4adb695372008493692b3d72739e2759fdb0bb53bf6a648fe
verifies148 science/24 controls, original native54548846, all19 runtime packages,
exact resources, wrapper/script and distinct scientific arm claims. The initial
queue had zero allocatable GPUs; the second fresh preview after control submit
verified existing2+new2<=4.

Control job54659007 uses intent8c01dc6f5a748e1bcfcfd6dbb06ea315, plan SHA
6f5237183173aa34bbff74aaed0031fb911f3dfa953645b2ade8d24376ee33fd, receipt
submit-20261004T184954149588Z.json. Treatment54659010 uses intent
95d7e6e70a4a5ebeed4d5aafbb376e0e, plan SHA
e59545f3b425afb5945746b2777c800feecb6abf20f8504458b645de89468c2d, receipt
submit-20261004T185046166446Z.json. Preserve exact plan/receipt directories under
`.sdsc/student-batch-probe/`; each was submitted once and must never be resubmitted.

## Verified completed outcome

Both job/batch/extern rows are COMPLETED0:0, elapsed5746/5714s. Final empty own-job
queues were verified at2026-10-04T20:32:21Z/20:32:22Z. The foreground observer72184
ended verified_complete/exit0 after204 polls; never re-arm it or resubmit either
intent. Both arms completed32 updates,2048 rows,1,587,844 tokens and4608 responses.

Bounded fetches are `.sdsc/fetched/54659007/fetch-rxyjtq4f/`(40,813,816B) and
`.sdsc/fetched/54659010/fetch-m910ypr3/`(40,945,363B). Their status snapshots
retain completed queue entries; preserve them and use the later final statuses.
Publication hashes are
e1eec0c680b0c86cd68653cef5afb05abbb17bffae65ebb4c22beb0f4b6c9da1 and
2848bc69b2fef7504166271807151fb82983e39768f9a308b024e037fb9af410.
Independent raw audits reconstruct all2048 fit encodings,32 windows,768 prompts
and4608 responses per arm, with hashes
41562e93e6686366333571973d937a3f3f48a9c9ab5f7b1371a041c27448d9db and
88ae19fa01b695ef534e24f61b67d338c91bb714834dbf21d3dbaeab46e3dca3.

Independent execution review
ea4ad21e3a9be675134678fa2d450f28fea5efa504c23d7caf461e054a1bedbb
passes64 rank-update records, step4 genuine restoration, six dense exports,
12 exact311-key FP32 reloads,24 native-BF16 parity observations and12 finite2454
forwards per arm. All181 own-cgroup samples per arm pass with noOOM/failcnt;
peak142.468/142.197GiB. Each13-file large inventory totals70,654,669,674B with
persistent readback evidence. Publication234.842/226.542s fits600s. Early startup
121.872/73.434s fits the accepted300s. Large tensors were not locally fetched or
GPU arithmetic independently recomputed. These are execution facts, not a
qualified common initial.

| Fixed steps6/7/8 proof | Control | Treatment | Difference, percentage points | Paired treatment/control wins |
| --- | --- | --- | --- | --- |
| Rename | 49.48% | 45.83% | -3.646 | 40/54 |
| Fact order | 49.22% | 54.17% | +4.948 | 39/20 |
| Rule order | 48.44% | 50.00% | +1.562 | 47/41 |

IID increases1.042pp. These are three separate prespecified contrasts; the same128
bases recur across checkpoints. At step32 control/treatment proof counts out of128
are IID124/120, rename107/104, fact123/119, rule123/121, paraphrase126/119 and
OOD-distractors122/113. No consistent batch-spreading improvement is established.
Frozen comparison JSON is6b6490b792fddd1f2775d29405a6396f9395463d428cb9e23fc43bfea3484010;
branch analysis is651b149932a1f612f82f190b1fccba6281d6631a92694680010b82ac717927e2.
All scientific summaries and evidence are under the batch diagnostics directory.

Primary IID chain counts138/144 and133/144 contrast with depth3/4 branch0/78
and2/78. Exact first-branch prefix denominators34/32 yield next canonical step
0/3 and immediately invalid next step31/28, with no missing next step. At step32,
IID branches rise to39/42 and34/42, but overall ability is above the unchanged
common-initial band. Remaining step32 rename gaps are mostly branch; early rename
gaps also include substantial chain losses. Finite training loss falls to
0.002757/0.001673; the lower treatment training loss does not improve its final
development performance. There is no observed recurrence of the historical
fixed-batch loss explosion and no new optimizer defect established here.

The next preparation will test a targeted data change:75% branch,12.5% chain,
12.5% DAG instead of50/25/25, with three branch signed pairs and one alternating
nonbranch pair per window. It preserves the anchored control order, all view
fractions, targets, optimizer and thresholds. This aims to learn difficult branch
transitions before easy structures saturate; improvement remains unproven.
A new full preparation and original qualification are required. No diagnostic
checkpoint has been selected or accepted.
