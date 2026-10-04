# Controlled early order-learning diagnostic

V4 fit54643699 completed successfully as computation but rejected every candidate:
order robustness matured only after IID accuracy exceeded the preserved common-
initial band. The complete raw audit and bounded causal interpretation are in
`sdsc_student_order_failure_20261004.md`. No historical checkpoint is promoted.

This prospective diagnostic tests one change: independently randomizing both
fact and rule presentation order for every training view. Its protocol is
`prereg/amendments/qwen3_student_order_probe_v1.json`; the implementation lives in
`experiments/protocols/student_order_probe.py` and the five
`tools/sdsc_student_order_probe*.py` controls (controller, node, worker, auditor,
startup). It uses one plan/publication per arm, with no additional recovery layer.

## Paired treatment

Both arms begin from the same native1.7B artifact54548846 with fresh optimizer,
RNG and source cursors. They share256 fresh fit bases and128 fresh development
bases. Control uses the original V4 eight-view construction. Treatment applies
independent fact and rule permutations to each of those same views. Labels,
canonical proof targets, symbol maps, depth/structure, distractors4..8,50% renamed
content and12.5% paraphrases remain equal. Every64-slot window still contains
two signed branch pairs, one signed chain pair and one signed DAG pair; this
experiment does not simultaneously alter independent-pair batching.

Namespaces were checked unused and frozen before measurement:270000042 fit,
280000042 development,290000042 fit transforms,300000042 development transforms.
Control retains rename+1/fact+3/rule+4 offsets. Treatment uses
case_seed+100+2*declared_view_index for facts and+101+2*declared_view_index for rules,
with offsets indexed by declared view name independently of the preserved
negative-sibling rotation. No seed, row or view is filtered
or replaced in response to measured lengths or outcomes. Both arms reuse the
original permutation helper: shuffle once, then rotate one position if the
result equals the original order. Distinct per-view seeds are not rerolled to
force unique realized orders. The frozen treatment happens to produce eight
unique joint fact/rule arrangements for each of the256 bases.

The complete2048-view population is encoded and audited in each arm; the fixed
prefix768 is consumed in12 global64 updates. Both arms have1,618,412 full input
tokens and606,232 consumed tokens; maximum training input1165 is within1536 and
the full population remains within2,000,000. Shared development has44 chain,
50 branch and34 DAG bases,768 views, and maximum prefix+256 generation2450 within
2454. All canonical targets verify. Complete prior populations and their views
remain excluded. The actual hashed144000 original family is streamed only for
exclusion keys, never training/evaluation selection; formal896 generated answers
and scores are not consulted.

## Measurements and interpretation

Use the original full-parameter FP32/W2 FULL_SHARD update, microbatch4 per rank,
accumulation8, response/EOS sequence-mean loss and constant AdamW5e-5 with original
betas/epsilon, no warmup, clipping or weight decay. Evaluate exactly at4/6/7/8/12,
all128 bases and all six original views:3840 raw responses per arm. Generation
remains deterministic native-BF16, no autocast/cache/truncation,256 completion
limit and the original parser/verifier. Complete-base-block rank sharding stays
fixed,384 responses per rank at every observation.

The predeclared descriptive primary contrast averages, across steps6/7/8,
treatment-minus-control of IID proof rate minus mean(fact-order,rule-order) proof
rate. Report absolute IID/fact/rule rates and paired per-example wins/losses;
a smaller gap caused only by lowering IID is not an improvement. Report both
arms at all five steps, every view and every IID structure. There is no
significance threshold, adaptive extension, early model selection or acceptance
criterion at this smaller denominator. Formal thresholds are unchanged.

`passed`, `stage_complete` and `diagnostic_complete` mean complete diagnostic
execution regardless of correctness scores. `preparation_complete` is false,
`selected_checkpoint` is null, and student/formal/G0/pilot/factorial/execution-
class acceptance flags remain false. A zero-correct diagnostic is still valid
evidence if all execution, publication and independent replay checks pass.

## Execution and evidence

Each arm is one fresh permanent claim and2 H100/24CPU/384GiB/2h, under the existing
SDSC account/shared partition/QoS, at most four concurrently allocatable GPUs.
The new entrypoint integrates the reviewed300-second early-child limit,12-thread
environment, import timing/stacks, raw CUDA checks, rank startup/exit records and
bounded owned-child cleanup. It explicitly binds its own accepted worker bytes;
it never presents the new worker as historical V4 or mutates frozen module globals.

Step4 performs actual two-rank full-state restoration. Every observation checks
all311 saved FP32 master tensors before one BF16 cast, exact export-logit parity
and2454-token finite forward. Preserve the cgroup/headroom, node-local192GiB free
space,600-second publication reserve,128GiB large artifact and224MiB small-fetch
bounds. Required results must persist with read-back hashes before zero exit.
The independent auditor reconstructs the shared development population, actual
per-arm training manifest and all3840 decoded parser/verifier results; it does
not claim to rerun GPU arithmetic or download large weights.

All130 historical science paths remain unchanged. Add the six named new science
paths and pin only the eleven actually used external helper files. The proposed
protocol requires a genuine implementation commit, independent cross-review,
and a distinct review-only acceptance before deployment/submission. CPU tests
establish implementation evidence only. A future full preparation requires a
fresh prospective protocol and its original complete selection/qualification
sequence; neither diagnostic arm yields a common initial model.

## Implementation verification

The integrated CPU suite passes274 tests in237.70s:60 protocol/population,
109 transport/node/startup,39 worker and66 independent raw-auditor cases.
Ruff, formatting and Python3.12 AST checks pass. All130 historical scientific
paths,11 helper files and five accepted parent protocols match their frozen
bytes. Full population isolation and pinned-tokenizer measurements are retained
under `.sdsc/diagnostics/student-order-v4/`.

Independent cross-review covers each author surface. It found and repaired two
completion-validation gaps: raw optimizer records now require exactly12 updates
and two ranks with matching64 slots, microsteps, finite positive updates and
local/global/cumulative token counts; startup raw-log events must share one
monotonic-minus-elapsed clock origin. Rehashed contradictory evidence is rejected.
These repairs validate already-required evidence and change no training setting.
Real CPU seams include two-process Gloo loss/restore, tiny311-key FP32 export and
BF16 parity, fixed-tokenizer rows, full3840-response replay, actual negative-node
publication, rank startup/exit and owned-descendant termination. CUDA/FSDP model
fixtures are explicitly mocked where appropriate; this is not H100 evidence.

Review records are in `.sdsc/diagnostics/student-order-v4/order-probe-v1-review/`
and `order-probe-startup-independent-review.json`. Implementation b81b62a82ab8fa7b786f082796f9692026fbab7d received exact-commit
independent review and distinct acceptance ebb1fc5c25ac39ce536b51ef4b699af3247699d9.
The accepted protocol core is646efc0d80c63440172abf4140ac3ebf8b47d95f583054f027cdbefb84c0af3c;
accepted artifact SHA2bb8f3c9993eea7dc6a760639d0607f61896f5fd22599edd3f76bc268ac2d3e0.

## Submitted diagnostic

Both exact dry-runs and independent deployment review passed before submission.
Release20261004T140714Z-d2326e59f530-5ebc8a86 contains768 files/10,866,456 bytes;
genuine provenance983f2ec8129e060237911d5b5615c03ded84a94b807b59222398293ba5872cb6
was uploaded and verified against the accepted HEAD. All19 runtime pins match.

Control54648255 was submitted once at14:19:00Z with intent
21fc86300215de2e3b26b7d23e89078e, plan SHA
67606001ab0e0ab1b64dbbba1ccc7085e17f6903fe3081da8da0c36801ec93ad.
Treatment54648257 was submitted once at14:19:27Z with intent
75aaacbf055cfd68a5b729dfb7648101, plan SHA
f5d536f937366d899651410de7e4d344c220711e935eadc9b4694dd961c20c66.
Both are RUNNING at14:19:52UTC. A finite foreground observer checks both every60s,
with absolute deadline20:00UTC, and stops on any failure/unknown/connection loss.
It never submits, retries, cancels or changes either allocation. Its launch/state
are under `.sdsc/diagnostics/student-order-probe-v1/watch-pair-54648255-54648257/`.

The observer and prespecified paired comparator pass49 independently rerun CPU
fixtures. Root review fixes an observer receipt-path mismatch by binding actual
controller-generated timestamped submit/reconcile receipts explicitly. Their
ignored diagnostic files and review are under `.sdsc/diagnostics/student-order-probe-v1/`.
No actual GPU completion, causal improvement or student acceptance is established
by these submissions; full terminal accounting, artifacts and raw replay remain.
