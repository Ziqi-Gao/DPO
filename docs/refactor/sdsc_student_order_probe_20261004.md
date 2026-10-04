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
Both were RUNNING at14:19:52UTC. A finite foreground observer checked both every60s,
with absolute deadline20:00UTC, and stops on any failure/unknown/connection loss.
It never submits, retries, cancels or changes either allocation. Its launch/state
are under `.sdsc/diagnostics/student-order-probe-v1/watch-pair-54648255-54648257/`.

The observer and prespecified paired comparator pass49 independently rerun CPU
fixtures. Root review fixes an observer receipt-path mismatch by binding actual
controller-generated timestamped submit/reconcile receipts explicitly. Their
ignored diagnostic files and review are under `.sdsc/diagnostics/student-order-probe-v1/`.
The terminal accounting and independent raw replay results are recorded below.
These diagnostic runs never select or accept a common initial model.


## Completed paired diagnostic

Control54648255 and treatment54648257 completed with job/batch/extern0:0 in
5305/5353 seconds. Final verified empty own-job queues were observed at
2026-10-04T15:53:51Z and15:54:51Z. The finite observer ended verified_complete,
exit0,191 status queries. Preserve its terminal state and both permanent claims.
Both runs completed12 updates/606,232 input tokens, all five checkpoints and
3840 generated responses; selected_checkpoint remains null and every preparation,
student, formal, G0, pilot, factorial and execution-class acceptance flag is false.

Small results, excluding model weights, are36,372,330 bytes under
`.sdsc/fetched/54648255/fetch-duprqhff/` and36,377,938 bytes under
`.sdsc/fetched/54648257/fetch-ha53j27_/`. Publication hashes are respectively
03cbf6973ce8f41350d1a0763d3c84cd36c0dc37ced10f24c5df3526b9d697e0 and
21cb982bb112c87b51e8769e9cb739be90609aa96d646a87ac1e3aa1267dd2ec.
The frozen auditor reconstructs all768 prompts and3840 responses per arm:
control audit SHA4fee3f1fc03ecf4486dcd347e73aba5b5da0d719fc08d7f965b5a3caa813e051;
treatment SHA5a895119ee042574c929464d63bad5f16c727204dd3c37df7cef1ec305adf46c.
The fixed comparator JSON SHA is
1678eeb1e77880e951fc77f34378d744242d2af705a72a6f5189f877dcc9bb9d.
These checks recompute parser/verifier evidence and artifact bindings; they do
not rerun GPU arithmetic or independently download/hash large model files.

Proof counts out of128, control/treatment:

| Step | IID | Rename | Fact order | Rule order | Paraphrase | Distractors |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 29/2 | 18/16 | 18/5 | 32/2 | 31/31 | 34/0 |
| 6 | 64/59 | 62/36 | 53/65 | 50/54 | 69/58 | 56/53 |
| 7 | 64/59 | 62/33 | 50/62 | 54/52 | 72/62 | 50/56 |
| 8 | 75/62 | 76/44 | 62/67 | 65/59 | 81/68 | 59/63 |
| 12 | 85/80 | 80/71 | 82/89 | 81/79 | 85/84 | 85/77 |

The declared steps6/7/8 gap contrast is-9.245 percentage points, decomposing
into IID-5.990pp and mean(fact,rule)+3.255pp. Fact proof improves+7.552pp;
rule changes-1.042pp; rename declines-22.656pp. Thus some absolute fact-order
improvement is real within this one paired diagnostic, but much of the reduced
gap comes from lower IID, and other capabilities regress. This is descriptive
single-seed evidence without a significance or acceptance claim.

Across6/7/8 rename pairs, treatment wins28 and loses115; proof counts113 versus200,
while format-valid counts355 versus354 and length stops27 versus30. Observed
antecedent_mismatch is205 versus140, unknown_citation19 versus3, and first-invalid
step S01 occurs91 versus39. This implicates premise/citation behavior more than
format or response truncation; it does not uniquely prove an internal mechanism.
Full evidence is `completed-error-patterns.json`, SHA
0de0e644c4c660e30ec338a69ee18e90e04942fef9c750620935e1190b77da21,
under `.sdsc/diagnostics/student-order-probe-v1/`.

The next prospective diagnostic restores two original-order training anchors
(identity and rename-only) while retaining independent joint permutations for
six other views. It keeps50% renamed coverage and all numerical settings fixed,
uses new excluded fit/dev populations, and reports separate rename/fact/rule
absolute-rate contrasts. The intervention jointly tests restored coverage,
compound difficulty and position cues; no outcome or unique mechanism is assumed.
The current diagnostic checkpoints cannot be promoted. A later complete fit and
original selection/qualification gates remain required.


Independent execution review also passes, SHA
24414b765795a162e579538605fe1a3010748283b920cd999a66d4555cf63731.
It rehashes25 small files per arm and verifies24 finite/nonzero rank updates,
actual step4 full-state restoration, ten311-FP32-tensor reload observations,
20 exact native-BF16 parity observations and ten2454-token finite forwards.
All171/172 own-job memory samples pass without OOM counters; peaks127.142/
127.146GiB within384GiB. Each arm persists12 large files,62,527,562,769/
62,527,563,281 bytes, with producer read-back hash evidence; publication191.24/
212.37s stays within600s. Early probes take133.850/118.882s. The control therefore
provides direct evidence that a legitimate startup can exceed the historical
120s bound while completing within the accepted300s limit. This does not
retrospectively identify the exact cause of every prior startup failure.
