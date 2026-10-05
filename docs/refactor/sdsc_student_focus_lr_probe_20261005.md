# Paired constant-learning-rate diagnostic

V5 job54663831 completed training but had no eligible checkpoint. At step2,
1533/1536 responses reached the token cap with repeated output. Later steps
recovered formatting but retained premise/citation errors under renaming,
permutation and distractors. This motivates a bounded LR test, not a claim that
LR uniquely explains the rejection or that a smaller value will fix it.

The new proposed protocol is
`prereg/amendments/qwen3_student_focus_lr_probe_v1.json`. Both arms start from
the same native Qwen3-1.7B checkpoint of54548846 with fresh optimizer/RNG/cursors.
Control uses constant AdamW5e-5; treatment uses2.5e-5. All other training
semantics match: full FP32 parameters, W2 FULL_SHARD, global64/micro4/accum8,
response-and-EOS sequence-mean cross-entropy, betas0.9/0.95, eps1e-8, no weight
decay, clipping or warmup. Each actual optimizer group's LR is recorded before
and after all32 updates and independently checked against its arm.

## Fixed data and measurements

Both arms consume the same ordered2048 rows, token IDs, masks and rank ownership,
including every intermediate exposure prefix. Fit has256 independent bases:
192 branch,32 chain and32 DAG, preserving V5's eight views/two anchors,
50% renamed coverage and75% jointly permuted coverage. Fresh fixed namespaces
430000042/440000042 generate fit/dev and450000042/460000042 generate transforms.
No seed search, filtering or reroll is allowed. New dev has128 bases and768 views.

Actual tokenizer checks give1,660,712 input tokens over32 windows, below the
unchanged2M budget. Fit maximum model input is1163; dev maximum prefix2198 plus
256 generated tokens exactly meets2454. All canonical targets pass the original
verifier. Full CPU exclusion checks pass against all144000 original family
examples, teacher8192/512 and historical preparation/diagnostic bases and views,
including V5. Actual GPU jobs must separately verify the persisted input bytes.
The shared dataset manifest is
`84227f0858cad084ed7e4ef72317d02c1e0ab18875628b4065c938bc4a510669`.

Checkpoints2/4/6/8/12/16/32 produce5376 responses per arm. Every checkpoint
requires exact FP32 reload, BF16 logit parity and finite2454-token probes on
both ranks. Step4 requires actual complete optimizer/model/runtime save/restore.
All32 updates are mandatory regardless of intermediate scores.

The primary descriptive report keeps all seven same-step comparisons and each
of the five transformed views separate, with IID as a companion. It reports
absolute proof/answer/format rates, paired wins/losses, signed IID-view gap
decompositions and each structure. Cap/error counts and exact response-text
duplicate multiplicity describe generation, especially step2; legitimate proofs
can repeat, so duplicate counts alone do not establish collapse. Additional
fixed comparisons are control4/6/8 versus treatment8/12/16, and complete-exposure
step32. Equal LR-times-steps does not mean equal model distance; those nominal
comparisons have different data exposure. No averaging across views or adaptive
extension is allowed. Worse IID performance alone is not robustness improvement.

Repeated development adaptation remains exploratory despite fresh seeds.
Previously exposed formal896 responses are not read for this diagnostic. No
diagnostic checkpoint may be selected, promoted or used as a common initial
model; all model/G0/pilot/factorial/class-acceptance flags remain false.
Original numerical gates are unchanged. No optimum-LR claim is licensed.

## Execution and verification

Each arm has a fresh permanent scientific claim and submission intent under
`.sdsc/student-focus-lr-probe/`: two H100,24 CPUs,384GiB and03:00:00,
accountnwu181, nairr-gpu-shared/nairr-gpu-shared-normal. At most four GPUs may
be concurrently allocatable. The finite worker budget is10200 seconds plus a
600-second publication reserve; early startup has its existing300-second cap.
Readiness gates cover the exact native parent, runtime19 package pins, GPU-node
mounts,192GiB node-local free space and own-job memory. Required artifacts are
read back on persistent storage; bounded reports are fetched without weights.
Missing submission acknowledgement requires reconciliation, never retry.

Author verification covers82 core,183 transport and169 worker/auditor cases.
Independent reviewers ran18 core,74 controller/launch and34 worker/auditor
cases, with no blocking source finding. These include actual tiny-model AdamW,
two-rank CPU32-update/restore and worker-to-auditor5376-response seams. CPU
evidence is not real H100 completion or a scientific outcome.

The ignored prospective observer and paired analysis are under
`.sdsc/diagnostics/student-focus-lr-probe-v1/`. They must be separately reviewed
and hash-pinned before use. The observer only reads exact plans/receipts/status,
stops on SSH loss, changed controls, unknown state or failure, and never submits,
cancels or retries. It retains raw unknown statuses before rejecting them.
The root session performs any subsequent evidence-led repair.

All154 parent scientific files, nine historical protocol artifacts and11 shared
helper pins remain immutable. The new protocol binds six additive scientific
files, requiring an implementation commit followed by independent review and a
distinct review-only acceptance commit. This document describes the proposed
implementation; current acceptance, deployment and job identities belong in
`docs/refactor/current_handoff.md` and the immutable local receipts.
