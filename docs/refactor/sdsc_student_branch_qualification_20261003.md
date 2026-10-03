# Qualification of the branch-prepared common student

Status: prospective implementation scaffolding. Preparation job54615110 is
running; no checkpoint has been selected or accepted. Qualification submission
remains fail-closed until a complete successful V3 fit and independent replay
of all18,432 development responses identify one unique earliest eligible model.
The user authorized the common1.7B preparation, subsequent gated training path,
monitoring and evidence-led repair. This document grants no new authorization.

## Scientific scope

The additive `student_branch_qualification` protocol and
`tools/sdsc_student_branch_qualify*.py` adapters preserve the125 named V3 parent
science files and four accepted historical protocol artifacts. They change only
admission and exact loading of the newly prepared checkpoint. The original
896 qualification prompts, task, tokenizer, rendering, parser, verifier,
generation and numerical gates remain unchanged.

Previous job54606205 already exposed all896 formal responses. Subsequent
preparation was adapted using that failure; the formal population is therefore
not an unexposed holdout. New development populations do not erase that exposure.
All later methods must share the exact same finally accepted initial bytes.

Require the selected model from actual V3 fit54615110, never its preflight
checkpoint or an unselected alternative. Freeze actual parent intent, plan,
publication, report and independent raw audit hashes plus the selected step,
relative path, size, dense-file SHA and matching two-rank FP32 master-state SHA.
No step4 assumption, checkpoint-list index assumption or placeholder acceptance
is permitted. Preserve the rejected V1/V2 models and all original evidence.

## Qualification and limits

Run inference only on the original validation128 and anti-shortcut IID128 plus
five transformations of each IID example:896 responses in total. The regenerated
prompt JSONL must remain13,466,230 bytes with SHA
`ac58b320c219c8611943d84fa194c4658b24d641e665c6a9f3c9f0b0a783c51b`.
Load exact saved FP32 masters before one native-BF16 inference copy. Preserve
256 greedy new tokens, no cache/autocast/explicit mask/truncation, seed42 once,
validation1536 and anti-shortcut2244 total context. Preparation2454 does not
change the qualification envelope.

The validation floor is13/128 correct answers, with no upper accuracy ceiling.
Anti-shortcut uses complete-proof reward: IID>=.10, transformed mean>=.08,
each transformation>=.05 and signed IID-minus-transformed-mean<=.05. Negative
gaps pass. V3 development's upper band, individual-view gaps and structure floors
are selection rules and must not be added to formal qualification. Retain all896
responses even when a gate fails. Independent raw replay verifies outcomes;
a complete replay can pass while the model fails qualification.

One H100/24CPU/192GiB/2h is the proposed existing inference envelope. Verify
actual node mounts, at least32GiB node-local free space, actual selected file
within8GiB and the own-job memory headroom. Parent admission uses the V3
W2/384GiB accounting contract and32MiB-prompt/224MiB-total fetch limits; child
qualification uses its W1/192GiB accounting and16MiB-file/48MiB-total limits.
These are separate contracts. Stage only the selected checkpoint under the
neutral local name `prepared-selected.pt`, plus bound report/audit and original
inputs. Start the node deadline before source and parent validation; reserve300s
for durable publication, subject to the inherited180s early TERM boundary.

Distinct implementation and independent review-only acceptance commits are
required before submission. Use a fresh claim and matching release/submit preview,
existing-master SSH, verified persistent publication and bounded fetch. At most
four allocatable GPUs, no duplicate submission/retry or cancellation. A successful
qualification permits later calibration/cohort/circuit/G0 gates; it does not
itself accept formal initial, G0, pilot, factorial or an execution class.

## Verification status

Generic implementation and independent non-author reviews are complete. All337
focused CPU tests pass together (149.97s):110 protocol,106 transport/node and121
worker/auditor cases. Ruff, pinned-Python AST and diff checks also pass. Meaningful
fixtures include genuine distinct Git acceptance and reselection rejection,
parent/child resource and file-bound separation, non-step4 node-to-isolated-worker
inputs, real tiny-Qwen311-key FP32 restore before BF16, and actual producer-to-auditor
replay of896 pinned-tokenizer responses for pass, base failure and anti-shortcut
failure. GPU, large-model and future accepted-candidate boundaries remain explicit
fixture substitutions; these CPU tests establish no real model result.

The original validate_config, runtime, prepare_prompts, recording_generate,
validation_summary and score_original helper ASTs are unchanged. The independent
new auditor reconstructs prompt tokens, parser/verifier traces and original gates,
and decodes the complete parent audit bytes/lineage. All125 parent source blobs
and four accepted parent protocols remain unchanged. No new candidate protocol
JSON is present, so operational preparation fails closed until one real selected
model and its completed audit are bound and independently accepted.

The prospective read-only qualification observer has49 independently passing
fixtures. Cross-node review found that qualification publishes `after_inference`
while the initial copied observer expected `after_training`; both reader and
summary were corrected before activation and a real-reader regression was added.
Final source SHA is
`d1dd7058897ceab51f8f918f5e5c40b18f082bec3d19b54545ac98a50e0f1826`.
It is unstarted and needs a future actual plan and submission receipt. It watches
one qualification every60s for at most4h including queue, with self/control pins,
own-job memory and no submission, cancellation or retry. Current fit monitoring
and all producer controls were untouched.

Root integration and observer evidence is under
`.sdsc/diagnostics/student-branch-qualify-v1/`; worker/auditor author and non-author
reviews are under `.sdsc/diagnostics/student-branch-v3/`. The later bound candidate
still requires an implementation commit and distinct review-only acceptance.
No real qualification run or model acceptance is claimed.
