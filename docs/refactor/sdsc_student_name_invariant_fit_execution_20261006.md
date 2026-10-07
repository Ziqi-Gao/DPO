# Name-invariant student preparation: full fit after recovered preflight

The recovered two-H100 preflight, job `54692520`, completed with accounting
`COMPLETED`, exit `0:0`, at `2026-10-06T05:36:47Z`. It ran for 676 seconds.
Fresh accounting and publication checks on October 7 UTC confirmed its four
optimizer updates, 256 consumed training views and 203,084 non-padding input
tokens. Both ranks completed full-parameter updates, actual Accelerate
save/load, exact master-checkpoint reload and bitwise logit parity. Native
before/after origins and the finite BF16 context probes passed. Peak recorded
host memory was 99.72 GiB within the original 384-GiB envelope.

The unchanged raw auditor independently reconstructed the training order and
replayed all eight training-only smoke responses. Its report is
`.sdsc/diagnostics/student-name-invariant-preparation-v1/preflight-54692520-independent-raw-audit.json`,
SHA `a1360693838e1a1ebee8a9a38747df7eb553438903ec2e91497c19c722e2188d`.
This replay does not independently recompute GPU numerics or historical runtime
isolation. Status rechecks published small bytes and large-file type/size against
the producer's recorded full readback; it does not freshly rehash or download the
30-GB checkpoint and optimizer set. The publication receipt SHA is
`aad46e81aee94b33f3e9af0863a8cd2a4ccdb8c5f6a875369fe4ace9fb7d66f5`.
Bounded terminal results are in `.sdsc/fetched/54692520/fetch-fuk6kze0/`.

## Why full fit needs a separate adapter

The successful recovery preserved the original unknown request's permanent
scientific claim. Both the frozen original controller and original node require
the preflight claim to match the new successful intent. Those gates deliberately
reject this recovered preflight. Deleting or rewriting the old claim would erase
the submission uncertainty and defeat duplicate prevention.

The new fit execution contract instead verifies the exact successful recovery,
accounting, original publication, raw audit, normal wrapper exit and retained
execution fence. It starts one normal original full-fit stage with fresh fit
scientific and execution claims. It never repeats either preflight, resets a
claim, changes a scientific threshold or promotes the four-step checkpoint.

The inner fit plan retains the exact old science release and genuine v2 history
at `dbd97a27ad0b836144b9bab2671c1c8b7bc50cd4`. The outer plan binds a separate
reviewed execution release and genuine v2 history. Both histories are restored
and validated. The new node uses explicit versioned orchestration: its scientific
body is the original body with named helper calls. It does not patch an imported
gate or invoke the old node's incompatible entry guard.

## Unchanged science and resource envelope

- Qwen3-1.7B, fresh original untrained initialization from the accepted parent;
  learning rate `2.5e-5`, full-parameter training, two ranks, global batch 64,
  32 optimizer updates, original 2,000,000-token ceiling.
- Original data, seeds, role-name intervention, checkpoint schedule, development
  evaluation and selection thresholds; original worker, startup, native audit
  and raw auditor bytes.
- Two H100s, 24 CPUs, 384 GiB, eight hours, account `nwu181`, partition
  `nairr-gpu-shared`, QoS `nairr-gpu-shared-normal`; 12 threads per rank.
- Original worker/staging clock 28,200 seconds and TERM at 180 seconds before
  allocation end. Wrapper setup is checked against 60 seconds before model work,
  runs outside that original clock, and consumes the original 600-second reserve.
  This is not a hard blocked-I/O deadline or a guarantee of a full remaining
  600-second reserve.
- The unresolved old request still reserves two GPUs. No other allocatable job
  may coexist with the new two-GPU fit under the four-GPU ceiling.

## Implementation and review boundary

New files are `tools/sdsc_student_name_invariant_fit_contract.py`,
`tools/sdsc_student_name_invariant_fit.py`,
`tools/sdsc_student_name_invariant_fit_job.py` and
`prereg/amendments/qwen3_student_name_invariant_fit_execution_v1.json`.
The contract pins 35 frozen dependencies alongside the original 203 science
files, covering 219 distinct existing paths. Its four named controls include
the unchanged bounded command observer. The original recovery contract and
science remain immutable.

Independent node review caught a real error: a saved live binding was compared
to a newly queried binding with a new timestamp. The corrected node validates
saved evidence through the frozen pure predicate, performs no Slurm query, and
tests both historical timestamps and mismatched identities. Independent AST
comparison confirms the full scientific body matches the original after
normalizing explicit helper qualification.

The contract has 40 CPU cases, the node 36, and the transport 51, including real
dual-history v2 export/restore rather than mocked Git ancestry. These validate
contracts and orchestration; they do not establish GPU fit success or student
qualification. Acceptance requires an implementation commit and a distinct
later review-only contract acceptance commit. The actual deployed plan and
remote dry-run must be reviewed before the single authorized submission.

## Continuation

Use the new controller's `prepare`, then matching `submit --dry-run` and one
authorized `submit`. Preserve every raw response and bounded command observation.
Missing or partial acknowledgement permits reconciliation only, never retry.
Continue known jobs with `status`, `logs` and `fetch`. Original scientific output
names are preserved; additional orchestration records use the `execution/`
namespace when fetched.

Full fit still needs accounting, verified publication and the unchanged full
raw/native audit before its preparation outcome is accepted. A fit outcome does
not grant held-out qualification, shared-initial acceptance, G0, OPD/RL pilot or
factorial readiness. No new supervisor, daemon or runtime installation is part
of this adapter.
