# Student quality diagnosis after calibration 54548846

The user authorized continued diagnosis, repair and necessary fresh SDSC runs
until the model satisfies the existing acceptance criteria. Teacher qualification
54496291 and student calibration 54548846 remain immutable completed evidence.
No original threshold, response, checkpoint or unsuccessful job is overwritten.

## Observed symptom and feedback loop

Calibration 54548846 completed 33 global-64 updates and 1,961,368 nonpadding
input tokens. Its step-20 strict answer/proof/format measurements are all zero.
There is no step-33 evaluation: the original trainer evaluates every 20 updates
or at max_steps=120, but this run stopped earlier at the exact token budget.
The old evaluator retained aggregate scores, not raw generated responses.

The real CPU evaluation boundary accepts positive and negative canonical proofs
with all three metrics equal to one. Truncating responses through the same
boundary produces all zeros. New regression tests compare the diagnostic and
original evaluator's exact prompts, generation arguments, token limits, strict
metrics and model-mode restoration. Malformed/tag-only outputs remain rejected.

Strict answer accuracy is not an independent binary-label-only metric: the
existing verifier returns answer_correct=false when the response cannot be
parsed or its proof fails early. Any extra extracted-answer diagnostic therefore
has a distinct name and cannot replace the original acceptance metric.

## Ranked, falsifiable hypotheses

1. Prompt/decoding or evaluator mismatch: canonical replay or independent
   checkpoint inference should disagree with the original evaluation. CPU
   replay has not found a template, prompt-slicing or parsing defect.
2. Output truncation: the same greedy 256-token continuation should preserve
   the 128-token prefix and complete previously length-limited valid proofs.
   Only 6/128 canonical validation targets exceed 128 tokens, so that bound
   alone cannot explain every failed response.
3. Training-label/mask error: a direct loss replay should supervise a different
   token span. Actual collation/supervision replay instead matches reference CE,
   including the first response token and EOS, with no prompt/padding gradients.
4. Optimization or insufficient learning: compare actual initial/step20/step33
   outputs and canonical-target likelihood on both training and validation.
   The fixed inherited learning rate is 5e-4 with constant LambdaLR and no
   warmup/clipping. First four v6 losses are 1.0599, 4.9146, 4.2811 and 11.9122;
   the first update norm is 18.75 and probe KL is 7.82. These observations make
   optimization instability plausible, not a proven cause or a license to
   change the accepted settings without independent scientific review.
   Git history traces 5e-4 to the shared smoke default; student v1 through v6
   explicitly preserve it. It was not accidentally copied from the adapted
   teacher's distinct LoRA optimizer.

A real CPU tiny-Qwen/Accelerate/FSDP test also reproduces a direct `.generate()`
boundary failure: the method is bound to the inner HF model and bypasses the
root FSDP forward, producing an empty embedding-storage error. This is W=1
CPU evidence, not an explanation of the actual W=2 job's zero scores: that job
did not raise this error. Full checkpoint inference avoids this boundary.
A bounded two-rank CPU/Gloo attempt stopped at FSDP preparation with
`cpu vs cpu:0` device inconsistency, before generation; it supplies no W=2
generation result. The scripts, per-rank observations and original logs remain
under `.sdsc/diagnostics/student-quality-v1/`.
A second bounded attempt normalizes only the CPU device and reaches real
two-rank Gloo/FSDP setup, then exits with a native SIGSEGV without per-rank
results. It is inconclusive and was not retried. PyTorch's FULL_SHARD root-buffer
retention differs from the W=1 NO_SHARD path, so the single-rank reproduction
does not establish the cause of production W=2 zero scores.

The two seed-42 v5/v6 runs diverge numerically from update three. Their logs
warn about CuBLAS configuration and nondeterministic attention. The memory
envelope change has not been shown to cause that difference; identical seeds
are not evidence of bitwise-identical arithmetic.

## Fixed first GPU diagnosis

The separate `tools/sdsc_student_quality.py` control path and its node wrapper
run `tools/sdsc_student_quality_probe.py` without changing the accepted science
files or historical tools. The first diagnostic is fixed before seeing results:

- Source: a fresh content-hashed release after matching sync preview.
- Parent: completed calibration 54548846, verified actual publication and
  initial/step20/step33 checkpoint SHA/size. Never retrain or resume this parent.
- Runtime: existing Python 3.12.13 / Torch 2.8.0+cu128 / Transformers 4.56.2 /
  Accelerate 1.10.1; pinned Qwen3-1.7B model/tokenizer revision, offline only.
- Resources: one H100, 24 CPUs, 192 GiB, at most two hours; account nwu181,
  nairr-gpu-shared / nairr-gpu-shared-normal. All Slurm commands run over the
  existing Quest SSH master; no authentication fallback or submission retry.
- Cohorts: original validation rows [0:128] and original training rows [0:32],
  identical for all checkpoints. Validation is already exposed, not a fresh
  independent confirmation or a checkpoint-selection set.
- Generation: greedy, original non-thinking prompt, cache disabled, 1536 input
  envelope; separate 128-token original-budget and 256-token diagnostic arms.
  The original training monitor uses 128, but the existing G0 probe scorer and
  initial-checkpoint anti-shortcut evaluator already use 256. The latter arm
  still remains diagnostic here; it does not replace the full G0 evaluation.
- Precision: load and compare every saved tensor in its actual BF16 initial /
  FP32 trained dtype, then explicitly convert to BF16 forward parameters.
  CPU hooks confirm BF16 parameters/activations in original FSDP forwards.
  Single-GPU inference is still not bitwise replay of two-rank FSDP arithmetic.
- Evidence: full response IDs/text, exact prompt IDs and graph/target records,
  EOS/length reason, original parser/verifier traces, strict metrics, separate
  answer-tag diagnostics and paired-prefix equality. Canonical-target NLL and
  token accuracy include EOS and are separate from teacher-demo training loss.
- Storage: stage source and verified inputs on the real node-local work disk;
  verify input/persistent mounts there. Publish and read-back verify reports
  and raw evidence before allocation exit. Fetch only bounded small evidence,
  including the complete prompt/response JSONL files; never fetch model weights.

`passed` means diagnostic execution/publication completed. Student acceptance,
G0, pilot and factorial flags stay false regardless of measured accuracy.
Missing Slurm acknowledgement requires reconciliation; permanent parent/source
claims prevent duplicate submission. No old supervisor is restarted.

## Acceptance and next decision

The next repair must follow the actual raw evidence. A budget, learning-rate,
training-population or other scientific change requires its own proposed
successor and independent implementation/acceptance review; original protocols
and thresholds remain immutable. Reusing exposed validation during debugging
must remain disclosed and cannot become an independent holdout claim.

The existing G0 gates include original base strict answer accuracy at least the
configured minimum (0.10), calibrated strict answer accuracy greater than base,
teacher readiness, anti-shortcut and causal-circuit checks, plus actual
checkpoint/resume and artifact evidence. Calibration execution alone satisfies
none of the missing scientific checks. G0/pilot adapters still require coherent
migration to the accepted teacher/student contracts before formal progression.

Current mutable state and actual run/job identities belong in
`docs/refactor/current_handoff.md` and `.sdsc/diagnostics/student-quality-v1/`.

The first diagnostic has real job ID **54557365**, submitted once at
2026-09-30T21:22:47Z after independent review of implementation `d76daee`, the
actual runtime, parent artifacts and deployment. Its immutable source run is
`20260930T211723Z-49907235ba25-c8e1e004` and its plan is
`.sdsc/student-quality/d47f1122872cad16aeb8e772f2e37ea2/plan.json`.
Use the separate controller's `status` and `fetch` commands with that plan;
the ordinary `tools/sdsc` task registry does not own this diagnostic. No old
flow is restarted. The saved submission receipt makes resubmission invalid.

## Independent local replay

After the controller's terminal fetch, run `tools/sdsc_student_quality_replay.py`
with `--fetch-dir`, `--plan`, the actual `--job-id`, the externally reviewed `--plan-sha256`, the
verified publication's `--receipt-sha256`, `--tokenizer-metadata` and a new
`--output` path. The actual small metadata snapshot is
`.sdsc/diagnostics/student-quality-v1/pinned-tokenizer-metadata.json`; it binds
the model/tokenizer revision, EOS 151645 and maximum position 40960.

Replay checks complete artifact hashes, all ordered cohorts/checkpoints/arms,
raw responses, original parser/verifier step traces, strict and answer-tag
summaries, EOS/length stops and paired prefixes. Twenty-eight CPU tests cover
producer-shaped records, altered traces/counts/hashes, duplicate IDs, ordering,
metric type confusion, missing records, nonfinite values and acceptance
overclaims. CPU fixtures are not actual diagnostic or model acceptance.

The output explicitly discloses what is not replayed: full-tokenizer decoding
of response IDs, teacher-forced logits/NLL, full source dataset membership,
accounting, GPU inference and checkpoint/memory measurements. The original
controller's verified publication remains the trust anchor for those inputs.
No result turns exposed validation into an independent holdout. A fixed
initial checkpoint failing its original base-capability gate cannot be repaired
by relabeling a trained checkpoint or borrowing the teacher's PASS.

## First GPU failure and repair

Job 54557365 terminated FAILED/1:0 after 26m05s. The original initial model's
four arms produced 320 retained responses; both validation (128 examples) and
training (32) have zero strict successes at both caps. All 256-token arms used
the full requested cap, and validation format validity rises only to 13/128.
These are partial diagnostic observations, not a complete G0 decision.

The next checkpoint load failed before inference: passing a FP32 model config
into the real loader violates its BF16/SDPA contract. The earlier CPU tests did
not exercise that production-loader boundary. The repair must leave that guard
intact, construct with the original BF16 configuration, promote the CPU model
to the saved dtype before strict checkpoint loading, compare every loaded
tensor exactly and only then convert to BF16 for the original diagnostic
forward policy. New regression tests exercise the real configuration guard
and actual tiny-Qwen FP32 values that BF16 loading would round.

The final cgroup peak is 70.628 GiB with 121.372 GiB headroom and zero OOM kills.
All five published file hashes were independently verified after fetching;
the original failure path had not required this verification automatically.
The controller repair requires receipt-bound hashes for failed publications as
well as successful ones. Completed-arm progress should remain in a failed
report and small Slurm summaries, preserving NLL evidence if a later arm fails.
None of these repairs changes the existing teacher/student science or turns the
failed job into success. A new worker hash gives a fresh permanent submission
intent; the old release, claim, receipt and outputs remain immutable.

The repaired six-file CPU suite passes **114 tests**. Separate independent
reviews bind the worker/loading tests and the node/progress/replay changes;
review records are under `.sdsc/diagnostics/student-quality-v2/`. Progress
observes the same child under one fixed deadline and never restarts it. New
fetches are confined to `.sdsc/fetched/<actual-job-id>/fetch-*`. The original
failed fetch remains at its historical path. All 49 accepted student science
files and the nested 47-file teacher contract remain unchanged.

Independent replay of the partial initial results verifies the actual pinned
tokenizer's full prompt encoding and response decoding as well as every
original verifier trace. Canonical targets all fit 256 tokens; 318/320 generated
responses already start with `<proof>`. A colon-only explanatory transformation
still gives zero strict successes in both 256-token cohorts: invalid citations,
wrong antecedents and incorrect conclusions persist. This does not modify the
original responses or scoring. The report is
`.sdsc/diagnostics/student-quality-v1/partial-initial-audit.json`, SHA
`d233e378db405a852e91018b93c7e0d515b264e6ba6a3cccd393cf6679ecec7f`.
Initial canonical NLL was lost on the original exception path and remains
unknown; neither trained checkpoint has yet been measured.
