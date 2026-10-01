# Original initial checkpoint: native BF16 validation128 check

Status: implementation candidate requiring independent review and one fresh, authorized
execution identity. This document does not record a submitted job or a scientific PASS.

This check removes one execution difference in quality diagnosis 54558773: that
worker wrapped generation in BF16 autocast; the original formal scorer does not.
Its initial validation256 arm scored 0/128 for the original answer metric and
0/128 for full proof. Those exposed examples and that failure remain evidence.
The check neither selects a model/prompt nor resumes the rejected instruction
candidate. Learning-rate changes cannot alter this fixed initial checkpoint.

## Frozen scientific inputs and computation

Restore genuine scientific HEAD `6c04f804b302184b8ff95d00fab404e0531ed8d6`,
using the parent 54548846 provenance and original accepted history. The worker
must run with that verified science directory as cwd. Before loading any model,
validate the actual original student protocol, including its 49 named files and
distinct implementation/acceptance history. The transport snapshot is separate.

- Initial checkpoint: SHA-256
  `85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4`;
  3,441,276,375 bytes, without replacing it by a trained checkpoint.
- Original resolved configuration: SHA-256
  `05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe`;
  7,923 bytes, unchanged even though its historical output paths are not used.
- Family manifest: SHA-256
  `bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189`.
  Stage and validate all seven original splits with `load_dataset_family`.
  Evaluate only `family.examples("validation")[:128]`, in unchanged file order.
- Validation file: SHA-256
  `8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3`;
  75,081,250 bytes. No filtering, resampling, early statistical stopping or other
  split inference. Loading the family is not an evaluation of its other splits.
- Qwen3-1.7B model/tokenizer revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16/SDPA, offline original loader.
  Original instruction SHA-256
  `8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6`,
  unchanged Facts/Rules/Query and pinned non-thinking chat template.

Directly call the original `score_probe_candidates._score_examples` and
`aggregate_verification`. Load using the original `load_model_and_tokenizer`,
`move_model_to_local_cuda`, and `load_checkpoint_into_hf_model`; no FP32
promotion, FSDP, autocast, optimizer, NLL selection, or alternate generation loop.
Every original `generate` call retains exactly `input_ids`, `max_new_tokens=256`,
`do_sample=False`, original pad/EOS, and `use_cache=False`. Do not add an attention
mask or truncate a prompt. Require each full prompt plus 256 to fit within 1536
and the actual model context before generation. An observer transparently calls
the original bound `generate` and returns the same output object.

The worker separately pins nine scoring/loading/parser/verifier entry files.
The scorer and loader are additional to the old training 49-file contract.
Import-time matching of restored Python source to actual HEAD is deployment
integrity, not a new scientific fingerprint or an expansion of any certificate.
The node's complete provenance restoration remains the trusted source boundary.

## Exact criterion and limits

`initial_base_gate_satisfied` is true exactly when all 128 results are complete
and at least 13 have original `VerificationResult.answer_correct=True`:
`answer_accuracy >= 0.10`. Preserve the original definition. Syntax or invalid
proof steps make the answer false, but an empty/incomplete otherwise valid proof
can have a correct answer and zero proof reward. Thus answer accuracy, full-proof
reward and answer-tag accuracy are not interchangeable. Report original format,
answer and proof aggregates, without changing the criterion or verifier.

`passed` and `diagnostic_complete` mean the execution and artifact contract
completed, even if the base criterion is false. Always keep these eight flags
false: `student_accepted`, `g0_passed`, `pilot_passed`, `factorial_ready`,
`teacher_accepted`, `teacher_accepted_under_candidate`, `accepted_science`,
`formal_prompt_accepted`. A failure also forces the base-criterion flag false.

Neither outcome establishes anti-shortcut robustness on IID/transformed tests,
calibration improvement, pair-complete base-capable/challenge circuit cohorts,
causal evidence, full G0, student acceptance or permission for factorial/Gemma.
No retry to search for a passing result, and no holdout-driven prompt change.

## Execution and review interface

Fixed allocation: one H100, eight CPUs, 64 GiB host memory, 30 minutes; account
`nwu181`, partition `nairr-gpu-shared`, discovered matching QoS/runtime. Preserve
Slurm's CUDA visibility. Use node-local input/workspace and persist bounded
outputs before exit. Quest retains source editing and control; no remote daemon.

Worker CLI: `--inputs-json PATH --output-dir PATH`.
Input schema `quest-sdsc-student-initial-inputs-v1` binds parent/job/run/source and
plan hashes, verified science root, original config/initial file records
`{path,size,sha256}`, full dataset root/manifest hash and staged offline HF cache.
The worker generates `initial-probe.json` (schema
`quest-sdsc-student-initial-probe-v1`), `initial-prompts.jsonl` (128 original
examples, raw/model prompts and complete prompt IDs), and
`initial-records.jsonl` (128 complete response IDs/text, parsed and verified
traces). Node/memory reports and a publication receipt bind all small results.
Failures retain completed rows and false flags. Progress is bounded to every
16 responses and contains counts only, not generated text.

An independent auditor must verify identity/receipt/file hashes and all 128
rows, match the cohort to already verified quality prompts (SHA-256
`5a0f55af4a6af1d6364eb94188fd9c95975b0f8016f123dad0296c7ec3d050fc`),
re-encode prompts, decode actual response IDs, run the original parser/verifier,
and recalculate all metrics and the 13/128 criterion. CPU fixtures verify the
interface and semantic distinctions; they are not real native-H100 results.
