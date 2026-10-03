# Exact prepared-student loading for future training

`models/prepared_student.py` adds a low-level loader for the model-only V3 dense
student export. Existing OPD/GRPO CLIs load BF16 before restoring prepared FP32
weights, which rounds trained masters. The new loader directly constructs the
pinned offline Qwen3 architecture on CPU in FP32, then restores all311 tensors
before a caller creates an optimizer or FSDP wrapper. It never loads native BF16
weights first. The caller's original BF16 forward configuration stays unchanged;
forward mixed precision remains the training engine's responsibility.

`PreparedStudentCheckpoint` is an immutable set of supplied file, master-state,
step, parent, dataset and protocol expectations. It is not an acceptance
certificate. The loader checks a regular nonsymlink file, actual size<=8GiB,
stable file identity and streamed SHA before and after deserialization. Only
`student_branch_dense_v3` model-only payloads with the declared metadata and false
formal flags are accepted. No preparation optimizer, scheduler, RNG or cursor
state is consumed.

CPU FP32 tensor keys, shapes, finite values, full trainable coverage and the exact
master digest are checked before restore. Tied storage must contain identical
bits, including signed zero; inconsistent aliases fail before mutation. Restore
uses copying into existing parameters, followed by bitwise/state-hash and alias
checks. Architecture construction and restoration preserve the caller's CPU RNG
and explicitly use CPU even when another default device is set. No CUDA,
optimizer construction or training is performed by the loader.

The91 focused CPU tests pass, with Ruff, format and AST checks. They include real
28-layer tiny-Qwen311-state tied/untied restore, one-ULP/signed-zero cases, actual
forward/backward/AdamW after restore, malformed state/metadata/file rejection,
8GiB boundaries, immutable caller config and success/failure CPU RNG preservation.
The production constructor boundary is explicitly substituted in tiny fixtures;
no full native1.7B/GPU execution is claimed. Independent review also checks that
the130 existing preparation/qualification sources and active controls stay fixed.

This additive primitive is currently unused by existing training CLIs. Future
method adapters must first admit an actually qualified selected student and bind
its genuine dense identity separately from native HF ancestry. They still need
accepted method protocols, learning-rate repair, actual method-shaped updates
and save/resume evidence, dense teacher loading for OPD, and all original G0 and
pilot gates. The loader alone supplies none of those approvals or scientific
results. Current fit54615110 is unchanged and remains separately monitored.
