# Core-v2 migration

`prereg/core_v2.yaml` is the active scientific registration. `core_v0.yaml` and `core_v1.yaml` stay
version-controlled as an audit trail but are not valid inputs to formal runs.

Core-v2 changes the scientific meaning of several artifacts:

- ProofGraph uses `proofgraph-v3-signed-paired`; both labels have nonempty proofs and pair siblings
  share the query and structure.
- Datasets use `proofgraph-dataset-v2-paired` and retain atomic pair-group hashes.
- Circuit probes use `circuit-probe-v2-stage-sequence`, a support-path swap, explicit stage targets,
  full target-token sequences, positions, and tokenizer-specific hashes.
- Trajectory stores use format version 2 and carry prereg, generator, label-semantics, and circuit
  schema bindings.
- Production run manifests bind the Git commit and SHA-256 of `prereg/core_v2.yaml`; a missing,
  uncommitted, or dirty preregistration is a hard refusal.

There is intentionally no in-place conversion of v1 datasets or circuits. Regenerate paired splits,
rerun pre-training student/teacher gates, freeze new probe cohorts, and rediscover circuits. Formal
loaders fail closed on absent/mismatched core-v2 fields and on content-hash mismatch.
