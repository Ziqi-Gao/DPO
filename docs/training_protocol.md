# Training protocol

`FactorialTrainer` is the only controlled-cell update loop. A state source returns tokenized
trajectories with response spans. A supervisor prepares targets and computes one of three losses.
Prompt and padding tokens are excluded by the same causal shift and response mask in every cell.

Hard supervision uses teacher top-1 next-token targets at every visited response prefix. Soft
supervision minimizes forward KL to cached teacher top-k distributions. In renormalized mode both
teacher and student are conditioned on the retained set; tail-bucket mode adds one omitted-mass
category. OPD rejects a verifier object so a reward or policy-gradient term cannot leak into the
objective. Teacher retained mass, entropy, student mass on teacher top-k, and top-1 overlap are
logged.

Verified replay computes response NLL only for exact-verifier successes. Sequence normalization is
the default; token normalization is an ablation. Sampling retries live in the trainer, continue up
to the configured limit, and record generated/successful counts, effective sequences/tokens, reward
rate, and retries. Exhaustion raises a dedicated diagnostic exception.

Canonical SFT reads only an independent teacher-demo store. The store is produced by teacher
candidate generation followed by the exact ProofGraph verifier; no fixed-bank successes are reused.
Its manifest pins teacher ID/revision/resolved commit, sampling seed and parameters, candidates per
prompt, verifier version, retention rate, and prompt-manifest hash. SFT uses the same response
masking, optimizer family, scheduler family, token accounting, and checkpoint format. Canonical GRPO is delegated to TRL's
`GRPOTrainer`; beta, generations, temperature, completion length, loss type, reward scaling, and
accumulation are explicit. The primary settings use exact reward, beta zero, no SFT auxiliary, and
no teacher loss. Random-matched and format-only controls expose no semantic verifier signal.

Each collection round runs `steps_per_round` training microsteps. Gradients accumulate across
rounds until `gradient_accumulation_steps` microsteps have completed; `global_step` and
`optimizer_updates` advance only at a real optimizer boundary. Prompt, generation, response-token,
supervised-token, FLOP, and wall-clock counters are cumulative.

Online sources assign an exact policy version at each refresh and reject records older than
`max_policy_lag`; the main grid uses zero. Retry requests receive distinct deterministic seeds
without forcing a policy refresh. Offline sources return copies from an immutable common bank. All
state-source cursors, refresh state, request counters, and the prompt scheduler are checkpointed.

Checkpoints include model, optimizer, scheduler, scaler slot, Python/NumPy/Torch CPU/CUDA RNG,
prompt cursor, policy version, rollout round, resolved configuration, artifact hashes, and runtime
version. Production schedules save densely at 0, 1, 2, 5, 10, 20, 40, 60, 80, and 100 percent and
may add accuracy milestones. `SIGTERM` requests an atomic checkpoint before exit.

