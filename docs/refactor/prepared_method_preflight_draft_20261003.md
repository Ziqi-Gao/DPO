# Prepared-student OPD / GRPO preflight design draft

This draft records constraints for the next prepared-student training adapters.
The qualified student and method-specific execution envelope must be established
before implementation acceptance and deployment. The low-level factories in
`learning/training/prepared_method_settings.py` now construct an actual OPD AdamW
and constant scheduler, or an explicitly configured original `TrlGrpoBackend`.
The training entrypoints still need to consume these objects and the qualified
FP32 initial model.

The prospective preflight LR is5e-5, not a demonstrated optimum. OPD must retain
W4×microbatch4×accumulation4=64 sequences, current-policy lag0, response-prefix
top128 renormalized forward KL including EOS, completion128 and a constant
schedule. Retained mass remains readiness evidence, not a new training-token
filter. GRPO must retain official TRL0.22.2, W4×batch8×accumulation8=256 sequences
in32 groups of8, prompt2048 plus completion256, exact verifier reward,
DAPO/beta0/unscaled rewards, and linear zero-warmup scheduling with its original
120-step horizon. Both retain2M global nonpadding model-input tokens and120 safety
steps. OPD uses accepted dense-teacher scores on student responses; GRPO needs no
teacher weights. Neither method requires generated standard-proof targets.
OPD retains AdamW betas(.9,.95); official GRPO retains(.9,.999). Both actual
parameter groups use weight decay0 and epsilon1e-8.

Nine CPU behavior tests exercise real OPD gradient updates and optimizer/scheduler
state continuation, and the installed official TRL0.22.2 constructor followed by
actual optimizer creation. They confirm5e-5 reaches the optimizer and GRPO's next
two scheduler values use119/120 and118/120. The latter uses a local parameter
objective, without GRPO rollouts; its actual CPU world size is1 and generation
batch64. Production W4, the qualified model, method losses and distributed resume
still require their own execution evidence. The existing project test environment
uses Python3.12.14, separately from the GPU runtime's3.12.13.

The bounded preflight design is two reference optimizer windows, a complete
nonterminal checkpoint at window1, then one fresh same-world resume reproducing
window2: three physical updates. Count replay compute separately from each
scientific token cursor. GRPO reserves589824 tokens per window; three physical
windows have a1769472-token worst-case envelope. Preserve the120-step schedule
when stopping the diagnostic after two windows. Stop only at an optimizer
boundary; record the diagnostic stop separately from a fallback
`max_steps_safety_limit` string and never override an earlier budget stop.
Equal rewards can produce zero advantage: report missing learning signal rather
than changing rewards, resampling until success or counting only weight decay.
Transformers4.56.2 enables gradient checkpointing again when training starts.
The existing GRPO backend leaves `gradient_checkpointing_kwargs=None`, whose
model default is `use_reentrant=True`; this would override the prepared loader's
non-reentrant setting. The future engine adapter must explicitly review and bind
the actual checkpointing policy. Constructor/optimizer tests do not cover this
training-start path or establish FSDP safety.

The new input binding must verify the actual preparation selection and full18432
replay, same-candidate896 qualification, and the existing accepted teacher's
inventory and genuine producer origin through the existing APIs. Raw replay PASS does not
imply qualification PASS. Prior896 exposure remains disclosed and cannot become
a tuning population. Actual evidence bindings, training prompt identities/order/
shapes, runtime/launcher, CPU/RAM/GPU/time/storage resources, and resume comparison
tolerances remain unfrozen. Do not invent values to enable execution. Formal
training still requires complete newly bound G0 and existing scientific gates.

Implementation must explicitly assemble the prepared model, optimizer settings
and method engine. Next tests must cover the actual method losses, production
batch and model identities, complete checkpoints and same-world resume, using
the existing scientific admission rules.
