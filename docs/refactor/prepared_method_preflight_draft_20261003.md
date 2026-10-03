# Prepared-student OPD / GRPO preflight design draft

This draft records constraints for the next prepared-student training adapters.
The qualified student and method-specific execution envelope must be established
before implementation acceptance and deployment. The learning-rate and loading
changes below still need to reach the actual optimizer and training entrypoints.

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

The new input binding must verify the actual preparation selection and full18432
replay, same-candidate896 qualification, and the existing accepted teacher's
inventory and genuine producer origin through the existing APIs. Raw replay PASS does not
imply qualification PASS. Prior896 exposure remains disclosed and cannot become
a tuning population. Actual evidence bindings, training prompt identities/order/
shapes, runtime/launcher, CPU/RAM/GPU/time/storage resources, and resume comparison
tolerances remain unfrozen. Do not invent values to enable execution. Formal
training still requires complete newly bound G0 and existing scientific gates.

Implementation must explicitly assemble the prepared model, optimizer settings
and method engine. Focused tests must show that actual AdamW and official TRL
receive the reviewed learning rate, schedule, batch and model identities, and
exercise real constructor, update and resume behavior.
