# Experimental identification strategy

The controlled experiment crosses two state sources with three supervision signals. Offline cells
learn on one frozen bank sampled from a calibrated behavior policy. Online cells draw from the
current policy with zero allowed policy lag. Hard teacher, soft teacher, and verified replay plug
into the same trainer; optimizer, schedule, response masks, prompt order, maximum completion length,
checkpoint format, and full-parameter update behavior do not change across cells.

The fixed bank is shared across all offline cells so `offline_soft - offline_hard` changes the
information in the signal without changing visited prefixes. Giving each cell its own bank would
confound signal with sampling noise and policy behavior. The bank deliberately includes verifier
successes and failures; the calibration target is 20–60% success.

Teacher demonstrations are separate from this grid. Canonical SFT changes both the states and the
targets: it follows verified teacher trajectories rather than behavior/student trajectories.
Likewise, verified replay and canonical GRPO are both necessary. Replay isolates reward information
while retaining a behavior-cloning update on successful sampled sequences. GRPO adds the standard
group-relative policy-gradient update on current-policy generations.

## Planned contrasts

- `online_hard - offline_hard`, `online_soft_opd - offline_soft`, and
  `online_verified_replay - offline_verified_replay` identify the effect of changing state source at
  a fixed learning signal.
- `online_soft_opd - online_hard` compares distributional versus top-1 teacher information on
  current states.
- `online_soft_opd - online_verified_replay` compares dense teacher information with sparse exact
  outcome information under the same source class.
- `online_verified_replay - canonical_grpo` connects reward-gated imitation to the policy-gradient
  rule, but natural trajectories can diverge; the shared-state local fork is the sharper update-rule
  diagnostic.

Circuit locking is evaluated by raw progress and at matched validation accuracy, output KL from the
initial model, parameter-update norm, generated-token budget, and supervised-token budget. A raw
step comparison alone is not an identified causal comparison.

This design cannot establish backbone universality, real-world reasoning validity, or a complete
free-form chain-of-thought circuit. Circuit estimators are measurements with error; exact patching,
random controls, bootstrap stability, and cross-mask transfer constrain interpretation but do not
turn a synthetic-task result into a biological or universal mechanism claim.

