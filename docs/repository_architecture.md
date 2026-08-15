# Repository architecture

The package has four dependency directions. `core` contains immutable records, configuration,
hashing, RNG, and provenance. `tasks`, `data`, and `rewards` depend only on `core`. `rollout`,
`teacher`, `supervision`, and `training` compose those domain interfaces. `circuits` and `analysis`
consume saved run artifacts and never reach into trainer internals.

The six controlled cells are data, not classes: configuration selects one `StateSource` and one
`Supervisor`, while `FactorialTrainer` owns the optimizer/update/checkpoint loop. Canonical SFT
reuses response masking and checkpointing. Canonical GRPO is kept behind a TRL adapter because its
policy-gradient semantics must not drift into an in-house approximation.

Artifacts are content-addressed where practical. Dataset and bank manifests bind seeds,
configuration, hashes, and producer versions. Run directories bind resolved configuration, git
state, environment, metrics, and checkpoints. Circuit artifacts bind checkpoint, pair split,
compatibility report, backend revision, full scores, and intervention settings.

