# Implementation plan

1. Build and test the deterministic Phase-0 ProofGraph vertical slice.
2. Route all six factorial cells through one trainer using composable state sources and supervisors.
3. Add canonical SFT and an official-TRL GRPO adapter with independently testable rewards.
4. Add immutable trajectory stores, checkpoint/RNG restoration, run provenance, and local forks.
5. Provide circuit scoring/patching interfaces, tiny-model interventions, and production MIB/TL adapters.
6. Resolve every configuration, exercise two-step CPU smoke runs, and gate production scale behind
   explicit confirmation plus a readiness report.

The CPU smoke backend is explicitly named `tiny_qwen`; it is not an algorithmic substitute for a
production backend. GPU validation and scientific inference remain outside repository construction.

