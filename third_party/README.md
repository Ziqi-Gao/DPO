# Third-party research code

No external repository is vendored.

MIB/EAP-IG is integrated through an adapter expecting an external checkout of
[`hannamw/MIB-circuit-track`](https://github.com/hannamw/MIB-circuit-track) pinned to commit
`b759df34433c9e31043ba9e02908ce0bf20e894f` (verified upstream merge commit, 2025-06-30).
Set `MIB_REPOSITORY` to that checkout. The adapter verifies `git rev-parse HEAD` and refuses a
different revision. Upstream owns EAP-IG computation; this project owns serialization,
counterfactual preparation, provenance, and evaluation.

TransformerLens is consumed as the exact package pin in `pyproject.toml`. Its GQA representation
keeps K/V heads distinct from query heads; production compatibility tests must validate the mapping
and logit parity instead of assuming one K/V head per query head.

