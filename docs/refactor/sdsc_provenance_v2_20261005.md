# Versioned full-bundle provenance successor

The accepted LR diagnostic uses127 unpublished commits under v1's fixed128
limit. Its genuine bundle is3,576,211 bytes, below the unchanged64MiB bound.
A further scientific implementation/acceptance pair needs a new transport
version. Moving the public ref, dropping history, editing the active verifier or
silently increasing its limit would invalidate existing provenance.

`tools/sdsc_provenance_v2.py` is an additive full-bundle producer/verifier with
schema `quest-sdsc-git-provenance-v2` and a fixed256-commit limit. Its local
namespace is `.sdsc/provenance-v2/`. The separate
`tools/sdsc_provenance_upload_v2.py` pins the specifically named v2 verifier in
the deployed release and uploads only to
`/home/zgao12/quest-runs/OPD/provenance-v2/<manifest-sha256>/`.

The successor preserves genuine HEAD/public tracking ref, full linear history,
the final implementation/acceptance pair, inspection of every historical tree,
accepted ancestors, exact source/wrapper bindings, external manifest hashes,
zero-prerequisite bundles, isolated restoration and read-only publication.
File, bundle and archive byte bounds remain4/64/72MiB. No runtime override,
submission, retry, credential copying or scientific acceptance is added.

Current and historical consumers remain on v1, including original native-parent
verification. A future task must explicitly include v2 in its deployed controls,
executed-science inventory and protocol; its controller must call the v2 verifier
and admit the new namespace. The existing node-to-controller restoration pattern
can be reused. Merely adding these utilities does not activate them for any job.

Author tests cover46 cases, including actual129/256-commit export and restore,
257 rejection before export/import, unchanged v1 rejection above128, forbidden
files added then deleted in history, binding corruption, and execution of the
versioned uploader receiver against a real129-commit fixture. These are CPU and
local transport checks; no production artifact has been exported or uploaded
with v2 at this stage. Independent review also reran all46 cases successfully
and found no blocker; evidence is
`.sdsc/diagnostics/student-focus-lr-probe-v1/provenance-v2-independent-review.json`
(SHAcd0238fcab9d7938f457e3943f526c8f09665d2ddce86c32025ffc9aa2bea606).
A new consumer still requires its own integration review. No current training
outcome follows from these checks.

Incremental bundles are deferred: v1 deliberately rejects prerequisites, and a
delta format would need independently trusted base artifacts, bounded dependency
chains and different restore/recovery contracts. Present evidence supports the
smaller versioned full-bundle change.
