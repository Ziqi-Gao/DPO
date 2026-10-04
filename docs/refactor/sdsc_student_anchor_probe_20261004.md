# Original-order anchor diagnostic

The completed paired order experiment54648255/54648257 improved fact-permutation
proof accuracy but reduced IID and renamed-symbol accuracy. Its full results are
in `sdsc_student_order_probe_20261004.md`. Neither diagnostic checkpoint is a
candidate for the common initial model.

This prospective experiment tests one training policy: preserve the original
fact/rule display order in two of each base's eight views. Control independently
permutes facts and rules in all eight slots, reproducing the preceding treatment
algorithm on fresh data. Treatment skips that final permutation only for
`identity` and `entity_symbol_renaming`. The other six prompts, targets and masks
remain identical across arms. Both arms retain50% renamed samples,12.5%
paraphrases, complete signed pairs and unchanged canonical proof targets.
The policy simultaneously restores ordered coverage, reduces compound
transformation difficulty and reintroduces order cues; the result cannot
separately identify those mechanisms.

The protocol is `prereg/amendments/qwen3_student_anchor_probe_v1.json`, with core
`experiments/protocols/student_anchor_probe.py` and five new
`tools/sdsc_student_anchor_probe*.py` controls. All136 accepted historical science
files, six parent protocols and eleven external helpers remain frozen. There is
one flat plan/publication per arm, with separate permanent claims and no recovery
wrapper or automatic retry.

## Frozen design

Namespaces310000042/320000042 generate256 fit and128 development bases;
330000042/340000042 control fit/development transforms. These namespaces were
checked unused before population measurement. Each arm audits all2048 fit views
against the1536-token input and2M total-token limits, and consumes exactly the
first768 views in12 global64 updates. Both arms measure1,625,484 full-population
input tokens and606,988 executed
tokens; maximum training input1165 is within1536. The shared development panel
has48 chain/30 branch/50 DAG bases and maximum prefix2190 plus256 generation,
2446 within2454. All canonical targets verify. No filtering, seed replacement
or adaptive extension is allowed. Data isolation includes the original144000
family, teacher,
all previous preparation bases/views, and both preceding diagnostic fit arms
plus their shared development views. The original family is read only for
exclusion keys; formal896 generated responses and scores are not consulted.

Each arm starts from the same native1.7B artifact54548846 with fresh optimizer,
RNG and cursors. Preserve full-parameter FP32, W2 FULL_SHARD, microbatch4 per rank,
accumulation8, response/EOS sequence-mean loss and constant AdamW5e-5 with the
original betas/epsilon and zero weight decay, warmup or clipping. Negative
siblings retain their rotated slot order and all64-slot signed-pair semantics.
Both anchors occupy even declared slots: under the preserved rotation/rank
sharding, rank0 receives eight positive anchors and rank1 eight negative anchors
per window. Global anchors are8/8; each rank still has16 positive and16 negative
rows overall and four of every view. Both arms use this same allocation; global
loss/gradient averaging is unchanged. Do not claim label balance within each
rank's anchor subset.

Observe exactly steps4/6/7/8/12, with all128 bases and six unchanged development
views at every checkpoint:3840 raw responses per arm. Native-BF16 greedy decoding
retains256 completion tokens,2454 preparation context, no cache/autocast/truncation
and the original parser/verifier. Step4 performs actual W2 full-state restoration;
every checkpoint verifies all311 FP32 master tensors before casting, exact
native-BF16 logit parity and the2454-token finite forward on both ranks.

Three separate prespecified descriptive contrasts average treatment-minus-control
absolute proof-rate changes at steps6/7/8: renamed symbols, fact permutation and
rule permutation. They must not be combined into a score. Report all five steps,
six views, IID structures, answer/proof/format rates and same-example wins/losses.
For each view, decompose the changed IID-minus-view gap as deltaIID minus
deltaView. Lowering IID alone is not improvement. There is no significance,
noninferiority or acceptance threshold and no checkpoint selection.

## Execution and acceptance boundary

Each arm uses2H100/24CPU/384GiB/2h with accountnwu181 and the existing shared
partition/QoS, at most four concurrently allocatable GPUs. The preceding two
jobs completed in5305/5353s and published their required artifacts within the
same bound. Preserve300s early startup,12-thread policy,192GiB node-local free
space, original cgroup/headroom checks,600s publication reserve,128GiB large
artifact limit and224MiB bounded small fetch. Read-back hashes on persistent
storage and successful terminal accounting are required before completion.

Diagnostic completion is independent of correctness scores. Every model,
preparation, formal, G0, pilot, factorial and execution-class acceptance flag
stays false; selected_checkpoint is null. The frozen raw auditor reconstructs
the actual arm dataset and shared prompts and independently replays every
parser/verifier result. It does not rerun GPU arithmetic or download weights.
A later common initial still requires a fresh complete preparation, original
earliest eligible selection and unchanged formal896/2244 qualification.

## Implementation evidence

All296 unique focused CPU tests pass:68 core/population,114 transport/node/startup,
42 worker and72 raw-auditor tests. Root independently reran114 transport tests;
worker-to-auditor and node-to-isolated-worker tests use the actual new interfaces.
The finite read-only observer additionally passes35 tests. Ruff, formatting,
Python3.12 AST and diff checks pass. A new test initially assumed label balance
inside each rank's anchor subset; the corrected test asserts the inherited
allocation described above. No data, model or batching code was altered to make
that mistaken assertion pass.

Complete CPU data isolation SHA is
f475f141b0d052f79aefd3cd47ba0bb5e17ca8cc50144273657ffcb80814f888;
token/target evidence SHA is
d6516f4193429140ec74522046cbbc66fb4eb84512a299af176aa407960c7f8d.
These deterministic CPU reconstructions do not substitute for production
verification of the actual persisted original-family bytes on the GPU node.

Independent reviews under `.sdsc/diagnostics/student-anchor-probe-v1/`:

- core:7cfa707c1a2c4a44d7dfc43bb518dd2f9c0bf55ba3728bcb4b9ed3909399ad79;
- raw auditor:57ec52a8b715ef7565aa8b5d5e2490867d50e462a374d9415d8f4e11f0989716;
- worker/startup:9100bc8576a6b72e8790a34fd3adf415a72854cae8a693801ebf99bfaad5e46a;
- controller/node:bfd2cf206d33d150cbc4aca058d4cbdb1f65870d9dd1d5581588de677893f5d4;
- actual rank exposure/sequence scaling:af552b13c8713f882d3aa13273681f4a041a3373936b94b1d5213568176fef15.

Protocol core534eee4b464f8bd3e08d433319c71f35246561af643da9a6aab8e3a4f1c2ab64
was implemented in c383b6f9a25dc44d3416e4f8e7436526aa52374e. Exact-commit
cross-review passes, SHA
ace9888c912b354d59a87d74e651c51c0bbcd90c87db723c69fabbb8c62d5bb5.
Separate review-only acceptance80534bb59abefa7b7766def21a978c71de2265ed
produces accepted artifact
82e4a5e52fa291a3514531f6b0d2074c47ddfe636a810f6056798a7c6d75148d;
scientific content outside review is unchanged.

## Actual deployment and submission

Release20261004T161558Z-10ba8793444e-b3cc46db has780 files/11,266,364 bytes,
code SHA10ba8793444e1a5a696cf08bc114f37adc219b8f963142adcc203589331d2c20
and canonical manifest SHA
8f28b2b7a0d54dc4a4ee3f8228a55519ef754bf5bad2f2fa9cfccff55b38f368.
Genuine provenance SHA
c046d5f0c89756b32e8555a7211e01c10133dc9c2d43b37a80c2aa8803c95778
binds the accepted producer HEAD80534bb59abefa7b7766def21a978c71de2265ed.
Independent deployment/two-plan review passes, SHA
38b6950e7f684a175e329bc9ceb1855f7ceb414057f068825dfa07611095b0f8.
This supplementary report arrived after the first submission; the prior root
pre-submit review already checked both actual plans/source/runtime/resource
bindings, SHA1c6964dff87043d0923010715ef5df5f2c06cb645972bbc87a60bc3c60a848d0.
All19 remote runtime pins and the verified native parent54548846 were checked.
Node-local mounts/free space remain actual runtime gates.

| Arm | Job | Intent | Submission UTC |
| --- | --- | --- | --- |
| Control |54655732|481c8ca7356156e7fbbcbd22acb2190f|2026-10-04T16:28:20Z|
| Treatment |54655735|2e243530a9ff5c6576f209d16370192e|2026-10-04T16:29:28Z|

Plans and timestamped submission receipts are under
`.sdsc/student-anchor-probe/<intent>/`. Plan SHAs are
control daf7d82f93ed9ff83996a5d816b2ecc72adca595c780a86049dc65c4c17c0264
and treatment40422ad3db68e86c697ce2413ba14f8411784673a4b98c6295b2deb1d5f55b3f.
Both allocations are2H100/24CPU/384GiB/02:00:00 on accountnwu181,
partitionnairr-gpu-shared/QoSnairr-gpu-shared-normal. The fresh second-arm
dry-run counted the first arm's2GPUs plus2newGPUs against the4GPU limit.
These are two distinct acknowledged submissions; never submit either again.
Required outputs use persistent
`/expanse/lustre/projects/nwu181/zgao12/OPD/student-anchor-probe/<intent>`.

The reviewed foreground observer started16:35UTC, session8280, state
`.sdsc/diagnostics/student-anchor-probe-v1/watch-pair-54655732-54655735/`,
with a22:00UTC deadline and one-minute status interval. Both arms were observed
RUNNING at16:35:14Z/16:35:15Z. It stops on unknown state, failure, SSH loss,
changed controls or deadline, and never submits/retries/cancels/fetches. Root
continues result collection and raw review separately. Reviewed observer,
one-shot bounded live reader and comparator pass35/23/22 fixtures respectively.
No GPU completion or capability outcome is established by these submissions.
Existing unrelated tracked and legacy untracked changes were preserved.
