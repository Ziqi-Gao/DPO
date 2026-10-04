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
and proposed artifact3e14900a3f0da12bd505ea3d4d2ec71a207785cb991696205eaeecb291753b9f
still require an implementation commit followed by exact-commit review and a
distinct review-only acceptance. No GPU outcome or submission is recorded here.
