# Name-cue diagnostic startup failure and runtime recovery

Job54681802 failed before any model was staged or inference performed. Its
300-second startup child read driver580.178.04 and began `import_torch`, then
continued importing NumPy and Torch until the deadline. The child was terminated
and reaped at299.381seconds; node execution took320.615seconds and accounting
records324seconds, job/batchFAILED1:0 and externCOMPLETED0:0. Fresh20:42:37Z
accounting has an empty own-job queue. This is an infrastructure failure and
supplies no name-cue result or model-quality evidence.

The original source, plan, submission, consumed scientific claim and negative
publication remain immutable. Plan614f6f48bd498983ce820f9c86856b2494e349b8d9dc625b6f5fb75d431fe858
belongs to intent43b6665d68c9b525320fc8e60ed16c2b. Publication
`a6750c0cd9afcbc9c42a0768f5f3ce32f37e2e0736ec1aa0ef038bbedea2074e`
was hash-verified and fetched into `.sdsc/fetched/54681802/fetch-udmbb982/`.
The bounded fetch is234,316bytes. Independent replay of original publication,
trace and memory validators is recorded in
`.sdsc/diagnostics/student-name-cue-probe-v1/startup54681802-independent-diagnosis.json`,
SHA7690707aacaa574e119f48d6ad376a71280b2ba0a7fc7336c26e5f073d48052e.

The complete129,547-byte startup log contains1,088 completed import-time records
with289.738seconds of summed self time. All nine30-second stack samples remain
in advancing imports: six filesystem-stat paths, two module reads and one NumPy
extension load. The final completed record is `torch.compiler`; there is no
`import_torch` completion or explicit CUDA API. These observations support slow
shared-runtime loading, with metadata/file I/O prominent in sampled stacks.
They do not apportion all wall time to I/O or exclude cold-cache, native-library
loading and CPU-scheduling contributions. No cluster-wide outage is established.
Twelve own-job memory samples pass with peak773,857,280bytes and noOOM/failcnt.

The early thread policy already setOMP/MKL/OPENBLAS12. Earlier successful LR
jobs54673886/54673887 imported Torch from the same runtime path in105.081/26.397s;
those are different allocations and cache states, not paired causal evidence.
Historical import diagnostic54643629 already showed53.419/1.559/1.571s for its
first/subsequent/subsequent imports. Increasing the timeout again would leave
this demonstrated shared-storage dependency in place.

## Proposed execution repair

Create an exact, bounded snapshot of the entire existing Conda prefix outside a
GPU allocation. Preserve regular-file bytes and safe internal relative symlinks;
do not install packages, rewrite embedded prefixes, or mutate the original runtime.
Copy the archive sequentially to owned, verified node-local storage, extract and
validate every entry, then run the native relocated interpreter. Explicitly record
its actual executable and prefix and validate module/native-library origins so
no dependency silently falls back to the original Lustre prefix. Host system
libraries remain those of the native compute environment.

This path requires actual relocation proof, not an assumption that Conda is
fully relocatable. It must preserve the300-second early-probe limit and include
staging inside the4800-second worker budget, leaving600seconds for publication.
Both fixed CP32 inputs, all168 named scientific files, original generation,
paired factors and raw auditor remain unchanged. A separate reviewed execution
contract must own one recovery claim tied to the exact failed no-inference job;
changing a run ID alone does not permit another attempt.

Read-only login-node discovery found `bwrap` but private-namespace creation failed
with `No space left on device`. Singularity rejected `/` as a sandbox. Neither
restriction was changed or bypassed. The documented installed PyTorch SIF could
run the pinned interpreter with a read-only runtime bind, but it usesglibc2.39;
that introduces a userspace change, so it is not the selected repair. Native
byte-preserving relocation is being implemented and has not yet been accepted,
used on a GPU node, or shown to cure this failure.

The first off-GPU pack request stopped before creating its snapshot or claim:
the project filesystem inherited SGID on the new private directory, producing
mode02700. Read-only reconciliation confirmed the original snapshot directory,
four request/event sidecars, archive and manifest were all absent. The source
Conda prefix and its sampled subdirectories have mode02755. The additive utility
now preserves SGID on directories only; SUID/sticky directories and special-mode
files remain rejected. It does not chmod existing storage or alter the source
runtime. Author and independent62-case tests pass, including real SGID directory
roundtrip and negative mode cases. This compatibility repair is preparation
evidence, not proof of native relocation or a successful GPU run.

The additive recovery contract, controller, node wrapper, runtime snapshotter and
independent execution auditor now have153 passing combined CPU cases. A separate
real Git fixture also restores the complete v2 history and validates distinct
implementation/acceptance ancestry. Original33 frozen execution dependencies and
all168 named scientific files remain exact. Independent review found and repaired
Git replacement-object handling and a missing publication-to-raw-node job/plan
cross-check. The standalone execution auditor requires an external receipt hash;
the original scientific raw auditor remains a separate required check.

The second off-GPU pack request was
`7ef839cabc924bd89851ee07a0f724b2`, from preparatory release
`20261005T211101Z-50c8d43ae489-546fd0a1`. It failed at22:16UTC after3600.033seconds,
with actual foreground exec1327 returning1 and a matching remote timeout record.
Initial inventory consumed about51–52minutes; subsequent serial packing left
265,339,753bytes without a manifest or completion. Preserve this partial output.
The fixed request/start/error bytes, complete local logs and a later complete
own-process scan establish termination. Two initially conservative unknown
reconciliations are retained; the reviewed local successor predicate permits an
earlier incomplete scan only when it has no contradictory process evidence and
the final scan is complete. It issues no retry or artifact-acceptance permission.

The additive I/O repair overlaps at most four metadata lookups and caches32
parent-directory descriptors. All source stamps, file bytes/modes/links, sorted
archive order, final inventory and full readback remain required. CPU fixtures
cover actual overlap, bounded descriptors, source mutation, partial submission,
and a task whose submission starts work but raises before returning its Future.
The whole worker pool is drained before borrowed directory descriptors close;
an external child-process supervisor remains necessary for blocked I/O.

The proposed single CPU successor has10800seconds total,10760seconds work,
10770seconds for cleanup attempts and30seconds publication reserve; local SSH
acknowledgement is bounded at10820seconds. Three hours provides engineering
headroom, not a measured completion guarantee. Original GPU startup300seconds,
worker4800seconds and publication600seconds remain unchanged. A permanent claim
keyed to the failed parent request blocks alternate UUID/code/release retries.
Missing acknowledgement or unreaped work stays unknown. Activation requires actual failure and fresh deployment identities to be reviewed.

OffGPU preparation cannot submit Slurm or certify native execution. Completion requires the
entire source inventory and archive readback, followed by a separate CPU rehearsal
that imports the original packages under the actual relocated Python and checks
all observed module/native-library bytes and origins. That rehearsal explicitly
does not prove GPU-node locality, CUDA, model inference or scientific success.


## Actual snapshot and CPU relocation evidence, 2026-10-06

Snapshot2a3797d216384301877dfaf6a398431a completed in4631.884seconds with
49,474 files,5,034 directories and1,154 internal symlinks. The7,825,971,697-byte
archive SHA is7151cc2839d518474b3ab968064a14773a43728c1671aea462b53234da1db83d;
manifest SHA is2cc4fa21afe2dc180379456d8dd2965e6a694ffb50d52cf0a11802bac0e910cd.
The first CPU restoration onto Lustre advanced too slowly; root stopped only
its owned child, retained the partial tree, and verified actual exit/reaping.
A separately reviewed login-local restoration preserved the same utility and
checks. It completed restoration and imports within148.119seconds, but the
native-origin collector failed before producing its file/library inventory.

The raw error reports `FileNotFoundError` for `_classes.py`. Torch2.8 defines
`torch.classes` and `torch.ops` as singleton ModuleType subclasses with inherited
placeholder `__file__` values `_classes.py` and `_ops.py`. They are dynamic
namespaces, with no actual module spec; their implementations are the real,
manifest-hashed `torch/_classes.py` and `torch/_ops.py`. The collector incorrectly
interpreted a placeholder as a current-directory file. All19 runtime versions
were reported and all restored file bytes had already passed validation. This
does not validate native origins: collection stopped before mapped libraries
and file hashes were recorded.

Actual `runtime-rehearsal-result-v3.json`, its genuine tool terminal, three
hash-verified fetched proofs and independent diagnosis live under
`.sdsc/diagnostics/student-name-cue-probe-v1/`. The execution-only fix now
recognizes only the exact Torch singleton aliases, retains and verifies their
backing source files, and rejects ordinary relative origins. The affected node
suite passes74 author and74 independent tests;8 unchanged CPU-scope cases pass.
A fresh real native rehearsal is still required before acceptance. The original failed GPU job54681802, both CPU
failure receipts, all scientific inputs and the300/4800/600 GPU limits remain
unchanged. No new GPU job has been submitted.
