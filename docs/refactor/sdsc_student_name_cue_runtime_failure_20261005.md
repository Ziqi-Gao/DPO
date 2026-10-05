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

The active off-GPU pack request is
`7ef839cabc924bd89851ee07a0f724b2`, from preparatory release
`20261005T211101Z-50c8d43ae489-546fd0a1`. It is finite and foreground, owns a fresh
snapshot namespace, and rechecks the old negative reconciliation before claiming.
It cannot submit Slurm or certify native execution. Completion requires the
entire source inventory and archive readback, followed by a separate CPU rehearsal
that imports the original packages under the actual relocated Python and checks
all observed module/native-library bytes and origins. That rehearsal explicitly
does not prove GPU-node locality, CUDA, model inference or scientific success.
