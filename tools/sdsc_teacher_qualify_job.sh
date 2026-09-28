#!/bin/bash
# Fixed four-H100 qualification of one independently reviewed adapted teacher.
set -euo pipefail
if [[ $# != 9 || $4 != /* || ! -x $4 ]]; then
    printf '%s\n' 'usage: sdsc_teacher_qualify_job.sh RELEASE SUBMISSION RESULTS ABSOLUTE_PYTHON RUN_ID CODE_SHA256 HF_HOME PROVENANCE_DIR PROVENANCE_SHA256' >&2
    exit 2
fi
exec "$4" -I -B -u - "$@" <<'PY'
import contextlib, hashlib, importlib.util, json, os, pathlib, pwd, re, selectors, shutil, signal, stat, subprocess, sys, tempfile, time

release, submission, results, python, run_id, code_hash, original_hf_home, provenance_dir, provenance_hash = sys.argv[1:]
qualify_task = "qwen3-v2-teacher-qualify"
qualify_contract = None
prerequisite = None
claims = None
job_id = os.environ.get("SLURM_JOB_ID", "")
project = pathlib.Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
models = (
    ("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("models--Qwen--Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
identity = dict(task=qualify_task, job_id=job_id, run_id=run_id,
                code_sha256=code_hash, hf_home=original_hf_home, provenance_dir=provenance_dir,
                provenance_manifest_sha256=provenance_hash)
process = None
work = None
log = None
result_root = None
exit_code = 1
wrapper_started = time.monotonic()
wrapper_phase = "preflight"
phase_timings = {}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def safe(path):
    path = pathlib.Path(path)
    if (not path.is_absolute() or ".." in path.parts
            or any(ord(char) < 32 for char in str(path))
            or any(part.is_symlink() for part in (path, *path.parents))):
        raise ValueError("unsafe path: %s" % path)
    return path

def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def atomic_json(path, value):
    safe(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def mount(path):
    return json.loads(subprocess.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True, capture_output=True, text=True, timeout=10).stdout)["filesystems"][0]

@contextlib.contextmanager
def timed_phase(name):
    global wrapper_phase
    wrapper_phase = name
    started = time.monotonic()
    completed = False
    print(json.dumps(dict(identity, phase=name, event="started", timestamp_unix=time.time(),
                          wrapper_elapsed_seconds=started - wrapper_started)), flush=True)
    try:
        yield
        completed = True
    finally:
        phase_timings[name] = time.monotonic() - started
        print(json.dumps(dict(identity, phase=name, event="finished", timestamp_unix=time.time(),
                              elapsed_seconds=phase_timings[name], completed=completed)), flush=True)

def forward_output(pipe, stream):
    # An unbuffered binary pipe avoids losing TextIOWrapper/readline buffers
    # when a signal interrupts the read and the failure handler takes over.
    while True:
        block = os.read(pipe.fileno(), 64 * 1024)
        if not block:
            break
        stream.write(block)
        stream.flush()
        sys.stdout.buffer.write(block)
        sys.stdout.buffer.flush()

def stop_child(grace_seconds=20, kill_seconds=10, drain_seconds=2, tail_limit=4 * 1024 * 1024):
    if process is None:
        return dict(child_started=False)
    started = time.monotonic()
    tail = bytearray()
    received = 0
    killed = False
    eof = process.stdout is None or process.stdout.closed
    exited_at = None
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Drain during the TERM grace period, not after wait(): a handler emitting
    # more than a pipe buffer would otherwise block before its final evidence.
    with selectors.DefaultSelector() as selector:
        if not eof:
            descriptor = process.stdout.fileno()
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        deadline = started + grace_seconds
        while True:
            now = time.monotonic()
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = now
                if eof or now >= exited_at + drain_seconds:
                    break
            elif now >= deadline:
                if killed:
                    break
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                killed = True
                deadline = now + kill_seconds
            timeout = min(0.1, max(0, (exited_at + drain_seconds if exited_at is not None else deadline) - now))
            for key, _events in selector.select(timeout):
                try:
                    block = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(key.fd)
                    eof = True
                    continue
                received += len(block)
                tail.extend(block)
                if len(tail) > tail_limit:
                    del tail[:-tail_limit]
    if not eof:
        # A descendant may retain the pipe after the direct worker exits.
        # The session belongs to this attempt; never wait indefinitely on it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        killed = True
    if process.stdout is not None:
        process.stdout.close()
    dropped = received - len(tail)
    if tail and log is not None:
        # The original with-log has already unwound. Reopen a verified regular
        # file rather than attempting to write through that closed stream.
        path = safe(log)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "ab") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("worker log is not a regular file")
            if dropped:
                stream.write(("\n[shutdown log truncated: %s earlier bytes omitted]\n" % dropped).encode())
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(child_started=True, exit_code=process.poll(), sigkill_sent=killed,
                pipe_drained=eof, received_bytes=received, retained_bytes=len(tail),
                omitted_bytes=dropped, elapsed_seconds=time.monotonic() - started)

def interrupted(signum, _frame):
    raise RuntimeError("teacher qualification interrupted by signal %s" % signum)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def failure_progress(output):
    diagnostics = dict(progress_available=False, not_a_completion_report=True, not_a_quality_estimate=True)
    if output is None:
        return diagnostics
    try:
        path = safe(output / "progress.json")
        with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("worker progress is not a regular file")
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("worker progress exceeds diagnostic size bound")
        progress = json.loads(raw)
        if (not isinstance(progress, dict)
                or any(progress.get(key) != identity[key] for key in ("task", "job_id", "run_id", "code_sha256"))
                or progress.get("artifact_kind") != "teacher_qualification_progress"
                or progress.get("not_a_completion_report") is not True
                or any(progress.get(key) is not False for key in ("passed", "accepted_science", "full_teacher_ready", "g0_passed"))):
            raise ValueError("worker progress identity or nonaccepting scope differs")
        diagnostics.update(progress=progress, progress_available=True)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as error:
        diagnostics["progress_error"] = str(error)[:200]
    return diagnostics

def stage_source(root, destination):
    manifest = json.loads(safe(root / "manifest.json").read_text())
    files = manifest["files"]
    if (hashlib.sha256(canonical(files)).hexdigest() != code_hash
            or manifest["code_sha256"] != code_hash or manifest["run_id"] != run_id):
        raise ValueError("release identity or content hash mismatch")
    seen = set()
    for entry in files:
        name = pathlib.PurePosixPath(entry["path"])
        if (name.is_absolute() or not name.parts or ".." in name.parts
                or str(name) != entry["path"] or str(name) in seen):
            raise ValueError("unsafe or duplicate manifest path")
        seen.add(str(name))
        path = safe(root / "source" / name)
        info = path.stat()
        if (not stat.S_ISREG(info.st_mode) or entry["mode"] not in (420, 493)
                or stat.S_IMODE(info.st_mode) != entry["mode"] or info.st_size > 4 * 1024 * 1024):
            raise ValueError("invalid source file metadata")
        data = path.read_bytes()
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("source differs from manifest: %s" % name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(entry["mode"])
        if file_hash(target) != entry["sha256"]:
            raise OSError("staged source failed read-back")
    if not {"tools/sdsc_teacher_qualify_job.sh", "tools/sdsc_teacher_qualify.py",
            "tools/sdsc_teacher_qualify_contract.py", "tools/sdsc_provenance.py"} <= seen:
        raise ValueError("qualification transport/provenance files are missing")
    # This snapshot is transport provenance. Science is restored from genuine
    # accepted Git below; every named transport byte must also match that review.
    for directory, directories, filenames in os.walk(destination, topdown=False):
        for name in filenames:
            path = pathlib.Path(directory) / name
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        pathlib.Path(directory).chmod(0o555)

def stage_models(cache, destination):
    entries = []
    for model, revision in models:
        snapshot = safe(cache / "hub" / model / "snapshots" / revision)
        if not snapshot.is_dir():
            raise ValueError("pinned model snapshot is missing")
        for directory, directories, filenames in os.walk(snapshot, followlinks=False):
            if any((pathlib.Path(directory) / name).is_symlink() for name in directories):
                raise ValueError("symlink directory in model snapshot")
            for name in filenames:
                if name not in {"config.json", "generation_config.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                        "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja",
                        "model.safetensors", "model.safetensors.index.json"} and not re.fullmatch(
                            r"model-[0-9]+-of-[0-9]+\.safetensors", name):
                    continue
                original = pathlib.Path(directory) / name
                resolved = original.resolve(strict=True)
                if cache not in resolved.parents or not stat.S_ISREG(resolved.stat().st_mode):
                    raise ValueError("model snapshot link escapes its cache or is not a regular file")
                entries.append((original, resolved, destination / original.relative_to(cache)))
    total = sum(source.stat().st_size for _, source, _ in entries)
    if not entries or shutil.disk_usage(destination.parent).free < total + 12 * 1024 ** 3:
        raise ValueError("model cache is empty or node-local free space lacks adaptation checkpoint headroom")
    print(json.dumps(dict(phase="stage_models", files=len(entries), bytes=total)), flush=True)
    for original, source, target in entries:
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                digest.update(block)
                outgoing.write(block)
        if file_hash(target) != digest.hexdigest():
            raise OSError("staged model failed read-back: %s" % original.name)
    return dict(files=len(entries), bytes=total)

def validate_report(report, status):
    if status != 0:
        raise ValueError("qualification worker did not exit successfully")
    for key in ("task", "job_id", "run_id", "code_sha256", "provenance_manifest_sha256", "science_git_head", "bundle_sha256"):
        if report.get(key) != identity[key]:
            raise ValueError("qualification worker identity differs: " + key)
    qualify_contract.validate_report(report, prerequisite)
    if any(report.get(key) != intent[key] for key in ("prerequisites_sha256", "claims_sha256")):
        raise ValueError("qualification worker proof/claim hashes differ")
    qualify_contract.validate_acceptance_file(output, report, claims)
    qualify_contract.validate_inventory(output, report)

def publish(output, log, report, passed):
    # Only this fresh attempt writes this directory. Every copied byte is read back.
    # Diagnostic report is durable before copying weights; receipt is written last.
    if process is not None and process.poll() is None:
        raise ValueError("cannot publish while the adaptation worker is writing")
    expected = {"artifacts/" + name: row for name, row in qualify_contract.validate_inventory(output, report).items()} if passed else {}
    atomic_json(result_root / "teacher-qualify.json", report)
    records = []
    sources = [("worker.log", log)] if log and log.is_file() else []
    if output and output.is_dir():
        for directory, directories, filenames in os.walk(output, followlinks=False):
            if any((pathlib.Path(directory) / name).is_symlink() for name in directories):
                raise ValueError("symlink output directory")
            for name in filenames:
                source = safe(pathlib.Path(directory) / name)
                relative = str(source.relative_to(output))
                # Preserve the original worker report too, including failed phase evidence.
                sources.append(("artifacts/" + relative, source))
    if sum(source.stat().st_size for _, source in sources) > 8 * 1024 ** 3:
        raise ValueError("adaptation output exceeds the 8 GiB publication budget")
    if passed and sum(source.stat().st_size for name, source in sources
                      if name not in expected and name != "artifacts/qualification-manifest.json"
                      and name != "artifacts/teacher-qualify.json") > qualify_contract.MAX_LOG_BYTES:
        raise ValueError("qualification diagnostics/logs exceed the 64 MiB limit")
    for name, source in sources:
        if (not stat.S_ISREG(source.stat().st_mode)
                or source.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}):
            raise ValueError("qualification output must be regular evidence, never duplicate model/trainer weights")
        target = safe(result_root / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe(target.with_name("." + target.name + ".tmp"))
        digest = hashlib.sha256()
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming, temporary.open("xb") as outgoing:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("output is not a regular file")
            total = 0
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                total += len(block)
                if total > before.st_size:
                    raise ValueError("output grew during publication")
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
        if total != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("output changed during publication")
        if name in expected and (digest.hexdigest(), total) != (expected[name]["sha256"], expected[name]["size"]):
            raise ValueError("teacher checkpoint content differs from the worker manifest")
        os.replace(temporary, target)
        if file_hash(target) != digest.hexdigest():
            raise OSError("persistent result failed read-back")
        records.append(dict(path=name, size=target.stat().st_size, sha256=digest.hexdigest()))
    records.append(dict(path="teacher-qualify.json", size=(result_root / "teacher-qualify.json").stat().st_size,
                        sha256=file_hash(result_root / "teacher-qualify.json")))
    if json.loads((result_root / "teacher-qualify.json").read_text()) != report:
        raise OSError("persistent teacher fit report failed read-back")
    receipt = dict(identity, passed=passed, persisted=True, persistent_read_back_verified=True, files=records)
    atomic_json(result_root / "receipt.json", receipt)
    if json.loads((result_root / "receipt.json").read_text()) != receipt:
        raise OSError("publication receipt failed read-back")
    print(json.dumps(dict(identity, phase="publication", passed=passed, persisted=True,
                          result_dir=str(result_root), teacher_qualification_sha256=records[-1]["sha256"])), flush=True)

try:
    if (not re.fullmatch(r"[1-9][0-9]*", job_id)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id)
            or not re.fullmatch(r"[a-f0-9]{64}", code_hash)):
        raise ValueError("real Slurm allocation and valid deployment identity required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "24"
            or os.environ.get("SLURM_MEM_PER_NODE") != "196608"
            or len(visible) != 4 or len(set(visible)) != 4
            or any(value in {"", "-1"} or value.strip() != value for value in visible)):
        raise ValueError("teacher qualification requires exactly four distinct visible GPUs, 24 CPUs and 192 GiB")
    result_root = safe(results)
    cache = safe(original_hf_home)
    for path in (result_root, cache):
        if project not in (path, *path.parents) or not path.is_dir():
            raise ValueError("input/output must exist within confirmed persistent project storage")
    persistent_mount = mount(result_root)
    input_mount = mount(cache)
    if any(value["fstype"] not in {"lustre", "nfs", "nfs4", "ceph"}
           for value in (persistent_mount, input_mount)):
        raise ValueError("input/output mount is not verified shared persistent storage")
    probe = result_root / ".node-write-read-check"
    with probe.open("xb") as stream:
        stream.write(code_hash.encode())
        stream.flush()
        os.fsync(stream.fileno())
    if probe.read_bytes() != code_hash.encode():
        raise OSError("GPU-node persistent write/read failed")
    probe.unlink()
    tmp = os.environ.get("TMPDIR") or "/scratch/%s/job_%s" % (pwd.getpwuid(os.getuid()).pw_name, job_id)
    scratch = safe(tmp)
    if not scratch.is_dir() or scratch.stat().st_uid != os.getuid():
        raise ValueError("owned node-local scratch is unavailable")
    scratch_mount = mount(scratch)
    if scratch_mount["fstype"] not in {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs"}:
        raise ValueError("scratch is not verified node-local storage")
    work = pathlib.Path(tempfile.mkdtemp(prefix="opd-teacher-qualify-%s-" % job_id, dir=scratch))
    source = work / "source"
    source.mkdir()
    with timed_phase("source_staging"):
        stage_source(safe(release), source)
    specification = importlib.util.spec_from_file_location("teacher_qualify_contract", source / "tools/sdsc_teacher_qualify_contract.py")
    qualify_contract = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(qualify_contract)
    submission_root = safe(submission)
    intent = json.loads(safe(submission_root / "intent.json").read_bytes())
    if any(intent.get(key) != identity[key] for key in ("task", "run_id", "code_sha256", "provenance_dir", "provenance_manifest_sha256")):
        raise ValueError("qualification submission identity differs")
    if intent.get("prerequisites_path") != str(submission_root / "prerequisites.json") or intent.get("claims_path") != str(submission_root / "claims.json"):
        raise ValueError("fixed qualification proof/claim paths differ")
    prerequisite_raw = qualify_contract.small_bytes(submission_root / "prerequisites.json")
    claims_raw = qualify_contract.small_bytes(submission_root / "claims.json")
    if (hashlib.sha256(prerequisite_raw).hexdigest() != intent.get("prerequisites_sha256")
            or hashlib.sha256(claims_raw).hexdigest() != intent.get("claims_sha256")):
        raise ValueError("qualification proof/claims changed")
    prerequisite, claims = json.loads(prerequisite_raw), json.loads(claims_raw)
    if (prerequisite.get("schema") != "quest-sdsc-teacher-qualification-prerequisites-v1"
            or prerequisite.get("task") != qualify_task
            or prerequisite.get("target") != {key: intent[key] for key in ("run_id", "code_sha256", "intent_id")}
            or prerequisite.get("teacher_job_id") != intent.get("teacher_job_id")):
        raise ValueError("qualification prerequisite identity differs")
    binding = qualify_contract.verify_claims(claims)
    if any(binding.get(key) != intent.get(key) for key in ("run_id", "code_sha256", "intent_id", "teacher_job_id", "adapted_teacher_sha256", "protocol_sha256", "prerequisites_sha256")):
        raise ValueError("qualification reservations differ from submission")
    atomic_json(work / "prerequisites.json", prerequisite)
    atomic_json(work / "claims.json", claims)
    verifier_spec = importlib.util.spec_from_file_location("qualify_provenance", source / "tools/sdsc_provenance.py")
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    science = work / "science"
    with timed_phase("genuine_science_restore"):
        restored = verifier.verify(safe(provenance_dir), provenance_hash, code_hash, science)
        protocol = qualify_contract.resolved_protocol(science, intent["science_git_head"])
        if protocol != prerequisite["protocol"]:
            raise ValueError("accepted protocol differs from submission")
        for name, expected_hash in protocol["science_file_sha256"].items():
            if file_hash(safe(source / name)) != expected_hash:
                raise ValueError("transport snapshot differs from reviewed named science: " + name)
    identity.update(science_git_head=restored["git_head"], bundle_sha256=restored["bundle_sha256"])
    selected = prerequisite["selected_checkpoint"]
    if selected["adapted_teacher_sha256"] != intent["adapted_teacher_sha256"]:
        raise ValueError("selected checkpoint identity differs")
    selected_root = safe(selected["checkpoint_root"])
    if project not in selected_root.parents or mount(selected_root)["fstype"] not in {"lustre", "nfs", "nfs4", "ceph"}:
        raise ValueError("selected checkpoint is not on confirmed shared persistent storage")
    if shutil.disk_usage(work).free < 72 * 1024 ** 3:
        raise ValueError("qualification lacks 72 GiB node-local staging/headroom")
    with timed_phase("selected_checkpoint_staging"):
        checkpoint_staging = qualify_contract.stage_checkpoint(selected, work / "selected-checkpoint")
    with timed_phase("fit_evidence_staging"):
        fit_staging = qualify_contract.stage_fit_evidence(prerequisite["fit_evidence"], work / "fit-evidence")
    staged_cache = work / "huggingface"
    with timed_phase("model_staging"):
        model_staging = stage_models(cache, staged_cache)
    output = work / "outputs" / "qwen3-v2"
    log = work / "worker.log"
    temporary = work / "tmp"
    temporary.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(HF_HOME=str(staged_cache), HF_HUB_CACHE=str(staged_cache / "hub"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6",
        NUMEXPR_NUM_THREADS="6", TMPDIR=str(temporary),
        XDG_CACHE_HOME=str(work / "cache"), TORCH_EXTENSIONS_DIR=str(work / "torch-extensions"),
        TRITON_CACHE_DIR=str(work / "triton-cache"))
    worker = str(source / "tools/sdsc_teacher_qualify.py")
    arguments = ["--science-root", str(science), "--work-dir", str(work), "--output-dir", str(output),
        "--hf-home", str(staged_cache), "--run-id", run_id, "--code-sha256", code_hash,
        "--job-id", job_id, "--provenance-manifest", str(safe(provenance_dir) / "manifest.json"),
        "--provenance-manifest-sha256", provenance_hash,
        "--checkpoint-root", str(work / "selected-checkpoint"), "--checkpoint-sha256", intent["adapted_teacher_sha256"],
        "--prerequisites", str(work / "prerequisites.json"), "--claims", str(work / "claims.json"),
        "--fit-evidence-root", str(work / "fit-evidence")]
    validation_command = [python, "-I", "-B", "-u", worker, *arguments, "--validate-only"]
    training_command = [python, "-I", "-B", "-u", "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=4", "--max-restarts=0", worker, *arguments]
    print(json.dumps(dict(identity, phase="launch", scratch_mount=scratch_mount,
                          input_mount=input_mount, persistent_mount=persistent_mount)), flush=True)
    with log.open("xb") as stream:
        for phase, command in (("validate-only", validation_command), ("teacher-qualification", training_command)):
            with timed_phase(phase):
                process = subprocess.Popen(command, cwd=source, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=0, start_new_session=True)
                forward_output(process.stdout, stream)
                exit_code = process.wait()
                process.stdout.close()
                if exit_code != 0:
                    raise RuntimeError("%s worker exited %s" % (phase, exit_code))
    report = json.loads(safe(output / "teacher-qualify.json").read_text())
    validate_report(report, exit_code)
    report.update(identity, exit_code=0, resumable=False,
                  prerequisites_sha256=intent["prerequisites_sha256"], prerequisites_path=intent["prerequisites_path"],
                  teacher_job_id=intent["teacher_job_id"], preflight_job_id=None,
                  claims_path=intent["claims_path"], claims_sha256=intent["claims_sha256"],
                  staged_hf_home=str(staged_cache), model_staging=model_staging,
                  checkpoint_staging=checkpoint_staging, fit_evidence_staging=fit_staging,
                  scratch_mount=scratch_mount, input_mount=input_mount, persistent_mount=persistent_mount,
                  wrapper_timing_seconds=phase_timings,
                  wrapper_elapsed_before_publication_seconds=time.monotonic() - wrapper_started)
    publish(output, log, report, True)
except BaseException as error:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        shutdown = stop_child()
    except BaseException as shutdown_error:
        shutdown = dict(error="%s: %s" % (type(shutdown_error).__name__, shutdown_error))
    exit_code = 1
    diagnostics = failure_progress(work / "outputs" / "qwen3-v2" if work else None)
    failure = dict(identity, passed=False, exit_code=1, g0_passed=False,
        execution_class_certified=False, exploratory=True, accepted_science=False, formal_teacher_accepted=False,
        artifact_kind="adapted_teacher_qualification_execution", scientific_evidence_verified=False,
        readiness_artifact_produced=False, student_training_started=False,
        training_started=False, teacher_training_started=False, original_128_token_readiness_pass_claim=False,
        full_teacher_ready=False, resumable=False, error="%s: %s" % (type(error).__name__, error),
        failure_stage=wrapper_phase, wrapper_timing_seconds=phase_timings, worker_shutdown=shutdown,
        execution_diagnostics=diagnostics,
        wrapper_elapsed_before_publication_seconds=time.monotonic() - wrapper_started)
    print(json.dumps(failure), file=sys.stderr, flush=True)
    try:
        if result_root is not None and work is not None and not (result_root / "receipt.json").exists():
            publish(work / "outputs" / "qwen3-v2", work / "worker.log", failure, False)
    except BaseException as publication_error:
        failure["publication_error"] = str(publication_error)
        print(json.dumps(failure), file=sys.stderr, flush=True)
    try:
        atomic_json(safe(pathlib.Path(submission) / "control-result.json"), failure)
    except (OSError, ValueError):
        pass
sys.exit(exit_code)
PY
