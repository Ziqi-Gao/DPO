#!/bin/bash
# Fixed two-H100 canonical-SFT calibration; never complete G0 or a certificate. Invoked only by the remote submission boundary.
set -euo pipefail
if [[ $# != 11 || $4 != /* || ! -x $4 ]]; then
    printf '%s\n' 'usage: sdsc_calibration_job.sh RELEASE SUBMISSION RESULTS ABSOLUTE_PYTHON RUN_ID CODE_SHA256 HF_HOME PROVENANCE_DIR PROVENANCE_MANIFEST_SHA256 PREREQUISITES PREREQUISITES_SHA256' >&2
    exit 2
fi
exec "$4" -I -B -u - "$@" <<'PY'
import hashlib, importlib.util, json, os, pathlib, pwd, re, shutil, signal, stat, subprocess, sys, tempfile

release, submission, results, python, run_id, code_hash, original_hf_home, provenance_dir, provenance_hash, prerequisites_path, prerequisites_hash = sys.argv[1:]
job_id = os.environ.get("SLURM_JOB_ID", "")
project = pathlib.Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
control_provenance = pathlib.Path("/home/zgao12/quest-runs/OPD/provenance")
control_releases = pathlib.Path("/home/zgao12/quest-runs/OPD/releases")
models = (
    ("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("models--Qwen--Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
identity = dict(task="qwen3-v2-g0-calibration", job_id=job_id, run_id=run_id,
                code_sha256=code_hash, hf_home=original_hf_home, provenance_dir=provenance_dir,
                provenance_manifest_sha256=provenance_hash, prerequisites_path=prerequisites_path,
                prerequisites_sha256=prerequisites_hash)
process = None
work = None
result_root = None
exit_code = 1

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

def stop_child():
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        # The worker may have exited while a launcher descendant remains alive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

def interrupted(signum, _frame):
    raise RuntimeError("calibration interrupted by signal %s" % signum)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

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
    if not {"tools/sdsc_calibration_job.sh", "tools/sdsc_g0_calibration.py", "tools/sdsc_teacher_prepare.py", "tools/sdsc_training_preflight.py", "tools/sdsc_provenance.py"} <= seen:
        raise ValueError("teacher preparation worker/provenance verifier is missing from release")

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
    if not entries or shutil.disk_usage(destination.parent).free < total + 256 * 1024 ** 3:
        raise ValueError("model cache is empty or node-local free space lacks checkpoint headroom")
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

def restore_science(source, destination):
    artifact = safe(provenance_dir)
    if control_provenance not in artifact.parents or not artifact.is_dir():
        raise ValueError("provenance must be a separate artifact in the project control directory")
    specification = importlib.util.spec_from_file_location("sdsc_bound_provenance", source / "tools/sdsc_provenance.py")
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    evidence = helper.verify(artifact, provenance_hash, code_hash, destination=destination)
    raw, _ = helper.read_regular(artifact / "manifest.json")
    if hashlib.sha256(raw).hexdigest() != provenance_hash:
        raise ValueError("provenance manifest changed after verification")
    local_manifest = destination.parent / "provenance-manifest.json"
    with local_manifest.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if file_hash(local_manifest) != provenance_hash:
        raise OSError("local provenance manifest failed read-back")
    return evidence, local_manifest

def validate_report(report, status):
    for key in ("task", "job_id", "run_id", "code_sha256", "provenance_manifest_sha256", "science_git_head", "bundle_sha256", "prerequisites_sha256"):
        if str(report.get(key)) != identity[key]:
            raise ValueError("worker report identity mismatch: %s" % key)
    if (status != 0 or report.get("passed") is not True or report.get("exit_code") != 0
            or report.get("g0_passed") is not False or report.get("execution_class_certified") is not False
            or report.get("factorial_ready") is not False or report.get("pilot_passed") is not False):
        raise ValueError("calibration failed or claims unsupported G0/pilot/certification")

def publish(output, log, report, passed):
    # Only this fresh attempt writes this directory. Every copied byte is read back.
    # Small diagnosis is durable before copying checkpoints; receipt is written last.
    require(process is None or process.poll() is not None, "cannot publish while a worker is writing")
    atomic_json(result_root / "g0-calibration.json", report)
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
    for name, source in sources:
        if not stat.S_ISREG(source.stat().st_mode):
            raise ValueError("output is not a regular file")
        target = safe(result_root / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe(target.with_name("." + target.name + ".tmp"))
        digest = hashlib.sha256()
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming, temporary.open("xb") as outgoing:
            before = os.fstat(incoming.fileno())
            require(stat.S_ISREG(before.st_mode), "output is not a regular file")
            total = 0
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                total += len(block)
                require(total <= before.st_size, "output grew during publication")
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
        require(total == before.st_size and (before.st_size, before.st_mtime_ns) ==
                (after.st_size, after.st_mtime_ns), "output changed during publication")
        os.replace(temporary, target)
        if file_hash(target) != digest.hexdigest():
            raise OSError("persistent result failed read-back")
        records.append(dict(path=name, size=target.stat().st_size, sha256=digest.hexdigest()))
    records.append(dict(path="g0-calibration.json", size=(result_root / "g0-calibration.json").stat().st_size,
                        sha256=file_hash(result_root / "g0-calibration.json")))
    if json.loads((result_root / "g0-calibration.json").read_text()) != report:
        raise OSError("persistent calibration report failed read-back")
    receipt = dict(identity, passed=passed, persisted=True, persistent_read_back_verified=True, files=records)
    atomic_json(result_root / "receipt.json", receipt)
    if json.loads((result_root / "receipt.json").read_text()) != receipt:
        raise OSError("publication receipt failed read-back")
    print(json.dumps(dict(identity, phase="publication", passed=passed, persisted=True,
                          result_dir=str(result_root), calibration_sha256=records[-1]["sha256"])), flush=True)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def bounded_bytes(path, maximum=4 * 1024 ** 2):
    path = safe(path)
    require(stat.S_ISREG(path.lstat().st_mode), "expected a regular proof file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "proof file exceeds limit")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) <= maximum and (before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), "proof changed while reading")
    return raw

def document(path, expected_hash, expected_value=None):
    require(re.fullmatch(r"[a-f0-9]{64}", expected_hash), "invalid proof SHA")
    raw = bounded_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == expected_hash, "proof SHA mismatch")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON proof key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(value, dict), "proof must be an object")
    json.dumps(value, allow_nan=False)
    require(expected_value is None or value == expected_value, "proof bytes differ from bound document")
    return value, raw

def safe_relative(value):
    require(isinstance(value, str) and value and not any(ord(c) < 32 for c in value), "bad artifact path")
    path = pathlib.PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value
            and "\\" not in value, "unsafe artifact path")
    return path

def checked_copy(source, target, entry):
    source, target = safe(source), safe(target)
    require(stat.S_ISREG(source.lstat().st_mode), "artifact is not a regular file")
    require(type(entry.get("size")) is int and entry["size"] >= 0
            and re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")), "bad artifact record")
    target.parent.mkdir(parents=True, exist_ok=True)
    value, total = hashlib.sha256(), 0
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming:
        before = os.fstat(incoming.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == entry["size"], "artifact size differs")
        with target.open("xb") as outgoing:
            for chunk in iter(lambda: incoming.read(8 * 1024 ** 2), b""):
                total += len(chunk)
                require(total <= entry["size"], "artifact grew while copying")
                value.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and total == entry["size"] and value.hexdigest() == entry["sha256"], "artifact SHA differs")
    require(file_hash(target) == entry["sha256"], "staged artifact failed read-back")
    target.chmod(0o444)

def verify_upstream(role, proof, proofs):
    receipt, status = proof["receipt"], proof["status"]
    expected_task = {"teacher": "qwen3-v2-teacher-prepare", "preflight": "qwen3-v2-preflight"}[role]
    upstream_job = str(receipt["job_id"])
    require(re.fullmatch(r"[1-9][0-9]*", upstream_job), "invalid upstream job ID")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", receipt["run_id"])
            and re.fullmatch(r"[a-f0-9]{64}", receipt["code_sha256"]), "bad upstream release identity")
    require(receipt["task"] == expected_task and status.get("task") == expected_task
            and status.get("success") is True and status.get("state") == "COMPLETED"
            and str(status.get("job_id")) == upstream_job
            and status.get("run_id") == receipt["run_id"], "upstream status identity/finality differs")
    require(status["queue"]["returncode"] == 0 and status["accounting"]["returncode"] == 0,
            "upstream Slurm queries did not succeed")
    require(not any(line.split("|", 1)[0].strip() == upstream_job
                    for line in status["queue"]["stdout"].splitlines()), "upstream job is still queued")
    rows = [line.split("|") for line in status["accounting"]["stdout"].splitlines() if line.strip()]
    relevant = [row for row in rows if row[0] == upstream_job or row[0].startswith(upstream_job + ".")]
    require(sum(row[0] == upstream_job for row in relevant) == 1 and
            all(len(row) >= 3 and row[1] == "COMPLETED" and row[2] == "0:0" for row in relevant),
            "upstream accounting is not successful for allocation and every step")
    require(status["result"].get("verified") is True
            and status["result"].get("result_sha256") == proof["report_sha256"], "upstream result is unverified")
    root = safe(proof["result_dir"])
    require(root == safe(receipt["result_dir"]) and project in root.parents and root.is_dir(),
            "upstream results escape project storage")
    require(mount(root)["fstype"] in {"lustre", "nfs", "nfs4", "ceph"}, "upstream storage is not persistent")
    name = "teacher-prepare.json" if role == "teacher" else "preflight.json"
    require(safe(proof["report_path"]) == root / name
            and safe(proof["publication_receipt_path"]) == root / "receipt.json", "upstream proof path differs")
    publication, publication_raw = document(root / "receipt.json", proof["publication_receipt_sha256"], proof["publication_receipt"])
    report, report_raw = document(root / name, proof["report_sha256"], proof["report"])
    require(publication.get("passed") is True and publication.get("persisted") is True
            and publication.get("persistent_read_back_verified") is True
            and report.get("passed") is True and report.get("exit_code") == 0,
            "upstream publication did not complete")
    for value in (publication, report):
        require(all(str(value.get(key)) == str(receipt[key]) for key in ("task", "run_id", "job_id", "code_sha256")),
                "upstream report/receipt identity differs")
        require(value.get("hf_home") == receipt.get("hf_home"), "upstream model-cache identity differs")
    release_path = safe(proof["release_manifest_path"])
    expected_release = control_releases / receipt["run_id"] / "manifest.json"
    require(release_path == expected_release, "upstream release location differs")
    manifest, manifest_raw = document(release_path, proof["release_manifest_sha256"], proof["release_manifest"])
    require(manifest.get("run_id") == receipt["run_id"] and manifest.get("code_sha256") == receipt["code_sha256"]
            and hashlib.sha256(canonical(manifest["files"])).hexdigest() == receipt["code_sha256"],
            "upstream release manifest identity differs")
    files = publication.get("files", [])
    require(isinstance(files, list) and files, "upstream publication inventory missing")
    paths = [str(safe_relative(entry["path"])) for entry in files]
    require(len(paths) == len(set(paths)), "duplicate publication paths")
    matches = [entry for entry in files if entry["path"] == name]
    require(len(matches) == 1 and matches[0]["size"] == len(report_raw)
            and matches[0]["sha256"] == proof["report_sha256"], "report is not in publication inventory")
    for suffix, raw in (("report", report_raw), ("publication-receipt", publication_raw), ("release-manifest", manifest_raw)):
        with (proofs / (role + "-" + suffix + ".json")).open("xb") as stream:
            stream.write(raw)
    return root, files

def stage_prerequisites(proofs, destination):
    path = safe(prerequisites_path)
    require(path == safe(pathlib.Path(submission) / "prerequisites.json"), "prerequisites must be submission-bound")
    value, raw = document(path, prerequisites_hash)
    require(value.get("schema") == "quest-sdsc-calibration-prerequisites-v1"
            and value.get("task") == identity["task"], "wrong calibration prerequisites schema/task")
    require(value.get("target") == {"run_id": run_id, "code_sha256": code_hash, "intent_id": pathlib.Path(submission).name},
            "prerequisites target differs from submission")
    require(set(value["upstream"]) == {"teacher", "preflight"}, "both upstream proofs are required")
    identity.update(teacher_job_id=str(value["upstream"]["teacher"]["receipt"]["job_id"]),
                    preflight_job_id=str(value["upstream"]["preflight"]["receipt"]["job_id"]))
    with (proofs / "prerequisites.json").open("xb") as stream:
        stream.write(raw)
    teacher_root, entries = verify_upstream("teacher", value["upstream"]["teacher"], proofs)
    verify_upstream("preflight", value["upstream"]["preflight"], proofs)
    selected = [entry for entry in entries if entry["path"].startswith(("artifacts/dataset/", "artifacts/teacher_demos/"))]
    require(selected and any(e["path"] == "artifacts/dataset/manifest.json" for e in selected)
            and any(e["path"] == "artifacts/teacher_demos/manifest.json" for e in selected), "complete teacher inputs absent")
    observed = set()
    for name in ("dataset", "teacher_demos"):
        base = safe(teacher_root / "artifacts" / name)
        require(base.is_dir(), "teacher input directory missing")
        for folder, directories, filenames in os.walk(base, followlinks=False):
            require(not any((pathlib.Path(folder) / child).is_symlink() for child in directories), "teacher input symlink directory")
            for child in filenames:
                source = safe(pathlib.Path(folder) / child)
                require(stat.S_ISREG(source.lstat().st_mode), "teacher input is not a regular file")
                observed.add(str(source.relative_to(teacher_root)))
    require(observed == {entry["path"] for entry in selected}, "teacher input tree differs from publication inventory")
    total = sum(entry["size"] for entry in selected)
    require(shutil.disk_usage(proofs.parent).free >= total + 256 * 1024 ** 3, "scratch lacks training checkpoint headroom")
    for entry in selected:
        relative = safe_relative(entry["path"])
        checked_copy(teacher_root / relative, destination / pathlib.Path(*relative.parts[1:]), entry)
    return value, {"files": len(selected), "bytes": total, "read_back_verified": True}

try:
    if (not re.fullmatch(r"[1-9][0-9]*", job_id)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id)
            or not re.fullmatch(r"[a-f0-9]{64}", code_hash)
            or not re.fullmatch(r"[a-f0-9]{64}", provenance_hash)
            or not re.fullmatch(r"[a-f0-9]{64}", prerequisites_hash)):
        raise ValueError("real Slurm allocation and valid deployment identity required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "24"
            or os.environ.get("SLURM_MEM_PER_NODE") != "196608"
            or len(visible) != 2 or len(set(visible)) != 2
            or any(value in {"", "-1"} or value.strip() != value for value in visible)):
        raise ValueError("calibration requires exactly two visible GPUs, 24 CPUs and 192 GiB")
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
    if scratch_mount["fstype"] not in {"ext2", "ext3", "ext4", "xfs", "btrfs", "btrfs"}:
        raise ValueError("scratch is not verified node-local storage")
    work = pathlib.Path(tempfile.mkdtemp(prefix="opd-calibration-%s-" % job_id, dir=scratch))
    source = work / "source"
    source.mkdir()
    stage_source(safe(release), source)
    science = work / "science"
    provenance, local_provenance_manifest = restore_science(source, science)
    identity["science_git_head"] = provenance["git_head"]
    identity["bundle_sha256"] = provenance["bundle_sha256"]
    proofs = work / "proofs"
    proofs.mkdir()
    teacher_root = work / "inputs" / "qwen3-v2"
    prerequisites, staged_inputs = stage_prerequisites(proofs, teacher_root)
    staged_cache = work / "huggingface"
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
        OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", TMPDIR=str(temporary),
        XDG_CACHE_HOME=str(work / "cache"), TORCH_EXTENSIONS_DIR=str(work / "torch-extensions"),
        TRITON_CACHE_DIR=str(work / "triton-cache"))
    command = [python, "-I", "-B", "-u", str(source / "tools/sdsc_g0_calibration.py"),
        "--science-root", str(science), "--work-dir", str(work), "--output-dir", str(output),
        "--hf-home", str(staged_cache), "--run-id", run_id, "--code-sha256", code_hash,
        "--python", python, "--provenance-manifest", str(local_provenance_manifest),
        "--provenance-manifest-sha256", provenance_hash,
        "--teacher-root", str(teacher_root),
        "--teacher-report", str(proofs / "teacher-report.json"),
        "--teacher-report-sha256", prerequisites["upstream"]["teacher"]["report_sha256"],
        "--preflight-report", str(proofs / "preflight-report.json"),
        "--preflight-report-sha256", prerequisites["upstream"]["preflight"]["report_sha256"],
        "--preflight-snapshot-manifest", str(proofs / "preflight-release-manifest.json"),
        "--prerequisites", str(proofs / "prerequisites.json"),
        "--prerequisites-sha256", prerequisites_hash]
    print(json.dumps(dict(identity, phase="launch", scratch_mount=scratch_mount,
                          input_mount=input_mount, persistent_mount=persistent_mount)), flush=True)
    with log.open("x") as stream:
        for phase, arguments in (("validate-only", ["--validate-only"]), ("calibration", ["--execute-calibration"])):
            print(json.dumps(dict(identity, phase=phase)), flush=True)
            process = subprocess.Popen(command + arguments, cwd=science, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
            if exit_code != 0:
                raise RuntimeError("%s worker exited %s" % (phase, exit_code))
    report = json.loads(safe(output / "g0-calibration.json").read_text())
    validate_report(report, exit_code)
    report.update(identity, exit_code=0, resumable=False, staged_prerequisites_path=str(proofs / "prerequisites.json"),
                  staged_hf_home=str(staged_cache), model_staging=model_staging, staged_teacher_inputs=staged_inputs,
                  scratch_mount=scratch_mount, input_mount=input_mount, persistent_mount=persistent_mount)
    publish(output, log, report, True)
except BaseException as error:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    stop_child()
    exit_code = 1
    failure = dict(identity, passed=False, exit_code=1, g0_passed=False,
        execution_class_certified=False, factorial_ready=False, pilot_passed=False,
        resumable=False, error="%s: %s" % (type(error).__name__, error))
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
