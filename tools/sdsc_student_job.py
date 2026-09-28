#!/usr/bin/env python3
"""Finite two-H100 student wrapper: immutable inputs, node-local work, durable results."""

import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import pwd
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

process = None
work = None
log = None
result_root = None
identity = {}
wrapper_started = 0
wrapper_phase = "preflight"
phase_timings = {}
models = (
    ("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("models--Qwen--Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
project = pathlib.Path("/expanse/lustre/projects/nwu181/zgao12/OPD")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def safe(path):
    path = pathlib.Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(ord(char) < 32 for char in str(path))
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError(f"unsafe path: {path}")
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
    return json.loads(
        subprocess.run(
            ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    )["filesystems"][0]


@contextlib.contextmanager
def timed_phase(name):
    global wrapper_phase
    wrapper_phase = name
    started = time.monotonic()
    completed = False
    print(
        json.dumps(
            dict(
                identity,
                phase=name,
                event="started",
                timestamp_unix=time.time(),
                wrapper_elapsed_seconds=started - wrapper_started,
            )
        ),
        flush=True,
    )
    try:
        yield
        completed = True
    finally:
        phase_timings[name] = time.monotonic() - started
        print(
            json.dumps(
                dict(
                    identity,
                    phase=name,
                    event="finished",
                    timestamp_unix=time.time(),
                    elapsed_seconds=phase_timings[name],
                    completed=completed,
                )
            ),
            flush=True,
        )


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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
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
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                killed = True
                deadline = now + kill_seconds
            timeout = min(
                0.1, max(0, (exited_at + drain_seconds if exited_at is not None else deadline) - now)
            )
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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
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
                stream.write((f"\n[shutdown log truncated: {dropped} earlier bytes omitted]\n").encode())
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(
        child_started=True,
        exit_code=process.poll(),
        sigkill_sent=killed,
        pipe_drained=eof,
        received_bytes=received,
        retained_bytes=len(tail),
        omitted_bytes=dropped,
        elapsed_seconds=time.monotonic() - started,
    )


def stage_source(root, destination):
    manifest = json.loads(safe(root / "manifest.json").read_text())
    files = manifest["files"]
    if (
        hashlib.sha256(canonical(files)).hexdigest() != code_hash
        or manifest["code_sha256"] != code_hash
        or manifest["run_id"] != run_id
    ):
        raise ValueError("release identity or content hash mismatch")
    seen = set()
    for entry in files:
        name = pathlib.PurePosixPath(entry["path"])
        if (
            name.is_absolute()
            or not name.parts
            or ".." in name.parts
            or str(name) != entry["path"]
            or str(name) in seen
        ):
            raise ValueError("unsafe or duplicate manifest path")
        seen.add(str(name))
        path = safe(root / "source" / name)
        info = path.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or entry["mode"] not in (420, 493)
            or stat.S_IMODE(info.st_mode) != entry["mode"]
            or info.st_size > 4 * 1024 * 1024
        ):
            raise ValueError("invalid source file metadata")
        data = path.read_bytes()
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"source differs from manifest: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(entry["mode"])
        if file_hash(target) != entry["sha256"]:
            raise OSError("staged source failed read-back")
    if (
        not {
            "tools/sdsc_student_job.sh",
            "tools/sdsc_student_job.py",
            "tools/sdsc_student_contract.py",
            "tools/sdsc_provenance.py",
            "tools/sdsc_adapted_training_preflight.py",
            "tools/sdsc_adapted_calibration.py",
        }
        <= seen
    ):
        raise ValueError("student transport/provenance files are missing")
    # This snapshot is transport provenance. Science is restored from genuine
    # accepted Git below; every named transport byte must also match that review.
    for directory, _directories, filenames in os.walk(destination, topdown=False):
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
                if name not in {
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.model",
                    "chat_template.jinja",
                    "model.safetensors",
                    "model.safetensors.index.json",
                } and not re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name):
                    continue
                original = pathlib.Path(directory) / name
                resolved = original.resolve(strict=True)
                if cache not in resolved.parents or not stat.S_ISREG(resolved.stat().st_mode):
                    raise ValueError("model snapshot link escapes its cache or is not a regular file")
                entries.append((original, resolved, destination / original.relative_to(cache)))
    total = sum(source.stat().st_size for _, source, _ in entries)
    if not entries or shutil.disk_usage(destination.parent).free < total + 12 * 1024**3:
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
            raise OSError(f"staged model failed read-back: {original.name}")
    return dict(files=len(entries), bytes=total)


def load_helper(root, name):
    spec = importlib.util.spec_from_file_location("_student_job_" + name, root / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(output, log, report, contract):
    """Publish only a stopped attempt; durable receipt follows all read-back checks."""
    if (result_root / "receipt.json").exists():
        raise ValueError("student attempt is already published")
    if process is not None and process.poll() is None:
        raise ValueError("cannot publish a live worker")
    result_name = contract.RESULTS[report["task"]]
    atomic_json(result_root / result_name, report)
    sources = [("worker.log", log)] if log and log.is_file() else []
    if output and output.is_dir():
        for directory, directories, filenames in os.walk(output, followlinks=False):
            if any((pathlib.Path(directory) / name).is_symlink() for name in directories):
                raise ValueError("symlink output directory")
            for name in filenames:
                path = safe(pathlib.Path(directory) / name)
                sources.append(("artifacts/" + str(path.relative_to(output)), path))
    total = sum(path.lstat().st_size for _, path in sources)
    if total > contract.MAX_RESULTS or len(sources) > 9998:
        raise ValueError("student publication exceeds output budget")
    records = []
    for name, path in sources:
        target = safe(result_root / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + ".tmp")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("nonregular student output")
            hasher, count = hashlib.sha256(), 0
            with temporary.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(8 * 1024**2), b""):
                    count += len(chunk)
                    if count > before.st_size:
                        raise ValueError("student output grew during publication")
                    hasher.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if (
                contract.stable(before) != contract.stable(os.fstat(incoming.fileno()))
                or contract.stable(before) != contract.stable(path.lstat())
                or count != before.st_size
            ):
                raise ValueError("student output changed during publication")
        os.replace(temporary, target)
        record = contract.file_record(target, maximum=contract.MAX_RESULTS)
        if record["sha256"] != hasher.hexdigest() or record["size"] != count:
            raise ValueError("persistent student output failed read-back")
        records.append(dict(record, path=name))
    records.append(contract.file_record(result_root / result_name))
    receipt = dict(
        identity,
        passed=report.get("passed") is True,
        persisted=True,
        persistent_read_back_verified=True,
        files=records,
    )
    if receipt["passed"]:
        contract.validate_publication(result_root, report, receipt)
    atomic_json(result_root / "receipt.json", receipt)
    if contract.document(result_root / "receipt.json") != receipt:
        raise ValueError("student publication receipt failed read-back")
    print(
        json.dumps(
            dict(
                identity,
                phase="publication",
                passed=receipt["passed"],
                persisted=True,
                bytes=total,
                result_dir=str(result_root),
            )
        ),
        flush=True,
    )


def interrupted(signum, _frame):
    raise RuntimeError(f"student job interrupted by signal {signum}")


def main(argv=None):
    global process, work, log, result_root, identity, wrapper_started, code_hash, run_id
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 9:
        raise ValueError("nine fixed student wrapper arguments required")
    (
        release,
        submission,
        results,
        python,
        run_id,
        code_hash,
        original_hf_home,
        provenance_dir,
        provenance_hash,
    ) = args
    wrapper_started = time.monotonic()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    contract = load_helper(pathlib.Path(__file__).parent.parent, "sdsc_student_contract")
    intent = contract.document(safe(submission) / "intent.json")
    task = intent.get("task")
    contract.require(task in contract.TASKS, "unknown student task")
    identity = dict(
        task=task,
        job_id=job_id,
        run_id=run_id,
        code_sha256=code_hash,
        hf_home=original_hf_home,
        provenance_dir=provenance_dir,
        provenance_manifest_sha256=provenance_hash,
    )
    output = None
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        contract.require(
            re.fullmatch(r"[1-9][0-9]*", job_id)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id)
            and re.fullmatch(r"[a-f0-9]{64}", code_hash),
            "real Slurm/deployment identity required",
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        contract.require(
            os.environ.get("SLURM_CPUS_PER_TASK") == "24"
            and os.environ.get("SLURM_MEM_PER_NODE") == "196608"
            and len(visible) == len(set(visible)) == 2
            and all(x and x != "-1" and x.strip() == x for x in visible),
            "student task requires two GPUs, 24 CPUs and 192 GiB",
        )
        contract.require(
            all(intent.get(k) == v for k, v in identity.items() if k != "job_id"),
            "student submission identity differs",
        )
        contract.require(
            intent.get("prerequisites_path") == str(safe(submission) / "prerequisites.json"),
            "student fixed prerequisite path differs",
        )
        proof = contract.document(intent["prerequisites_path"], intent["prerequisites_sha256"])
        contract.require(
            proof.get("schema") == "quest-sdsc-adapted-student-prerequisites-v1"
            and proof.get("task") == task
            and proof.get("target") == {k: intent[k] for k in ("run_id", "code_sha256", "intent_id")},
            "student prerequisite identity differs",
        )
        identity.update(
            {
                key: intent[key]
                for key in (
                    *contract.BINDINGS,
                    "teacher_job_id",
                    "preflight_job_id",
                    "prerequisites_path",
                    "prerequisites_sha256",
                )
            }
        )
        result_root = safe(results)
        cache = safe(original_hf_home)
        for path in (result_root, cache):
            contract.require(
                project in (path, *path.parents) and path.is_dir(), "persistent project paths required"
            )
        contract.require(not any(result_root.iterdir()), "student result directory is already used")
        persistent_mount, input_mount = mount(result_root), mount(cache)
        contract.require(
            all(x["fstype"] in {"lustre", "nfs", "nfs4", "ceph"} for x in (persistent_mount, input_mount)),
            "GPU input/output mounts are not shared persistent storage",
        )
        probe = result_root / ".node-write-read-check"
        with probe.open("xb") as stream:
            stream.write(code_hash.encode())
            stream.flush()
            os.fsync(stream.fileno())
        contract.require(probe.read_bytes() == code_hash.encode(), "GPU persistent write/read failed")
        probe.unlink()
        scratch = safe(
            os.environ.get("TMPDIR") or f"/scratch/{pwd.getpwuid(os.getuid()).pw_name}/job_{job_id}"
        )
        contract.require(
            scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch unavailable"
        )
        scratch_mount = mount(scratch)
        contract.require(
            scratch_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}, "node-local disk required"
        )
        contract.require(
            shutil.disk_usage(scratch).free >= 256 * 1024**3, "student task lacks 256 GiB scratch headroom"
        )
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"opd-student-{job_id}-", dir=scratch))
        source = work / "source"
        source.mkdir()
        with timed_phase("source_staging"):
            stage_source(safe(release), source)
        contract = load_helper(source, "sdsc_student_contract")
        verifier = load_helper(source, "sdsc_provenance")
        science = work / "science"
        with timed_phase("genuine_science_restore"):
            restored = verifier.verify(safe(provenance_dir), provenance_hash, code_hash, science)
            protocol = contract.resolved_protocol(science, intent["science_git_head"])
            contract.require(protocol == proof["protocol"], "accepted student protocol changed")
            for name, expected_hash in protocol["science_file_sha256"].items():
                contract.require(
                    file_hash(safe(source / name)) == expected_hash,
                    "reviewed science/transport differs: " + name,
                )
        identity.update(science_git_head=restored["git_head"], bundle_sha256=restored["bundle_sha256"])
        bindings = dict(
            student_protocol_sha256=protocol["protocol_sha256"],
            student_protocol_artifact_sha256=protocol["artifact_sha256"],
            adapted_teacher_sha256=contract.DENSE_SHA256,
            teacher_acceptance_sha256=contract.ACCEPTED_SHA256,
            teacher_acceptance_inventory_sha256=contract.INVENTORY_SHA256,
        )
        contract.require(
            all(intent.get(k) == v for k, v in bindings.items()),
            "student submission scientific binding differs",
        )
        proof_record, proof_bytes = contract.file_record(intent["prerequisites_path"], contents=True)
        contract.require(
            proof_record["sha256"] == intent["prerequisites_sha256"], "prerequisite bytes changed"
        )
        with (work / "prerequisites.json").open("xb") as stream:
            stream.write(proof_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        contract.require(
            file_hash(work / "prerequisites.json") == intent["prerequisites_sha256"],
            "proof serialization changed",
        )
        plans, dataset = contract.input_plans(proof["qualification"])
        contract.require(
            plans == proof["student_inputs"] and dataset == proof["dataset_inputs"],
            "student input plan changed",
        )
        for name in (contract.ACCEPTANCE_ROOT, contract.QUALIFICATION_ROOT, contract.DATASET_ROOT):
            contract.require(
                project in name.parents and mount(name)["fstype"] in {"lustre", "nfs", "nfs4", "ceph"},
                "persistent student inputs unavailable on GPU node",
            )
        with timed_phase("teacher_inputs_staging"):
            teacher_staging = contract.stage_inputs(plans, work / "teacher-inputs", maximum=32 * 1024**2)
            acceptance = contract.verify_teacher_acceptance(
                work / "teacher-inputs/acceptance", expected_inventory_sha256=contract.INVENTORY_SHA256
            )
            contract.require(acceptance == proof["acceptance"], "staged teacher acceptance changed")
        checkpoint_staging = dataset_staging = None
        if contract.TASKS[task] == "preflight":
            selected = proof["selected_checkpoint"]
            contract.require(
                selected["adapted_teacher_sha256"] == contract.DENSE_SHA256, "wrong selected learned teacher"
            )
            checkpoint_root = safe(selected["checkpoint_root"])
            contract.require(
                project in checkpoint_root.parents
                and mount(checkpoint_root)["fstype"] in {"lustre", "nfs", "nfs4", "ceph"},
                "learned teacher is not on persistent project storage",
            )
            with timed_phase("selected_checkpoint_staging"):
                checkpoint_staging = load_helper(source, "sdsc_teacher_qualify_contract").stage_checkpoint(
                    selected, work / "selected-checkpoint"
                )
        else:
            with timed_phase("dataset_staging"):
                dataset_staging = contract.stage_inputs(
                    dataset, work / "inputs/qwen3-v2/dataset", maximum=2 * 1024**3
                )
        staged_cache = work / "huggingface"
        with timed_phase("model_staging"):
            model_staging = stage_models(cache, staged_cache)
        output = work / "outputs/qwen3-v2"
        log = work / "worker.log"
        temporary = work / "tmp"
        temporary.mkdir()
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment.update(
            HF_HOME=str(staged_cache),
            HF_HUB_CACHE=str(staged_cache / "hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="12",
            MKL_NUM_THREADS="12",
            OPENBLAS_NUM_THREADS="12",
            NUMEXPR_NUM_THREADS="12",
            TMPDIR=str(temporary),
            XDG_CACHE_HOME=str(work / "cache"),
            TORCH_EXTENSIONS_DIR=str(work / "torch-extensions"),
            TRITON_CACHE_DIR=str(work / "triton-cache"),
        )
        arguments = [
            "--science-root",
            str(science),
            "--science-git-head",
            identity["science_git_head"],
            "--work-dir",
            str(work),
            "--output-dir",
            str(output),
            "--hf-home",
            str(staged_cache),
            "--run-id",
            run_id,
            "--code-sha256",
            code_hash,
            "--student-protocol-sha256",
            protocol["protocol_sha256"],
        ]
        if contract.TASKS[task] == "preflight":
            command = [
                python,
                "-I",
                "-B",
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "--max-restarts=0",
                str(source / "tools/sdsc_adapted_training_preflight.py"),
                *arguments,
                "--teacher-checkpoint-root",
                str(work / "selected-checkpoint"),
                "--teacher-checkpoint-sha256",
                contract.DENSE_SHA256,
                "--teacher-acceptance",
                str(work / "teacher-inputs/acceptance"),
                "--teacher-acceptance-sha256",
                contract.INVENTORY_SHA256,
            ]
        else:
            command = [
                python,
                "-I",
                "-B",
                "-u",
                str(source / "tools/sdsc_adapted_calibration.py"),
                *arguments,
                "--teacher-input-root",
                str(work / "teacher-inputs"),
                "--dataset-root",
                str(work / "inputs/qwen3-v2/dataset"),
                "--prerequisites",
                str(work / "prerequisites.json"),
                "--prerequisites-sha256",
                intent["prerequisites_sha256"],
                "--execute-calibration",
            ]
        with log.open("xb") as stream, timed_phase(contract.TASKS[task]):
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                start_new_session=True,
            )
            forward_output(process.stdout, stream)
            status = process.wait()
            process.stdout.close()
            contract.require(status == 0, "student worker exited " + str(status))
        report = contract.document(output / contract.RESULTS[task])
        contract.validate_report(report, proof)
        contract.require(
            all(
                report.get(k) == identity[k]
                for k in ("task", "job_id", "run_id", "code_sha256", "science_git_head")
            ),
            "student worker deployment identity differs",
        )
        report.update(
            identity,
            model_staging=model_staging,
            checkpoint_staging=checkpoint_staging,
            teacher_staging=teacher_staging,
            dataset_staging=dataset_staging,
            scratch_mount=scratch_mount,
            input_mount=input_mount,
            persistent_mount=persistent_mount,
            wrapper_timing_seconds=phase_timings,
            resumable=False,
        )
        with timed_phase("publication"):
            publish(output, log, report, contract)
        return 0
    except BaseException as error:
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, signal.SIG_IGN)
        try:
            shutdown = stop_child()
        except BaseException as shutdown_error:
            shutdown = dict(error=str(shutdown_error))
        failure = dict(
            identity,
            passed=False,
            exit_code=1,
            g0_passed=False,
            pilot_passed=False,
            factorial_ready=False,
            execution_class_certified=False,
            resumable=False,
            error=f"{type(error).__name__}: {error}",
            failure_stage=wrapper_phase,
            wrapper_timing_seconds=phase_timings,
            worker_shutdown=shutdown,
        )
        print(json.dumps(failure), file=sys.stderr, flush=True)
        try:
            if result_root and work and not (result_root / "receipt.json").exists():
                publish(output, log, failure, contract)
        except BaseException as publication_error:
            failure["publication_error"] = str(publication_error)
            print(json.dumps(failure), file=sys.stderr, flush=True)
        with contextlib.suppress(OSError, ValueError):
            atomic_json(safe(submission) / "control-result.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
