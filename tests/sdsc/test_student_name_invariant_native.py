"""Real tiny runtime staging, raw origin capture and complete flat execution evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "_name_invariant_test_" + name, ROOT / "tools" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    node, snapshot, control = (
        load(n)
        for n in (
            "sdsc_student_name_invariant_native",
            "sdsc_runtime_snapshot",
            "sdsc_student_name_invariant",
        )
    )
    source_root, archive_root, work = (tmp_path / n for n in ("runtime-source", "archives", "work"))
    for path in (source_root, archive_root, work):
        path.mkdir(mode=0o700)
    prefix = source_root / "environment"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib/python3.12/site-packages").mkdir(parents=True)
    (prefix / "bin/python3.12").write_bytes(b"fixture executable bytes")
    (prefix / "bin/python3.12").chmod(0o755)
    (prefix / "bin/python").symlink_to("python3.12")
    for name in ("torch", "numpy", "transformers", "datasets"):
        (prefix / f"lib/python3.12/site-packages/{name}.py").write_text("VALUE = 42\n")
    (prefix / "lib/native.so").write_bytes(b"fixture mapped native")
    descriptor = snapshot.prepare(
        prefix, archive_root / "snapshot", source_root=source_root, artifact_root=archive_root
    )
    monkeypatch.setattr(
        snapshot,
        "mount_info",
        lambda p: dict(target=str(work), fstype="ext4", source="/dev/test", options="rw"),
    )
    stage = snapshot.stage(
        archive_root / "snapshot",
        descriptor["manifest"]["sha256"],
        work / "runtime",
        artifact_root=archive_root,
        destination_root=work,
    )
    raw = Path(descriptor["manifest"]["path"]).read_bytes()
    pins = {name: node.sha((ROOT / name).read_bytes()) for name in control.TOOLS}
    plan = dict(
        mode="preflight",
        code_sha256="c" * 64,
        python=str(prefix / "bin/python3.12"),
        control_sha256=pins,
        protocol={"science_file_sha256": dict(pins)},
        runtime_snapshot=descriptor,
    )
    context = node.context_value(
        plan, stage, work, {"job_id": "12345", "cuda_visible_devices": "2,3"}, control
    )
    # CPU fixture substitutes only allocation UID, not production runtime/source checks.
    context["uid"] = 543540
    manifest = node.validate_stage(plan, stage, raw)
    node.validate_context(plan, context, stage)
    return dict(
        node=node,
        audit=node,
        control=control,
        snapshot=snapshot,
        plan=plan,
        stage=stage,
        manifest=manifest,
        manifest_raw=raw,
        context=context,
        work=work,
    )


def produce_origin(b, monkeypatch, phase="early", rank=None):
    node, control, stage = b["node"], b["control"], b["stage"]
    prefix = Path(stage["destination"])
    modules = {}
    for name in ("torch", "numpy", "transformers", "datasets"):
        path = str(prefix / f"lib/python3.12/site-packages/{name}.py")
        modules[name] = SimpleNamespace(__file__=path, __spec__=SimpleNamespace(origin=path))
    state = SimpleNamespace(
        executable=stage["python"],
        prefix=str(prefix),
        base_prefix=str(prefix),
        path=[str(prefix / "lib/python3.12")],
        version_info=(3, 12, 13),
        dont_write_bytecode=True,
        modules=modules,
    )
    monkeypatch.setattr(node, "sys", state)
    monkeypatch.setattr(node, "helper", load)
    monkeypatch.setattr(node.importlib.metadata, "version", lambda name: control.RUNTIME_PACKAGES[name])
    original = Path.read_text

    def read_text(path, *a, **kw):
        if str(path) == "/proc/self/maps":
            return "0000-1000 r-xp 0000 00:00 0 " + str(prefix / "lib/native.so") + "\n"
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text)
    for key in node.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in dict(
        CUDA_VISIBLE_DEVICES="2,3",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        OPENBLAS_NUM_THREADS="12",
        TMPDIR=str(b["work"]),
    ).items():
        monkeypatch.setenv(key, value)
    return node.capture_runtime(
        b["plan"], stage, b["context"], b["manifest"], phase=phase, rank=rank, control=control
    )


def complete_execution_files(b, monkeypatch):
    """Real capture and frozen startup producers, with only CUDA/hardware mocked."""
    import contextlib
    import io

    startup = load("sdsc_student_name_invariant_startup")
    control = b["control"]
    node = b["node"]
    work = b["work"]
    execution = work / "artifacts"
    execution.mkdir(exist_ok=True)
    files = {
        "runtime-stage.json": control.canonical(b["stage"]),
        "runtime-manifest.json": b["manifest_raw"],
        "runtime-context.json": control.canonical(b["context"]),
    }
    for phase, rank, name in [
        ("early", None, "runtime-early.json"),
        *(
            (phase, rank, f"runtime-rank-{rank}-{phase}.json")
            for rank in (0, 1)
            for phase in ("before", "after")
        ),
    ]:
        with monkeypatch.context() as mp:
            observed = produce_origin(b, mp, phase=phase, rank=rank)
            assert observed["passed"], observed["error"]
            files[name] = control.canonical(observed)

    class FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 2

        def init(self):
            return None

        def get_device_name(self, index):
            return "NVIDIA H100 fixture"

        def get_device_capability(self, index):
            return (9, 0)

        def get_device_properties(self, index):
            return SimpleNamespace(total_memory=80 * 1024**3, uuid="fixture")

    fake = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"), cuda=FakeCuda())
    with monkeypatch.context() as mp:
        mp.setattr(startup, "import_torch", lambda: fake)
        mp.setattr(startup.platform, "python_version", lambda: "3.12.13")
        mp.setattr(startup.sys, "executable", b["stage"]["python"])
        for name in ("enable", "dump_traceback_later", "cancel_dump_traceback_later"):
            mp.setattr(startup.faulthandler, name, lambda *a, **kw: None)
        for key in startup.ENVIRONMENT_KEYS:
            mp.delenv(key, raising=False)
        for key, value in dict(
            SLURM_JOB_ID="12345",
            SLURM_CPUS_PER_TASK="24",
            SLURM_MEM_PER_NODE="393216",
            SLURM_JOB_ACCOUNT="nwu181",
            SLURM_JOB_PARTITION="nairr-gpu-shared",
            CUDA_VISIBLE_DEVICES="2,3",
            OMP_NUM_THREADS="12",
            MKL_NUM_THREADS="12",
            OPENBLAS_NUM_THREADS="12",
            TMPDIR=str(work),
        ).items():
            mp.setenv(key, value)
        out = io.StringIO()
        before = time.monotonic()
        with contextlib.redirect_stdout(out):
            assert startup.main(["probe", *node.frozen_probe_args(b["plan"], b["context"])]) == 0
        raw = out.getvalue().encode()
        events = [
            json.loads(line)
            for line in raw.splitlines()
            if json.loads(line).get("schema") == startup.PROGRESS_SCHEMA
        ]
        meta = dict(
            schema=startup.TRACE_SCHEMA,
            job_id="12345",
            cuda_visible_devices="2,3",
            python=b["stage"]["python"],
            worker_sha256=b["plan"]["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
            adapter_sha256=startup.file_sha(startup.__file__),
            argv=node.probe_argv(b["plan"], b["context"], b["stage"]),
            maximum_probe_seconds=300,
            stack_interval_seconds=30,
            time_limit_seconds=300,
            shutdown_reserve_seconds=5,
            elapsed_seconds=time.monotonic() - before + 0.01,
            child_environment=startup.environment(),
            pid=os.getpid(),
            exit_code=0,
            timed_out=False,
            reaped=True,
            events_truncated=False,
            probe_passed=True,
            error=None,
            shutdown_error=None,
            adapter_events=events,
            log=dict(
                path="startup.log",
                bytes_seen=len(raw),
                retained_bytes=len(raw),
                truncated=False,
                sha256=control.sha(raw),
            ),
            **startup.FLAGS,
        )
        files.update(
            {
                "trace-meta.json": control.canonical(meta),
                "startup.log": raw,
                "early-node-startup.json": (execution / "early-node-startup.json").read_bytes(),
            }
        )
        original = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(work / "artifacts")]
        for rank in (0, 1):
            mp.setenv("RANK", str(rank))
            mp.setenv("LOCAL_RANK", str(rank))
            mp.setenv("WORLD_SIZE", "2")
            with contextlib.redirect_stdout(io.StringIO()):
                report = startup.startup("run", "12345", "2,3", meta["worker_sha256"])
            assert report["cuda_ready"], report
            report["original_argv"] = original
            files[f"rank-{rank}-startup.json"] = control.canonical(report)
            files[f"rank-{rank}-exit.json"] = control.canonical(
                dict(
                    schema=startup.EXIT_SCHEMA,
                    job_id="12345",
                    rank=rank,
                    original_worker_sha256=meta["worker_sha256"],
                    original_argv=original,
                    worker_invoked=True,
                    exit_code=0,
                    **startup.FLAGS,
                )
            )
    files["node-result.json"] = control.canonical(
        dict(
            job_id="12345",
            plan_sha256=control.sha(control.canonical(b["plan"])),
            work_dir=str(work),
            runtime=dict(
                source_python=b["plan"]["python"],
                effective_python=b["stage"]["python"],
                runtime_stage_sha256=control.sha(control.canonical(b["stage"])),
            ),
        )
    )
    return files


def test_actual_snapshot_capture_flat_complete_evidence(bundle, monkeypatch):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    summary = b["node"].validate_evidence(b["plan"], files, "12345")
    assert summary["passed"] and summary["worker_exit_verified"] == 0
    assert len(summary["native"]) == 5 and not summary["execution_acceptance_claim"]


@pytest.mark.parametrize(
    "change",
    [
        "job",
        "uid",
        "after",
        "python",
        "tmpdir",
        "foreign_path",
        "generated_missing",
        "science_unpinned",
        "node_root",
        "snapshot",
        "helper_control",
        "helper_science",
    ],
)
def test_complete_evidence_fails_closed(bundle, monkeypatch, change):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    if change in ("helper_control", "helper_science"):
        target = (
            b["plan"]["control_sha256"]
            if change == "helper_control"
            else b["plan"]["protocol"]["science_file_sha256"]
        )
        target["tools/sdsc_torch_generated_origins.py"] = "0" * 64
    else:
        name = "runtime-rank-1-after.json"
        if change in ("job", "uid"):
            name = "runtime-context.json"
        if change == "python":
            name = "rank-1-startup.json"
        if change == "node_root":
            name = "node-result.json"
        if change == "snapshot":
            name = "runtime-stage.json"
        value = json.loads(files[name])
        if change == "job":
            value["job_id"] = "54321"
        elif change == "uid":
            value["uid"] = True
        elif change == "after":
            value.update(passed=False, error="fixture after failure")
        elif change == "python":
            value["python"] = b["plan"]["python"]
        elif change == "tmpdir":
            value["environment"]["TMPDIR"] = "/tmp"
        elif change == "foreign_path":
            value["sys_path"].append("/tmp/foreign")
        elif change == "generated_missing":
            value["modules"].append(dict(name="_remote_module_non_scriptable", file=None, spec_origin=None))
        elif change == "science_unpinned":
            path = str(b["work"] / "science/src/unknown.py")
            value["files"].append(dict(path=path, size=10, sha256="d" * 64, uid=543540, mode=0o600))
            value["files"].sort(key=lambda r: r["path"])
        elif change == "node_root":
            value["work_dir"] = "/tmp/other"
        elif change == "snapshot":
            value["node_local_verified"] = False
        files[name] = b["node"].canonical(value)
    with pytest.raises(ValueError):
        b["node"].validate_evidence(b["plan"], files, "12345")


@pytest.mark.parametrize(
    "name",
    [
        "runtime-stage.json",
        "runtime-manifest.json",
        "runtime-context.json",
        "runtime-early.json",
        "runtime-rank-0-before.json",
        "runtime-rank-1-after.json",
        "early-node-startup.json",
        "rank-0-startup.json",
        "rank-1-exit.json",
        "trace-meta.json",
        "node-result.json",
    ],
)
def test_complete_evidence_rejects_duplicate_json_fields(bundle, monkeypatch, name):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    raw = files[name]
    # A duplicate invisible to json.loads must not be accepted after publication hashing.
    files[name] = b'{"fixture_duplicate":0,"fixture_duplicate":1,' + raw[1:]
    with pytest.raises(ValueError):
        b["node"].validate_evidence(b["plan"], files, "12345")


@pytest.mark.parametrize("bad", [True, 1, 3, -1, 2.0])
def test_worker_exit_interpretation_is_not_generic_override(bundle, bad):
    with pytest.raises(ValueError, match="unreviewed worker exit"):
        bundle["node"].validate_evidence(bundle["plan"], {}, "12345", expected_worker_exit=bad)


def test_full_fit_scientific_rejection_retains_native_audit_without_success_claim(bundle, monkeypatch):
    b = bundle
    b["plan"]["mode"] = "fit"
    b["context"]["plan_sha256"] = b["node"].sha(b["node"].canonical(b["plan"]))
    files = complete_execution_files(b, monkeypatch)
    for rank in (0, 1):
        name = f"rank-{rank}-exit.json"
        value = json.loads(files[name])
        value["exit_code"] = 2
        files[name] = b["node"].canonical(value)
    with pytest.raises(ValueError, match="rank execution failed"):
        b["node"].validate_evidence(b["plan"], files, "12345")
    result = b["node"].validate_evidence(b["plan"], files, "12345", expected_worker_exit=2)
    assert result["worker_exit_verified"] == 2 and result["execution_acceptance_claim"] is False


def test_preflight_never_accepts_exit_two(bundle):
    with pytest.raises(ValueError, match="unreviewed worker exit"):
        bundle["node"].validate_evidence(bundle["plan"], {}, "12345", expected_worker_exit=2)


@pytest.mark.parametrize("raw", [b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}'])
def test_nonfinite_json_is_rejected(bundle, raw):
    with pytest.raises(ValueError, match="nonfinite JSON"):
        bundle["node"].strict_json(raw)


def test_changed_actual_runtime_file_is_retained_negative(bundle, monkeypatch):
    b = bundle
    (Path(b["stage"]["destination"]) / "lib/python3.12/site-packages/torch.py").write_text("CHANGED=True")
    value = produce_origin(b, monkeypatch)
    assert value["passed"] is False and "runtime bytes differ" in value["error"]
    assert value["files"] and value["modules"]
