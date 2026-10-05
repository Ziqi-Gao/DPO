"""Actual runtime inventory/origin producer-consumer seams and bounded negative publication."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_recovery_tests_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    return [
        load(name)
        for name in (
            "sdsc_student_name_cue_recovery_job",
            "sdsc_student_name_cue_recovery_audit",
            "sdsc_runtime_snapshot",
            "sdsc_student_name_cue_recovery",
        )
    ]


def fixture_control(control, audit):
    return SimpleNamespace(
        FLAGS=control.FLAGS,
        canonical=control.canonical,
        sha=control.sha,
        require=control.require,
        write_once=control.write_once,
        read=control.read,
        document=control.document,
        science=control.science,
        helper=lambda name: audit if name == "sdsc_student_name_cue_recovery_audit" else control.helper(name),
    )


@pytest.fixture
def bundle(modules, tmp_path, monkeypatch):
    node, audit, snapshot, control = modules
    source_root, archive_root, work = (tmp_path / name for name in ("runtime-source", "archives", "work"))
    for path in (source_root, archive_root, work):
        path.mkdir()
    prefix = source_root / "environment"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib/python3.12/site-packages").mkdir(parents=True)
    (prefix / "bin/python3.12").write_bytes(b"fixture native executable bytes\x00")
    (prefix / "bin/python3.12").chmod(0o755)
    (prefix / "bin/python").symlink_to("python3.12")
    for name in ("torch", "numpy", "transformers", "datasets"):
        (prefix / f"lib/python3.12/site-packages/{name}.py").write_text("VALUE = 42\n")
    (prefix / "lib/native.so").write_bytes(b"fixture mapped native bytes")
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
    manifest = json.loads(raw)
    plan = dict(
        task=control.TASK,
        intent_id="a" * 32,
        runtime_snapshot=descriptor,
        science_plan=dict(
            code_sha256="c" * 64,
            run_id="fixture",
            control_sha256={
                "tools/sdsc_student_name_cue_probe_worker.py": control.sha(
                    (ROOT / "tools/sdsc_student_name_cue_probe_worker.py").read_bytes()
                )
            },
            protocol={"science_file_sha256": {}},
            result_dir=str(tmp_path / "results"),
        ),
        execution={"control_file_sha256": {}},
    )
    context = node.context_value(
        plan, stage, work, {"job_id": "12345", "cuda_visible_devices": "2,3"}, control
    )
    audit.validate_stage(plan, stage, raw)
    audit.validate_context(plan, context, stage)
    return dict(
        node=node,
        audit=audit,
        snapshot=snapshot,
        control=fixture_control(control, audit),
        plan=plan,
        stage=stage,
        manifest=manifest,
        manifest_raw=raw,
        context=context,
        work=work,
    )


def produce_origin(bundle, monkeypatch, phase="early", rank=None):
    b = bundle
    node, control, stage = b["node"], b["control"], b["stage"]
    prefix = Path(stage["destination"])
    observed_modules = {}
    for name in ("torch", "numpy", "transformers", "datasets"):
        path = str(prefix / f"lib/python3.12/site-packages/{name}.py")
        observed_modules[name] = SimpleNamespace(__file__=path, __spec__=SimpleNamespace(origin=path))
    state = SimpleNamespace(
        executable=stage["python"],
        prefix=str(prefix),
        base_prefix=str(prefix),
        path=[str(prefix / "lib/python3.12")],
        version_info=(3, 12, 13),
        dont_write_bytecode=True,
        modules=observed_modules,
    )
    monkeypatch.setattr(node, "sys", state)
    monkeypatch.setattr(
        node.importlib.metadata, "version", lambda name: control.science.RUNTIME_PACKAGES[name]
    )
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/maps":
            return "0000-1000 r-xp 0000 00:00 0 " + str(prefix / "lib/native.so") + "\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    for key in b["audit"].ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in dict(
        CUDA_VISIBLE_DEVICES="2,3",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        OPENBLAS_NUM_THREADS="12",
    ).items():
        monkeypatch.setenv(key, value)
    return node.capture_runtime(
        b["plan"], stage, b["context"], b["manifest"], phase=phase, rank=rank, control=control
    )


def test_real_snapshot_stage_origin_producer_to_independent_auditor(bundle, monkeypatch):
    b = bundle
    value = produce_origin(b, monkeypatch)
    assert value["passed"], value["error"]
    summary = b["audit"].validate_runtime_origin(
        b["plan"], b["stage"], b["manifest"], b["context"], value, phase="early"
    )
    assert summary["module_count"] == 4 and summary["native_library_count"] == 1
    assert b["stage"]["descriptor"] == b["plan"]["runtime_snapshot"]


@pytest.mark.parametrize(
    "change",
    [
        lambda b, v: v.update(executable=b["plan"]["runtime_snapshot"]["source_prefix"] + "/bin/python3.12"),
        lambda b, v: v.update(prefix=b["plan"]["runtime_snapshot"]["source_prefix"]),
        lambda b, v: v["sys_path"].append("/tmp/foreign-python"),
        lambda b, v: v["sys_path"].append(b["plan"]["runtime_snapshot"]["source_prefix"] + "/lib/python3.12"),
        lambda b, v: v["environment"].update(PYTHONPATH="/tmp/import"),
        lambda b, v: v["environment"].update(
            LD_LIBRARY_PATH=b["plan"]["runtime_snapshot"]["source_prefix"] + "/lib"
        ),
        lambda b, v: v["environment"].update(PYTHONDONTWRITEBYTECODE=None),
        lambda b, v: v["environment"].update(CUDA_VISIBLE_DEVICES="3,2"),
        lambda b, v: v["environment"].update(OMP_NUM_THREADS="24"),
        lambda b, v: v.update(dont_write_bytecode=False),
        lambda b, v: v["packages"].update(torch="2.7"),
        lambda b, v: v["files"][0].update(sha256="0" * 64),
        lambda b, v: v["modules"][0].update(spec_origin="/tmp/foreign.so"),
        lambda b, v: v.update(rank=0),
        lambda b, v: v.update(student_accepted=True),
        lambda b, v: v.update(passed=1),
        lambda b, v: v.update(native_libraries=[]),
    ],
)
def test_reject_origin_corruption(bundle, monkeypatch, change):
    b = bundle
    value = produce_origin(b, monkeypatch)
    assert value["passed"], value["error"]
    change(b, value)
    with pytest.raises(ValueError):
        b["audit"].validate_runtime_origin(
            b["plan"], b["stage"], b["manifest"], b["context"], value, phase="early"
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(node_local_verified=False),
        lambda s: s.update(schema="quest-sdsc-native-runtime-rehearsal-v1"),
        lambda s: s["local_mount"].update(fstype="lustre"),
        lambda s: s.update(archive_verified=False),
        lambda s: s.update(files_verified=False),
        lambda s: s.update(python="/original/bin/python3.12"),
    ],
)
def test_rehearsal_or_unverified_stage_never_satisfies_production(bundle, change):
    b = bundle
    stage = copy.deepcopy(b["stage"])
    change(stage)
    with pytest.raises(ValueError):
        b["audit"].validate_stage(b["plan"], stage, b["manifest_raw"])


def test_changed_runtime_file_produces_retained_negative_evidence(bundle, monkeypatch):
    b = bundle
    (Path(b["stage"]["destination"]) / "lib/python3.12/site-packages/torch.py").write_text("CHANGED = True\n")
    value = produce_origin(b, monkeypatch)
    assert value["passed"] is False and "bytes differ" in value["error"]
    assert value["files"] and value["modules"]


def test_both_file_and_spec_origin_are_observed(bundle, monkeypatch):
    b = bundle
    value = produce_origin(b, monkeypatch)
    state = b["node"].sys
    state.modules["torch"].__spec__.origin = str(
        Path(b["plan"]["runtime_snapshot"]["source_prefix"]) / "lib/python3.12/site-packages/torch.py"
    )
    value = b["node"].capture_runtime(
        b["plan"], b["stage"], b["context"], b["manifest"], phase="early", rank=None, control=b["control"]
    )
    assert not value["passed"] and "original runtime" in value["error"]


def test_local_python_used_for_both_probe_and_torchrun_without_inner_mutation(bundle):
    b = bundle
    before = copy.deepcopy(b["plan"])
    n = b["node"]
    probe = n.probe_argv(b["plan"], b["context"], b["stage"])
    worker = n.worker_argv(b["plan"], b["context"], b["stage"])
    assert probe[:6] == [b["stage"]["python"], "-I", "-B", "-u", "-X", "importtime"]
    assert worker[:8] == [
        b["stage"]["python"],
        "-B",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
    ]
    assert b["plan"] == before


def test_actual_bounded_failure_publication_roundtrip(bundle, tmp_path):
    b = bundle
    execution = tmp_path / "execution"
    execution.mkdir()
    result_root = Path(b["plan"]["science_plan"]["result_dir"])
    result_root.mkdir()
    (result_root / "receipt.json").write_bytes(b"original scientific publication")
    (execution / "startup.log").write_bytes(b"failed import raw bytes\xff")
    result = dict(
        schema="quest-sdsc-student-name-cue-recovery-node-v1",
        job_id="12345",
        plan_sha256=b["control"].sha(b["control"].canonical(b["plan"])),
        science_plan_sha256=b["control"].sha(b["control"].canonical(b["plan"]["science_plan"])),
        stage_complete=False,
        exit_code=1,
        **dict.fromkeys(b["control"].FLAGS, False),
    )
    publication = b["node"].persist_execution(
        b["plan"], b["control"], execution, result, {"initial": {"failed": True}}
    )
    root = result_root / "execution"
    files = {row["path"]: (root / row["path"]).read_bytes() for row in publication["files"]}
    b["audit"].validate_execution_publication(b["plan"], publication, files)
    assert publication["passed"] is False and files["startup.log"].endswith(b"\xff")
    assert publication["scientific_publication_sha256"] == b["control"].sha(
        (result_root / "receipt.json").read_bytes()
    )
    assert set(files) == {"startup.log", "recovery-node-result.json", "memory.json"}
    with pytest.raises(ValueError):
        b["audit"].validate_execution_evidence(b["plan"], files)
    files["startup.log"] += b"changed"
    with pytest.raises(ValueError, match="bytes differ"):
        b["audit"].validate_execution_publication(b["plan"], publication, files)


def test_false_success_is_downgraded_with_negative_evidence(bundle, tmp_path):
    b = bundle
    execution = tmp_path / "execution"
    execution.mkdir()
    root = Path(b["plan"]["science_plan"]["result_dir"])
    root.mkdir()
    (root / "receipt.json").write_bytes(b"inner")
    result = dict(
        schema="quest-sdsc-student-name-cue-recovery-node-v1",
        plan_sha256=b["control"].sha(b["control"].canonical(b["plan"])),
        science_plan_sha256=b["control"].sha(b["control"].canonical(b["plan"]["science_plan"])),
        job_id="12345",
        stage_complete=True,
        exit_code=0,
        **dict.fromkeys(b["control"].FLAGS, False),
    )
    publication = b["node"].persist_execution(b["plan"], b["control"], execution, result, {})
    assert publication["passed"] is False and result["exit_code"] == 1 and result["evidence_error"]


def test_supervisor_reaps_only_owned_child_and_preserves_deadline(modules):
    node, _, _, control = modules
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    started = time.monotonic()
    with pytest.raises(ValueError, match="deadline"):
        node.supervise(child, control, deadline=started + 5.1, measure=lambda _: None)
    assert child.poll() is not None and time.monotonic() - started < 7


def complete_execution_files(b, monkeypatch):
    """Real capture and frozen startup producers, with only CUDA/hardware mocked."""
    import contextlib
    import io
    import math
    import os

    startup = load("sdsc_student_name_cue_probe_startup")
    control = b["control"]
    node, audit = b["node"], b["audit"]
    work = b["work"]
    execution = work / "execution"
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
            worker_sha256=b["plan"]["science_plan"]["control_sha256"][
                "tools/sdsc_student_name_cue_probe_worker.py"
            ],
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
    limit = 384 * 1024**3
    peak = 50 * 1024**3
    path = "/sys/fs/cgroup/slurm/job_12345"
    evidence = dict(
        passed=True,
        expected_limit_bytes=limit,
        limit_bytes=limit,
        path=path,
        current_bytes=peak,
        peak_bytes=peak,
        headroom_bytes=limit - peak,
        minimum_headroom_bytes=math.ceil(limit * 0.2),
        observed_at_unix=1,
        ancestors=[
            dict(
                path=path,
                limit_bytes=limit,
                current_bytes=peak,
                peak_bytes=peak,
                memory_events={"oom": 0, "oom_kill": 0},
            )
        ],
    )
    files["memory.json"] = control.canonical(
        {
            name: copy.deepcopy(evidence)
            for name in (
                "initial",
                "after_cuda_probe",
                "after_staging",
                "after_inference",
                "final",
                "after_publication",
            )
        }
    )
    files["recovery-node-result.json"] = control.canonical(
        dict(
            schema="quest-sdsc-student-name-cue-recovery-node-v1",
            job_id="12345",
            plan_sha256=control.sha(control.canonical(b["plan"])),
            science_plan_sha256=control.sha(control.canonical(b["plan"]["science_plan"])),
            stage_complete=True,
            exit_code=0,
            elapsed_seconds=30,
            **dict.fromkeys(control.FLAGS, False),
        )
    )
    assert set(files) == set(audit.NAMES)
    return files


def test_complete_original_startup_origin_producers_independent_execution_audit(bundle, monkeypatch):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    report = b["audit"].validate_execution_evidence(b["plan"], files)
    assert report["execution_complete"] and report["scientific_acceptance"] is False
    assert len(report["native_origins"]) == 5
    execution = b["work"] / "published-evidence"
    execution.mkdir()
    for name, raw in files.items():
        if name not in ("recovery-node-result.json", "memory.json"):
            (execution / name).write_bytes(raw)
    root = Path(b["plan"]["science_plan"]["result_dir"])
    root.mkdir()
    (root / "receipt.json").write_bytes(b"original-scientific-receipt")
    publication = b["node"].persist_execution(
        b["plan"],
        b["control"],
        execution,
        json.loads(files["recovery-node-result.json"]),
        json.loads(files["memory.json"]),
    )
    assert publication["passed"] is True
    published = {row["path"]: (root / "execution" / row["path"]).read_bytes() for row in publication["files"]}
    b["audit"].validate_execution_publication(b["plan"], publication, published)
    b["audit"].validate_execution_evidence(b["plan"], published)


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("runtime-rank-1-after.json", "prefix", "/original/runtime"),
        ("runtime-rank-0-after.json", "rank", 1),
        ("runtime-rank-1-after.json", "rank", True),
        ("rank-1-exit.json", "exit_code", 1),
        ("trace-meta.json", "python", "/original/bin/python3.12"),
        ("recovery-node-result.json", "elapsed_seconds", 4801),
        ("runtime-context.json", "job_id", "54321"),
    ],
)
def test_complete_evidence_rejects_cross_rank_final_or_startup_drift(bundle, monkeypatch, name, key, value):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    row = json.loads(files[name])
    row[key] = value
    files[name] = b["control"].canonical(row)
    with pytest.raises(ValueError):
        b["audit"].validate_execution_evidence(b["plan"], files)


def test_fixed_probe_suffix_uses_actual_frozen_trace_contract(bundle):
    b = bundle
    argv = b["node"].probe_argv(b["plan"], b["context"], b["stage"])
    assert argv[-8:] == b["node"].frozen_probe_args(b["plan"], b["context"])


def test_origin_publication_uses_16mib_bound_not_frozen_startup_one_mib(bundle, tmp_path):
    b = bundle
    value = {"large": "x" * (1024**2 + 1)}
    path = tmp_path / "origin.json"
    b["node"].publish_origin(b["control"], path, value)
    assert json.loads(path.read_bytes()) == value
    with pytest.raises(ValueError, match="bound"):
        b["node"].publish_origin(b["control"], tmp_path / "oversized.json", {"large": "x" * (16 * 1024**2)})


def test_actual_node_initial_failure_publishes_both_bound_receipts(tmp_path, monkeypatch):
    path = ROOT / "tests/sdsc/test_student_name_cue_recovery_transport.py"
    spec = importlib.util.spec_from_file_location("_recovery_transport_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    c = fixture.c.__wrapped__(monkeypatch)
    plan = fixture.plan.__wrapped__(c, tmp_path, monkeypatch)
    node = load("sdsc_student_name_cue_recovery_job")
    original = c.helper("sdsc_student_name_cue_probe_job")
    original_helper = c.helper
    monkeypatch.setattr(
        c,
        "helper",
        lambda name, *a: original if name == "sdsc_student_name_cue_probe_job" else original_helper(name, *a),
    )
    monkeypatch.setattr(original, "allocation", lambda *a: dict(job_id="12345", cuda_visible_devices="2,3"))
    monkeypatch.setattr(original, "mount", lambda *a: dict(fstype="lustre"))
    monkeypatch.setattr(original, "memory_envelope", lambda *a: dict(fixture_failure=True))
    monkeypatch.setattr(node, "load_control", lambda *a: (c, plan))
    monkeypatch.setattr(
        c, "verify_source", lambda *a: (_ for _ in ()).throw(ValueError("fixture invalid release"))
    )
    monkeypatch.setattr(node.signal, "signal", lambda *a: None)
    assert node.node_main(["fixture", "0" * 64]) == 1
    root = Path(plan["science_plan"]["result_dir"])
    scientific = c.document(root / "receipt.json")
    execution = c.document(root / "execution/receipt.json")
    assert scientific["passed"] is execution["passed"] is False
    assert scientific["large_files"] == execution["large_files"] == []
    audit = c.helper("sdsc_student_name_cue_recovery_audit")
    files = {r["path"]: (root / "execution" / r["path"]).read_bytes() for r in execution["files"]}
    audit.validate_execution_publication(plan, execution, files)
    assert set(files) == {"recovery-node-result.json", "memory.json"}
    assert execution["scientific_publication_sha256"] == c.sha((root / "receipt.json").read_bytes())
    assert "fixture invalid release" in json.loads(files["recovery-node-result.json"])["error"]


def publication_fixture(b, files, *, job="12345"):
    c = b["control"]
    plan = b["plan"]
    inner = plan["science_plan"]
    audit = b["audit"]
    return dict(
        schema=audit.PUBLICATION_SCHEMA,
        task=plan["task"],
        recovery_intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        science_plan_sha256=c.sha(c.canonical(inner)),
        job_id=job,
        run_id=inner["run_id"],
        code_sha256=inner["code_sha256"],
        execution_complete=True,
        stage_complete=True,
        passed=True,
        scientific_publication_sha256="f" * 64,
        persistent_read_back_verified=True,
        files=[dict(path=n, size=len(raw), sha256=c.sha(raw)) for n, raw in files.items()],
        large_files=[],
        **dict.fromkeys(c.FLAGS, False),
    )


def test_internally_valid_foreign_job_cannot_attach_to_publication(bundle, monkeypatch):
    b = bundle
    c = b["control"]
    audit = b["audit"]
    files = complete_execution_files(b, monkeypatch)
    files = {name: raw.replace(b"12345", b"54321") for name, raw in files.items()}
    context = json.loads(files["runtime-context.json"])
    context_sha = c.sha(c.canonical(context))
    for name in (
        "runtime-early.json",
        *(f"runtime-rank-{r}-{phase}.json" for r in (0, 1) for phase in ("before", "after")),
    ):
        row = json.loads(files[name])
        row["context_sha256"] = context_sha
        files[name] = c.canonical(row)
    trace = json.loads(files["trace-meta.json"])
    trace["log"]["sha256"] = c.sha(files["startup.log"])
    files["trace-meta.json"] = c.canonical(trace)
    assert audit.validate_execution_evidence(b["plan"], files)["execution_complete"]
    publication = publication_fixture(b, files)
    with pytest.raises(ValueError, match="publication/node identity"):
        audit.validate_execution_publication(b["plan"], publication, files)


def test_publication_requires_real_readback_bool(bundle, monkeypatch):
    b = bundle
    files = complete_execution_files(b, monkeypatch)
    publication = publication_fixture(b, files)
    publication["persistent_read_back_verified"] = 1
    with pytest.raises(ValueError, match="identity"):
        b["audit"].validate_execution_publication(b["plan"], publication, files)


@pytest.mark.parametrize("wrong", ["outer_sha", "inner_job", "inner_plan"])
def test_cli_receipt_binding_rejects_untrusted_outer_or_inner(bundle, monkeypatch, tmp_path, wrong):
    b = bundle
    c = b["control"]
    audit = b["audit"]
    b["plan"]["science_plan"]["intent_id"] = "a" * 32
    b["context"] = b["node"].context_value(
        b["plan"], b["stage"], b["work"], {"job_id": "12345", "cuda_visible_devices": "2,3"}, c
    )
    files = complete_execution_files(b, monkeypatch)
    pub = publication_fixture(b, files)
    inner = b["plan"]["science_plan"]
    science = c.science
    inner_pub = dict(
        schema="quest-sdsc-student-name-cue-probe-publication-v1",
        task=science.TASK,
        mode="probe",
        job_id="12345",
        run_id=inner["run_id"],
        intent_id=inner["intent_id"],
        code_sha256=inner["code_sha256"],
        plan_sha256=c.sha(c.canonical(inner)),
        preparation_complete=False,
        selected_checkpoint=None,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=[],
        diagnostic_complete=True,
        stage_complete=True,
        passed=True,
        files=[],
        **dict.fromkeys(c.FLAGS, False),
    )
    if wrong == "inner_job":
        inner_pub["job_id"] = "54321"
    if wrong == "inner_plan":
        inner_pub["plan_sha256"] = "0" * 64
    root = tmp_path / "fetch"
    (root / "execution").mkdir(parents=True)
    raw = c.canonical(inner_pub)
    (root / "receipt.json").write_bytes(raw)
    pub["scientific_publication_sha256"] = c.sha(raw)
    for name, raw in files.items():
        (root / "execution" / name).write_bytes(raw)
    raw = c.canonical(pub)
    (root / "execution/receipt.json").write_bytes(raw)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(c.canonical(b["plan"]))
    original = audit.helper
    adapted = SimpleNamespace(
        validate_plan=lambda p: p, read=c.read, science=science, write_once=c.write_once
    )
    monkeypatch.setattr(
        audit, "helper", lambda name: adapted if name == "sdsc_student_name_cue_recovery" else original(name)
    )
    argv = [
        "--plan",
        str(plan_path),
        "--fetched-dir",
        str(root),
        "--receipt-sha256",
        "0" * 64 if wrong == "outer_sha" else c.sha(raw),
        "--output",
        str(tmp_path / "audit.json"),
    ]
    with pytest.raises(ValueError, match="receipt SHA|publication scope/identity"):
        audit.main(argv)
    assert not (tmp_path / "audit.json").exists()
