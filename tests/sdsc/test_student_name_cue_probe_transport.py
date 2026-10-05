"""Bounded inference transport, historical parent pins, v2 provenance and real startup seams."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"sbatch", "sacct", "squeue", "scontrol", "srun", "scancel", "ssh"}:
            pytest.fail("CPU fixture attempted remote or Slurm command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load("sdsc_student_name_cue_probe")


@pytest.fixture
def n():
    return load("sdsc_student_name_cue_probe_job")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    raw = (ROOT / c.PROTOCOL_PATH).read_bytes()
    protocol = json.loads(raw)
    core = {k: v for k, v in protocol.items() if k != "review"}
    pins = {name: c.sha((ROOT / name).read_bytes()) for name in c.TOOLS}
    pins.update(c.FROZEN)
    binding = dict(
        protocol_sha256=c.sha(c.canonical(core)),
        artifact_sha256=c.sha(raw),
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        head="2" * 40,
        science_file_sha256={name: c.sha((ROOT / name).read_bytes()) for name in protocol["science_files"]},
    )
    parents = {}
    for arm, pin in c.PARENTS.items():
        receipt = dict(
            arm=arm,
            task=c._parent.TASK,
            mode="probe",
            job_id=pin["job_id"],
            intent_id=pin["intent_id"],
            plan_sha256=pin["plan_sha256"],
            result_dir=str(c.PROJECT / "student-focus-lr-probe" / pin["intent_id"]),
            code_sha256="3" * 64,
            run_id="historical",
            resources=c._parent.resources(),
            received_at="2026-10-05T00:00:00Z",
            reconciled=False,
        )
        parents[arm] = dict(
            plan_path=str(c.CONTROL / "student-focus-lr-probe-submissions" / pin["intent_id"] / "plan.json"),
            plan_sha256=pin["plan_sha256"],
            publication_sha256=pin["publication_sha256"],
            report_sha256=pin["report_sha256"],
            checkpoint=copy.deepcopy(pin["checkpoint"]),
            receipt=receipt,
        )
    contract = c.helper("sdsc_student_contract")
    result = dict(
        schema="quest-sdsc-student-name-cue-probe-plan-v1",
        task=c.TASK,
        mode="probe",
        run_id="new-cue-run",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        created_at="2026-10-05T00:00:00Z",
        resources=c.resources(),
        parents=parents,
        resolved_config=dict(
            path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=c.CONFIG_SHA
        ),
        protocol=binding,
        dataset_inputs=[
            dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES
        ],
        control_sha256=pins,
        provenance=dict(
            directory=str(c.CONTROL / "provenance-v2" / ("e" * 64)), manifest_sha256="e" * 64, head="2" * 40
        ),
        science_identity=c.science_identity(binding),
        python=str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12"),
        hf_home=str(c.PROJECT / "cache/huggingface"),
        **dict.fromkeys(c.FLAGS, False),
    )
    intent = c.execution_intent(result)
    result.update(
        intent_id=intent,
        release=str(c.CONTROL / "releases" / result["run_id"]),
        submission_dir=str(c.CONTROL / "student-name-cue-probe-submissions" / intent),
        claim=str(c.CONTROL / "student-name-cue-probe-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-name-cue-probe" / intent),
        job_name="opd-snc-" + intent,
    )
    result["scientific_claim"] = str(c.scientific_claim_path(result))
    c.validate_plan(result)
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


def write_once(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


@pytest.fixture
def startup_setup(tmp_path, monkeypatch):
    node = load("sdsc_student_name_cue_probe_job")
    probe = load("sdsc_student_name_cue_probe_startup")
    helpers = {"sdsc_student_name_cue_probe_startup": probe}

    def helper(name):
        if name not in helpers:
            helpers[name] = load(name)
        return helpers[name]

    c = SimpleNamespace(
        helper=helper,
        require=require,
        write_once=write_once,
        canonical=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")).encode(),
        sha=lambda raw: hashlib.sha256(raw).hexdigest(),
    )
    for key in probe.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    environment = dict(os.environ)
    environment.update(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        OMP_NUM_THREADS="7",
        MKL_NUM_THREADS="8",
        TMPDIR=str(tmp_path),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    source = tmp_path / "release/source/tools"
    source.mkdir(parents=True)
    script = source / "sdsc_student_name_cue_probe_startup.py"
    script.write_text(
        """import importlib.util,sys,platform,os,time
from types import SimpleNamespace
path=sys.argv.pop(1) if False else """
        + repr(str(ROOT / "tools/sdsc_student_name_cue_probe_startup.py"))
        + """
spec=importlib.util.spec_from_file_location("real_probe",path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
platform.python_version=lambda:"3.12.13"
mode=os.environ.get("FIXTURE_MODE","good")
class FakeCuda:
    def is_available(self): return mode!="cuda_false"
    def device_count(self): return 2
    def init(self): return None
    def get_device_name(self,index): return "NVIDIA H100 fixture"
    def get_device_capability(self,index): return (9,0)
    def get_device_properties(self,index): return SimpleNamespace(total_memory=80*1024**3,uuid="fixture")
torch=SimpleNamespace(__version__="2.8.0+cu128",version=SimpleNamespace(cuda="12.8"),cuda=FakeCuda())
def import_torch():
    if mode=="hang": time.sleep(30)
    if mode=="import_error": raise ImportError("fixture import failure")
    if mode=="mutate_env": os.environ["OMP_NUM_THREADS"]="99"
    return torch
probe.import_torch=import_torch
if mode=="flood": print("x"*(2*1024**2),flush=True)
raise SystemExit(probe.main())
"""
    )
    execution = tmp_path / "work/execution"
    execution.mkdir(parents=True)
    environment["TMPDIR"] = str(execution.parent)
    outer = {
        "python": sys.executable,
        "release": str(tmp_path / "release"),
        "control_sha256": {
            "tools/sdsc_student_name_cue_probe_worker.py": hashlib.sha256(
                (ROOT / "tools/sdsc_student_name_cue_probe_worker.py").read_bytes()
            ).hexdigest()
        },
    }
    identity = {"job_id": "12345", "cuda_visible_devices": "2,3"}
    return node, probe, c, outer, identity, execution, environment


def run(startup_setup, mode="good", budget=20):
    node, probe, c, outer, identity, execution, environment = startup_setup
    environment["FIXTURE_MODE"] = mode
    meta = node.run_probe(
        outer, c, execution, identity, environment, deadline=time.monotonic() + budget, measure=lambda _: None
    )
    probe.validate_trace_meta(
        meta,
        worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        expected_python=sys.executable,
        expected_argv=node.probe_argv(outer, execution, identity),
        log_bytes=(execution / "startup.log").read_bytes(),
    )
    return meta


def test_real_adapter_calls_frozen_worker_and_records_policy_without_changing_parent(startup_setup):
    node, probe, c, outer, identity, execution, environment = startup_setup
    before = dict(environment)
    meta = run(startup_setup)
    assert meta["probe_passed"] is True and meta["reaped"] is True and meta["exit_code"] == 0
    assert meta["child_environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert all(meta["child_environment"][key] == "12" for key in probe.THREAD_KEYS)
    assert {key: environment.get(key) for key in probe.THREAD_KEYS} == {
        key: before.get(key) for key in probe.THREAD_KEYS
    }
    assert meta["child_environment"]["TMPDIR"] == str(execution.parent)
    raw = json.loads((execution / "early-node-startup.json").read_bytes())
    probe.validate_report(
        raw,
        worker_sha256=outer["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        scope="early_node",
    )
    assert raw["wrapper_sha256"] == probe.file_sha(probe.__file__)
    assert all(raw[key] is False for key in probe.FLAGS)
    assert b"import time:" in (execution / "startup.log").read_bytes()


@pytest.mark.parametrize("mode", ["cuda_false", "import_error", "mutate_env"])
def test_real_negative_probe_retains_trace_without_claiming_ready(startup_setup, mode):
    meta = run(startup_setup, mode)
    assert meta["probe_passed"] is False and meta["reaped"] is True and meta["exit_code"] == 1
    assert json.loads((startup_setup[5] / "trace-meta.json").read_bytes()) == meta
    assert (startup_setup[5] / "early-node-startup.json").exists()


def test_timeout_reaps_child_and_keeps_original_global_budget(startup_setup):
    meta = run(startup_setup, "hang", budget=5.8)
    assert meta["timed_out"] is True and meta["reaped"] is True and meta["probe_passed"] is False
    assert 5 < meta["time_limit_seconds"] <= 5.8
    assert meta["elapsed_seconds"] < 2
    assert meta["maximum_probe_seconds"] == 300
    assert any(row["event"] == "frozen_worker_begin" for row in meta["adapter_events"])
    assert not (startup_setup[5] / "early-node-startup.json").exists()
    assert not startup_setup[2].helper("sdsc_student_lr_job").group_has_live_members(meta["pid"])


def test_exhausted_budget_rejects_before_popen_and_preserves_negative_trace(startup_setup, monkeypatch):
    monkeypatch.setattr(
        startup_setup[0].subprocess, "Popen", lambda *a, **kw: pytest.fail("no remaining child budget")
    )
    meta = run(startup_setup, budget=4)
    assert meta["pid"] is None and meta["probe_passed"] is False
    assert "exhausted" in meta["error"]


def test_large_log_retains_tail_and_exact_omission_count(startup_setup):
    meta = run(startup_setup, "flood")
    assert meta["probe_passed"] is True
    assert meta["log"]["bytes_seen"] > 2 * 1024**2 and meta["log"]["retained_bytes"] == 1024**2
    assert meta["log"]["truncated"] is True
    assert len((startup_setup[5] / "startup.log").read_bytes()) == 1024**2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(elapsed_seconds=m["time_limit_seconds"] + 2),
        lambda m: m.update(reaped=False),
        lambda m: m.update(timed_out=True),
        lambda m: m.update(events_truncated=True),
        lambda m: m.update(exit_code=1),
        lambda m: m.update(student_accepted=True),
        lambda m: m["child_environment"].update(SLURM_MEM_PER_NODE="16384"),
        lambda m: m["adapter_events"][-1].update(returncode=1),
        lambda m: m["adapter_events"].reverse(),
        lambda m: m["adapter_events"][0].update(utc="2000-01-01T00:00:00+00:00"),
        lambda m: m["adapter_events"][-1].update(elapsed_seconds=9999),
        lambda m: m["adapter_events"][2].update(original_argv=["run"]),
        lambda m: m["log"].update(sha256="0" * 64),
    ],
)
def test_trace_rejects_forged_success_and_changed_bytes(startup_setup, mutation):
    meta = copy.deepcopy(run(startup_setup))
    mutation(meta)
    with pytest.raises((ValueError, TypeError)):
        startup_setup[1].validate_trace_meta(
            meta,
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=(startup_setup[5] / "startup.log").read_bytes(),
        )


def test_source_staging_has_explicit_manifest_inputs_and_readonly_modes(tmp_path):
    node = load("sdsc_student_name_cue_probe_job")
    contract = load("sdsc_student_contract")
    release = tmp_path / "release"
    (release / "source/tools").mkdir(parents=True)
    rows = []
    for name, mode in [("main.txt", 0o644), ("tools/run.py", 0o755)]:
        data = (name + "\n").encode()
        p = release / "source" / name
        p.write_bytes(data)
        p.chmod(mode)
        rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), mode=mode))
    manifest = dict(code_sha256="a" * 64, run_id="fixture", files=rows)
    payload = json.dumps(manifest).encode()
    (release / "manifest.json").write_bytes(payload)
    control = SimpleNamespace(
        safe=Path,
        require=require,
        manifest_records=lambda m: {r["path"]: r for r in m["files"]},
        document=lambda path, digest: json.loads(path.read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest
        else None,
        helper=lambda name: contract,
    )
    result = node.stage_source(
        dict(
            release=str(release),
            code_sha256="a" * 64,
            run_id="fixture",
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
        ),
        tmp_path / "staged",
        control,
    )
    assert result["read_back_verified"] is True and result["files"] == 2
    assert (tmp_path / "staged/main.txt").stat().st_mode & 0o777 == 0o444
    assert (tmp_path / "staged/tools/run.py").stat().st_mode & 0o777 == 0o555
    assert (tmp_path / "staged").stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize("worker_code", [0, 2])
def test_actual_rank_entry_validates_new_worker_and_records_exact_exit(
    startup_setup, tmp_path, monkeypatch, worker_code
):
    _, startup, _, plan, identity, _, environment = startup_setup
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in startup.THREAD_KEYS:
        monkeypatch.setenv(key, "12")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    spec = importlib.util.spec_from_file_location(
        "_cuda_fixture", ROOT / "tests/sdsc/test_cuda_diagnostic_worker.py"
    )
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    monkeypatch.setattr(startup, "import_torch", lambda: fakes.FakeTorch())
    monkeypatch.setattr(startup.platform, "python_version", lambda: "3.12.13")
    calls = []
    monkeypatch.setattr(
        startup, "invoke_original", lambda argv, digest: (calls.append((argv, digest)) or worker_code)
    )
    expected_sha = plan["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"]
    args = [
        "run",
        "--worker-sha256",
        expected_sha,
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        "2,3",
        "--execution-output-dir",
        str(tmp_path),
        "--inputs-json",
        str(tmp_path / "inputs.json"),
        "--output-dir",
        str(tmp_path),
    ]
    assert startup.main(args) == worker_code
    assert calls == [
        (["--inputs-json", str(tmp_path / "inputs.json"), "--output-dir", str(tmp_path)], expected_sha)
    ]
    report = json.loads((tmp_path / "rank-1-startup.json").read_bytes())
    startup.validate_report(
        report,
        worker_sha256=expected_sha,
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        scope="rank_entry",
        rank=1,
        original_argv=calls[0][0],
    )
    exit_row = json.loads((tmp_path / "rank-1-exit.json").read_bytes())
    assert exit_row["exit_code"] == worker_code and exit_row["original_worker_sha256"] == expected_sha
    assert all(exit_row[key] is False for key in startup.FLAGS)


def test_wrong_accepted_worker_digest_stops_before_training(startup_setup, tmp_path, monkeypatch):
    _, startup, _, _, identity, _, environment = startup_setup
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in startup.THREAD_KEYS:
        monkeypatch.setenv(key, "12")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(startup, "invoke_original", lambda *a: pytest.fail("mismatched worker invoked"))
    args = [
        "run",
        "--worker-sha256",
        "f" * 64,
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        "2,3",
        "--execution-output-dir",
        str(tmp_path),
        "--inputs-json",
        str(tmp_path / "inputs.json"),
        "--output-dir",
        str(tmp_path),
    ]
    assert startup.main(args) == 1
    row = json.loads((tmp_path / "rank-0-startup.json").read_bytes())
    assert row["cuda_ready"] is False and "error" in row
    assert not (tmp_path / "rank-0-exit.json").exists()


def test_rehashed_raw_log_cannot_forge_elapsed_clock_origin(startup_setup):
    meta = run(startup_setup)
    probe = startup_setup[1]
    root = startup_setup[5]
    raw = []
    for line in (root / "startup.log").read_bytes().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raw.append(line)
            continue
        if isinstance(row, dict) and row.get("schema") == probe.PROGRESS_SCHEMA:
            row["elapsed_seconds"] = 0.0
            line = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        raw.append(line)
    raw = b"\n".join(raw) + b"\n"
    for event in meta["adapter_events"]:
        event["elapsed_seconds"] = 0.0
    meta["log"].update(bytes_seen=len(raw), retained_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="origins"):
        probe.validate_trace_meta(
            meta,
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=raw,
        )


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_ssh_launcher_executes_actual_controller_or_rejects_hash_before_runpy(c, plan, tmp_path, wrong_hash):
    # Execute the exact controller-generated launcher against actual source.
    # Its real Quest-only guard is the stop point, before request handling or
    # any scheduler/SSH action. The fake CLI replaces only the SSH transport.
    release = tmp_path / "local-release"
    release.mkdir()
    (release / "source").symlink_to(ROOT, target_is_directory=True)
    plan["release"], plan["python"] = str(release), sys.executable
    controller_path = "tools/sdsc_student_name_cue_probe.py"
    controller_sha = c.sha((ROOT / controller_path).read_bytes())
    plan["control_sha256"][controller_path] = "0" * 64 if wrong_hash else controller_sha
    startup_sha = plan["control_sha256"]["tools/sdsc_student_name_cue_probe_startup.py"]
    assert startup_sha != controller_sha
    calls = []

    def local_ssh(argv, *, data, timeout):
        assert argv[:4] == [sys.executable, "-I", "-B", "-c"]
        assert argv[-2] == str(release / "source" / controller_path)
        assert json.loads(data) == dict(action="dry-run", plan=plan, authorize=False, dry_run=None)
        assert timeout == 240
        result = subprocess.run(argv, input=data, capture_output=True, timeout=10)
        calls.append(result)
        return result

    expected = "AssertionError" if wrong_hash else "Slurm operations cannot run on Quest"
    with pytest.raises(ValueError, match=expected):
        c.ssh_operation(SimpleNamespace(ssh_call=local_ssh), plan, "dry-run")
    assert len(calls) == 1 and calls[0].returncode != 0
    if wrong_hash:
        assert b"Slurm operations cannot run on Quest" not in calls[0].stderr
        assert b"sdsc_student_name_cue_probe.py" not in calls[0].stderr
    else:
        assert b"AssertionError" not in calls[0].stderr
        assert b"sdsc_student_name_cue_probe.py" in calls[0].stderr


def test_ssh_launcher_forwards_remote_argv_request_and_checked_named_payload(c, plan, tmp_path):
    release = tmp_path / "release"
    script = release / "source/tools/sdsc_student_name_cue_probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "print(json.dumps(dict(argv=sys.argv,request=request,checked_payload_executed=True)))\n"
    )
    plan["release"], plan["python"] = str(release), sys.executable
    plan["control_sha256"]["tools/sdsc_student_name_cue_probe.py"] = c.sha(script.read_bytes())
    # Keep the real distinct startup pin: reordering TOOLS must not determine
    # which file's bytes authorize the controller path.
    calls = []

    def local_ssh(argv, *, data, timeout):
        calls.append((list(argv), data, timeout))
        return subprocess.run(argv, input=data, capture_output=True, timeout=10)

    proof = {"dry_run": True, "sentinel": "unchanged request"}
    result = c.ssh_operation(SimpleNamespace(ssh_call=local_ssh), plan, "submit", True, proof)
    assert result == dict(
        argv=[str(script), "remote"],
        request=dict(action="submit", plan=plan, authorize=True, dry_run=proof),
        checked_payload_executed=True,
    )
    assert len(calls) == 1


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_actual_isolated_node_load_uses_named_controller_hash_before_any_action(
    c, plan, tmp_path, wrong_hash
):
    source = tmp_path / "node-source"
    source.mkdir()
    node = source / "sdsc_student_name_cue_probe_job.py"
    node.write_bytes((ROOT / "tools/sdsc_student_name_cue_probe_job.py").read_bytes())
    controller = source / "sdsc_student_name_cue_probe.py"
    controller.write_text("def validate_plan(plan):\n    return dict(plan, real_controller_called=True)\n")
    plan["control_sha256"]["tools/sdsc_student_name_cue_probe.py"] = (
        "0" * 64 if wrong_hash else c.sha(controller.read_bytes())
    )
    # Different startup bytes must never authorize the controller's filename.
    assert plan["control_sha256"]["tools/sdsc_student_name_cue_probe_startup.py"] != c.sha(
        controller.read_bytes()
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(c.canonical(plan))
    code = (
        "import importlib.util,json,sys;"
        "spec=importlib.util.spec_from_file_location('new_node',sys.argv[1]);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "control,plan=m.load_control(sys.argv[2],sys.argv[3]);"
        "print(json.dumps(plan))"
    )
    observed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(node), str(plan_path), c.sha(plan_path.read_bytes())],
        capture_output=True,
        timeout=15,
    )
    if wrong_hash:
        assert observed.returncode != 0 and b"execution controller changed" in observed.stderr
        assert not observed.stdout
    else:
        assert observed.returncode == 0, observed.stderr.decode()
        assert json.loads(observed.stdout)["real_controller_called"] is True


def test_inference_plan_resources_and_parent_only_claim(c, plan):
    assert c.validate_plan(plan) == plan
    assert c.resources()["time"] == "01:30:00"
    assert c.worker_budget_seconds(plan) == 4800
    assert c.resources()["gpus"] == 2
    assert c.resources()["cpus"] == 24
    assert c.resources()["mem_gib"] == 384
    assert plan["science_identity"]["optimizer_steps"] == 0
    assert not any(plan[k] for k in c.FLAGS)
    assert "arm" not in plan and len(plan["parents"]) == 2
    assert all(cp["step"] == 32 for cp in c.checkpoint_inputs(plan, Path("checkpoints")).values())
    assert len(c.required_raw_names("probe")) == 8
    assert c.MAX_FETCH == 128 * 1024**2


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["resources"].update(time="03:00:00"),
        lambda p: p["resources"].update(gpus=1),
        lambda p: p["resources"].update(cpus=4),
        lambda p: p["resources"].update(mem_gib=384.0),
        lambda p: p.update(arm="control"),
        lambda p: p.update(student_accepted=True),
        lambda p: p["parents"].pop("control"),
        lambda p: p["parents"]["control"]["checkpoint"].update(path="checkpoints/step-00000016.pt"),
        lambda p: p["parents"]["control"]["checkpoint"].update(sha256="0" * 64),
        lambda p: p["parents"]["treatment"]["receipt"].update(job_id="54663831"),
        lambda p: p["parents"]["control"].update(publication_sha256="0" * 64),
        lambda p: p["parents"]["control"].update(plan_sha256="0" * 64),
        lambda p: p["provenance"].update(
            directory=p["provenance"]["directory"].replace("provenance-v2", "provenance")
        ),
        lambda p: p["protocol"]["science_file_sha256"].pop("tools/sdsc_provenance_v2.py"),
        lambda p: p["protocol"].update(implementation_commit=p["protocol"]["acceptance_commit"]),
    ],
)
def test_plan_changes_fail_closed(c, plan, change):
    change(plan)
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(plan)


def test_explicit_v2_dispatch_never_uses_historical_v1(c, plan, tmp_path, monkeypatch):
    calls = []

    def verify(directory, manifest, wrapper, destination):
        calls.append((directory, manifest, wrapper, destination))
        return {"git_head": plan["provenance"]["head"]}

    def helper(name):
        assert name == "sdsc_provenance_v2"
        return SimpleNamespace(verify=verify)

    monkeypatch.setattr(c, "helper", helper)
    monkeypatch.setattr(c, "protocol_binding", lambda *a: plan["protocol"])
    c.verify_science(plan, tmp_path / "science")
    assert len(calls) == 1 and "provenance-v2" in str(calls[0][0])
    assert calls[0][1:3] == (plan["provenance"]["manifest_sha256"], plan["code_sha256"])


@pytest.mark.parametrize("corruption", ["head", "protocol"])
def test_v2_restored_binding_is_checked(c, plan, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(
        c,
        "helper",
        lambda name: SimpleNamespace(
            verify=lambda *a: {"git_head": "f" * 40 if corruption == "head" else plan["provenance"]["head"]}
        ),
    )
    monkeypatch.setattr(
        c, "protocol_binding", lambda *a: {} if corruption == "protocol" else plan["protocol"]
    )
    with pytest.raises(ValueError):
        c.verify_science(plan, tmp_path / "science")


def test_parent_validation_keeps_v1_verifier_and_accounting(c, plan, tmp_path, monkeypatch):
    calls = []
    parentplans = {arm: {"arm": arm} for arm in c.ARMS}
    publications = {arm: {"large_files": [plan["parents"][arm]["checkpoint"]]} for arm in c.ARMS}
    reports = {arm: {"selected_checkpoint": None, "preparation_complete": False} for arm in c.ARMS}

    def document(path, expected=None):
        path = Path(path)
        for arm, row in plan["parents"].items():
            if path == Path(row["plan_path"]):
                assert expected == row["plan_sha256"]
                return parentplans[arm]
            root = Path(row["receipt"]["result_dir"])
            if path == root / "receipt.json":
                assert expected == row["publication_sha256"]
                return publications[arm]
            if path == root / "prepare-report.json":
                assert expected == row["report_sha256"]
                return reports[arm]
            if path == root / "memory.json":
                return {}
        pytest.fail(str(path))

    monkeypatch.setattr(c, "document", document)
    parent = c._parent
    monkeypatch.setattr(parent, "validate_plan", lambda p: p)
    for name in (
        "validate_submission_receipt",
        "verify_source",
        "validate_worker_report",
        "validate_update_records",
        "validate_large_outputs",
        "validate_memory",
        "validate_startup_evidence",
    ):
        monkeypatch.setattr(parent, name, lambda *a, _name=name: calls.append(_name))
    monkeypatch.setattr(parent, "verify_science", lambda p: calls.append("historical_v1:" + p["arm"]))
    monkeypatch.setattr(parent, "publication_records", lambda *a: {})
    monkeypatch.setattr(parent, "read", lambda *a: b"")
    monkeypatch.setattr(parent, "inspect_job", lambda *a: dict(success=True, accounting_complete=True))
    result = c.verify_parent(plan, accounting=True)
    assert (
        len(result) == 2
        and calls.count("historical_v1:control") == calls.count("historical_v1:treatment") == 1
    )
    monkeypatch.setattr(parent, "inspect_job", lambda *a: dict(success=False, accounting_complete=False))
    with pytest.raises(ValueError, match="accounting incomplete"):
        c.verify_parent(plan, accounting=True)


def test_submit_dry_run_required_before_claim(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: pytest.fail("must reject before admission"))
    with pytest.raises(ValueError, match="matching dry-run"):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=None))
    assert not Path(plan["scientific_claim"]).exists()


def test_unknown_submission_is_claimed_once_without_retry(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: {})
    calls = []
    monkeypatch.setattr(
        c, "run", lambda argv: (calls.append(argv) or dict(returncode=1, stdout="", stderr="lost"))
    )
    proof = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))
    assert len(calls) == 1
    assert Path(plan["claim"]).exists() and Path(plan["scientific_claim"]).exists()
    assert (Path(plan["submission_dir"]) / "unknown.json").exists()
    with pytest.raises(FileExistsError):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))
    assert len(calls) == 1


def test_inference_launcher_has_no_optimizer_or_fsdp(c, n, plan, tmp_path):
    argv = n.worker_argv(
        plan,
        tmp_path / "source",
        tmp_path / "science",
        tmp_path / "inputs.json",
        tmp_path / "artifacts",
        tmp_path / "artifacts",
        dict(job_id="12345", cuda_visible_devices="0,1"),
    )
    assert argv[argv.index("-m") + 1] == "torch.distributed.run"
    assert "--nproc-per-node=2" in argv and "--standalone" in argv
    assert "accelerate.commands.launch" not in argv and "--config_file" not in argv
    assert "--inputs-json" in argv and "--output-dir" in argv
    script = c.job_script(plan)
    path = tmp_path / "job.sh"
    path.write_bytes(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert b"CUDA_VISIBLE_DEVICES=" not in script


def test_actual_isolated_worker_accepts_node_built_input_envelope(c, n, plan, tmp_path, monkeypatch):
    work = tmp_path / "work"
    science = work / "science"
    science.mkdir(parents=True)
    output = work / "artifacts"
    output.mkdir()
    hf = work / "huggingface"
    hf.mkdir()
    (work / "dataset").mkdir()
    originals = {}
    for index in range(49):
        name = f"src/original-{index}.py"
        path = science / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"value={index}\n")
        originals[name] = c.sha(path.read_bytes())
    protocol = science / c.PROTOCOL_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((ROOT / c.PROTOCOL_PATH).read_bytes())
    originaldoc = c.document

    def doc(path, expected=None):
        if str(path) == plan["parents"]["control"]["plan_path"]:
            return {"protocol": {"artifact_sha256": "a" * 64}}
        if Path(path).name == "prepare-report.json":
            return {"dataset_sha256": "b" * 64}
        return originaldoc(path, expected)

    monkeypatch.setattr(c, "document", doc)
    inputs = n.build_worker_inputs(plan, work, science, dict(job_id="12345"), originals, c)
    worker = load("sdsc_student_name_cue_probe_worker")
    assert set(inputs) == worker.INPUT_KEYS and len(inputs) == 14
    path = work / "inputs.json"
    path.write_text(json.dumps(inputs))
    code = """import importlib.util,json,sys;from pathlib import Path
p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('isolated_worker',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=json.loads(Path(sys.argv[2]).read_text())
work=m.staged_science(x,Path(sys.argv[3]))
assert work==Path(sys.argv[3]).parent.resolve()
print('validated')
"""
    env = {**os.environ, "HF_HOME": str(hf), "SLURM_JOB_ID": "12345", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    args = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        code,
        str(ROOT / "tools/sdsc_student_name_cue_probe_worker.py"),
        str(path),
        str(output),
    ]
    result = subprocess.run(args, cwd=science, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    inputs["optimizer"] = "forbidden"
    path.write_text(json.dumps(inputs))
    bad = subprocess.run(args, cwd=science, env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "input fields differ" in bad.stderr


def test_inference_memory_uses_original_raw_guard_without_mutation(c, monkeypatch):
    seen = []
    original = {
        "initial": {},
        "after_staging": {},
        "after_inference": {},
        "final": {},
        "after_publication": {},
    }
    monkeypatch.setattr(c._lr, "validate_memory", lambda value, job: seen.append((value, job)))
    c.validate_memory(original, "12345")
    assert "after_inference" in original and "after_training" not in original
    assert "after_training" in seen[0][0] and "after_inference" not in seen[0][0]
    with pytest.raises(ValueError):
        c.validate_memory({"after_training": {}}, "12345")


def test_failure_publication_contains_only_bounded_raw_evidence(c, n, plan, tmp_path):
    dest = Path(plan["result_dir"])
    dest.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "name-cue-report.json").write_text('{"passed":false}')
    result = dict(job_id="12345", stage_complete=False, exit_code=1, **dict.fromkeys(c.FLAGS, False))
    n.persist(plan, c, artifacts, None, result, {}, lambda stage: None)
    pub = c.document(dest / "receipt.json")
    records = c.publication_records(pub, plan, "12345")
    assert (
        pub["large_files"] == []
        and pub["large_files_read_back_verified"] is True
        and pub["diagnostic_complete"] is False
    )
    assert set(records) == {"name-cue-report.json", "node-result.json", "memory.json"}
    for name, row in records.items():
        assert c.sha((dest / name).read_bytes()) == row["sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("large_files", [{"path": "weights.pt"}]),
        ("selected_checkpoint", "control"),
        ("preparation_complete", True),
        ("student_accepted", True),
    ],
)
def test_publication_never_promotes_or_publishes_weights(c, n, plan, tmp_path, field, value):
    dest = Path(plan["result_dir"])
    dest.mkdir(parents=True)
    n.persist(
        plan,
        c,
        None,
        None,
        dict(job_id="12345", stage_complete=False, exit_code=1, **dict.fromkeys(c.FLAGS, False)),
        {},
        lambda _: None,
    )
    publication = c.document(dest / "receipt.json")
    publication[field] = value
    with pytest.raises(ValueError):
        c.publication_records(publication, plan, "12345")


@pytest.fixture
def valid_bound_report(c, plan):
    path = Path(__file__).with_name("test_student_name_cue_probe_worker_audit.py")
    spec = importlib.util.spec_from_file_location("_name_cue_worker_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.valid_report()
    report.update(
        job_id="12345",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
    )
    return report, {row["path"]: row for row in report["raw_artifacts"]}


def test_actual_worker_report_validator_accepts_only_complete_bound_inference(c, plan, valid_bound_report):
    report, records = valid_bound_report
    c.validate_worker_report(report, plan, "12345", records)
    assert report["optimizer_steps"] == 0 and report["selected_checkpoint"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(optimizer_steps=1),
        lambda r: r.update(training_input_tokens=64),
        lambda r: r.update(world_size=1),
        lambda r: r.update(total_responses=512),
        lambda r: r.update(plan_sha256="0" * 64),
        lambda r: r.update(protocol_artifact_sha256="0" * 64),
        lambda r: r.update(selected_checkpoint="control"),
        lambda r: r["models"].pop("treatment"),
        lambda r: r["models"]["control"].update(checkpoint_sha256="0" * 64),
        lambda r: r["models"]["control"].update(source_step=16),
        lambda r: r["models"]["treatment"]["ranks"][1]["completed_ordinals"].pop(),
        lambda r: r["models"]["control"]["ranks"][0]["exact_saved_master_reload"].update(
            all_tensors_exact=False
        ),
        lambda r: r["raw_artifacts"].pop(),
        lambda r: r["raw_artifacts"][0].update(sha256="0" * 64),
    ],
)
def test_real_report_binding_rejections(c, plan, valid_bound_report, change):
    report, records = valid_bound_report
    records = copy.deepcopy(records)
    change(report)
    with pytest.raises((ValueError, KeyError)):
        c.validate_worker_report(report, plan, "12345", records)


def test_actual_node_failed_early_probe_publishes_before_large_input_staging(
    startup_setup, c, plan, tmp_path, monkeypatch
):
    node, _, _, startup_plan, identity, _, environment = startup_setup
    plan["python"] = startup_plan["python"]
    plan["release"] = startup_plan["release"]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("FIXTURE_MODE", "cuda_false")
    monkeypatch.setattr(node, "load_control", lambda *a: (c, plan))
    monkeypatch.setattr(node, "allocation", lambda *a: identity)
    monkeypatch.setattr(c, "verify_source", lambda *a: None)
    monkeypatch.setattr(c, "verify_parent", lambda *a: None)
    monkeypatch.setattr(c, "verify_runtime", lambda *a: {"fixture": True})
    original = c.document

    def document(path, expected=None):
        if str(path) == plan["parents"]["control"]["plan_path"]:
            return {"parent": {"receipt": {}}}
        return original(path, expected)

    monkeypatch.setattr(c, "document", document)
    monkeypatch.setattr(node, "mount", lambda path, c: {"fstype": "ext4" if path == scratch else "lustre"})
    monkeypatch.setattr(node.shutil, "disk_usage", lambda path: SimpleNamespace(free=192 * 1024**3))
    monkeypatch.setattr(node, "memory_envelope", lambda *a: {"fixture": True})
    monkeypatch.setattr(node.signal, "signal", lambda *a: None)
    monkeypatch.setattr(
        node, "stage_source", lambda *a: pytest.fail("failed CUDA probe reached source staging")
    )
    assert node.main(["fixture", "0" * 64]) == 1
    root = Path(plan["result_dir"])
    publication = c.document(root / "receipt.json")
    rows = c.publication_records(publication, plan, identity["job_id"])
    assert not publication["diagnostic_complete"] and publication["large_files"] == []
    assert set(rows) == {
        "early-node-startup.json",
        "startup.log",
        "trace-meta.json",
        "node-result.json",
        "memory.json",
    }
    assert c.document(root / "trace-meta.json")["reaped"] is True
    assert c.document(root / "early-node-startup.json")["cuda_ready"] is False


def test_publication_memory_is_measured_after_raw_readback(c, n, plan, tmp_path):
    dest = Path(plan["result_dir"])
    dest.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "name-cue-report.json").write_text("{}")
    result = dict(job_id="12345", stage_complete=True, exit_code=0, **dict.fromkeys(c.FLAGS, False))

    def measure(stage):
        assert stage == "after_publication" and (dest / "name-cue-report.json").read_text() == "{}"
        assert not (dest / "receipt.json").exists()
        raise ValueError("fixture memory failure")

    n.persist(plan, c, artifacts, None, result, {}, measure)
    publication = c.document(dest / "receipt.json")
    assert (
        publication["diagnostic_complete"] is False
        and c.document(dest / "node-result.json")["exit_code"] == 1
    )


def test_actual_v2_full_history_restores_new_accepted_consumer_beyond128(c, tmp_path):
    """Real temporary history keeps every genuine anchor; project refs remain untouched."""
    import shutil

    provenance = load("sdsc_provenance_v2")
    legacy = load("sdsc_provenance")
    root = tmp_path / "source"
    env = provenance.git_env()

    def git(*args, cwd=root):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, env=env).stdout

    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(ROOT / ".opd-git"), str(root)],
        check=True,
        capture_output=True,
        env=env,
    )
    git("checkout", "--quiet", "51e7cafca43c8b88ddf1b4eef5198e3abc6ec550")
    git("config", "user.name", "Transport fixture")
    git("config", "user.email", "fixture@example.invalid")
    public = "0215c356355b29b5e2b407978a207db2156719e1"
    git("update-ref", "refs/remotes/public/master", public)
    payload = json.loads((ROOT / c.PROTOCOL_PATH).read_text())
    for name in [*payload["science_files"], c.PROTOCOL_PATH]:
        dst = root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    git("add", "--", *payload["science_files"], c.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Fixture name cue implementation")
    implementation = git("rev-parse", "HEAD").decode().strip()
    payload["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent fixture",
        reviewed_at_utc="2026-10-05T00:00:00Z",
        rationale="Temporary fixture acceptance only",
    )
    (root / c.PROTOCOL_PATH).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    git("add", "--", c.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Fixture review-only acceptance")
    acceptance = git("rev-parse", "HEAD").decode().strip()
    binding = c.protocol_binding(root, acceptance)
    names = git("ls-files", "-z").decode().split("\0")[:-1]
    records = [
        dict(
            path=name,
            size=(root / name).stat().st_size,
            sha256=c.sha((root / name).read_bytes()),
            mode=0o755 if (root / name).stat().st_mode & 0o111 else 0o644,
        )
        for name in sorted(names)
    ]
    wrapper = dict(
        schema="quest-sdsc-snapshot-v1",
        run_id="fixture-real-history",
        git_head=acceptance,
        files=records,
        code_sha256=c.sha(c.canonical(records)),
        total_bytes=sum(r["size"] for r in records),
    )
    manifest = root / ".sdsc/wrapper.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_bytes(c.canonical(wrapper) + b"\n")
    options = dict(
        accepted_ancestors=["51e7cafca43c8b88ddf1b4eef5198e3abc6ec550"],
        reviewed_local_implementation=implementation,
        reviewed_local_acceptance=acceptance,
    )
    with pytest.raises(legacy.ProvenanceError, match="commit limit"):
        legacy.prepare(root, manifest, **options)
    artifact = provenance.prepare(root, manifest, **options)
    document = json.loads((Path(artifact["artifact"]) / "manifest.json").read_text())
    assert len(document["local_successor"]["commits"]) > 128
    plan = dict(
        provenance=dict(
            directory=artifact["artifact"], manifest_sha256=artifact["manifest_sha256"], head=acceptance
        ),
        code_sha256=artifact["wrapper_code_sha256"],
        protocol=binding,
    )
    restored = tmp_path / "restored"
    try:
        result = c.verify_science(plan, restored)
        assert result["git_head"] == acceptance and result["verified"] is True
        assert c.protocol_binding(restored, acceptance) == binding
        assert git("rev-parse", "refs/remotes/public/master").decode().strip() == public
    finally:
        for folder, _dirs, files in os.walk(tmp_path):
            Path(folder).chmod(0o700)
            for name in files:
                path = Path(folder) / name
                if not path.is_symlink():
                    path.chmod(0o600)


@pytest.mark.parametrize(
    "changed_key,value",
    [(None, None)]
    + [
        (key, value)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        for value in (None, "11")
    ],
)
def test_actual_startup_captures_and_checks_each_thread_variable(monkeypatch, changed_key, value):
    startup = load("sdsc_student_name_cue_probe_startup")
    for key in startup.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    declared = dict(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        OPENBLAS_NUM_THREADS="12",
    )
    for key, expected in declared.items():
        monkeypatch.setenv(key, expected)
    if changed_key:
        if value is None:
            monkeypatch.delenv(changed_key)
        else:
            monkeypatch.setenv(changed_key, value)

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
            return SimpleNamespace(total_memory=80 * 1024**3, uuid=f"fixture-{index}")

    monkeypatch.setattr(startup.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(
        startup,
        "import_torch",
        lambda: SimpleNamespace(
            __version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"), cuda=FakeCuda()
        ),
    )
    worker_sha = startup.file_sha(ROOT / "tools/sdsc_student_name_cue_probe_worker.py")
    before = dict(os.environ)
    report = startup.startup("probe", "12345", "2,3", worker_sha)
    assert dict(os.environ) == before
    assert set(startup.THREAD_KEYS) <= report["environment"].keys()
    assert report["environment_after"] == report["environment"]
    assert report["diagnostic_complete"] is True
    assert report["checks"]["allocation_environment_matches"] is (changed_key is None)
    assert report["cuda_ready"] is (changed_key is None)
    if changed_key is None:
        startup.validate_report(
            report,
            worker_sha256=worker_sha,
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            scope="early_node",
        )
    else:
        assert report["environment"][changed_key] == value
        assert report["error"]["type"] == "ValueError"
        assert report["devices"] == []
