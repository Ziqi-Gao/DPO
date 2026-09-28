"""Real cgroup file fixtures distinguish allocation limits from workload budgets."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_fit_memory", Path(__file__).resolve().parents[2] / "tools/sdsc_teacher_fit_memory.py"
)
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)


def tree(tmp_path, *, version=1, limit=1024 * 1024**3, peak=80 * 1024**3):
    root = tmp_path / "cgroup"
    base = root / "memory" if version == 1 else root
    own = base / "slurm/uid_543540/job_54493777"
    step = own / "step_batch"
    leaf = step / "task_0"
    leaf.mkdir(parents=True)
    names = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
        if version == 1
        else ("memory.max", "memory.current", "memory.peak")
    )

    def write(path, maximum, current, highwater):
        path.mkdir(parents=True, exist_ok=True)
        for name, value in zip(names, (maximum, current, highwater), strict=True):
            (path / name).write_text(str(value))

    write(own, limit, 20 * 1024**3, peak)
    write(step, limit, 10 * 1024**3, peak - 1)
    write(leaf, 2**63 - 4096 if version == 1 else "max", 1024, 2048)
    # Historical uid usage belongs to other jobs, never to this workload.
    write(own.parent, 2**63 - 4096 if version == 1 else "max", 30 * 1024**3, 900 * 1024**3)
    proc = tmp_path / "proc-cgroup"
    prefix = "5:memory:" if version == 1 else "0::"
    proc.write_text(prefix + "/slurm/uid_543540/job_54493777/step_batch/task_0\n")
    return root, proc, own, step, names, write


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("limit", [192 * 1024**3, 1024 * 1024**3])
def test_true_kernel_limit_and_own_workload_peak_are_reported_separately(tmp_path, version, limit):
    root, proc, own, step, *_ = tree(tmp_path, version=version, limit=limit)
    evidence = tmp_path / "evidence.json"
    actual = memory.memory_envelope(job_id="54493777", proc=proc, root=root, evidence_path=evidence)
    assert actual["path"] == str(own) and actual["step_path"] == str(step)
    assert actual["effective_limit_bytes"] == limit
    assert actual["peak_bytes"] == 80 * 1024**3  # not the uid's historical900GiB
    assert actual["headroom_bytes"] == 112 * 1024**3
    assert actual["minimum_headroom_bytes"] == memory.HEADROOM
    assert actual["kernel_cap_equals_budget"] is (limit == memory.BUDGET)
    assert actual["software_budget_only"] is (limit > memory.BUDGET)
    assert json.loads(evidence.read_text()) == actual


@pytest.mark.parametrize(
    "fault",
    [
        "small",
        "headroom",
        "zero",
        "peak",
        "missing",
        "wrong_job",
        "wrong_step",
        "own_unlimited",
        "ancestor_limit",
    ],
)
def test_missing_or_inadequate_memory_evidence_fails_and_preserves_raw_counters(tmp_path, fault):
    root, proc, own, step, names, write = tree(tmp_path)
    if fault == "small":
        write(own, 128 * 1024**3, 20 * 1024**3, 80 * 1024**3)
    elif fault == "headroom":
        write(own, 1024 * 1024**3, 100 * 1024**3, 160 * 1024**3)
    elif fault == "zero":
        write(step, 1024 * 1024**3, 0, 80 * 1024**3)
    elif fault == "peak":
        write(own, 1024 * 1024**3, 100 * 1024**3, 80 * 1024**3)
    elif fault == "missing":
        (step / names[2]).unlink()
    elif fault == "wrong_job":
        proc.write_text(proc.read_text().replace("54493777", "12345"))
    elif fault == "wrong_step":
        proc.write_text(proc.read_text().replace("step_batch", "step_extern"))
    elif fault == "own_unlimited":
        for path in (own, step):
            (path / names[0]).write_text(str(2**63 - 4096))
        write(own.parent, 1024 * 1024**3, 50 * 1024**3, 900 * 1024**3)
    else:
        write(own.parent, 128 * 1024**3, 30 * 1024**3, 100 * 1024**3)
    evidence = tmp_path / "failure.json"
    with pytest.raises(ValueError):
        memory.memory_envelope(job_id="54493777", proc=proc, root=root, evidence_path=evidence)
    captured = json.loads(evidence.read_text())
    assert captured["passed"] is False and captured["error"]
    assert captured["budget_bytes"] == memory.BUDGET
    if fault not in {"wrong_job", "wrong_step"}:
        assert len(captured["ancestors"]) >= 4


def test_historical_shared_guard_remains_strict_192_gib(tmp_path):
    original_spec = importlib.util.spec_from_file_location(
        "_shared_memory_guard", Path(__file__).resolve().parents[2] / "tools/sdsc_training_preflight.py"
    )
    original = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original)
    root, proc, *_ = tree(tmp_path)
    with pytest.raises(ValueError, match="finite 192 GiB and real peak"):
        original.memory_envelope(proc=proc, root=root)


@pytest.mark.parametrize("adequate", [True, False])
def test_first_validate_only_creates_output_and_retains_actual_memory_evidence(
    tmp_path, monkeypatch, adequate
):
    root, proc, *_ = tree(tmp_path, limit=(1024 if adequate else 128) * 1024**3)
    work = tmp_path / "node-work"
    home = work / "hf-home"
    home.mkdir(parents=True)
    output = work / "outputs/qwen3-v2"
    args = SimpleNamespace(work_dir=work, output_dir=output, hf_home=home, science_root=work / "source")
    guards = SimpleNamespace(real_path=lambda path: Path(path).resolve())
    monkeypatch.delenv("RANK", raising=False)
    actual_memory = memory.memory_envelope
    monkeypatch.setattr(
        memory, "memory_envelope", lambda **kw: actual_memory(job_id="54493777", proc=proc, root=root, **kw)
    )
    monkeypatch.setattr(
        memory.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(stdout=json.dumps({"filesystems": [{"fstype": "ext4"}]})),
    )
    monkeypatch.setattr(memory, "pinned_cache", lambda _home: {"verified_fixture": True})
    assert not output.exists()
    if adequate:
        observed = memory.local_environment(args, guards)
        assert observed["cgroup_memory"]["effective_limit_bytes"] == 1024 * 1024**3
        assert memory.os.environ["OMP_NUM_THREADS"] == "6"
    else:
        with pytest.raises(ValueError, match="smaller than"):
            memory.local_environment(args, guards)
    report = json.loads((output / "memory-environment-validate.json").read_text())
    assert report["passed"] is adequate
    assert report["ancestors"]
