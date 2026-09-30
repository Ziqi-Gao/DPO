"""Independent completion validation consumes actual cgroup measurements."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("student_memory_contract", ROOT / "tools/sdsc_student_contract.py")
fixtures = load("memory_contract_fixture", Path(__file__).with_name("test_student_memory.py"))


@pytest.fixture
def report(tmp_path):
    proc, root, _ = fixtures.fixture(tmp_path)
    stages = {}
    for stage in ("initial", "after_export", "after_training", "after_validation"):
        stages[stage] = fixtures.memory.memory_envelope(
            expected_bytes=384 * fixtures.GIB, proc=proc, root=root
        )
    return {
        "job_id": "123456", "allocation": {
            "job_id": "123456", "cpus": 24, "world_size": 2, "memory_mib": 393216,
        },
        "memory_stages": stages,
        "initial_cgroup_memory": stages["initial"],
        "final_cgroup_memory": stages["after_validation"],
    }


def test_completed_calibration_binds_all_measured_stages(report):
    contract.validate_calibration_memory(report)


@pytest.mark.parametrize("change", ["limit", "missing_stage", "failed", "lost_counters", "aggregate_peak", "stage_binding", "job", "time", "mixed_path"])
def test_completed_calibration_rejects_wrong_or_missing_memory_evidence(report, change):
    report = copy.deepcopy(report)
    stages = report["memory_stages"]
    if change == "limit":
        report["allocation"]["memory_mib"] = 196608
    elif change == "missing_stage":
        del stages["after_training"]
    elif change == "failed":
        stages["after_training"]["passed"] = False
    elif change == "lost_counters":
        del stages["after_export"]["memory_stat"]
    elif change == "aggregate_peak":
        stages["after_training"]["ancestors"][2]["peak_bytes"] += fixtures.GIB
    elif change == "stage_binding":
        report["final_cgroup_memory"] = {}
    elif change == "job":
        report["allocation"]["job_id"] = "123457"
    elif change == "time":
        stages["after_training"]["observed_at_unix"] = 1
    elif change == "mixed_path":
        stages["after_training"]["path"] += "/other"
    with pytest.raises(ValueError):
        contract.validate_calibration_memory(report)


def test_same_job_can_change_selected_ancestor_after_an_initial_peak_tie(report):
    initial = report["memory_stages"]["initial"]
    step, job = initial["ancestors"][1:3]
    step["peak_bytes"] = job["peak_bytes"]
    for key in ("path", "limit_bytes", "current_bytes", "peak_bytes", "memory_stat", "memory_events", "memory_events_local", "memory_oom_control", "memory_failcnt"):
        initial[key] = step[key]
    initial["headroom_bytes"] = initial["limit_bytes"] - initial["peak_bytes"]
    assert initial["path"] != report["final_cgroup_memory"]["path"]
    contract.validate_calibration_memory(report)
