"""Four-H100 teacher fit transport gates; all tests run without SSH or GPUs."""

import ast
import copy
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


contract = module("fit_boundary_contract", REPO / "tools/sdsc_teacher_fit_contract.py")
client_fixture = module("fit_client_fixture", Path(__file__).with_name("test_cli.py"))
remote_fixture = module("fit_remote_fixture", Path(__file__).with_name("test_remote.py"))
cli, remote = client_fixture.cli, remote_fixture.remote


def manifest_fixture():
    return {
        "files": [
            {"path": path, "sha256": "%064x" % (index + 1)}
            for index, path in enumerate(contract.KERNEL_PATHS)
        ]
    }


def test_execution_plan_binds_every_named_kernel_and_excludes_unrelated_deployment_identity():
    manifest = manifest_fixture()
    expected = contract.execution_plan(manifest)
    manifest.update(run_id="a-new-deployment", code_sha256="f" * 64)
    manifest["files"].append({"path": "docs/unrelated.md", "sha256": "e" * 64})
    assert contract.execution_plan(manifest) == expected
    for index in range(len(contract.KERNEL_PATHS)):
        changed = manifest_fixture()
        changed["files"][index]["sha256"] = "d" * 64
        assert contract.sha256_value(contract.execution_plan(changed)) != contract.sha256_value(expected)


def test_execution_plan_rejects_missing_duplicate_and_bad_kernel_hash():
    changed = manifest_fixture()
    changed["files"].pop()
    with pytest.raises(ValueError, match="missing or invalid"):
        contract.execution_plan(changed)
    changed = manifest_fixture()
    changed["files"].append(changed["files"][0])
    with pytest.raises(ValueError, match="duplicate"):
        contract.execution_plan(changed)
    changed = manifest_fixture()
    changed["files"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="missing or invalid"):
        contract.execution_plan(changed)


def test_actual_modes_keep_exact_data_epochs_batch_updates_and_schedule():
    preflight = contract.actual_plan("preflight")
    fit = contract.actual_plan("full-fit")
    assert (preflight["train_examples"], preflight["dev_examples"], preflight["optimizer_steps"]) == (
        512,
        32,
        8,
    )
    assert (fit["train_examples"], fit["dev_examples"], fit["optimizer_steps"]) == (8192, 512, 512)
    for plan in (preflight, fit):
        assert plan["optimizer_steps"] * plan["global_batch_size"] == plan["train_examples"] * plan["epochs"]
        assert (plan["seed"], plan["raw_fit_seed_start"], plan["raw_dev_seed_start"]) == (
            271828,
            70000042,
            80000042,
        )
        assert plan["development_max_new_tokens"] == 256
        assert plan["complete_all_scheduled_optimizer_steps"] is True
        assert plan["formal_teacher_accepted"] is False
    assert (preflight["schedule"], preflight["warmup_steps"], preflight["checkpoint_steps"]) == (
        "constant",
        0,
        [8],
    )
    assert (fit["schedule"], fit["warmup_steps"], fit["checkpoint_steps"]) == (
        "cosine",
        16,
        [128, 256, 384, 512],
    )
    with pytest.raises(ValueError):
        contract.actual_plan("unreviewed-fit")
    fit["checkpoint_steps"].append(1024)
    assert contract.actual_plan("full-fit")["checkpoint_steps"] == [128, 256, 384, 512]


def write_value(path, value):
    raw = contract.canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return remote.digest(raw)


def file_record(root, path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": remote.digest(path.read_bytes()),
    }


def memory_fixture(job_id="12345", effective_limit=1024 * 1024**3):
    path = "/sys/fs/cgroup/memory/slurm/uid_543540/job_" + job_id
    budget = 192 * 1024**3
    current, peak = 32 * 1024**3, 64 * 1024**3
    return {
        "passed": True,
        "policy": "exclusive_node_workload_budget_v1",
        "job_id": job_id,
        "path": path,
        "step_path": path + "/step_batch",
        "budget_bytes": budget,
        "effective_limit_bytes": effective_limit,
        "current_bytes": current,
        "peak_bytes": peak,
        "headroom_bytes": budget - peak,
        "minimum_headroom_bytes": contract.FIXED_EXECUTION["allocation"]["minimum_host_headroom_bytes"],
        "kernel_headroom_bytes": effective_limit - peak,
        "kernel_cap_equals_budget": effective_limit == budget,
        "software_budget_only": effective_limit > budget,
        "ancestors": [
            {
                "path": path + "/step_batch",
                "limit_bytes": effective_limit,
                "current_bytes": current - 4096,
                "peak_bytes": peak - 4096,
            },
            {"path": path, "limit_bytes": effective_limit, "current_bytes": current, "peak_bytes": peak},
            {
                "path": str(Path(path).parent),
                "limit_bytes": None,
                "current_bytes": current + 4096,
                "peak_bytes": 1000 * 1024**3,
            },
        ],
    }


def test_exclusive_node_memory_keeps_budget_distinct_from_kernel_limit_and_ignores_uid_history():
    for limit in (192 * 1024**3, 1024 * 1024**3):
        value = memory_fixture(effective_limit=limit)
        contract.validate_memory(value, "12345")
        assert value["headroom_bytes"] == 128 * 1024**3
        assert value["software_budget_only"] is (limit > 192 * 1024**3)
    assert "tools/sdsc_teacher_fit_memory.py" in contract.KERNEL_PATHS
    assert contract.FIXED_EXECUTION["allocation"]["threads_per_rank"] == 6
    assert (
        contract.FIXED_EXECUTION["allocation"]["host_memory_scope"]
        == "workload_budget_not_claimed_kernel_cap"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(policy="legacy_exact_192"),
        lambda v: v.update(budget_bytes=1024 * 1024**3),
        lambda v: v.update(effective_limit_bytes=128 * 1024**3),
        lambda v: v.update(kernel_cap_equals_budget=True),
        lambda v: v.update(software_budget_only=False),
        lambda v: v.update(kernel_headroom_bytes=123),
        lambda v: v.update(job_id="another-job"),
        lambda v: v.update(job_id="54321"),
        lambda v: v.update(path=str(Path(v["path"]).parent)),
        lambda v: v.update(step_path=v["path"] + "/step_extern"),
        lambda v: v["ancestors"].pop(1),
        lambda v: v["ancestors"][0].update(current_bytes=0),
        lambda v: v["ancestors"][0].update(peak_bytes=2 * 1024**3),
        lambda v: v["ancestors"][0].update(path=v["path"] + "0/step_batch"),
        lambda v: v["ancestors"][2].update(limit_bytes=128 * 1024**3),
        lambda v: v["ancestors"].append(copy.deepcopy(v["ancestors"][0])),
    ],
)
def test_exclusive_node_memory_rejects_missing_or_forged_budget_job_and_raw_counters(mutation):
    value = memory_fixture()
    mutation(value)
    with pytest.raises(ValueError):
        contract.validate_memory(value, "12345")


def test_exclusive_node_memory_requires_own_finite_limit_and_preserves_192_gib_peak_gate():
    value = memory_fixture()
    for row in value["ancestors"][:2]:
        row["limit_bytes"] = None
    value["ancestors"][2]["limit_bytes"] = value["effective_limit_bytes"]
    with pytest.raises(ValueError, match="finite kernel boundary"):
        contract.validate_memory(value, "12345")
    value = memory_fixture()
    peak = 160 * 1024**3
    for row in value["ancestors"][:2]:
        row["peak_bytes"] = peak
    value.update(
        peak_bytes=peak,
        headroom_bytes=value["budget_bytes"] - peak,
        kernel_headroom_bytes=value["effective_limit_bytes"] - peak,
    )
    with pytest.raises(ValueError, match="headroom"):
        contract.validate_memory(value, "12345")


def test_each_rank_memory_binds_exact_report_job():
    rows = [{"rank": rank, "cgroup_memory": memory_fixture("54321")} for rank in range(4)]
    contract.validate_rank_memory(rows, "54321")
    rows[2]["cgroup_memory"] = memory_fixture("12345")
    with pytest.raises(ValueError, match="job identity"):
        contract.validate_rank_memory(rows, "54321")


@pytest.fixture
def isolated_fixture_dataset_pins(monkeypatch):
    """Tiny transport artifacts use test-only pins, never change production constants."""
    monkeypatch.setattr(contract, "EXPECTED_DATASETS", fixture_dataset_pins())


def fixture_dataset_pins():
    pins = {}
    for mode in ("preflight", "full-fit"):
        plan = contract.actual_plan(mode)
        dataset = {
            "teacher_fit": {"count": plan["train_examples"], "examples_sha256": "c" * 64},
            "teacher_dev": {"count": plan["dev_examples"], "examples_sha256": "d" * 64},
        }
        pins[mode] = {
            "dataset_manifest_sha256": contract.sha256_value(dataset),
            "teacher_fit_examples_sha256": "c" * 64,
            "teacher_dev_examples_sha256": "d" * 64,
        }
    return pins


def test_production_data_pins_bind_complete_reviewed_populations():
    assert (
        contract.actual_plan("preflight")["dataset_manifest_sha256"]
        == "208f279053605acfed08b4bbd027b23b23b3e39337ecd70dbc87455cd9814368"
    )
    assert (
        contract.actual_plan("full-fit")["dataset_manifest_sha256"]
        == "742b62a1ee328c8d8f660106265e4a342458fe5145243ecb08368eaa745f502d"
    )


def output_fixture(output, identity, execution, mode="preflight"):
    from posttrain_circuits.learning.teacher.adaptation_fit import FitPlan

    output.mkdir(parents=True, exist_ok=True)
    plan = contract.actual_plan(mode)
    dataset = {
        "teacher_fit": {"count": plan["train_examples"], "examples_sha256": "c" * 64},
        "teacher_dev": {"count": plan["dev_examples"], "examples_sha256": "d" * 64},
    }
    report = dict(
        identity,
        task=contract.task_for_mode(mode),
        mode=mode,
        artifact_kind="teacher_adaptation_fit_execution",
        passed=True,
        exit_code=0,
        exploratory=True,
        accepted_science=False,
        formal_teacher_accepted=False,
        full_teacher_ready=False,
        g0_passed=False,
        execution_class_certified=False,
        student_training_started=False,
        readiness_artifact_produced=False,
        teacher_training_started=True,
        training_started=True,
        world_size=4,
        execution_plan=execution,
        execution_plan_sha256=contract.sha256_value(execution),
        actual_plan=plan,
        actual_plan_sha256=contract.sha256_value(plan),
        adaptation_dataset=dataset,
        adaptation_dataset_sha256=contract.sha256_value(dataset),
        optimizer_steps=plan["optimizer_steps"],
        consumed_tokens=plan["optimizer_steps"] * 12345,
        consumed_sequences=plan["optimizer_steps"] * 64,
        completed_epochs=plan["epochs"],
        selection_does_not_stop_training=True,
        original_128_token_readiness_pass_claim=False,
        teacher_readiness_thresholds=contract.READINESS_THRESHOLDS,
        same_world_resume_performed_this_attempt=mode == "preflight",
        same_world_resume_prerequisite_job_id=None if mode == "preflight" else "12344",
        checks={name: True for name in contract.EXECUTION_CHECKS},
        allocation={
            "job_id": identity["job_id"],
            "cpus": 24,
            "requested_cpus_per_task": 24,
            "allocated_cpus_on_node": 72,
            "allocated_cpus_slurm_value": "72",
            "memory_mib": 196608,
            "cuda_visible_devices": "0,1,2,3",
        },
    )
    report["trainer_identity"] = contract.trainer_identity(mode, report["execution_plan_sha256"], "c" * 64)
    report["rank_summaries"] = [
        {
            "rank": rank,
            "cgroup_memory": memory_fixture(identity["job_id"]),
            "training": {
                "completed_updates": plan["optimizer_steps"],
                "completed_samples": plan["optimizer_steps"] * 64,
                "consumed_tokens": report["consumed_tokens"],
                "world_size": 4,
                "same_world_resume_verified": mode == "preflight",
            },
        }
        for rank in range(4)
    ]
    write_value(output / "data-isolation.json", {"passed": True})
    write_value(
        output / "train-metrics.jsonl", {"step": plan["optimizer_steps"], "tokens": report["consumed_tokens"]}
    )
    write_value(output / "progress.json", {"passed": False, "not_a_completion_report": True})
    checkpoints = []
    for step in plan["checkpoint_steps"]:
        directory = output / "checkpoints" / f"step-{step:06d}"
        directory.mkdir(parents=True)
        names = [
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "trainer.pt",
            "train-metrics.jsonl",
            *[f"rng-rank{rank}.pt" for rank in range(4)],
            *[f"fit-samples-rank{rank}.jsonl" for rank in range(4)],
        ]
        for name in names:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{step}/{name}: immutable fixture bytes".encode())
        metrics_sha = remote.digest((directory / "train-metrics.jsonl").read_bytes())
        trainer_plan = asdict(FitPlan.preflight() if mode == "preflight" else FitPlan())
        state = {
            "artifact_kind": "teacher_adapter_fit_checkpoint",
            "format_version": 1,
            "accepted_science": False,
            "formal_teacher_accepted": False,
            "metadata": {
                "identity": report["trainer_identity"],
                "plan": trainer_plan,
                "plan_sha256": contract.sha256_value(trainer_plan),
                "world_size": 4,
                "completed_updates": step,
                "completed_samples": step * 64,
                "consumed_tokens": step * 12345,
                "train_metrics_sha256": metrics_sha,
                "train_examples": plan["train_examples"],
                "total_updates": plan["optimizer_steps"],
            },
            "files": [file_record(directory, directory / name) for name in names],
        }
        state["sha256"] = contract.sha256_value(state)
        state_sha = write_value(directory / "manifest.json", state)
        for name in (
            "merged/config.json",
            "merged/tokenizer.json",
            "merged/tokenizer_config.json",
            "merged/model.safetensors",
        ):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{step}/{name}: dense fixture bytes".encode())
        dense = {
            "artifact_kind": "adapted_dense_teacher",
            "base_revision": contract.BASE_REVISION,
            "adaptation_plan_sha256": report["actual_plan_sha256"],
            "formal_teacher_accepted": False,
            "training_provenance": {
                **{key: report[key] for key in ("run_id", "job_id", "code_sha256")},
                "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
                "train_metrics_sha256": metrics_sha,
                "optimizer_steps": step,
                "consumed_tokens": step * 12345,
            },
            "files": [
                file_record(directory, path)
                for folder in ("merged", "adapter")
                for path in sorted((directory / folder).rglob("*"))
                if path.is_file()
            ],
        }
        dense["sha256"] = contract.sha256_value(dense)
        dense_sha = write_value(directory / "dense-manifest.json", dense)
        checks = {
            name: True
            for name in (
                "answer_accuracy",
                "exact_proof_accuracy",
                "first_rule_top1_accuracy",
                "intermediate_top1_accuracy",
                "topk_mass",
                "topk_target_coverage",
                "corrupted_prefix_recovery",
                "causal_shift",
            )
        }
        checks["causal_shift"] = step >= 256
        development = {
            "checks": checks,
            "metrics_passed": all(checks.values()),
            "thresholds": contract.READINESS_THRESHOLDS,
            "adapted_teacher_sha256": dense["sha256"],
            "evaluation_max_new_tokens": 256,
            "formal_teacher_accepted": False,
            "original_128_token_readiness_pass_claim": False,
        }
        dev_sha = write_value(directory / "dev-capability.json", development)
        checkpoints.append(
            {
                "step": step,
                "optimizer_steps": step,
                "directory": directory.relative_to(output).as_posix(),
                "consumed_tokens": step * 12345,
                "adapted_teacher_sha256": dense["sha256"],
                "dense_manifest_sha256": dense_sha,
                "trainer_state_manifest_sha256": state_sha,
                "training_metrics_sha256": metrics_sha,
                "dev_metrics_sha256": dev_sha,
                "dev_metrics_passed": development["metrics_passed"],
                "cgroup_memory_by_rank": [
                    {"rank": rank, "cgroup_memory": memory_fixture(identity["job_id"])} for rank in range(4)
                ],
            }
        )
    selected = (
        next((row for row in checkpoints if row["dev_metrics_passed"]), None) if mode == "full-fit" else None
    )
    report["selected_checkpoint_sha256"] = selected["adapted_teacher_sha256"] if selected else None
    write_value(
        output / "checkpoint-selection.json",
        {
            "selection_rule": plan["checkpoint_selection"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "selected_step": selected["step"] if selected else None,
            "formal_teacher_accepted": False,
            "original_128_token_readiness_pass_claim": False,
            "evaluation_max_new_tokens": 256,
        },
    )
    manifest = {
        "artifact_kind": "teacher_adaptation_checkpoint_set",
        "inventory_scope": "checkpoint_and_training_artifacts",
        "formal_teacher_accepted": False,
        "identity": {
            key: report[key]
            for key in ("run_id", "job_id", "code_sha256", "execution_plan_sha256", "actual_plan_sha256")
        },
        "base_revision": contract.BASE_REVISION,
        "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
        "checkpoints": checkpoints,
        "files": [
            file_record(output, path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "progress.json"
        ],
    }
    manifest["sha256"] = contract.sha256_value(manifest)
    report["checkpoint_manifest_sha256"] = write_value(output / "checkpoint-manifest.json", manifest)
    report["checkpoint_set_sha256"] = manifest["sha256"]
    write_value(output / "teacher-fit.json", report)
    for rank in ("validate", "0", "1", "2", "3"):
        write_value(output / f"memory-environment-{rank}.json", memory_fixture(identity["job_id"]))
    return report


@pytest.mark.parametrize("mode", ["preflight", "full-fit"])
def test_real_file_checkpoint_contract_and_nonaccepting_selection(
    tmp_path, mode, isolated_fixture_dataset_pins
):
    execution = contract.execution_plan(manifest_fixture())
    identity = dict(run_id="fit-fixture", job_id="12345", code_sha256="a" * 64)
    report = output_fixture(tmp_path, identity, execution, mode)
    contract.validate_report(report, execution, contract.task_for_mode(mode))
    records = contract.checkpoint_records(tmp_path, report)
    assert "checkpoint-manifest.json" in records
    assert "progress.json" not in records
    assert (
        report["selected_checkpoint_sha256"] is None
        if mode == "preflight"
        else report["selected_checkpoint_sha256"]
    )
    for changed in (
        dict(report, accepted_science=True),
        dict(report, formal_teacher_accepted=True),
        {key: value for key, value in report.items() if key != "formal_teacher_accepted"},
        dict(report, full_teacher_ready=True),
        dict(report, optimizer_steps=1),
        dict(report, actual_plan_sha256="f" * 64),
        dict(report, checks={"isolated_data": True}),
        dict(report, teacher_training_started=False),
        dict(report, same_world_resume_performed_this_attempt=mode != "preflight"),
        dict(report, teacher_readiness_thresholds={}),
        dict(report, allocation={}),
        dict(report, allocation=dict(report["allocation"], allocated_cpus_on_node=24)),
        dict(
            report,
            allocation=dict(report["allocation"], allocated_cpus_on_node=16, allocated_cpus_slurm_value="16"),
        ),
        dict(report, allocation=dict(report["allocation"], requested_cpus_per_task=72)),
        dict(report, allocation=dict(report["allocation"], cuda_visible_devices=None)),
    ):
        with pytest.raises(ValueError):
            contract.validate_report(changed, execution, contract.task_for_mode(mode))
    checkpoint = tmp_path / "checkpoints" / f"step-{contract.actual_plan(mode)['checkpoint_steps'][0]:06d}"
    (checkpoint / "train-metrics.jsonl").write_text("changed training metrics")
    with pytest.raises(ValueError):
        contract.checkpoint_records(tmp_path, report)


class FitClientTests(unittest.TestCase):
    def setUp(self):
        self.fixture = client_fixture.ClientTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        manifest = dict(manifest_fixture(), run_id="run-test", code_sha256="a" * 64)
        for row in manifest["files"]:
            if row["path"] == "tools/sdsc_teacher_fit_contract.py":
                row["sha256"] = cli.sha((REPO / row["path"]).read_bytes())
        cli.write_json(cli.state_root() / "runs/run-test.json", {"state": "deployed", "manifest": manifest})

    def arguments(self, mode):
        args = self.fixture.preflight_args(True)
        args.task = contract.task_for_mode(mode)
        args.gpus, args.partition, args.qos = 4, "nairr-gpu", "nairr-gpu-normal"
        args.teacher_job_id = "54493015"
        args.preflight_job_id = None if mode == "preflight" else "12344"
        args.time = "01:00:00" if mode == "preflight" else "04:00:00"
        return args

    def test_exact_four_rank_command_and_mode_specific_grace_no_remote_dryrun(self):
        for mode, grace in (("preflight", 300), ("full-fit", 900)):
            with (
                self.subTest(mode=mode),
                patch.object(cli, "remote") as remote_call,
                redirect_stdout(io.StringIO()),
            ):
                cli.submit_command(self.arguments(mode))
            remote_call.assert_not_called()
            plan = cli.read_json(cli.state_root() / "submit-plan.json")
            self.assertIn("--gpus=h100:4", plan["sbatch_argv"])
            self.assertIn(f"--signal=B:TERM@{grace}", plan["sbatch_argv"])
            self.assertEqual(plan["sbatch_argv"][-1], mode)
            index = next(
                i for i, value in enumerate(plan["sbatch_argv"]) if value.endswith("/sdsc_teacher_fit_job.sh")
            )
            self.assertEqual(len(plan["sbatch_argv"][index + 1 :]), 8)

    def test_wrong_resource_or_missing_prerequisite_fails_locally(self):
        for mode, key, value in (
            ("preflight", "gpus", 1),
            ("preflight", "time", "01:00:01"),
            ("full-fit", "time", "24:00:01"),
            ("full-fit", "preflight_job_id", None),
            ("preflight", "teacher_job_id", "54345604"),
        ):
            args = self.arguments(mode)
            setattr(args, key, value)
            with self.subTest(mode=mode, key=key), self.assertRaises(cli.UserError):
                cli.submit_command(args)


class FitRemoteTests(unittest.TestCase):
    def setUp(self):
        self.fixture = remote_fixture.RemoteBoundaryTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.enable_preflight()
        self.request = self.fixture.request
        pins = fixture_dataset_pins()
        pin_patch = patch.object(contract, "EXPECTED_DATASETS", pins)
        pin_patch.start()
        self.addCleanup(pin_patch.stop)
        self.request.update(
            task=contract.task_for_mode("preflight"), teacher_job_id="54493015", preflight_job_id=None
        )
        self.request["resources"].update(
            gpus=4, partition="nairr-gpu", qos="nairr-gpu-normal", time="01:00:00"
        )
        files = {name: b"# immutable kernel fixture\n" for name in contract.KERNEL_PATHS}
        files["tools/sdsc_teacher_fit_job.sh"] = b"#!/bin/bash\nexit 0\n"
        files["tools/sdsc_teacher_fit_contract.py"] = (
            (REPO / "tools/sdsc_teacher_fit_contract.py").read_bytes()
            + b"\nEXPECTED_DATASETS = "
            + repr(pins).encode()
            + b"\n"
        )
        remote.upload(self.request, self.fixture.archive(files))
        _, manifest = remote.release_manifest(self.fixture.root, self.request)
        self.execution = contract.execution_plan(manifest)
        self.request.update(
            execution_plan_sha256=contract.sha256_value(self.execution),
            actual_plan_sha256=contract.sha256_value(contract.actual_plan("preflight")),
        )

    def full_fit(self):
        self.request.update(
            task=contract.task_for_mode("full-fit"),
            preflight_job_id="12344",
            actual_plan_sha256=contract.sha256_value(contract.actual_plan("full-fit")),
        )
        self.request["resources"]["time"] = "04:00:00"

    def upstream(self, root, job_id, task, role):
        return {
            "receipt": {"run_id": "prior-" + job_id, "job_id": job_id, "task": task},
            "report": {"execution_plan_sha256": self.request["execution_plan_sha256"]},
            "status": {"state": "COMPLETED", "success": True},
        }

    def submit(self, queue=""):
        def command(argv, **kwargs):
            return {
                "returncode": 0,
                "stdout": queue if argv[0] == "squeue" else "12345;expanse\n",
                "stderr": "",
            }

        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "upstream_evidence", side_effect=self.upstream),
            patch.object(remote, "run", side_effect=command) as run,
        ):
            result = remote.submit(self.request)
        self.assertEqual(run.call_args_list[-1].args[0][0], "sbatch")
        self.assertEqual(run.call_args_list[-1].args[0][-1], remote.TEACHER_FIT_TASKS[self.request["task"]])
        self.assertEqual(run.call_count, 2)
        return result

    def publication(self, receipt):
        root = Path(receipt["result_dir"])
        identity = {key: receipt[key] for key in ("run_id", "job_id", "code_sha256", "hf_home")}
        mode = remote.TEACHER_FIT_TASKS[receipt["task"]]
        report = output_fixture(root / "artifacts", identity, self.execution, mode)
        report.update({key: receipt[key] for key in remote.PREREQUISITE_BINDINGS})
        raw = remote.canonical(report)
        (root / "teacher-fit.json").write_bytes(raw)
        records = [
            file_record(root, path) for path in sorted((root / "artifacts").rglob("*")) if path.is_file()
        ]
        records.append(dict(path="teacher-fit.json", size=len(raw), sha256=remote.digest(raw)))
        remote.atomic_json(
            root / "receipt.json",
            dict(
                identity,
                task=receipt["task"],
                passed=True,
                persisted=True,
                persistent_read_back_verified=True,
                files=records,
            ),
        )
        return report

    def test_four_gpu_preflight_requires_fixed_one_gpu_proof_and_publication(self):
        receipt = self.submit()
        self.publication(receipt)
        self.assertTrue(remote.verify_result(receipt)["verified"])
        proof = json.loads(Path(receipt["prerequisites_path"]).read_text())
        self.assertEqual(proof["teacher_job_id"], "54493015")
        self.assertIsNone(proof["preflight_job_id"])

    def test_full_fit_binds_matching_new_four_gpu_preflight_and_complete_checkpoint_set(self):
        self.full_fit()
        receipt = self.submit()
        report = self.publication(receipt)
        self.assertTrue(remote.verify_result(receipt)["verified"])
        self.assertEqual(report["same_world_resume_prerequisite_job_id"], receipt["preflight_job_id"])
        (Path(receipt["result_dir"]) / "artifacts/checkpoint-manifest.json").write_text("{}")
        self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_missing_upstream_blocks_before_claim_or_slurm(self):
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "upstream_evidence", side_effect=ValueError("upstream failed")),
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "upstream failed"),
        ):
            remote.submit(self.request)
        run.assert_not_called()
        self.assertFalse((self.fixture.root / "run-claims").exists())

    def test_changed_intended_kernel_or_different_preflight_blocks_before_claim(self):
        self.full_fit()

        def different(*args):
            result = self.upstream(*args)
            result["report"]["execution_plan_sha256"] = "f" * 64
            return result

        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "upstream_evidence", side_effect=different),
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            remote.submit(self.request)
        run.assert_not_called()
        self.assertFalse((self.fixture.root / "run-claims").exists())
        self.request["execution_plan_sha256"] = "f" * 64
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "upstream_evidence") as upstream,
            self.assertRaisesRegex(ValueError, "intended release"),
        ):
            remote.submit(self.request)
        upstream.assert_not_called()

    def test_existing_gpu_or_unknown_queue_blocks_before_claim_or_sbatch(self):
        for queue in ("54493015|RUNNING|gres:gpu:h100:1\n", "malformed\n"):
            with (
                self.subTest(queue=queue),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "upstream_evidence", side_effect=self.upstream),
                patch.object(
                    remote, "run", return_value={"returncode": 0, "stdout": queue, "stderr": ""}
                ) as run,
                self.assertRaises(ValueError),
            ):
                remote.submit(self.request)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], "squeue")
            self.assertFalse((self.fixture.root / "run-claims").exists())

    def test_fetch_keeps_weights_and_rank_state_remote(self):
        receipt = self.submit()
        self.publication(receipt)
        self.request["job_id"] = receipt["job_id"]
        result = remote.fetch(self.request)
        names = {row["path"] for row in result["files"]}
        self.assertEqual(names, remote.TEACHER_FIT_SMALL_RESULTS | {"teacher-fit.json", "receipt.json"})
        self.assertFalse(any(name.endswith((".pt", ".safetensors")) for name in names))

    def test_failed_validate_only_memory_evidence_is_fetchable_without_large_artifacts(self):
        receipt = self.submit()
        root = Path(receipt["result_dir"])
        raw = dict(memory_fixture(receipt["job_id"]), passed=False, error="actual finite memory unknown")
        write_value(root / "artifacts/memory-environment-validate.json", raw)
        write_value(root / "teacher-fit.json", dict(passed=False, failure_stage="validate-only"))
        self.request["job_id"] = receipt["job_id"]
        result = remote.fetch(self.request)
        names = {row["path"] for row in result["files"]}
        assert "memory-environment-validate.json" in names
        assert "teacher-fit.json" in names
        assert remote.TEACHER_FIT_SMALL_RESULTS == cli.TEACHER_FIT_SMALL_RESULTS


def wrapper_functions():
    path = REPO / "tools/sdsc_teacher_fit_job.sh"
    source = path.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    namespace = {}
    exec(
        compile(
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)
                ],
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_wrapper_publishes_verified_checkpoints_and_bounds_failure(tmp_path, isolated_fixture_dataset_pins):
    helpers = wrapper_functions()
    identity = dict(
        task=contract.task_for_mode("preflight"), run_id="fit-fixture", job_id="12345", code_sha256="a" * 64
    )
    execution = contract.execution_plan(manifest_fixture())
    output, persistent = tmp_path / "output", tmp_path / "persistent"
    persistent.mkdir()
    report = output_fixture(output, identity, execution)
    helpers.update(
        identity=identity,
        fit_task=identity["task"],
        fit_contract=contract,
        expected_execution_plan=execution,
        result_root=persistent,
        process=None,
    )
    helpers["validate_report"](report, 0)
    with redirect_stdout(io.StringIO()):
        helpers["publish"](output, None, report, True)
    receipt = json.loads((persistent / "receipt.json").read_text())
    assert receipt["persisted"] and receipt["persistent_read_back_verified"]
    for row in receipt["files"]:
        assert remote.digest((persistent / row["path"]).read_bytes()) == row["sha256"]


def test_wrapper_rejects_same_size_weight_mutation_and_preserves_failure_without_checkpoint(
    tmp_path, isolated_fixture_dataset_pins
):
    helpers = wrapper_functions()
    execution = contract.execution_plan(manifest_fixture())
    identity = dict(run_id="fit-fixture", job_id="12345", code_sha256="a" * 64)
    report = output_fixture(tmp_path / "output", identity, execution)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    helpers.update(identity=identity, fit_contract=contract, result_root=persistent, process=None)
    weight = tmp_path / "output/checkpoints/step-000008/merged/model.safetensors"
    weight.write_bytes(b"x" * weight.stat().st_size)
    with redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="content differs"):
        helpers["publish"](tmp_path / "output", None, report, True)
    assert not (persistent / "receipt.json").exists()
    failure = tmp_path / "failure"
    failure.mkdir()
    helpers["result_root"] = failure
    with redirect_stdout(io.StringIO()):
        helpers["publish"](None, None, {"passed": False}, False)
    assert json.loads((failure / "receipt.json").read_text())["passed"] is False


def test_fit_wrapper_retains_actual_term_tail_and_bounds_native_shutdown(tmp_path):
    fixture = module("fit_signal_fixture", Path(__file__).with_name("test_teacher_probe_wrapper.py"))
    with patch.object(fixture, "SCRIPT", REPO / "tools/sdsc_teacher_fit_job.sh"):
        tail = tmp_path / "tail"
        tail.mkdir()
        fixture.test_interrupted_wrapper_preserves_shutdown_pipe_after_original_log_closes(tail)
        native = tmp_path / "native"
        native.mkdir()
        fixture.test_native_block_forces_kill_and_preserves_faulthandler_stderr(native)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_world",
        "wrong_steps",
        "missing_rng",
        "wrong_dense_plan",
        "late_selection",
        "missing_rank_memory",
        "low_headroom",
    ],
)
def test_semantic_checkpoint_mismatch_rejected_after_outer_hashes_rebound(
    tmp_path, isolated_fixture_dataset_pins, mutation
):
    execution = contract.execution_plan(manifest_fixture())
    identity = dict(run_id="fit-fixture", job_id="12345", code_sha256="a" * 64)
    report = output_fixture(tmp_path, identity, execution, "full-fit")
    manifest = json.loads((tmp_path / "checkpoint-manifest.json").read_text())
    checkpoint = manifest["checkpoints"][0]
    directory = tmp_path / checkpoint["directory"]
    if mutation in {"wrong_world", "wrong_steps", "missing_rng"}:
        path = directory / "manifest.json"
        changed = json.loads(path.read_text())
        if mutation == "wrong_world":
            changed["metadata"]["world_size"] = 3
        elif mutation == "wrong_steps":
            changed["metadata"]["completed_updates"] = 127
        else:
            changed["files"] = [row for row in changed["files"] if row["path"] != "rng-rank3.pt"]
        changed["sha256"] = contract.sha256_value(
            {key: value for key, value in changed.items() if key != "sha256"}
        )
        checkpoint["trainer_state_manifest_sha256"] = write_value(path, changed)
    elif mutation == "wrong_dense_plan":
        path = directory / "dense-manifest.json"
        changed = json.loads(path.read_text())
        changed["adaptation_plan_sha256"] = "f" * 64
        changed["sha256"] = contract.sha256_value(
            {key: value for key, value in changed.items() if key != "sha256"}
        )
        checkpoint["dense_manifest_sha256"] = write_value(path, changed)
        checkpoint["adapted_teacher_sha256"] = changed["sha256"]
    elif mutation == "late_selection":
        path = tmp_path / "checkpoint-selection.json"
        changed = json.loads(path.read_text())
        chosen = manifest["checkpoints"][-1]
        changed.update(
            selected_step=chosen["step"], selected_checkpoint_sha256=chosen["adapted_teacher_sha256"]
        )
        report["selected_checkpoint_sha256"] = chosen["adapted_teacher_sha256"]
        write_value(path, changed)
    elif mutation == "missing_rank_memory":
        checkpoint["cgroup_memory_by_rank"].pop()
        path = directory / "manifest.json"
    else:
        checkpoint["cgroup_memory_by_rank"][0]["cgroup_memory"]["headroom_bytes"] = 1
        path = directory / "manifest.json"
    relative = path.relative_to(tmp_path).as_posix()
    manifest["files"] = [
        file_record(tmp_path, path) if row["path"] == relative else row for row in manifest["files"]
    ]
    manifest["sha256"] = contract.sha256_value(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    report["checkpoint_manifest_sha256"] = write_value(tmp_path / "checkpoint-manifest.json", manifest)
    report["checkpoint_set_sha256"] = manifest["sha256"]
    with pytest.raises(ValueError):
        contract.checkpoint_records(tmp_path, report)
