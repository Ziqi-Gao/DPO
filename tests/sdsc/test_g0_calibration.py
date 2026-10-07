"""Candidate planning guards only: no models, GPUs, SSH, or Slurm commands."""

import contextlib
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_g0_calibration.py"
SPEC = importlib.util.spec_from_file_location("sdsc_g0_calibration", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def memory_fixture():
    limit = 192 * 1024**3
    current, peak = 40 * 1024**3, 80 * 1024**3
    selected = {"path": "/job", "limit_bytes": limit, "current_bytes": current, "peak_bytes": peak}
    return {
        **selected,
        "headroom_bytes": limit - peak,
        "minimum_headroom_bytes": math.ceil(limit * 0.2),
        "passed": True,
        "ancestors": [selected, {"path": "/", "limit_bytes": None}],
    }


def preflight_fixture():
    guards = worker.helper("sdsc_training_preflight")
    report = {
        "kind": "sdsc_h100_training_preflight_candidate_v1",
        "task": "qwen3-v2-preflight",
        "passed": True,
        "exit_code": 0,
        "world_size": 2,
        "g0_passed": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate": False,
        "job_id": "12345",
        "run_id": "real-preflight",
        "code_sha256": "a" * 64,
    }
    contract = {"world_size": 2, "samples_by_rank": [32, 32], "global_logical_batch_size": 64}
    reports = []
    for rank in range(2):
        reports.append(
            {
                **report,
                "allocation": {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 2,
                    "threads_per_rank": 12,
                    "job_id": "12345",
                    "cuda_visible_devices": "GPU-a,GPU-b",
                },
                "runtime": {"python": "3.12.13", "packages": guards.DEPENDENCIES},
                "pinned_cache": {model: {"revision": revision} for model, revision in guards.MODELS.items()},
                "gpu": {
                    "name": "NVIDIA H100 80GB HBM3",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 80 * 1024**3,
                    "logical_device": rank,
                },
                "nccl_all_reduce": True,
                "finite_gradients": True,
                "parameter_update_nonzero": True,
                "full_state_optimizer_scheduler_rng_restore": True,
                "local_parameter_sha256_before": "b" * 64,
                "local_parameter_sha256_after": "c" * 64,
                "losses": [0.5] * 8,
                "batch_contract": contract,
                "local_global_slots": list(range(rank, 64, 2)),
                "reserved_global_nonpadding_tokens": 64 * 1536,
                "fsdp": {
                    "requested_fsdp_sharding_strategy": "FULL_SHARD",
                    "effective_fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_wrapper_count": 29,
                },
                "rank_zero_teacher": {"finite": True, "revision": guards.MODELS["Qwen/Qwen3-8B"]},
                "checkpoint": {
                    "world_size": 2,
                    "all_files_verified_before_restore": True,
                    "local_restored_parameter_sha256": "c" * 64,
                    "local_restored_optimizer_sha256": "d" * 64,
                    "files": [
                        {"path": path, "size": 1024, "sha256": "e" * 64}
                        for path in guards.checkpoint_names(2)
                    ],
                },
                "node_local_mount": {"fstype": "ext4"},
                "initial_cgroup_memory": memory_fixture(),
                "cgroup_memory": memory_fixture(),
                "peak_gpu_allocated_bytes": 20 * 1024**3,
                "peak_gpu_reserved_bytes": 40 * 1024**3,
                "node": "h100-test-node",
            }
        )
    report["ranks"] = reports
    return report, contract


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(
            science_root=self.root / "science",
            work_dir=self.root,
            output_dir=self.root / "qwen3-v2/out",
            teacher_root=self.root / "qwen3-v2/teacher",
            hf_home=self.root / "huggingface",
            python=Path("/runtime/bin/python3.12"),
        )

    def test_unexported_checkpoint_produces_no_runnable_training_command(self):
        plan = worker.build_plan(self.args, ["experiment=canonical_sft"])
        self.assertIsNone(plan["train_argv"])
        self.assertIsNone(plan["runtime_initial_checkpoint_sha256"])
        self.assertFalse(plan["execution_enabled"])
        self.assertFalse(plan["g0_passed"])
        self.assertFalse(plan["factorial_ready"])

    def test_actual_hash_is_separate_from_base_hash_and_argv_is_literal(self):
        self.args.output_dir = self.root / "qwen3-v2/out;$(touch NO)"
        overrides = ["experiment=canonical_sft", *worker.storage_overrides(self.args)]
        plan = worker.build_plan(self.args, overrides, initial_checkpoint_sha256="f" * 64)
        self.assertEqual(plan["base_science_config_sha256"], worker.BASE_SCIENCE_SHA256)
        self.assertEqual(plan["runtime_initial_checkpoint_sha256"], "f" * 64)
        self.assertFalse(plan["initial_checkpoint_hash_is_inherited_science_acceptance"])
        argv = plan["train_argv"]
        self.assertEqual(argv[argv.index("--num_processes") + 1], "2")
        self.assertEqual(argv[argv.index("--main_process_port") + 1], "0")
        self.assertIn(str(self.args.output_dir / "canonical_sft"), argv)
        self.assertIn("production_safety.initial_checkpoint_hash=" + "f" * 64, argv)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", plan["environment_updates"])
        self.assertEqual(plan["preserve_environment_exactly"], ["CUDA_VISIBLE_DEVICES"])
        self.assertFalse((self.root / "NO").exists())

    def test_initial_checkpoint_hash_cannot_be_placeholder(self):
        with self.assertRaisesRegex(ValueError, "initial checkpoint SHA"):
            worker.build_plan(self.args, [], initial_checkpoint_sha256="TODO")

    def test_current_conda_python_alias_resolves_but_other_interpreter_is_rejected(self):
        alias = self.root / "python"
        alias.symlink_to(Path(sys.executable).resolve())
        self.assertEqual(worker.resolved_interpreter(alias), Path(sys.executable).resolve())
        with self.assertRaisesRegex(ValueError, "interpreter differs"):
            worker.resolved_interpreter(self.root / "other-python")

    def test_execute_refuses_before_io_or_scientific_import(self):
        argv = []
        for name in (
            "science-root",
            "work-dir",
            "output-dir",
            "hf-home",
            "provenance-manifest",
            "teacher-root",
            "teacher-report",
            "preflight-report",
            "preflight-snapshot-manifest",
        ):
            argv.extend(["--" + name, str(self.root / "absent")])
        for name in (
            "provenance-manifest-sha256",
            "teacher-report-sha256",
            "preflight-report-sha256",
            "code-sha256",
        ):
            argv.extend(["--" + name, "a" * 64])
        argv.extend(["--run-id", "candidate", "--execute-calibration"])
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(worker, "helper", side_effect=AssertionError("unexpected input/model IO")),
            self.assertRaisesRegex(ValueError, "real Slurm allocation required"),
        ):
            worker.main(argv)

    def test_report_read_is_hash_anchored_and_rejects_duplicate_keys(self):
        path = self.root / "report.json"
        path.write_text('{"passed":true}')
        self.assertTrue(worker.json_document(path, worker.sha(path.read_bytes()))["passed"])
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            worker.json_document(path, "a" * 64)
        path.write_text('{"passed":false,"passed":true}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            worker.json_document(path)

    def test_report_symlinks_are_rejected(self):
        path = self.root / "report.json"
        path.write_text("{}")
        link = self.root / "linked.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.json_document(link)

    def test_snapshot_binds_actual_science_bytes_across_different_wrappers(self):
        scientific = {"path": "src/science.py", "sha256": "a" * 64, "size": 20, "mode": 420}
        files = [scientific, {"path": "tools/wrapper.py", "sha256": "b" * 64, "size": 30, "mode": 420}]
        snapshot = {
            "schema": "quest-sdsc-snapshot-v1",
            "git_head": "c" * 40,
            "run_id": "preflight",
            "code_sha256": worker.sha(worker.canonical(files)),
            "files": files,
        }
        report = {"run_id": snapshot["run_id"], "code_sha256": snapshot["code_sha256"]}
        provenance = {"git_head": "c" * 40, "scientific_files": [scientific]}
        worker.validate_snapshot(snapshot, report, provenance)
        wrong = copy.deepcopy(provenance)
        wrong["scientific_files"][0]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "scientific file bytes"):
            worker.validate_snapshot(snapshot, report, wrong)
        wrong = copy.deepcopy(snapshot)
        wrong["files"][0]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            worker.validate_snapshot(wrong, report, provenance)
        wrong = dict(snapshot, git_head="f" * 40)
        with self.assertRaisesRegex(ValueError, "HEAD differs"):
            worker.validate_snapshot(wrong, report, provenance)

    def test_passed_two_rank_evidence_has_no_certification_side_effect(self):
        report, contract = preflight_fixture()
        actual = worker.validate_preflight(report, contract)
        self.assertTrue(actual["report_validated"])
        self.assertEqual(actual["world_size"], 2)
        self.assertNotIn("execution_class_certified", actual)

    def test_preflight_rejects_wrong_runtime_hardware_slots_and_resume(self):
        mutations = [
            ("runtime", "python", "3.12.3"),
            ("gpu", "compute_capability", [12, 0]),
            ("allocation", "cuda_visible_devices", "GPU-b,GPU-a"),
            ("checkpoint", "local_restored_parameter_sha256", "f" * 64),
            ("fsdp", "effective_fsdp_sharding_strategy", "NO_SHARD"),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                report, contract = preflight_fixture()
                report["ranks"][1][section][key] = value
                with self.assertRaises(ValueError):
                    worker.validate_preflight(report, contract)
        report, contract = preflight_fixture()
        report["ranks"][1]["local_global_slots"] = list(range(32))
        with self.assertRaisesRegex(ValueError, "batch/token"):
            worker.validate_preflight(report, contract)

    def test_preflight_rejects_missing_rank_failed_teacher_or_checkpoint_file(self):
        report, contract = preflight_fixture()
        report["ranks"].pop()
        with self.assertRaisesRegex(ValueError, "both rank"):
            worker.validate_preflight(report, contract)
        report, contract = preflight_fixture()
        report["ranks"][0]["rank_zero_teacher"]["finite"] = False
        with self.assertRaisesRegex(ValueError, "teacher forward"):
            worker.validate_preflight(report, contract)
        report, contract = preflight_fixture()
        report["ranks"][0]["checkpoint"]["files"].pop()
        with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
            worker.validate_preflight(report, contract)

    def test_preflight_rejects_unverified_parent_memory_and_forged_headroom(self):
        report, contract = preflight_fixture()
        report["ranks"][0]["cgroup_memory"]["ancestors"].append(
            {"path": "/parent", "limit_bytes": 128 * 1024**3}
        )
        with self.assertRaisesRegex(ValueError, "ancestor"):
            worker.validate_preflight(report, contract)
        report, contract = preflight_fixture()
        report["ranks"][0]["cgroup_memory"]["peak_bytes"] = 190 * 1024**3
        with self.assertRaisesRegex(ValueError, "memory/headroom"):
            worker.validate_preflight(report, contract)

    def test_teacher_failure_or_different_science_refused_before_bank_import(self):
        with self.assertRaisesRegex(ValueError, "teacher report"):
            worker.validate_teacher({"passed": False}, self.args, {}, {"code_commit": "a" * 40}, None)

    def test_upstream_prerequisites_bind_receipts_and_real_accounting(self):
        self.args.run_id, self.args.code_sha256 = "new-calibration", "f" * 64
        self.args.prerequisites = self.root / "prerequisites.json"
        reports = {}
        upstream = {}
        for name, job_id, filename in (
            ("teacher", "111", "teacher-prepare.json"),
            ("preflight", "222", "preflight.json"),
        ):
            report = {"job_id": job_id, "run_id": name + "-run", "code_sha256": "a" * 64}
            reports[name] = report
            report_hash = worker.sha(worker.canonical(report))
            setattr(self.args, name + "_report_sha256", report_hash)
            upstream[name] = {
                "report": report,
                "report_sha256": report_hash,
                "receipt": report,
                "publication_receipt_sha256": "b" * 64,
                "publication_receipt": {
                    **report,
                    "passed": True,
                    "persisted": True,
                    "persistent_read_back_verified": True,
                    "files": [{"path": filename, "sha256": report_hash}],
                },
                "status": {
                    "success": True,
                    "state": "COMPLETED",
                    "job_id": job_id,
                    "run_id": report["run_id"],
                    "result": {"verified": True, "result_sha256": report_hash},
                    "queue": {"returncode": 0, "stdout": ""},
                    "accounting": {
                        "returncode": 0,
                        "stdout": job_id + "|COMPLETED|0:0|\n" + job_id + ".batch|COMPLETED|0:0|\n",
                    },
                },
            }
        document = {
            "schema": "quest-sdsc-calibration-prerequisites-v1",
            "target": {
                "run_id": self.args.run_id,
                "code_sha256": self.args.code_sha256,
                "intent_id": "c" * 32,
            },
            "upstream": upstream,
        }

        def write(value):
            self.args.prerequisites.write_bytes(worker.canonical(value))
            self.args.prerequisites_sha256 = worker.sha(self.args.prerequisites.read_bytes())

        write(document)
        self.assertTrue(worker.validate_prerequisites(self.args, reports)["upstream_verified"])
        bad = copy.deepcopy(document)
        bad["upstream"]["teacher"]["status"]["accounting"]["stdout"] += "111.extern|FAILED|1:0|\n"
        write(bad)
        with self.assertRaisesRegex(ValueError, "Slurm job/step"):
            worker.validate_prerequisites(self.args, reports)
        bad = copy.deepcopy(document)
        bad["upstream"]["preflight"]["status"]["queue"]["stdout"] = "222|RUNNING|node"
        write(bad)
        with self.assertRaisesRegex(ValueError, "inconclusive"):
            worker.validate_prerequisites(self.args, reports)
        bad = copy.deepcopy(document)
        bad["target"]["run_id"] = "another-run"
        write(bad)
        with self.assertRaisesRegex(ValueError, "immutable invocation"):
            worker.validate_prerequisites(self.args, reports)

    def test_staging_preserves_inputs_and_rejects_external_symlinks(self):
        for name in ("dataset", "teacher_demos"):
            folder = self.args.teacher_root / name
            folder.mkdir(parents=True)
            (folder / "fixture.json").write_text('{"test":true}')
        destination = self.root / "copied"
        worker.stage_inputs(self.args.teacher_root, destination)
        copied = destination / "dataset/fixture.json"
        original = self.args.teacher_root / "dataset/fixture.json"
        self.assertEqual(copied.read_bytes(), original.read_bytes())
        self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
        original.write_text("changed")
        self.assertNotEqual(copied.read_bytes(), original.read_bytes())
        (self.args.teacher_root / "dataset/external").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.stage_inputs(self.args.teacher_root, self.root / "refused")

    def test_input_staging_refuses_recursive_output_under_input(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            worker.stage_inputs(self.args.teacher_root, self.args.teacher_root / "dataset/output")

    def test_child_waits_in_wrapper_group_and_nonzero_exit_is_failure(self):
        process = Mock()
        process.wait.return_value = 1
        process.poll.return_value = 1
        with (
            patch.object(worker.subprocess, "Popen", return_value=process) as launch,
            self.assertRaisesRegex(ValueError, "scientific command failed"),
        ):
            worker.run_command(
                ["/python", "--literal"],
                cwd=self.root,
                environment={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"},
                log=self.root / "failure.log",
            )
        self.assertIs(launch.call_args.kwargs["start_new_session"], False)
        process.wait.assert_called_once_with()

    def test_interrupted_child_is_bounded_and_reaped(self):
        process = Mock()
        process.wait.side_effect = [InterruptedError("TERM"), subprocess.TimeoutExpired("python", 30), 1]
        process.poll.return_value = None
        with (
            patch.object(worker.subprocess, "Popen", return_value=process),
            self.assertRaises(InterruptedError),
        ):
            worker.run_command(["/python"], cwd=self.root, environment={}, log=self.root / "interrupt.log")
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args.kwargs, {"timeout": 10})

    def execution_fixture(self):
        config = {"production_safety": {"initial_checkpoint_hash": ""}}
        runtime = {"production_safety": {"initial_checkpoint_hash": "e" * 64}}
        binding = {"code_commit": "a" * 40}
        teacher = SimpleNamespace(
            publish=worker.helper("sdsc_teacher_prepare").publish,
            validate_checkout=lambda args: {"science_git_head": "a" * 40},
        )
        guards = SimpleNamespace(memory_envelope=memory_fixture)
        api = SimpleNamespace(
            compose=lambda *args, **kwargs: runtime,
            binding=lambda config: binding,
            config_sha=lambda config: "d" * 64,
        )
        report = {
            "g0_passed": False,
            "factorial_ready": False,
            "pilot_passed": False,
            "execution_class_certified": False,
        }
        validator = Mock()
        validator.validate_factorial_run_artifacts.return_value = SimpleNamespace(
            content_binding=lambda: {"validated": True}
        )
        stack = contextlib.ExitStack()
        stack.enter_context(
            patch.object(
                worker,
                "helper",
                side_effect=lambda name: teacher if name == "sdsc_teacher_prepare" else guards,
            )
        )
        stack.enter_context(
            patch.object(
                worker,
                "allocation_identity",
                return_value={"job_id": "999", "cuda_visible_devices": "GPU-a,GPU-b"},
            )
        )
        stack.enter_context(patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}))
        stack.enter_context(patch.object(worker, "gpu_identity", return_value=[{}, {}]))
        stack.enter_context(patch.object(worker, "stage_inputs", return_value=1000))
        stack.enter_context(patch.object(worker, "validate_teacher"))
        stack.enter_context(
            patch.object(worker, "initial_checkpoint_identity", return_value={"sha256": "e" * 64, "size": 10})
        )
        stack.enter_context(patch.object(worker.importlib, "import_module", return_value=validator))
        commands = stack.enter_context(patch.object(worker, "run_command"))
        return stack, config, binding, api, report, validator, commands

    def test_execution_accepts_only_original_artifact_validator_and_never_claims_g0(self):
        stack, config, binding, api, report, validator, commands = self.execution_fixture()
        with stack:
            result = worker.execute_calibration(self.args, config, [], binding, None, api, report, {})
        self.assertTrue(result["passed"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(commands.call_count, 2)
        kwargs = validator.validate_factorial_run_artifacts.call_args.kwargs
        self.assertEqual(kwargs["expected_world_size"], 2)
        self.assertEqual(kwargs["expected_code_commit"], "a" * 40)
        self.assertEqual(kwargs["expected_resume_ancestry"], [])
        self.assertEqual(
            kwargs["expected_resolved_config"]["production_safety"]["initial_checkpoint_hash"], "e" * 64
        )
        for key in ("g0_passed", "factorial_ready", "pilot_passed", "execution_class_certified"):
            self.assertFalse(result[key])
        self.assertTrue((self.args.output_dir / "g0-calibration.json").is_file())

    def test_failed_export_never_starts_training_and_preserves_failure(self):
        stack, config, binding, api, report, validator, commands = self.execution_fixture()
        with stack:
            commands.side_effect = ValueError("export failed")
            with self.assertRaisesRegex(ValueError, "export failed"):
                worker.execute_calibration(self.args, config, [], binding, None, api, report, {})
        self.assertEqual(commands.call_count, 1)
        validator.validate_factorial_run_artifacts.assert_not_called()
        failure = json.loads((self.args.output_dir / "g0-calibration.json").read_text())
        self.assertFalse(failure["passed"])
        self.assertFalse(failure["training_executed"])

    def test_zero_exit_training_with_invalid_artifacts_remains_failure(self):
        stack, config, binding, api, report, validator, commands = self.execution_fixture()
        with stack:
            validator.validate_factorial_run_artifacts.side_effect = ValueError("invalid final checkpoint")
            with self.assertRaisesRegex(ValueError, "invalid final checkpoint"):
                worker.execute_calibration(self.args, config, [], binding, None, api, report, {})
        self.assertEqual(commands.call_count, 2)
        failure = json.loads((self.args.output_dir / "g0-calibration.json").read_text())
        self.assertFalse(failure["passed"])
        self.assertTrue(failure["training_executed"])

    def test_existing_output_is_not_reused_or_modified(self):
        self.args.output_dir.mkdir(parents=True)
        marker = self.args.output_dir / "retained.txt"
        marker.write_text("preserve")
        stack, config, binding, api, report, _validator, commands = self.execution_fixture()
        with stack, self.assertRaisesRegex(ValueError, "implicit resume"):
            worker.execute_calibration(self.args, config, [], binding, None, api, report, {})
        commands.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")

    def test_real_composer_freezes_science_and_exposes_runtime_hash_delta(self):
        repository = SCRIPT.parent.parent
        python = repository / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("repository CPU dependency environment unavailable")
        code = """
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'src'))
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.artifacts.execution_science_protocol import canonical_science_config_sha256
from posttrain_circuits.learning.training.execution_safety_kernel import batch_token_contract
spec = importlib.util.spec_from_file_location('candidate', repo / 'tools/sdsc_g0_calibration.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
teacher = candidate.helper('sdsc_teacher_prepare')
args = SimpleNamespace(science_root=repo, output_dir=Path('/tmp/qwen3-v2/output'),
                       teacher_root=Path('/tmp/qwen3-v2/teacher'), hf_home=Path('/tmp/hf'),
                       python=Path(sys.executable))
resolved = SimpleNamespace(binding=SimpleNamespace(hydra_override_vector=teacher.OVERRIDES,
                           storage_neutral_resolved_config_sha256=candidate.BASE_SCIENCE_SHA256))
api = SimpleNamespace(resolve=lambda **kw: resolved, compose=compose_config,
                      config_sha=canonical_science_config_sha256,
                      binding=lambda config: {'code_commit': 'a' * 40})
config, overrides, binding, protocol = candidate.compose_base(args, api, {'science_git_head': 'a' * 40})
assert config['trainer']['max_steps'] == 120
assert config['trainer']['token_budget'] == 2000000
assert config['trainer']['global_batch_size'] == 64
assert config['trainer']['checkpoint_every'] == config['trainer']['evaluation_every'] == 20
assert config['g0']['full_parameter_training'] is True
assert config['experiment']['name'] == 'canonical_sft'
assert config['teacher']['generation_seed'] == 31415
assert sum(config['task']['split_sizes'].values()) == 144000
assert batch_token_contract(2, 256)['samples_by_rank'] == [32, 32]
plan = candidate.build_plan(args, overrides, initial_checkpoint_sha256='f' * 64)
runtime_config = compose_config(plan['training_overrides'], config_root=repo / 'configs')
assert canonical_science_config_sha256(runtime_config) != candidate.BASE_SCIENCE_SHA256
runtime_config['production_safety']['initial_checkpoint_hash'] = ''
assert runtime_config == config
assert candidate.build_plan(args, overrides)['train_argv'] is None
print(json.dumps({'base': canonical_science_config_sha256(config), 'no_model_loaded': True}))
"""
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", code, str(repository)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        self.assertEqual(json.loads(result.stdout)["base"], worker.BASE_SCIENCE_SHA256)


if __name__ == "__main__":
    unittest.main()
