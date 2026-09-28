"""CPU fixtures for reviewed teacher consumption and calibration admission."""

import contextlib
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/sdsc_adapted_calibration.py"
SPEC = importlib.util.spec_from_file_location("adapted_calibration", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "adapted_preflight_fixtures", Path(__file__).with_name("test_adapted_training_preflight.py")
)
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


class AdaptedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            science_root=self.root / "science",
            science_git_head="1" * 40,
            work_dir=self.root,
            output_dir=self.root / "output",
            hf_home=self.root / "hf",
            teacher_input_root=self.root / "input",
            dataset_root=self.root / "dataset",
            python=Path(sys.executable).resolve(),
            run_id="new-calibration",
            code_sha256="a" * 64,
            student_protocol_sha256="0" * 64,
            prerequisites=self.root / "prerequisites.json",
        )
        for path in (self.args.science_root, self.args.teacher_input_root):
            path.mkdir()

    def upstream(self):
        report, expected = fixtures.preflight_fixture()
        identity = {key: report[key] for key in ("task", "job_id", "run_id", "code_sha256")}
        upstream = {
            "report": report,
            "report_sha256": "7" * 64,
            "receipt": dict(identity),
            "publication_receipt_sha256": "8" * 64,
            "publication_receipt": {
                **identity,
                "passed": True,
                "persisted": True,
                "persistent_read_back_verified": True,
                "files": [{"path": "adapted-preflight.json", "size": 1234, "sha256": "7" * 64}],
            },
            "status": {
                **{key: report[key] for key in ("task", "job_id", "run_id")},
                "success": True,
                "state": "COMPLETED",
                "result": {"verified": True, "result_sha256": "7" * 64},
                "queue": {"returncode": 0, "stdout": ""},
                "accounting": {"returncode": 0, "stdout": "12345|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n"},
            },
        }
        return upstream, expected

    def test_worker_bootstrap_never_imports_model_or_science(self):
        code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {'torch', 'transformers', 'posttrain_circuits'}:
            raise AssertionError('premature scientific import: ' + name)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.TASK)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), worker.TASK)

    def test_training_argv_uses_fixed_two_rank_original_cli_and_actual_initial_hash(self):
        before = worker.build_plan(self.args, [])
        self.assertIsNone(before["train_argv"])
        self.assertFalse(before["execution_enabled"])
        self.args.output_dir = self.root / "literal ; $(touch NEVER)"
        overrides = ["adapted_teacher=qwen3_accepted_student_v3", *worker.storage_overrides(self.args)]
        plan = worker.build_plan(self.args, overrides, initial_checkpoint_sha256="f" * 64)
        command = plan["train_argv"]
        self.assertIn("posttrain_circuits.cli.train", command)
        self.assertIn("production_safety.initial_checkpoint_hash=" + "f" * 64, command)
        self.assertEqual(command[command.index("--num_processes") + 1], "2")
        self.assertEqual(command[command.index("--num_cpu_threads_per_process") + 1], "12")
        self.assertIn(
            str(self.args.science_root / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"), command
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", plan["environment_updates"])
        self.assertFalse((self.root / "NEVER").exists())
        self.assertIn(
            "state_source.store_path=" + json.dumps(str(self.args.output_dir / "teacher_demos")), overrides
        )

    def test_matching_completed_preflight_is_accepted_without_fabricated_status_hash(self):
        upstream, expected = self.upstream()
        self.assertNotIn("code_sha256", upstream["status"])
        self.assertTrue(worker.validate_preflight_upstream(upstream, expected)["verified"])

    def test_missing_queue_job_or_failed_accounting_step_cannot_pass(self):
        upstream, expected = self.upstream()
        changes = (
            ("success", False),
            ("state", "FAILED"),
            ("queue", {"returncode": 0, "stdout": "12345|RUNNING"}),
            ("accounting", {"returncode": 0, "stdout": "12345|COMPLETED|0:0\n12345.batch|FAILED|1:0"}),
            ("accounting", {"returncode": 0, "stdout": ""}),
            ("result", {"verified": True, "result_sha256": "9" * 64}),
        )
        for name, value in changes:
            bad = copy.deepcopy(upstream)
            bad["status"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                worker.validate_preflight_upstream(bad, expected)

    def test_prerequisite_target_acceptance_and_full_protocol_origin_are_bound(self):
        upstream, expected = self.upstream()
        acceptance = {
            "teacher_identity": expected["teacher_identity"],
            "accepted_teacher_sha256": "c" * 64,
            "inventory_sha256": "b" * 64,
        }
        document = {
            "schema": "quest-sdsc-adapted-student-prerequisites-v1",
            "task": worker.TASK,
            "target": {
                "run_id": self.args.run_id,
                "code_sha256": self.args.code_sha256,
                "intent_id": "f" * 32,
            },
            "protocol": {
                "head": self.args.science_git_head,
                "protocol_sha256": self.args.student_protocol_sha256,
            },
            "acceptance": acceptance,
            "preflight": upstream,
        }

        def save(value):
            raw = worker.canonical(value)
            self.args.prerequisites.write_bytes(raw)
            self.args.prerequisites_sha256 = worker.sha(raw)

        save(document)
        self.assertTrue(worker.validate_prerequisites(self.args, acceptance)[1]["preflight"]["verified"])
        document["protocol"]["protocol_path"] = worker.PROTOCOL
        save(document)
        with patch.object(worker, "validate_preflight_upstream", return_value={"verified": True}) as verify:
            worker.validate_prerequisites(self.args, acceptance)
        self.assertEqual(verify.call_args.args[1]["student_protocol_path"], worker.PROTOCOL)
        bad = copy.deepcopy(document)
        bad["target"]["run_id"] = "another-release"
        save(bad)
        with self.assertRaisesRegex(ValueError, "immutable invocation"):
            worker.validate_prerequisites(self.args, acceptance)
        save(document)
        with self.assertRaisesRegex(ValueError, "staged publication"):
            worker.validate_prerequisites(self.args, {**acceptance, "inventory_sha256": "9" * 64})
        binding = SimpleNamespace(
            head="1" * 40,
            protocol_sha256="2" * 64,
            sha256="3" * 64,
            reviewed_implementation_commit="4" * 40,
            git_commit="5" * 40,
            science_file_sha256={"src/science.py": "6" * 64},
        )
        proof = {
            "head": binding.head,
            "protocol_sha256": binding.protocol_sha256,
            "artifact_sha256": binding.sha256,
            "implementation_commit": binding.reviewed_implementation_commit,
            "acceptance_commit": binding.git_commit,
            "science_file_sha256": binding.science_file_sha256,
        }
        worker.validate_protocol_proof(proof, binding)
        with self.assertRaisesRegex(ValueError, "protocol origin"):
            worker.validate_protocol_proof({**proof, "acceptance_commit": "9" * 40}, binding)

    def test_teacher_copy_preserves_nine_original_files_and_rejects_symlinks(self):
        originals = {}
        for index in range(9):
            path = self.args.teacher_input_root / f"original-{index}.json"
            path.write_bytes(f"original bytes {index}\n".encode())
            originals[path.name] = path.read_bytes()
        target = self.root / "copy"
        result = worker.stage_teacher_inputs(self.args.teacher_input_root, target)
        self.assertEqual(result["files"], 9)
        self.assertTrue(result["read_back_verified"])
        self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, originals)
        with self.assertRaisesRegex(ValueError, "fresh"):
            worker.stage_teacher_inputs(self.args.teacher_input_root, target)
        (self.args.teacher_input_root / "original-0.json").unlink()
        (self.args.teacher_input_root / "original-0.json").symlink_to(target / "original-0.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.stage_teacher_inputs(self.args.teacher_input_root, self.root / "bad-copy")

    def test_child_process_is_foreground_and_nonzero_is_failure(self):
        log = self.root / "child.log"
        worker.run_command(
            [sys.executable, "-I", "-B", "-c", "print('finite child')"],
            cwd=self.root,
            environment=dict(os.environ),
            log=log,
        )
        self.assertEqual(log.read_text().strip(), "finite child")
        with self.assertRaisesRegex(ValueError, "scientific command failed"):
            worker.run_command(
                [sys.executable, "-I", "-B", "-c", "raise SystemExit(7)"],
                cwd=self.root,
                environment=dict(os.environ),
                log=self.root / "failed.log",
            )

    def test_real_composer_preserves_training_contract_and_binds_runtime_hash_only(self):
        python = ROOT / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("scientific CPU runtime unavailable")
        code = """
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.artifacts.adapted_student_protocol import validate_student_config
from posttrain_circuits.learning.training.execution_safety_kernel import batch_token_contract
spec = importlib.util.spec_from_file_location('worker', root / 'tools/sdsc_adapted_calibration.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
args = SimpleNamespace(science_root=root, science_git_head='1' * 40,
    output_dir=Path('/tmp/qwen3-v2/output'), dataset_root=Path('/tmp/qwen3-v2/dataset'),
    teacher_input_root=Path('/tmp/qwen3-v2/input'), hf_home=Path('/tmp/hf'),
    python=Path(sys.executable), student_protocol_sha256='0' * 64)
reviewed = SimpleNamespace(head=args.science_git_head, protocol_sha256=args.student_protocol_sha256)
def validate(config):
    validate_student_config(config)
    return reviewed
api = SimpleNamespace(resolve=lambda *a, **k: reviewed, validate=validate,
    compose=compose_config, binding=lambda config: {'code_commit': args.science_git_head})
config, overrides, binding, protocol = worker.compose_base(args, api)
assert config['trainer']['max_steps'] == 120
assert config['trainer']['token_budget'] == 2000000
assert config['trainer']['global_batch_size'] == 64
assert config['trainer']['max_completion_length'] == 128
assert config['trainer']['checkpoint_every'] == config['trainer']['evaluation_every'] == 20
assert config['state_source']['store_path'] == '/tmp/qwen3-v2/output/teacher_demos'
assert sum(config['task']['split_sizes'].values()) == 144000
plan = worker.build_plan(args, overrides, initial_checkpoint_sha256='f' * 64)
runtime = compose_config(plan['training_overrides'], config_root=root / 'configs')
validate_student_config(runtime)
assert runtime['production_safety']['initial_checkpoint_hash'] == 'f' * 64
runtime['production_safety']['initial_checkpoint_hash'] = ''
assert runtime == config
preflight = worker.helper('sdsc_adapted_training_preflight')
assert preflight.expected_batch_contract() == batch_token_contract(2, 256)
print(json.dumps({'pure_config_pass': True, 'actual_model_loaded': False}))
"""
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", code, str(ROOT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        self.assertEqual(json.loads(result.stdout), {"pure_config_pass": True, "actual_model_loaded": False})

    def execution_fixture(self, *, validator_error=None):
        config = {"production_safety": {"initial_checkpoint_hash": ""}, "model": {"id": "fixture"}}
        binding = {"code_commit": self.args.science_git_head}
        reviewed = SimpleNamespace(head=self.args.science_git_head)
        runtime = copy.deepcopy(config)
        runtime["production_safety"]["initial_checkpoint_hash"] = "f" * 64
        api = SimpleNamespace(
            compose=Mock(return_value=runtime),
            binding=Mock(return_value=binding),
            validate=Mock(return_value=reviewed),
            resolve=Mock(return_value=reviewed),
        )
        report = {
            "g0_passed": False,
            "pilot_passed": False,
            "factorial_ready": False,
            "execution_class_certified": False,
            "teacher_inputs": {"verified": True},
        }
        validator = SimpleNamespace(
            validate_factorial_run_artifacts=Mock(
                side_effect=validator_error,
                return_value=SimpleNamespace(content_binding=lambda: {"actual": "validated"}),
            )
        )
        stack = contextlib.ExitStack()
        stack.enter_context(
            patch.object(
                worker,
                "allocation_identity",
                return_value={"job_id": "23456", "cuda_visible_devices": "GPU-a,GPU-b"},
            )
        )
        stack.enter_context(patch.object(worker, "gpu_identity", return_value=["fixture", "fixture"]))
        original_helper = worker.helper
        guard = original_helper("sdsc_training_preflight")
        guard.memory_envelope = Mock(return_value={"passed": True})
        stack.enter_context(
            patch.object(
                worker,
                "helper",
                side_effect=lambda name: guard
                if name == "sdsc_training_preflight"
                else original_helper(name),
            )
        )
        stack.enter_context(patch.object(worker, "stage_teacher_inputs", return_value={"files": 9}))
        stack.enter_context(patch.object(worker, "teacher_inputs", return_value=report["teacher_inputs"]))
        stack.enter_context(
            patch.object(worker, "initial_checkpoint_identity", return_value={"sha256": "f" * 64})
        )
        commands = stack.enter_context(patch.object(worker, "run_command"))
        original_import = worker.importlib.import_module
        stack.enter_context(
            patch.object(
                worker.importlib,
                "import_module",
                side_effect=lambda name: validator
                if name == "posttrain_circuits.cli.factorial_run_validation"
                else original_import(name),
            )
        )
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}))
        return stack, config, binding, reviewed, api, report, validator, commands

    def test_success_requires_actual_artifact_validator_with_exact_config_and_source(self):
        stack, config, binding, reviewed, api, report, validator, commands = self.execution_fixture()
        with stack:
            result = worker.execute_calibration(self.args, config, [], binding, reviewed, api, report)
        self.assertEqual(commands.call_count, 2)
        self.assertTrue(result["passed"])
        self.assertTrue(result["training_executed"])
        self.assertTrue(
            all(
                result[key] is False
                for key in ("g0_passed", "pilot_passed", "factorial_ready", "execution_class_certified")
            )
        )
        kwargs = validator.validate_factorial_run_artifacts.call_args.kwargs
        self.assertEqual(kwargs["expected_world_size"], 2)
        self.assertEqual(kwargs["expected_code_commit"], self.args.science_git_head)
        self.assertEqual(kwargs["expected_resume_ancestry"], [])
        self.assertEqual(
            kwargs["expected_resolved_config"]["production_safety"]["initial_checkpoint_hash"], "f" * 64
        )

    def test_zero_exit_with_invalid_artifacts_never_passes(self):
        stack, config, binding, reviewed, api, report, _validator, commands = self.execution_fixture(
            validator_error=ValueError("bad checkpoint evidence")
        )
        with stack, self.assertRaisesRegex(ValueError, "bad checkpoint evidence"):
            worker.execute_calibration(self.args, config, [], binding, reviewed, api, report)
        self.assertEqual(commands.call_count, 2)
        final = json.loads((self.args.output_dir / worker.RESULT).read_text())
        self.assertFalse(final["passed"])
        self.assertEqual(final["exit_code"], 1)

    def test_failed_export_cannot_start_training_or_claim_training_executed(self):
        stack, config, binding, reviewed, api, report, validator, commands = self.execution_fixture()
        commands.side_effect = ValueError("export failed")
        with stack, self.assertRaisesRegex(ValueError, "export failed"):
            worker.execute_calibration(self.args, config, [], binding, reviewed, api, report)
        self.assertEqual(commands.call_count, 1)
        self.assertFalse(report["training_executed"])
        validator.validate_factorial_run_artifacts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
