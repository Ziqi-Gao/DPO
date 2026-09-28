"""CPU admission fixtures for the new dense-teacher H100 execution canary."""

import ast
import copy
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/sdsc_adapted_training_preflight.py"
SPEC = importlib.util.spec_from_file_location("sdsc_adapted_preflight", SCRIPT)
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
    guards = worker.guards
    report = {
        "kind": worker.KIND,
        "task": worker.TASK,
        "passed": True,
        "exit_code": 0,
        "world_size": 2,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate": False,
        "job_id": "12345",
        "run_id": "real-preflight",
        "code_sha256": "a" * 64,
    }
    contract = worker.expected_batch_contract()
    identity = {
        "kind": "learned_dense_checkpoint",
        "teacher_checkpoint_sha256": "d" * 64,
        "base_revision": guards.MODELS["Qwen/Qwen3-8B"],
        "tokenizer_id": "Qwen/Qwen3-8B",
        "tokenizer_revision": guards.MODELS["Qwen/Qwen3-8B"],
        "tokenizer_fingerprint": "e" * 64,
        "chat_template_sha256": "f" * 64,
        "prompt_protocol": "qwen3_non_thinking_v1",
    }
    bindings = {
        "teacher_identity": identity,
        "accepted_teacher_sha256": "c" * 64,
        "teacher_acceptance_inventory_sha256": "b" * 64,
        "student_protocol_sha256": "0" * 64,
        "science_git_head": "1" * 40,
    }
    report.update(bindings)
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
                "rank_zero_teacher": worker.teacher_probe_evidence(identity),
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
    return report, bindings


class AdaptedTrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = {
            "kind": "learned_dense_checkpoint",
            "teacher_checkpoint_sha256": "d" * 64,
            "base_revision": worker.MODELS["Qwen/Qwen3-8B"],
            "tokenizer_id": "Qwen/Qwen3-8B",
            "tokenizer_revision": worker.MODELS["Qwen/Qwen3-8B"],
            "tokenizer_fingerprint": "e" * 64,
            "chat_template_sha256": "f" * 64,
            "prompt_protocol": "qwen3_non_thinking_v1",
        }
        self.accepted = {
            "sha256": "c" * 64,
            "formal_teacher_accepted": True,
            "original_128_token_readiness_pass_claim": False,
            "teacher_identity": self.identity,
        }
        self.verified = {
            "accepted_teacher": self.accepted,
            "teacher_identity": self.identity,
            "inventory_sha256": "b" * 64,
            "accepted_teacher_sha256": "c" * 64,
        }
        self.contract = SimpleNamespace(verify_teacher_acceptance=Mock(return_value=self.verified))
        self.args = SimpleNamespace(
            work_dir=self.root,
            science_root=self.root / "science",
            teacher_checkpoint_root=self.root / "teacher",
            teacher_acceptance=self.root / "acceptance",
            teacher_acceptance_sha256="b" * 64,
            teacher_checkpoint_sha256="d" * 64,
            code_sha256="a" * 64,
            student_protocol_sha256="0" * 64,
            science_git_head="1" * 40,
            run_id="new-independent-run",
        )
        for name in ("science_root", "teacher_checkpoint_root", "teacher_acceptance"):
            getattr(self.args, name).mkdir()

    def test_bootstrap_has_no_numerical_or_scientific_imports(self):
        code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {'torch', 'transformers', 'posttrain_circuits', 'yaml'}:
            raise AssertionError('premature model/science import: ' + name)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.TASK)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "qwen3-v2-adapted-preflight")

    def test_acceptance_inventory_and_genuine_dense_identity_are_bound(self):
        result = worker.validate_inputs(self.args, self.contract)
        self.contract.verify_teacher_acceptance.assert_called_once_with(
            self.args.teacher_acceptance, expected_inventory_sha256="b" * 64
        )
        self.assertEqual(result["teacher_identity"], self.identity)
        self.assertEqual(result["accepted_teacher_sha256"], "c" * 64)
        self.assertEqual(result["student_protocol_sha256"], "0" * 64)
        self.assertEqual(result["science_git_head"], "1" * 40)

    def test_wrong_inventory_checkpoint_or_acceptance_is_rejected(self):
        changes = [
            ("inventory_sha256", "9" * 64),
            ("accepted_teacher_sha256", "9" * 64),
            ("accepted_teacher.formal_teacher_accepted", False),
            ("accepted_teacher.original_128_token_readiness_pass_claim", True),
            ("teacher_identity.kind", "pinned_hf_base"),
            ("teacher_identity.teacher_checkpoint_sha256", "9" * 64),
        ]
        for name, value in changes:
            with self.subTest(name=name):
                document = copy.deepcopy(self.verified)
                target = document
                *parents, leaf = name.split(".")
                for key in parents:
                    target = target[key]
                target[leaf] = value
                contract = SimpleNamespace(verify_teacher_acceptance=Mock(return_value=document))
                with self.assertRaisesRegex(ValueError, "accepted teacher publication"):
                    worker.validate_inputs(self.args, contract)

    def test_external_or_symlink_teacher_path_rejected_before_acceptance_read(self):
        external = self.root.parent
        linked = self.root / "linked"
        linked.symlink_to(self.args.teacher_checkpoint_root, target_is_directory=True)
        for path in (external, linked, Path("relative"), self.root / "absent"):
            with self.subTest(path=path):
                args = copy.copy(self.args)
                args.teacher_checkpoint_root = path
                with self.assertRaisesRegex(ValueError, "real staged directory"):
                    worker.validate_inputs(args, self.contract)
        self.contract.verify_teacher_acceptance.assert_not_called()

    def test_invalid_protocol_or_head_rejected_before_acceptance_read(self):
        for name, value in (("student_protocol_sha256", "proposed"), ("science_git_head", "929fb14")):
            args = copy.copy(self.args)
            setattr(args, name, value)
            with self.assertRaisesRegex(ValueError, "invalid"):
                worker.validate_inputs(args, self.contract)
        self.contract.verify_teacher_acceptance.assert_not_called()

    def loaded(self):
        return SimpleNamespace(
            teacher_checkpoint_sha256=self.identity["teacher_checkpoint_sha256"],
            base_revision=self.identity["base_revision"],
            tokenizer_id=self.identity["tokenizer_id"],
            requested_tokenizer_revision=self.identity["tokenizer_revision"],
            tokenizer_hash=self.identity["tokenizer_fingerprint"],
            chat_template_sha256=self.identity["chat_template_sha256"],
            prompt_protocol=self.identity["prompt_protocol"],
        )

    def test_actual_loader_identity_must_match_all_acceptance_fields(self):
        worker.validate_loaded_teacher(self.loaded(), self.identity)
        for name in vars(self.loaded()):
            with self.subTest(name=name):
                loaded = self.loaded()
                setattr(loaded, name, "wrong")
                with self.assertRaisesRegex(ValueError, "differs from independent acceptance"):
                    worker.validate_loaded_teacher(loaded, self.identity)
        with self.assertRaises(ValueError):
            worker.validate_loaded_teacher(
                SimpleNamespace(resolved_model_commit=worker.MODELS["Qwen/Qwen3-8B"]), self.identity
            )

    def test_teacher_probe_reports_dense_identity_and_no_fictitious_hub_commit(self):
        result = worker.teacher_probe_evidence(self.identity)
        self.assertEqual(result["teacher_identity"], self.identity)
        self.assertTrue(result["dense_checkpoint_verified"])
        self.assertTrue(result["base_revision_is_ancestry_only"])
        self.assertNotIn("revision", result)
        self.assertNotIn("resolved_model_commit", result)

    def test_physical_student_update_and_full_state_restore_are_unchanged(self):
        """Review seam: compare actual numerical AST after the teacher forward."""
        old_tree = ast.parse((ROOT / "tools/sdsc_training_preflight.py").read_text())
        new_tree = ast.parse(SCRIPT.read_text())

        def numerical_tail(tree):
            function = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_canary"
            )
            start = next(
                i
                for i, node in enumerate(function.body)
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "contract"
            )
            return ast.dump(ast.Module(body=function.body[start:], type_ignores=[]), include_attributes=False)

        self.assertEqual(numerical_tail(old_tree), numerical_tail(new_tree))

    def test_completed_report_requires_dense_binding_and_all_physical_checks(self):
        report, expected = preflight_fixture()
        self.assertTrue(worker.validate_completed_report(report, expected)["report_validated"])
        changes = (
            ("teacher_identity", {"kind": "pinned_hf_base"}),
            ("student_protocol_sha256", "9" * 64),
            ("science_git_head", "9" * 40),
            ("nccl_all_reduce", False),
            ("full_state_optimizer_scheduler_rng_restore", False),
            ("rank_zero_teacher", {"finite": True, "revision": worker.MODELS["Qwen/Qwen3-8B"]}),
            ("reserved_global_nonpadding_tokens", 1536),
        )
        for key, value in changes:
            bad = copy.deepcopy(report)
            bad["ranks"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_completed_report(bad, expected)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            worker.validate_completed_report(report, {})

    def test_allocation_guard_remains_exact_two_rank_shared_partition(self):
        env = {
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "196608",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
        }
        self.assertEqual(worker.guards.allocation_identity(env)["threads_per_rank"], 12)
        for key, value in (("WORLD_SIZE", "4"), ("SLURM_MEM_PER_NODE", "0"), ("CUDA_VISIBLE_DEVICES", "0")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.guards.allocation_identity({**env, key: value})


if __name__ == "__main__":
    unittest.main()
