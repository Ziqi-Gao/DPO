"""CPU-only teacher preparation guards; no teacher/model/GPU execution."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_teacher_prepare.py"
SPEC = importlib.util.spec_from_file_location("sdsc_teacher_prepare", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class TeacherPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.science = self.root / "science"
        self.science.mkdir()
        self.home = self.root / "huggingface"
        self.home.mkdir()
        self.output = self.root / "outputs" / "qwen3-v2"
        self.manifest = self.root / "provenance.json"
        self.manifest.write_text("{}")
        self.argv = [
            "--science-root",
            str(self.science),
            "--work-dir",
            str(self.root),
            "--output-dir",
            str(self.output),
            "--hf-home",
            str(self.home),
            "--provenance-manifest",
            str(self.manifest),
            "--provenance-manifest-sha256",
            "b" * 64,
            "--run-id",
            "teacher-test",
            "--code-sha256",
            "a" * 64,
        ]
        self.args = worker.parser().parse_args(self.argv)
        self.provenance = {
            "science_git_head": "c" * 40,
            "bundle_sha256": "d" * 64,
            "provenance_manifest_sha256": "b" * 64,
        }
        self.config = {
            "seed": 42,
            "task": {
                "num_examples": 256,
                "split_sizes": {
                    "train": 100000,
                    "validation": 10000,
                    "iid_test": 10000,
                    "ood_depth_test": 10000,
                    "ood_structure_test": 10000,
                    "circuit_discovery": 2000,
                    "circuit_validation": 2000,
                },
            },
            "teacher": {"generation_seed": 31415},
            "state_source": {"num_candidates": 8, "max_prompt_tokens": 1246, "max_new_tokens": 256},
        }
        self.binding = {"code_commit": "c" * 40, "protocol_track": "qwen3_v2"}
        self.resolved = types.SimpleNamespace(
            binding=types.SimpleNamespace(
                hydra_override_vector=worker.OVERRIDES,
                storage_neutral_resolved_config_sha256="e" * 64,
                protocol_id="prompt-v3",
            ),
            sha256="f" * 64,
            acceptance_commit="1" * 40,
            reviewed_implementation_commit="2" * 40,
        )
        self.api = types.SimpleNamespace(
            resolve=lambda **kwargs: self.resolved,
            compose=lambda *args, **kwargs: self.config,
            config_sha=lambda config: "e" * 64,
            binding=lambda config: self.binding,
        )

    def environment(self):
        return {
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "196608",
            "CUDA_VISIBLE_DEVICES": "GPU-assigned",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }

    def orchestration(self, *, generation_error=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        calls = []
        stack.enter_context(patch.dict(os.environ, self.environment()))
        for name, value in (
            ("runtime_identity", {"python": "fixture"}),
            ("validate_checkout", self.provenance),
            ("scientific_api", self.api),
            ("local_environment", {"node_local_mount": {"fstype": "ext4"}}),
            ("gpu_identity", {"name": "fixture H100"}),
            ("final_usage", {"peak_gpu_allocated_bytes": 123, "cgroup_memory": {"passed": True}}),
        ):
            stack.enter_context(patch.object(worker, name, return_value=value))

        def dataset(*_args):
            calls.append("dataset")
            (self.output / "dataset").mkdir()
            return None, [], {"ordered_prompt_ids": [str(i) for i in range(256)]}

        def prompts(*_args):
            calls.append("prompts")
            return {"tokenizer_fingerprint": "3" * 64}

        def generate(*_args):
            calls.append("generate")
            if generation_error:
                raise generation_error

        stack.enter_context(patch.object(worker, "prepare_dataset", side_effect=dataset))
        stack.enter_context(patch.object(worker, "validate_prompts", side_effect=prompts))
        stack.enter_context(patch.object(worker, "generate_teacher", side_effect=generate))
        stack.enter_context(
            patch.object(worker, "validate_store", return_value={"ready_for_formal_sft": True})
        )
        return calls

    def test_import_and_help_are_stdlib_only(self):
        code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {'torch', 'transformers', 'posttrain_circuits'}:
            raise AssertionError('premature scientific import')
sys.meta_path.insert(0, Block())
spec=importlib.util.spec_from_file_location('teacher', sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.parser().print_help()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(SCRIPT)], capture_output=True, text=True, check=True
        )
        self.assertIn("--validate-only", result.stdout)

    def test_binding_preflight_preserves_scientific_seeds_and_population(self):
        config, overrides, binding, audit = worker.validate_binding(self.args, self.api, self.provenance)
        self.assertEqual(config["teacher"]["generation_seed"], 31415)
        self.assertEqual(sum(config["task"]["split_sizes"].values()), 144000)
        self.assertEqual(tuple(overrides[:5]), worker.OVERRIDES)
        self.assertIn("protocol_amendment_path=" + worker.AMENDMENT, overrides)
        self.assertEqual(binding, self.binding)
        self.assertEqual(audit["storage_neutral_resolved_config_sha256"], "e" * 64)

    def test_config_hash_mismatch_rejected_before_binding_or_generation(self):
        self.api.config_sha = lambda config: "0" * 64
        with patch.object(self.api, "binding") as binding:
            with self.assertRaisesRegex(ValueError, "resolved science configuration"):
                worker.validate_binding(self.args, self.api, self.provenance)
            binding.assert_not_called()

    def test_changed_seed_count_length_or_full_split_rejected(self):
        for section, key, value in (
            ("teacher", "generation_seed", 42),
            ("state_source", "num_candidates", 1),
            ("state_source", "max_prompt_tokens", 2000),
            ("task", "num_examples", 64),
        ):
            with (
                self.subTest(key=key),
                patch.dict(self.config[section], {key: value}),
                self.assertRaises(ValueError),
            ):
                worker.validate_binding(self.args, self.api, self.provenance)
        self.config["task"]["split_sizes"]["train"] = 256
        with self.assertRaisesRegex(ValueError, "144000"):
            worker.validate_binding(self.args, self.api, self.provenance)

    def test_validate_only_never_generates_data_or_touches_cuda(self):
        calls = self.orchestration()
        with patch.object(worker, "gpu_identity") as gpu, patch.object(worker, "allocation") as allocation:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(worker.main([*self.argv, "--validate-only"]), 0)
            gpu.assert_not_called()
            allocation.assert_not_called()
        self.assertEqual(calls, [])
        self.assertFalse(self.output.exists())
        report = json.loads(output.getvalue())
        self.assertTrue(report["binding_validated"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["g0_passed"])
        self.assertFalse(report["execution_class_certified"])

    def test_success_order_and_audit_do_not_claim_g0_or_persistence(self):
        calls = self.orchestration()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(self.argv), 0)
        self.assertEqual(calls, ["dataset", "prompts", "generate"])
        report = json.loads((self.output / "teacher-prepare.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["job_id"], "123")
        self.assertEqual(report["science_git_head"], "c" * 40)
        self.assertFalse(report["g0_passed"])
        self.assertFalse(report["execution_class_certified"])
        self.assertEqual(report["persistent_storage_publication"], "caller_required")
        self.assertTrue((self.output / "teacher-prepare-start.json").is_file())

    def test_generation_failure_reports_no_partial_ledger_and_preserves_dataset(self):
        self.orchestration(generation_error=RuntimeError("generation interrupted"))
        with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
            worker.main(self.argv)
        report = json.loads((self.output / "teacher-prepare.json").read_text())
        self.assertFalse(report["passed"])
        self.assertFalse(report["ledger_written"])
        self.assertFalse(report["partial_attempt_ledger_guaranteed"])
        self.assertTrue((self.output / "dataset").is_dir())

    def test_prompt_failure_precedes_generation(self):
        self.orchestration()
        with (
            patch.object(worker, "validate_prompts", side_effect=ValueError("overlong prompt")),
            patch.object(worker, "generate_teacher") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "overlong prompt"):
                worker.main(self.argv)
            generate.assert_not_called()

    def test_late_binding_change_rejected_before_generation(self):
        self.orchestration()
        with (
            patch.object(self.api, "binding", side_effect=[self.binding, {"code_commit": "9" * 40}]),
            patch.object(worker, "generate_teacher") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "binding changed before"):
                worker.main(self.argv)
            generate.assert_not_called()

    def test_allocation_cannot_use_login_or_two_gpu_visibility(self):
        with patch.dict(os.environ, self.environment()):
            original = os.environ["CUDA_VISIBLE_DEVICES"]
            self.assertEqual(worker.allocation(self.args)["cuda_visible_devices"], original)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], original)
            for change in (
                {"SLURM_JOB_ID": ""},
                {"CUDA_VISIBLE_DEVICES": "0,1"},
                {"WORLD_SIZE": "2"},
                {"SLURM_CPUS_PER_TASK": "4"},
                {"SLURM_MEM_PER_NODE": "16384"},
                {"LOCAL_RANK": "1"},
            ):
                with (
                    self.subTest(change=change),
                    patch.dict(os.environ, change),
                    self.assertRaises(ValueError),
                ):
                    worker.allocation(self.args)

    def test_no_clobber_output_and_safe_paths(self):
        self.orchestration()
        self.output.mkdir(parents=True)
        (self.output / "retained").write_text("existing")
        with self.assertRaisesRegex(ValueError, "already used"):
            worker.main(self.argv)
        self.assertEqual((self.output / "retained").read_text(), "existing")
        self.assertFalse((self.output / "teacher-prepare.json").exists())
        linked = self.root / "link"
        linked.symlink_to(self.science, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.real_path(linked)

    def test_provenance_hash_fails_before_git(self):
        (self.science / ".git").mkdir()
        with patch.object(worker, "git") as git:
            with self.assertRaisesRegex(ValueError, "provenance manifest SHA"):
                worker.validate_checkout(self.args)
            git.assert_not_called()

    def test_real_git_and_byte_hash_gate_reject_dirty_science(self):
        environment = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")

        def git(*args):
            return subprocess.run(
                ["/usr/bin/git", "-C", str(self.science), *args],
                capture_output=True,
                env=environment,
                check=True,
            ).stdout

        git("init", "--quiet", "--template=")
        git("config", "user.name", "Fixture")
        git("config", "user.email", "fixture@example.invalid")
        (self.science / "src").mkdir()
        source = self.science / "src/science.py"
        source.write_bytes(b"VALUE = 1\n")
        git("add", ".")
        git("commit", "--quiet", "-m", "Scientific fixture")
        head = git("rev-parse", "HEAD").decode().strip()
        payload = {
            "schema": "quest-sdsc-git-provenance-v1",
            "git_head": head,
            "wrapper": {"code_sha256": "a" * 64},
            "bundle": {"sha256": "d" * 64},
            "scientific_files": [
                {
                    "path": "src/science.py",
                    "size": 10,
                    "sha256": worker.digest(source.read_bytes()),
                    "mode": 0o644,
                }
            ],
        }
        raw = json.dumps(payload).encode()
        self.manifest.write_bytes(raw)
        self.args.provenance_manifest_sha256 = worker.digest(raw)
        self.assertEqual(worker.validate_checkout(self.args)["science_git_head"], head)
        git("update-index", "--assume-unchanged", "src/science.py")
        source.write_bytes(b"VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "scientific file bytes differ"):
            worker.validate_checkout(self.args)

    def test_store_formal_and_population_bindings_are_rechecked(self):
        store = self.output / "teacher_demos"
        store.mkdir(parents=True)
        (store / "manifest.json").write_text("fixture")
        formal = {"code_commit": "c" * 40, "teacher_revision": "d" * 40}
        dataset = {
            "dataset_family_sha256": "a" * 64,
            "train_examples_file_sha256": "b" * 64,
            "ordered_prompt_ids": [str(i) for i in range(256)],
        }
        prompt = {"tokenizer_fingerprint": "f" * 64}
        manifest = {
            "protocol_bindings": {
                **formal,
                "dataset_family_sha256": dataset["dataset_family_sha256"],
                "train_examples_file_sha256": dataset["train_examples_file_sha256"],
            },
            "ordered_prompt_ids": dataset["ordered_prompt_ids"],
            "attempt_count": 2048,
            "tokenizer_hash": "f" * 64,
            "sha256": "0" * 64,
            "ledger_file_sha256": "1" * 64,
            "accepted_view_file_sha256": "2" * 64,
            "behavior_policy": {"id": "Qwen/Qwen3-8B", "revision": "d" * 40, "resolved_commit": "d" * 40},
            "teacher_demo_generation": {
                "sampling_request_seed": 31415,
                "candidates_per_prompt": 8,
                "max_prompt_tokens": 1246,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
        }
        calls = []

        def read(root, *, require_formal):
            calls.append(require_formal)
            return list(range(256)), manifest

        module = types.SimpleNamespace(read_teacher_demo_store=read)
        with patch.dict(sys.modules, {"posttrain_circuits.datasets.teacher_demos.store": module}):
            self.assertTrue(worker.validate_store(self.args, formal, dataset, prompt)["ready_for_formal_sft"])
            self.assertEqual(calls, [True])
            manifest["protocol_bindings"]["code_commit"] = "e" * 40
            with self.assertRaisesRegex(ValueError, "formal/dataset bindings differ"):
                worker.validate_store(self.args, formal, dataset, prompt)

    def test_prompt_length_preflight_only_loads_offline_tokenizer(self):
        config = {
            "teacher": {
                "tokenizer_name_or_path": "Qwen/Qwen3-8B",
                "tokenizer_revision": "d" * 40,
                "tokenizer_fingerprint": "f" * 64,
                "prompt_protocol": {"chat_template_sha256": "e" * 64},
            },
            "model": {"tokenizer_fingerprint": "f" * 64},
        }
        lengths = [1246]
        tokenizer = types.SimpleNamespace(
            pad_token_id=1, eos_token_id=2, encode=lambda *args, **kwargs: [1] * lengths[0]
        )
        kwargs_seen = []

        def tokenizer_load(*args, **kwargs):
            kwargs_seen.append(kwargs)
            return tokenizer

        modules = {
            "transformers": types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_load)
            ),
            "posttrain_circuits.datasets.proofgraph.generation": types.SimpleNamespace(
                ProofGraphTask=lambda: types.SimpleNamespace(render=lambda example: "prompt")
            ),
            "posttrain_circuits.models.loading": types.SimpleNamespace(
                tokenizer_fingerprint=lambda _: "f" * 64
            ),
            "posttrain_circuits.models.prompt_protocol": types.SimpleNamespace(
                chat_template_sha256=lambda _: "e" * 64,
                format_model_prompt=lambda *args: types.SimpleNamespace(model_facing_prompt="prompt"),
            ),
        }
        examples = [types.SimpleNamespace(example_id="example")]
        with patch.dict(sys.modules, modules):
            self.assertEqual(worker.validate_prompts(config, examples)["maximum_prompt_tokens"], 1246)
            self.assertTrue(kwargs_seen[0]["local_files_only"])
            self.assertFalse(kwargs_seen[0]["trust_remote_code"])
            lengths[0] = 1247
            with self.assertRaisesRegex(ValueError, "token bound"):
                worker.validate_prompts(config, examples)

    def test_runtime_metadata_rejects_smoke_container_without_importing_torch(self):
        with (
            patch.object(worker.platform, "python_version", return_value="3.12.3"),
            self.assertRaisesRegex(ValueError, "Python 3.12.13"),
        ):
            worker.runtime_identity()
        with (
            patch.object(worker.platform, "python_version", return_value="3.12.13"),
            patch.object(worker.importlib.metadata, "version", side_effect=worker.DEPENDENCIES.__getitem__),
        ):
            self.assertEqual(worker.runtime_identity()["packages"]["torch"], "2.8.0+cu128")

    def test_real_composer_namespace_and_frozen_science_digest(self):
        repository = SCRIPT.parent.parent
        python = repository / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("repository CPU dependency environment is unavailable")
        code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.artifacts.execution_science_protocol import canonical_science_config_sha256
overrides = ['g0=qwen3_v2_eap_separation', 'experiment=canonical_sft', 'task.num_examples=256',
             'state_source.num_candidates=8', 'seed=42',
             'protocol_amendment_path=prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml']
root = Path(sys.argv[1]) / 'configs'
try:
    compose_config(overrides + ['output_root=/tmp/outputs'], config_root=root)
except ValueError:
    pass
else:
    raise AssertionError('namespace-less output was accepted')
config = compose_config(overrides + ['output_root=/tmp/outputs/qwen3-v2'], config_root=root)
expected = '6c3942f4d6cf87329e1c70ba32b0b8d4cdbd526a3bbcfdcb1afd3e1674662657'
assert canonical_science_config_sha256(config) == expected
assert config['teacher']['generation_seed'] == 31415
assert sum(config['task']['split_sizes'].values()) == 144000
print('real-composer-passed')
"""
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", code, str(repository)], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "real-composer-passed")


if __name__ == "__main__":
    unittest.main()
