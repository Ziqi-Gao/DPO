"""CPU regressions for the candidate runtime-field binding; no model loading."""

import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_science_binding.py"
SPEC = importlib.util.spec_from_file_location("science_binding", SCRIPT)
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)


def projection(config):
    projected = copy.deepcopy(config)
    projected.pop("output_root", None)
    projected["production_safety"].pop("initial_checkpoint_path", None)
    return projected


class ScienceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "qwen3-v2"
        self.output.mkdir()
        self.checkpoint = self.output / "initial_checkpoint.pt"
        self.checkpoint.write_bytes(b"fixture: already independently validated initial model")
        self.checkpoint_hash = binding.digest(self.checkpoint.read_bytes())
        self.base = {
            "seed": 42,
            "trainer": {"global_batch_size": 64, "token_budget": 2000000, "max_steps": 120},
            "output_root": "/approved/base/qwen3-v2",
            "production_safety": {"initial_checkpoint_path": "", "initial_checkpoint_hash": ""},
        }
        self.runtime = copy.deepcopy(self.base)
        self.runtime["output_root"] = str(self.output)
        self.runtime["production_safety"] = {
            "initial_checkpoint_path": str(self.checkpoint),
            "initial_checkpoint_hash": self.checkpoint_hash,
        }
        self.base_hash = binding.digest(binding.canonical(projection(self.base)))

    def verify(self, **changes):
        arguments = {
            "expected_base_science_sha256": self.base_hash,
            "expected_checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_path": self.checkpoint,
            "science_projection": projection,
        }
        arguments.update(changes)
        return binding.verify_runtime_binding(self.base, self.runtime, **arguments)

    def test_only_verified_runtime_field_differs_and_inputs_remain_unchanged(self):
        original_base, original_runtime = copy.deepcopy(self.base), copy.deepcopy(self.runtime)
        result = self.verify()
        self.assertEqual(self.base, original_base)
        self.assertEqual(self.runtime, original_runtime)
        self.assertEqual(result["base_science_config_sha256"], self.base_hash)
        self.assertNotEqual(result["runtime_science_config_sha256"], self.base_hash)
        self.assertEqual(result["initial_checkpoint"]["sha256"], self.checkpoint_hash)
        self.assertEqual(result["normalized_fields"], ["production_safety.initial_checkpoint_hash"])
        self.assertFalse(result["checkpoint_content_semantics_verified"])
        for field in (
            "g0_passed",
            "pilot_passed",
            "factorial_ready",
            "execution_class_certified",
            "old_finalizer_modified",
        ):
            self.assertFalse(result[field])
        self.assertEqual(
            result["sha256"],
            binding.digest(
                binding.canonical({key: value for key, value in result.items() if key != "sha256"})
            ),
        )

    def test_corrupted_actual_checkpoint_cannot_be_normalized_away(self):
        self.checkpoint.write_bytes(b"X" * self.checkpoint.stat().st_size)
        with self.assertRaisesRegex(ValueError, "actual checkpoint bytes"):
            self.verify()

    def test_config_and_artifact_agreement_does_not_replace_external_hash(self):
        self.checkpoint.write_bytes(b"a substituted checkpoint")
        self.runtime["production_safety"]["initial_checkpoint_hash"] = binding.digest(
            self.checkpoint.read_bytes()
        )
        with self.assertRaisesRegex(ValueError, "independently expected"):
            self.verify()

    def test_empty_missing_or_invalid_runtime_hash_is_rejected(self):
        for value in ("", None, "TODO", 123):
            with self.subTest(value=value):
                self.runtime["production_safety"]["initial_checkpoint_hash"] = value
                with self.assertRaisesRegex(ValueError, "independently expected"):
                    self.verify()

    def test_other_scientific_changes_are_rejected(self):
        baseline = copy.deepcopy(self.runtime)
        changes = [
            ("seed", 43),
            ("trainer", {**baseline["trainer"], "max_steps": 119}),
            ("trainer", {**baseline["trainer"], "global_batch_size": 32}),
            ("trainer", {**baseline["trainer"], "token_budget": 1999999}),
            ("unreviewed_exclusion", "anything"),
        ]
        for name, value in changes:
            with self.subTest(name=name, value=value):
                self.runtime = {**baseline, name: value}
                with self.assertRaisesRegex(ValueError, "other than"):
                    self.verify()

    def test_type_changes_cannot_hide_behind_python_equality(self):
        self.base["scientific_switch"] = 1
        self.base_hash = binding.digest(binding.canonical(projection(self.base)))
        self.runtime["scientific_switch"] = True
        with self.assertRaisesRegex(ValueError, "other than"):
            self.verify()

    def test_modified_base_is_rejected_even_when_runtime_matches_it(self):
        self.base["seed"] = self.runtime["seed"] = 99
        with self.assertRaisesRegex(ValueError, "accepted hash"):
            self.verify()

    def test_base_must_retain_its_empty_runtime_field(self):
        self.base["production_safety"]["initial_checkpoint_hash"] = self.checkpoint_hash
        with self.assertRaisesRegex(ValueError, "frozen empty"):
            self.verify()

    def test_wrong_checkpoint_locator_is_rejected_despite_same_bytes(self):
        other = self.output / "other.pt"
        other.write_bytes(self.checkpoint.read_bytes())
        with self.assertRaisesRegex(ValueError, "workspace mapping"):
            self.verify(checkpoint_path=other)
        self.runtime["production_safety"]["initial_checkpoint_path"] = str(other)
        with self.assertRaisesRegex(ValueError, "exact attempt locator"):
            self.verify(checkpoint_path=other)

    def test_replay_requires_explicit_mapping_and_original_relative_filename(self):
        extracted = self.root / "extracted"
        extracted.mkdir()
        actual = extracted / "initial_checkpoint.pt"
        actual.write_bytes(self.checkpoint.read_bytes())
        with self.assertRaisesRegex(ValueError, "workspace mapping"):
            self.verify(checkpoint_path=actual)
        result = self.verify(checkpoint_path=actual, relocated_workspace=extracted)
        self.assertEqual(result["configured_initial_checkpoint_path"], str(self.checkpoint))
        self.assertEqual(result["initial_checkpoint"]["path"], str(actual))
        self.assertEqual(result["explicit_relocated_workspace"], str(extracted))

    def test_symlink_file_and_parent_are_rejected(self):
        original = self.output / "original.pt"
        self.checkpoint.rename(original)
        self.checkpoint.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.verify()
        linked = self.root / "linked-output"
        linked.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            binding.file_identity(linked / "original.pt")

    def test_fifo_and_empty_or_oversized_files_are_rejected_without_blocking(self):
        empty = self.root / "empty"
        empty.touch()
        with self.assertRaisesRegex(ValueError, "nonempty regular"):
            binding.file_identity(empty)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular"):
            binding.file_identity(fifo)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            binding.file_identity(self.checkpoint, limit=2)

    def test_replaced_inode_is_rejected_even_with_same_content(self):
        replacement = self.output / "replacement.pt"
        replacement.write_bytes(self.checkpoint.read_bytes())
        actual_fstat = os.fstat
        calls = 0

        def replace_on_second_read(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                os.replace(replacement, self.checkpoint)
            return actual_fstat(fd)

        with (
            patch.object(binding.os, "fstat", side_effect=replace_on_second_read),
            self.assertRaisesRegex(ValueError, "changed during"),
        ):
            self.verify()

    def test_duplicate_or_nonfinite_json_fields_are_rejected(self):
        path = self.root / "binding.json"
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                binding.json_document(path)

    def test_new_head_does_not_relabel_historical_teacher_origin(self):
        expected = "a" * 40
        original = {"code_commit": expected, "model_revision": "pinned"}
        self.assertEqual(
            binding.producer_origin([original], expected_science_head=expected)[0]["producer_code_commit"],
            expected,
        )
        with self.assertRaisesRegex(ValueError, "cross-commit review"):
            binding.producer_origin([original], expected_science_head="b" * 40)
        self.assertEqual(original["code_commit"], expected)
        with self.assertRaisesRegex(ValueError, "formal model"):
            binding.producer_origin(
                [original],
                expected_science_head=expected,
                expected_formal_binding={"model_revision": "different"},
            )

    def test_producer_formal_binding_distinguishes_booleans_numbers_and_missing_fields(self):
        head = "a" * 40
        expected = {"code_commit": head, "enable_thinking": False}
        for actual in ({"code_commit": head, "enable_thinking": 0}, {"code_commit": head}):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "formal model"):
                binding.producer_origin(
                    [actual], expected_science_head=head, expected_formal_binding=expected
                )
        self.assertTrue(
            binding.producer_origin([expected], expected_science_head=head, expected_formal_binding=expected)
        )

    def test_assume_unchanged_cannot_hide_modified_science(self):
        repo = self.root / "science"
        repo.mkdir()
        for command in (
            ["git", "init", "--quiet", "--template=", str(repo)],
            ["git", "-C", str(repo), "config", "user.name", "fixture"],
            ["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"],
        ):
            subprocess.run(command, check=True, capture_output=True)
        (repo / "src").mkdir()
        source = repo / "src/science.py"
        source.write_text("accepted = True\n")
        for command in (
            ["add", "src/science.py"],
            ["commit", "-qm", "fixture"],
            ["update-index", "--assume-unchanged", "src/science.py"],
        ):
            subprocess.run(["git", "-C", str(repo), *command], check=True, capture_output=True)
        head = binding.git(repo, "rev-parse", "HEAD").decode().strip()
        binding.verify_source_bytes(repo, head)
        source.write_text("accepted = False\n")
        self.assertEqual(binding.git(repo, "status", "--porcelain"), b"")
        with self.assertRaisesRegex(ValueError, "Uncommitted scientific"):
            binding.verify_source_bytes(repo, head)

    def test_real_composer_reproduces_old_gate_and_accepts_only_bound_hash(self):
        repository = SCRIPT.parent.parent
        python = repository / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("repository CPU dependency environment unavailable")
        code = """
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'src'))
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.artifacts.execution_science_protocol import (
    canonical_science_config_projection, canonical_science_config_sha256)
spec = importlib.util.spec_from_file_location('candidate', repo / 'tools/sdsc_science_binding.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
base = compose_config(['g0=qwen3_v2_eap_separation', 'experiment=canonical_sft',
    'task.num_examples=256', 'state_source.num_candidates=8', 'seed=42',
    'protocol_amendment_path=' + candidate.AMENDMENT], config_root=repo / 'configs')
assert canonical_science_config_sha256(base) == candidate.FROZEN_BASE_SHA256
with tempfile.TemporaryDirectory() as temporary:
    workspace = Path(temporary)
    checkpoint = workspace / 'initial_checkpoint.pt'
    checkpoint.write_bytes(b'no model: fixture for byte identity only')
    actual = candidate.file_identity(checkpoint)['sha256']
    runtime = copy.deepcopy(base)
    runtime['output_root'] = str(workspace)
    runtime['production_safety']['initial_checkpoint_path'] = str(checkpoint)
    runtime['production_safety']['initial_checkpoint_hash'] = actual
    assert canonical_science_config_sha256(runtime) != candidate.FROZEN_BASE_SHA256
    kwargs = dict(expected_base_science_sha256=candidate.FROZEN_BASE_SHA256,
        expected_checkpoint_sha256=actual, checkpoint_path=checkpoint,
        science_projection=canonical_science_config_projection)
    evidence = candidate.verify_runtime_binding(base, runtime, **kwargs)
    assert evidence['configuration_binding_validated'] and not evidence['g0_passed']
    assert runtime['production_safety']['initial_checkpoint_hash'] == actual
    deltas = [('trainer','token_budget',1999999), ('trainer','max_steps',119),
        ('trainer','global_batch_size',32), ('model','model_revision','a'*40),
        ('teacher','generation_seed',31416), ('supervision','normalization','token'),
        ('state_source','max_prompt_tokens',1245)]
    for section, key, value in deltas:
        changed = copy.deepcopy(runtime)
        changed[section][key] = value
        try:
            candidate.verify_runtime_binding(base, changed, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('unreviewed scientific delta was accepted: ' + key)
    print(json.dumps({'base': evidence['base_science_config_sha256'], 'no_model_loaded': True}))
"""
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", code, str(repository)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        self.assertEqual(json.loads(result.stdout)["base"], binding.FROZEN_BASE_SHA256)


if __name__ == "__main__":
    unittest.main()
