from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from posttrain_circuits.artifacts.execution_science_protocol import (
    APPROVED_STORAGE_LOCATOR_NAMES,
    PROPOSED_REVIEW,
    REVIEW_MECHANISM,
    SCHEDULER_PROVENANCE_NAMES,
    ExecutionScienceProtocolError,
    build_execution_science_protocol,
    canonical_science_config_projection,
    canonical_science_config_sha256,
    execution_science_protocol_core_sha256,
    load_execution_science_protocol_json_bytes,
    load_execution_science_protocol_yaml_bytes,
    resolve_accepted_execution_science_protocol,
    validate_count_neutral_identity,
    validate_execution_science_protocol_bytes,
    validate_execution_science_protocol_review_transition,
    validate_execution_science_protocol_relative_path,
    validate_hydra_override_vector,
)


IMPLEMENTATION_COMMIT = "a" * 40
ACCEPTANCE_COMMIT = "b" * 40
HASHES = {
    "descriptor": "1" * 64,
    "fingerprint": "2" * 64,
    "certification": "3" * 64,
    "certification_core": "4" * 64,
}


class ExecutionScienceProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolved_config = {
            "seed": 42,
            "output_root": "/data/del6500/OPD/runs/first",
            "model": {
                "model_name_or_path": "Qwen/Qwen3-1.7B",
                "model_revision": "revision-a",
            },
            "trainer": {
                "global_batch_size": 64,
                "max_model_input_length": 1536,
            },
            "state_source": {
                "name": "teacher_demo",
                "store_path": "/data/del6500/OPD/stores/first",
            },
            "task": {
                "name": "proofgraph",
                "dataset_family_path": "/data/del6500/OPD/datasets/first",
            },
            "scheduler_g0": {
                "request_git_commit": "c" * 40,
                "job_id": "opaque-job-a",
                "attempt": 1,
                "execution_safety_supported_world_sizes": [1, 2, 3, 4],
            },
            "workflow_id": "workflow-a",
            "unit_id": "g0",
        }
        self.overrides = [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "task.num_examples=256",
            "seed=42",
        ]
        self.proposed = build_execution_science_protocol(
            protocol_id="qwen3-v2-g0-seed-42-v1",
            unit_id="g0",
            seed=42,
            execution_class_id="qwen3-v2-elastic-training-v1",
            descriptor_path=(
                "prereg/execution_safety/"
                "qwen3_v2_elastic_training_v1.descriptor.json"
            ),
            descriptor_sha256=HASHES["descriptor"],
            fingerprint_sha256=HASHES["fingerprint"],
            certification_id=(
                "qwen3-v2-elastic-training-v1-initial-certification"
            ),
            certification_path=(
                "prereg/execution_safety/"
                "qwen3_v2_elastic_training_v1.certification.yaml"
            ),
            certification_core_sha256=HASHES["certification_core"],
            resolved_config=self.resolved_config,
            hydra_override_vector=self.overrides,
        )

    @staticmethod
    def _accepted(proposed: dict[str, object]) -> dict[str, object]:
        accepted = copy.deepcopy(proposed)
        accepted["review"] = {
            "status": "accepted",
            "reviewed_implementation_commit": IMPLEMENTATION_COMMIT,
            "reviewer": "independent-science-reviewer",
            "reviewed_at_utc": "2026-09-07T01:23:45Z",
            "rationale": "Accepted the exact scientific protocol binding.",
        }
        return accepted

    def test_builds_proposed_allocation_neutral_binding(self) -> None:
        self.assertEqual(self.proposed["review"], PROPOSED_REVIEW)
        self.assertEqual(
            self.proposed["review_contract"]["mechanism"], REVIEW_MECHANISM
        )
        self.assertEqual(
            self.proposed["execution_class"]["supported_world_sizes"],
            [1, 2, 3, 4],
        )
        serialized = json.dumps(self.proposed, sort_keys=True).lower()
        for forbidden_field in (
            '"execution_profile"',
            '"resources"',
            '"gpu_count"',
            '"gpu_uuid"',
            '"cuda_visible_devices"',
        ):
            self.assertNotIn(forbidden_field, serialized)
        self.assertEqual(
            self.proposed["science_config"]["excluded_storage_locator_paths"],
            list(APPROVED_STORAGE_LOCATOR_NAMES),
        )
        self.assertEqual(
            self.proposed["science_config"][
                "excluded_scheduler_provenance_paths"
            ],
            list(SCHEDULER_PROVENANCE_NAMES),
        )

    def test_json_and_yaml_round_trip_with_exact_schema(self) -> None:
        json_raw = (json.dumps(self.proposed, sort_keys=True) + "\n").encode()
        yaml_raw = yaml.safe_dump(self.proposed, sort_keys=False).encode()
        self.assertEqual(
            load_execution_science_protocol_json_bytes(json_raw), self.proposed
        )
        self.assertEqual(
            load_execution_science_protocol_yaml_bytes(yaml_raw), self.proposed
        )
        binding = validate_execution_science_protocol_bytes(
            yaml_raw, format="yaml"
        )
        self.assertEqual(binding.protocol_id, "qwen3-v2-g0-seed-42-v1")
        self.assertEqual(binding.unit_id, "g0")
        self.assertEqual(binding.seed, 42)
        self.assertEqual(binding.review_status, "proposed")
        self.assertEqual(binding.hydra_override_vector, tuple(self.overrides))
        self.assertEqual(len(binding.artifact_sha256), 64)

    def test_accepted_binding_requires_complete_review(self) -> None:
        accepted = self._accepted(self.proposed)
        raw = yaml.safe_dump(accepted, sort_keys=False).encode()
        binding = validate_execution_science_protocol_bytes(
            raw, require_accepted=True
        )
        self.assertEqual(binding.review_status, "accepted")
        self.assertEqual(
            binding.reviewed_implementation_commit, IMPLEMENTATION_COMMIT
        )
        with self.assertRaisesRegex(
            ExecutionScienceProtocolError, "remains proposed"
        ):
            validate_execution_science_protocol_bytes(
                yaml.safe_dump(self.proposed, sort_keys=False).encode(),
                require_accepted=True,
            )
        incomplete = copy.deepcopy(accepted)
        incomplete["review"]["reviewer"] = None
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "reviewer"):
            load_execution_science_protocol_yaml_bytes(
                yaml.safe_dump(incomplete, sort_keys=False).encode()
            )

    def test_review_transition_is_pure_and_non_self_referential(self) -> None:
        accepted = self._accepted(self.proposed)
        validate_execution_science_protocol_review_transition(
            proposed=self.proposed,
            accepted=accepted,
            implementation_commit=IMPLEMENTATION_COMMIT,
            acceptance_commit=ACCEPTANCE_COMMIT,
        )
        mutated = copy.deepcopy(accepted)
        mutated["scope"]["seed"] = 43
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "non-review"):
            validate_execution_science_protocol_review_transition(
                proposed=self.proposed,
                accepted=mutated,
                implementation_commit=IMPLEMENTATION_COMMIT,
                acceptance_commit=ACCEPTANCE_COMMIT,
            )
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "must differ"):
            validate_execution_science_protocol_review_transition(
                proposed=self.proposed,
                accepted=accepted,
                implementation_commit=IMPLEMENTATION_COMMIT,
                acceptance_commit=IMPLEMENTATION_COMMIT,
            )

    def test_core_digest_is_review_neutral(self) -> None:
        accepted = self._accepted(self.proposed)
        self.assertEqual(
            execution_science_protocol_core_sha256(self.proposed),
            execution_science_protocol_core_sha256(accepted),
        )

    def test_storage_and_scheduler_provenance_do_not_change_science_digest(self) -> None:
        relocated = copy.deepcopy(self.resolved_config)
        relocated["output_root"] = "/data/del6500/OPD/runs/second"
        relocated["state_source"]["store_path"] = "/scr/del6500/OPD/store"
        relocated["task"]["dataset_family_path"] = "/scr/del6500/OPD/data"
        relocated["scheduler_g0"] = {
            "request_git_commit": "d" * 40,
            "job_id": "opaque-job-b",
            "attempt": 9,
        }
        relocated["workflow_id"] = "workflow-b"
        relocated["unit_id"] = "different-storage-unit"
        self.assertEqual(
            canonical_science_config_sha256(relocated),
            canonical_science_config_sha256(self.resolved_config),
        )
        projection = canonical_science_config_projection(relocated)
        self.assertNotIn("output_root", projection)
        self.assertNotIn("scheduler_g0", projection)
        self.assertNotIn("workflow_id", projection)
        self.assertNotIn("store_path", projection["state_source"])

    def test_seed_and_scientific_config_remain_in_science_digest(self) -> None:
        changed_seed = copy.deepcopy(self.resolved_config)
        changed_seed["seed"] = 43
        changed_shape = copy.deepcopy(self.resolved_config)
        changed_shape["trainer"]["max_model_input_length"] = 2048
        base = canonical_science_config_sha256(self.resolved_config)
        self.assertNotEqual(canonical_science_config_sha256(changed_seed), base)
        self.assertNotEqual(canonical_science_config_sha256(changed_shape), base)

    def test_projection_does_not_mutate_resolved_config(self) -> None:
        original = copy.deepcopy(self.resolved_config)
        canonical_science_config_projection(self.resolved_config)
        self.assertEqual(self.resolved_config, original)

    def test_builder_rejects_resource_steering_hidden_in_config(self) -> None:
        for path, value in (
            (("resources",), {"gpu_count": 4}),
            (("trainer", "gpu_count"), 4),
            (("trainer", "world_size"), 4),
            (("trainer", "worldSize"), 4),
            (("torchrun", "nproc-per-node"), 4),
            (("runtime", "numGPUs"), 4),
            (("runtime", "local_rank"), 0),
            (("model", "device_map"), "auto"),
            (("runtime", "cuda_device"), 2),
            (("runtime", "cpu-thread-count"), 24),
            (("runtime", "host_memory_mib"), 196608),
        ):
            with self.subTest(path=path):
                config = copy.deepcopy(self.resolved_config)
                parent = config
                for part in path[:-1]:
                    parent = parent.setdefault(part, {})
                parent[path[-1]] = value
                with self.assertRaisesRegex(
                    ExecutionScienceProtocolError, "resource steering"
                ):
                    canonical_science_config_sha256(config)

    def test_override_vector_is_ordered_strict_and_resource_neutral(self) -> None:
        self.assertEqual(
            validate_hydra_override_vector(self.overrides), tuple(self.overrides)
        )
        invalid_vectors = (
            ["seed=42", "seed=43"],
            ["trainer.gpu_count=4"],
            ["trainer.world-size=4"],
            ["trainer.worldSize=4"],
            ["torchrun.nproc_per_node=4"],
            ["runtime.numGPUs=4"],
            ["runtime.local_rank=0"],
            ["model.device_map=auto"],
            ["runtime.cuda_device=2"],
            ["runtime.cpu_thread_count=24"],
            ["runtime.gpu_memory_mib=81920"],
            ["launcher={nproc_per_node: 4}"],
            ["execution_profile=large"],
            ["resources.memory_mib=4096"],
            ["server_scheduler.hint=fast"],
            [" seed=42"],
            ["seed"],
            ["seed=.nan"],
            ["+seed=42"],
        )
        for vector in invalid_vectors:
            with self.subTest(vector=vector), self.assertRaises(
                ExecutionScienceProtocolError
            ):
                validate_hydra_override_vector(vector)

    def test_exact_schema_rejects_resource_and_profile_fields(self) -> None:
        for key, value in (
            ("resources", {"gpu_count": 4}),
            ("execution_profile", "four-gpu"),
            ("gpu_count", 4),
        ):
            with self.subTest(key=key):
                tampered = copy.deepcopy(self.proposed)
                tampered[key] = value
                with self.assertRaisesRegex(
                    ExecutionScienceProtocolError, "top-level fields"
                ):
                    load_execution_science_protocol_json_bytes(
                        json.dumps(tampered).encode()
                    )

    def test_strict_decoders_reject_duplicates_and_nonfinite_values(self) -> None:
        duplicate_json = b'{"schema_version":1,"schema_version":1}'
        duplicate_yaml = b"schema_version: 1\nschema_version: 1\n"
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "duplicate"):
            load_execution_science_protocol_json_bytes(duplicate_json)
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "duplicate"):
            load_execution_science_protocol_yaml_bytes(duplicate_yaml)
        nonfinite = copy.deepcopy(self.proposed)
        nonfinite["scope"]["seed"] = float("nan")
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "finite"):
            load_execution_science_protocol_yaml_bytes(
                yaml.safe_dump(nonfinite, sort_keys=False).encode()
            )

    def test_invalid_reference_and_exclusion_metadata_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.proposed)
        tampered["execution_class"]["descriptor_path"] = "../descriptor.json"
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "relative path"):
            load_execution_science_protocol_yaml_bytes(
                yaml.safe_dump(tampered, sort_keys=False).encode()
            )
        tampered = copy.deepcopy(self.proposed)
        tampered["science_config"]["excluded_storage_locator_paths"] = [
            "output_root",
            "unreviewed.path",
        ]
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "approved set"):
            load_execution_science_protocol_yaml_bytes(
                yaml.safe_dump(tampered, sort_keys=False).encode()
            )
        tampered = copy.deepcopy(self.proposed)
        tampered["execution_class"]["supported_world_sizes"] = [True, 2, 3, 4]
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "world sizes"):
            load_execution_science_protocol_json_bytes(json.dumps(tampered).encode())

    def test_safety_fingerprint_is_referenced_but_not_recomputed_here(self) -> None:
        changed = copy.deepcopy(self.proposed)
        changed["execution_class"]["fingerprint_sha256"] = "f" * 64
        parsed = load_execution_science_protocol_json_bytes(
            json.dumps(changed).encode()
        )
        self.assertEqual(
            parsed["execution_class"]["fingerprint_sha256"], "f" * 64
        )
        self.assertNotEqual(
            execution_science_protocol_core_sha256(changed),
            execution_science_protocol_core_sha256(self.proposed),
        )

    def test_protocol_identity_and_filename_must_be_count_neutral(self) -> None:
        for value in ("experiment-w3", "experiment-4gpu", "gpu_2-pilot"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ExecutionScienceProtocolError, "GPU count"
            ):
                validate_count_neutral_identity(value, name="test identity")
        self.assertEqual(
            validate_count_neutral_identity(
                "qwen3-v2-g0-seed-42-v1", name="test identity"
            ),
            "qwen3-v2-g0-seed-42-v1",
        )


class ExecutionScienceProtocolGitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="science-protocol-git-")
        self.root = Path(self.temporary.name)
        self.relative = "prereg/execution_science/experiment_seed42_v1.yaml"
        self.path = self.root / self.relative
        self.path.parent.mkdir(parents=True)
        (self.root / "docs/refactor").mkdir(parents=True)
        (self.root / "docs/refactor/current_handoff.md").write_text(
            "initial\n", encoding="utf-8"
        )
        (self.root / "prereg/amendments").mkdir(parents=True)
        (self.root / "prereg/execution_safety").mkdir(parents=True)
        self._git("init", "-q")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "user.name=Science Test",
                "-c",
                "user.email=science@example.invalid",
                *arguments,
            ),
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _proposed(self) -> dict[str, object]:
        config = {
            "seed": 42,
            "model": {"name": "student"},
            "trainer": {"global_batch_size": 64},
        }
        return build_execution_science_protocol(
            protocol_id="experiment-seed42-v1",
            unit_id="g0",
            seed=42,
            execution_class_id="qwen3-v2-elastic-training-v1",
            descriptor_path="prereg/execution_safety/descriptor.json",
            descriptor_sha256="1" * 64,
            fingerprint_sha256="2" * 64,
            certification_id="qwen3-v2-elastic-certification",
            certification_path="prereg/execution_safety/certification.yaml",
            certification_core_sha256="4" * 64,
            resolved_config=config,
            hydra_override_vector=["seed=42"],
        )

    def _write_yaml(self, path: Path, payload: object) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _commit_proposed(self) -> tuple[dict[str, object], str]:
        proposed = self._proposed()
        self._write_yaml(self.path, proposed)
        self._git("add", ".")
        self._git("commit", "-q", "-m", "implement science protocol")
        return proposed, self._git("rev-parse", "HEAD")

    def _accept(
        self,
        proposed: dict[str, object],
        implementation: str,
        *,
        mutate_science: bool = False,
        joint_class_review: bool = False,
    ) -> str:
        accepted = copy.deepcopy(proposed)
        accepted["review"] = {
            "status": "accepted",
            "reviewed_implementation_commit": implementation,
            "reviewer": "independent-science-reviewer",
            "reviewed_at_utc": "2026-09-07T01:23:45Z",
            "rationale": "Accepted exact experiment science.",
        }
        if mutate_science:
            accepted["scope"]["seed"] = 43
        self._write_yaml(self.path, accepted)
        if joint_class_review:
            amendment = self.root / (
                "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml"
            )
            certification = self.root / (
                "prereg/execution_safety/"
                "qwen3_v2_elastic_training_v1.certification.yaml"
            )
            amendment.write_text("review: accepted\n", encoding="utf-8")
            certification.write_text("review: accepted\n", encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "-q", "-m", "accept science protocol")
        return self._git("rev-parse", "HEAD")

    def test_resolves_review_acceptance_after_intervening_commit(self) -> None:
        proposed, implementation = self._commit_proposed()
        (self.root / "docs/refactor/current_handoff.md").write_text(
            "intervening handoff\n", encoding="utf-8"
        )
        self._git("add", "docs/refactor/current_handoff.md")
        self._git("commit", "-q", "-m", "update handoff")
        acceptance = self._accept(proposed, implementation)
        binding = resolve_accepted_execution_science_protocol(
            code_root=self.root,
            configured_path=self.relative,
            expected_head=acceptance,
        )
        self.assertEqual(binding.acceptance_commit, acceptance)
        self.assertEqual(binding.reviewed_implementation_commit, implementation)
        self.assertEqual(binding.binding.protocol_id, "experiment-seed42-v1")

    def test_joint_execution_class_review_paths_are_allowed(self) -> None:
        proposed, implementation = self._commit_proposed()
        acceptance = self._accept(
            proposed,
            implementation,
            joint_class_review=True,
        )
        binding = resolve_accepted_execution_science_protocol(
            code_root=self.root,
            configured_path=self.relative,
        )
        self.assertEqual(binding.acceptance_commit, acceptance)

    def test_rejects_non_review_acceptance_and_later_artifact_change(self) -> None:
        proposed, implementation = self._commit_proposed()
        self._accept(proposed, implementation, mutate_science=True)
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "non-review"):
            resolve_accepted_execution_science_protocol(
                code_root=self.root,
                configured_path=self.relative,
            )

        # Rebuild a valid history, then prove a later touch cannot be hidden.
        self.tearDown()
        self.setUp()
        proposed, implementation = self._commit_proposed()
        self._accept(proposed, implementation)
        accepted = self.path.read_text(encoding="utf-8")
        self.path.write_text(accepted + "\n", encoding="utf-8")
        self._git("add", self.relative)
        self._git("commit", "-q", "-m", "touch reviewed protocol")
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "exactly once"):
            resolve_accepted_execution_science_protocol(
                code_root=self.root,
                configured_path=self.relative,
            )

    def test_rejects_scientific_source_change_after_review(self) -> None:
        proposed, implementation = self._commit_proposed()
        self._accept(proposed, implementation)
        source = self.root / "src" / "experiment.py"
        source.parent.mkdir(parents=True)
        source.write_text("SCIENTIFIC_BEHAVIOR = 2\n", encoding="utf-8")
        self._git("add", "src/experiment.py")
        self._git("commit", "-q", "-m", "change scientific implementation")
        with self.assertRaisesRegex(
            ExecutionScienceProtocolError, "implementation changed after review"
        ):
            resolve_accepted_execution_science_protocol(
                code_root=self.root,
                configured_path=self.relative,
            )

    def test_rejects_dirty_checkout_and_unconfined_or_count_shaped_path(self) -> None:
        proposed, implementation = self._commit_proposed()
        self._accept(proposed, implementation)
        (self.root / "docs/refactor/current_handoff.md").write_text(
            "tracked dirty change\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ExecutionScienceProtocolError, "clean checkout"):
            resolve_accepted_execution_science_protocol(
                code_root=self.root,
                configured_path=self.relative,
            )
        for path in (
            "../experiment.yaml",
            "prereg/execution_science/nested/experiment.yaml",
            "prereg/execution_science/experiment_w4.yaml",
        ):
            with self.subTest(path=path), self.assertRaises(
                ExecutionScienceProtocolError
            ):
                validate_execution_science_protocol_relative_path(path)


if __name__ == "__main__":
    unittest.main()
