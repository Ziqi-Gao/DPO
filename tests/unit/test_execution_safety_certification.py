from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.protocol_amendments import (
    CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256,
    EXECUTION_CLASS_AMENDMENT_ID,
    PROPOSED_REVIEW,
    ProtocolAmendmentError,
    candidate_e_scientific_config_sha256,
    load_execution_class_amendment_bytes,
    load_protocol_amendment_bytes,
    validate_elastic_g0_config,
    validate_execution_class_review_transition,
)
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.learning.training import evaluation as evaluation_module
from posttrain_circuits.learning.training.execution_safety_kernel import (
    distributed_resume_checks,
)
from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_RELATIVE_PATH,
    DESCRIPTOR_RELATIVE_PATH,
    EXECUTION_CLASS_ID,
    IMPLEMENTATION_FILE_PATHS,
    INVALIDATING_DIMENSIONS,
    NON_INVALIDATING_DIMENSIONS,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
    ExecutionSafetyCertificationError,
    build_execution_safety_descriptor,
    condense_legacy_gpu_preflight_evidence,
    execution_safety_config_projection,
    execution_safety_world_size_contract,
    load_execution_safety_certification_bytes,
    validate_execution_safety_descriptor_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_COMMIT = "a" * 40
ACCEPTANCE_COMMIT = "b" * 40
LEGACY_EVIDENCE_PATHS = {
    1: (
        Path("/data/del6500/OPD/workflows/plans/qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407/b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72.json"),
        Path("/data/del6500/OPD/workflows/outputs/qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407/b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72/gpu-preflight/gpu_preflight.json"),
        Path("/data/del6500/OPD/workflows/completions/qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407/b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72/gpu-preflight.json"),
    ),
    2: (
        Path("/data/del6500/OPD/workflows/plans/qwen3-v2-gpu-preflight-elastic-700b6054e4dcfff45a56841d6a90dbce/2107c67c07132a2a6a4383c6bd4ebf78f79961d5ed1ce8b920d9c545d414886d.json"),
        Path("/data/del6500/OPD/workflows/outputs/qwen3-v2-gpu-preflight-elastic-700b6054e4dcfff45a56841d6a90dbce/2107c67c07132a2a6a4383c6bd4ebf78f79961d5ed1ce8b920d9c545d414886d/gpu-preflight/gpu_preflight.json"),
        Path("/data/del6500/OPD/workflows/completions/qwen3-v2-gpu-preflight-elastic-700b6054e4dcfff45a56841d6a90dbce/2107c67c07132a2a6a4383c6bd4ebf78f79961d5ed1ce8b920d9c545d414886d/gpu-preflight.json"),
    ),
}


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


class ExecutionSafetyCertificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = build_execution_safety_descriptor(PROJECT_ROOT)
        self.descriptor_raw = _json_bytes(self.descriptor)
        self.certificate = yaml.safe_load(
            (PROJECT_ROOT / CERTIFICATION_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        self.certificate["execution_class"]["descriptor_sha256"] = __import__(
            "hashlib"
        ).sha256(self.descriptor_raw).hexdigest()
        self.certificate["execution_class"]["fingerprint_sha256"] = self.descriptor[
            "fingerprint_sha256"
        ]
        self.certificate["legacy_evidence_condensation"][
            "attested_execution_fingerprint_sha256"
        ] = self.descriptor["fingerprint_sha256"]

    def _certificate_raw(self, *, accepted: bool) -> bytes:
        payload = copy.deepcopy(self.certificate)
        if accepted:
            payload["review"] = {
                "status": "accepted",
                "reviewed_implementation_commit": IMPLEMENTATION_COMMIT,
                "reviewer": "independent-execution-safety-reviewer",
                "reviewed_at_utc": "2026-09-06T23:00:00Z",
                "rationale": "Accepted exact reusable execution class.",
            }
        return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")

    def test_descriptor_recomputes_from_checkout_and_covers_required_surfaces(self) -> None:
        binding = validate_execution_safety_descriptor_bytes(
            self.descriptor_raw, code_root=PROJECT_ROOT
        )
        self.assertEqual(binding.execution_class_id, EXECUTION_CLASS_ID)
        self.assertEqual(binding.supported_world_sizes, (1, 2, 3, 4))
        subject = binding.payload["subject"]
        for field in (
            "implementation_files",
            "fixed_runtime",
            "model_and_sequence_shape",
            "batch_partition",
            "loss_scaling",
            "token_accounting",
            "fsdp",
            "checkpoint_resume",
            "memory_envelope",
            "optimizer_and_precision",
            "supported_world_sizes",
        ):
            self.assertIn(field, subject)
        self.assertIn(
            "scripts/server_scheduler/qwen3-v2-g0-handler.py",
            subject["implementation_files"],
        )
        self.assertEqual(
            set(subject["implementation_files"]), set(IMPLEMENTATION_FILE_PATHS)
        )
        self.assertIn(
            "src/posttrain_circuits/scheduler_adapter/registry.py",
            subject["implementation_files"],
        )
        self.assertNotIn(
            "configs/model/qwen3_v2_teacher_8b.yaml",
            subject["implementation_files"],
        )

    def test_certification_reader_rejects_symlinked_file_and_ancestor(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".execution-cert-reader-", dir=PROJECT_ROOT
        ) as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            target = real / "certificate.yaml"
            target.write_bytes(b"review: proposed\n")
            self.assertEqual(
                read_regular_bytes_nofollow(target, context="certificate"),
                b"review: proposed\n",
            )
            file_link = root / "file-link.yaml"
            file_link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "no-follow"):
                read_regular_bytes_nofollow(file_link, context="certificate")
            directory_link = root / "directory-link"
            directory_link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "no-follow"):
                read_regular_bytes_nofollow(
                    directory_link / target.name, context="certificate"
                )

    def test_distributed_evaluation_respects_certified_sequence_envelope(self) -> None:
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            @staticmethod
            def __call__(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    input_ids=torch.ones((1, 1536), dtype=torch.long)
                )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(max_position_embeddings=32768)
                self.generated = False

            def generate(self, **kwargs: object) -> torch.Tensor:
                self.generated = True
                return torch.ones((1, 1537), dtype=torch.long)

        task = SimpleNamespace(render=lambda _example: "prompt")
        with (
            mock.patch.object(evaluation_module, "ProofGraphTask", return_value=task),
            mock.patch.object(
                evaluation_module,
                "format_model_prompt",
                return_value=SimpleNamespace(model_facing_prompt="prompt"),
            ),
        ):
            evaluator = evaluation_module.build_proofgraph_evaluator(
                [object()],
                Tokenizer(),
                max_completion_length=128,
                max_model_input_length=1536,
            )
            model = Model()
            with self.assertRaisesRegex(ValueError, "model-input envelope"):
                evaluator(model)
            self.assertIs(model.generated, False)

    def test_candidate_e_science_is_separate_from_topology_certificate(self) -> None:
        config = compose_config(
            [
                "g0=qwen3_v2_eap_separation",
                "experiment=canonical_sft",
                "task.num_examples=256",
                "state_source.num_candidates=8",
            ]
        )
        self.assertEqual(
            candidate_e_scientific_config_sha256(config),
            CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256,
        )
        relocated = copy.deepcopy(config)
        relocated["output_root"] = "/data/different/output"
        relocated["state_source"]["store_path"] = "/data/different/store"
        self.assertEqual(
            candidate_e_scientific_config_sha256(relocated),
            CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256,
        )
        changed_seed = copy.deepcopy(config)
        changed_seed["seed"] = 43
        self.assertNotEqual(
            candidate_e_scientific_config_sha256(changed_seed),
            CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256,
        )
        base_safety = execution_safety_config_projection(config)
        self.assertEqual(
            execution_safety_config_projection(changed_seed), base_safety
        )
        renamed_experiment = copy.deepcopy(config)
        renamed_experiment["experiment"]["name"] = "replication_seed_43"
        self.assertEqual(
            execution_safety_config_projection(renamed_experiment), base_safety
        )
        relocated_safety = execution_safety_config_projection(relocated)
        self.assertEqual(relocated_safety, base_safety)
        changed_shape = copy.deepcopy(config)
        changed_shape["trainer"]["max_model_input_length"] = 2048
        self.assertNotEqual(
            execution_safety_config_projection(changed_shape), base_safety
        )
        bounded_science_change = copy.deepcopy(config)
        bounded_science_change["state_source"]["num_candidates"] = 4
        bounded_science_change["task"]["depth_range"] = [3, 5]
        bounded_science_change["task"]["structures"] = ["chain"]
        bounded_science_change["task"]["num_examples"] = 128
        bounded_science_change["task"]["split_sizes"]["train"] = 90000
        self.assertEqual(
            execution_safety_config_projection(bounded_science_change), base_safety
        )
        changed_memory = copy.deepcopy(config)
        changed_memory["state_source"]["num_candidates"] = 16
        with self.assertRaisesRegex(
            ExecutionSafetyCertificationError, "candidate count"
        ):
            execution_safety_config_projection(changed_memory)
        changed_dataset_envelope = copy.deepcopy(config)
        changed_dataset_envelope["task"]["split_sizes"]["train"] = 200000
        with self.assertRaisesRegex(
            ExecutionSafetyCertificationError, "split sizes"
        ):
            execution_safety_config_projection(changed_dataset_envelope)
        unaligned_population = copy.deepcopy(config)
        unaligned_population["task"]["num_examples"] = 255
        with self.assertRaisesRegex(
            ExecutionSafetyCertificationError, "prompt population"
        ):
            execution_safety_config_projection(unaligned_population)
        self.assertNotIn(
            "configs/g0/qwen3_v2_eap_separation.yaml",
            self.descriptor["subject"]["implementation_files"],
        )

    def test_world_size_contracts_are_exact_and_include_three_rank_tail(self) -> None:
        expected_samples = {
            1: [64],
            2: [32, 32],
            3: [22, 21, 21],
            4: [16, 16, 16, 16],
        }
        for world_size, samples in expected_samples.items():
            with self.subTest(world_size=world_size):
                contract = execution_safety_world_size_contract(
                    self.descriptor, world_size
                )
                self.assertEqual(contract["rank_local_samples"], samples)
                self.assertEqual(contract["fsdp_wrapper_count"], 29)
                self.assertEqual(contract["wrapped_transformer_blocks"], 28)
                self.assertEqual(
                    contract["effective_fsdp_sharding_strategy"],
                    "NO_SHARD" if world_size == 1 else "FULL_SHARD",
                )

    def test_resume_evidence_is_bound_to_the_actual_world_size(self) -> None:
        resume = {
            "format_version": 3,
            "passed": True,
            "world_size": 2,
            "max_optimizer_steps": 120,
            "checks": {"checkpoint_state_exact": True},
            "fsdp_sharding_contract": {
                "requested_fsdp_sharding_strategy": "FULL_SHARD",
                "effective_fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_wrapper_count": 29,
            },
        }
        self.assertTrue(
            all(
                distributed_resume_checks(
                    resume,
                    max_optimizer_steps=120,
                    world_size=2,
                    certified_fsdp_wrapper_count=29,
                ).values()
            )
        )
        self.assertFalse(
            all(
                distributed_resume_checks(
                    resume,
                    max_optimizer_steps=120,
                    world_size=3,
                    certified_fsdp_wrapper_count=29,
                ).values()
            )
        )

    def test_safety_change_invalidates_descriptor(self) -> None:
        tampered = copy.deepcopy(self.descriptor)
        tampered["subject"]["model_and_sequence_shape"][
            "max_model_input_length"
        ] = 2048
        tampered["fingerprint_sha256"] = __import__(
            "posttrain_circuits.artifacts.hashing", fromlist=["sha256_value"]
        ).sha256_value(tampered["subject"])
        with self.assertRaisesRegex(
            ExecutionSafetyCertificationError, "semantic subject"
        ):
            validate_execution_safety_descriptor_bytes(_json_bytes(tampered))

    def test_job_seed_replication_and_output_are_explicitly_non_invalidating(self) -> None:
        partition = self.descriptor["subject"]["parameter_partition"]
        self.assertEqual(
            partition["non_invalidating_dimensions"],
            list(NON_INVALIDATING_DIMENSIONS),
        )
        self.assertEqual(
            partition["invalidating_dimensions"], list(INVALIDATING_DIMENSIONS)
        )
        for name in ("job_id", "seed", "replication_id", "output_directory"):
            self.assertNotIn(name, self.descriptor["subject"]["implementation_files"])

    def test_proposed_certificate_is_not_operational(self) -> None:
        with self.assertRaisesRegex(
            ExecutionSafetyCertificationError, "remains proposed"
        ):
            load_execution_safety_certification_bytes(
                self._certificate_raw(accepted=False),
                self.descriptor_raw,
                require_accepted=True,
                code_root=PROJECT_ROOT,
            )

    def test_accepted_certificate_binds_exact_descriptor(self) -> None:
        binding = load_execution_safety_certification_bytes(
            self._certificate_raw(accepted=True),
            self.descriptor_raw,
            require_accepted=True,
            code_root=PROJECT_ROOT,
        )
        self.assertEqual(binding.fingerprint_sha256, self.descriptor["fingerprint_sha256"])
        self.assertEqual(binding.reviewed_implementation_commit, IMPLEMENTATION_COMMIT)
        self.assertEqual(
            binding.evidence_basis["real_gpu_pilots"]["3"]["status"],
            "not_yet_observed",
        )
        self.assertIs(
            self.certificate["legacy_evidence_condensation"][
                "source_reports_embed_execution_fingerprint"
            ],
            False,
        )

    @unittest.skipUnless(
        all(path.is_file() for pair in LEGACY_EVIDENCE_PATHS.values() for path in pair),
        "host legacy W1/W2 evidence is unavailable",
    )
    def test_checked_in_w1_w2_condensation_is_reproduced_from_raw_evidence(self) -> None:
        for world_size, (
            plan_path,
            report_path,
            completion_path,
        ) in LEGACY_EVIDENCE_PATHS.items():
            with self.subTest(world_size=world_size):
                condensed = condense_legacy_gpu_preflight_evidence(
                    plan_path=plan_path,
                    report_path=report_path,
                    completion_path=completion_path,
                    expected_world_size=world_size,
                    descriptor_raw=self.descriptor_raw,
                    code_root=PROJECT_ROOT,
                )
                self.assertEqual(
                    condensed["real_gpu_pilot"],
                    self.certificate["evidence_basis"]["real_gpu_pilots"][
                        str(world_size)
                    ],
                )
                self.assertEqual(
                    condensed["legacy_evidence_condensation"],
                    self.certificate["legacy_evidence_condensation"]["evidence"][
                        str(world_size)
                    ],
                )

    def test_certificate_rejects_descriptor_substitution(self) -> None:
        other = copy.deepcopy(self.descriptor)
        other["subject"]["implementation_files"][
            "scripts/server_scheduler/qwen3-v2-g0-handler.py"
        ] = "0" * 64
        other["fingerprint_sha256"] = __import__(
            "posttrain_circuits.artifacts.hashing", fromlist=["sha256_value"]
        ).sha256_value(other["subject"])
        with self.assertRaises(ExecutionSafetyCertificationError):
            load_execution_safety_certification_bytes(
                self._certificate_raw(accepted=True),
                _json_bytes(other),
                require_accepted=True,
            )

    def test_successor_is_proposed_and_does_not_rewrite_accepted_v1(self) -> None:
        successor = load_execution_class_amendment_bytes(
            (PROJECT_ROOT / SUCCESSOR_AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        predecessor = load_protocol_amendment_bytes(
            (PROJECT_ROOT / "prereg/amendments/qwen3_v2_g0_elastic_v1.yaml").read_bytes()
        )
        self.assertEqual(successor["amendment_id"], EXECUTION_CLASS_AMENDMENT_ID)
        self.assertEqual(successor["review"], PROPOSED_REVIEW)
        self.assertEqual(predecessor["review"]["status"], "accepted")
        self.assertEqual(
            successor["evidence_and_reuse"][
                "repeat_four_real_pilots_for_each_new_experiment"
            ],
            False,
        )
        self.assertEqual(
            successor["candidate_e_migration"]["current_real_gpu_evidence"],
            "W1_and_W2_accepted",
        )

    def test_joint_acceptance_rejects_certificate_evidence_edit(self) -> None:
        proposed_amendment = load_execution_class_amendment_bytes(
            (PROJECT_ROOT / SUCCESSOR_AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        accepted_amendment = copy.deepcopy(proposed_amendment)
        accepted_certificate = copy.deepcopy(self.certificate)
        review = {
            "status": "accepted",
            "reviewed_implementation_commit": IMPLEMENTATION_COMMIT,
            "reviewer": "independent-execution-safety-reviewer",
            "reviewed_at_utc": "2026-09-06T23:00:00Z",
            "rationale": "Accepted exact reusable execution class.",
        }
        accepted_amendment["review"] = review
        accepted_certificate["review"] = review
        accepted_certificate["residual_risk"]["world_sizes_without_real_gpu_pilot"] = []
        with self.assertRaisesRegex(ProtocolAmendmentError, "evidence changed"):
            validate_execution_class_review_transition(
                proposed_amendment=proposed_amendment,
                accepted_amendment=accepted_amendment,
                proposed_certification=self.certificate,
                accepted_certification=accepted_certificate,
                implementation_commit=IMPLEMENTATION_COMMIT,
                current_commit=ACCEPTANCE_COMMIT,
                changed_paths=(
                    str(SUCCESSOR_AMENDMENT_RELATIVE_PATH),
                    str(CERTIFICATION_RELATIVE_PATH),
                ),
                implementation_is_ancestor=True,
            )

    def test_v2_config_binds_class_not_per_job_preflight_matrix(self) -> None:
        amendment = load_execution_class_amendment_bytes(
            (PROJECT_ROOT / SUCCESSOR_AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        config = compose_config(
            [
                "g0=qwen3_v2_eap_separation",
                "experiment=canonical_sft",
                "task.num_examples=256",
                "state_source.num_candidates=8",
            ]
        )
        binding = amendment["execution_safety_certification"]
        config["protocol_amendment_path"] = str(SUCCESSOR_AMENDMENT_RELATIVE_PATH)
        config["scheduler_g0"] = {
            "allocation_contract": "manifest_driven_scheduler_gpu_v1",
            "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
            "execution_safety_class_id": binding["execution_class_id"],
            "execution_safety_descriptor_sha256": binding["descriptor_sha256"],
            "execution_safety_certification_sha256": "c" * 64,
            "execution_safety_fingerprint_sha256": binding["fingerprint_sha256"],
            "execution_safety_supported_world_sizes": [1, 2, 3, 4],
            "execution_science_protocol_git_commit": "d" * 40,
            "execution_science_protocol_id": "candidate-e-seed-42",
            "execution_science_protocol_path": (
                "prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml"
            ),
            "execution_science_protocol_reviewed_implementation_commit": "a" * 40,
            "execution_science_protocol_sha256": "e" * 64,
            "artifact_namespace": "qwen3-v2",
            "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "protocol_amendment_id": "qwen3_v2_g0_execution_class_v2",
            "protocol_amendment_sha256": "f" * 64,
            "request_git_commit": "d" * 40,
            "reviewed_implementation_commit": "a" * 40,
            "task": "qwen3_v2_g0",
            "teacher_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_fingerprint": (
                "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
            ),
        }
        validate_elastic_g0_config(config, amendment)
        config["scheduler_g0"]["gpu_preflight_matrix"] = {}
        with self.assertRaisesRegex(ProtocolAmendmentError, "per-experiment"):
            validate_elastic_g0_config(config, amendment)


if __name__ == "__main__":
    unittest.main()
