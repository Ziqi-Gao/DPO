from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from posttrain_circuits.causal_circuits.contracts import CircuitArtifact
from posttrain_circuits.cli.build_rollout_bank import (
    _rollout_generation_batch_contract,
)
from posttrain_circuits.cli.finalize_g0 import (
    _execution_safety_fsdp_wrapper_count,
    _load_execution_science_protocol,
    _qwen3_qk_norm_hooks_pass,
    _runtime_execution_class_checks,
    _stage_compatibility_passes,
)
from posttrain_circuits.artifacts.execution_safety_certification import (
    EXECUTION_CLASS_ID,
    build_execution_safety_descriptor,
)
from posttrain_circuits.artifacts.execution_science_protocol import (
    build_execution_science_protocol,
)
from posttrain_circuits.cli.score_teacher import (
    _compose_score_teacher_config,
    _source_formal_metadata,
)
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    LEGACY_BATCH_PARTITION_PROTOCOL,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.learning.teacher.evaluation import (
    TeacherReadinessThresholds,
    evaluate_teacher_readiness,
)
from posttrain_circuits.learning.contracts import (
    PromptBatch,
    SamplingCursor,
    SamplingRequest,
)
from posttrain_circuits.learning.state_sources.generation import (
    HF_SAMPLING_PROTOCOL_ID,
    hf_generate_trajectories,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _circuit_artifact(**overrides: object) -> CircuitArtifact:
    values: dict[str, object] = {
        "run_id": "run",
        "checkpoint_id": "checkpoint",
        "task_manifest_hash": "task",
        "pair_manifest_hash": "pairs",
        "backend_version": "backend",
        "model_compatibility_hash": "compatibility",
        "node_or_edge_level": "node",
        "integrated_gradient_steps": 5,
        "ablation_baseline": "counterfactual_replacement",
        "scores": {"layer.0.a": 1.0},
        "score_uncertainty": {"layer.0.a": 0.1},
        "semantic_probe_manifest": {"probes": ["semantic"]},
        "tokenized_probe_manifest": {"probes": ["tokenized"]},
        "discovery_pair_manifest": {"pairs": ["pair"]},
        "probe_cohort_manifest": {"cohort": "base_capable"},
        "model_compatibility": {"passed": True},
    }
    values.update(overrides)
    return CircuitArtifact(**values)  # type: ignore[arg-type]


class G0StageContractTests(unittest.TestCase):
    def test_finalizer_binds_g0_fsdp_tree_to_certified_execution_class(self) -> None:
        descriptor = build_execution_safety_descriptor(Path(__file__).resolve().parents[2])
        for world_size in (1, 2, 3, 4):
            with self.subTest(world_size=world_size):
                self.assertEqual(
                    _execution_safety_fsdp_wrapper_count(
                        descriptor,
                        world_size=world_size,
                    ),
                    29,
                )
        with self.assertRaisesRegex(ValueError, "outside 1..4"):
            _execution_safety_fsdp_wrapper_count(descriptor, world_size=5)

    def test_finalizer_compares_runtime_resume_evidence_to_each_world_contract(self) -> None:
        descriptor = build_execution_safety_descriptor(Path(__file__).resolve().parents[2])
        config = {"trainer": {"max_steps": 120}}
        for world_size in (1, 2, 3, 4):
            with self.subTest(world_size=world_size):
                wrappers = _execution_safety_fsdp_wrapper_count(
                    descriptor,
                    world_size=world_size,
                )
                resume = {
                    "format_version": 3,
                    "passed": True,
                    "world_size": world_size,
                    "max_optimizer_steps": 120,
                    "checks": {"checkpoint_state_exact": True},
                    "fsdp_sharding_contract": {
                        "requested_fsdp_sharding_strategy": "FULL_SHARD",
                        "effective_fsdp_sharding_strategy": (
                            "NO_SHARD" if world_size == 1 else "FULL_SHARD"
                        ),
                        "fsdp_wrapper_count": wrappers,
                    },
                }
                self.assertEqual(
                    _runtime_execution_class_checks(
                        resume,
                        config=config,
                        world_size=world_size,
                        certified_fsdp_wrapper_count=wrappers,
                    ),
                    {
                        "distributed_checkpoint_resume": True,
                        "distributed_resume_fsdp_strategy": True,
                    },
                )
                resume["fsdp_sharding_contract"]["fsdp_wrapper_count"] = wrappers - 1
                self.assertFalse(
                    _runtime_execution_class_checks(
                        resume,
                        config=config,
                        world_size=world_size,
                        certified_fsdp_wrapper_count=wrappers,
                    )["distributed_resume_fsdp_strategy"]
                )

    def test_finalizer_binds_accepted_science_protocol_to_config_and_class(self) -> None:
        fingerprint = "a" * 64
        descriptor_sha256 = "b" * 64
        certification_sha256 = "c" * 64
        science_implementation_commit = "d" * 40
        science_acceptance_commit = "e" * 40
        config: dict[str, object] = {
            "seed": 42,
            "trainer": {"max_steps": 120},
        }
        protocol = build_execution_science_protocol(
            protocol_id="qwen3-v2-g0-seed-42-v1",
            unit_id="g0",
            seed=42,
            execution_class_id=EXECUTION_CLASS_ID,
            descriptor_path="prereg/execution_safety/descriptor.json",
            descriptor_sha256=descriptor_sha256,
            fingerprint_sha256=fingerprint,
            certification_id="qwen3-v2-elastic-training-v1-certification",
            certification_path="prereg/execution_safety/certification.yaml",
            certification_core_sha256="f" * 64,
            resolved_config=config,
            hydra_override_vector=("seed=42",),
            review={
                "status": "accepted",
                "reviewed_implementation_commit": science_implementation_commit,
                "reviewer": "independent-review",
                "reviewed_at_utc": "2026-09-06T00:00:00Z",
                "rationale": "accepted fixture",
            },
        )
        raw = yaml.safe_dump(protocol, sort_keys=False).encode("utf-8")
        import hashlib

        config["scheduler_g0"] = {
            "execution_safety_class_id": EXECUTION_CLASS_ID,
            "execution_safety_fingerprint_sha256": fingerprint,
            "execution_safety_descriptor_sha256": descriptor_sha256,
            "execution_safety_certification_sha256": certification_sha256,
            "execution_science_protocol_id": protocol["protocol_id"],
            "execution_science_protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "execution_science_protocol_git_commit": science_acceptance_commit,
            "execution_science_protocol_reviewed_implementation_commit": (
                science_implementation_commit
            ),
        }
        context = SimpleNamespace(
            execution_safety_class_id=EXECUTION_CLASS_ID,
            execution_safety_fingerprint_sha256=fingerprint,
            execution_safety_descriptor_sha256=descriptor_sha256,
            execution_safety_certification_sha256=certification_sha256,
            execution_science_protocol_id=protocol["protocol_id"],
            execution_science_protocol_sha256=hashlib.sha256(raw).hexdigest(),
            execution_science_protocol_git_commit=science_acceptance_commit,
            execution_science_protocol_reviewed_implementation_commit=(
                science_implementation_commit
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix=".g0-science-finalizer-", dir=Path.cwd()
        ) as temporary:
            path = Path(temporary) / "execution_science_protocol.yaml"
            path.write_bytes(raw)
            binding, acceptance_commit = _load_execution_science_protocol(
                path,
                config=config,  # type: ignore[arg-type]
                scientific_context=context,  # type: ignore[arg-type]
            )
            self.assertEqual(binding.protocol_id, protocol["protocol_id"])
            self.assertEqual(acceptance_commit, science_acceptance_commit)
            config["seed"] = 43
            with self.assertRaisesRegex(ValueError, "execution science protocol"):
                _load_execution_science_protocol(
                    path,
                    config=config,  # type: ignore[arg-type]
                    scientific_context=context,  # type: ignore[arg-type]
                )

    @staticmethod
    def _compatibility(*, logit_error: float) -> dict[str, object]:
        payload: dict[str, object] = {
            "passed": True,
            "hf_identity_passed": True,
            "transformerlens_parity_passed": True,
            "transformerlens_logit_error": logit_error,
            "hook_positions": {
                "query_projection": [
                    "model.layers.0.self_attn.q_proj:output:pre_q_norm_pre_rope"
                ],
                "key_projection": [
                    "model.layers.0.self_attn.k_proj:output:pre_k_norm_pre_rope"
                ],
            },
        }
        from posttrain_circuits.artifacts.hashing import sha256_value

        payload["sha256"] = sha256_value(payload)
        return payload

    def test_stage_compatibilities_are_independent_and_fail_closed(self) -> None:
        final = self._compatibility(logit_error=0.001)
        process = self._compatibility(logit_error=0.002)
        self.assertNotEqual(final["sha256"], process["sha256"])
        final_circuit = {"model_compatibility_hash": final["sha256"]}
        process_circuit = {"model_compatibility_hash": process["sha256"]}

        self.assertTrue(_stage_compatibility_passes(final, final_circuit))
        self.assertTrue(_stage_compatibility_passes(process, process_circuit))
        self.assertFalse(_stage_compatibility_passes(final, process_circuit))
        self.assertTrue(_qwen3_qk_norm_hooks_pass(final))

        missing_key_hooks = dict(process)
        missing_key_hooks["hook_positions"] = {
            "query_projection": [
                "model.layers.0.self_attn.q_proj:output:pre_q_norm_pre_rope"
            ]
        }
        self.assertFalse(_qwen3_qk_norm_hooks_pass(missing_key_hooks))

    def test_rollout_batch_contract_uses_elastic_science_fields(self) -> None:
        size, binding = _rollout_generation_batch_contract(
            {
                "trainer": {
                    "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
                    "global_batch_size": 64,
                    "max_microbatch_size": 4,
                },
                "state_source": {"temperature": 0.7},
            },
            example_count=256,
            generations_per_prompt=4,
        )
        self.assertEqual(size, 4)
        self.assertEqual(binding["global_logical_batch_size"], 64)
        self.assertEqual(binding["max_generation_prompt_chunk_size"], 4)
        self.assertEqual(binding["expanded_generation_sequences_per_full_chunk"], 16)
        self.assertEqual(binding["max_simultaneous_generation_sequences"], 1)
        self.assertNotIn("max_generation_microbatch_size", binding)
        self.assertNotIn("batch_size", binding)
        with self.assertRaisesRegex(ValueError, "exact multiple"):
            _rollout_generation_batch_contract(
                {
                    "trainer": {
                        "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
                        "global_batch_size": 64,
                        "max_microbatch_size": 4,
                    },
                    "state_source": {"temperature": 0.7},
                },
                example_count=255,
                generations_per_prompt=4,
            )
        with self.assertRaisesRegex(ValueError, "sampled serial"):
            _rollout_generation_batch_contract(
                {
                    "trainer": {
                        "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
                        "global_batch_size": 64,
                        "max_microbatch_size": 4,
                    },
                    "state_source": {"temperature": 0.0},
                },
                example_count=256,
                generations_per_prompt=4,
            )

        legacy_size, legacy = _rollout_generation_batch_contract(
            {
                "trainer": {
                    "batch_partition_protocol": LEGACY_BATCH_PARTITION_PROTOCOL,
                    "batch_size": 3,
                }
            },
            example_count=9,
            generations_per_prompt=4,
        )
        self.assertEqual(legacy_size, 3)
        self.assertEqual(legacy["generation_batch_size"], 3)

    def test_four_by_four_sampled_rollout_expansion_generates_serially(self) -> None:
        tokenizer = build_tiny_tokenizer()
        model = build_tiny_qwen(43).eval()
        prompt_ids = tuple(
            f"prompt-{prompt_index}"
            for prompt_index in range(4)
            for _ in range(4)
        )
        prompts = PromptBatch(prompt_ids, tuple("A" for _ in prompt_ids))
        sampling_request = SamplingRequest(
            sampling_request_seed=42,
            sampling_protocol_id=HF_SAMPLING_PROTOCOL_ID,
            cursors=tuple(
                SamplingCursor(
                    optimizer_step=0,
                    retry_index=0,
                    prompt_id=f"prompt-{prompt_index}",
                    generation_index=generation_index,
                )
                for prompt_index in range(4)
                for generation_index in range(4)
            ),
        )
        observed_batch_sizes: list[int] = []
        original_generate = model.generate

        def tracked_generate(*args: object, **kwargs: object) -> object:
            input_ids = kwargs["input_ids"]
            observed_batch_sizes.append(int(input_ids.shape[0]))  # type: ignore[attr-defined]
            return original_generate(*args, **kwargs)

        with mock.patch.object(model, "generate", side_effect=tracked_generate):
            records = hf_generate_trajectories(
                model,
                tokenizer,
                prompts,
                policy_version=0,
                sampling_request=sampling_request,
                max_new_tokens=1,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                policy_id="local-fixture",
                policy_revision="local-fixture-v1",
            )

        self.assertEqual(len(records), 16)
        self.assertEqual(observed_batch_sizes, [1] * 16)

    def test_fixed_bank_teacher_scoring_drops_only_candidate_override(self) -> None:
        fixed = _compose_score_teacher_config(
            ["experiment=offline_soft", "state_source.num_candidates=8"]
        )
        self.assertEqual(fixed["state_source"]["name"], "fixed_bank")
        self.assertNotIn("num_candidates", fixed["state_source"])

        teacher_demo = _compose_score_teacher_config(
            ["experiment=canonical_sft", "state_source.num_candidates=3"]
        )
        self.assertEqual(teacher_demo["state_source"]["num_candidates"], 3)

    def test_score_teacher_propagates_complete_formal_binding_and_rejects_tamper(self) -> None:
        binding = {
            "protocol_track": "qwen3_v2",
            "artifact_namespace": "qwen3-v2",
            "model_revision": "1" * 40,
            "teacher_revision": "2" * 40,
            "tokenizer_revision": "1" * 40,
            "tokenizer_fingerprint": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "code_commit": "5" * 40,
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_version": "qwen3_v2",
            "prereg_commit": "6" * 40,
            "prereg_sha256": "7" * 64,
            "protocol_amendment_id": "qwen3-v2-g0-elastic-v1",
            "protocol_amendment_path": "prereg/amendments/qwen3_v2_g0_elastic_v1.yaml",
            "protocol_amendment_git_commit": "8" * 40,
            "protocol_amendment_sha256": "9" * 64,
            "reviewed_implementation_commit": "a" * 40,
        }
        self.assertEqual(
            _source_formal_metadata(dict(binding), expected=binding),
            binding,
        )
        tampered = dict(binding)
        tampered["protocol_amendment_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "protocol_amendment_sha256"):
            _source_formal_metadata(tampered, expected=binding)

    def test_circuit_artifact_supports_optional_complete_amendment_binding(self) -> None:
        amendment = {
            "protocol_amendment_id": "qwen3-v2-g0-elastic-v1",
            "protocol_amendment_path": "prereg/amendments/qwen3_v2_g0_elastic_v1.yaml",
            "protocol_amendment_git_commit": "1" * 40,
            "protocol_amendment_sha256": "2" * 64,
            "reviewed_implementation_commit": "3" * 40,
        }
        artifact = _circuit_artifact(**amendment)
        self.assertEqual(artifact.protocol_amendment_id, amendment["protocol_amendment_id"])
        _circuit_artifact()  # Legacy artifacts omit all five fields.
        with self.assertRaisesRegex(ValueError, "complete or omitted"):
            _circuit_artifact(protocol_amendment_id="qwen3-v2-g0-elastic-v1")

    def test_qwen_v2_teacher_readiness_requires_complete_optional_amendment_binding(self) -> None:
        task = ProofGraphTask()
        examples = list(task.generate_pair(73, {"depth": 2}))
        responses = {
            example.example_id: task.canonical_target(example) for example in examples
        }
        bindings = {
            "teacher_model_revision": "teacher",
            "tokenizer_revision": "teacher-tokenizer",
            "dataset_hash": "dataset",
            "prefix_probe_hash": "probes",
            "code_commit": "1" * 40,
            "prereg_commit": "2" * 40,
            "teacher_model_id": "Qwen/Qwen3-8B",
            "student_model_revision": "student",
            "student_model_id": "Qwen/Qwen3-1.7B",
            "student_tokenizer_revision": "student-tokenizer",
            "teacher_tokenizer_revision": "teacher-tokenizer",
            "tokenizer_fingerprint": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "protocol_track": "qwen3_v2",
            "artifact_namespace": "qwen3-v2",
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_version": "qwen3_v2",
            "prereg_sha256": "5" * 64,
            "protocol_amendment_id": "qwen3-v2-g0-elastic-v1",
            "protocol_amendment_path": "prereg/amendments/qwen3_v2_g0_elastic_v1.yaml",
            "protocol_amendment_git_commit": "6" * 40,
            "protocol_amendment_sha256": "7" * 64,
            "reviewed_implementation_commit": "8" * 40,
        }
        artifact = evaluate_teacher_readiness(
            examples,
            responses,
            [],
            TeacherReadinessThresholds(),
            bindings=bindings,
        )
        self.assertEqual(
            artifact["bindings"]["protocol_amendment_sha256"],
            bindings["protocol_amendment_sha256"],
        )
        incomplete = dict(bindings)
        incomplete.pop("reviewed_implementation_commit")
        with self.assertRaisesRegex(ValueError, "bindings are incomplete"):
            evaluate_teacher_readiness(
                examples,
                responses,
                [],
                TeacherReadinessThresholds(),
                bindings=incomplete,
            )
        legacy = {
            key: value
            for key, value in bindings.items()
            if not key.startswith("protocol_amendment_")
            and key != "reviewed_implementation_commit"
        }
        evaluate_teacher_readiness(
            examples,
            responses,
            [],
            TeacherReadinessThresholds(),
            bindings=legacy,
        )


if __name__ == "__main__":
    unittest.main()
