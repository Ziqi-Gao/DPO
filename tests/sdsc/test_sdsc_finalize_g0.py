"""CPU decision/contract fixtures; these do not claim real G0 or GPU success."""

import ast
import copy
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sdsc_finalize_g0_test", ROOT / "tools/sdsc_finalize_g0.py")
final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final)


def sha(value):
    return hashlib.sha256(final.canonical(value)).hexdigest()


def original_pure_helpers():
    """Use the genuine original pure helper bodies without importing Torch."""
    tree = ast.parse((ROOT / "src/posttrain_circuits/cli/finalize_g0.py").read_text())
    functions = {
        "_require_formal_binding",
        "_teacher_readiness_formal_binding",
        "_stage_compatibility_passes",
        "_qwen3_qk_norm_hooks_pass",
    }
    constants = {
        "G0_CHECK_NAMES",
        "_AMENDMENT_ID",
        "_BATCH_PARTITION_PROTOCOL",
        "_GLOBAL_BATCH_SIZE",
        "_MAX_MICROBATCH_SIZE",
        "_MICROBATCHES_BY_WORLD_SIZE",
        "_SAMPLES_BY_WORLD_SIZE",
        "_SHA256",
    }
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    namespace = {"re": re, "sha256_value": sha}
    parsed = ast.parse("from __future__ import annotations\n")
    parsed.body.extend(selected)
    exec(compile(parsed, "original_pure_finalizer_helpers", "exec"), namespace)
    return SimpleNamespace(**{key: value for key, value in namespace.items() if key != "__builtins__"})


class ScientificDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.api = original_pure_helpers()
        self.config = {
            "prereg_version": "core_v2",
            "protocol_track": "qwen3_v2",
            "model": {
                "model_name_or_path": "student",
                "model_revision": "s" * 40,
                "tokenizer_revision": "t" * 40,
                "artifact_namespace": "qwen3-v2",
                "tokenizer_fingerprint": "f" * 64,
                "prompt_protocol": {"name": "qwen3_non_thinking_v1", "chat_template_sha256": "c" * 64},
            },
            "teacher": {
                "model_name_or_path": "teacher",
                "model_revision": "a" * 40,
                "tokenizer_revision": "t" * 40,
            },
            "anti_shortcut": {"minimum_iid_accuracy": 0.55, "max_shortcut_gap": 0.1},
            "g0": {
                "minimum_bootstrap_spearman": 0.8,
                "minimum_selected_vs_random_cpr_margin": 0.0,
                "minimum_attribution_exact_spearman": 0.5,
                "minimum_spearman_bootstrap_lower_bound": 0.2,
            },
            "task": {"num_examples": 256},
            "trainer": {"max_steps": 120},
        }
        keys = [
            "protocol_track",
            "artifact_namespace",
            "model_revision",
            "teacher_revision",
            "tokenizer_revision",
            "tokenizer_fingerprint",
            "chat_template_sha256",
            "prompt_protocol",
            "enable_thinking",
            "code_commit",
            "prereg_path",
            "prereg_version",
            "prereg_commit",
            "prereg_sha256",
            "protocol_amendment_id",
            "protocol_amendment_path",
            "protocol_amendment_git_commit",
            "protocol_amendment_sha256",
            "reviewed_implementation_commit",
        ]
        self.formal = dict.fromkeys(keys, "fixture")
        self.formal.update(
            protocol_track="qwen3_v2",
            artifact_namespace="qwen3-v2",
            prompt_protocol="qwen3_non_thinking_v1",
            enable_thinking=False,
            code_commit=final.SCIENCE_HEAD,
            chat_template_sha256="c" * 64,
            tokenizer_fingerprint="f" * 64,
            protocol_amendment_id=final.binding.AMENDMENT.split("/")[-1][:-5],
            model_revision="s" * 40,
            teacher_revision="a" * 40,
            tokenizer_revision="t" * 40,
        )
        self.initial = self.root / "initial_checkpoint.pt"
        self.initial.write_bytes(b"CPU fixture only; not a model")
        self.initial_hash = hashlib.sha256(self.initial.read_bytes()).hexdigest()
        self.documents = {}
        self.write(
            "probe_scores.json",
            initial_validation_metrics={"answer_accuracy": 0.6},
            calibrated_validation_metrics={"answer_accuracy": 0.7},
            initial_checkpoint_sha256=self.initial_hash,
            sha256="score",
            eligibility_evidence_ancestry="ancestry",
        )
        self.write(
            "teacher_scores/manifest.json",
            teacher_topk_mass={"minimum": 0.95},
            total_trajectories=10,
            reward_distribution={"positive": 5},
        )
        teacher_binding = {
            **self.formal,
            "student_model_revision": "s" * 40,
            "teacher_model_revision": "a" * 40,
            "student_tokenizer_revision": "t" * 40,
        }
        self.write(
            "teacher_readiness.json",
            passed=True,
            bindings=teacher_binding,
            tokenized_prefix_manifest={"sha256": sha({})},
            metrics={},
        )
        self.write("label_leakage.json", passed=True, dataset_hash="dataset", metrics={})
        self.write(
            "anti_shortcut.json",
            passed=True,
            shortcut_gap=0.01,
            iid_accuracy=0.6,
            transformed_accuracy=0.6,
            dataset_hash="dataset",
            suite_hash="suite",
        )
        self.write(
            "probes/manifest.json",
            cohorts={"base_capable": {"discovery": {"num_examples": 1}, "validation": {"num_examples": 1}}},
            scoring_manifest_hash="score",
            eligibility_evidence_ancestry="ancestry",
            frozen_before_training=True,
        )
        for stage in ("final_answer", "first_rule_selection"):
            compatibility = self.write(
                f"circuits/{stage}/mib_raw/compatibility.json",
                passed=True,
                hf_identity_passed=True,
                transformerlens_parity_passed=True,
                hook_positions={
                    "query_projection": ["pre_q_norm_pre_rope"],
                    "key_projection": ["pre_k_norm_pre_rope"],
                },
            )
            compatibility["sha256"] = sha(compatibility)
            self.flush(f"circuits/{stage}/mib_raw/compatibility.json")
            self.write(
                f"circuits/{stage}/circuit.json",
                bootstrap_score_vectors=[0.9],
                probe_stage=stage,
                stage_target_manifest_hash=stage,
                model_compatibility_hash=compatibility["sha256"],
                checkpoint_sha256=self.initial_hash,
            )
            self.write(
                f"circuits/{stage}/exact_patching.json",
                selected_vs_matched_random_cpr_margin=0.1,
                sanity_checks={"identity_passed": True, "full_corruption_passed": True},
                attribution_patching_spearman=0.7,
                attribution_patching_spearman_ci={"lower": 0.4},
                artifacts={"checkpoint_sha256": self.initial_hash},
            )
        self.resume = self.write(
            "distributed_resume.json",
            format_version=3,
            passed=True,
            world_size=2,
            max_optimizer_steps=120,
            checks={"all_three_runs_verified": True},
            fsdp_sharding_contract={
                "requested_fsdp_sharding_strategy": "FULL_SHARD",
                "effective_fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_wrapper_count": 29,
            },
        )
        self.api._read = lambda path: copy.deepcopy(self.documents[str(path.relative_to(self.root))])
        self.api.validate_probe_score_artifact = self.api._read
        self.api.TrajectoryStore = lambda path: SimpleNamespace(
            check_integrity=lambda: self.api._read(path / "manifest.json")
        )
        self.api.require_scientific_artifact = lambda *_a, **_kw: None
        self.api.validate_label_leakage_artifact = lambda value: value
        self.api.validate_teacher_readiness_artifact = lambda value, **_kw: value
        self.api.validate_anti_shortcut_report = lambda path, **_kw: self.api._read(path)
        self.api.validate_probe_cohort_manifest = lambda path, **_kw: self.api._read(path)
        self.api.sha256_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.api.sha256_value = sha
        self.api.estimate_estimator_noise_floor = lambda value, **_kw: {
            "within_checkpoint_full_score_spearman": value[0]
        }
        self.api.REQUESTED_FSDP_SHARDING_STRATEGY = "FULL_SHARD"
        self.api.effective_fsdp_sharding_strategy = lambda _world: "FULL_SHARD"
        self.api._batch_token_contract = lambda _config, _world: {
            "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
            "requested_fsdp_sharding_strategy": "FULL_SHARD",
            "effective_fsdp_sharding_strategy": "FULL_SHARD",
            "global_logical_batch_size": 64,
            "max_per_rank_microbatch_size": 4,
            "max_model_input_length": 1536,
            "full_parameter_training": True,
            "prompt_population_size": 256,
            "prompt_ids_unique": True,
            "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
            "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
            "microbatch_schedule_by_rank": [[4] * 8, [4] * 8],
            "optimizer_microsteps": 8,
            "samples_by_rank": [32, 32],
            "world_size": 2,
            "token_budget": 2000000,
            "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
            "max_optimizer_steps": 120,
        }
        self.api._runtime_execution_class_checks = lambda value, **_kw: {
            "distributed_checkpoint_resume": value["passed"],
            "distributed_resume_fsdp_strategy": True,
        }

    def write(self, name, **values):
        self.documents[name] = {**self.formal, **values}
        self.flush(name)
        return self.documents[name]

    def flush(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.documents[name]))

    def decision(self):
        resume_api = SimpleNamespace(validate_resume_report=lambda *_a, **_kw: self.resume)
        with patch.object(final, "helper", return_value=resume_api):
            return final.scientific_decision(self.root, self.config, self.formal, self.api)

    def test_exact_26_scientific_checks_can_pass_without_historical_context(self):
        checks, metrics, _paths, _artifacts = self.decision()
        self.assertEqual(len(checks), 26)
        self.assertTrue(all(checks.values()))
        self.assertEqual(set(checks), set(self.api.G0_CHECK_NAMES) - final.OLD_BACKEND_CHECKS)
        self.assertEqual(metrics["teacher_topk_mass_minimum"], 0.95)

    def test_each_stage_margin_must_strictly_exceed_threshold(self):
        for stage, check in (
            ("final_answer", "final_stage_eap_beats_matched_random"),
            ("first_rule_selection", "process_stage_eap_beats_matched_random"),
        ):
            name = f"circuits/{stage}/exact_patching.json"
            self.documents[name]["selected_vs_matched_random_cpr_margin"] = 0.0
            self.flush(name)
            self.assertFalse(self.decision()[0][check])

    def test_teacher_mass_fixed_point_nine_and_calibration_improvement(self):
        self.documents["teacher_scores/manifest.json"]["teacher_topk_mass"]["minimum"] = 0.8999
        self.documents["probe_scores.json"]["calibrated_validation_metrics"]["answer_accuracy"] = 0.6
        checks = self.decision()[0]
        self.assertFalse(checks["teacher_topk_mass"])
        self.assertFalse(checks["calibration_anchor_improves_accuracy"])

    def test_attribution_confidence_and_bootstrap_thresholds_not_relaxed(self):
        self.documents["circuits/final_answer/exact_patching.json"]["attribution_patching_spearman_ci"][
            "lower"
        ] = 0.199
        self.documents["circuits/first_rule_selection/circuit.json"]["bootstrap_score_vectors"] = [0.799]
        checks = self.decision()[0]
        self.assertFalse(checks["attribution_exact_calibration"])
        self.assertFalse(checks["bootstrap_stability"])

    def test_formal_boolean_alias_and_cross_head_are_rejected(self):
        for key, value in (("enable_thinking", 0), ("code_commit", "a" * 40)):
            with self.subTest(key=key):
                original = self.documents["probe_scores.json"][key]
                self.documents["probe_scores.json"][key] = value
                with self.assertRaises(ValueError):
                    self.decision()
                self.documents["probe_scores.json"][key] = original

    def test_missing_or_failed_resume_does_not_get_g0(self):
        self.resume["passed"] = False
        self.assertFalse(self.decision()[0]["distributed_checkpoint_resume"])
        with (
            patch.object(
                final,
                "helper",
                return_value=SimpleNamespace(
                    validate_resume_report=lambda *_a, **_kw: (_ for _ in ()).throw(
                        ValueError("missing second resume")
                    )
                ),
            ),
            self.assertRaisesRegex(ValueError, "missing second resume"),
        ):
            final.scientific_decision(self.root, self.config, self.formal, self.api)

    def test_strict_json_scan_rejects_nonfinite_and_duplicate_fields(self):
        path = self.root / "label_leakage.json"
        for raw in ('{"passed":true,"passed":true}', '{"x":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                self.decision()

    def test_readiness_retains_fifteen_original_required_checks(self):
        checks, metrics, _paths, _artifacts = self.decision()
        evidence = final.readiness_evidence(checks, metrics, {})
        self.assertEqual(len(evidence), 15)
        self.assertTrue(all(row[0] is True for row in evidence.values()))
        checks["distributed_checkpoint_resume"] = False
        self.assertFalse(final.readiness_evidence(checks, metrics, {})["checkpoint_resume_verified"][0])


class CompletionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_replay_uses_exact_reconstructed_decision_even_after_resigning(self):
        reconstructed = {
            "schema": final.SCHEMA,
            "passed": True,
            "checks": {"teacher_correctness": True},
            "execution_class_certified": False,
        }
        report = {**copy.deepcopy(reconstructed), "created_at": "2026-09-18T00:00:00Z"}
        report["sha256"] = sha(report)
        self.assertIs(final.validate_replay(report, reconstructed), report)
        for mutation in (
            {"execution_class_certified": True},
            {"checks": {"teacher_correctness": 1}},
            {"extra": 1},
        ):
            candidate = {**report, **mutation}
            candidate["sha256"] = sha({key: value for key, value in candidate.items() if key != "sha256"})
            with self.assertRaises(ValueError):
                final.validate_replay(candidate, reconstructed)

    def test_failure_cannot_be_replayed_as_success(self):
        report = {"passed": False, "created_at": "now"}
        report["sha256"] = sha(report)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            final.validate_replay(report, {"passed": False})

    def test_workspace_refuses_symlink_and_records_byte_change(self):
        path = self.root / "evidence.json"
        path.write_text("one")
        before = final.require_safe_tree(self.root)
        path.write_text("two")
        self.assertNotEqual(before, final.require_safe_tree(self.root))
        (self.root / "link").symlink_to(path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            final.require_safe_tree(self.root)

    def test_relative_locator_rejects_escape_and_absolute(self):
        for value in ("../secret", "/tmp/secret", "a/../secret", "a//b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                final.contained(self.root, value)

    def test_review_requires_external_hash_and_exact_current_adapter_bytes(self):
        fake = {
            "reviewer": "independent-review",
            "scope": "single_invocation_h100_g0",
            "files": dict.fromkeys(final.REQUIRED_ADAPTERS, "a" * 64),
        }
        with self.assertRaisesRegex(ValueError, "externally selected"):
            final.validate_adapter_review(fake, "b" * 64)
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            final.validate_adapter_review(fake, "a" * 64)
        fake["files"]["../outside.py"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "unsupported"):
            final.validate_adapter_review(fake, "a" * 64)

    def test_g0_adapter_has_no_historical_context_or_certificate_calls(self):
        tree = ast.parse((ROOT / "tools/sdsc_finalize_g0.py").read_text())
        forbidden = {
            "ScientificInvocationContext",
            "_load_execution_safety",
            "_load_execution_science_protocol",
            "replay_g0_decision",
            "load_execution_safety_certification_bytes",
        }
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
        }
        self.assertFalse(called & forbidden)

    def test_preflight_requires_empty_queue_and_every_step_completed(self):
        report = {"job_id": "123", "run_id": "run"}
        status = {
            "success": True,
            "state": "COMPLETED",
            **report,
            "result": {"verified": True, "result_sha256": "a" * 64},
            "queue": {"returncode": 0, "stdout": ""},
            "accounting": {"returncode": 0, "stdout": "123|COMPLETED|0:0|\n123.batch|COMPLETED|0:0|\n"},
        }
        final.validate_preflight_status(status, report, "a" * 64)
        for section, key, value in (
            ("queue", "stdout", "123|RUNNING"),
            ("accounting", "stdout", "123|COMPLETED|0:0|\n123.batch|FAILED|1:0|"),
            ("accounting", "returncode", 1),
        ):
            candidate = copy.deepcopy(status)
            candidate[section][key] = value
            with self.assertRaises(ValueError):
                final.validate_preflight_status(candidate, report, "a" * 64)


if __name__ == "__main__":
    unittest.main()
