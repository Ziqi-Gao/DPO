from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.cli import factorial_run_validation as run_validation
from posttrain_circuits.cli import compare_distributed_resume as resume_comparison
from posttrain_circuits.cli import finalize_g0


class G0FactorialRunValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".g0-run-validation-",
            dir=Path.cwd(),
        )
        self.workspace = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def _artifacts(
        self,
        name: str,
        *,
        ancestry: list[str],
        source_sha256: str,
    ) -> run_validation.FactorialRunArtifacts:
        root = self.workspace / name
        checkpoint_path = root / "checkpoints" / "step-00000120.pt"
        accelerator_path = checkpoint_path.with_suffix(".accelerate")
        accelerator_path.mkdir(parents=True)
        (accelerator_path / "state.bin").write_bytes(b"same-state")
        for path, raw in (
            (root / "manifest.json", b"manifest\n"),
            (root / "resolved_config.yaml", b"config\n"),
            (root / "config_binding.json", b"binding\n"),
            (checkpoint_path, b"checkpoint\n"),
            (root / "factorial_update_evidence.json", b"evidence\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self._write_json(
            root / "metrics.jsonl",
            {"step": 120.0, "canonical_sft_loss": 0.25},
        )
        token_budget = {
            "budget": 2_000_000,
            "unit": "global_nonpadding_model_input_tokens_processed",
            "consumed": 1000,
            "accepted_optimizer_updates": 120,
            "stop_reason": "max_steps_safety_limit",
        }
        runtime_common = {
            "model": {"weight": "same"},
            "optimizer": {"state": "same"},
            "scheduler": {"state": "same"},
            "rng": {"state": "same"},
            "trainer_state": {
                "batch_partition": {
                    "protocol": "exact",
                    "requested_fsdp_sharding_strategy": "FULL_SHARD",
                    "effective_fsdp_sharding_strategy": "NO_SHARD",
                    "fsdp_wrapper_count": 29,
                }
            },
            "token_budget": token_budget,
            "prompt_scheduler_by_rank": [{"rank": 0}],
            "state_source_by_rank": [{"rank": 0}],
            "manifest_hashes": {"dataset": "same"},
            "trainer_state_by_rank": [{"rank": 0}],
            "scaler": None,
            "rank_shard_hashes": ["a" * 64],
            "accelerate_state_files": {"state.bin": "b" * 64},
            "accelerate_state_sha256": "c" * 64,
        }
        checkpoint = {
            **runtime_common,
            "format": "accelerate_fsdp_full_export_v1",
            "world_size": 1,
            "global_step": 120,
            "policy_version": 0,
            "online_rollout_round": 0,
            "optimizer_state_key_type": "parameter_id",
            "resume_ancestry": ancestry,
            "update_norm_baseline_checkpoint_sha256": source_sha256,
            "final_model_state_hash": "d" * 64,
        }
        manifest = {
            "sha256": "e" * 64,
            "resolved_config_sha256": "7" * 64,
            "training_stop_reason": "max_steps_safety_limit",
        }
        evidence = {"sha256": "f" * 64}
        return run_validation.FactorialRunArtifacts(
            root=root,
            manifest_path=root / "manifest.json",
            manifest=manifest,
            resolved_config_path=root / "resolved_config.yaml",
            config_binding_path=root / "config_binding.json",
            metrics_path=root / "metrics.jsonl",
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint,
            evidence_path=root / "factorial_update_evidence.json",
            evidence=evidence,
            experiment_binding=SimpleNamespace(scientific_sha256="1" * 64),
            accelerator_state_path=accelerator_path,
        )

    def _source_binding(self, source: Path, source_sha256: str) -> dict[str, object]:
        content: dict[str, object] = {
            "accelerator_state": {
                "files": {"state.bin": "b" * 64},
                "logical_path": "calibration/checkpoints/step-00000020.accelerate",
                "sha256": "c" * 64,
            },
            "batch_partition": {"protocol": "exact"},
            "checkpoint": {
                "format": "accelerate_fsdp_full_export_v1",
                "logical_path": "calibration/checkpoints/step-00000020.pt",
                "sha256": source_sha256,
            },
            "config_binding_sha256": "1" * 64,
            "experiment_binding_sha256": "2" * 64,
            "git_commit": "3" * 40,
            "global_step": 20,
            "implementation_dirty": False,
            "manifest_hashes": {"dataset": "same"},
            "rank_shard_hashes": ["4" * 64],
            "resolved_config_sha256": "5" * 64,
            "resume_ancestry": [],
            "runtime_state_hashes": {"model": "6" * 64},
            "token_budget": {"accepted_optimizer_updates": 20},
            "world_size": 1,
        }
        return {**content, "sha256": sha256_value(content)}

    def test_strict_json_rejects_duplicate_nonfinite_and_symlink_input(self) -> None:
        path = self.workspace / "artifact.json"
        path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            run_validation.strict_json_object(path, name="artifact")
        path.write_text('{"a": NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run_validation.strict_json_object(path, name="artifact")
        target = self.workspace / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.unlink()
        path.symlink_to(target.name)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            run_validation.strict_json_object(path, name="artifact")

    def test_probe_scores_reject_duplicate_nonfinite_and_hash_tampering(self) -> None:
        path = self.workspace / "probe_scores.json"
        content = {
            "initial_validation_metrics": {"answer_accuracy": 0.5},
            "calibrated_validation_metrics": {"answer_accuracy": 0.6},
            "scores": {
                "example-1": {
                    "initial_correct": True,
                    "learnable_after_post_training": True,
                }
            },
        }
        self._write_json(path, {**content, "sha256": sha256_value(content)})
        self.assertEqual(finalize_g0.validate_probe_score_artifact(path)["scores"], content["scores"])

        path.write_text('{"scores": {}, "scores": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            finalize_g0.validate_probe_score_artifact(path)
        for token in ("NaN", "1e999"):
            with self.subTest(token=token):
                path.write_text(
                    '{"initial_validation_metrics":{"answer_accuracy":'
                    + token
                    + '},"calibrated_validation_metrics":{"answer_accuracy":0.6},'
                    '"scores":{"x":{"initial_correct":true,'
                    '"learnable_after_post_training":true}},"sha256":"'
                    + "0" * 64
                    + '"}',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    finalize_g0.validate_probe_score_artifact(path)
        self._write_json(path, {**content, "sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            finalize_g0.validate_probe_score_artifact(path)

    def test_terminal_state_rejects_signal_and_inconsistent_limit_reason(self) -> None:
        budget = {
            "budget": 100,
            "consumed": 80,
            "stop_reason": "max_steps_safety_limit",
        }
        self.assertEqual(
            run_validation._terminal_state(
                global_step=120,
                max_steps=120,
                token_budget=budget,
            ),
            "max_steps_safety_limit",
        )
        signal = {**budget, "stop_reason": "signal_at_optimizer_boundary"}
        with self.assertRaisesRegex(ValueError, "accepted scientific stop reason"):
            run_validation._terminal_state(
                global_step=120,
                max_steps=120,
                token_budget=signal,
            )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            run_validation._terminal_state(
                global_step=119,
                max_steps=120,
                token_budget=budget,
            )

    def test_accelerator_state_replays_at_new_root_and_rejects_tampering(self) -> None:
        source_workspace = Path(
            "/scr/del6500/OPD/tmp/qwen3-v2-g0-source0001/qwen3-v2"
        )
        replay_workspace = self.workspace / "fresh-replay" / "qwen3-v2"
        run_root = replay_workspace / "calibration"
        checkpoint = run_root / "checkpoints" / "step-00000020.pt"
        state_root = checkpoint.with_suffix(".accelerate")
        state_root.mkdir(parents=True)
        state_file = state_root / "state.bin"
        state_file.write_bytes(b"locally-validated-state")
        state_files = {
            "state.bin": __import__("hashlib").sha256(state_file.read_bytes()).hexdigest()
        }
        payload = {
            "format": "accelerate_fsdp_full_export_v1",
            "accelerate_state_dir": str(
                source_workspace
                / "calibration/checkpoints/step-00000020.accelerate"
            ),
            "accelerate_state_files": state_files,
            "accelerate_state_sha256": sha256_value(state_files),
        }
        observed = run_validation._validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint,
            checkpoint_payload=payload,
            resolved_config={"trainer": {"backend": "accelerate"}},
            path_relocation=(source_workspace, replay_workspace),
        )
        self.assertEqual(observed, state_files)

        state_file.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "content differs"):
            run_validation._validate_factorial_accelerator_state(
                run_root,
                checkpoint_path=checkpoint,
                checkpoint_payload=payload,
                resolved_config={"trainer": {"backend": "accelerate"}},
                path_relocation=(source_workspace, replay_workspace),
            )

    def test_final_cadence_accepts_parameter_names_and_rejects_step_tampering(self) -> None:
        optimizer = {
            "state": {
                "model.layers.0.weight": {"step": 20, "exp_avg": "opaque"},
                "model.layers.0.bias": {"step": 20.0, "exp_avg": "opaque"},
            },
            "param_groups": [
                {
                    "params": [
                        "model.layers.0.weight",
                        "model.layers.0.bias",
                    ]
                }
            ],
        }
        scheduler = {"last_epoch": 20, "_step_count": 21}
        run_validation._validate_optimizer_scheduler_cadence(
            optimizer_state=optimizer,
            scheduler_state=scheduler,
            global_step=20,
        )

        for field, value, message in (
            ("last_epoch", 19, "last_epoch"),
            ("_step_count", 20, "step count"),
        ):
            with self.subTest(field=field):
                changed_scheduler = {**scheduler, field: value}
                with self.assertRaisesRegex(ValueError, message):
                    run_validation._validate_optimizer_scheduler_cadence(
                        optimizer_state=optimizer,
                        scheduler_state=changed_scheduler,
                        global_step=20,
                    )

        changed_optimizer = copy.deepcopy(optimizer)
        changed_optimizer["state"]["model.layers.0.bias"]["step"] = 19
        with self.assertRaisesRegex(ValueError, "AdamW parameter step"):
            run_validation._validate_optimizer_scheduler_cadence(
                optimizer_state=changed_optimizer,
                scheduler_state=scheduler,
                global_step=20,
            )

    def test_step20_source_validator_invokes_the_same_cadence_contract(self) -> None:
        payload = {
            "world_size": 1,
            "global_step": 20,
            "trainer_state": {
                "batch_partition": {
                    "protocol": "allocation_neutral_exact_global_batch_v1"
                }
            },
            "optimizer": {
                "state": {"model.weight": {"step": 20}},
                "param_groups": [{"params": ["model.weight"]}],
            },
            "scheduler": {"last_epoch": 20, "_step_count": 21},
        }
        with mock.patch.object(resume_comparison, "_validate_factorial_rank_states"):
            self.assertIs(
                resume_comparison._validate_exact_checkpoint(
                    payload,
                    name="source",
                    world_size=1,
                    max_steps=120,
                ),
                payload,
            )
            payload["scheduler"]["_step_count"] = 20
            with self.assertRaisesRegex(ValueError, "step count"):
                resume_comparison._validate_exact_checkpoint(
                    payload,
                    name="source",
                    world_size=1,
                    max_steps=120,
                )

    def test_resume_comparison_binds_full_state_and_uninterrupted_reference(self) -> None:
        source_sha256 = "9" * 64
        reference = self._artifacts(
            "calibration",
            ancestry=[],
            source_sha256="8" * 64,
        )
        ancestry = [f"sha256:{source_sha256}"]
        left = self._artifacts("resume-a", ancestry=ancestry, source_sha256=source_sha256)
        right = self._artifacts("resume-b", ancestry=ancestry, source_sha256=source_sha256)
        source = self.workspace / "calibration" / "checkpoints" / "step-00000020.pt"
        source.write_bytes(b"source")
        config = {"trainer": {"max_steps": 120}}
        with mock.patch.object(
            resume_comparison,
            "formal_artifact_binding",
            return_value={"code_commit": "2" * 40},
        ):
            payload = resume_comparison._build_comparison_payload(
                config=config,
                world_size=1,
                tolerance=1e-6,
                workspace=self.workspace,
                resume_source=source,
                resume_source_sha256=source_sha256,
                resume_source_binding=self._source_binding(source, source_sha256),
                formal_binding={"code_commit": "2" * 40},
                reference=reference,
                left=left,
                right=right,
            )
        self.assertIs(payload["passed"], True)
        self.assertTrue(all(payload["checks"].values()))
        self.assertIn("optimizer", payload["runs"]["left"]["runtime_state_hashes"])
        self.assertIn("scaler", payload["runs"]["left"]["runtime_state_hashes"])
        self.assertNotIn(
            "accelerate_state_files",
            resume_comparison._CORE_RUNTIME_STATE_NAMES,
        )

        left.checkpoint_payload["accelerate_state_files"] = {
            "state.bin": "4" * 64
        }
        left.checkpoint_payload["accelerate_state_sha256"] = "5" * 64
        right.checkpoint_payload["accelerate_state_files"] = {
            "state.bin": "6" * 64
        }
        right.checkpoint_payload["accelerate_state_sha256"] = "7" * 64
        equivalent_serializations = resume_comparison._build_comparison_payload(
            config=config,
            world_size=1,
            tolerance=1e-6,
            workspace=self.workspace,
            resume_source=source,
            resume_source_sha256=source_sha256,
            resume_source_binding=self._source_binding(source, source_sha256),
            formal_binding={"code_commit": "2" * 40},
            reference=reference,
            left=left,
            right=right,
        )
        self.assertIs(equivalent_serializations["passed"], True)
        self.assertNotEqual(
            equivalent_serializations["runs"]["left"]["accelerator_state"],
            equivalent_serializations["runs"]["right"]["accelerator_state"],
        )
        serialized = json.dumps(equivalent_serializations, sort_keys=True)
        self.assertNotIn(str(self.workspace), serialized)
        self.assertNotIn('"workspace"', serialized)

        changed = copy.deepcopy(right)
        changed.checkpoint_payload["optimizer"] = {"state": "changed"}
        with mock.patch.object(
            resume_comparison,
            "formal_artifact_binding",
            return_value={"code_commit": "2" * 40},
        ):
            rejected = resume_comparison._build_comparison_payload(
                config=config,
                world_size=1,
                tolerance=1e-6,
                workspace=self.workspace,
                resume_source=source,
                resume_source_sha256=source_sha256,
                resume_source_binding=self._source_binding(source, source_sha256),
                formal_binding={"code_commit": "2" * 40},
                reference=reference,
                left=left,
                right=changed,
            )
        self.assertIs(rejected["passed"], False)
        self.assertIs(
            rejected["checks"]["core_runtime_state_matches_uninterrupted_reference"],
            False,
        )

        changed_strategy = copy.deepcopy(right)
        changed_strategy.checkpoint_payload["trainer_state"]["batch_partition"][
            "effective_fsdp_sharding_strategy"
        ] = "FULL_SHARD"
        with mock.patch.object(
            resume_comparison,
            "formal_artifact_binding",
            return_value={"code_commit": "2" * 40},
        ):
            rejected_strategy = resume_comparison._build_comparison_payload(
                config=config,
                world_size=1,
                tolerance=1e-6,
                workspace=self.workspace,
                resume_source=source,
                resume_source_sha256=source_sha256,
                resume_source_binding=self._source_binding(source, source_sha256),
                formal_binding={"code_commit": "2" * 40},
                reference=reference,
                left=left,
                right=changed_strategy,
            )
        self.assertIs(
            rejected_strategy["checks"]["fsdp_sharding_contract_identical"],
            False,
        )

    def test_report_validation_rebuilds_inputs_instead_of_trusting_passed_flag(self) -> None:
        source = self.workspace / "calibration" / "checkpoints" / "step-00000020.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        source_sha256 = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        reference = self._artifacts(
            "calibration",
            ancestry=[],
            source_sha256="8" * 64,
        )
        # `_artifacts` preserves the already-created step-20 source.
        source.write_bytes(b"source")
        ancestry = [f"sha256:{source_sha256}"]
        left = self._artifacts("resume-a", ancestry=ancestry, source_sha256=source_sha256)
        right = self._artifacts("resume-b", ancestry=ancestry, source_sha256=source_sha256)
        config = {"trainer": {"max_steps": 120}}
        binding = {"code_commit": "2" * 40}
        with mock.patch.object(
            resume_comparison,
            "formal_artifact_binding",
            return_value=binding,
        ):
            payload = resume_comparison._build_comparison_payload(
                config=config,
                world_size=1,
                tolerance=1e-6,
                workspace=self.workspace,
                resume_source=source,
                resume_source_sha256=source_sha256,
                resume_source_binding=self._source_binding(source, source_sha256),
                formal_binding=binding,
                reference=reference,
                left=left,
                right=right,
            )
        payload["sha256"] = sha256_value(payload)
        report = self.workspace / "distributed_resume.json"
        self._write_json(report, payload)

        with (
            mock.patch.object(
                resume_comparison,
                "formal_artifact_binding",
                return_value=binding,
            ),
            mock.patch.object(
                resume_comparison,
                "_validated_runs",
                return_value=(reference, left, right),
            ),
            mock.patch.object(
                resume_comparison,
                "_validate_resume_source_artifacts",
                return_value=self._source_binding(source, source_sha256),
            ),
        ):
            validated = resume_comparison.validate_distributed_resume_report(
                report,
                config=config,
                expected_world_size=1,
            )
            self.assertEqual(validated, payload)
            forged = dict(payload)
            forged["global_step"] = 999
            forged["sha256"] = sha256_value(
                {key: value for key, value in forged.items() if key != "sha256"}
            )
            self._write_json(report, forged)
            with self.assertRaisesRegex(ValueError, "differs from revalidated"):
                resume_comparison.validate_distributed_resume_report(
                    report,
                    config=config,
                    expected_world_size=1,
                )

            forged = copy.deepcopy(payload)
            forged["tolerance"] = 1e-5
            forged["sha256"] = sha256_value(
                {key: value for key, value in forged.items() if key != "sha256"}
            )
            self._write_json(report, forged)
            with self.assertRaisesRegex(ValueError, "fixed value 1e-6"):
                resume_comparison.validate_distributed_resume_report(
                    report,
                    config=config,
                    expected_world_size=1,
                )

    def test_resume_comparison_rejects_nonfixed_tolerance_before_reporting(self) -> None:
        source = self.workspace / "calibration" / "checkpoints" / "step-00000020.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        source_sha256 = "9" * 64
        reference = self._artifacts("calibration", ancestry=[], source_sha256="8" * 64)
        ancestry = [f"sha256:{source_sha256}"]
        left = self._artifacts("resume-a", ancestry=ancestry, source_sha256=source_sha256)
        right = self._artifacts("resume-b", ancestry=ancestry, source_sha256=source_sha256)
        with self.assertRaisesRegex(ValueError, "fixed value 1e-6"):
            resume_comparison._build_comparison_payload(
                config={"trainer": {"max_steps": 120}},
                world_size=1,
                tolerance=1e-5,
                workspace=self.workspace,
                resume_source=source,
                resume_source_sha256=source_sha256,
                resume_source_binding=self._source_binding(source, source_sha256),
                formal_binding={"code_commit": "2" * 40},
                reference=reference,
                left=left,
                right=right,
            )

    def test_resume_source_rejects_checkpoint_with_tampered_semantic_step(self) -> None:
        source = self.workspace / "calibration" / "checkpoints" / "step-00000020.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        source_sha256 = __import__("hashlib").sha256(b"source").hexdigest()
        config = {"trainer": {"max_steps": 120}}
        payload = {key: None for key in run_validation._CHECKPOINT_FIELDS}
        payload.update(
            {
                "format": "accelerate_fsdp_full_export_v1",
                "global_step": 19,
                "resolved_config": config,
                "git_commit": "3" * 40,
                "implementation_dirty": False,
                "resume_ancestry": [],
            }
        )
        reference = SimpleNamespace(manifest={"git_commit": "3" * 40})
        with mock.patch("torch.load", return_value=payload):
            with self.assertRaisesRegex(ValueError, "semantic step-20"):
                resume_comparison._validate_resume_source_artifacts(
                    source,
                    source_sha256=source_sha256,
                    reference=reference,
                    config=config,
                    world_size=1,
                )


if __name__ == "__main__":
    unittest.main()
