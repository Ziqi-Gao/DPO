from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from posttrain_circuits.artifacts import protocol_amendments as amendment_module
from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_ID,
    AMENDMENT_RELATIVE_PATH,
    PROPOSED_REVIEW,
    ProtocolAmendmentError,
    load_protocol_amendment_bytes,
    resolve_accepted_protocol_amendment,
    validate_accepted_lineage_commit,
    validate_review_transition,
    validate_elastic_g0_config,
)
from posttrain_circuits.core.config import compose_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_COMMIT = "a" * 40
ACCEPTANCE_COMMIT = "b" * 40
REQUEST_COMMIT = "c" * 40
EXECUTION_COMMIT = "d" * 40
SOURCE_CHANGE_COMMIT = "e" * 40
SOURCE_REVERT_COMMIT = "f" * 40
MERGE_PARENT_COMMIT = "1" * 40


class ProtocolAmendmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = (PROJECT_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
        self.proposed = load_protocol_amendment_bytes(self.raw)

    def _accepted(self) -> dict[str, object]:
        accepted = copy.deepcopy(self.proposed)
        accepted["review"] = {
            "status": "accepted",
            "reviewed_implementation_commit": IMPLEMENTATION_COMMIT,
            "reviewer": "independent-reviewer-id",
            "reviewed_at_utc": "2026-09-04T05:00:00Z",
            "rationale": "Approved only for the scheduler-managed seed-42 G0 gate.",
        }
        return accepted

    def test_checked_in_amendment_is_exact_and_unaccepted(self) -> None:
        self.assertEqual(self.proposed["amendment_id"], AMENDMENT_ID)
        self.assertEqual(self.proposed["review"], PROPOSED_REVIEW)
        batch = self.proposed["batch_token_invariants"]
        allocation = self.proposed["allocation_semantics"]
        self.assertEqual(allocation["reviewed_world_sizes"], [1, 2, 3, 4])
        self.assertEqual(allocation["request_side_gpu_count"], "forbidden")
        self.assertEqual(
            allocation["request_side_execution_profile"], "forbidden"
        )
        self.assertEqual(allocation["request_side_resources"], "forbidden")
        self.assertEqual(
            allocation["fsdp_requested_sharding_strategy"], "FULL_SHARD"
        )
        self.assertEqual(
            allocation["fsdp_effective_sharding_strategy_by_world_size"],
            {
                "1": "NO_SHARD",
                "2": "FULL_SHARD",
                "3": "FULL_SHARD",
                "4": "FULL_SHARD",
            },
        )
        self.assertEqual(batch["prompt_population_size"], 256)
        self.assertEqual(batch["max_model_input_length"], 1536)
        self.assertEqual(batch["frozen_prompt_population_max_tokens"], 1246)
        self.assertEqual(batch["teacher_demo_max_new_tokens"], 256)
        self.assertEqual(batch["derived_max_model_input_tokens"], 1502)
        self.assertEqual(batch["preflight_model_input_tokens"], 1536)
        self.assertEqual(
            batch["overlength_policy"],
            "reject_without_truncation_before_any_training_forward",
        )
        self.assertIs(batch["prompt_ids_unique"], True)
        self.assertEqual(
            batch["accepted_view_prompt_order"],
            "exactly_manifest_ordered_prompt_ids",
        )
        self.assertIs(
            self.proposed["scientific_invariants"]["full_parameter_training"],
            True,
        )
        for world_size in (1, 2, 3, 4):
            self.assertEqual(
                sum(batch["per_rank_samples_by_world_size"][str(world_size)]),
                batch["global_logical_batch_size"],
            )
        checkpoint = self.proposed["checkpoint_retry_invariants"]
        self.assertEqual(checkpoint["state_source_kind"], "teacher_demo")
        self.assertEqual(
            checkpoint["state_source_cursor_protocol"],
            "teacher-demo-round-robin-v2-accepted-view",
        )
        self.assertEqual(checkpoint["state_source_rng"], "none")
        self.assertEqual(
            checkpoint["rank_local_trainer_state"],
            "checkpointed_and_restored_per_rank",
        )
        self.assertEqual(
            checkpoint["global_token_budget_state"],
            "identical_across_all_ranks",
        )
        with self.assertRaisesRegex(ProtocolAmendmentError, "remains proposed"):
            resolve_accepted_protocol_amendment(
                code_root=PROJECT_ROOT,
                configured_path=str(AMENDMENT_RELATIVE_PATH),
            )

    def test_resolved_config_preserves_global_batch_and_token_budget(self) -> None:
        config = compose_config(
            [
                "g0=qwen3_v2_eap_separation",
                "experiment=canonical_sft",
                "state_source.num_candidates=8",
            ]
        )
        config["scheduler_g0"] = {
            "allocation_contract": "manifest_driven_scheduler_gpu_v1",
            "batch_partition_protocol": (
                "allocation_neutral_exact_global_batch_v1"
            ),
        }
        validate_elastic_g0_config(config, self.proposed)
        self.assertEqual(config["trainer"]["global_batch_size"], 64)
        self.assertEqual(config["trainer"]["max_microbatch_size"], 4)
        self.assertEqual(config["trainer"]["max_model_input_length"], 1536)
        self.assertEqual(config["state_source"]["max_prompt_tokens"], 1246)
        self.assertEqual(config["state_source"]["max_new_tokens"], 256)
        self.assertIs(config["g0"]["full_parameter_training"], True)
        self.assertNotIn("batch_size", config["trainer"])
        self.assertNotIn("gradient_accumulation_steps", config["trainer"])
        self.assertNotIn("execution_profile", config["scheduler_g0"])
        self.assertNotIn("process_count", config["scheduler_g0"])

        wrong_population = copy.deepcopy(config)
        wrong_population["task"]["num_examples"] = 255
        with self.assertRaisesRegex(ProtocolAmendmentError, "prompt population"):
            validate_elastic_g0_config(wrong_population, self.proposed)

    def test_scientific_term_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.proposed)
        tampered["batch_token_invariants"]["global_logical_batch_size"] = 32
        with self.assertRaisesRegex(ProtocolAmendmentError, "batch/token"):
            load_protocol_amendment_bytes(
                yaml.safe_dump(tampered, sort_keys=False).encode("utf-8")
            )

    def test_fsdp_requested_and_effective_strategy_tampering_is_rejected(self) -> None:
        for field, replacement in (
            ("fsdp_requested_sharding_strategy", "NO_SHARD"),
            (
                "fsdp_effective_sharding_strategy_by_world_size",
                {
                    "1": "FULL_SHARD",
                    "2": "FULL_SHARD",
                    "3": "FULL_SHARD",
                    "4": "FULL_SHARD",
                },
            ),
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.proposed)
                tampered["allocation_semantics"][field] = replacement
                with self.assertRaisesRegex(
                    ProtocolAmendmentError, "allocation semantics"
                ):
                    load_protocol_amendment_bytes(
                        yaml.safe_dump(tampered, sort_keys=False).encode("utf-8")
                    )

    def test_model_input_bound_arithmetic_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.proposed)
        tampered["batch_token_invariants"]["derived_max_model_input_tokens"] = 1501
        with self.assertRaisesRegex(ProtocolAmendmentError, "batch/token"):
            load_protocol_amendment_bytes(
                yaml.safe_dump(tampered, sort_keys=False).encode("utf-8")
            )

    def test_review_transition_accepts_only_metadata_descendant(self) -> None:
        validate_review_transition(
            proposed=self.proposed,
            accepted=self._accepted(),
            implementation_commit=IMPLEMENTATION_COMMIT,
            current_commit=ACCEPTANCE_COMMIT,
            changed_paths=(
                str(AMENDMENT_RELATIVE_PATH),
                "docs/refactor/current_handoff.md",
            ),
            implementation_is_ancestor=True,
        )

    def test_review_transition_rejects_post_review_code_change(self) -> None:
        with self.assertRaisesRegex(ProtocolAmendmentError, "non-metadata"):
            validate_review_transition(
                proposed=self.proposed,
                accepted=self._accepted(),
                implementation_commit=IMPLEMENTATION_COMMIT,
                current_commit=ACCEPTANCE_COMMIT,
                changed_paths=(
                    str(AMENDMENT_RELATIVE_PATH),
                    "src/posttrain_circuits/cli/train.py",
                ),
                implementation_is_ancestor=True,
            )

    def test_review_transition_rejects_scientific_edit(self) -> None:
        accepted = self._accepted()
        accepted["batch_token_invariants"]["token_budget"] = 1
        with self.assertRaisesRegex(ProtocolAmendmentError, "batch/token"):
            validate_review_transition(
                proposed=self.proposed,
                accepted=accepted,
                implementation_commit=IMPLEMENTATION_COMMIT,
                current_commit=ACCEPTANCE_COMMIT,
                changed_paths=(str(AMENDMENT_RELATIVE_PATH),),
                implementation_is_ancestor=True,
            )

    def test_review_transition_rejects_self_reference(self) -> None:
        with self.assertRaisesRegex(ProtocolAmendmentError, "committed after"):
            validate_review_transition(
                proposed=self.proposed,
                accepted=self._accepted(),
                implementation_commit=IMPLEMENTATION_COMMIT,
                current_commit=IMPLEMENTATION_COMMIT,
                changed_paths=(str(AMENDMENT_RELATIVE_PATH),),
                implementation_is_ancestor=True,
            )

    def test_preflight_and_request_lineage_allow_only_handoff_after_acceptance(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return EXECUTION_COMMIT
            if arguments[:2] == ("show", f"{REQUEST_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return accepted_raw
            if arguments[:2] == ("show", f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return proposed_raw
            if arguments[:2] == ("show", f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return accepted_raw
            if arguments[0] == "rev-list":
                revision = arguments[4]
                if revision == f"{IMPLEMENTATION_COMMIT}..{REQUEST_COMMIT}":
                    return "\n".join((ACCEPTANCE_COMMIT, REQUEST_COMMIT))
                if revision == f"{REQUEST_COMMIT}..{EXECUTION_COMMIT}":
                    return EXECUTION_COMMIT
            raise AssertionError(arguments)

        parents = {
            ACCEPTANCE_COMMIT: (IMPLEMENTATION_COMMIT,),
            REQUEST_COMMIT: (ACCEPTANCE_COMMIT,),
            EXECUTION_COMMIT: (REQUEST_COMMIT,),
        }
        changed_paths = {
            (IMPLEMENTATION_COMMIT, ACCEPTANCE_COMMIT): (
                str(AMENDMENT_RELATIVE_PATH),
                "docs/refactor/current_handoff.md",
            ),
            (ACCEPTANCE_COMMIT, REQUEST_COMMIT): ("docs/refactor/current_handoff.md",),
            (REQUEST_COMMIT, EXECUTION_COMMIT): ("docs/refactor/current_handoff.md",),
        }

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                side_effect=lambda _root, commit: parents[commit],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_changed_paths",
                side_effect=lambda _root, parent, commit: changed_paths[(parent, commit)],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._unsafe_untracked_paths",
                return_value=(),
            ),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=REQUEST_COMMIT,
                current_binding=binding,
                expected_head=EXECUTION_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_post_acceptance_code_delta(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return EXECUTION_COMMIT
            if arguments[:2] == ("show", f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return accepted_raw
            if arguments[:2] == ("show", f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return proposed_raw
            if arguments[0] == "rev-list":
                revision = arguments[4]
                if revision == f"{IMPLEMENTATION_COMMIT}..{ACCEPTANCE_COMMIT}":
                    return ACCEPTANCE_COMMIT
                if revision == f"{ACCEPTANCE_COMMIT}..{EXECUTION_COMMIT}":
                    return EXECUTION_COMMIT
            raise AssertionError(arguments)

        parents = {
            ACCEPTANCE_COMMIT: (IMPLEMENTATION_COMMIT,),
            EXECUTION_COMMIT: (ACCEPTANCE_COMMIT,),
        }
        changed_paths = {
            (IMPLEMENTATION_COMMIT, ACCEPTANCE_COMMIT): (str(AMENDMENT_RELATIVE_PATH),),
            (ACCEPTANCE_COMMIT, EXECUTION_COMMIT): (
                "src/posttrain_circuits/cli/train.py",
            ),
        }

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                side_effect=lambda _root, commit: parents[commit],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_changed_paths",
                side_effect=lambda _root, parent, commit: changed_paths[(parent, commit)],
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "post-acceptance"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=EXECUTION_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_source_change_then_revert(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return ACCEPTANCE_COMMIT
            if arguments[:2] == ("show", f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return accepted_raw
            if arguments[:2] == ("show", f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return proposed_raw
            if arguments[0] == "rev-list":
                return "\n".join(
                    (SOURCE_CHANGE_COMMIT, SOURCE_REVERT_COMMIT, ACCEPTANCE_COMMIT)
                )
            raise AssertionError(arguments)

        parents = {
            SOURCE_CHANGE_COMMIT: (IMPLEMENTATION_COMMIT,),
            SOURCE_REVERT_COMMIT: (SOURCE_CHANGE_COMMIT,),
            ACCEPTANCE_COMMIT: (SOURCE_REVERT_COMMIT,),
        }
        changed_paths = {
            (IMPLEMENTATION_COMMIT, SOURCE_CHANGE_COMMIT): (
                "src/posttrain_circuits/cli/train.py",
            ),
            (SOURCE_CHANGE_COMMIT, SOURCE_REVERT_COMMIT): (
                "src/posttrain_circuits/cli/train.py",
            ),
            (SOURCE_REVERT_COMMIT, ACCEPTANCE_COMMIT): (str(AMENDMENT_RELATIVE_PATH),),
        }

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                side_effect=lambda _root, commit: parents[commit],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_changed_paths",
                side_effect=lambda _root, parent, commit: changed_paths[(parent, commit)],
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "pre-acceptance"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=ACCEPTANCE_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_merge_commit(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return ACCEPTANCE_COMMIT
            if arguments[:2] == ("show", f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return accepted_raw
            if arguments[:2] == ("show", f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}"):
                return proposed_raw
            if arguments[0] == "rev-list":
                return ACCEPTANCE_COMMIT
            raise AssertionError(arguments)

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                return_value=(IMPLEMENTATION_COMMIT, MERGE_PARENT_COMMIT),
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "merge"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=ACCEPTANCE_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_post_candidate_change_then_revert(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return EXECUTION_COMMIT
            if arguments[:2] == (
                "show",
                f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return accepted_raw
            if arguments[:2] == (
                "show",
                f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return proposed_raw
            if arguments[0] == "rev-list":
                revision = arguments[4]
                if revision == f"{IMPLEMENTATION_COMMIT}..{ACCEPTANCE_COMMIT}":
                    return ACCEPTANCE_COMMIT
                if revision == f"{ACCEPTANCE_COMMIT}..{EXECUTION_COMMIT}":
                    return "\n".join((SOURCE_CHANGE_COMMIT, EXECUTION_COMMIT))
            raise AssertionError(arguments)

        parents = {
            ACCEPTANCE_COMMIT: (IMPLEMENTATION_COMMIT,),
            SOURCE_CHANGE_COMMIT: (ACCEPTANCE_COMMIT,),
            EXECUTION_COMMIT: (SOURCE_CHANGE_COMMIT,),
        }
        changed_paths = {
            (IMPLEMENTATION_COMMIT, ACCEPTANCE_COMMIT): (str(AMENDMENT_RELATIVE_PATH),),
            (ACCEPTANCE_COMMIT, SOURCE_CHANGE_COMMIT): (
                "src/posttrain_circuits/cli/train.py",
            ),
            (SOURCE_CHANGE_COMMIT, EXECUTION_COMMIT): (
                "src/posttrain_circuits/cli/train.py",
            ),
        }

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                side_effect=lambda _root, commit: parents[commit],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_changed_paths",
                side_effect=lambda _root, parent, commit: changed_paths[(parent, commit)],
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "post-acceptance"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=EXECUTION_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_second_amendment_change(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return REQUEST_COMMIT
            if arguments[:2] in {
                ("show", f"{REQUEST_COMMIT}:{AMENDMENT_RELATIVE_PATH}"),
                ("show", f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}"),
            }:
                return accepted_raw
            if arguments[:2] == (
                "show",
                f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return proposed_raw
            if arguments[0] == "rev-list":
                return "\n".join((ACCEPTANCE_COMMIT, REQUEST_COMMIT))
            raise AssertionError(arguments)

        parents = {
            ACCEPTANCE_COMMIT: (IMPLEMENTATION_COMMIT,),
            REQUEST_COMMIT: (ACCEPTANCE_COMMIT,),
        }

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_commit_parents",
                side_effect=lambda _root, commit: parents[commit],
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git_changed_paths",
                return_value=(str(AMENDMENT_RELATIVE_PATH),),
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "more than once"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=REQUEST_COMMIT,
                current_binding=binding,
                expected_head=REQUEST_COMMIT,
                role="GPU preflight commit",
            )

    def test_preflight_lineage_rejects_empty_commit_history(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return ACCEPTANCE_COMMIT
            if arguments[:2] == (
                "show",
                f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return accepted_raw
            if arguments[:2] == (
                "show",
                f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return proposed_raw
            if arguments[0] == "rev-list":
                return ""
            raise AssertionError(arguments)

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._git",
                side_effect=git_result,
            ),
            mock.patch(
                "posttrain_circuits.artifacts.protocol_amendments._is_ancestor",
                return_value=True,
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "chain is empty"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=ACCEPTANCE_COMMIT,
                role="GPU preflight commit",
            )

    def test_git_subprocesses_use_isolated_environment(self) -> None:
        expected_environment = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        with mock.patch.object(
            amendment_module.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=b"result\n", returncode=0),
        ) as run:
            self.assertEqual(amendment_module._git(PROJECT_ROOT, "rev-parse", "HEAD"), "result")
        self.assertEqual(run.call_args.kwargs["env"], expected_environment)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("text", run.call_args.kwargs)

        with mock.patch.object(
            amendment_module.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=b"", returncode=0),
        ) as run:
            self.assertTrue(
                amendment_module._is_ancestor(
                    PROJECT_ROOT,
                    IMPLEMENTATION_COMMIT,
                    ACCEPTANCE_COMMIT,
                )
            )
        self.assertEqual(run.call_args.kwargs["env"], expected_environment)
        self.assertNotIn("text", run.call_args.kwargs)

    def test_changed_paths_are_nul_delimited_strict_utf8(self) -> None:
        newline_path = b"docs/refactor/current_handoff.md\nsrc/hidden.py"
        with mock.patch.object(
            amendment_module,
            "_git_bytes",
            return_value=newline_path + b"\0",
        ) as git_bytes:
            self.assertEqual(
                amendment_module._git_changed_paths(
                    PROJECT_ROOT,
                    IMPLEMENTATION_COMMIT,
                    ACCEPTANCE_COMMIT,
                ),
                (newline_path.decode("utf-8"),),
            )
        self.assertIn("-z", git_bytes.call_args.args)

        with (
            mock.patch.object(
                amendment_module,
                "_git_bytes",
                return_value=b"src/invalid-\xff.py\0",
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "strict UTF-8"),
        ):
            amendment_module._git_changed_paths(
                PROJECT_ROOT,
                IMPLEMENTATION_COMMIT,
                ACCEPTANCE_COMMIT,
            )

    def test_lineage_commit_limit_is_fail_closed(self) -> None:
        history = "\n".join(f"{number:040x}" for number in range(1, 258))
        with (
            mock.patch.object(amendment_module, "_git", return_value=history),
            mock.patch.object(amendment_module, "_is_ancestor", return_value=True),
            self.assertRaisesRegex(ProtocolAmendmentError, "exceeds 256 commits"),
        ):
            amendment_module._linear_commit_steps(
                code_root=PROJECT_ROOT,
                start_commit=IMPLEMENTATION_COMMIT,
                end_commit=ACCEPTANCE_COMMIT,
                role="test lineage",
                allow_empty=False,
            )

    def test_lineage_rechecks_head_before_return(self) -> None:
        accepted = self._accepted()
        accepted_raw = yaml.safe_dump(accepted, sort_keys=False).rstrip()
        proposed_raw = self.raw.decode("utf-8").rstrip()
        observed_heads = iter((ACCEPTANCE_COMMIT, EXECUTION_COMMIT))

        def git_result(_root: Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "HEAD"):
                return next(observed_heads)
            if arguments[:2] == (
                "show",
                f"{ACCEPTANCE_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return accepted_raw
            if arguments[:2] == (
                "show",
                f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
            ):
                return proposed_raw
            if arguments[0] == "rev-list":
                return ACCEPTANCE_COMMIT
            raise AssertionError(arguments)

        binding = SimpleNamespace(
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        with (
            mock.patch.object(amendment_module, "_git", side_effect=git_result),
            mock.patch.object(amendment_module, "_is_ancestor", return_value=True),
            mock.patch.object(
                amendment_module,
                "_git_commit_parents",
                return_value=(IMPLEMENTATION_COMMIT,),
            ),
            mock.patch.object(
                amendment_module,
                "_git_changed_paths",
                return_value=(str(AMENDMENT_RELATIVE_PATH),),
            ),
            self.assertRaisesRegex(ProtocolAmendmentError, "HEAD changed before completion"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=ACCEPTANCE_COMMIT,
                role="GPU preflight commit",
            )

    def test_resolver_rechecks_head_bytes_and_all_untracked_status(self) -> None:
        accepted_raw = yaml.safe_dump(self._accepted(), sort_keys=False).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="opd-amendment-resolve-") as raw_root:
            code_root = Path(raw_root)
            amendment_path = code_root / AMENDMENT_RELATIVE_PATH
            amendment_path.parent.mkdir(parents=True)
            amendment_path.write_bytes(accepted_raw)
            base_path = code_root / amendment_module.BASE_PREREG_RELATIVE_PATH
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_bytes(
                (PROJECT_ROOT / amendment_module.BASE_PREREG_RELATIVE_PATH).read_bytes()
            )
            status_calls: list[tuple[str, ...]] = []
            head_calls = 0

            def git_result(_root: Path, *arguments: str) -> str:
                nonlocal head_calls
                if arguments[0] == "status":
                    status_calls.append(arguments)
                    return ""
                if arguments == ("rev-parse", "HEAD"):
                    head_calls += 1
                    return ACCEPTANCE_COMMIT
                if arguments[:2] == (
                    "show",
                    f"{IMPLEMENTATION_COMMIT}:{AMENDMENT_RELATIVE_PATH}",
                ):
                    return self.raw.decode("utf-8")
                raise AssertionError(arguments)

            with (
                mock.patch.object(amendment_module, "_git", side_effect=git_result),
                mock.patch.object(
                    amendment_module,
                    "_unsafe_untracked_paths",
                    return_value=(),
                ),
                mock.patch.object(
                    amendment_module,
                    "_validate_acceptance_chain",
                    return_value=ACCEPTANCE_COMMIT,
                ),
            ):
                binding = resolve_accepted_protocol_amendment(
                    code_root=code_root,
                    configured_path=str(AMENDMENT_RELATIVE_PATH),
                    expected_head=ACCEPTANCE_COMMIT,
                )
            self.assertEqual(binding.git_commit, ACCEPTANCE_COMMIT)
            self.assertEqual(head_calls, 2)
            self.assertEqual(len(status_calls), 2)
            for arguments in status_calls:
                self.assertEqual(
                    arguments,
                    (
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                        "--ignore-submodules=none",
                    ),
                )

            def mutate_amendment(**_kwargs: object) -> str:
                amendment_path.write_bytes(accepted_raw + b"\n")
                return ACCEPTANCE_COMMIT

            head_calls = 0
            status_calls.clear()
            with (
                mock.patch.object(amendment_module, "_git", side_effect=git_result),
                mock.patch.object(
                    amendment_module,
                    "_unsafe_untracked_paths",
                    return_value=(),
                ),
                mock.patch.object(
                    amendment_module,
                    "_validate_acceptance_chain",
                    side_effect=mutate_amendment,
                ),
                self.assertRaisesRegex(ProtocolAmendmentError, "bytes changed"),
            ):
                resolve_accepted_protocol_amendment(
                    code_root=code_root,
                    configured_path=str(AMENDMENT_RELATIVE_PATH),
                    expected_head=ACCEPTANCE_COMMIT,
                )

    def test_untracked_enumeration_finds_git_info_exclude_bypass(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".amendment-untracked-",
            dir=PROJECT_ROOT,
        ) as raw_root:
            repository = Path(raw_root)
            subprocess.run(
                ("/usr/bin/git", "-C", str(repository), "init", "-q"),
                check=True,
                capture_output=True,
            )
            (repository / ".git" / "info" / "exclude").write_text(
                "src/ignored.py\n",
                encoding="utf-8",
            )
            ignored = repository / "src" / "ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
            self.assertEqual(
                amendment_module._unsafe_untracked_paths(repository),
                ("src/ignored.py",),
            )

    def test_real_merge_parent_outside_range_is_rejected(self) -> None:
        git_environment = {
            **amendment_module._GIT_ENVIRONMENT,
            "GIT_AUTHOR_EMAIL": "opd-test@invalid",
            "GIT_AUTHOR_NAME": "OPD test",
            "GIT_COMMITTER_EMAIL": "opd-test@invalid",
            "GIT_COMMITTER_NAME": "OPD test",
        }
        with tempfile.TemporaryDirectory(prefix="opd-lineage-merge-") as raw_root:
            repository = Path(raw_root)

            def git(*arguments: str) -> str:
                result = subprocess.run(
                    ("/usr/bin/git", "-C", str(repository), *arguments),
                    check=True,
                    capture_output=True,
                    env=git_environment,
                )
                return result.stdout.decode("utf-8", errors="strict").strip()

            git("init", "-q")
            fixture = repository / "fixture.txt"
            fixture.write_text("root\n", encoding="utf-8")
            git("add", "--", fixture.name)
            git("commit", "-q", "-m", "root")
            root_commit = git("rev-parse", "HEAD")
            fixture.write_text("implementation\n", encoding="utf-8")
            git("commit", "-q", "-am", "implementation")
            implementation_commit = git("rev-parse", "HEAD")
            fixture.write_text("acceptance\n", encoding="utf-8")
            git("commit", "-q", "-am", "acceptance")
            acceptance_commit = git("rev-parse", "HEAD")
            tree = git("write-tree")
            merge_commit = git(
                "commit-tree",
                tree,
                "-p",
                acceptance_commit,
                "-p",
                root_commit,
                "-m",
                "redundant merge with ancestor outside range",
            )
            self.assertEqual(
                git("show", "-s", "--format=%P", merge_commit).split(),
                [acceptance_commit, root_commit],
            )
            with self.assertRaisesRegex(ProtocolAmendmentError, "merge"):
                amendment_module._linear_commit_steps(
                    code_root=repository,
                    start_commit=implementation_commit,
                    end_commit=merge_commit,
                    role="real merge lineage",
                    allow_empty=False,
                )


if __name__ == "__main__":
    unittest.main()
