from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_ID,
    AMENDMENT_RELATIVE_PATH,
    PROPOSED_REVIEW,
    ProtocolAmendmentError,
    load_protocol_amendment_bytes,
    resolve_accepted_protocol_amendment,
    validate_accepted_lineage_commit,
    validate_review_transition,
    validate_two_gpu_g0_config,
)
from posttrain_circuits.core.config import compose_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_COMMIT = "a" * 40
ACCEPTANCE_COMMIT = "b" * 40
REQUEST_COMMIT = "c" * 40
EXECUTION_COMMIT = "d" * 40


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
            "rationale": "Approved only for the bounded two-GPU seed-42 G0 gate.",
        }
        return accepted

    def test_checked_in_amendment_is_exact_and_unaccepted(self) -> None:
        self.assertEqual(self.proposed["amendment_id"], AMENDMENT_ID)
        self.assertEqual(self.proposed["review"], PROPOSED_REVIEW)
        batch = self.proposed["batch_token_invariants"]
        resource = self.proposed["resource_amendment"]
        self.assertEqual(
            resource["amended_world_size"]
            * batch["per_device_batch_size"]
            * batch["amended_gradient_accumulation_steps"],
            batch["effective_global_batch_size"],
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
                "task.num_examples=256",
                "state_source.num_candidates=8",
            ]
        )
        config["scheduler_g0"] = {
            "process_count": 2,
            "execution_profile": "qwen3-v2-g0-2gpu",
        }
        validate_two_gpu_g0_config(config, self.proposed)
        self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 8)

    def test_scientific_term_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.proposed)
        tampered["batch_token_invariants"]["effective_global_batch_size"] = 32
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
            if arguments[0:2] == ("diff", "--name-only"):
                revision = arguments[3]
                if revision == f"{IMPLEMENTATION_COMMIT}..{REQUEST_COMMIT}":
                    return "\n".join(
                        (str(AMENDMENT_RELATIVE_PATH), "docs/refactor/current_handoff.md")
                    )
                if revision == f"{REQUEST_COMMIT}..{EXECUTION_COMMIT}":
                    return "docs/refactor/current_handoff.md"
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
            if arguments[0:2] == ("diff", "--name-only"):
                revision = arguments[3]
                if revision == f"{IMPLEMENTATION_COMMIT}..{ACCEPTANCE_COMMIT}":
                    return str(AMENDMENT_RELATIVE_PATH)
                if revision == f"{ACCEPTANCE_COMMIT}..{EXECUTION_COMMIT}":
                    return "src/posttrain_circuits/cli/train.py"
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
            self.assertRaisesRegex(ProtocolAmendmentError, "post-acceptance"),
        ):
            validate_accepted_lineage_commit(
                code_root=PROJECT_ROOT,
                candidate_commit=ACCEPTANCE_COMMIT,
                current_binding=binding,
                expected_head=EXECUTION_COMMIT,
                role="GPU preflight commit",
            )


if __name__ == "__main__":
    unittest.main()
