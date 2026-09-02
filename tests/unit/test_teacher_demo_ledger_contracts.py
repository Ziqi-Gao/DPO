from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_AVAILABLE,
    LOGPROB_FIXTURE_UNAVAILABLE,
    TeacherDemoAttempt,
)
from posttrain_circuits.datasets.teacher_demos.store import (
    read_teacher_demo_store,
    write_teacher_demo_store,
)
from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed

SHA_A = "a" * 64
SHA_B = "b" * 64
GENERATION = {
    "teacher_id": "teacher/id",
    "teacher_revision": "requested",
    "resolved_teacher_commit": "resolved",
    "sampling_request_seed": 17,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 0,
    "min_p": 0.0,
    "candidates_per_prompt": 2,
    "verifier_version": "proofgraph-exact-v1",
}


def _attempt(
    prompt_id: str,
    candidate_index: int,
    *,
    accepted: bool,
    logprob_status: str = LOGPROB_AVAILABLE,
) -> TeacherDemoAttempt:
    behavior_logprobs = [-0.25] if logprob_status == LOGPROB_AVAILABLE else None
    return TeacherDemoAttempt(
        attempt_id=f"{prompt_id}:candidate-{candidate_index:04d}",
        prompt_id=prompt_id,
        prompt_identity_sha256=SHA_A if prompt_id == "p0" else SHA_B,
        candidate_index=candidate_index,
        raw_prompt_text=f"raw {prompt_id}",
        model_facing_prompt_text=f"model {prompt_id}",
        input_ids=[1, 2],
        response_ids=[3],
        response_text="answer",
        response_token_mask=[True],
        behavior_logprobs=behavior_logprobs,
        logprob_status=logprob_status,
        finish_reason="eos",
        teacher_id="teacher/id",
        teacher_revision="requested",
        sampling_request_seed=17,
        actual_sampling_seed=100 + candidate_index,
        sampling_protocol_id="teacher-demo-v2-prompt-identity",
        sampling_temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        verifier_reward=1.0 if accepted else 0.0,
        verification_trace={"accepted": accepted},
        accepted=accepted,
        prompt_protocol="test-v1",
        enable_thinking=False,
        chat_template_sha256=SHA_A,
        raw_prompt_sha256=SHA_A,
        model_facing_prompt_sha256=SHA_B,
        tokenizer_fingerprint=SHA_B,
    )


class TeacherDemoLedgerContractTests(unittest.TestCase):
    def test_complete_ledger_and_accepted_reference_view_have_only_file_hashes(self) -> None:
        attempts = [
            _attempt("p0", 0, accepted=True),
            _attempt("p0", 1, accepted=False),
            _attempt("p1", 0, accepted=False),
            _attempt("p1", 1, accepted=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            manifest = write_teacher_demo_store(
                root,
                attempts,
                ordered_prompt_ids=["p0", "p1"],
                prompt_manifest_hash=SHA_A,
                tokenizer_hash=SHA_B,
                generation=GENERATION,
            )
            accepted, loaded = read_teacher_demo_store(root, require_formal=True)
            ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
            view = json.loads((root / "accepted_view.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["attempts"]), 4)
        self.assertEqual([attempt.attempt_id for attempt in accepted], [
            "p0:candidate-0000",
            "p1:candidate-0001",
        ])
        self.assertTrue(all("sha256" not in row for row in ledger["attempts"]))
        self.assertEqual(view["ledger_file_sha256"], manifest["ledger_file_sha256"])
        self.assertEqual(loaded["accepted_view_file_sha256"], manifest["accepted_view_file_sha256"])

    def test_zero_success_prompt_writes_diagnostic_manifest_then_fails_closed(self) -> None:
        attempts = [
            _attempt("p0", 0, accepted=True),
            _attempt("p0", 1, accepted=False),
            _attempt("p1", 0, accepted=False),
            _attempt("p1", 1, accepted=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            with self.assertRaisesRegex(ValueError, "zero-success"):
                write_teacher_demo_store(
                    root,
                    attempts,
                    ordered_prompt_ids=["p0", "p1"],
                    prompt_manifest_hash=SHA_A,
                    tokenizer_hash=SHA_B,
                    generation=GENERATION,
                )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["zero_success_prompt_ids"], ["p1"])
        self.assertFalse(manifest["ready_for_formal_sft"])

    def test_formal_read_rejects_explicit_fixture_unavailable_logprobs(self) -> None:
        attempts = [
            _attempt(
                prompt_id,
                candidate_index,
                accepted=candidate_index == 0,
                logprob_status=(
                    LOGPROB_FIXTURE_UNAVAILABLE
                    if candidate_index == 0
                    else LOGPROB_AVAILABLE
                ),
            )
            for prompt_id in ("p0", "p1")
            for candidate_index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            write_teacher_demo_store(
                root,
                attempts,
                ordered_prompt_ids=["p0", "p1"],
                prompt_manifest_hash=SHA_A,
                tokenizer_hash=SHA_B,
                generation=GENERATION,
            )
            accepted, _ = read_teacher_demo_store(root, require_formal=False)
            self.assertEqual(len(accepted), 2)
            with self.assertRaisesRegex(ValueError, "fixture-unavailable"):
                read_teacher_demo_store(root, require_formal=True)

    def test_candidate_seed_uses_prompt_identity_not_prompt_order(self) -> None:
        seeds_first_order = {
            prompt_hash: teacher_candidate_seed(17, prompt_hash, 1)
            for prompt_hash in (SHA_A, SHA_B)
        }
        seeds_reordered = {
            prompt_hash: teacher_candidate_seed(17, prompt_hash, 1)
            for prompt_hash in (SHA_B, SHA_A)
        }
        self.assertEqual(seeds_first_order, seeds_reordered)
        self.assertNotEqual(seeds_first_order[SHA_A], seeds_first_order[SHA_B])


if __name__ == "__main__":
    unittest.main()
