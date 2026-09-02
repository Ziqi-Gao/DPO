"""Pure prompt-identity seed derivation for teacher-demo candidates."""

from __future__ import annotations

from posttrain_circuits.artifacts.hashing import sha256_value

TEACHER_DEMO_SAMPLING_PROTOCOL_ID = "teacher-demo-v2-prompt-identity"


def teacher_candidate_seed(
    sampling_request_seed: int,
    prompt_identity_sha256: str,
    candidate_index: int,
) -> int:
    if type(sampling_request_seed) is not int:
        raise ValueError("teacher-demo request seed must be an integer")
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("teacher-demo candidate index must be non-negative")
    digest = sha256_value(
        {
            "sampling_protocol_id": TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
            "sampling_request_seed": sampling_request_seed,
            "prompt_identity_sha256": prompt_identity_sha256,
            "candidate_index": candidate_index,
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)
