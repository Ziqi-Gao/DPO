"""Teacher correctness readiness, independent from retained top-k mass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample


@dataclass(frozen=True)
class TeacherReadinessThresholds:
    minimum_teacher_answer_accuracy: float = 0.90
    minimum_teacher_exact_proof_accuracy: float = 0.85
    minimum_teacher_first_rule_top1_accuracy: float = 0.80
    minimum_teacher_intermediate_top1_accuracy: float = 0.80
    minimum_teacher_topk_mass: float = 0.90
    minimum_teacher_topk_target_coverage: float = 0.90
    minimum_corrupted_prefix_recovery_accuracy: float = 0.70
    minimum_causal_shift_logprob: float = 0.0


@dataclass(frozen=True)
class TeacherPrefixScore:
    probe_id: str
    stage: str
    prefix_kind: str
    target_ids: tuple[int, ...]
    top1_correct: bool
    target_in_topk: bool
    minimum_topk_mass: float
    causal_shift_valid: bool
    target_log_probability: float = 0.0
    alternative_log_probability: float = 0.0
    target_logprob_margin: float = 0.0
    causal_shift_logprob: float = 0.0


def evaluate_teacher_readiness(
    examples: list[TaskExample],
    generated_responses: dict[str, str],
    prefix_scores: list[TeacherPrefixScore],
    thresholds: TeacherReadinessThresholds,
    *,
    bindings: dict[str, str],
) -> dict[str, Any]:
    """Evaluate full-generation and step correctness as separate conditions."""

    if not examples:
        raise ValueError("teacher readiness requires a nonempty frozen validation set")
    if set(generated_responses) != {example.example_id for example in examples}:
        raise ValueError("teacher generation responses do not exactly cover validation examples")
    required_bindings = {
        "teacher_model_revision",
        "tokenizer_revision",
        "dataset_hash",
        "prefix_probe_hash",
        "code_commit",
        "prereg_commit",
    }
    if set(bindings) < required_bindings or any(not str(bindings[key]) for key in required_bindings):
        raise ValueError("teacher readiness bindings are incomplete")

    task = ProofGraphTask()
    answer_correct = 0
    exact_proof_correct = 0
    format_valid = 0
    rows = []
    for example in examples:
        parsed = task.parse_response(generated_responses[example.example_id])
        verified = task.verify(example, parsed)
        format_valid += int(parsed.parse_valid)
        derived_answer_correct = bool(parsed.parse_valid and parsed.answer == example.label)
        answer_correct += int(derived_answer_correct)
        exact_proof = bool(
            verified.proof_valid and verified.answer_correct and parsed.steps == example.canonical_proof
        )
        exact_proof_correct += int(exact_proof)
        rows.append(
            {
                "example_id": example.example_id,
                "format_valid": parsed.parse_valid,
                "answer_correct": derived_answer_correct,
                "exact_proof_correct": exact_proof,
                "verification_reward": verified.reward,
                "verification_error": verified.error_code,
            }
        )

    canonical = [score for score in prefix_scores if score.prefix_kind == "canonical"]
    corrupted = [score for score in prefix_scores if score.prefix_kind != "canonical"]

    def stage_accuracy(stage: str) -> float:
        selected = [score for score in canonical if score.stage == stage]
        if not selected:
            return 0.0
        return sum(score.top1_correct for score in selected) / len(selected)

    all_scores = prefix_scores
    topk_coverage = sum(score.target_in_topk for score in all_scores) / len(all_scores) if all_scores else 0.0
    minimum_mass = min((score.minimum_topk_mass for score in all_scores), default=0.0)
    recovery = sum(score.top1_correct for score in corrupted) / len(corrupted) if corrupted else 0.0
    minimum_causal_shift = min((score.causal_shift_logprob for score in all_scores), default=float("-inf"))
    metrics = {
        "answer_accuracy": answer_correct / len(examples),
        "exact_proof_accuracy": exact_proof_correct / len(examples),
        "format_validity": format_valid / len(examples),
        "first_rule_top1_accuracy": stage_accuracy("first_rule_selection"),
        "intermediate_top1_accuracy": stage_accuracy("intermediate_conclusion"),
        "topk_target_coverage": topk_coverage,
        "minimum_topk_mass": minimum_mass,
        "corrupted_prefix_recovery_accuracy": recovery,
        "minimum_causal_shift_logprob": minimum_causal_shift,
    }
    checks = {
        "answer_accuracy": metrics["answer_accuracy"] >= thresholds.minimum_teacher_answer_accuracy,
        "exact_proof_accuracy": metrics["exact_proof_accuracy"]
        >= thresholds.minimum_teacher_exact_proof_accuracy,
        "first_rule_top1_accuracy": metrics["first_rule_top1_accuracy"]
        >= thresholds.minimum_teacher_first_rule_top1_accuracy,
        "intermediate_top1_accuracy": metrics["intermediate_top1_accuracy"]
        >= thresholds.minimum_teacher_intermediate_top1_accuracy,
        "topk_mass": metrics["minimum_topk_mass"] >= thresholds.minimum_teacher_topk_mass,
        "topk_target_coverage": metrics["topk_target_coverage"]
        >= thresholds.minimum_teacher_topk_target_coverage,
        "corrupted_prefix_recovery": metrics["corrupted_prefix_recovery_accuracy"]
        >= thresholds.minimum_corrupted_prefix_recovery_accuracy,
        "causal_shift": bool(all_scores)
        and all(score.causal_shift_valid for score in all_scores)
        and metrics["minimum_causal_shift_logprob"] >= thresholds.minimum_causal_shift_logprob,
    }
    likely_next_action = None
    if not all(checks.values()):
        correctness = all(value for key, value in checks.items() if key not in {"topk_mass"})
        likely_next_action = "train/calibrate the teacher" if not correctness else "reduce task difficulty"
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(),
        "artifact_kind": "teacher_readiness",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "bindings": bindings,
        "full_generation_rows": rows,
        "prefix_scores": [asdict(score) for score in prefix_scores],
        "likely_next_action": likely_next_action,
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def validate_teacher_readiness_artifact(
    artifact: dict[str, Any],
    *,
    expected_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected = artifact.get("sha256")
    content = {key: value for key, value in artifact.items() if key != "sha256"}
    if expected != sha256_value(content):
        raise ValueError("teacher-readiness artifact hash mismatch")
    from posttrain_circuits.core.scientific_versions import require_core_v2_artifact

    require_core_v2_artifact(artifact, require_circuit_schema=True)
    if expected_bindings:
        for key, value in expected_bindings.items():
            if artifact.get("bindings", {}).get(key) != value:
                raise ValueError(f"teacher-readiness binding mismatch for {key}")
    return artifact
