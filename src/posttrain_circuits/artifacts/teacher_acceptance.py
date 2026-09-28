"""Replayable evidence for a separately identified adapted teacher.

This module does not generate tokens, read model weights, submit jobs, or infer
that an exposure was one-shot. Callers must first verify the published bytes,
accepted protocol, execution receipts and immutable exposure claims. Their
externally pinned hashes are required here. Evidence validation never issues its
own independent review, and existing base-teacher artifacts are not accepted.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import (
    TeacherIdentity,
    adapted_teacher_identity,
    pinned_base_teacher_identity,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS
from posttrain_circuits.datasets.teacher_demos.contracts import ATTEMPT_FIELDS, TeacherCandidateOutput
from posttrain_circuits.learning.teacher.adaptation_evaluation import build_adaptation_probes
from posttrain_circuits.learning.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    summarize_teacher_scores,
)
from posttrain_circuits.learning.teacher.seeding import (
    TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
    teacher_candidate_seed,
)
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import chat_template_sha256, format_model_prompt

COHORTS = {
    "development": (512, 0, "3bd34a2e3a743c41a5215317f220c0ae99dba66841776f81565c61768f16e4c5"),
    "train_probe": (32, 0, "3dfdee4142a1b12cb82cd9e002d631849578d5364a26de17292f249bf87ca9ed"),
    "supplemental": (128, 128, "532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073"),
    "formal_readiness": (128, 0, "ceecd7470309dbb78351f8ecba850b98b6f35bf749aa89ac101498ba4ce921e8"),
    "formal_store": (256, 0, "b2de1ba6a96f602f4ffd511f3c1e3e83104c426315123c2b3cca5bc9e216ea1c"),
}
FULL_FIT_DATASET_SHA256 = "742b62a1ee328c8d8f660106265e4a342458fe5145243ecb08368eaa745f502d"
V7_INSTRUCTION_SHA256 = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
CHECKPOINT_STEPS = (128, 256, 384, 512)
READINESS_ROLES = {"development", "supplemental", "formal_readiness"}
CANDIDATE_ROLES = {"train_probe", "formal_store"}
CANDIDATE_FIELDS = (ATTEMPT_FIELDS - {"teacher_id", "teacher_revision"}) | {"teacher_identity_sha256"}
GENERATION_FIELDS = {"example_id", "global_index", "actual_sampling_seed"} | {
    field.name for field in fields(TeacherCandidateOutput)
}
ENVELOPE_FIELDS = {
    "format_version",
    "artifact_kind",
    "role",
    "teacher_identity",
    "spec_sha256",
    "claim_id",
    "predecessor_sha256",
    "sha256",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CLAIM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def sealed(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical artifact identity; no implicit trust or scientific acceptance."""
    _require("sha256" not in payload, "cannot seal an already hashed payload")
    return {**payload, "sha256": sha256_value(payload)}


def _verify_hash(value: Any, expected: str) -> None:
    _require(isinstance(value, dict) and _sha(expected), "missing externally pinned evidence SHA")
    _require(
        value.get("sha256")
        == expected
        == sha256_value({key: item for key, item in value.items() if key != "sha256"}),
        "artifact differs from externally pinned content",
    )


def _runtime_contract(tokenizer: Any, teacher: dict[str, Any]) -> None:
    base = pinned_base_teacher_identity()
    _require(
        teacher.get("model_name_or_path") == base.base_model_id
        and teacher.get("model_revision") == base.base_revision
        and teacher.get("tokenizer_name_or_path") == base.tokenizer_id
        and teacher.get("tokenizer_revision") == base.tokenizer_revision,
        "teacher base/tokenizer provenance differs",
    )
    _require(
        tokenizer_fingerprint(tokenizer) == base.tokenizer_fingerprint
        and chat_template_sha256(tokenizer) == base.chat_template_sha256,
        "actual tokenizer or chat template differs",
    )
    _require(
        teacher.get("prompt_protocol", {}).get("name") == base.prompt_protocol
        and teacher["prompt_protocol"].get("enable_thinking") is False
        and teacher["prompt_protocol"].get("chat_template_sha256") == base.chat_template_sha256,
        "non-thinking inference protocol differs",
    )
    _require(
        hashlib.sha256(RESPONSE_FORMAT_INSTRUCTIONS.encode()).hexdigest() == V7_INSTRUCTION_SHA256,
        "v7 response instructions changed",
    )


def _prepare_spec(role, examples, tokenizer, teacher, protocol_sha256):
    _require(role in COHORTS and _sha(protocol_sha256), "unknown cohort or protocol identity")
    count, start, population_sha = COHORTS[role]
    _require(
        len(examples) == count
        and len({example.example_id for example in examples}) == count
        and sha256_value([asdict(example) for example in examples]) == population_sha,
        "examples differ from the frozen ordered population",
    )
    _runtime_contract(tokenizer, teacher)
    task = ProofGraphTask()
    prompt_rows = []
    for example in examples:
        prompt = format_model_prompt(task.render(example), tokenizer, teacher)
        ids = list(tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False))
        _require(0 < len(ids) <= 1246, "frozen evaluation prompt exceeds token envelope")
        prompt_rows.append({"example_id": example.example_id, "input_ids": ids, **prompt.manifest_fields()})
    probes, semantic, tokenized, skipped = [], None, None, []
    if role in READINESS_ROLES:
        base = pinned_base_teacher_identity()
        loaded = SimpleNamespace(
            tokenizer=tokenizer,
            tokenizer_id=base.tokenizer_id,
            requested_tokenizer_revision=base.tokenizer_revision,
        )
        semantic, probes, tokenized, skipped = build_adaptation_probes(examples, loaded, teacher)
    spec = sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_evaluation_spec",
            "role": role,
            "protocol_sha256": protocol_sha256,
            "population_sha256": population_sha,
            "ordered_example_ids": [example.example_id for example in examples],
            "source_indices": list(range(start, start + count)),
            "prompt_manifest_sha256": sha256_value(prompt_rows),
            "teacher_configuration_sha256": sha256_value(teacher),
            "instruction_sha256": V7_INSTRUCTION_SHA256,
            "max_prompt_tokens": 1246,
            "max_new_tokens": 256,
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "seed_base": 42,
                "seed_index": "original_source_index",
            }
            if role in READINESS_ROLES
            else {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "request_seed": 31415,
                "candidates_per_prompt": 8,
                "sampling_protocol_id": TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
            },
            "prefix_top_k": 128 if role in READINESS_ROLES else None,
            "semantic_prefix_manifest_sha256": semantic["sha256"] if semantic else None,
            "tokenized_prefix_manifest_sha256": tokenized["sha256"] if tokenized else None,
            "alignment_skips": skipped,
            "thresholds": asdict(TeacherReadinessThresholds()),
            "original_128_token_readiness_pass_claim": False,
        }
    )
    return spec, prompt_rows, probes


def build_evaluation_spec(role, examples, tokenizer, teacher_config, *, protocol_sha256):
    """Freeze a complete cohort before generation; protocol acceptance is external."""
    return _prepare_spec(role, examples, tokenizer, teacher_config, protocol_sha256)[0]


def _envelope(evidence, spec, identity, expected_binding, *, candidate):
    extra = {"attempts"} if candidate else {"generations", "prefix_scores"}
    _require(set(evidence) == ENVELOPE_FIELDS | extra, "evidence envelope fields differ")
    _require(
        set(expected_binding) == {"sha256", "claim_id", "predecessor_sha256"},
        "external evidence binding fields differ",
    )
    _verify_hash(evidence, expected_binding["sha256"])
    for name in ("claim_id", "predecessor_sha256"):
        _require(evidence[name] == expected_binding[name], "external claim/chain binding differs")
    _require(
        isinstance(evidence["claim_id"], str)
        and _CLAIM.fullmatch(evidence["claim_id"])
        and _sha(evidence["predecessor_sha256"]),
        "invalid external exposure claim",
    )
    identity = TeacherIdentity.from_mapping(identity.to_mapping())
    _require(identity.kind == "learned_dense_checkpoint", "base teacher cannot satisfy adapted acceptance")
    _require(
        evidence["format_version"] == 1
        and type(evidence["format_version"]) is int
        and evidence["artifact_kind"]
        == ("adapted_teacher_candidate_evidence" if candidate else "adapted_teacher_readiness_evidence")
        and evidence["teacher_identity"] == identity.to_mapping()
        and evidence["spec_sha256"] == spec["sha256"]
        and evidence["role"] == spec["role"],
        "evidence identity/spec differs",
    )


def _output(row, tokenizer):
    output = TeacherCandidateOutput(
        **{field.name: row[field.name] for field in fields(TeacherCandidateOutput)}
    )
    output.validate()
    _require(
        output.logprob_status == "available"
        and output.response_ids is not None
        and 0 < len(output.response_ids) <= 256,
        "actual bounded response tokens are required",
    )
    _require(all(value <= 1e-6 for value in output.token_logprobs), "positive token log probability")
    _require(
        tokenizer.decode(output.response_ids, skip_special_tokens=True) == output.response_text,
        "response text differs from actual token decode",
    )
    expected_finish = (
        "eos"
        if output.response_ids[-1] == tokenizer.eos_token_id
        else ("length" if len(output.response_ids) == 256 else "stopped")
    )
    _require(output.finish_reason == expected_finish, "stop reason differs from actual tokens")
    return output


def _prefix_scores(rows, specifications):
    selected = [
        spec for spec in specifications if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
    ]
    _require(
        isinstance(rows, list) and len(rows) == 2 * len(selected) > 0, "prefix side population is incomplete"
    )
    indexed = {}
    score_fields = {field.name for field in fields(TeacherPrefixScore)}
    for row in rows:
        _require(
            isinstance(row, dict) and set(row) == {"global_index", "side", "score"},
            "prefix row fields differ",
        )
        index, side = row["global_index"], row["side"]
        _require(
            type(index) is int
            and 0 <= index < len(selected)
            and type(side) is int
            and side in (0, 1)
            and (index, side) not in indexed,
            "duplicate or invalid prefix side",
        )
        value = row["score"]
        _require(isinstance(value, dict) and set(value) == score_fields, "prefix score fields differ")
        spec = selected[index]
        targets = spec.clean_target_ids if side == 0 else spec.corrupt_target_ids
        _require(
            value["probe_id"] == spec.probe_id
            and value["stage"] == spec.stage
            and value["prefix_kind"] == ("canonical" if side == 0 else "corrupted_or_initial_student")
            and isinstance(value["target_ids"], list | tuple)
            and all(type(token) is int for token in value["target_ids"])
            and tuple(value["target_ids"]) == tuple(targets),
            "prefix target/probe binding differs",
        )
        for name in ("top1_correct", "target_in_topk", "causal_shift_valid"):
            _require(type(value[name]) is bool, "prefix correctness must be boolean")
        for name in (
            "minimum_topk_mass",
            "target_log_probability",
            "alternative_log_probability",
            "target_logprob_margin",
            "causal_shift_logprob",
        ):
            _require(
                type(value[name]) in (int, float) and math.isfinite(value[name]), "nonfinite prefix score"
            )
        _require(
            0 <= value["minimum_topk_mass"] <= 1.000001
            and value["target_log_probability"] <= 1e-6
            and value["alternative_log_probability"] <= 1e-6,
            "invalid measured probability",
        )
        _require(not value["top1_correct"] or value["target_in_topk"], "top1 contradicts top128 coverage")
        indexed[index, side] = value
    result = []
    for (index, side), value in sorted(indexed.items()):
        other, spec = indexed[index, 1 - side], selected[index]
        shift = value["target_log_probability"] - other["alternative_log_probability"]
        margin = value["target_log_probability"] - value["alternative_log_probability"]
        targets = spec.clean_target_ids if side == 0 else spec.corrupt_target_ids
        inputs = spec.clean_input_ids if side == 0 else spec.corrupt_input_ids
        positions = spec.clean_metric_positions if side == 0 else spec.corrupt_metric_positions
        context = spec.clean_context if side == 0 else spec.corrupt_context
        start = len(inputs) - len(targets)
        valid = bool(context) and tuple(positions) == tuple(range(start, start + len(targets))) and shift > 0
        _require(
            math.isclose(value["causal_shift_logprob"], shift, rel_tol=1e-7, abs_tol=1e-6)
            and math.isclose(value["target_logprob_margin"], margin, rel_tol=1e-7, abs_tol=1e-6)
            and value["causal_shift_valid"] is valid,
            "causal shift/margin differs from both actual conditional histories",
        )
        result.append(
            TeacherPrefixScore(
                **{
                    **value,
                    "target_ids": tuple(targets),
                    "causal_shift_logprob": shift,
                    "causal_shift_valid": valid,
                }
            )
        )
    return result


def validate_readiness_evidence(
    spec, evidence, *, examples, tokenizer, teacher_config, identity, expected_binding, protocol_sha256
):
    """Recompute all eight reductions from actual generations and paired scores."""
    _require(spec.get("role") in READINESS_ROLES, "not a readiness cohort")
    expected, _, probes = _prepare_spec(spec["role"], examples, tokenizer, teacher_config, protocol_sha256)
    _verify_hash(spec, expected["sha256"])
    _require(spec == expected, "evaluation specification changed")
    _envelope(evidence, spec, identity, expected_binding, candidate=False)
    rows = evidence["generations"]
    _require(isinstance(rows, list) and len(rows) == len(examples), "generation population incomplete")
    generated = {}
    by_index = dict(zip(spec["source_indices"], examples, strict=True))
    for row in rows:
        _require(isinstance(row, dict) and set(row) == GENERATION_FIELDS, "generation fields differ")
        index = row["global_index"]
        _require(type(index) is int and index in by_index, "generation source index differs")
        example = by_index[index]
        _require(
            row["example_id"] == example.example_id
            and example.example_id not in generated
            and type(row["actual_sampling_seed"]) is int
            and row["actual_sampling_seed"] == 42 + index,
            "generation population or original-index seed differs",
        )
        generated[example.example_id] = _output(row, tokenizer).response_text
    scores = _prefix_scores(evidence["prefix_scores"], probes)
    return summarize_teacher_scores(examples, generated, scores, TeacherReadinessThresholds())


def validate_candidate_evidence(
    spec, evidence, *, examples, tokenizer, teacher_config, identity, expected_binding, protocol_sha256
):
    """Re-run the unchanged verifier on every candidate, including failures."""
    _require(spec.get("role") in CANDIDATE_ROLES, "not a candidate cohort")
    expected, prompts, _ = _prepare_spec(spec["role"], examples, tokenizer, teacher_config, protocol_sha256)
    _verify_hash(spec, expected["sha256"])
    _require(spec == expected, "candidate specification changed")
    _envelope(evidence, spec, identity, expected_binding, candidate=True)
    attempts = evidence["attempts"]
    _require(
        isinstance(attempts, list) and len(attempts) == len(examples) * 8,
        "candidate population is not exactly eight per prompt",
    )
    task, covered, accepted_ids = ProofGraphTask(), set(), []
    for offset, row in enumerate(attempts):
        _require(isinstance(row, dict) and set(row) == CANDIDATE_FIELDS, "candidate fields differ")
        example, prompt, candidate = examples[offset // 8], prompts[offset // 8], offset % 8
        prompt_identity = sha256_value(asdict(example))
        expected_fields = {
            "attempt_id": f"{example.example_id}:candidate-{candidate:04d}",
            "prompt_id": example.example_id,
            "prompt_identity_sha256": prompt_identity,
            "candidate_index": candidate,
            "input_ids": prompt["input_ids"],
            "raw_prompt_text": prompt["raw_prompt"],
            "model_facing_prompt_text": prompt["model_facing_prompt"],
            "raw_prompt_sha256": prompt["raw_prompt_sha256"],
            "model_facing_prompt_sha256": prompt["model_facing_prompt_sha256"],
            "prompt_protocol": identity.prompt_protocol,
            "enable_thinking": False,
            "chat_template_sha256": identity.chat_template_sha256,
            "tokenizer_fingerprint": identity.tokenizer_fingerprint,
            "teacher_identity_sha256": identity.sha256,
            "sampling_request_seed": 31415,
            "actual_sampling_seed": teacher_candidate_seed(31415, prompt_identity, candidate),
            "sampling_protocol_id": TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
            "sampling_temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
        _require(
            all(row[key] == value for key, value in expected_fields.items()),
            "candidate prompt/model/seed/sampling differs",
        )
        for name in ("candidate_index", "sampling_request_seed", "actual_sampling_seed", "top_k"):
            _require(type(row[name]) is int, "candidate integer metadata differs")
        for name in ("sampling_temperature", "top_p", "min_p"):
            _require(type(row[name]) in (int, float), "candidate numeric metadata differs")
        _require(
            isinstance(row["input_ids"], list) and all(type(token) is int for token in row["input_ids"]),
            "candidate input tokens must be integers",
        )
        _require(row["enable_thinking"] is False, "candidate thinking must be disabled")
        output = _output({**row, "token_logprobs": row["behavior_logprobs"]}, tokenizer)
        _require(
            row["response_token_mask"] == [True] * len(output.response_ids)
            and all(type(item) is bool for item in row["response_token_mask"]),
            "response mask differs",
        )
        result = task.verify(example, task.parse_response(output.response_text))
        accepted = result.reward == 1.0
        _require(
            type(row["accepted"]) is bool
            and row["accepted"] is accepted
            and type(row["verifier_reward"]) in (int, float)
            and row["verifier_reward"] == result.reward
            and sha256_value(row["verification_trace"]) == sha256_value(asdict(result)),
            "stored verifier decision differs from replay",
        )
        if accepted:
            covered.add(example.example_id)
            accepted_ids.append(row["attempt_id"])
    missing = [example.example_id for example in examples if example.example_id not in covered]
    return {
        "attempts": len(attempts),
        "prompts": len(examples),
        "covered_prompts": len(covered),
        "accepted_attempt_ids": accepted_ids,
        "zero_success_prompt_ids": missing,
        "metrics_passed": not missing,
    }


def validate_development_selection(
    *,
    protocol_sha256,
    fit_publication_sha256,
    fit_plan_sha256,
    checkpoints,
    expected_bindings,
    examples,
    spec,
    tokenizer,
    teacher_config,
):
    """Replay all scheduled dev evidence and freeze selection before any exposure."""
    _require(
        all(_sha(value) for value in (protocol_sha256, fit_publication_sha256, fit_plan_sha256)),
        "missing protocol/fit publication/plan identity",
    )
    _require(
        [row.get("step") for row in checkpoints] == list(CHECKPOINT_STEPS),
        "all four predeclared merged checkpoints are required",
    )
    labels = [f"development:{step}" for step in CHECKPOINT_STEPS]
    _require(set(expected_bindings) == set(labels), "development evidence population differs")
    claims = [expected_bindings[label]["claim_id"] for label in labels]
    _require(len(set(claims)) == len(claims), "duplicate exposure claim")
    measured, selected, previous = [], None, fit_publication_sha256
    for checkpoint in checkpoints:
        _require(set(checkpoint) == {"step", "manifest", "evidence"}, "checkpoint evidence fields differ")
        step, manifest = checkpoint["step"], checkpoint["manifest"]
        identity = adapted_teacher_identity(manifest, expected_manifest_sha256=manifest.get("sha256"))
        provenance = manifest.get("training_provenance")
        _require(
            isinstance(provenance, dict)
            and provenance.get("optimizer_steps") == step
            and provenance.get("dataset_manifest_sha256") == FULL_FIT_DATASET_SHA256
            and manifest["adaptation_plan_sha256"] == fit_plan_sha256,
            "dense checkpoint fit/data/step provenance differs",
        )
        label = f"development:{step}"
        _require(
            expected_bindings[label]["predecessor_sha256"] == previous, "development evidence chain differs"
        )
        summary = validate_readiness_evidence(
            spec,
            checkpoint["evidence"],
            examples=examples,
            tokenizer=tokenizer,
            teacher_config=teacher_config,
            identity=identity,
            expected_binding=expected_bindings[label],
            protocol_sha256=protocol_sha256,
        )
        previous = expected_bindings[label]["sha256"]
        measured.append(
            {
                "step": step,
                "teacher_identity": identity.to_mapping(),
                "evidence_sha256": previous,
                "summary": summary,
            }
        )
        if selected is None and summary["metrics_passed"]:
            selected = (step, identity)
    _require(selected is not None, "no scheduled checkpoint passed all eight development gates")
    selection = sealed(
        {
            "selection_rule": "first_scheduled_all_eight_dev_pass",
            "protocol_sha256": protocol_sha256,
            "fit_publication_sha256": fit_publication_sha256,
            "development_evidence_sha256": [row["evidence_sha256"] for row in measured],
            "selected_step": selected[0],
            "teacher_identity": selected[1].to_mapping(),
        }
    )
    return {"selection": selection, "development": measured}


def build_teacher_acceptance(
    *,
    protocol_sha256,
    fit_publication_sha256,
    fit_plan_sha256,
    checkpoints,
    stages,
    expected_bindings,
    examples_by_role,
    specs,
    tokenizer,
    teacher_config,
):
    """Build unaccepted evidence after the exact declared sequence passes.

    ``expected_bindings`` comes from reviewed publication/exposure records,
    never from the evidence being validated. Stage predecessors start at the
    computed selection SHA. Their claim IDs must be distinct; hidden earlier
    exposures remain the external workflow/reviewer's responsibility.
    """
    roles = ("train_probe", "supplemental", "formal_readiness", "formal_store")
    labels = [f"development:{step}" for step in CHECKPOINT_STEPS] + list(roles)
    _require(
        set(expected_bindings) == set(labels)
        and set(stages) == set(roles)
        and set(examples_by_role) == set(COHORTS)
        and set(specs) == set(COHORTS),
        "acceptance evidence population differs",
    )
    _require(
        len({expected_bindings[label]["claim_id"] for label in labels}) == len(labels),
        "duplicate exposure claim",
    )
    development = validate_development_selection(
        protocol_sha256=protocol_sha256,
        fit_publication_sha256=fit_publication_sha256,
        fit_plan_sha256=fit_plan_sha256,
        checkpoints=checkpoints,
        expected_bindings={label: expected_bindings[label] for label in labels[:4]},
        examples=examples_by_role["development"],
        spec=specs["development"],
        tokenizer=tokenizer,
        teacher_config=teacher_config,
    )
    selection = development["selection"]
    identity = TeacherIdentity.from_mapping(selection["teacher_identity"])
    previous, summaries = selection["sha256"], {}
    for role in roles:
        _require(
            expected_bindings[role]["predecessor_sha256"] == previous, "conditional exposure sequence differs"
        )
        validator = validate_candidate_evidence if role in CANDIDATE_ROLES else validate_readiness_evidence
        summaries[role] = validator(
            specs[role],
            stages[role],
            examples=examples_by_role[role],
            tokenizer=tokenizer,
            teacher_config=teacher_config,
            identity=identity,
            expected_binding=expected_bindings[role],
            protocol_sha256=protocol_sha256,
        )
        _require(summaries[role]["metrics_passed"], f"{role} scientific gate failed")
        previous = expected_bindings[role]["sha256"]
    return sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_acceptance_evidence",
            "protocol_sha256": protocol_sha256,
            "fit_publication_sha256": fit_publication_sha256,
            "fit_plan_sha256": fit_plan_sha256,
            "teacher_identity": identity.to_mapping(),
            "selection": selection,
            "development": development["development"],
            "stage_summaries": summaries,
            "evaluation_specs": {role: specs[role]["sha256"] for role in COHORTS},
            "external_evidence_bindings": expected_bindings,
            "scientific_evidence_verified": True,
            "formal_teacher_accepted": False,
            "student_training_started": False,
            "original_128_token_readiness_pass_claim": False,
            "one_shot_history_requires_external_claim_audit": True,
            "weights_and_accounting_require_external_publication_audit": True,
        }
    )


def validate_teacher_acceptance(artifact, *, expected_sha256, recomputed_evidence):
    """Reject rehashed summaries: callers supply independently replayed raw evidence."""
    _verify_hash(artifact, expected_sha256)
    _require(
        artifact == recomputed_evidence
        and artifact.get("artifact_kind") == "adapted_teacher_acceptance_evidence"
        and artifact.get("scientific_evidence_verified") is True
        and artifact.get("formal_teacher_accepted") is False,
        "acceptance artifact differs from complete raw-evidence replay",
    )
    return artifact


def apply_independent_acceptance(
    evidence,
    attestation,
    *,
    expected_evidence_sha256,
    expected_attestation_sha256,
    expected_implementation_commit,
    expected_acceptance_commit,
    recomputed_evidence,
):
    """Bind a separately reviewed decision; never manufacture independent review.

    The caller must resolve the accepted protocol/Git lineage and review-record
    hash independently, and first replay evidence with ``build_teacher_acceptance``.
    Neither a producer's PASS flag nor an unpinned review dictionary is enough.
    """
    validate_teacher_acceptance(
        evidence, expected_sha256=expected_evidence_sha256, recomputed_evidence=recomputed_evidence
    )
    _verify_hash(attestation, expected_attestation_sha256)
    required = {
        "format_version",
        "artifact_kind",
        "decision",
        "reviewer",
        "protocol_sha256",
        "reviewed_evidence_sha256",
        "implementation_commit",
        "acceptance_commit",
        "exposure_claim_audit_sha256",
        "publication_audit_sha256",
        "sha256",
    }
    _require(
        set(attestation) == required
        and attestation["format_version"] == 1
        and type(attestation["format_version"]) is int
        and attestation["artifact_kind"] == "adapted_teacher_independent_acceptance"
        and attestation["decision"] == "accepted"
        and isinstance(attestation["reviewer"], str)
        and bool(attestation["reviewer"].strip())
        and attestation["protocol_sha256"] == evidence.get("protocol_sha256")
        and attestation["reviewed_evidence_sha256"] == expected_evidence_sha256,
        "independent review does not accept this exact evidence/protocol",
    )
    _require(
        all(
            isinstance(attestation[key], str) and _COMMIT.fullmatch(attestation[key])
            for key in ("implementation_commit", "acceptance_commit")
        )
        and attestation["implementation_commit"] != attestation["acceptance_commit"]
        and all(
            _sha(attestation[key]) for key in ("exposure_claim_audit_sha256", "publication_audit_sha256")
        ),
        "independent review lacks distinct commits and external audits",
    )
    _require(
        attestation["implementation_commit"] == expected_implementation_commit
        and attestation["acceptance_commit"] == expected_acceptance_commit,
        "independent attestation differs from the resolved real protocol lineage",
    )
    _require(
        evidence.get("artifact_kind") == "adapted_teacher_acceptance_evidence"
        and evidence.get("scientific_evidence_verified") is True
        and evidence.get("formal_teacher_accepted") is False,
        "not unaccepted validated teacher evidence",
    )
    return sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_acceptance",
            "protocol_sha256": evidence["protocol_sha256"],
            "teacher_identity": evidence["teacher_identity"],
            "evidence_sha256": expected_evidence_sha256,
            "independent_attestation_sha256": expected_attestation_sha256,
            "formal_teacher_accepted": True,
            "original_128_token_readiness_pass_claim": False,
        }
    )
