"""Complete-cohort evidence replay with synthetic measurements, never GPU claims.

The tokenizer/probe fixture is deliberately synthetic; real graph generation,
canonical proofs, parsing, verification, cohort pins and candidate seed rules
are exercised. Existing tokenizer/probe suites cover their implementation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from posttrain_circuits.artifacts import teacher_acceptance as gate
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import (
    adapted_teacher_identity,
    pinned_base_teacher_identity,
)
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed
from posttrain_circuits.models.prompt_protocol import FormattedPrompt

PROTOCOL, PUBLICATION, PLAN = "a" * 64, "b" * 64, "c" * 64


class TextTokenizerFixture:
    eos_token_id = 0

    def __init__(self):
        self.texts = {}
        self.ids = {}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text not in self.ids:
            self.ids[text] = len(self.ids) + 1
            self.texts[self.ids[text]] = text
        return [self.ids[text]]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(self.texts[value] for value in ids if value != 0)


def synthetic_prompt(text, _tokenizer, _teacher):
    base = pinned_base_teacher_identity()
    return FormattedPrompt(
        text,
        "chat:" + text,
        base.prompt_protocol,
        False,
        base.chat_template_sha256,
        sha256_value(text),
        sha256_value("chat:" + text),
    )


def synthetic_probes(examples, _loaded, _teacher):
    specs = [
        SimpleNamespace(
            probe_id=f"{example.example_id}:{stage}",
            stage=stage,
            clean_target_ids=(3, 4),
            corrupt_target_ids=(3, 5),
            clean_input_ids=(1, 2, 3),
            corrupt_input_ids=(6, 7, 8, 3),
            clean_metric_positions=(1, 2),
            corrupt_metric_positions=(2, 3),
            clean_context="clean",
            corrupt_context="corrupt",
        )
        for example in examples[::2]
        for stage in ("first_rule_selection", "intermediate_conclusion")
    ]
    return {"sha256": "d" * 64}, specs, {"sha256": "e" * 64}, []


@pytest.fixture(scope="module")
def data():
    config = compose_config(
        [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "task.num_examples=256",
            "state_source.num_candidates=8",
            "seed=42",
        ]
    )
    task = ProofGraphTask()
    train = build_split(task, "train", 256, 42, config["task"])
    validation = build_split(task, "validation", 256, 42, config["task"])
    examples = {
        "development": build_split(task, "validation", 512, 70_000_042, config["task"]),
        "train_probe": train[:32],
        "formal_store": train,
        "supplemental": validation[128:256],
        "formal_readiness": validation[:128],
    }
    return config["teacher"], examples


@pytest.fixture
def context(data, monkeypatch):
    teacher, examples = data
    monkeypatch.setattr(gate, "_runtime_contract", lambda *_args: None)
    monkeypatch.setattr(gate, "format_model_prompt", synthetic_prompt)
    monkeypatch.setattr(gate, "build_adaptation_probes", synthetic_probes)
    tokenizer = TextTokenizerFixture()
    specs = {
        role: gate.build_evaluation_spec(role, rows, tokenizer, teacher, protocol_sha256=PROTOCOL)
        for role, rows in examples.items()
    }
    return SimpleNamespace(teacher=teacher, examples=examples, tokenizer=tokenizer, specs=specs)


def dense_manifest(step):
    return gate.sealed(
        {
            "artifact_kind": "adapted_dense_teacher",
            "base_revision": pinned_base_teacher_identity().base_revision,
            "adaptation_plan_sha256": PLAN,
            "formal_teacher_accepted": False,
            "training_provenance": {
                "optimizer_steps": step,
                "dataset_manifest_sha256": gate.FULL_FIT_DATASET_SHA256,
            },
            "files": [{"path": "merged/model.safetensors", "size": 123, "sha256": sha256_value(step)}],
        }
    )


def identity(step=128):
    manifest = dense_manifest(step)
    return adapted_teacher_identity(manifest, expected_manifest_sha256=manifest["sha256"])


def envelope(ctx, role, model, predecessor, *, passed=True):
    rows, spec, task = ctx.examples[role], ctx.specs[role], ProofGraphTask()
    payload = {
        "format_version": 1,
        "artifact_kind": "adapted_teacher_candidate_evidence"
        if role in gate.CANDIDATE_ROLES
        else "adapted_teacher_readiness_evidence",
        "role": role,
        "teacher_identity": model.to_mapping(),
        "spec_sha256": spec["sha256"],
        "claim_id": role + ":" + model.teacher_checkpoint_sha256[:12],
        "predecessor_sha256": predecessor,
    }
    if role in gate.READINESS_ROLES:
        generations = []
        for index, example in zip(spec["source_indices"], rows, strict=True):
            text = task.canonical_target(example) if passed else "invalid model output"
            generations.append(
                {
                    "example_id": example.example_id,
                    "global_index": index,
                    "actual_sampling_seed": 42 + index,
                    "response_text": text,
                    "response_ids": [*ctx.tokenizer.encode(text), 0],
                    "token_logprobs": [-0.1, -0.1],
                    "logprob_status": "available",
                    "finish_reason": "eos",
                }
            )
        scores = []
        for index, probe in enumerate(synthetic_probes(rows, None, None)[1]):
            for side in (0, 1):
                scores.append(
                    {
                        "global_index": index,
                        "side": side,
                        "score": {
                            "probe_id": probe.probe_id,
                            "stage": probe.stage,
                            "prefix_kind": "canonical" if side == 0 else "corrupted_or_initial_student",
                            "target_ids": list(
                                probe.clean_target_ids if side == 0 else probe.corrupt_target_ids
                            ),
                            "top1_correct": True,
                            "target_in_topk": True,
                            "minimum_topk_mass": 0.99,
                            "causal_shift_valid": True,
                            "target_log_probability": -0.2,
                            "alternative_log_probability": -2.0,
                            "target_logprob_margin": 1.8,
                            "causal_shift_logprob": 1.8,
                        },
                    }
                )
        payload.update(generations=generations, prefix_scores=scores)
    else:
        attempts = []
        for example in rows:
            text = task.canonical_target(example) if passed else "invalid model output"
            verification = task.verify(example, task.parse_response(text))
            prompt = synthetic_prompt(task.render(example), None, None)
            prompt_hash = sha256_value(asdict(example))
            for candidate in range(8):
                attempts.append(
                    {
                        "attempt_id": f"{example.example_id}:candidate-{candidate:04d}",
                        "prompt_id": example.example_id,
                        "prompt_identity_sha256": prompt_hash,
                        "candidate_index": candidate,
                        "raw_prompt_text": prompt.raw_prompt,
                        "model_facing_prompt_text": prompt.model_facing_prompt,
                        "input_ids": ctx.tokenizer.encode(prompt.model_facing_prompt),
                        "response_ids": [*ctx.tokenizer.encode(text), 0],
                        "response_text": text,
                        "response_token_mask": [True, True],
                        "behavior_logprobs": [-0.1, -0.1],
                        "logprob_status": "available",
                        "finish_reason": "eos",
                        "teacher_identity_sha256": model.sha256,
                        "sampling_request_seed": 31415,
                        "actual_sampling_seed": teacher_candidate_seed(31415, prompt_hash, candidate),
                        "sampling_protocol_id": gate.TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
                        "sampling_temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 20,
                        "min_p": 0.0,
                        "verifier_reward": verification.reward,
                        "verification_trace": asdict(verification),
                        "accepted": verification.reward == 1.0,
                        "prompt_protocol": model.prompt_protocol,
                        "enable_thinking": False,
                        "chat_template_sha256": model.chat_template_sha256,
                        "raw_prompt_sha256": prompt.raw_prompt_sha256,
                        "model_facing_prompt_sha256": prompt.model_facing_prompt_sha256,
                        "tokenizer_fingerprint": model.tokenizer_fingerprint,
                    }
                )
        payload["attempts"] = attempts
    return gate.sealed(payload)


def pin(evidence):
    return {name: evidence[name] for name in ("sha256", "claim_id", "predecessor_sha256")}


def reseal(evidence):
    evidence["sha256"] = sha256_value({key: value for key, value in evidence.items() if key != "sha256"})


def replay(ctx, evidence, *, expected_binding=None):
    role = evidence["role"]
    function = (
        gate.validate_candidate_evidence if role in gate.CANDIDATE_ROLES else gate.validate_readiness_evidence
    )
    return function(
        ctx.specs[role],
        evidence,
        examples=ctx.examples[role],
        tokenizer=ctx.tokenizer,
        teacher_config=ctx.teacher,
        identity=identity(),
        expected_binding=expected_binding or pin(evidence),
        protocol_sha256=PROTOCOL,
    )


@pytest.mark.parametrize("role", list(gate.COHORTS))
def test_complete_fixed_cohorts_replay_actual_graph_verifier_and_seed_contract(context, role):
    evidence = envelope(context, role, identity(), PUBLICATION)
    result = replay(context, evidence)
    assert result["metrics_passed"] is True
    if role in gate.CANDIDATE_ROLES:
        assert result["attempts"] == len(context.examples[role]) * 8
        assert result["covered_prompts"] == len(context.examples[role])
    else:
        assert all(result["checks"].values())
        assert result["metrics"]["exact_proof_accuracy"] == 1.0


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "source_seed",
        "decode",
        "stop",
        "logprobs",
        "causal_fabrication",
        "zero_causal",
        "mean_mass",
        "bool",
        "side",
        "old_pass",
    ],
)
def test_readiness_cannot_hide_incomplete_or_inconsistent_raw_evidence(context, fault):
    evidence = envelope(context, "supplemental", identity(), PUBLICATION)
    score = evidence["prefix_scores"][-1]["score"]
    if fault == "missing":
        evidence["generations"].pop()
    elif fault == "duplicate":
        evidence["generations"][-1] = copy.deepcopy(evidence["generations"][0])
    elif fault == "source_seed":
        evidence["generations"][0]["actual_sampling_seed"] = 42
    elif fault == "decode":
        evidence["generations"][0]["response_text"] += " fabricated"
    elif fault == "stop":
        evidence["generations"][0]["finish_reason"] = "length"
    elif fault == "logprobs":
        evidence["generations"][0]["token_logprobs"][0] = float("nan")
    elif fault == "causal_fabrication":
        score["causal_shift_logprob"] = 200.0
    elif fault == "zero_causal":
        score.update(
            target_log_probability=-2.0,
            target_logprob_margin=0.0,
            causal_shift_logprob=0.0,
            causal_shift_valid=False,
        )
    elif fault == "mean_mass":
        score["minimum_topk_mass"] = 0.89
    elif fault == "bool":
        score["top1_correct"] = 1
    elif fault == "side":
        evidence["prefix_scores"][-1]["side"] = 0
    elif fault == "old_pass":
        evidence["passed"] = True
    reseal(evidence)
    if fault in {"zero_causal", "mean_mass"}:
        result = replay(context, evidence)
        assert not result["metrics_passed"]
        assert not result["checks"]["causal_shift" if fault == "zero_causal" else "topk_mass"]
    else:
        with pytest.raises(ValueError):
            replay(context, evidence)


@pytest.mark.parametrize(
    "fault", ["missing", "order", "seed", "identity", "tokens", "verifier", "mask", "legacy"]
)
def test_candidate_raw_attempts_require_full_identity_and_verifier_replay(context, fault):
    evidence = envelope(context, "train_probe", identity(), PUBLICATION)
    row = evidence["attempts"][0]
    if fault == "missing":
        evidence["attempts"].pop()
    elif fault == "order":
        evidence["attempts"][:2] = list(reversed(evidence["attempts"][:2]))
    elif fault == "seed":
        row["actual_sampling_seed"] += 1
    elif fault == "identity":
        row["teacher_identity_sha256"] = "0" * 64
    elif fault == "tokens":
        row["input_ids"] = [999999]
    elif fault == "verifier":
        row["verification_trace"]["reward"] = 0.0
    elif fault == "mask":
        row["response_token_mask"] = [1, 1]
    elif fault == "legacy":
        row["teacher_revision"] = pinned_base_teacher_identity().base_revision
    reseal(evidence)
    with pytest.raises(ValueError):
        replay(context, evidence)


def test_candidate_failure_is_measured_not_relaxed_to_completion(context):
    evidence = envelope(context, "train_probe", identity(), PUBLICATION, passed=False)
    result = replay(context, evidence)
    assert result["attempts"] == 256 and result["covered_prompts"] == 0 and not result["metrics_passed"]


def test_changed_external_hash_or_cohort_and_thresholds_cannot_be_self_resealed(context):
    evidence = envelope(context, "formal_readiness", identity(), PUBLICATION)
    binding = pin(evidence)
    evidence["claim_id"] = "another-claim"
    reseal(evidence)
    with pytest.raises(ValueError, match="externally pinned"):
        replay(context, evidence, expected_binding=binding)
    context.specs["formal_readiness"]["thresholds"]["minimum_teacher_answer_accuracy"] = 0.1
    reseal(context.specs["formal_readiness"])
    with pytest.raises(ValueError, match="externally pinned|specification changed"):
        replay(context, evidence)
    with pytest.raises(ValueError, match="frozen ordered population"):
        gate.build_evaluation_spec(
            "train_probe",
            list(reversed(context.examples["train_probe"])),
            context.tokenizer,
            context.teacher,
            protocol_sha256=PROTOCOL,
        )


def complete_inputs(ctx):
    checkpoints, bindings, previous = [], {}, PUBLICATION
    for step in gate.CHECKPOINT_STEPS:
        evidence = envelope(ctx, "development", identity(step), previous, passed=step != 128)
        checkpoints.append({"step": step, "manifest": dense_manifest(step), "evidence": evidence})
        bindings[f"development:{step}"] = pin(evidence)
        previous = evidence["sha256"]
    selection_kwargs = dict(
        protocol_sha256=PROTOCOL,
        fit_publication_sha256=PUBLICATION,
        fit_plan_sha256=PLAN,
        checkpoints=checkpoints,
        expected_bindings=bindings,
        examples=ctx.examples["development"],
        spec=ctx.specs["development"],
        tokenizer=ctx.tokenizer,
        teacher_config=ctx.teacher,
    )
    selection = gate.validate_development_selection(**selection_kwargs)["selection"]
    assert selection["selected_step"] == 256
    stages, previous = {}, selection["sha256"]
    for role in ("train_probe", "supplemental", "formal_readiness", "formal_store"):
        evidence = envelope(ctx, role, identity(256), previous)
        stages[role], bindings[role], previous = evidence, pin(evidence), evidence["sha256"]
    return dict(
        protocol_sha256=PROTOCOL,
        fit_publication_sha256=PUBLICATION,
        fit_plan_sha256=PLAN,
        checkpoints=checkpoints,
        stages=stages,
        expected_bindings=bindings,
        examples_by_role=ctx.examples,
        specs=ctx.specs,
        tokenizer=ctx.tokenizer,
        teacher_config=ctx.teacher,
    )


def test_complete_replay_selects_first_scheduled_pass_and_requires_independent_acceptance(context):
    arguments = complete_inputs(context)
    evidence = gate.build_teacher_acceptance(**arguments)
    assert evidence["scientific_evidence_verified"] is True
    assert evidence["formal_teacher_accepted"] is False
    assert evidence["selection"]["selected_step"] == 256
    assert evidence["stage_summaries"]["formal_store"]["attempts"] == 2048
    gate.validate_teacher_acceptance(
        evidence, expected_sha256=evidence["sha256"], recomputed_evidence=evidence
    )
    attestation = gate.sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_independent_acceptance",
            "decision": "accepted",
            "reviewer": "separate-test-reviewer",
            "protocol_sha256": PROTOCOL,
            "reviewed_evidence_sha256": evidence["sha256"],
            "implementation_commit": "1" * 40,
            "acceptance_commit": "2" * 40,
            "exposure_claim_audit_sha256": "3" * 64,
            "publication_audit_sha256": "4" * 64,
        }
    )
    accepted = gate.apply_independent_acceptance(
        evidence,
        attestation,
        expected_evidence_sha256=evidence["sha256"],
        expected_attestation_sha256=attestation["sha256"],
        expected_implementation_commit="1" * 40,
        expected_acceptance_commit="2" * 40,
        recomputed_evidence=evidence,
    )
    assert accepted["formal_teacher_accepted"] is True
    assert accepted["original_128_token_readiness_pass_claim"] is False
    with pytest.raises(ValueError):
        gate.apply_independent_acceptance(
            evidence,
            attestation,
            expected_evidence_sha256=evidence["sha256"],
            expected_attestation_sha256="0" * 64,
            expected_implementation_commit="1" * 40,
            expected_acceptance_commit="2" * 40,
            recomputed_evidence=evidence,
        )
    for implementation, acceptance in (("3" * 40, "2" * 40), ("1" * 40, "3" * 40)):
        with pytest.raises(ValueError, match="resolved real protocol lineage"):
            gate.apply_independent_acceptance(
                evidence,
                attestation,
                expected_evidence_sha256=evidence["sha256"],
                expected_attestation_sha256=attestation["sha256"],
                expected_implementation_commit=implementation,
                expected_acceptance_commit=acceptance,
                recomputed_evidence=evidence,
            )


@pytest.mark.parametrize("fault", ["later_checkpoint", "missing_dev", "chain", "duplicate_claim", "bad_fit"])
def test_end_to_end_acceptance_cannot_reselect_or_skip_conditional_evidence(context, fault):
    arguments = complete_inputs(context)
    if fault == "later_checkpoint":
        row = arguments["stages"]["supplemental"]
        row["teacher_identity"] = identity(384).to_mapping()
        reseal(row)
        arguments["expected_bindings"]["supplemental"] = pin(row)
    elif fault == "missing_dev":
        arguments["checkpoints"].pop()
    elif fault == "chain":
        arguments["expected_bindings"]["formal_store"]["predecessor_sha256"] = PUBLICATION
    elif fault == "duplicate_claim":
        arguments["expected_bindings"]["formal_store"]["claim_id"] = arguments["expected_bindings"][
            "train_probe"
        ]["claim_id"]
    elif fault == "bad_fit":
        arguments["fit_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        gate.build_teacher_acceptance(**arguments)


def test_real_runtime_boundary_rejects_unpinned_tokenizer_without_any_generation(data):
    teacher, _ = data
    with pytest.raises((ValueError, AttributeError)):
        gate._runtime_contract(TextTokenizerFixture(), teacher)


def test_base_only_identity_cannot_enter_adapted_evidence(context):
    evidence = envelope(context, "formal_readiness", identity(), PUBLICATION)
    with pytest.raises(ValueError, match="base teacher"):
        gate.validate_readiness_evidence(
            context.specs["formal_readiness"],
            evidence,
            examples=context.examples["formal_readiness"],
            tokenizer=context.tokenizer,
            teacher_config=context.teacher,
            identity=pinned_base_teacher_identity(),
            expected_binding=pin(evidence),
            protocol_sha256=PROTOCOL,
        )
