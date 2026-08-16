from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from posttrain_circuits.circuits.probes import (
    PROBE_STAGES,
    CircuitProbeSpec,
    build_semantic_probe_specs,
    semantic_probe_manifest,
    sequence_log_probability,
    tokenize_probe_specs,
)
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.scientific_versions import (
    require_core_v2_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.core.types import TrajectoryBatch
from posttrain_circuits.data.splits import assert_split_isolation, build_split, load_frozen_split
from posttrain_circuits.supervision.verified_replay import VerifiedReplaySupervisor
from posttrain_circuits.tasks.proofgraph.label_leakage import audit_label_leakage
from posttrain_circuits.tasks.proofgraph.schemas import Literal
from posttrain_circuits.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    evaluate_teacher_readiness,
    validate_teacher_readiness_artifact,
)
from posttrain_circuits.training.grpo_backend import GrpoSettings, resolve_grpo_batch_contract
from posttrain_circuits.training.local_fork import (
    SharedTrajectoryCenteredPolicyGradientSupervisor,
    SharedTrajectoryUncenteredReinforceDiagnostic,
)
from posttrain_circuits.utils.smoke import build_grouped_fork_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


@pytest.mark.unit
def test_formal_loaders_reject_v1_dataset_and_circuit_artifacts(tmp_path: Path) -> None:
    split_root = tmp_path / "validation"
    split_root.mkdir()
    (split_root / "examples.jsonl").write_text("", encoding="utf-8")
    (split_root / "manifest.json").write_text(
        json.dumps(
            {
                "split_name": "validation",
                "prereg_version": "core_v1",
                "generator_version": "proofgraph-v2",
                "label_semantics": "binary_provability",
                "dataset_schema_version": "proofgraph-dataset-v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible with core_v2"):
        load_frozen_split(split_root, expected_split="validation")

    v1_circuit = {
        "prereg_version": "core_v1",
        "generator_version": "proofgraph-v2",
        "label_semantics": "binary_provability",
        "circuit_probe_schema_version": "circuit-probe-v1-prompt-end",
    }
    with pytest.raises(ValueError, match="incompatible with core_v2"):
        require_core_v2_artifact(v1_circuit, require_circuit_schema=True)

    valid = {**scientific_compatibility_fields(), "scores": {"h0": 1.0}}
    valid["sha256"] = sha256_value(valid)
    require_core_v2_artifact(valid, require_circuit_schema=True, require_hash=True)
    valid["scores"] = {"h0": 2.0}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_core_v2_artifact(valid, require_circuit_schema=True, require_hash=True)


@pytest.mark.unit
def test_paired_signed_entailment_hundreds_have_symmetric_nonempty_proofs() -> None:
    task_examples = build_split(
        __import__(
            "posttrain_circuits.tasks.proofgraph.generator", fromlist=["ProofGraphTask"]
        ).ProofGraphTask(),
        "train",
        400,
        700,
        {"depth_range": [2, 4], "distractor_range": [1, 3]},
    )
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
    from posttrain_circuits.tasks.proofgraph.verifier import closure

    task = ProofGraphTask()
    by_pair: dict[str, list] = {}
    for example in task_examples:
        by_pair.setdefault(example.pair_group_id, []).append(example)
        assert bool(example.query in closure(example)) != bool(example.query.flipped() in closure(example))
        assert example.canonical_proof
        expected = example.query if example.label == 1 else example.query.flipped()
        assert example.canonical_proof[-1].conclusion == expected
        assert task.verify(example, task.parse_response(task.canonical_target(example))).reward == 1.0
        assert "UNPROVABLE" not in task.render(example)
    assert len(by_pair) == 200
    for siblings in by_pair.values():
        positive, negative = sorted(siblings, key=lambda example: example.label, reverse=True)
        assert positive.query == negative.query
        assert positive.rules == negative.rules
        assert positive.metadata["topology_hash"] == negative.metadata["topology_hash"]
        assert len(positive.canonical_proof) == len(negative.canonical_proof)
    assert len({row.metadata["role_to_symbol"]["positive_support"] for row in task_examples}) > 20
    audit = audit_label_leakage(task_examples)
    assert audit["passed"] is True
    assert audit["metrics"]["query_only_accuracy"] == 0.5


@pytest.mark.unit
def test_pair_groups_never_cross_splits_and_fixed_support_leak_is_detected() -> None:
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask

    task = ProofGraphTask()
    discovery = build_split(task, "circuit_discovery", 40, 19, {"depth": 2})
    validation = build_split(task, "circuit_validation", 40, 19, {"depth": 2})
    assert_split_isolation({"circuit_discovery": discovery, "circuit_validation": validation})
    leaked = copy.deepcopy(build_split(task, "validation", 200, 81, {"depth": 2}))
    for example in leaked:
        if example.label == 1:
            example.facts["F01"] = Literal("FIXED_POSITIVE_SUPPORT")
    report = audit_label_leakage(leaked)
    assert report["passed"] is False
    assert report["metrics"]["bow_accuracy"] > report["thresholds"]["maximum_bow_accuracy"]


@pytest.mark.unit
def test_all_circuit_stages_use_explicit_aligned_sequence_targets() -> None:
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask

    task = ProofGraphTask()
    tokenizer = build_tiny_tokenizer()
    pair = task.make_counterfactual(
        task.generate(101, {"depth": 3, "positive": True}),
        "active_support_path_swap",
        102,
    )
    assert pair.clean_example.query == pair.corrupt_example.query
    semantic = semantic_probe_manifest(build_semantic_probe_specs([pair], subset="discovery"))
    probes = tokenize_probe_specs(
        semantic,
        tokenizer,
        tokenizer_id="local/tiny-tokenizer",
        tokenizer_revision="local-random-v1",
    )
    assert {probe.stage for probe in probes} == set(PROBE_STAGES)
    for probe in probes:
        assert len(probe.clean_input_ids) == len(probe.corrupt_input_ids)
        assert len(probe.clean_target_ids) == len(probe.clean_metric_positions)
        assert len(probe.clean_target_ids) == len(probe.corrupt_target_ids)
        assert probe.corruption_type == "active_support_path_swap"
        assert probe.changed_semantic_field == "facts.active_support"
    final = next(probe for probe in probes if probe.stage == "final_answer")
    assert final.clean_context.endswith("<answer>")
    assert final.clean_metric_positions[-1] == len(final.clean_input_ids) - 1
    process = next(probe for probe in probes if probe.stage == "intermediate_conclusion")
    assert len(process.clean_target_ids) > 1

    logits = torch.zeros((1, 4, 6))
    logits[0, 1, 2] = 2.0
    logits[0, 2, 3] = 3.0
    expected = logits[0, 1].log_softmax(-1)[2] + logits[0, 2].log_softmax(-1)[3]
    assert sequence_log_probability(logits, (2, 3), (1, 2)).item() == pytest.approx(expected.item())

    invalid = {**final.__dict__}
    invalid.update(
        stage="first_rule_selection",
        clean_target="1",
        corrupt_target="0",
    )
    with pytest.raises(ValueError, match="process-stage"):
        CircuitProbeSpec(**invalid)
    auxiliary = task.make_counterfactual(pair.clean_example, "query_flip", 103)
    assert auxiliary.corrupt_example.metadata["corruption_class"] == "query_routing_only"


@pytest.mark.unit
def test_tokenizer_alignment_rejects_then_deterministically_selects_next_pair(monkeypatch) -> None:
    from posttrain_circuits.cli import discover_circuit as discovery
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask

    task = ProofGraphTask()
    pairs = [
        task.make_counterfactual(
            task.generate(seed, {"depth": 3, "positive": True}),
            "active_support_path_swap",
            seed + 10_000,
        )
        for seed in (501, 502)
    ]
    tokenizer = build_tiny_tokenizer()
    real_tokenize = discovery.tokenize_probe_specs
    rejected_group = pairs[0].clean_example.pair_group_id

    def reject_first(manifest, *args, **kwargs):
        first = manifest["probes"][0]
        if first["semantic_pair_group_id"] == rejected_group:
            raise ValueError("synthetic tokenizer alignment failure")
        return real_tokenize(manifest, *args, **kwargs)

    monkeypatch.setattr(discovery, "tokenize_probe_specs", reject_first)
    selected, audit = discovery._select_tokenizer_aligned_pairs(
        pairs,
        count=1,
        subset="discovery",
        tokenizer=tokenizer,
        tokenizer_id="local/tiny-tokenizer",
        tokenizer_revision="local-random-v1",
    )
    assert selected == [pairs[1]]
    assert audit["rejected"][0]["pair_group_id"] == rejected_group
    assert audit["selection"] == "frozen_candidate_order_first_tokenizer_aligned"


@pytest.mark.unit
def test_teacher_correctness_and_mass_are_independent_hash_bound_gates() -> None:
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask

    task = ProofGraphTask()
    examples = list(task.generate_pair(777, {"depth": 2}))
    responses = {example.example_id: task.canonical_target(example) for example in examples}
    bindings = {
        "teacher_model_revision": "teacher-commit",
        "tokenizer_revision": "tokenizer-commit",
        "dataset_hash": "dataset-hash",
        "prefix_probe_hash": "prefix-hash",
        "code_commit": "code-commit",
        "prereg_commit": "prereg-commit",
    }
    correct_low_mass = [
        TeacherPrefixScore(
            probe_id=f"p-{stage}",
            stage=stage,
            prefix_kind="canonical",
            target_ids=(1,),
            top1_correct=True,
            target_in_topk=True,
            minimum_topk_mass=0.5,
            causal_shift_valid=True,
        )
        for stage in ("first_rule_selection", "intermediate_conclusion")
    ]
    artifact = evaluate_teacher_readiness(
        examples,
        responses,
        correct_low_mass,
        TeacherReadinessThresholds(),
        bindings=bindings,
    )
    assert artifact["checks"]["answer_accuracy"] is True
    assert artifact["checks"]["topk_mass"] is False
    assert artifact["passed"] is False
    tampered = copy.deepcopy(artifact)
    tampered["metrics"]["answer_accuracy"] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_teacher_readiness_artifact(tampered)


@pytest.mark.unit
def test_trl_global_generation_batch_contract() -> None:
    settings = GrpoSettings(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=8,
    )
    contract = resolve_grpo_batch_contract(settings, world_size=4)
    assert contract["effective_global_batch_size"] == 16
    assert contract["groups_per_update"] == 2
    with pytest.raises(ValueError, match="divisible"):
        resolve_grpo_batch_contract(
            GrpoSettings(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                num_generations=3,
            ),
            world_size=4,
        )


def _gradient(model: torch.nn.Module, supervisor: object, batch: TrajectoryBatch) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    prepared = supervisor.prepare_targets(batch, None, None)  # type: ignore[attr-defined]
    output = supervisor.compute_loss(model, prepared)  # type: ignore[attr-defined]
    output.loss.backward()
    flattened = torch.cat(
        [parameter.grad.detach().flatten() for parameter in model.parameters() if parameter.grad is not None]
    )
    assert torch.isfinite(flattened).all()
    return flattened


@pytest.mark.unit
def test_centered_pg_has_negative_advantages_and_non_replay_gradient_geometry() -> None:
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen(91)
    records = build_grouped_fork_bank(
        build_smoke_examples(1, seed=92),
        tokenizer,
        seed=93,
        group_size=4,
    )
    for record in records:
        tokens = record.input_ids + record.response_ids
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([tokens])).logits[0].float().log_softmax(-1)
        start = len(record.input_ids) - 1
        record.behavior_logprobs = [
            float(logits[start + index, token]) for index, token in enumerate(record.response_ids)
        ]
    trajectories = TrajectoryBatch(records, policy_version=0)
    centered = SharedTrajectoryCenteredPolicyGradientSupervisor(tokenizer.pad_token_id)
    prepared = centered.prepare_targets(trajectories, None, None)
    advantages = prepared.metadata["frozen_advantages"]
    assert torch.equal(advantages > 0, torch.tensor([True, True, False, False]))
    assert torch.equal(advantages < 0, torch.tensor([False, False, True, True]))

    replay_gradient = _gradient(
        model,
        VerifiedReplaySupervisor(tokenizer.pad_token_id),
        trajectories,
    )
    centered_gradient = _gradient(model, centered, trajectories)
    uncentered_gradient = _gradient(
        model,
        SharedTrajectoryUncenteredReinforceDiagnostic(tokenizer.pad_token_id),
        trajectories,
    )
    replay_centered_cosine = float(functional.cosine_similarity(replay_gradient, centered_gradient, dim=0))
    replay_uncentered_cosine = float(
        functional.cosine_similarity(replay_gradient, uncentered_gradient, dim=0)
    )
    assert float(centered_gradient.norm()) > 0.0
    assert math.isfinite(replay_centered_cosine)
    assert replay_centered_cosine < 0.999
    assert replay_uncentered_cosine == pytest.approx(1.0, abs=1e-5)

    positive_only = TrajectoryBatch(records[:2], policy_version=0)
    positive_gradient = _gradient(
        model,
        VerifiedReplaySupervisor(tokenizer.pad_token_id),
        positive_only,
    )
    assert torch.allclose(replay_gradient, positive_gradient, atol=1e-7, rtol=1e-5)
