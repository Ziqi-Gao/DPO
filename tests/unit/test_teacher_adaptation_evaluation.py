"""Preserve original scoring while extending the distinct teacher-dev population."""

import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.learning.teacher.adaptation_evaluation import (
    build_adaptation_probes,
    combine_development_shards,
    measure_development_shard,
)
from posttrain_circuits.learning.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    summarize_teacher_scores,
)
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


@pytest.fixture(scope="module")
def population():
    rows = build_split(ProofGraphTask(), "validation", 512, 70_000_042, {"depth": 3, "distractors": 4})
    loaded = SimpleNamespace(
        tokenizer=build_tiny_tokenizer(), tokenizer_id="tiny", requested_tokenizer_revision="tiny-v1"
    )
    return rows, loaded


def test_full_512_probe_population_is_preserved_and_small_population_matches_original(population):
    examples, loaded = population
    actual = build_adaptation_probes(examples, loaded, {})
    assert len(actual[1]) == 768
    assert not actual[3]
    spec = importlib.util.spec_from_file_location(
        "_original_capability_eval", Path(__file__).resolve().parents[2] / "tools/sdsc_teacher_capability.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with pytest.raises(ValueError, match="population"):
        original.prefix_probes(examples, ProofGraphTask(), loaded, {})
    expected = original.prefix_probes(examples[:128], ProofGraphTask(), loaded, {})
    assert build_adaptation_probes(examples[:128], loaded, {}) == expected


def make_shards(examples, specs):
    selected = [s for s in specs if s.stage in {"first_rule_selection", "intermediate_conclusion"}]
    shards = []
    for rank in range(4):
        rows = []
        for index in range(rank, len(selected), 4):
            spec = selected[index]
            for side in (0, 1):
                score = TeacherPrefixScore(
                    probe_id=spec.probe_id,
                    stage=spec.stage,
                    prefix_kind="canonical" if side == 0 else "corrupted_or_initial_student",
                    target_ids=spec.clean_target_ids if side == 0 else spec.corrupt_target_ids,
                    top1_correct=True,
                    target_in_topk=True,
                    minimum_topk_mass=0.95,
                    causal_shift_valid=True,
                    causal_shift_logprob=0.1,
                )
                rows.append({"global_index": index, "side": side, "score": asdict(score)})
        shards.append(
            {
                "rank": rank,
                "world_size": 4,
                "generated": {e.example_id: ProofGraphTask().canonical_target(e) for e in examples[rank::4]},
                "scores": rows,
            }
        )
    return shards


@pytest.fixture(scope="module")
def measurement(population):
    examples, loaded = population
    examples = examples[:8]
    specs = build_adaptation_probes(examples, loaded, {})[1]
    return examples, specs, make_shards(examples, specs)


def test_four_rank_development_reduction_cannot_claim_original_budget_or_acceptance(measurement):
    examples, specs, shards = measurement
    result = combine_development_shards(examples, specs, list(reversed(shards)))
    assert result["metrics_passed"]
    assert result["metrics"]["exact_proof_accuracy"] == 1
    assert result["evaluation_max_new_tokens"] == 256
    assert result["historical_evaluation_max_new_tokens"] == 128
    assert result["formal_teacher_accepted"] is False
    assert result["original_128_token_readiness_pass_claim"] is False
    expected = summarize_teacher_scores(
        examples,
        {e.example_id: ProofGraphTask().canonical_target(e) for e in examples},
        [TeacherPrefixScore(**s) for s in result["prefix_scores"]],
        TeacherReadinessThresholds(),
    )
    assert result["metrics"] == expected["metrics"]
    assert result["checks"] == expected["checks"]


@pytest.mark.parametrize(
    "fault", ["rank", "world", "generation", "duplicate", "missing", "wrong_probe", "wrong_side"]
)
def test_measurement_shards_fail_closed(measurement, fault):
    examples, specs, original = measurement
    shards = copy.deepcopy(original)
    if fault == "rank":
        shards[1]["rank"] = 0
    elif fault == "world":
        shards[0]["world_size"] = 1
    elif fault == "generation":
        shards[0]["generated"].pop(next(iter(shards[0]["generated"])))
    elif fault == "duplicate":
        shards[0]["scores"].append(shards[0]["scores"][0])
    elif fault == "missing":
        shards[0]["scores"].pop()
    elif fault == "wrong_probe":
        shards[0]["scores"][0]["score"]["probe_id"] = "unrelated"
    elif fault == "wrong_side":
        shards[0]["scores"][0]["score"]["target_ids"] = (999,)
    with pytest.raises(ValueError):
        combine_development_shards(examples, specs, shards)


def test_strict_causal_gate_and_minimum_mass_are_not_averaged_away(measurement):
    examples, specs, original = measurement
    shards = copy.deepcopy(original)
    score = shards[3]["scores"][-1]["score"]
    score.update(causal_shift_valid=False, causal_shift_logprob=0.0, minimum_topk_mass=0.89)
    result = combine_development_shards(examples, specs, shards)
    assert not result["checks"]["causal_shift"]
    assert not result["checks"]["topk_mass"]
    assert not result["metrics_passed"]


def test_nan_cannot_hide_in_minimum_reduction(measurement):
    examples, specs, original = measurement
    shards = copy.deepcopy(original)
    shards[3]["scores"][-1]["score"]["minimum_topk_mass"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        combine_development_shards(examples, specs, shards)


def test_measurements_preserve_global_seeds_and_use_the_declared_teacher_only_budget(
    measurement, population, tmp_path, monkeypatch
):
    from posttrain_circuits.cli import evaluate_teacher_readiness as cli
    from posttrain_circuits.datasets.teacher_demos.contracts import TeacherCandidateOutput
    from posttrain_circuits.learning.teacher import demo_generation

    examples, specs, original = measurement
    _, loaded = population
    seeds = {}
    lookup = {}
    selected = [s for s in specs if s.stage in {"first_rule_selection", "intermediate_conclusion"}]
    for shard in original:
        for row in shard["scores"]:
            lookup[(row["global_index"], row["side"])] = TeacherPrefixScore(**row["score"])

    class Generator:
        def __init__(self, model, tokenizer, *, max_new_tokens, model_config):
            assert max_new_tokens == 256
            assert tokenizer is loaded.tokenizer
            assert model_config == {"teacher_only_test": True}

        def __call__(self, *, example, **values):
            assert values["temperature"] == 0 and values["top_p"] == 1
            assert values["top_k"] == values["min_p"] == values["candidate_index"] == 0
            assert example.example_id not in seeds
            seeds[example.example_id] = values["actual_sampling_seed"]
            return TeacherCandidateOutput(
                ProofGraphTask().canonical_target(example), [10, 20], [-0.1, -0.2], "available", "eos"
            )

    def score(_model, spec, *, top_k):
        assert top_k == 128
        index = next(index for index, value in enumerate(selected) if value.probe_id == spec.probe_id)
        return tuple(lookup[index, side] for side in (0, 1))

    monkeypatch.setattr(demo_generation, "HfTeacherCandidateGenerator", Generator)
    monkeypatch.setattr(cli, "_prefix_scores", score)
    observer = SimpleNamespace(phase=lambda *_args, **_kwargs: None)
    shards = [
        measure_development_shard(
            object(),
            loaded,
            {"teacher_only_test": True},
            examples,
            specs,
            tmp_path / str(rank),
            rank=rank,
            world_size=4,
            observe=observer,
        )
        for rank in range(4)
    ]
    assert seeds == {example.example_id: 42 + index for index, example in enumerate(examples)}
    assert combine_development_shards(examples, specs, shards)["metrics_passed"]
