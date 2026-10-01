"""Fail-closed data, prospective selection, and genuine review lineage."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_example, render_target
from posttrain_circuits.experiments.protocols import student_preparation as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return prep.make_examples(prep.DIFFICULTY)


def test_proposed_artifact_matches_code_and_cannot_authorize_execution():
    raw = (ROOT / prep.PROTOCOL_PATH).read_bytes()
    actual = prep.load_student_preparation_protocol(raw)
    assert actual == prep.proposed_student_preparation_protocol()
    with pytest.raises(prep.StudentPreparationError, match="not accepted"):
        prep.load_student_preparation_protocol(raw, require_accepted=True)
    assert actual["scope"]["teacher_generated_targets"] is False
    assert actual["preserved_base"]["downstream_student_rollout_max_completion_length"] == 128
    assert actual["development"]["max_new_tokens"] == 256


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "optimizer_steps", 64),
        ("training", "input_token_budget", 2000001),
        ("training", "world_size", 1),
        ("training", "seed", True),
        ("development", "minimum_answer_correct_count", 51),
        ("development", "maximum_answer_correct_count", 308),
        ("scope", "preparation_is_student_or_g0_acceptance", True),
        ("scope", "previous_validation128_exposure_disclosed", False),
        ("data", "fit_examples", 8192),
        ("data", "filter_truncate_or_repair_rows", True),
    ],
)
def test_protocol_cannot_change_scope_budget_population_or_threshold(section, key, value):
    protocol = prep.proposed_student_preparation_protocol()
    protocol[section][key] = value
    with pytest.raises(prep.StudentPreparationError, match="differs"):
        prep.validate_student_preparation_protocol(protocol)


def test_duplicate_json_and_malformed_review_are_rejected():
    with pytest.raises(prep.StudentPreparationError, match="duplicate"):
        prep.load_student_preparation_protocol(b'{"review":null,"review":{}}')
    protocol = prep.proposed_student_preparation_protocol()
    protocol["review"] = {"status": "accepted"}
    with pytest.raises(prep.StudentPreparationError, match="review fields"):
        prep.validate_student_preparation_protocol(protocol)


def test_generated_canonical_population_is_frozen_balanced_complete_and_disjoint(examples):
    manifest = prep.dataset_manifest(examples)
    assert sha256_value(manifest) == prep.DATASET_MANIFEST_SHA256
    for role, count, seed in (("student_fit", 2048, 90000042), ("student_dev", 512, 100000042)):
        rows = examples[role]
        assert len(rows) == count
        assert Counter(row.label for row in rows) == {0: count // 2, 1: count // 2}
        assert set(Counter(row.pair_group_id for row in rows).values()) == {2}
        assert min(row.metadata["pair_seed"] for row in rows) == seed
        assert sha256_value([asdict(row) for row in rows]) == prep.EXAMPLES_SHA256[role]
    inventory = prep.isolation_inventory(examples)
    assert len(inventory["example_id"]) == 2560
    assert len(inventory["pair_group_id"]) == len(inventory["pair_seed"]) == 1280


@pytest.mark.parametrize("kind", ["order", "drop", "duplicate", "canonical_target", "seed"])
def test_preparation_population_cannot_be_filtered_reordered_or_rewritten(examples, kind):
    changed = {role: list(rows) for role, rows in examples.items()}
    if kind == "order":
        changed["student_dev"].reverse()
    elif kind == "drop":
        changed["student_fit"].pop()
    elif kind == "duplicate":
        changed["student_fit"][1] = changed["student_fit"][0]
    else:
        row = copy.deepcopy(changed["student_fit"][0])
        if kind == "seed":
            row.metadata["pair_seed"] = 42
        else:
            row.canonical_proof = []
        changed["student_fit"][0] = row
    with pytest.raises(ValueError):
        prep.validate_examples(changed)


@pytest.mark.parametrize("dimension", ["semantic", "example_id", "pair_group_id", "pair_seed"])
def test_each_isolation_dimension_rejects_a_collision(examples, dimension):
    original = examples["student_fit"][0]
    external = ProofGraphTask().generate_pair(42, prep.DIFFICULTY)[0]
    inventory = {key: set() for key in ("semantic", "example_id", "pair_group_id", "pair_seed")}
    if dimension == "semantic":
        inventory[dimension].add(prep.canonical_semantic_key(original))
        external = copy.deepcopy(original)
    elif dimension == "pair_seed":
        inventory[dimension].add(original.metadata[dimension])
        external.metadata[dimension] = original.metadata[dimension]
    else:
        inventory[dimension].add(getattr(original, dimension))
        setattr(external, dimension, getattr(original, dimension))
    with pytest.raises(prep.StudentPreparationError, match=dimension):
        prep.reject_overlap(inventory, external)


def test_isolation_requires_the_entire_original_and_teacher_populations(examples):
    with pytest.raises(prep.StudentPreparationError, match="incomplete original_family"):
        prep.audit_isolation(examples, [], {"teacher_fit": [], "teacher_dev": []})


class BoundaryTokenizer:
    chat_template = "unit-test-nonthinking"
    eos_token_id = 999
    formatted = "prefix-only<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, example, prefix=7, response=5):
        self.raw, self.target = render_example(example), render_target(example)
        self.prefix, self.response = prefix, response
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert messages == [{"role": "user", "content": self.raw}]
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return self.formatted

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls.append(text)
        if text == self.formatted:
            return [101] * self.prefix
        assert text == self.target, "joined-string retokenization is forbidden"
        return [201] * self.response


def test_encoding_keeps_exact_boundary_masks_prompt_and_supervises_eos(examples):
    tokenizer = BoundaryTokenizer(examples["student_fit"][0])
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }
    encoded = prep.encode_example(examples["student_fit"][0], tokenizer, config)
    assert encoded["input_ids"] == [101] * 7 + [201] * 5 + [999]
    assert encoded["labels"] == [-100] * 7 + [201] * 5 + [999]
    assert encoded["response_length"] == 6
    assert tokenizer.calls == [tokenizer.formatted, tokenizer.target]
    tokenizer.prefix = 1345
    with pytest.raises(prep.StudentPreparationError, match="truncation is forbidden"):
        prep.encode_example(examples["student_fit"][0], tokenizer, config)


def encoded_fit():
    return [
        {
            "example_id": str(index),
            "input_ids": [101, 201, 999],
            "labels": [-100, 201, 999],
            "prefix_length": 1,
            "response_length": 2,
            "eos_token_id": 999,
        }
        for index in range(2048)
    ]


def test_all_fit_windows_are_audited_before_training_without_budget_truncation():
    rows = encoded_fit()
    result = prep.validate_encoded_fit(rows)
    assert result["window_input_tokens"] == [192] * 32
    assert result["total_input_tokens"] == 6144
    for row in rows:
        row.update(input_ids=[101] * 1024 + [201, 999], labels=[-100] * 1024 + [201, 999], prefix_length=1024)
    with pytest.raises(prep.StudentPreparationError, match="exceeds2M"):
        prep.validate_encoded_fit(rows)


def test_preparation_development_context_keeps_long_rows_without_expanding_training(examples):
    tokenizer = BoundaryTokenizer(examples["student_dev"][0], prefix=1292, response=162)
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }
    row = prep.encode_example(examples["student_dev"][0], tokenizer, config)
    assert len(row["input_ids"]) == 1455
    protocol = prep.proposed_student_preparation_protocol()
    assert row["prefix_length"] + 256 > protocol["data"]["max_model_input_tokens"] == 1536
    assert row["prefix_length"] + 256 <= protocol["development"]["max_model_input_tokens"] == 1600
    tokenizer.response = 255
    with pytest.raises(prep.StudentPreparationError, match="truncation is forbidden"):
        prep.encode_example(examples["student_dev"][0], tokenizer, config)


@pytest.mark.parametrize("corruption", ["prompt_label", "masked_response", "eos", "duplicate", "drop"])
def test_fit_encoding_corruption_fails_before_training(corruption):
    rows = encoded_fit()
    if corruption == "prompt_label":
        rows[0]["labels"][0] = 101
    elif corruption == "masked_response":
        rows[0]["labels"][1] = -100
    elif corruption == "eos":
        rows[0]["eos_token_id"] = 998
    elif corruption == "duplicate":
        rows[0]["example_id"] = rows[1]["example_id"]
    else:
        rows.pop()
    with pytest.raises(prep.StudentPreparationError):
        prep.validate_encoded_fit(rows)


def development(counts=(51, 52, 307, 400)):
    return [
        {
            "step": step,
            "checkpoint_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
            "examples_sha256": prep.EXAMPLES_SHA256["student_dev"],
            "num_examples": 512,
            "answer_correct": count,
            "proof_correct": count,
            "format_valid": 512,
        }
        for step, count in zip(prep.CHECKPOINT_STEPS, counts, strict=True)
    ]


def test_earliest_eligible_checkpoint_wins_even_when_later_performance_is_better():
    rows = development()
    assert prep.select_checkpoint(rows) == rows[1]
    assert prep.select_checkpoint(development((307, 100, 300, 512)))["step"] == 4
    assert prep.select_checkpoint(development((51, 308, 400, 512))) is None


@pytest.mark.parametrize(
    "corruption",
    ["missing", "reorder", "denominator", "population", "float", "reused", "answer_only", "contradiction"],
)
def test_checkpoint_selection_rejects_incomplete_or_unbound_development(corruption):
    rows = development()
    if corruption == "missing":
        rows.pop()
    elif corruption == "reorder":
        rows.reverse()
    elif corruption == "denominator":
        rows[0]["num_examples"] = 128
    elif corruption == "population":
        rows[0]["examples_sha256"] = "0" * 64
    elif corruption == "float":
        rows[0]["answer_correct"] = 51.0
    elif corruption == "reused":
        rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    elif corruption == "answer_only":
        rows[0].pop("proof_correct")
    else:
        rows[0]["format_valid"] = 0
    with pytest.raises(prep.StudentPreparationError):
        prep.select_checkpoint(rows)


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in ("prereg/qwen3_v2.yaml", "prereg/amendments/qwen3_adapted_student_calibration_v6.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(prep, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(prep, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(prep, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged reviewed implementation\n")
    protocol = prep.proposed_student_preparation_protocol()
    path = tmp_path / prep.PROTOCOL_PATH
    path.write_text(json.dumps(protocol))
    git("add", ".")
    git("commit", "--quiet", "-m", "Proposed implementation")
    implementation = git("rev-parse", "HEAD")
    protocol["review"] = {
        "status": "accepted",
        "reviewed_implementation_commit": implementation,
        "reviewer": "independent-test-reviewer",
        "reviewed_at_utc": "2026-10-01T03:00:00Z",
        "rationale": "Reviewed fixture implementation only.",
    }
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Independent review only")
    return tmp_path, git, implementation


def test_real_git_distinct_review_resolves_and_unrelated_followup_is_allowed(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = prep.resolve_student_preparation_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_preparation_protocol(root).acceptance_commit == result.acceptance_commit


@pytest.mark.parametrize("mutation", ["dirty", "staged", "committed", "symlink"])
def test_real_git_source_mutations_are_rejected(reviewed_repo, mutation):
    root, git, _ = reviewed_repo
    source = root / "kernel.py"
    if mutation == "symlink":
        source.unlink()
        source.symlink_to(root / "prereg/qwen3_v2.yaml")
    else:
        source.write_text("unreviewed changed behavior\n")
        if mutation in {"staged", "committed"}:
            git("add", "kernel.py")
        if mutation == "committed":
            git("commit", "--quiet", "-m", "Unreviewed change")
    with pytest.raises((ValueError, OSError)):
        prep.resolve_student_preparation_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentPreparationError, match="exactly one"):
        prep.resolve_student_preparation_protocol(root)
