"""Prospective names-only fit construction and unchanged preparation gates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.contracts import Literal, ProofStep, Rule
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return p.make_examples(p.DIFFICULTY)


@pytest.fixture(scope="module")
def views(examples):
    return {
        "student_fit": p.fit_views(examples["student_fit"]),
        "student_dev": p.development_views(examples["student_dev"]),
    }


def test_actual_proposed_artifact_and_fixed_training_selection_contract():
    proposal = p.load_student_name_invariant_preparation_protocol((ROOT / p.PROTOCOL_PATH).read_bytes())
    assert proposal == p.proposed_student_name_invariant_preparation_protocol()
    with pytest.raises(p.StudentNameInvariantPreparationError, match="not accepted"):
        p.validate_student_name_invariant_preparation_protocol(proposal, require_accepted=True)
    old = p.previous.proposed_student_focus_preparation_protocol()
    expected = copy.deepcopy(old["training"])
    expected["optimizer"]["learning_rate"] = 2.5e-5
    assert proposal["training"] == expected
    for key in ("qualification", "model"):
        assert proposal[key] == old[key]
    assert len(p.name_diagnostic.SCIENCE_PATHS) == 168
    assert len(p.FROZEN_SCIENCE_PATHS) == 194
    assert all(proposal["execution"][k] == v for k, v in old["execution"].items() if "2454" not in k)
    assert all(
        proposal["development"][k] == v
        for k, v in old["development"].items()
        if k not in ("max_model_input_tokens", "larger_context_scope")
    )
    assert proposal["development"]["max_model_input_tokens"] == 2456
    assert proposal["execution"]["development_context_probe_tokens"] == 2456
    assert proposal["execution"]["native_runtime_snapshot"] == p.NATIVE_RUNTIME_SNAPSHOT
    assert proposal["scope"]["failed_name_diagnostic_execution_accepted"] is False
    assert proposal["data"]["training_name_assignment"]["unmodified_branch_name_anchor_guarantee"] is False
    assert proposal["data"]["original_base_provenance_included_in_isolation"] is True
    assert len(p.FROZEN_PROTOCOL_SHA256) == 11


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "optimizer_steps", 64),
        ("training", "input_token_budget", 3_000_000),
        ("training", "complete_all_32_steps_even_if_candidate_qualifies", False),
        ("data", "fit_raw_pair_seed_start", 470000042),
        ("data", "filter_truncate_or_repair_rows", True),
        ("data", "fit_original_order_anchor_sequence_fraction", 0),
        ("development", "maximum_answer_correct_count", 256),
        ("development", "maximum_iid_minus_each_transformation_gap", 0.10),
        ("development", "minimum_each_structure_iid_proof_accuracy", 0),
        ("development", "reselection_after_formal_exposure", True),
        ("scope", "failed_name_diagnostic_execution_accepted", True),
        ("scope", "formal_generated_responses_or_scores_read", True),
        ("execution", "gpu_count", True),
        ("execution", "early_startup_timeout_seconds", 600),
    ],
)
def test_unreviewed_contract_mutations_reject(section, key, value):
    proposal = p.proposed_student_name_invariant_preparation_protocol()
    proposal[section][key] = value
    with pytest.raises(p.StudentNameInvariantPreparationError, match="differs"):
        p.validate_student_name_invariant_preparation_protocol(proposal)


@pytest.mark.parametrize("raw", [b'{"review":{},"review":{}}', b"[]", b"", b"{", b"NaN"])
def test_strict_json_rejects_invalid_inputs(raw):
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.load_student_name_invariant_preparation_protocol(raw)


def test_exact_unfiltered_uniform_mapping_and_signed_pair_consistency(examples):
    seen = {}
    for source in examples["student_fit"]:
        mapping = p.training_name_mapping(source)
        atoms = p.name_diagnostic._atoms(source)
        assert set(mapping) == set(mapping.values()) == atoms
        key = source.pair_group_id
        if key in seen:
            assert mapping == seen[key]
        seen[key] = mapping
        if source.metadata["structure"] != "branch":
            assert all(a == b for a, b in mapping.items())
            assert p.name_permuted_base(source) == source
            continue
        roles, depth = source.metadata["role_to_symbol"], source.metadata["depth"]
        pool = sorted(
            roles[f"{polarity}_intermediate_{level:02d}"] + "_" + side
            for polarity in ("positive", "negative")
            for level in range(1, depth)
            for side in ("L", "R")
        )
        target = list(pool)
        random.Random(530000042 + source.metadata["pair_seed"] - 490000042).shuffle(target)
        assert [mapping[x] for x in pool] == target
        assert all(mapping[x] == x for x in atoms - set(pool))
    assert len(seen) == 128
    assert p.name_mapping_audit(examples["student_fit"]) == p.NAME_MAPPING_AUDIT
    assert p.NAME_MAPPING_AUDIT["branch_by_depth"]["2"].get("multilevel_paths", 0) == 0
    assert sum(c["retained_same_stem_pairs"] for c in p.NAME_MAPPING_AUDIT["branch_by_depth"].values()) == 57
    assert (
        sum(
            c.get("retained_constant_suffix_paths", 0)
            for c in p.NAME_MAPPING_AUDIT["branch_by_depth"].values()
        )
        == 78
    )


def test_every_base_is_exact_graph_proof_isomorphism_with_negation_and_order(examples):
    task = ProofGraphTask()
    for source in examples["student_fit"]:
        result = p.name_permuted_base(source)
        mapping = p.training_name_mapping(source)
        inverse = {b: a for a, b in mapping.items()}

        def undo(x, inverse=inverse):
            return Literal(inverse[x.atom], x.negated)

        restored = copy.deepcopy(result)
        restored.facts = {k: undo(v) for k, v in result.facts.items()}
        restored.rules = {
            k: Rule(k, tuple(undo(v) for v in r.antecedents), undo(r.consequent))
            for k, r in result.rules.items()
        }
        restored.query = undo(result.query)
        restored.canonical_proof = [
            ProofStep(s.step_id, s.rule_id, s.citations, undo(s.conclusion)) for s in result.canonical_proof
        ]
        assert asdict(restored) == asdict(source)
        assert list(result.facts) == list(source.facts) and list(result.rules) == list(source.rules)
        assert result.metadata == source.metadata and result.query == source.query
        assert task.verify(result, task.parse_response(render_target(result))).reward == 1
        assert [s.citations for s in result.canonical_proof] == [s.citations for s in source.canonical_proof]
        assert [s.conclusion.negated for s in result.canonical_proof] == [
            s.conclusion.negated for s in source.canonical_proof
        ]


@pytest.mark.parametrize("mutation", ["seed", "depth", "structure", "role_collision"])
def test_training_map_rejects_nonfit_or_corrupt_provenance(examples, mutation):
    source = copy.deepcopy(next(x for x in examples["student_fit"] if x.metadata["structure"] == "branch"))
    if mutation == "seed":
        source.metadata["pair_seed"] = p.DEV_SEED_START
    elif mutation == "depth":
        source.metadata["depth"] = True
    elif mutation == "structure":
        source.metadata["structure"] = "unknown"
    else:
        source.metadata["role_to_symbol"]["negative_intermediate_01"] = source.metadata["role_to_symbol"][
            "positive_intermediate_01"
        ]
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.training_name_mapping(source)


def test_all_eight_views_use_working_base_and_original_order_anchors(examples, views):
    task = ProofGraphTask()
    net = Counter()
    for i, source in enumerate(examples["student_fit"]):
        working = p.name_permuted_base(source)
        group = {v.view: v for v in views["student_fit"][8 * i : 8 * i + 8]}
        assert set(group) == set(p.FIT_VIEW_NAMES)
        seed = p.FIT_TRANSFORM_SEED + i * 10000
        renamed = p._renamed(working, seed + 1)
        for name, v in group.items():
            assert v.source_example_id == source.example_id
            assert v.example.pair_group_id == source.pair_group_id
            assert v.example.metadata["training_name_mapping_sha256"] == sha256_value(
                p.training_name_mapping(source)
            )
            reference = (
                renamed
                if name
                in ("entity_symbol_renaming", "rename_fact_order", "rename_rule_order", "joint_rename_order")
                else working
            )
            assert v.example.facts == reference.facts and v.example.rules == reference.rules
            assert v.example.canonical_proof == reference.canonical_proof
            assert task.verify(v.example, task.parse_response(render_target(v.example))).reward == 1
            for kind in ("facts", "rules"):
                if name in p.ANCHOR_VIEW_NAMES:
                    assert list(getattr(v.example, kind)) == list(getattr(source, kind))
                else:
                    net[kind] += list(getattr(v.example, kind)) != list(getattr(source, kind))
    assert dict(net) == p.NET_ORDER_CHANGED_ROWS == {"facts": 1532, "rules": 1536}
    assert sha256_value([asdict(v) for v in views["student_fit"]]) == p.VIEWS_SHA256["student_fit"]


def test_fresh_development_retains_original_suite_not_training_permutation(examples, views):
    for i, source in enumerate(examples["student_dev"]):
        group = views["student_dev"][i * 6 : i * 6 + 6]
        assert group[0].example == source
        assert [v.view for v in group] == list(p.DEV_VIEW_NAMES)
        for view in group:
            assert (
                ProofGraphTask()
                .verify(view.example, ProofGraphTask().parse_response(render_target(view.example)))
                .reward
                == 1
            )
    assert sha256_value([asdict(v) for v in views["student_dev"]]) == p.VIEWS_SHA256["student_dev"]
    manifest = p.dataset_manifest(examples)
    assert sha256_value(manifest) == p.DATASET_MANIFEST_SHA256
    assert manifest["student_fit"]["name_mapping_sha256"] == p.NAME_MAPPING_SHA256


def test_global_windows_keep_structure_sign_rank_balance(examples, views):
    assert Counter(x.metadata["structure"] for x in examples["student_fit"]) == dict(
        branch=192, chain=32, converging_dag=32
    )
    for start in range(0, 2048, 64):
        window = views["student_fit"][start : start + 64]
        other = "chain" if (start // 64) % 2 == 0 else "converging_dag"
        assert Counter(v.example.metadata["structure"] for v in window) == {"branch": 48, other: 16}
        assert Counter(v.example.label for v in window) == {0: 32, 1: 32}
        for rank in (0, 1):
            shard = window[rank::2]
            assert Counter(v.view for v in shard) == dict.fromkeys(p.FIT_VIEW_NAMES, 4)
            assert Counter(v.example.label for v in shard) == {0: 16, 1: 16}
            assert Counter(v.example.metadata["structure"] for v in shard) == {"branch": 24, other: 8}
            assert sum(v.view in p.ANCHOR_VIEW_NAMES for v in shard) == 8


@pytest.mark.parametrize("mutation", ["drop", "reverse", "duplicate", "target", "seed"])
def test_base_reselection_rejected(examples, mutation):
    changed = copy.deepcopy(examples)
    if mutation == "drop":
        changed["student_fit"].pop()
    elif mutation == "reverse":
        changed["student_dev"].reverse()
    elif mutation == "duplicate":
        changed["student_fit"][1] = changed["student_fit"][0]
    elif mutation == "target":
        changed["student_fit"][0].canonical_proof = []
    else:
        changed["student_fit"][0].metadata["pair_seed"] = 42
    with pytest.raises(ValueError):
        p.validate_examples(changed)


def test_history_excludes_original_and_mapped_progenitors_and_current_cue(examples, views):
    inventory = p.isolation_inventory(examples)
    for row in (
        examples["student_fit"][0],
        p.name_permuted_base(examples["student_fit"][0]),
        views["student_fit"][0].example,
    ):
        with pytest.raises(p.StudentNameInvariantPreparationError, match="overlaps"):
            p.reject_overlap(inventory, row)
    seen_base, seen_view = {}, {}
    for name, rows, is_view in p.historical_populations([], {"teacher_fit": [], "teacher_dev": []}):
        if name in ("original_family", "teacher_fit", "teacher_dev"):
            continue
        count = 0
        for row in rows:
            p.reject_overlap(inventory, row.example if is_view else row)
            count += 1
        (seen_view if is_view else seen_base)[name] = count
    settings = p.proposed_student_name_invariant_preparation_protocol()["data"]
    assert seen_base == {
        k: v
        for k, v in settings["isolation_population_counts"].items()
        if k not in ("original_family", "teacher_fit", "teacher_dev")
    }
    assert seen_view == settings["isolation_excluded_view_counts"]
    assert seen_base["name_cue_student_dev"] == 64 and seen_view["name_cue_student_dev_views"] == 512


def count(n, correct):
    return dict(num_examples=n, answer_correct=correct, proof_correct=correct, format_valid=n)


def dev_records(examples):
    sizes = Counter(x.metadata["structure"] for x in examples["student_dev"])
    strata = {k: count(n, n // 3) for k, n in sizes.items()}
    correct = sum(r["proof_correct"] for r in strata.values())
    return [
        dict(
            step=step,
            checkpoint_sha256=f"{i + 1:064x}",
            examples_sha256=p.EXAMPLES_SHA256["student_dev"],
            **count(256, correct),
            per_view_counts={k: count(256, correct) for k in p.DEV_VIEW_NAMES},
            structure_counts=copy.deepcopy(strata),
        )
        for i, step in enumerate(p.CHECKPOINT_STEPS)
    ]


def test_earliest_eligible_requires_complete_twelve_and_unchanged_gap(examples):
    rows = dev_records(examples)
    rows[0]["per_view_counts"]["entity_symbol_renaming"] = count(256, 10)
    assert p.select_checkpoint(rows)["step"] == 2
    assert all(p.checkpoint_eligible(r) == p.previous.checkpoint_eligible(r) for r in rows)
    for r in rows:
        r["per_view_counts"]["entity_symbol_renaming"] = count(256, 10)
    assert p.select_checkpoint(rows) is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrongdev", "bool", "badstratum"])
def test_selection_rejects_partial_or_inconsistent_evidence(examples, mutation):
    rows = dev_records(examples)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    elif mutation == "wrongdev":
        rows[0]["examples_sha256"] = "0" * 64
    elif mutation == "bool":
        rows[0]["answer_correct"] = True
    else:
        rows[0]["structure_counts"]["branch"]["num_examples"] += 1
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.select_checkpoint(rows)


def test_real_pinned_tokenizer_full2048_training_and1536_dev_envelopes(views):
    spec = importlib.util.spec_from_file_location(
        "_name_prep_tokenizer", ROOT / "tools/sdsc_student_prepare_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.load_tokenizer()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    rows = [p.encode_view(v, tokenizer, config) for v in views["student_fit"]]
    audit = p.validate_encoded_fit(rows)
    assert (
        audit["fit_rows"] == 2048
        and audit["optimizer_windows"] == 32
        and audit["total_input_tokens"] <= 2_000_000
    )
    assert max(x["response_length"] for x in rows) <= 256
    assert max(x["prefix_length"] for x in rows) <= 1344
    assert max(len(x["input_ids"]) for x in rows) <= 1536
    dev = [p.encode_view(v, tokenizer, config, training=False) for v in views["student_dev"]]
    assert max(x["prefix_length"] + 256 for x in dev) == 2456
    assert max(x["response_length"] for x in dev) <= 256
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.validate_encoded_fit(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[0]["labels"][0] = bad[0]["input_ids"][0]
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.validate_encoded_fit(bad)


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for name in ("prereg/qwen3_v2.yaml", *p.FROZEN_PROTOCOL_SHA256, *p.FROZEN_HELPER_SHA256):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(p, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(p, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(p, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged scientific source\n")
    proposal = p.proposed_student_name_invariant_preparation_protocol()
    path = tmp_path / p.PROTOCOL_PATH
    path.write_text(json.dumps(proposal))
    git("add", ".")
    git("commit", "--quiet", "-m", "Implementation")
    implementation = git("rev-parse", "HEAD")
    proposal["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent-fixture",
        reviewed_at_utc="2026-10-06T03:00:00Z",
        rationale="Fixture-only review.",
    )
    path.write_text(json.dumps(proposal))
    git("add", p.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Independent review only")
    return tmp_path, git, implementation


def test_real_git_review_and_unrelated_followup(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = p.resolve_student_name_invariant_preparation_protocol(root)
    assert result.implementation_commit == implementation and result.acceptance_commit == git(
        "rev-parse", "HEAD"
    )
    (root / "unrelated.txt").write_text("outside named science\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated")
    assert (
        p.resolve_student_name_invariant_preparation_protocol(root).acceptance_commit
        == result.acceptance_commit
    )


@pytest.mark.parametrize(
    "mutation", ["dirty", "staged", "committed", "symlink", "parent_protocol", "helper", "second_review"]
)
def test_real_git_rejects_unreviewed_source_and_frozen_parent_mutations(reviewed_repo, mutation):
    root, git, _ = reviewed_repo
    target = root / "kernel.py"
    if mutation == "symlink":
        target.unlink()
        target.symlink_to(root / "prereg/qwen3_v2.yaml")
    elif mutation == "parent_protocol":
        target = root / p.name_diagnostic.PROTOCOL_PATH
        target.write_bytes(target.read_bytes() + b"\n")
    elif mutation == "helper":
        target = root / next(iter(p.FROZEN_HELPER_SHA256))
        target.write_bytes(target.read_bytes() + b"\n")
    elif mutation == "second_review":
        target = root / p.PROTOCOL_PATH
        v = json.loads(target.read_bytes())
        v["review"]["rationale"] += " changed"
        target.write_text(json.dumps(v))
        git("add", p.PROTOCOL_PATH)
        git("commit", "--quiet", "-m", "Second review")
    else:
        target.write_text("changed\n")
        if mutation in ("staged", "committed"):
            git("add", "kernel.py")
        if mutation == "committed":
            git("commit", "--quiet", "-m", "Unreviewed")
    with pytest.raises((ValueError, OSError)):
        p.resolve_student_name_invariant_preparation_protocol(root)


def test_exact_new_development_context_boundary_retains_training_and_completion_limits(views):
    view = views["student_dev"][0]

    class Tokenizer:
        chat_template = "fixture"
        eos_token_id = 999
        prefix = 2200
        response = 200
        formatted = "prefix<|im_start|>assistant\n<think>\n\n</think>\n\n"

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": view.prompt}]
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return self.formatted

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.formatted:
                return [101] * self.prefix
            assert text == render_target(view.example)
            return [201] * self.response

    token = Tokenizer()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(token.chat_template.encode()).hexdigest(),
        }
    }
    assert p.encode_view(view, token, config, training=False)["prefix_length"] + 256 == 2456
    with pytest.raises(p.StudentNameInvariantPreparationError, match="training view"):
        p.encode_view(view, token, config)
    token.prefix = 2201
    with pytest.raises(p.StudentNameInvariantPreparationError, match="exceeds2456"):
        p.encode_view(view, token, config, training=False)
    token.prefix = 1000
    token.response = 256
    with pytest.raises(p.StudentNameInvariantPreparationError, match="target exceeds"):
        p.encode_view(view, token, config, training=False)


def test_learning_rate_cannot_silently_revert_to_control():
    proposal = p.proposed_student_name_invariant_preparation_protocol()
    proposal["training"]["optimizer"]["learning_rate"] = 5e-5
    with pytest.raises(p.StudentNameInvariantPreparationError):
        p.validate_student_name_invariant_preparation_protocol(proposal)
