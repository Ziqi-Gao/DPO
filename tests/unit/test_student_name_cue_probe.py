"""Actual matched-name construction, pinned token feasibility and review boundaries."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.experiments.protocols import student_name_cue_probe as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def population():
    bases = prep.make_examples()
    views = prep.development_views(bases["student_dev"])
    return bases, views


def test_declared_proposal_and_source_checkpoint_pins():
    expected = prep.proposed_student_name_cue_probe_protocol()
    assert prep.load_student_name_cue_probe_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes()) == expected
    assert len(prep.SCIENCE_PATHS) == 168 and len(prep.FROZEN_SCIENCE_PATHS) == 160
    assert expected["data"]["isolation_dimensions"] == [
        "semantic",
        "example_id",
        "pair_group_id",
        "pair_seed",
    ]
    assert expected["scope"]["training"] is False
    assert expected["analysis"]["scientific_acceptance"] is False
    assert expected["analysis"]["selected_checkpoint"] is None
    assert set(prep.SOURCE_CHECKPOINTS) == {"control", "treatment"}
    assert {x["step"] for x in prep.SOURCE_CHECKPOINTS.values()} == {32}
    assert {x["size"] for x in prep.SOURCE_CHECKPOINTS.values()} == {8127108953}
    assert {x["job_id"] for x in prep.SOURCE_CHECKPOINTS.values()} == {"54673886", "54673887"}
    assert len({x["sha256"] for x in prep.SOURCE_CHECKPOINTS.values()}) == 2
    for path, expected_sha in prep.FROZEN_PROTOCOL_SHA256.items():
        import hashlib

        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected_sha


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("scope", "training", True),
        ("scope", "checkpoint_step", 16),
        ("scope", "checkpoint_selection", True),
        ("scope", "formal_generated_responses_or_scores_read", True),
        ("data", "base_raw_pair_seed_start", 42),
        ("data", "branch_examples", 64),
        ("data", "filter_truncate_repair_rows_or_seed_search", True),
        ("development", "max_new_tokens", 512),
        ("development", "max_model_input_tokens", 4096),
        ("execution", "gpu_count", True),
        ("execution", "walltime_minutes", 180),
        ("execution", "worker_deadline_seconds", 4900),
        ("analysis", "scientific_acceptance", True),
    ],
)
def test_strict_contract_rejects_unreviewed_scope_and_resource_changes(section, key, new):
    value = prep.proposed_student_name_cue_probe_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentNameCueProbeError, match="differs"):
        prep.validate_student_name_cue_probe_protocol(value)


@pytest.mark.parametrize("raw", [b'{"review":{},"review":{}}', b"[]", b"", b"{", b"NaN"])
def test_bounded_strict_json(raw):
    with pytest.raises(prep.StudentNameCueProbeError):
        prep.load_student_name_cue_probe_protocol(raw)


def test_actual_fixed_population_and_every_target_semantics(population):
    bases, views = population
    assert len(bases["student_dev"]) == 64 and len(views) == 512
    assert Counter(x.metadata["structure"] for x in bases["student_dev"]) == {"branch": 48, "chain": 16}
    assert {x.metadata["depth"] for x in bases["student_dev"]} == {4}
    assert prep.dataset_manifest(bases)["views_sha256"] == prep.VIEWS_SHA256
    assert sha256_value(prep.dataset_manifest(bases)) == prep.DATASET_MANIFEST_SHA256
    task = ProofGraphTask()
    for index, source in enumerate(bases["student_dev"]):
        group = views[index * 8 : index * 8 + 8]
        assert [(v.block, v.condition) for v in group] == [
            (b, c) for b in prep.BLOCKS for c in prep.CONDITIONS
        ]
        frequencies = [prep._literal_frequencies(v.example) for v in group]
        assert all(value == frequencies[0] for value in frequencies)
        for view in group:
            assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1
            assert list(view.example.facts) == list(source.facts)
            assert list(view.example.rules) == list(source.rules)
            assert [(x.step_id, x.rule_id, x.citations) for x in view.example.canonical_proof] == [
                (x.step_id, x.rule_id, x.citations) for x in source.canonical_proof
            ]
            assert view.prompt == task.render(view.example)


def test_branch_factors_are_nontrivial_within_polarity_and_map_changes_middle_stem(population):
    bases, _ = population
    for source in bases["student_dev"][:48]:
        roles = source.metadata["role_to_symbol"]
        for polarity in ("positive", "negative"):
            atoms = [roles[f"{polarity}_intermediate_{level:02d}"] for level in (1, 2, 3)]
            maps = {
                (block, c): prep.name_mapping(source, block, c)
                for block in prep.BLOCKS
                for c in prep.CONDITIONS
            }
            for block in prep.BLOCKS:
                normal = maps[block, "preserve"]
                paired = maps[block, "break_pairs"]
                paths = maps[block, "break_paths"]
                both = maps[block, "break_both"]
                for level, atom in enumerate(atoms):
                    left, right = atom + "_L", atom + "_R"
                    assert normal[left][:-2] == normal[right][:-2]
                    assert paired[left] == normal[left]
                    assert paired[right] == normal[atoms[(level + 1) % 3] + "_R"]
                    assert paired[left][:-2] != paired[right][:-2]
                    if level == 1:
                        assert (paths[left], paths[right]) == (normal[right], normal[left])
                        assert (both[left], both[right]) == (paired[right], paired[left])
                    else:
                        assert (paths[left], paths[right]) == (normal[left], normal[right])
                        assert (both[left], both[right]) == (paired[left], paired[right])
                assert {normal[x + side] for x in atoms for side in ("_L", "_R")} == {
                    both[x + side] for x in atoms for side in ("_L", "_R")
                }
            for atom in atoms:
                assert maps[0, "preserve"][atom + "_L"] != maps[1, "preserve"][atom + "_L"]


def test_chain_controls_permute_names_but_do_not_invent_branch_labels(population):
    bases, _ = population
    for source in bases["student_dev"][48:]:
        for block in prep.BLOCKS:
            maps = [prep.name_mapping(source, block, c) for c in prep.CONDITIONS]
            assert len({tuple(sorted(m.items())) for m in maps}) == 4
            assert len({tuple(sorted(m.values())) for m in maps}) == 1
            assert all(not name.endswith(("_L", "_R")) for m in maps for name in m.values())


def test_rank_assignment_balances_each_condition_block_structure_and_label(population):
    _, views = population
    all_ordinals = []
    for rank in (0, 1):
        ordinals = prep.generation_ordinals(rank)
        assert len(ordinals) == 256 and ordinals == sorted(ordinals)
        all_ordinals.extend(ordinals)
        for block in prep.BLOCKS:
            for condition in prep.CONDITIONS:
                cell = [
                    views[i].example
                    for i in ordinals
                    if views[i].block == block and views[i].condition == condition
                ]
                assert Counter((x.metadata["structure"], x.label) for x in cell) == {
                    ("branch", 0): 12,
                    ("branch", 1): 12,
                    ("chain", 0): 4,
                    ("chain", 1): 4,
                }
    assert sorted(all_ordinals) == list(range(512))
    for pair_index in range(32):
        for within in range(8):
            assert prep.rank_for_ordinal(16 * pair_index + within) == prep.rank_for_ordinal(
                16 * pair_index + 8 + within
            )


@pytest.mark.parametrize("ordinal", [-1, 512, True, 1.5])
def test_invalid_global_ordinals_rejected(ordinal):
    with pytest.raises(prep.StudentNameCueProbeError):
        prep.rank_for_ordinal(ordinal)


@pytest.mark.parametrize("mutation", ["drop", "order", "target", "seed", "structure"])
def test_no_filter_or_repair_of_frozen_bases(population, mutation):
    values = copy.deepcopy(population[0])
    rows = values["student_dev"]
    if mutation == "drop":
        rows.pop()
    elif mutation == "order":
        rows.reverse()
    elif mutation == "target":
        rows[0].canonical_proof = []
    elif mutation == "seed":
        rows[0].metadata["pair_seed"] += 100
    else:
        rows[0].metadata["structure"] = "chain"
    with pytest.raises(ValueError):
        prep.validate_examples(values)


@pytest.fixture(scope="module")
def actual_encodings(population):
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("pinned tokenizer artifact absent; local acceptance requires actual feasibility evidence")
    spec = importlib.util.spec_from_file_location(
        "name_cue_pinned_tokenizer", ROOT / "tools/sdsc_student_prepare_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.load_tokenizer()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": prep.CHAT_TEMPLATE_SHA256,
        }
    }
    views = population[1]
    return [prep.encode_view(view, tokenizer, config) for view in views]


def test_actual_pinned_full_model_prefix_and_canonical_EOS_lengths_match(population, actual_encodings):
    evidence = prep.validate_encoded_views(actual_encodings, population[1])
    assert evidence["passed"] is True and evidence["scientific_acceptance"] is False
    assert evidence["maximum_prefix_tokens"] == 1350
    assert evidence["maximum_canonical_response_tokens"] == 212
    assert evidence["maximum_prefix_plus_cap"] == 1606
    assert (
        evidence["input_token_lengths_sha256"]
        == "6e5cbe1f76937b5fef40266d326719269e11ed9d92ea079e06bafbbf3a8bacaa"
    )


@pytest.mark.parametrize("mutation", ["length", "mask", "eos", "identity", "drop"])
def test_encoded_evidence_cannot_hide_a_length_or_mask_mismatch(population, actual_encodings, mutation):
    rows = copy.deepcopy(actual_encodings)
    if mutation == "length":
        rows[1]["input_ids"].insert(0, 17)
        rows[1]["labels"].insert(0, -100)
        rows[1]["prefix_length"] += 1
    elif mutation == "mask":
        rows[0]["labels"][0] = rows[0]["input_ids"][0]
    elif mutation == "eos":
        rows[0]["input_ids"][-1] = rows[0]["labels"][-1] = 17
    elif mutation == "identity":
        rows[0]["example_id"] = rows[1]["example_id"]
    else:
        rows.pop()
    with pytest.raises(prep.StudentNameCueProbeError):
        prep.validate_encoded_views(rows, population[1])


@pytest.mark.parametrize("dimension", ["semantic", "example_id", "pair_group_id", "pair_seed"])
def test_history_collision_rejected_in_each_dimension(population, dimension):
    inventory = prep.isolation_inventory(population[0])
    row = prep.previous.make_examples(prep.DIFFICULTY)["student_dev"][0]
    values = dict(
        semantic=prep.canonical_semantic_key(row),
        example_id=row.example_id,
        pair_group_id=row.pair_group_id,
        pair_seed=row.metadata["pair_seed"],
    )
    inventory[dimension].add(values[dimension])
    with pytest.raises(prep.StudentNameCueProbeError, match="overlaps " + dimension):
        prep.reject_overlap(inventory, row)


def test_actual_historical_producers_cover_every_old_population_and_view(population):
    inventory = prep.isolation_inventory(population[0])
    counts, view_counts = {}, {}
    for name, rows, is_view in prep.historical_populations([], {"teacher_fit": [], "teacher_dev": []}):
        observed = view_counts if is_view else counts
        observed[name] = len(rows)
        for row in rows:
            prep.reject_overlap(inventory, row.example if is_view else row)
    expected = prep.proposed_student_name_cue_probe_protocol()["data"]
    assert view_counts == expected["isolation_excluded_view_counts"]
    assert counts == {
        **expected["isolation_population_counts"],
        "original_family": 0,
        "teacher_fit": 0,
        "teacher_dev": 0,
    }


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in ("prereg/qwen3_v2.yaml", *prep.FROZEN_PROTOCOL_SHA256, *prep.FROZEN_HELPER_SHA256):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(prep, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(prep, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(prep, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged reviewed implementation\n")
    protocol = prep.proposed_student_name_cue_probe_protocol()
    path = tmp_path / prep.PROTOCOL_PATH
    path.write_text(json.dumps(protocol))
    git("add", ".")
    git("commit", "--quiet", "-m", "Proposed implementation")
    implementation = git("rev-parse", "HEAD")
    protocol["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent-test-reviewer",
        reviewed_at_utc="2026-10-05T03:00:00Z",
        rationale="Reviewed fixture implementation only.",
    )
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Independent review only")
    return tmp_path, git, implementation


def test_real_two_commit_acceptance_and_unrelated_later_commit(reviewed_repo):
    root, git, implementation = reviewed_repo
    first = prep.resolve_student_name_cue_probe_protocol(root)
    assert first.implementation_commit == implementation and first.acceptance_commit == git(
        "rev-parse", "HEAD"
    )
    (root / "unrelated.txt").write_text("not scientific\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated later note")
    assert prep.resolve_student_name_cue_probe_protocol(root).acceptance_commit == first.acceptance_commit


@pytest.mark.parametrize("mutation", ["science", "staged", "protocol", "frozen", "prereg"])
def test_real_accepted_lineage_rejects_changed_control_bytes(reviewed_repo, mutation):
    root, git, _ = reviewed_repo
    if mutation in {"science", "staged"}:
        (root / "kernel.py").write_text("changed scientific code\n")
        if mutation == "staged":
            git("add", "kernel.py")
    elif mutation == "protocol":
        path = root / prep.PROTOCOL_PATH
        data = json.loads(path.read_text())
        data["data"]["base_examples"] = 128
        path.write_text(json.dumps(data))
    elif mutation == "frozen":
        (root / next(iter(prep.FROZEN_HELPER_SHA256))).write_text("changed helper\n")
    else:
        (root / "prereg/qwen3_v2.yaml").write_text("changed base prereg\n")
    with pytest.raises(ValueError):
        prep.resolve_student_name_cue_probe_protocol(root)
