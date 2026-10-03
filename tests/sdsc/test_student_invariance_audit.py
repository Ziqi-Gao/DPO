"""Full real-tokenizer replay, semantic selection boundaries and evidence corruption."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_step

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("sdsc_student_invariance_audit")
producer = module("sdsc_student_quality_probe")


@pytest.fixture(scope="module")
def actual_population():
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    tokenizer = audit.load_tokenizer()
    return tokenizer, audit.population(tokenizer, "fit"), audit.population(tokenizer, "preflight")


def make_fixture(actual_population, fit=True, all_failed=False):
    tokenizer, fit_population, preflight_population = actual_population
    prompts, examples, dev_sha = fit_population if fit else preflight_population
    task, development, records = ProofGraphTask(), [], {}
    steps = audit.STEPS if fit else (4,)
    checkpoints = [dict(step=step, sha256=str(i + 1) * 64) for i, step in enumerate(steps)]
    for index, checkpoint in enumerate(checkpoints):
        step = checkpoint["step"]
        ranks = [[], []]
        groups = {
            name: dict(num_examples=0, answer_correct=0, proof_correct=0, format_valid=0)
            for name in audit.VIEWS
        }
        strata = {
            name: dict(num_examples=0, answer_correct=0, proof_correct=0, format_valid=0)
            for name in audit.STRUCTURES
        }
        for ordinal, (prompt, example) in enumerate(zip(prompts, examples, strict=True)):
            successes = (40, 100, 140, 200, 220, 240)[index] if fit else 8
            if fit and index == 0 and prompt["view"] != "identity":
                successes = 20
            good = not all_failed and (ordinal // 6 if fit else ordinal) < successes
            text = task.canonical_target(example) if good else ""
            # One legitimate incomplete proof distinguishes answer_correct from
            # full proof reward; the audit must retain the original definitions.
            if good and fit and ordinal == 0:
                body = "\n".join(render_step(x) for x in example.canonical_proof[:-1])
                text = f"<proof>\n{body}\n</proof>\n<answer>{example.label}</answer>"
            tokens = [*tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id]
            row = producer.response_record(
                example,
                tokens,
                tokenizer,
                prompt_ids=prompt["prompt_ids"],
                prompt_text=prompt["prompt_text"],
                cap=256,
            )
            assigned_rank = (ordinal // 6) % 2 if fit else ordinal % 2
            row.update(
                step=step,
                ordinal=ordinal,
                rank=assigned_rank,
                checkpoint_sha256=checkpoint["sha256"],
                cohort="student_invariance_dev" if fit else "student_invariance_fit_preflight_first8",
                source_example_id=prompt["source_example_id"],
                view=prompt["view"],
                parsed_trace=asdict(task.parse_response(text)),
            )
            ranks[assigned_rank].append(row)
            if fit:
                result = row["verification"]
                targets = [groups[prompt["view"]]]
                if prompt["view"] == "identity":
                    targets.append(strata[example.metadata["structure"]])
                for target in targets:
                    target["num_examples"] += 1
                    target["answer_correct"] += int(result["answer_correct"])
                    target["proof_correct"] += int(result["reward"] == 1.0)
                    target["format_valid"] += int(result["parse_valid"])
        records[step] = ranks
        if fit:
            development.append(
                dict(
                    step=step,
                    checkpoint_sha256=checkpoint["sha256"],
                    examples_sha256=dev_sha,
                    **groups["identity"],
                    per_view_counts=groups,
                    structure_counts=strata,
                )
            )
    selected = development[1] if fit and not all_failed else None
    report = dict(
        schema="quest-sdsc-student-invariance-report-v2",
        mode="fit" if fit else "preflight",
        passed=selected is not None if fit else True,
        checkpoints=checkpoints,
        development=development,
        selected_checkpoint=selected,
        development_generation=dict(
            max_new_tokens=256,
            max_model_input_length=2454,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
            truncation=False,
        ),
        **dict.fromkeys(audit.FLAGS, False),
    )
    return report, prompts if fit else [], records, tokenizer


@pytest.fixture(scope="module")
def full_fixture(actual_population):
    return make_fixture(actual_population)


def test_all_9216_responses_six_views_and_real_tokenizer_replay(full_fixture):
    result = audit.replay(*full_fixture)
    assert result["prompts_replayed"] == 1536
    assert result["responses_replayed"] == 9216
    assert result["selected_checkpoint"]["step"] == 8
    first = result["development"][0]
    assert first["answer_correct"] == first["proof_correct"] + 1
    assert all(result[k] is False for k in audit.FLAGS)
    assert result["gpu_numerics_independently_recomputed"] is False


def test_preflight_uses_only_first_eight_training_views(actual_population):
    result = audit.replay(*make_fixture(actual_population, fit=False))
    assert result["mode"] == "preflight"
    assert result["responses_replayed"] == result["prompts_replayed"] == 8
    assert result["selected_checkpoint"] is None and result["development"] == []


def test_complete_failed_fit_remains_a_successful_nonaccepting_raw_audit(actual_population, monkeypatch):
    monkeypatch.setattr(audit, "population", lambda tokenizer, mode: actual_population[1])
    result = audit.replay(*make_fixture(actual_population, all_failed=True))
    assert result["raw_replay_passed"] is True
    assert result["selected_checkpoint"] is None
    assert all(result[k] is False for k in audit.FLAGS)


@pytest.mark.parametrize(
    "corruption",
    [
        "tokens",
        "verifier",
        "parser",
        "rank",
        "missing_rank_row",
        "missing_step",
        "order",
        "view",
        "source",
        "checkpoint",
        "prompt",
        "population",
        "prefix",
        "later_selection",
        "structure",
        "group",
        "overclaim",
        "eos",
        "length",
        "tag",
        "hash",
        "summary",
        "generation_sampling",
        "generation_context",
        "generation_autocast",
    ],
)
def test_corrupted_evidence_rejected(full_fixture, actual_population, monkeypatch, corruption):
    monkeypatch.setattr(audit, "population", lambda tokenizer, mode: actual_population[1])
    report, prompts, original, tokenizer = full_fixture
    report = copy.deepcopy(report)
    prompts = list(prompts)
    records = {step: [list(rank) for rank in ranks] for step, ranks in original.items()}
    row = copy.deepcopy(records[4][0][0])
    records[4][0][0] = row
    if corruption == "tokens":
        row["response_ids"][0] = 0
    elif corruption == "verifier":
        row["verification"]["answer_correct"] = False
    elif corruption == "parser":
        row["parsed_trace"]["answer"] = 1 - row["parsed_trace"]["answer"]
    elif corruption == "rank":
        row["rank"] = 1
    elif corruption == "missing_rank_row":
        records[4][1].pop()
    elif corruption == "missing_step":
        del records[128]
    elif corruption == "order":
        records[4][0][0], records[4][0][1] = records[4][0][1], row
    elif corruption == "view":
        row["view"] = "rule_order_permutation"
    elif corruption == "source":
        row["source_example_id"] = "different"
    elif corruption == "checkpoint":
        row["checkpoint_sha256"] = "f" * 64
    elif corruption == "prompt":
        prompts[0] = {**prompts[0], "prompt_text": "wrong"}
    elif corruption == "population":
        prompts[0] = copy.deepcopy(prompts[0])
        prompts[0]["example"]["label"] ^= 1
    elif corruption == "prefix":
        row["prompt_ids"][0] = 0
    elif corruption == "later_selection":
        report["selected_checkpoint"] = report["development"][2]
    elif corruption == "structure":
        report["development"][0]["structure_counts"]["branch"]["proof_correct"] += 1
    elif corruption == "group":
        report["development"][0]["per_view_counts"]["entity_symbol_renaming"]["proof_correct"] += 1
    elif corruption == "overclaim":
        report["student_accepted"] = True
    elif corruption == "eos":
        row["response_ids"].insert(0, tokenizer.eos_token_id)
    elif corruption == "length":
        row["response_tokens"] += 1
    elif corruption == "tag":
        row["diagnostic_answer_tag_correct"] = False
    elif corruption == "hash":
        row["prompt_token_sha256"] = "f" * 64
    elif corruption == "generation_sampling":
        report["development_generation"]["do_sample"] = True
    elif corruption == "generation_context":
        report["development_generation"]["max_model_input_length"] = 2455
    elif corruption == "generation_autocast":
        report["development_generation"]["explicit_autocast"] = True
    elif corruption == "summary":
        report["development"][0]["answer_correct"] += 1
    with pytest.raises(ValueError):
        audit.replay(report, prompts, records, tokenizer)


def selection_row(iid=50, transformed=None, answers=None, branch=3):
    transformed = [50] * 5 if transformed is None else transformed

    def counts(n, proof, answer=None):
        return dict(
            num_examples=n,
            answer_correct=proof if answer is None else answer,
            proof_correct=proof,
            format_valid=256,
        )

    groups = {"identity": counts(256, iid, answers)}
    groups.update(
        {name: counts(256, value) for name, value in zip(audit.VIEWS[1:], transformed, strict=True)}
    )
    return dict(
        per_view_counts=groups,
        structure_counts={name: counts(30, branch if name == "branch" else 5) for name in audit.STRUCTURES},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(answers=25),
        dict(answers=154),
        dict(iid=25, answers=26),
        dict(iid=26, transformed=[20, 20, 20, 21, 21]),
        dict(iid=26, transformed=[12, 30, 30, 30, 30]),
        dict(iid=50, transformed=[37, 54, 54, 54, 54]),
        dict(iid=50, transformed=[37] * 5),
        dict(branch=2),
    ],
)
def test_original_floor_and_stronger_prospective_boundaries(kwargs):
    assert audit.eligible(selection_row(**kwargs)) is False


def test_negative_gap_and_exact_discrete_boundary_preserved():
    assert audit.eligible(selection_row(iid=26, transformed=[100] * 5)) is True
    assert audit.eligible(selection_row(iid=50, transformed=[38] * 5)) is True
    assert audit.eligible(selection_row(iid=50, transformed=[37] * 5)) is False


def publication_fixture(full_fixture):
    report = copy.deepcopy(full_fixture[0])
    binding = SimpleNamespace(
        protocol_sha256="1" * 64,
        artifact_sha256="2" * 64,
        implementation_commit="3" * 40,
        acceptance_commit="4" * 40,
        science_file_sha256={"src/science.py": "5" * 64},
    )
    protocol = {key: getattr(binding, key) for key in vars(binding)}
    protocol["head"] = binding.acceptance_commit
    plan = dict(
        schema="quest-sdsc-student-invariance-plan-v2",
        task="qwen3-v2-student-invariance-v2",
        mode="fit",
        run_id="release",
        intent_id="opaque",
        code_sha256="6" * 64,
        protocol=protocol,
        control_sha256={
            "tools/sdsc_student_invariance_audit.py": audit.sha(Path(audit.__file__).read_bytes())
        },
        provenance={"head": binding.acceptance_commit},
        **dict.fromkeys(audit.FLAGS, False),
    )
    report.update(
        mode="fit",
        job_id="123",
        run_id=plan["run_id"],
        plan_sha256=audit.sha(audit.canonical(plan)),
        source_code_sha256=plan["code_sha256"],
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        initial_checkpoint_sha256=audit.INITIAL_SHA,
    )
    large = []
    for checkpoint in report["checkpoints"]:
        checkpoint.update(path=f"checkpoints/step-{checkpoint['step']:08d}.pt", size=100)
        large.append({key: checkpoint[key] for key in ("path", "size", "sha256")})
    files = {"prepare-report.json": audit.canonical(report)}
    publication = dict(
        task=plan["task"],
        mode="fit",
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=audit.sha(audit.canonical(plan)),
        code_sha256=plan["code_sha256"],
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=large,
        files=[dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()],
        **dict.fromkeys(audit.FLAGS, False),
    )
    return plan, publication, files, binding


def test_valid_persistent_checkpoint_lineage_and_report_binding(full_fixture):
    plan, pub, files, binding = publication_fixture(full_fixture)
    raw = audit.canonical(pub)
    report, published = audit.validate_publication(plan, raw, audit.sha(raw), files, binding)
    assert report["job_id"] == published["job_id"] == "123"


@pytest.mark.parametrize(
    "corruption",
    [
        "receipt",
        "plan_sha",
        "task",
        "mode",
        "intent",
        "source",
        "protocol",
        "science",
        "review",
        "missing_checkpoint",
        "checkpoint_sha",
        "path",
        "duplicate",
        "report_job",
        "report_parent",
        "overclaim",
        "auditor",
        "file_hash",
        "unbound_file",
        "size",
        "readback",
    ],
)
def test_publication_binding_corruptions_rejected(full_fixture, corruption):
    plan, pub, files, binding = publication_fixture(full_fixture)
    if corruption == "plan_sha":
        pub["plan_sha256"] = "f" * 64
    elif corruption == "task":
        pub["task"] = "other"
    elif corruption == "mode":
        pub["mode"] = "preflight"
    elif corruption == "intent":
        pub["intent_id"] = "other"
    elif corruption == "source":
        pub["code_sha256"] = "f" * 64
    elif corruption == "protocol":
        binding.protocol_sha256 = "f" * 64
    elif corruption == "science":
        binding.science_file_sha256 = {}
    elif corruption == "review":
        binding.acceptance_commit = "f" * 40
    elif corruption == "missing_checkpoint":
        pub["large_files"].pop()
    elif corruption == "checkpoint_sha":
        pub["large_files"][0]["sha256"] = "f" * 64
    elif corruption == "path":
        pub["large_files"][0]["path"] = "checkpoints/../bad.pt"
    elif corruption == "duplicate":
        pub["files"].append(pub["files"][0])
    elif corruption in ("report_job", "report_parent"):
        report = audit.document(files["prepare-report.json"])
        report["job_id" if corruption == "report_job" else "initial_checkpoint_sha256"] = "wrong"
        files["prepare-report.json"] = audit.canonical(report)
        pub["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    elif corruption == "overclaim":
        pub["formal_initial_accepted"] = True
    elif corruption == "auditor":
        plan["control_sha256"]["tools/sdsc_student_invariance_audit.py"] = "f" * 64
    elif corruption == "file_hash":
        pub["files"][0]["sha256"] = "f" * 64
    elif corruption == "unbound_file":
        files["unbound.json"] = b"{}"
    elif corruption == "size":
        pub["large_files"][0]["size"] = True
    elif corruption == "readback":
        pub["large_files_read_back_verified"] = False
    raw = audit.canonical(pub)
    receipt = "f" * 64 if corruption == "receipt" else audit.sha(raw)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, receipt, files, binding)


def test_strict_json_and_filesystem(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        audit.document(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):
        audit.document(b'{"x":NaN}')
    path = tmp_path / "valid.json"
    path.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        audit.read(link)
