"""V5 twelve-checkpoint real-tokenizer replay, selection and corruption rejection."""

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


audit = module("sdsc_student_focus_audit")
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
    checkpoints = [dict(step=step, sha256=f"{i + 1:064x}") for i, step in enumerate(steps)]
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
            successes = (40, 100, 140, *([200] * (len(audit.STEPS) - 3)))[index] if fit else 8
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
                cohort="student_focus_dev" if fit else "student_focus_fit_preflight_first8",
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
        schema="quest-sdsc-student-focus-report-v1",
        mode="fit" if fit else "preflight",
        passed=selected is not None if fit else True,
        execution_complete=True,
        preparation_complete=fit,
        optimizer_steps=32 if fit else 4,
        fit_base_examples=256,
        fit_training_views=2048,
        consumed_fit_training_views=2048 if fit else 256,
        development_base_examples=256,
        development_views=1536,
        source_kind="symbolic_canonical_focus_preparation",
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


def test_all_18432_responses_six_views_and_real_tokenizer_replay(full_fixture):
    result = audit.replay(*full_fixture)
    assert result["prompts_replayed"] == 1536
    assert result["responses_replayed"] == 18432
    assert result["selected_checkpoint"]["step"] == 2
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
        "old_v3_report",
        "old_v3_cohort",
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
    if corruption == "old_v3_report":
        report["schema"] = "quest-sdsc-student-branch-report-v3"
    elif corruption == "old_v3_cohort":
        row["cohort"] = "student_branch_dev"
    elif corruption == "tokens":
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
        del records[32]
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
        schema="quest-sdsc-student-focus-plan-v1",
        task="qwen3-v2-student-focus-preparation-v5",
        mode="fit",
        run_id="release",
        intent_id="opaque",
        code_sha256="6" * 64,
        protocol=protocol,
        control_sha256={"tools/sdsc_student_focus_audit.py": audit.sha(Path(audit.__file__).read_bytes())},
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
        schema="quest-sdsc-student-focus-publication-v1",
        passed=True,
        stage_complete=True,
        preparation_complete=True,
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
        "old_v3_plan",
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
    if corruption == "old_v3_plan":
        plan["schema"] = "quest-sdsc-student-branch-plan-v3"
    elif corruption == "plan_sha":
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
        plan["control_sha256"]["tools/sdsc_student_focus_audit.py"] = "f" * 64
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


@pytest.fixture(scope="module")
def training_fixtures(actual_population):
    tokenizer = actual_population[0]
    lengths, token_audit = audit.training_population(tokenizer)
    result = {}
    for mode, steps in (("preflight", 4), ("fit", 32)):
        updates, consumed = [], 0
        for step, count in enumerate(token_audit["window_input_tokens"][:steps], 1):
            consumed += count
            ranks = []
            for rank in (0, 1):
                slots = list(range((step - 1) * 64 + rank, step * 64, 2))
                ranks.append(
                    dict(
                        rank=rank,
                        step=step,
                        global_slots=slots,
                        global_input_tokens=count,
                        optimizer_calls=[step - 1] * 7 + [step],
                        metric=dict(
                            step=float(step),
                            optimizer_updates=float(step),
                            model_facing_input_tokens_processed=float(consumed),
                            model_facing_input_tokens_this_update=count,
                            local_model_facing_input_tokens_this_update=sum(lengths[i] for i in slots),
                            token_budget_consumed=consumed,
                            token_budget=2_000_000,
                            token_budget_remaining=2_000_000 - consumed,
                        ),
                        token_budget=dict(
                            accepted_optimizer_updates=step,
                            budget=2_000_000,
                            consumed=consumed,
                            stop_reason=None,
                            unit="global_nonpadding_model_input_tokens_processed",
                        ),
                    )
                )
            updates.append(dict(step=step, ranks=ranks))
        report = dict(
            mode=mode,
            optimizer_steps=steps,
            consumed_fit_training_views=steps * 64,
            world_size=2,
            global_batch_size=64,
            training_input_tokens=consumed,
            data_audit=dict(token_envelope=copy.deepcopy(token_audit)),
        )
        files = {
            "prepare-token-audit.json": audit.canonical(token_audit),
            "prepare-updates.jsonl": b"\n".join(audit.canonical(row) for row in updates) + b"\n",
        }
        result[mode] = (report, files, lengths, token_audit)
    return result


@pytest.mark.parametrize("mode", ["preflight", "fit"])
def test_actual_training_order_and_tokens_are_independently_reconstructed(
    actual_population, training_fixtures, mode
):
    report, files, _, _ = training_fixtures[mode]
    result = audit.validate_training_evidence(report, files, actual_population[0])
    assert result["training_order_reconstructed"] is True
    assert result["training_rows_reconstructed"] == 2048
    assert result["consumed_training_rows"] == (256 if mode == "preflight" else 2048)
    assert result["optimizer_windows_replayed"] == (4 if mode == "preflight" else 32)
    assert result["training_input_tokens"] == report["training_input_tokens"]
    assert result["gpu_training_independently_recomputed"] is False
    assert set(result["checkpoint_cumulative_input_tokens"]) == (
        {"4"} if mode == "preflight" else {str(x) for x in audit.STEPS}
    )


@pytest.mark.parametrize("mode", ["preflight", "fit"])
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_mode_count",
        "missing_last",
        "rank_swap",
        "slot_swap",
        "local_token",
        "global_token",
        "budget_cursor",
        "accumulation",
        "float_count",
        "report_total",
        "last_step",
        "truncated_full_audit",
        "contradictory_report_envelope",
    ],
)
def test_training_mode_rank_and_token_corruption_rejected(
    actual_population, training_fixtures, monkeypatch, mode, mutation
):
    original, files, lengths, token_audit = training_fixtures[mode]
    monkeypatch.setattr(audit, "training_population", lambda tokenizer: (lengths, token_audit))
    report, files = copy.deepcopy(original), dict(files)
    updates = audit.lines(files["prepare-updates.jsonl"], report["optimizer_steps"])
    if mutation == "wrong_mode_count":
        report["optimizer_steps"] = 32 if mode == "preflight" else 4
    elif mutation == "missing_last":
        updates.pop()
    elif mutation == "rank_swap":
        updates[-1]["ranks"].reverse()
    elif mutation == "slot_swap":
        updates[0]["ranks"][0]["global_slots"][0] = 1
    elif mutation == "local_token":
        updates[-1]["ranks"][1]["metric"]["local_model_facing_input_tokens_this_update"] += 1
    elif mutation == "global_token":
        updates[-1]["ranks"][0]["global_input_tokens"] += 1
    elif mutation == "budget_cursor":
        updates[-1]["ranks"][1]["token_budget"]["consumed"] -= 64
    elif mutation == "accumulation":
        updates[-1]["ranks"][0]["optimizer_calls"][0] = report["optimizer_steps"]
    elif mutation == "float_count":
        report["consumed_fit_training_views"] = float(report["consumed_fit_training_views"])
    elif mutation == "report_total":
        report["training_input_tokens"] -= 1
    elif mutation == "last_step":
        updates[-1]["step"] -= 1
    elif mutation == "contradictory_report_envelope":
        windows = report["data_audit"]["token_envelope"]["window_input_tokens"]
        windows[0] += 1
        windows[1] -= 1
        assert sum(windows) == token_audit["total_input_tokens"]
    else:
        bad = copy.deepcopy(token_audit)
        bad["window_input_tokens"] = bad["window_input_tokens"][:4]
        files["prepare-token-audit.json"] = audit.canonical(bad)
    files["prepare-updates.jsonl"] = b"\n".join(audit.canonical(row) for row in updates) + b"\n"
    with pytest.raises(ValueError):
        audit.validate_training_evidence(report, files, actual_population[0])


@pytest.mark.parametrize(
    "mutation",
    ["old_v4", "old_batch", "arm", "diagnostic_complete", "missing_completion", "wrong_consumption"],
)
def test_prior_or_incomplete_worker_identity_rejected(full_fixture, mutation):
    report, prompts, records, tokenizer = full_fixture
    report = copy.deepcopy(report)
    if mutation == "old_v4":
        report["schema"] = "quest-sdsc-student-order-report-v4"
    elif mutation == "old_batch":
        report["schema"] = "quest-sdsc-student-batch-probe-report-v1"
    elif mutation == "arm":
        report["arm"] = "control"
    elif mutation == "diagnostic_complete":
        report["diagnostic_complete"] = True
    elif mutation == "missing_completion":
        report["execution_complete"] = False
    else:
        report["consumed_fit_training_views"] = 768
    with pytest.raises(ValueError):
        audit.replay(report, prompts, records, tokenizer)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "arm",
        "diagnostic_complete",
        "selected_checkpoint",
        "stage_integer",
        "passed_mismatch",
        "preparation_mismatch",
    ],
)
def test_flat_publication_rejects_stale_or_contradictory_fields(full_fixture, mutation):
    plan, pub, files, binding = publication_fixture(full_fixture)
    if mutation == "schema":
        pub["schema"] = "quest-sdsc-student-batch-probe-publication-v1"
    elif mutation == "arm":
        pub["arm"] = "control"
    elif mutation == "diagnostic_complete":
        pub["diagnostic_complete"] = True
    elif mutation == "selected_checkpoint":
        pub["selected_checkpoint"] = {}
    elif mutation == "stage_integer":
        pub["stage_complete"] = 1
    elif mutation == "passed_mismatch":
        pub["passed"] = False
    else:
        pub["preparation_complete"] = False
    raw = audit.canonical(pub)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding)


def test_complete_scientifically_failed_fit_retains_negative_publication_for_replay(full_fixture):
    plan, pub, files, binding = publication_fixture(full_fixture)
    report = audit.document(files["prepare-report.json"])
    report.update(passed=False, selected_checkpoint=None, preparation_complete=True, execution_complete=True)
    files["prepare-report.json"] = audit.canonical(report)
    pub.update(passed=False, stage_complete=False, preparation_complete=False)
    pub["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    raw = audit.canonical(pub)
    checked, _ = audit.validate_publication(plan, raw, audit.sha(raw), files, binding)
    assert checked["passed"] is False and checked["preparation_complete"] is True
    pub.update(passed=True, stage_complete=True, preparation_complete=True)
    raw = audit.canonical(pub)
    with pytest.raises(ValueError, match="contradicts"):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding)


def test_model_facing_tokens_do_not_trust_producer_encoder(actual_population, monkeypatch):
    from posttrain_circuits.experiments.protocols import student_focus_preparation as p

    original = p.encode_view

    def changed(*args, **kwargs):
        row = copy.deepcopy(original(*args, **kwargs))
        row["input_ids"][-2] += 1
        row["labels"][-2] += 1
        return row

    monkeypatch.setattr(p, "encode_view", changed)
    with pytest.raises(ValueError, match="independent model-facing"):
        audit.training_population(actual_population[0])


def test_pair_order_does_not_trust_producer_sorting(actual_population, monkeypatch):
    from posttrain_circuits.experiments.protocols import student_focus_preparation as p

    original = p.fit_views

    def changed(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0], rows[1] = rows[1], rows[0]
        return rows

    monkeypatch.setattr(p, "fit_views", changed)
    with pytest.raises(ValueError, match="complete-pair row order"):
        audit.training_population(actual_population[0])


@pytest.fixture(scope="module")
def actual_dataset():
    from posttrain_circuits.experiments.protocols import student_focus_preparation as protocol

    return protocol.dataset_manifest(protocol.make_examples(protocol.DIFFICULTY))


@pytest.mark.parametrize("mutation", [None, "prior_v4", "changed_order_rehashed"])
def test_dataset_publication_binds_fresh_population_even_when_rehashed(actual_dataset, mutation):
    value = copy.deepcopy(actual_dataset)
    if mutation == "prior_v4":
        from posttrain_circuits.experiments.protocols import student_order_preparation as prior

        value = prior.dataset_manifest(prior.make_examples(prior.DIFFICULTY))
    elif mutation == "changed_order_rehashed":
        ids = value["student_fit"]["ordered_ids"]
        ids[0], ids[1] = ids[1], ids[0]
    raw = audit.canonical(value)
    report = dict(dataset_sha256=audit.sha(raw))
    files = {"prepare-dataset-manifest.json": raw}
    if mutation is None:
        assert audit.validate_dataset(report, files) == report["dataset_sha256"]
    else:
        with pytest.raises(ValueError, match="published training population"):
            audit.validate_dataset(report, files)


@pytest.mark.parametrize("mode", ["preflight", "fit"])
@pytest.mark.parametrize(
    "mutation", [None, "old_key", "contradictory_report", "missing_history", "formal_claim", "bad_digest"]
)
def test_actual_isolation_producer_receipt_cross_binding(mode, mutation):
    path = ROOT / ".sdsc/diagnostics/student-focus-v5/focus-cpu-population-isolation.json"
    if not path.exists():
        pytest.skip("bounded real isolation producer evidence is not present")
    isolation = audit.document(path.read_bytes())["audit"]
    digest = isolation["dataset_manifest_sha256"]
    if mutation == "old_key":
        isolation["preparation_manifest_sha256"] = isolation.pop("dataset_manifest_sha256")
    elif mutation == "missing_history":
        isolation["excluded_view_counts"].pop("batch_probe_treatment_student_fit_views")
    elif mutation == "formal_claim":
        isolation["formal_generated_responses_or_scores_read"] = True
    elif mutation == "bad_digest":
        isolation["population_stream_sha256"]["original_family"] = "invalid"
    report = dict(mode=mode, data_audit=dict(passed=True, isolation=copy.deepcopy(isolation)))
    if mutation == "contradictory_report":
        report["data_audit"]["isolation"]["population_stream_sha256"]["original_family"] = "0" * 64
    files = {"prepare-data-isolation.json": audit.canonical(isolation)}
    if mutation is None:
        result = audit.validate_isolation(report, files, digest)
        assert result["producer_isolation_consistency_verified"] is True
        assert result["historical_isolation_independently_recomputed"] is False
    else:
        with pytest.raises(ValueError):
            audit.validate_isolation(report, files, digest)
