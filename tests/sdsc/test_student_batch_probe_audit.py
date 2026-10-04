"""Six-checkpoint two-arm raw replay and rejection of mixed or promoted evidence."""

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


audit = module("sdsc_student_batch_probe_audit")
producer = module("sdsc_student_quality_probe")


@pytest.fixture(scope="module")
def actual_population():
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    tokenizer = audit.load_tokenizer()
    return tokenizer, audit.population(tokenizer, "probe")


def make_fixture(actual_population, arm="control", all_failed=False):
    tokenizer, (prompts, examples, dev_sha) = actual_population
    task, development, records = ProofGraphTask(), [], {}
    steps = audit.STEPS
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
            successes = (40, 60, 70, 100, 120, 128)[index]
            if index == 0 and prompt["view"] != "identity":
                successes = 20
            good = not all_failed and ordinal // 6 < successes
            text = task.canonical_target(example) if good else ""
            # One legitimate incomplete proof distinguishes answer_correct from
            # full proof reward; the audit must retain the original definitions.
            if good and ordinal == 0:
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
            assigned_rank = (ordinal // 6) % 2
            row.update(
                step=step,
                ordinal=ordinal,
                rank=assigned_rank,
                checkpoint_sha256=checkpoint["sha256"],
                cohort="student_batch_probe_dev",
                arm=arm,
                source_example_id=prompt["source_example_id"],
                view=prompt["view"],
                parsed_trace=asdict(task.parse_response(text)),
            )
            ranks[assigned_rank].append(row)
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
    report = dict(
        schema="quest-sdsc-student-batch-probe-report-v1",
        mode="probe",
        arm=arm,
        passed=True,
        execution_complete=True,
        diagnostic_complete=True,
        preparation_complete=False,
        optimizer_steps=32,
        fit_training_views=2048,
        consumed_fit_training_views=2048,
        development_base_examples=128,
        development_views=768,
        checkpoints=checkpoints,
        development=development,
        selected_checkpoint=None,
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
    return report, prompts, records, tokenizer


@pytest.fixture(scope="module")
def full_fixture(actual_population):
    return make_fixture(actual_population)


def test_all_4608_responses_six_views_and_real_tokenizer_replay(full_fixture):
    result = audit.replay(*full_fixture)
    assert result["prompts_replayed"] == 768
    assert result["responses_replayed"] == 4608
    assert result["selected_checkpoint"] is None
    assert result["arm"] == "control"
    assert result["diagnostic_complete"] is True and result["preparation_complete"] is False
    first = result["development"][0]
    assert first["answer_correct"] == first["proof_correct"] + 1
    assert all(result[k] is False for k in audit.FLAGS)
    assert result["gpu_numerics_independently_recomputed"] is False


def test_treatment_replays_same_population_but_binds_separate_arm(actual_population):
    result = audit.replay(*make_fixture(actual_population, arm="treatment"))
    assert result["mode"] == "probe" and result["arm"] == "treatment"
    assert result["responses_replayed"] == 4608
    assert result["selected_checkpoint"] is None


def test_zero_correct_diagnostic_is_complete_without_model_acceptance(actual_population, monkeypatch):
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
        "wrong_arm",
        "unknown_arm",
        "preparation",
        "incomplete",
        "wrong_count",
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
    if corruption == "wrong_arm":
        row["arm"] = "treatment"
    elif corruption == "unknown_arm":
        report["arm"] = "adaptive"
    elif corruption == "preparation":
        report["preparation_complete"] = True
    elif corruption == "incomplete":
        report["diagnostic_complete"] = False
    elif corruption == "wrong_count":
        report["consumed_fit_training_views"] = 768
    elif corruption == "old_v3_report":
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
        del records[12]
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
        schema="quest-sdsc-student-batch-probe-plan-v1",
        task="qwen3-v2-student-batch-probe-v1",
        mode="probe",
        arm="control",
        run_id="release",
        intent_id="opaque",
        code_sha256="6" * 64,
        protocol=protocol,
        control_sha256={
            "tools/sdsc_student_batch_probe_audit.py": audit.sha(Path(audit.__file__).read_bytes())
        },
        provenance={"head": binding.acceptance_commit},
        **dict.fromkeys(audit.FLAGS, False),
    )
    report.update(
        mode="probe",
        arm="control",
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
        schema="quest-sdsc-student-batch-probe-publication-v1",
        passed=True,
        stage_complete=True,
        diagnostic_complete=True,
        preparation_complete=False,
        selected_checkpoint=None,
        task=plan["task"],
        mode="probe",
        arm="control",
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
        "arm",
        "report_arm",
        "publication_schema",
        "publication_incomplete",
        "publication_candidate",
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
    if corruption == "publication_schema":
        pub["schema"] = "quest-sdsc-student-order-publication-v4"
    elif corruption == "publication_incomplete":
        pub["diagnostic_complete"] = False
    elif corruption == "publication_candidate":
        pub["selected_checkpoint"] = {"step": 7}
    elif corruption == "arm":
        pub["arm"] = "treatment"
    elif corruption == "old_v3_plan":
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
    elif corruption in ("report_job", "report_parent", "report_arm"):
        report = audit.document(files["prepare-report.json"])
        report[
            {"report_job": "job_id", "report_parent": "initial_checkpoint_sha256", "report_arm": "arm"}[
                corruption
            ]
        ] = "wrong"
        files["prepare-report.json"] = audit.canonical(report)
        pub["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    elif corruption == "overclaim":
        pub["formal_initial_accepted"] = True
    elif corruption == "auditor":
        plan["control_sha256"]["tools/sdsc_student_batch_probe_audit.py"] = "f" * 64
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
def arm_datasets():
    from posttrain_circuits.experiments.protocols import student_batch_probe as protocol

    splits = protocol.make_examples(protocol.DIFFICULTY)
    return {arm: protocol.dataset_manifest(splits, arm=arm) for arm in ("control", "treatment")}


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_actual_arm_manifest_binds_training_population(arm_datasets, arm):
    raw = audit.canonical(arm_datasets[arm])
    report = dict(arm=arm, dataset_sha256=audit.sha(raw))
    files = {"prepare-dataset-manifest.json": raw}
    assert audit.validate_arm_dataset(report, files) == report["dataset_sha256"]
    opposite = "treatment" if arm == "control" else "control"
    files["prepare-dataset-manifest.json"] = audit.canonical(arm_datasets[opposite])
    with pytest.raises(ValueError, match="arm population"):
        audit.validate_arm_dataset(report, files)
    files["prepare-dataset-manifest.json"] = raw
    report["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="arm digest"):
        audit.validate_arm_dataset(report, files)


@pytest.mark.parametrize("surface", ["plan", "publication", "report", "cohort"])
def test_completed_previous_anchor_probe_cannot_be_relabelled_as_batch(full_fixture, surface):
    # A completed predecessor is valid evidence of a different experiment.
    # Rehashing that predecessor's identity must not make it this diagnostic.
    if surface in ("report", "cohort"):
        source_report, prompts, source_records, tokenizer = full_fixture
        report = copy.deepcopy(source_report)
        records = {step: [list(rank) for rank in ranks] for step, ranks in source_records.items()}
        records[4][0][0] = copy.deepcopy(records[4][0][0])
        if surface == "report":
            report["schema"] = "quest-sdsc-student-anchor-probe-report-v1"
        else:
            records[4][0][0]["cohort"] = "student_anchor_probe_dev"
        with pytest.raises(ValueError):
            audit.replay(report, prompts, records, tokenizer)
        return
    plan, pub, files, binding = publication_fixture(full_fixture)
    if surface == "plan":
        plan["schema"] = "quest-sdsc-student-anchor-probe-plan-v1"
        plan["task"] = "qwen3-v2-student-anchor-probe-v1"
        pub["task"] = plan["task"]
        pub["plan_sha256"] = audit.sha(audit.canonical(plan))
    else:
        pub["schema"] = "quest-sdsc-student-anchor-probe-publication-v1"
    raw = audit.canonical(pub)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding)


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_previous_anchor_dataset_rejected_even_with_self_consistent_digest(arm):
    from posttrain_circuits.experiments.protocols import student_anchor_probe as historical

    splits = historical.make_examples(historical.DIFFICULTY)
    raw = audit.canonical(historical.dataset_manifest(splits, arm=arm))
    report = dict(arm=arm, dataset_sha256=audit.sha(raw))
    with pytest.raises(ValueError, match="arm population"):
        audit.validate_arm_dataset(report, {"prepare-dataset-manifest.json": raw})


@pytest.mark.parametrize("corruption", ["old_prefix", "missing_step32", "duplicate_step32"])
def test_full_run_cannot_be_replaced_by_old_twelve_step_diagnostic(full_fixture, corruption):
    report, prompts, source_records, tokenizer = full_fixture
    report = copy.deepcopy(report)
    records = {step: [list(rank) for rank in ranks] for step, ranks in source_records.items()}
    if corruption == "old_prefix":
        report["optimizer_steps"] = 12
        report["consumed_fit_training_views"] = 768
    elif corruption == "missing_step32":
        records.pop(32)
        report["checkpoints"].pop()
        report["development"].pop()
    else:
        report["checkpoints"][-1]["sha256"] = report["checkpoints"][-2]["sha256"]
    with pytest.raises(ValueError):
        audit.replay(report, prompts, records, tokenizer)


@pytest.fixture(scope="module")
def training_fixtures(actual_population):
    tokenizer = actual_population[0]
    result = {}
    for arm in ("control", "treatment"):
        lengths, token_audit = audit.training_population(tokenizer, arm)
        updates, consumed = [], 0
        for step, count in enumerate(token_audit["window_input_tokens"], 1):
            consumed += count
            ranks = []
            for rank in (0, 1):
                slots = list(range((step - 1) * 64 + rank, step * 64, 2))
                metric = dict(
                    step=float(step),
                    optimizer_updates=float(step),
                    model_facing_input_tokens_processed=float(consumed),
                    model_facing_input_tokens_this_update=count,
                    local_model_facing_input_tokens_this_update=sum(lengths[i] for i in slots),
                    token_budget_consumed=consumed,
                    token_budget=2_000_000,
                    token_budget_remaining=2_000_000 - consumed,
                )
                ranks.append(
                    dict(
                        rank=rank,
                        step=step,
                        global_slots=slots,
                        global_input_tokens=count,
                        optimizer_calls=[step - 1] * 7 + [step],
                        metric=metric,
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
            arm=arm,
            optimizer_steps=32,
            consumed_fit_training_views=2048,
            world_size=2,
            global_batch_size=64,
            training_input_tokens=consumed,
        )
        files = {
            "prepare-token-audit.json": audit.canonical(token_audit),
            "prepare-updates.jsonl": b"\n".join(audit.canonical(row) for row in updates) + b"\n",
        }
        result[arm] = (report, files, lengths, token_audit)
    return result


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_full_training_order_and_cumulative_tokens_independently_reconstructed(
    actual_population, training_fixtures, arm
):
    report, files, _, _ = training_fixtures[arm]
    result = audit.validate_training_evidence(report, files, actual_population[0])
    assert result["optimizer_windows_replayed"] == 32
    assert result["training_rows_reconstructed"] == 2048
    assert result["identical_scientific_row_multiset"] is True
    assert result["checkpoint_cumulative_input_tokens"]["32"] == report["training_input_tokens"]
    assert result["gpu_training_independently_recomputed"] is False
    other = training_fixtures["treatment" if arm == "control" else "control"][0]
    assert result["training_input_tokens"] == other["training_input_tokens"]


@pytest.mark.parametrize(
    "corruption",
    [
        "old_prefix",
        "missing_last",
        "rank_swap",
        "slot_swap",
        "local_token",
        "window_token",
        "budget_cursor",
        "accumulation",
        "other_arm_token_windows",
        "report_total",
        "late_update_order",
    ],
)
def test_training_order_and_token_corruption_rejected(
    actual_population, training_fixtures, monkeypatch, corruption
):
    report, files, lengths, token_audit = training_fixtures["treatment"]
    monkeypatch.setattr(audit, "training_population", lambda tokenizer, arm: (lengths, token_audit))
    report, files = copy.deepcopy(report), dict(files)
    updates = audit.lines(files["prepare-updates.jsonl"], 32)
    if corruption == "old_prefix":
        updates = updates[:12]
        report["optimizer_steps"] = 12
        report["consumed_fit_training_views"] = 768
    elif corruption == "missing_last":
        updates.pop()
    elif corruption == "rank_swap":
        updates[-1]["ranks"].reverse()
    elif corruption == "slot_swap":
        updates[0]["ranks"][0]["global_slots"][0] = 1
    elif corruption == "local_token":
        updates[-1]["ranks"][1]["metric"]["local_model_facing_input_tokens_this_update"] += 1
    elif corruption == "window_token":
        updates[-1]["ranks"][0]["global_input_tokens"] += 1
    elif corruption == "budget_cursor":
        updates[-1]["ranks"][1]["token_budget"]["consumed"] -= 64
    elif corruption == "accumulation":
        updates[-1]["ranks"][0]["optimizer_calls"][0] = 32
    elif corruption == "other_arm_token_windows":
        files["prepare-token-audit.json"] = training_fixtures["control"][1]["prepare-token-audit.json"]
        assert files["prepare-token-audit.json"] != audit.canonical(token_audit)
    elif corruption == "report_total":
        report["training_input_tokens"] -= 1
    elif corruption == "late_update_order":
        updates[-1]["step"] = 31
    files["prepare-updates.jsonl"] = b"\n".join(audit.canonical(row) for row in updates) + b"\n"
    with pytest.raises(ValueError):
        audit.validate_training_evidence(report, files, actual_population[0])
