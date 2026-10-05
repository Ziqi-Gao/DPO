"""Independent full-population replay and corruption rejection, with no GPU."""

import base64
import copy
import importlib.util
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.artifacts.compatibility import scientific_compatibility_fields
from posttrain_circuits.datasets.proofgraph.anti_shortcut import (
    TRANSFORMATIONS,
    evaluate_anti_shortcut_suite,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.metrics import aggregate_verification
from posttrain_circuits.datasets.proofgraph.rendering import render_step

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qualify_audit", ROOT / "tools/sdsc_student_focus_qualify_audit.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def completion_fixture(value):
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    plan = dict(
        schema="quest-sdsc-student-focus-plan-v1",
        task="qwen3-v2-student-focus-preparation-v5",
        mode="fit",
        intent_id=value["fit_intent"],
        run_id="synthetic-fit",
        code_sha256="c" * 64,
        control_sha256=dict(q.FROZEN_EXECUTION_SHA256),
        provenance=dict(head=value["fit_source_head"]),
        protocol=dict(
            protocol_sha256=q.PARENT_PROTOCOL_CORE_SHA256,
            artifact_sha256=q.PARENT_PROTOCOL_SHA256,
            implementation_commit=q.PARENT_IMPLEMENTATION_COMMIT,
            acceptance_commit=q.PARENT_HEAD,
            head=value["fit_source_head"],
        ),
        **dict.fromkeys(q.FLAGS, False),
    )
    value["fit_plan_sha256"] = audit.sha(audit.canonical(plan))
    pub = dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=plan["task"],
        mode="fit",
        intent_id=value["fit_intent"],
        run_id=plan["run_id"],
        code_sha256=plan["code_sha256"],
        job_id=value["fit_job_id"],
        plan_sha256=value["fit_plan_sha256"],
        passed=True,
        stage_complete=True,
        preparation_complete=True,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=[
            dict(path="prepare-report.json", size=1, sha256=value["fit_report_sha256"]),
            dict(path="prepare-data-isolation.json", size=1, sha256="e" * 64),
        ],
        large_files=[
            dict(
                path=value["checkpoint_path"],
                size=value["checkpoint_size"],
                sha256=value["checkpoint_sha256"],
            )
        ],
        **dict.fromkeys(q.FLAGS, False),
    )
    value["fit_publication_sha256"] = audit.sha(audit.canonical(pub))
    status = dict(
        accounting_complete=True,
        success=True,
        stage_complete=True,
        preparation_complete=True,
        publication_verified=True,
        artifact_hashes_verified=True,
        state="COMPLETED",
        job_id=value["fit_job_id"],
        result_sha256=value["fit_report_sha256"],
        queue=dict(returncode=0, stdout=""),
        accounting=dict(returncode=0),
        publication=pub,
        publication_sha256=value["fit_publication_sha256"],
        **dict.fromkeys(q.FLAGS, False),
    )
    return {
        "fit_plan": audit.canonical(plan),
        "verified_status": audit.canonical(status),
        "publication": audit.canonical(pub),
    }


def parent_development(value):
    from collections import Counter

    from posttrain_circuits.experiments.protocols import student_focus_preparation as p

    sizes = Counter(x.metadata["structure"] for x in p._build_bases()["student_dev"])
    rows = []
    for step in p.CHECKPOINT_STEPS:
        proof = 0 if step < value["checkpoint_step"] else 30
        counts = dict(
            num_examples=256, format_valid=256, answer_correct=40 if proof else 0, proof_correct=proof
        )
        strata = {
            name: dict(
                num_examples=sizes[name],
                format_valid=sizes[name],
                answer_correct=ans if proof else 0,
                proof_correct=score if proof else 0,
            )
            for name, ans, score in [("branch", 10, 8), ("chain", 13, 10), ("converging_dag", 17, 12)]
        }
        rows.append(
            dict(
                step=step,
                checkpoint_sha256=value["checkpoint_sha256"]
                if step == value["checkpoint_step"]
                else f"{step:064x}",
                examples_sha256=p.EXAMPLES_SHA256["student_dev"],
                per_view_counts={view: dict(counts) for view in p.DEV_VIEW_NAMES},
                structure_counts=strata,
                **counts,
            )
        )
    return rows


def candidate_fixture(**changes):
    value = dict(
        fit_job_id="99900001",
        fit_intent="1" * 32,
        fit_source_head="f" * 40,
        fit_plan_sha256="2" * 64,
        fit_publication_sha256="3" * 64,
        fit_report_sha256="4" * 64,
        independent_raw_audit_sha256="5" * 64,
        checkpoint_step=2,
        checkpoint_path="checkpoints/step-00000002.pt",
        checkpoint_sha256="9" * 64,
        checkpoint_size=8127108889,
        master_model_state_sha256="6" * 64,
        selection="frozen_earliest_development_candidate_before_this_candidate_qualification",
        checkpoint_reselection_or_retraining=False,
        load="exact_FP32_saved_masters_before_one_native_BF16_inference_copy",
        identity="dense_checkpoint_bytes_never_native_HF_revision",
    )
    value.update(changes)
    return value


CHECKPOINT_SHA256 = candidate_fixture()["checkpoint_sha256"]


@pytest.fixture(scope="module")
def original_population():
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    tokenizer = audit.load_tokenizer()
    return tokenizer, audit.population(tokenizer)


def fixture(original_population, *, validation=13, iid=20, transformed=None, partial=True):
    tokenizer, population = original_population
    prompts, examples, iid_examples, cases = population
    transformed = dict.fromkeys(TRANSFORMATIONS, 16) if transformed is None else transformed
    task, records, results = ProofGraphTask(), [], []
    responses = {}
    for prompt, example in zip(prompts, examples, strict=True):
        ordinal = prompt["ordinal"]
        good = (
            ordinal < validation
            if ordinal < 128
            else (
                ordinal - 128 < iid
                if ordinal < 256
                else (ordinal - 256) // 5 < transformed[prompt["transformation"]]
            )
        )
        text = task.canonical_target(example) if good else ""
        # One correct answer with a valid but incomplete proof distinguishes the
        # base answer metric from anti-shortcut's stricter exact proof reward.
        if ordinal == 0 and good and partial:
            body = "\n".join(render_step(x) for x in example.canonical_proof[:-1])
            text = f"<proof>\n{body}\n</proof>\n<answer>{example.label}</answer>"
        tokens = [*tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id]
        parsed = task.parse_response(text)
        verified = task.verify(example, parsed)
        results.append(verified)
        responses[example.example_id] = text
        records.append(
            dict(
                ordinal=ordinal,
                cohort=prompt["cohort"],
                transformation=prompt["transformation"],
                source_example_id=prompt["source_example_id"],
                example_id=example.example_id,
                pair_group_id=example.pair_group_id,
                prepared_checkpoint_sha256=CHECKPOINT_SHA256,
                prompt_ids=prompt["prompt_ids"],
                prompt_text_sha256=prompt["prompt_text_sha256"],
                prompt_token_sha256=prompt["prompt_token_sha256"],
                raw_prompt_sha256=prompt["raw_prompt_sha256"],
                response_ids=tokens,
                response_tokens=len(tokens),
                response_text=text,
                stop_reason="eos",
                max_new_tokens=256,
                parsed_trace=asdict(parsed),
                parse_error=parsed.error_code,
                verification=asdict(verified),
            )
        )
    base = results[:128]
    validation_result = dict(
        num_examples=128,
        answer_correct=sum(x.answer_correct for x in base),
        proof_correct=sum(x.reward == 1.0 for x in base),
        format_valid=sum(x.parse_valid for x in base),
        **aggregate_verification(base),
        passed=sum(x.answer_correct for x in base) >= 13,
    )
    anti = evaluate_anti_shortcut_suite(
        iid_examples,
        cases,
        lambda example, _prompt: responses[example.example_id],
        max_shortcut_gap=0.05,
        model_checkpoint_hash=CHECKPOINT_SHA256,
        minimum_iid_accuracy=0.10,
        minimum_transformed_accuracy=0.08,
        minimum_per_transformation_accuracy=0.05,
        dataset_hash=audit.IID_SHA256,
        code_commit="a" * 40,
        prereg_commit="b" * 40,
    )
    anti.update(scientific_compatibility_fields("qwen3_v2"))
    anti["sha256"] = audit.sha(audit.canonical({k: v for k, v in anti.items() if k != "sha256"}))
    passed = validation_result["passed"] and anti["passed"]
    report = dict(
        schema="quest-sdsc-student-focus-qualification-report-v1",
        execution_complete=True,
        parameters_unchanged=True,
        prepared_initial=candidate_fixture(),
        loading=dict(
            prepared_checkpoint_sha256=CHECKPOINT_SHA256,
            master_model_state_sha256=candidate_fixture()["master_model_state_sha256"],
            forward_model_state_sha256="7" * 64,
            exact_saved_master_reload=dict(
                all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
            ),
            forward_parameter_dtype="torch.bfloat16",
            all_parameters_frozen=True,
            gradient_checkpointing=False,
            use_cache=False,
        ),
        prepared_checkpoint_sha256=CHECKPOINT_SHA256,
        validation=validation_result,
        anti_shortcut=anti,
        qualification_passed=passed,
        passed=passed,
        scientific_binding=dict(head="a" * 40, prereg_commit="b" * 40),
        generation=dict(
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
            explicit_attention_mask=False,
            explicit_autocast=False,
            precision="native_bfloat16",
            truncation=False,
            seed=42,
            validation_max_model_input_length=1536,
            anti_shortcut_max_model_input_length=2244,
        ),
        **dict.fromkeys(audit.FLAGS, False),
    )
    return report, prompts, records, tokenizer


@pytest.fixture(scope="module")
def passing(original_population):
    return fixture(original_population)


def test_all_896_real_tokenizer_prompts_and_verifications_replay(passing):
    result = audit.replay(*passing, candidate=candidate_fixture())
    assert result["qualification_passed"] is True
    assert result["responses_replayed"] == result["prompts_replayed"] == 896
    assert result["validation"]["answer_correct"] == 13
    assert result["validation"]["proof_correct"] == 12
    assert result["gpu_numerics_independently_recomputed"] is False
    assert all(result[key] is False for key in audit.FLAGS)


@pytest.mark.parametrize(
    "failure", ["validation", "iid", "per_transformation", "transformed_mean", "gap", "zero_zero"]
)
def test_complete_scientific_failure_replays_without_acceptance(original_population, monkeypatch, failure):
    # Population generation is exercised separately with the actual tokenizer;
    # these boundary tests keep it fixed while varying response outcomes only.
    monkeypatch.setattr(audit, "population", lambda _tokenizer: original_population[1])
    options = {}
    if failure == "validation":
        options["validation"] = 12
    elif failure == "iid":
        options["iid"] = 12
    elif failure == "per_transformation":
        options["transformed"] = {**dict.fromkeys(TRANSFORMATIONS, 16), TRANSFORMATIONS[0]: 6}
    elif failure == "transformed_mean":
        options.update(iid=13, transformed=dict.fromkeys(TRANSFORMATIONS, 10))
    elif failure == "gap":
        options.update(iid=23, transformed=dict.fromkeys(TRANSFORMATIONS, 16))
    else:
        options.update(iid=0, transformed=dict.fromkeys(TRANSFORMATIONS, 0))
    result = audit.replay(*fixture(original_population, **options), candidate=candidate_fixture())
    assert result["schema"] == "quest-sdsc-student-focus-qualification-raw-audit-v1"
    assert result["raw_replay_passed"] is True
    assert result["qualification_passed"] is False


def test_negative_shortcut_gap_is_not_an_absolute_gap_gate(original_population, monkeypatch):
    monkeypatch.setattr(audit, "population", lambda _tokenizer: original_population[1])
    result = audit.replay(
        *fixture(original_population, iid=13, transformed=dict.fromkeys(TRANSFORMATIONS, 120)),
        candidate=candidate_fixture(),
    )
    assert result["anti_shortcut"]["shortcut_gap"] < -0.8
    assert result["qualification_passed"] is True


@pytest.mark.parametrize(
    "corruption",
    [
        "verifier",
        "parser",
        "tokens",
        "checkpoint",
        "duplicate",
        "missing",
        "population",
        "prompt",
        "cap",
        "after_eos",
        "summary",
        "threshold",
        "overclaim",
        "generation",
        "source",
        "candidate",
        "master",
        "parameters",
        "reload",
    ],
)
def test_raw_response_corruptions_fail_closed(passing, original_population, monkeypatch, corruption):
    monkeypatch.setattr(audit, "population", lambda _tokenizer: original_population[1])
    report, prompts, records, tokenizer = passing
    report, prompts, records = copy.deepcopy((report, prompts, records))
    row = records[0]
    if corruption == "candidate":
        report["prepared_initial"]["checkpoint_step"] = 4
    elif corruption == "master":
        report["loading"]["master_model_state_sha256"] = "f" * 64
    elif corruption == "parameters":
        report["parameters_unchanged"] = False
    elif corruption == "reload":
        report["loading"]["exact_saved_master_reload"]["loaded_before_bf16_copy"] = False
    elif corruption == "verifier":
        row["verification"]["answer_correct"] = False
    elif corruption == "parser":
        row["parsed_trace"]["answer"] = 1 - row["parsed_trace"]["answer"]
    elif corruption == "tokens":
        row["response_ids"][0] = 0
    elif corruption == "checkpoint":
        row["prepared_checkpoint_sha256"] = "f" * 64
    elif corruption == "duplicate":
        records[1] = copy.deepcopy(row)
    elif corruption == "missing":
        records.pop()
    elif corruption == "population":
        prompts[0]["example"]["label"] = 1 - prompts[0]["example"]["label"]
    elif corruption == "prompt":
        prompts[0]["prompt_ids"] = prompts[0]["prompt_ids"][:-1]
    elif corruption == "cap":
        row["max_new_tokens"] = 128
    elif corruption == "after_eos":
        row["response_ids"] += [tokenizer.eos_token_id]
    elif corruption == "summary":
        report["validation"]["answer_correct"] += 1
    elif corruption == "threshold":
        report["anti_shortcut"]["minimum_per_transformation_accuracy"] = 0.0
    elif corruption == "overclaim":
        report["formal_initial_accepted"] = True
    elif corruption == "generation":
        report["generation"]["explicit_autocast"] = True
    else:
        report["scientific_binding"]["head"] = "c" * 40
    with pytest.raises(ValueError):
        audit.replay(report, prompts, records, tokenizer, candidate=candidate_fixture())


def publication_fixture():
    from posttrain_circuits.experiments.protocols import student_focus_qualification as qualification

    candidate = candidate_fixture()
    payload = qualification.proposed_student_focus_qualification_protocol(candidate)
    parent = payload["preserved_base"]
    completion_raw = completion_fixture(candidate)
    parent_plan = audit.document(completion_raw["fit_plan"])
    completion = {
        name: dict(sha256=audit.sha(raw), size=len(raw), base64=base64.b64encode(raw).decode())
        for name, raw in completion_raw.items()
        if name != "fit_plan"
    }
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation

    development = parent_development(candidate)
    parent_audit = dict(
        schema="quest-sdsc-student-focus-raw-audit-v1",
        preparation_complete=True,
        dataset_sha256=preparation.DATASET_MANIFEST_SHA256,
        development=development,
        training_evidence=dict(
            training_order_reconstructed=True,
            training_rows_reconstructed=2048,
            optimizer_windows_replayed=32,
            consumed_training_rows=2048,
            full_population_input_tokens=1665064,
            training_input_tokens=1665064,
            checkpoint_cumulative_input_tokens=qualification.PARENT_CHECKPOINT_INPUT_TOKENS,
            gpu_training_independently_recomputed=False,
        ),
        isolation_evidence=dict(
            dataset_manifest_sha256=preparation.DATASET_MANIFEST_SHA256,
            producer_evidence_sha256="e" * 64,
            producer_isolation_consistency_verified=True,
            historical_isolation_independently_recomputed=False,
        ),
        mode="fit",
        job_id=candidate["fit_job_id"],
        plan_sha256=candidate["fit_plan_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        auditor_sha256=parent["parent_auditor_sha256"],
        protocol_sha256=parent["parent_protocol_core_sha256"],
        protocol_artifact_sha256=parent["parent_student_protocol_sha256"],
        implementation_commit=parent["parent_implementation_commit"],
        acceptance_commit=parent["parent_acceptance_commit"],
        raw_replay_passed=True,
        prompts_replayed=1536,
        responses_replayed=18432,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
        selected_checkpoint=copy.deepcopy(development[1]),
        **dict.fromkeys(audit.FLAGS, False),
    )
    parent_raw = audit.canonical(parent_audit) + b"\n"
    candidate["independent_raw_audit_sha256"] = audit.sha(parent_raw)
    payload = qualification.proposed_student_focus_qualification_protocol(candidate)
    binding = SimpleNamespace(
        science_file_sha256={"auditor.py": "9" * 64},
        implementation_commit="a" * 40,
        acceptance_commit="b" * 40,
        protocol_sha256="c" * 64,
        artifact_sha256="d" * 64,
        head="1" * 40,
        payload=payload,
    )
    prereg_commit = "2" * 40
    task = "qwen3-v2-student-focus-qualification-v1"
    source = {key: value for key, value in vars(binding).items() if key != "payload"}
    source["head"] = "3" * 40
    plan = dict(
        prepared_checkpoint={
            "path": candidate["checkpoint_path"],
            "size": candidate["checkpoint_size"],
            "sha256": CHECKPOINT_SHA256,
        },
        fit=dict(
            completion=completion,
            plan=parent_plan,
            job_id=candidate["fit_job_id"],
            plan_sha256=candidate["fit_plan_sha256"],
            publication_sha256=candidate["fit_publication_sha256"],
            report_sha256=candidate["fit_report_sha256"],
            audit=dict(
                sha256=audit.sha(parent_raw),
                size=len(parent_raw),
                base64=base64.b64encode(parent_raw).decode(),
            ),
        ),
        schema="quest-sdsc-student-focus-qualification-plan-v1",
        task=task,
        protocol=source,
        provenance=dict(head=source["head"]),
        run_id="fixed-run",
        intent_id="4" * 32,
        code_sha256="e" * 64,
        **dict.fromkeys(audit.FLAGS, False),
    )
    fit_evidence = dict(
        completion={name: dict(size=len(raw), sha256=audit.sha(raw)) for name, raw in completion_raw.items()},
        fit_job_id=candidate["fit_job_id"],
        publication_sha256=candidate["fit_publication_sha256"],
        report_sha256=candidate["fit_report_sha256"],
        independent_audit_sha256=candidate["independent_raw_audit_sha256"],
        master_model_state_sha256=candidate["master_model_state_sha256"],
        checkpoint_path=candidate["checkpoint_path"],
        checkpoint_size=candidate["checkpoint_size"],
        selected_checkpoint=parent_audit["selected_checkpoint"],
    )
    report = dict(
        passed=True,
        qualification_passed=True,
        prepared_initial=candidate,
        prepared_checkpoint_sha256=CHECKPOINT_SHA256,
        fit_evidence=fit_evidence,
        preparation_protocol_sha256=parent["parent_protocol_core_sha256"],
        preparation_protocol_artifact_sha256=parent["parent_student_protocol_sha256"],
        task=task,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=audit.sha(audit.canonical(plan)),
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        scientific_binding={
            key: source[key]
            for key in ("head", "implementation_commit", "acceptance_commit", "science_file_sha256")
        },
        **dict.fromkeys(audit.FLAGS, False),
    )
    report["scientific_binding"]["prereg_commit"] = prereg_commit
    files = {
        name: audit.canonical(report if name == audit.RAW_NAMES[0] else {"example": 1})
        for name in audit.RAW_NAMES
    }
    publication = dict(
        schema="quest-sdsc-student-focus-qualification-publication-v1",
        passed=True,
        stage_complete=True,
        qualification_passed=True,
        task=task,
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=report["plan_sha256"],
        persistent_read_back_verified=True,
        files=[dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()],
        **dict.fromkeys(audit.FLAGS, False),
    )
    raw = audit.canonical(publication)
    return plan, raw, audit.sha(raw), files, binding, prereg_commit


def test_externally_bound_publication_validates():
    values = publication_fixture()
    report, publication = audit.validate_publication(*values)
    assert report["job_id"] == publication["job_id"] == "123"
    assert report["scientific_binding"]["head"] != values[4].head


@pytest.mark.parametrize(
    "corruption",
    [
        "external_hash",
        "plan",
        "source",
        "file",
        "duplicate",
        "unsafe",
        "readback",
        "receipt_task",
        "receipt_intent",
        "receipt_source",
        "receipt_overclaim",
        "bool_size",
        "missing",
    ],
)
def test_publication_corruption_fails_closed(corruption):
    plan, raw, expected, files, binding, prereg = publication_fixture()
    if corruption == "external_hash":
        expected = "0" * 64
    elif corruption == "plan":
        plan["run_id"] = "different"
    elif corruption == "source":
        binding.science_file_sha256 = {"auditor.py": "0" * 64}
    elif corruption == "file":
        files[audit.RAW_NAMES[0]] += b" "
    else:
        publication = audit.document(raw)
        if corruption == "duplicate":
            publication["files"].append(publication["files"][0])
        elif corruption == "unsafe":
            publication["files"].append(dict(path="../unsafe", size=1, sha256="0" * 64))
        elif corruption == "readback":
            publication["persistent_read_back_verified"] = False
        elif corruption == "receipt_task":
            publication["task"] = "different-task"
        elif corruption == "receipt_intent":
            publication["intent_id"] = "0" * 32
        elif corruption == "receipt_source":
            publication["code_sha256"] = "0" * 64
        elif corruption == "receipt_overclaim":
            publication["formal_initial_accepted"] = True
        elif corruption == "bool_size":
            publication["files"][0]["size"] = True
        else:
            publication["files"].pop()
        raw = audit.canonical(publication)
        expected = audit.sha(raw)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, expected, files, binding, prereg)


@pytest.mark.parametrize(
    "corruption",
    [
        "task",
        "schema",
        "plan_head",
        "science",
        "implementation",
        "acceptance",
        "core",
        "artifact",
        "plan_overclaim",
        "report_head",
        "report_implementation",
        "report_acceptance",
        "report_science",
        "prereg",
        "report_job",
        "report_source",
        "report_overclaim",
    ],
)
def test_rehashed_publication_still_rejects_false_scientific_bindings(corruption):
    plan, raw, _, files, binding, prereg = publication_fixture()
    report = audit.document(files[audit.RAW_NAMES[0]])
    if corruption in {"task", "schema"}:
        plan[corruption] = "another-identity"
    elif corruption == "plan_head":
        plan["provenance"]["head"] = "9" * 40
    elif corruption == "plan_overclaim":
        plan["formal_initial_accepted"] = True
    elif corruption in {"science", "implementation", "acceptance", "core", "artifact"}:
        key = {
            "science": "science_file_sha256",
            "implementation": "implementation_commit",
            "acceptance": "acceptance_commit",
            "core": "protocol_sha256",
            "artifact": "artifact_sha256",
        }[corruption]
        plan["protocol"][key] = {"auditor.py": "0" * 64} if corruption == "science" else "0" * 64
    elif corruption.startswith("report_"):
        key = corruption.removeprefix("report_")
        if key in {"head", "implementation", "acceptance", "science"}:
            key = {
                "head": "head",
                "implementation": "implementation_commit",
                "acceptance": "acceptance_commit",
                "science": "science_file_sha256",
            }[key]
            report["scientific_binding"][key] = (
                {"auditor.py": "0" * 64} if key == "science_file_sha256" else "9" * 40
            )
        elif key == "job":
            report["job_id"] = "456"
        elif key == "source":
            report["source_code_sha256"] = "0" * 64
        else:
            report["formal_initial_accepted"] = True
    else:
        report["scientific_binding"]["prereg_commit"] = "9" * 40
    report["plan_sha256"] = audit.sha(audit.canonical(plan))
    files[audit.RAW_NAMES[0]] = audit.canonical(report)
    publication = audit.document(raw)
    publication["plan_sha256"] = report["plan_sha256"]
    publication["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    raw = audit.canonical(publication)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding, prereg)


def test_prereg_provenance_uses_original_producer_commit_and_rejects_false_lineage(tmp_path):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Provenance fixture")
    (tmp_path / "prereg").mkdir()
    prereg = tmp_path / "prereg/qwen3_v2.yaml"
    prereg.write_text("original frozen preregistration\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Original preregistration")
    original = git("rev-parse", "HEAD")
    (tmp_path / "accepted.txt").write_text("qualification accepted\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Qualification acceptance")
    producer = git("rev-parse", "HEAD")
    prereg.write_text("later producer-unrelated preregistration\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Later observer state")
    current = git("rev-parse", "HEAD")
    binding = SimpleNamespace(acceptance_commit=producer, head=current)
    assert audit.prereg_commit_at(tmp_path, tmp_path / ".git", producer, binding) == original
    assert audit.prereg_commit_at(tmp_path, tmp_path / ".git", current, binding) == current
    with pytest.raises(ValueError, match="predates accepted"):
        audit.prereg_commit_at(tmp_path, tmp_path / ".git", original, binding)
    binding.head = producer
    with pytest.raises(ValueError, match="outside current accepted lineage"):
        audit.prereg_commit_at(tmp_path, tmp_path / ".git", current, binding)


def test_duplicate_nonfinite_json_and_wrong_population_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        audit.document(b'{"passed":true,"passed":false}')
    with pytest.raises(ValueError):
        audit.document(b'{"loss":NaN}')
    with pytest.raises(ValueError, match="population"):
        audit.lines(b"{}\n", 896)


@pytest.mark.parametrize(
    "corruption",
    [
        "bytes",
        "size",
        "newline",
        "count",
        "core",
        "lineage",
        "scope",
        "null",
        "step",
        "boolean_count",
        "duplicate_json",
    ],
)
def test_independent_parent_audit_decoder_rejects_rehashed_bad_evidence(corruption):
    plan, _, _, _, binding, _ = publication_fixture()
    candidate = copy.deepcopy(binding.payload["prepared_initial"])
    record = copy.deepcopy(plan["fit"]["audit"])
    raw = base64.b64decode(record["base64"])
    value = audit.document(raw)
    if corruption == "bytes":
        record["base64"] = base64.b64encode(raw + b" ").decode()
    elif corruption == "size":
        record["size"] = True
    else:
        if corruption == "count":
            value["responses_replayed"] -= 1
        elif corruption == "core":
            value["protocol_sha256"] = "f" * 64
        elif corruption == "lineage":
            value["acceptance_commit"] = "f" * 40
        elif corruption == "scope":
            value["execution_acceptance_claim"] = True
        elif corruption == "null":
            value["selected_checkpoint"] = None
        elif corruption == "step":
            value["selected_checkpoint"]["step"] = 4
        elif corruption == "boolean_count":
            value["prompts_replayed"] = True
        raw = audit.canonical(value) + (b"" if corruption == "newline" else b"\n")
        if corruption == "duplicate_json":
            raw = raw.replace(b"{", b'{"mode":"fit",', 1)
        record = dict(size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode())
        candidate["independent_raw_audit_sha256"] = record["sha256"]
    with pytest.raises(ValueError):
        audit.validate_fit_audit_record(record, candidate, binding.payload["preserved_base"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_status",
        "old_wrapper",
        "status_failed",
        "queue_retained",
        "old_publication",
        "selected_sha",
        "isolation_digest",
        "extra_bundle",
        "noncanonical_status",
    ],
)
def test_flat_completion_decoder_fails_closed_after_rehash(mutation):
    plan, _, _, _, binding, _ = publication_fixture()
    candidate = binding.payload["prepared_initial"]
    fit = plan["fit"]
    records = fit["completion"]
    if mutation == "missing_status":
        records.pop("verified_status")
    elif mutation == "old_wrapper":
        records["execution_plan"] = records.pop("verified_status")
    elif mutation == "extra_bundle":
        records["unexpected"] = copy.deepcopy(records["publication"])
    elif mutation == "noncanonical_status":
        raw = base64.b64decode(records["verified_status"]["base64"]) + b"\n"
        records["verified_status"] = dict(
            size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode()
        )
    else:
        status = audit.document(base64.b64decode(records["verified_status"]["base64"]))
        pub = audit.document(base64.b64decode(records["publication"]["base64"]))
        if mutation == "status_failed":
            status["success"] = False
        elif mutation == "queue_retained":
            status["queue"]["stdout"] = "99900001|COMPLETED"
        elif mutation == "old_publication":
            pub["schema"] = "quest-sdsc-student-order-execution-publication-v2"
        elif mutation == "selected_sha":
            pub["large_files"][0]["sha256"] = "a" * 64
        elif mutation == "isolation_digest":
            pub["files"][1]["sha256"] = "b" * 64
        if mutation == "isolation_digest":
            # Cross-binding to the independent full-fit raw audit is checked by
            # validate_publication, not the bare completion decoder.
            record = fit["audit"]
            raw = base64.b64decode(record["base64"])
            a = audit.document(raw)
            a["isolation_evidence"]["producer_evidence_sha256"] = "invalid"
            raw = audit.canonical(a) + b"\n"
            record.update(size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode())
            candidate["independent_raw_audit_sha256"] = record["sha256"]
            with pytest.raises(ValueError):
                audit.validate_fit_audit_record(record, candidate, binding.payload["preserved_base"])
            return
        pubraw = audit.canonical(pub)
        candidate["fit_publication_sha256"] = audit.sha(pubraw)
        status.update(publication=pub, publication_sha256=candidate["fit_publication_sha256"])
        for name, raw in [("publication", pubraw), ("verified_status", audit.canonical(status))]:
            records[name] = dict(size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode())
    with pytest.raises(ValueError):
        audit.decode_completion_records(records, candidate, fit["plan"])


def test_completion_preserves_exact_three_staged_records():
    plan, _, _, _, binding, _ = publication_fixture()
    fit = plan["fit"]
    rows = audit.decode_completion_records(
        fit["completion"], binding.payload["prepared_initial"], fit["plan"]
    )
    assert set(rows) == {"fit_plan", "verified_status", "publication"}
    assert rows["fit_plan"] == audit.canonical(fit["plan"])
    for name in fit["completion"]:
        assert rows[name] == base64.b64decode(fit["completion"][name]["base64"])


@pytest.mark.parametrize(
    "field",
    [
        "training_input_tokens",
        "full_population_input_tokens",
        "optimizer_windows_replayed",
        "consumed_training_rows",
        "training_rows_reconstructed",
        "checkpoint_cumulative_input_tokens",
    ],
)
def test_independent_full_training_audit_requires_all32_windows(field):
    plan, _, _, _, binding, _ = publication_fixture()
    candidate = binding.payload["prepared_initial"]
    record = plan["fit"]["audit"]
    value = audit.document(base64.b64decode(record["base64"]))
    value["training_evidence"][field] = 0
    raw = audit.canonical(value) + b"\n"
    record.update(size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode())
    candidate["independent_raw_audit_sha256"] = record["sha256"]
    with pytest.raises(ValueError):
        audit.validate_fit_audit_record(record, candidate, binding.payload["preserved_base"])


@pytest.mark.parametrize(
    "mutation",
    [
        "old_schema",
        "preflight_mode",
        "incomplete_preparation",
        "prefix_shift",
        "prefix_bool",
        "isolation_dataset",
        "isolation_recomputed",
        "isolation_unverified",
        "isolation_old_key",
        "missing_checkpoint",
        "reselected_later",
        "old_population",
    ],
)
def test_rehashed_parent_semantics_rejected(mutation):
    plan, _, _, _, binding, _ = publication_fixture()
    candidate = binding.payload["prepared_initial"]
    record = plan["fit"]["audit"]
    parent = audit.document(base64.b64decode(record["base64"]))
    if mutation == "old_schema":
        parent["schema"] = "quest-sdsc-student-order-raw-audit-v1"
    elif mutation == "preflight_mode":
        parent["mode"] = "preflight"
    elif mutation == "incomplete_preparation":
        parent["preparation_complete"] = False
    elif mutation == "prefix_shift":
        # Preserve full consumption but contradict an earlier checkpoint's exact budget.
        parent["training_evidence"]["checkpoint_cumulative_input_tokens"]["4"] += 1
    elif mutation == "prefix_bool":
        parent["training_evidence"]["checkpoint_cumulative_input_tokens"]["1"] = True
    elif mutation == "isolation_dataset":
        parent["isolation_evidence"]["dataset_manifest_sha256"] = "a" * 64
    elif mutation == "isolation_recomputed":
        parent["isolation_evidence"]["historical_isolation_independently_recomputed"] = True
    elif mutation == "isolation_unverified":
        parent["isolation_evidence"]["producer_isolation_consistency_verified"] = False
    elif mutation == "isolation_old_key":
        parent["isolation_evidence"]["preparation_manifest_sha256"] = parent["isolation_evidence"].pop(
            "dataset_manifest_sha256"
        )
    elif mutation == "missing_checkpoint":
        parent["development"].pop()
    elif mutation == "reselected_later":
        parent["selected_checkpoint"] = copy.deepcopy(parent["development"][2])
        candidate.update(
            checkpoint_step=3,
            checkpoint_sha256=parent["selected_checkpoint"]["checkpoint_sha256"],
            checkpoint_path="checkpoints/step-00000003.pt",
        )
    elif mutation == "old_population":
        parent["development"][0]["examples_sha256"] = "a" * 64
    raw = audit.canonical(parent) + b"\n"
    record.update(size=len(raw), sha256=audit.sha(raw), base64=base64.b64encode(raw).decode())
    candidate["independent_raw_audit_sha256"] = record["sha256"]
    with pytest.raises(ValueError):
        audit.validate_fit_audit_record(record, candidate, binding.payload["preserved_base"])


def test_parent_isolation_valid_digest_must_match_retained_publication_after_rehash():
    plan, raw, _, files, binding, prereg = publication_fixture()
    candidate = binding.payload["prepared_initial"]
    record = plan["fit"]["audit"]
    parent = audit.document(base64.b64decode(record["base64"]))
    parent["isolation_evidence"]["producer_evidence_sha256"] = "a" * 64
    parent_raw = audit.canonical(parent) + b"\n"
    record.update(
        size=len(parent_raw), sha256=audit.sha(parent_raw), base64=base64.b64encode(parent_raw).decode()
    )
    candidate["independent_raw_audit_sha256"] = record["sha256"]
    report = audit.document(files[audit.RAW_NAMES[0]])
    report["prepared_initial"] = copy.deepcopy(candidate)
    report["fit_evidence"]["independent_audit_sha256"] = record["sha256"]
    report["plan_sha256"] = audit.sha(audit.canonical(plan))
    files[audit.RAW_NAMES[0]] = audit.canonical(report)
    publication = audit.document(raw)
    publication["plan_sha256"] = report["plan_sha256"]
    publication["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    raw = audit.canonical(publication)
    with pytest.raises(ValueError, match="independent isolation audit differs"):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding, prereg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "quest-sdsc-student-order-qualification-publication-v1"),
        ("passed", 1),
        ("stage_complete", False),
        ("qualification_passed", None),
    ],
)
def test_rehashed_qualification_publication_has_exact_schema_and_typed_flags(field, value):
    plan, raw, _, files, binding, prereg = publication_fixture()
    publication = audit.document(raw)
    publication[field] = value
    raw = audit.canonical(publication)
    with pytest.raises(ValueError):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding, prereg)


@pytest.mark.parametrize("report_passed", [False, True])
def test_failed_qualification_publication_preserves_complete_raw_evidence(report_passed):
    plan, raw, _, files, binding, prereg = publication_fixture()
    report = audit.document(files[audit.RAW_NAMES[0]])
    report.update(passed=report_passed, qualification_passed=report_passed)
    files[audit.RAW_NAMES[0]] = audit.canonical(report)
    publication = audit.document(raw)
    publication.update(passed=False, stage_complete=False, qualification_passed=False)
    publication["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    raw = audit.canonical(publication)
    observed, retained = audit.validate_publication(plan, raw, audit.sha(raw), files, binding, prereg)
    assert observed["passed"] is report_passed and retained["passed"] is False


def test_successful_qualification_publication_requires_successful_report():
    plan, raw, _, files, binding, prereg = publication_fixture()
    report = audit.document(files[audit.RAW_NAMES[0]])
    report.update(passed=False, qualification_passed=False)
    files[audit.RAW_NAMES[0]] = audit.canonical(report)
    publication = audit.document(raw)
    publication["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in files.items()]
    raw = audit.canonical(publication)
    with pytest.raises(ValueError, match="lacks a successful scientific report"):
        audit.validate_publication(plan, raw, audit.sha(raw), files, binding, prereg)
