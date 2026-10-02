"""Independent full-population replay and corruption rejection, with no GPU."""

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
from posttrain_circuits.experiments.protocols.student_qualification import CHECKPOINT_SHA256

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qualify_audit", ROOT / "tools/sdsc_student_qualify_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


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
        schema="quest-sdsc-student-qualification-report-v1",
        execution_complete=True,
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
    result = audit.replay(*passing)
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
    result = audit.replay(*fixture(original_population, **options))
    assert result["raw_replay_passed"] is True
    assert result["qualification_passed"] is False


def test_negative_shortcut_gap_is_not_an_absolute_gap_gate(original_population, monkeypatch):
    monkeypatch.setattr(audit, "population", lambda _tokenizer: original_population[1])
    result = audit.replay(
        *fixture(original_population, iid=13, transformed=dict.fromkeys(TRANSFORMATIONS, 120))
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
    ],
)
def test_raw_response_corruptions_fail_closed(passing, original_population, monkeypatch, corruption):
    monkeypatch.setattr(audit, "population", lambda _tokenizer: original_population[1])
    report, prompts, records, tokenizer = passing
    report, prompts, records = copy.deepcopy((report, prompts, records))
    row = records[0]
    if corruption == "verifier":
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
        audit.replay(report, prompts, records, tokenizer)


def publication_fixture():
    binding = SimpleNamespace(
        science_file_sha256={"auditor.py": "f" * 64},
        implementation_commit="a" * 40,
        acceptance_commit="b" * 40,
        protocol_sha256="c" * 64,
        artifact_sha256="d" * 64,
        head="1" * 40,
    )
    prereg_commit = "2" * 40
    task = "qwen3-v2-student-qualification-v1"
    # The observer may be on a later, unrelated commit; provenance is bound to
    # the producer's accepted source HEAD, not the observer's current HEAD.
    source = dict(vars(binding), head="3" * 40)
    plan = dict(
        schema="quest-sdsc-student-qualification-plan-v1",
        task=task,
        protocol=source,
        provenance=dict(head=source["head"]),
        run_id="fixed-run",
        intent_id="4" * 32,
        code_sha256="e" * 64,
        **dict.fromkeys(audit.FLAGS, False),
    )
    report = dict(
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
