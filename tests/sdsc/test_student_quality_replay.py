"""CPU producer→independent verifier replay and semantic tampering regressions."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def case(tmp_path, monkeypatch):
    api = module("sdsc_student_quality_replay")
    worker = module("sdsc_student_quality_probe")
    task = ProofGraphTask()
    example = task.generate(42, {"depth": 1, "structure": "chain", "positive": True})
    examples, prompts = {}, []
    for cohort, count in api.COHORTS:
        examples[cohort] = []
        for i in range(count):
            item = replace(example, example_id=f"{cohort}-{i}", pair_group_id=f"pair-{cohort}-{i}")
            examples[cohort].append(item)
            ids, text = [10, i + 100], "<user>" + task.render(item) + "</user>"
            prompts.append(
                {
                    "cohort": cohort,
                    "example": asdict(item),
                    "prompt_text": text,
                    "prompt_ids": ids,
                    "prompt_token_sha256": api.sha(api.canonical(ids)),
                    "canonical_target": task.canonical_target(item),
                }
            )
    records, arms = [], []
    for label in api.LABELS:
        prompt_offset = 0
        for cohort, count in api.COHORTS:
            previous = None
            for cap in (128, 256):
                rows = []
                for i, item in enumerate(examples[cohort]):
                    # Every fourth answer has the right tag but fails strict proof
                    # verification. Synthetic IDs deliberately do not claim decode replay.
                    malformed = i % 4 == 0
                    text = f"<answer>{item.label}</answer>" if malformed else task.canonical_target(item)
                    ids = (
                        [8] * 128
                        if malformed and cap == 128
                        else [8] * 128 + [151645]
                        if malformed
                        else [9, 151645]
                    )
                    tokenizer = SimpleNamespace(
                        eos_token_id=151645, decode=lambda _ids, text=text, **_kwargs: text
                    )
                    prompt = prompts[prompt_offset + i]
                    row = worker.response_record(
                        item,
                        ids,
                        tokenizer,
                        prompt_ids=prompt["prompt_ids"],
                        prompt_text=prompt["prompt_text"],
                        cap=cap,
                    )
                    row.pop("prompt_ids")
                    rows.append(dict(checkpoint=label, cohort=cohort, cap=cap, **row))
                metrics = worker.summarize(rows)
                if previous is not None:
                    metrics["paired_prefix_equal_count"] = count
                arms.append(
                    {
                        "checkpoint": label,
                        "cohort": cohort,
                        "cap": cap,
                        "metrics": metrics,
                        "teacher_forced": {
                            "examples": count,
                            "response_tokens_including_eos": 50 * count,
                            "canonical_response_nll": 0.25,
                            "canonical_response_token_accuracy": 0.75,
                        },
                        "raw_file": "quality-records.jsonl",
                        "record_start": len(records),
                        "record_count": count,
                    }
                )
                records.extend(rows)
                previous = rows
            prompt_offset += count
    flags = {key: False for key in api.FALSE_FLAGS}
    plan = {
        "schema": "quest-sdsc-student-quality-plan-v1",
        "task": api.TASK,
        "scope": "inference_diagnostic_only",
        "parent": {"receipt": {"job_id": api.PARENT}},
        "run_id": "fixture-run",
        "intent_id": "a" * 32,
        "code_sha256": "b" * 64,
        "dataset_inputs": [{"path": "manifest.json", "size": 100, "sha256": "c" * 64}],
        "checkpoints": [
            {"label": label, "size": 100 + i, "sha256": str(i + 1) * 64} for i, label in enumerate(api.LABELS)
        ],
        **flags,
    }
    report = {
        "schema": "quest-sdsc-student-quality-probe-v1",
        "passed": True,
        "diagnostic_complete": True,
        "job_id": api.JOB,
        "parent_job_id": api.PARENT,
        "run_id": plan["run_id"],
        "source_code_sha256": plan["code_sha256"],
        "dataset_manifest_sha256": "c" * 64,
        "generation": {
            "do_sample": False,
            "use_cache": False,
            "caps": [128, 256],
            "max_model_input_length": 1536,
        },
        "cohorts": {
            cohort: {"count": count, "ordered_ids": [x.example_id for x in examples[cohort]]}
            for cohort, count in api.COHORTS
        },
        "checkpoint_evidence": [
            dict(
                row,
                saved_dtype="torch.bfloat16" if row["label"] == "initial" else "torch.float32",
                loaded_exactly=True,
                forward_parameter_dtype="torch.bfloat16",
            )
            for row in plan["checkpoints"]
        ],
        "raw_record_count": 960,
        "arms": arms,
        **flags,
    }
    node = dict(
        job_id=api.JOB,
        parent_job_id=api.PARENT,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        diagnostic_complete=True,
        exit_code=0,
        **flags,
    )
    metadata_items = []
    documents = {
        "config.json": {"eos_token_id": 151645, "max_position_embeddings": 40960},
        "tokenizer_config.json": {
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {"151645": {"content": "<|im_end|>"}},
        },
    }
    for name, value in documents.items():
        raw = api.canonical(value)
        metadata_items.append(
            dict(
                name=name,
                path=f"/cache/snapshots/{api.REVISION}/{name}",
                resolved_path="/cache/blobs/fixture",
                size=len(raw),
                sha256=api.sha(raw),
                text=raw.decode(),
            )
        )
    # Test-only pins for minimal metadata fixtures; production constants remain fixed.
    monkeypatch.setattr(api, "METADATA_SHA", {x["name"]: x["sha256"] for x in metadata_items})
    metadata = {"revision": api.REVISION, "items": metadata_items}
    value = SimpleNamespace(
        api=api,
        plan=plan,
        report=report,
        node=node,
        metadata=metadata,
        prompts=prompts,
        records=records,
        root=tmp_path,
    )

    def seal():
        payloads = {
            "quality-prompts.jsonl": b"".join(api.canonical(x) + b"\n" for x in value.prompts),
            "quality-records.jsonl": b"".join(api.canonical(x) + b"\n" for x in value.records),
            "memory.json": api.canonical({"fixture_only": True}),
        }
        raw_plan = api.canonical(plan)
        plan_sha = api.sha(raw_plan)
        node["plan_sha256"] = plan_sha
        report["raw_artifacts"] = [
            dict(path=name, size=len(payloads[name]), sha256=api.sha(payloads[name]))
            for name in ("quality-records.jsonl", "quality-prompts.jsonl")
        ]
        payloads.update(
            {"quality-probe.json": api.canonical(report), "node-result.json": api.canonical(node)}
        )
        receipt = dict(
            task=api.TASK,
            job_id=api.JOB,
            run_id=plan["run_id"],
            intent_id=plan["intent_id"],
            plan_sha256=plan_sha,
            code_sha256=plan["code_sha256"],
            passed=True,
            diagnostic_complete=True,
            persistent_read_back_verified=True,
            **flags,
            files=[
                dict(path=name, size=len(payloads[name]), sha256=api.sha(payloads[name]))
                for name in api.NAMES
            ],
        )
        receipt_raw = api.canonical(receipt)
        for name, raw in dict(
            payloads,
            **{
                "receipt.json": receipt_raw,
                "plan.json": raw_plan,
                "tokenizer-metadata.json": api.canonical(metadata),
            },
        ).items():
            (tmp_path / name).write_bytes(raw)
        return dict(
            plan_sha256=plan_sha,
            receipt_sha256=api.sha(receipt_raw),
            tokenizer_metadata=tmp_path / "tokenizer-metadata.json",
        )

    value.seal = seal
    value.replay = lambda: api.replay(tmp_path, tmp_path / "plan.json", **seal())
    return value


def test_full_producer_records_replay_real_parser_and_every_ordered_arm(case):
    observed = case.replay()
    assert observed["passed"] is True
    assert observed["prompt_count"] == 160 and observed["response_count"] == 960
    assert len(observed["arms"]) == 12
    for arm in observed["arms"]:
        assert arm["metrics"]["validation_accuracy"] == 0.75
        assert arm["metrics"]["diagnostic_answer_tag_accuracy"] == 1.0
        if arm["cap"] == 256:
            assert arm["metrics"]["paired_prefix_equal_count"] == arm["metrics"]["count"]
    assert all(observed[key] is False for key in (*case.api.FALSE_FLAGS, "new_test_holdout"))
    assert "not decoded" in observed["limitations"][0]
    assert "not recomputed" in observed["limitations"][1]


@pytest.mark.parametrize(
    "fault,match",
    [
        ("duplicate_prompt", "duplicate prompt ID"),
        ("record_order", "record identity/order"),
        ("missing_record", "JSONL row count"),
        ("missing_arm", "expected 12 arms"),
        ("trace", "verification trace"),
        ("tag", "answer-tag"),
        ("stop", "stop reason"),
        ("aggregate", "aggregate metrics"),
        ("prefix", "aggregate metrics"),
        ("checkpoint", "checkpoint/precision"),
        ("dataset", "dataset manifest"),
        ("canonical_target", "canonical target"),
        ("token_hash", "prompt token hash"),
        ("nll_cross_cap", "teacher-forced values"),
        ("bool_metric", "aggregate metrics"),
        ("acceptance", "acceptance overclaim"),
    ],
)
def test_resealed_semantic_tampering_is_rejected(case, fault, match):
    if fault == "duplicate_prompt":
        case.prompts[1] = copy.deepcopy(case.prompts[0])
    elif fault == "record_order":
        case.records[0], case.records[1] = case.records[1], case.records[0]
    elif fault == "missing_record":
        case.records.pop()
    elif fault == "missing_arm":
        case.report["arms"].pop()
    elif fault == "trace":
        case.records[1]["verification"]["step_results"][0]["valid"] = False
    elif fault == "tag":
        case.records[0]["diagnostic_answer_tag_correct"] = False
    elif fault == "stop":
        case.records[0]["stop_reason"] = "eos"
    elif fault == "aggregate":
        case.report["arms"][0]["metrics"]["validation_accuracy"] = 1.0
    elif fault == "prefix":
        case.records[128]["response_ids"][0] = 7
    elif fault == "checkpoint":
        case.report["checkpoint_evidence"][1]["sha256"] = "f" * 64
    elif fault == "dataset":
        case.report["dataset_manifest_sha256"] = "f" * 64
    elif fault == "canonical_target":
        case.prompts[0]["canonical_target"] += " extra"
    elif fault == "token_hash":
        case.prompts[0]["prompt_ids"][0] += 1
    elif fault == "nll_cross_cap":
        case.report["arms"][1]["teacher_forced"]["canonical_response_nll"] = 0.5
    elif fault == "bool_metric":
        case.report["arms"][0]["metrics"]["diagnostic_answer_tag_accuracy"] = True
    elif fault == "acceptance":
        case.report["g0_passed"] = True
    with pytest.raises(ValueError, match=match):
        case.replay()


def test_post_fetch_byte_change_and_wrong_external_receipt_are_rejected(case):
    kwargs = case.seal()
    raw = case.root / "quality-records.jsonl"
    raw.write_bytes(raw.read_bytes().replace(b"<answer>", b"<answerX", 1))
    with pytest.raises(ValueError, match="published file hash/size"):
        case.api.replay(case.root, case.root / "plan.json", **kwargs)
    kwargs = case.seal()
    kwargs["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt SHA"):
        case.api.replay(case.root, case.root / "plan.json", **kwargs)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'])
def test_duplicate_keys_and_nonfinite_values_rejected(raw):
    with pytest.raises(ValueError):
        module("sdsc_student_quality_replay").document(raw)


def test_pinned_eos_metadata_cannot_be_replaced(case):
    case.metadata["items"][0]["text"] = '{"eos_token_id":0,"max_position_embeddings":40960}'
    with pytest.raises(ValueError, match="tokenizer metadata bytes"):
        case.replay()
