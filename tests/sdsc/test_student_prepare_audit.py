"""Real tokenizer/parser/verifier replay and corruption rejection, without models."""

import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.experiments.protocols import student_preparation as protocol
from posttrain_circuits.models.prompt_protocol import format_model_prompt

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("sdsc_student_prepare_audit")
producer = module("sdsc_student_quality_probe")


@pytest.fixture(scope="module")
def fit_fixture():
    # The optional local tokenizer bytes have independent external hashes. Skip
    # in a clean CI checkout rather than downloading a mutable Hub snapshot.
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    tokenizer = audit.load_tokenizer()
    examples = protocol.make_examples(protocol.DIFFICULTY)["student_dev"]
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": protocol.CHAT_TEMPLATE_SHA256,
        }
    }
    task, prompts, records, development = ProofGraphTask(), [], [[], []], []
    steps = (4, 8, 16, 32)
    checkpoints = [dict(step=step, sha256=str(index + 1) * 64) for index, step in enumerate(steps)]
    for ordinal, example in enumerate(examples):
        text = format_model_prompt(task.render(example), tokenizer, config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append(dict(ordinal=ordinal, example=asdict(example), prompt_ids=ids))
        for step, successes, checkpoint in zip(steps, (60, 220, 400, 500), checkpoints, strict=True):
            answer = task.canonical_target(example) if ordinal < successes else ""
            tokens = [*tokenizer.encode(answer, add_special_tokens=False), tokenizer.eos_token_id]
            row = producer.response_record(
                example, tokens, tokenizer, prompt_ids=ids, prompt_text=text, cap=256
            )
            row.update(
                step=step,
                ordinal=ordinal,
                rank=ordinal % 2,
                checkpoint_sha256=checkpoint["sha256"],
                cohort="student_dev",
                parsed_trace=asdict(task.parse_response(answer)),
            )
            records[ordinal % 2].append(row)
    for rank in records:
        rank.sort(key=lambda row: (row["step"], row["ordinal"]))
    for step, successes, checkpoint in zip(steps, (60, 220, 400, 500), checkpoints, strict=True):
        development.append(
            dict(
                step=step,
                checkpoint_sha256=checkpoint["sha256"],
                examples_sha256=protocol.EXAMPLES_SHA256["student_dev"],
                num_examples=512,
                answer_correct=successes,
                proof_correct=successes,
                format_valid=successes,
            )
        )
    report = dict(
        mode="fit",
        passed=True,
        checkpoints=checkpoints,
        development=development,
        selected_checkpoint=development[0],
        **dict.fromkeys(audit.FLAGS, False),
    )
    return report, prompts, records, tokenizer


def test_actual_tokenizer_all_four_development_sets_replay(fit_fixture):
    result = audit.replay(*fit_fixture)
    assert result["responses_replayed"] == 2048
    assert result["selected_checkpoint"]["step"] == 4
    assert result["gpu_numerics_independently_recomputed"] is False
    assert result["formal_initial_accepted"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "verification",
        "tokens",
        "checkpoint",
        "rank",
        "missing",
        "population",
        "later_selection",
        "overclaim",
        "summary",
    ],
)
def test_complete_raw_evidence_rejects_corruption(fit_fixture, corruption):
    report, prompts, records, tokenizer = fit_fixture
    report, prompts, records = copy.deepcopy((report, prompts, records))
    row = records[0][0]
    if corruption == "verification":
        row["verification"]["answer_correct"] = False
    elif corruption == "tokens":
        row["response_ids"][0] = 0
    elif corruption == "checkpoint":
        row["checkpoint_sha256"] = "f" * 64
    elif corruption == "rank":
        row["rank"] = 1
    elif corruption == "missing":
        records[1].pop()
    elif corruption == "population":
        prompts[0]["example"]["label"] = 1 - prompts[0]["example"]["label"]
    elif corruption == "later_selection":
        report["selected_checkpoint"] = report["development"][1]
    elif corruption == "overclaim":
        report["formal_initial_accepted"] = True
    elif corruption == "summary":
        report["development"][0]["answer_correct"] += 1
    with pytest.raises(ValueError):
        audit.replay(report, prompts, records, tokenizer)


def test_json_duplicate_and_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        audit.document(b'{"passed":false,"passed":true}')
    with pytest.raises(ValueError):
        audit.document(b'{"loss":NaN}')
