from __future__ import annotations

import copy
import re
from dataclasses import asdict

import pytest

from posttrain_circuits.datasets.proofgraph.anti_shortcut import _paraphrased_prompt
from posttrain_circuits.datasets.proofgraph.contracts import Literal, ProofStep, Rule, TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import (
    RESPONSE_FORMAT_INSTRUCTIONS,
    render_example,
    render_target,
)


@pytest.mark.unit
@pytest.mark.parametrize("structure", ["chain", "branch", "converging_dag"])
@pytest.mark.parametrize("positive", [False, True])
def test_compact_prompt_preserves_the_complete_ordered_graph(structure: str, positive: bool) -> None:
    task = ProofGraphTask()
    example = task.generate(
        42,
        {"structure": structure, "depth": 4, "positive": positive, "distractors": 16},
    )
    before = asdict(example)
    prompt = render_example(example)
    assert prompt.endswith("\n\n" + RESPONSE_FORMAT_INSTRUCTIONS)
    graph = prompt.removesuffix("\n\n" + RESPONSE_FORMAT_INSTRUCTIONS)
    facts_text, rules_text, query_text = graph.removeprefix("FACTS\n").split("\n\n")
    recovered_facts = [
        (key, Literal.parse(value)) for key, value in (line.split(" ", 1) for line in facts_text.splitlines())
    ]
    recovered_rules = []
    for line in rules_text.removeprefix("RULES\n").splitlines():
        key, expression = line.split(" ", 1)
        antecedents, consequent = expression.split(" -> ")
        recovered_rules.append(
            (
                key,
                Rule(
                    key,
                    tuple(Literal.parse(item) for item in antecedents.split(" AND ")),
                    Literal.parse(consequent),
                ),
            )
        )
    assert recovered_facts == list(example.facts.items())
    assert recovered_rules == list(example.rules.items())
    assert Literal.parse(query_text.removeprefix("QUERY\n")) == example.query
    assert asdict(example) == before
    assert task.verify(example, task.parse_response(render_target(example))).reward == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("renderer", [render_example, _paraphrased_prompt])
def test_prompt_never_uses_hidden_label_or_reference_proof(renderer) -> None:  # type: ignore[no-untyped-def]
    example = ProofGraphTask().generate(42, {"depth": 3, "positive": True})
    mutated = copy.deepcopy(example)
    mutated.label = 1 - example.label
    mutated.canonical_proof = [ProofStep("S99", "R99", ("SECRET_REFERENCE",), Literal("SECRET_TARGET"))]
    mutated.metadata = {"oracle": "SECRET_TARGET"}
    mutated.example_id = "SECRET_ID"
    assert renderer(mutated) == renderer(example)
    assert renderer(example).endswith(RESPONSE_FORMAT_INSTRUCTIONS)


def _signed_conjunction_example(positive: bool) -> TaskExample:
    conclusion = Literal("Y", negated=not positive)
    return TaskExample(
        example_id="schema-fixture",
        facts={"F01": Literal("A"), "F02": Literal("B", True)},
        rules={
            "R01": Rule("R01", (Literal("A"), Literal("B", True)), Literal("X")),
            "R02": Rule("R02", (Literal("X"),), conclusion),
        },
        query=Literal("Y"),
        label=int(positive),
        canonical_proof=[
            ProofStep("S01", "R01", ("F01", "F02"), Literal("X")),
            ProofStep("S02", "R02", ("S01",), conclusion),
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("positive", [False, True])
def test_actual_rule_consequents_support_signed_conjunction_and_earlier_steps(positive: bool) -> None:
    task = ProofGraphTask()
    example = _signed_conjunction_example(positive)
    # Following the instructions copies actual consequents, not the old fake
    # schema's TRUE X. Bare positive literals are already valid parser inputs;
    # the frozen canonical targets retain their original explicit TRUE form.
    response = (
        f"<proof>\nS01: R01(F01,F02) -> {example.rules['R01'].consequent}\n"
        f"S02: R02(S01) -> {example.rules['R02'].consequent}\n"
        f"</proof>\n<answer>{int(positive)}</answer>"
    )
    signed = "TRUE Y" if positive else "NOT Y"
    assert render_target(example) == (
        f"<proof>\nS01: R01(F01,F02) -> TRUE X\nS02: R02(S01) -> {signed}\n"
        f"</proof>\n<answer>{int(positive)}</answer>"
    )
    result = task.verify(example, task.parse_response(response))
    assert result.reward == 1.0
    assert [step.established for step in result.step_results] == [
        Literal("X"),
        example.canonical_proof[-1].conclusion,
    ]


@pytest.mark.unit
def test_instructions_do_not_supply_an_invented_rule_application_to_copy() -> None:
    # Real teacher job 54345715 copied the former concrete schema line in
    # 1,468/2,048 attempts. Instructions must not supply real-looking fact/rule
    # IDs or a fabricated consequent, even when a task happens to use X.
    assert re.search(r"\b[FR]\d+\b", RESPONSE_FORMAT_INSTRUCTIONS) is None
    assert "TRUE X" not in RESPONSE_FORMAT_INSTRUCTIONS
    assert "NOT X" not in RESPONSE_FORMAT_INSTRUCTIONS
    assert not ProofGraphTask().parse_response(RESPONSE_FORMAT_INSTRUCTIONS).parse_valid


@pytest.mark.unit
def test_observed_copied_schema_is_rejected_even_before_an_otherwise_valid_proof() -> None:
    task = ProofGraphTask()
    example = task.generate(42, {"structure": "chain", "depth": 4, "positive": True, "distractors": 4})
    valid = render_target(example)
    # The actual failed first attempt started with this invented line; keeping
    # a correct proof later must not turn it into an accepted demonstration.
    copied = valid.replace("<proof>\n", "<proof>\nS01: R01(F01,F02) -> TRUE X\n", 1)
    result = task.verify(example, task.parse_response(copied))
    assert result.reward == 0.0
    assert result.error_code == "antecedent_mismatch"


@pytest.mark.unit
@pytest.mark.parametrize("positive", [False, True])
def test_proof_must_stop_at_the_proved_query_polarity(positive: bool) -> None:
    task = ProofGraphTask()
    example = _signed_conjunction_example(positive)
    target = render_target(example)
    # A valid, irrelevant final step is not a proof ending at the query;
    # proving the correct polarity does not license asserting the opposite.
    continued = target.replace("</proof>", "S03: R01(F01,F02) -> TRUE X\n</proof>")
    result = task.verify(example, task.parse_response(continued))
    assert result.reward == 0.0
    assert result.error_code == "final_conclusion_mismatch"
    contradictory = target.replace(
        "</proof>",
        f"S03: R02(S01) -> {example.rules['R02'].consequent.flipped()}\n</proof>",
    )
    result = task.verify(example, task.parse_response(contradictory))
    assert result.reward == 0.0
    assert result.error_code == "conclusion_mismatch"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("S01: F01 -> TRUE A", "step_syntax"),
        ("First, derive X from A and NOT B.", "step_syntax"),
        ("S01: R01(S02,F02) -> TRUE X", "unknown_citation"),
        ("S01: R01(F01,F01) -> TRUE X", "antecedent_mismatch"),
    ],
)
def test_observed_teacher_failure_forms_remain_rejected(replacement: str, error: str) -> None:
    task = ProofGraphTask()
    example = _signed_conjunction_example(True)
    target = render_target(example)
    malformed = target.replace(target.splitlines()[1], replacement, 1)
    result = task.verify(example, task.parse_response(malformed))
    assert result.reward == 0.0
    assert result.error_code == error
    assert not task.parse_response("Reasoning:\n" + target).parse_valid
