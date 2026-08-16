"""Semantics-preserving ProofGraph anti-shortcut transformations and metrics."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import Literal, ProofStep, Rule, TaskExample
from posttrain_circuits.tasks.proofgraph.verifier import closure

TRANSFORMATIONS = (
    "entity_symbol_renaming",
    "fact_order_permutation",
    "rule_order_permutation",
    "surface_template_paraphrase",
    "distractor_count_ood",
)


@dataclass(frozen=True)
class AntiShortcutCase:
    source_example_id: str
    transformation: str
    example: TaskExample
    prompt: str
    semantic_hash: str


def _rename_literal(literal: Literal, mapping: dict[str, str]) -> Literal:
    return Literal(mapping[literal.atom], literal.negated)


def _renamed(example: TaskExample, seed: int) -> TaskExample:
    transformed = copy.deepcopy(example)
    atoms = sorted(
        {
            literal.atom
            for literal in (
                [*example.facts.values(), example.query]
                + [item for rule in example.rules.values() for item in rule.antecedents]
                + [rule.consequent for rule in example.rules.values()]
            )
        }
    )
    shuffled = list(atoms)
    random.Random(seed).shuffle(shuffled)
    mapping = {
        atom: f"SYM_{index:03d}_{sha256_value([seed, replacement])[:6]}"
        for index, (atom, replacement) in enumerate(zip(atoms, shuffled, strict=True))
    }
    transformed.facts = {
        fact_id: _rename_literal(literal, mapping) for fact_id, literal in example.facts.items()
    }
    transformed.rules = {
        rule_id: Rule(
            rule_id,
            tuple(_rename_literal(item, mapping) for item in rule.antecedents),
            _rename_literal(rule.consequent, mapping),
        )
        for rule_id, rule in example.rules.items()
    }
    transformed.query = _rename_literal(example.query, mapping)
    transformed.canonical_proof = [
        ProofStep(
            step.step_id,
            step.rule_id,
            step.citations,
            _rename_literal(step.conclusion, mapping),
        )
        for step in example.canonical_proof
    ]
    return transformed


def _permuted_mapping(values: dict[str, Any], seed: int) -> dict[str, Any]:
    items = list(values.items())
    random.Random(seed).shuffle(items)
    if len(items) > 1 and [key for key, _ in items] == list(values):
        items = items[1:] + items[:1]
    return dict(items)


def _with_ood_distractors(example: TaskExample, count: int, seed: int) -> TaskExample:
    if count <= int(example.metadata.get("distractors", 0)):
        raise ValueError("OOD distractor count must exceed the IID distractor count")
    transformed = copy.deepcopy(example)
    current = int(example.metadata.get("distractors", 0))
    for index in range(current, count):
        fact_id = f"F{len(transformed.facts) + 1:02d}"
        rule_id = f"R{len(transformed.rules) + 1:02d}"
        literal = Literal(f"OOD_{seed}_{index}", bool(index % 2))
        transformed.facts[fact_id] = literal
        transformed.rules[rule_id] = Rule(
            rule_id,
            (literal,),
            Literal(f"OOD_RESULT_{seed}_{index}"),
        )
    transformed.metadata["distractors"] = count
    transformed.metadata["iid_distractors"] = current
    return transformed


def _paraphrased_prompt(example: TaskExample) -> str:
    facts = "\n".join(f"- [{key}] We know {value}." for key, value in example.facts.items())
    rules = "\n".join(
        f"- [{key}] Whenever {' plus '.join(str(item) for item in rule.antecedents)} holds, "
        f"infer {rule.consequent}."
        for key, rule in example.rules.items()
    )
    return (
        "KNOWN STATEMENTS\n"
        f"{facts}\n\nINFERENCE POLICY\n{rules}\n\nDECISION\n"
        f"Can {example.query} be derived from these statements?\n\n"
        "Reply using exactly this schema:\n<proof>\n"
        "S01: R01(F01,F02) -> CONCLUSION\n</proof>\n<answer>0 or 1</answer>"
    )


def _semantic_signature(example: TaskExample) -> dict[str, Any]:
    return {
        "label": example.label,
        "query_derivable": example.query in closure(example),
        "canonical_proof_valid": ProofGraphTask()
        .verify(
            example,
            ProofGraphTask().parse_response(ProofGraphTask().canonical_target(example)),
        )
        .reward,
    }


def _assert_preserved(source: TaskExample, transformed: TaskExample) -> None:
    source_signature = _semantic_signature(source)
    transformed_signature = _semantic_signature(transformed)
    if source_signature != transformed_signature:
        raise RuntimeError(
            "anti-shortcut transformation changed proof semantics: "
            f"source={source_signature}, transformed={transformed_signature}"
        )


def build_anti_shortcut_suite(
    examples: list[TaskExample],
    *,
    seed: int,
    distractor_ood_count: int,
) -> list[AntiShortcutCase]:
    """Create all five transformations for every fixed IID example."""

    if not examples:
        raise ValueError("anti-shortcut suite requires IID examples")
    task = ProofGraphTask()
    cases: list[AntiShortcutCase] = []
    for index, source in enumerate(examples):
        case_seed = seed + index * 10_000
        renamed = _renamed(source, case_seed + 1)
        fact_order = copy.deepcopy(source)
        fact_order.facts = _permuted_mapping(fact_order.facts, case_seed + 2)
        rule_order = copy.deepcopy(source)
        rule_order.rules = _permuted_mapping(rule_order.rules, case_seed + 3)
        paraphrased = copy.deepcopy(source)
        distractor_ood = _with_ood_distractors(source, distractor_ood_count, case_seed + 5)
        transformed = {
            "entity_symbol_renaming": (renamed, task.render(renamed)),
            "fact_order_permutation": (fact_order, task.render(fact_order)),
            "rule_order_permutation": (rule_order, task.render(rule_order)),
            "surface_template_paraphrase": (paraphrased, _paraphrased_prompt(paraphrased)),
            "distractor_count_ood": (distractor_ood, task.render(distractor_ood)),
        }
        for name in TRANSFORMATIONS:
            example, prompt = transformed[name]
            _assert_preserved(source, example)
            example.metadata = {
                **example.metadata,
                "anti_shortcut_transformation": name,
                "source_example_id": source.example_id,
            }
            example.example_id = f"{source.example_id}-as-{name}-{sha256_value(prompt)[:8]}"
            cases.append(
                AntiShortcutCase(
                    source_example_id=source.example_id,
                    transformation=name,
                    example=example,
                    prompt=prompt,
                    semantic_hash=sha256_value(
                        {
                            "source": source.example_id,
                            "transformation": name,
                            "label": example.label,
                            "query_derivable": example.query in closure(example),
                        }
                    ),
                )
            )
    return cases


def evaluate_anti_shortcut_suite(
    examples: list[TaskExample],
    cases: list[AntiShortcutCase],
    predict_response: Callable[[TaskExample, str], str],
    *,
    max_shortcut_gap: float,
    model_checkpoint_hash: str,
    minimum_iid_accuracy: float = 0.01,
    minimum_transformed_accuracy: float = 0.01,
    minimum_per_transformation_accuracy: float = 0.01,
    dataset_hash: str = "unspecified",
    code_commit: str = "unavailable",
    prereg_commit: str = "unavailable",
) -> dict[str, Any]:
    """Evaluate exact-verifier accuracy and return hash-bound gate evidence."""

    if not 0.0 <= max_shortcut_gap <= 1.0:
        raise ValueError("max shortcut gap must be in [0, 1]")
    task = ProofGraphTask()

    def correct(example: TaskExample, prompt: str) -> float:
        response = predict_response(example, prompt)
        return task.verify(example, task.parse_response(response)).reward

    iid_values = [correct(example, task.render(example)) for example in examples]
    by_transformation: dict[str, list[float]] = {name: [] for name in TRANSFORMATIONS}
    for case in cases:
        by_transformation[case.transformation].append(correct(case.example, case.prompt))
    if any(not values for values in by_transformation.values()):
        raise ValueError("anti-shortcut suite must include every required transformation")

    def wilson_lower(values: list[float]) -> float:
        n = len(values)
        proportion = sum(values) / n
        z = 1.959963984540054
        denominator = 1.0 + z * z / n
        center = proportion + z * z / (2.0 * n)
        radius = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        return max(0.0, (center - radius) / denominator)

    iid_accuracy = sum(iid_values) / len(iid_values)
    accuracies = {name: sum(values) / len(values) for name, values in by_transformation.items()}
    lower_bounds = {name: wilson_lower(values) for name, values in by_transformation.items()}
    transformed_accuracy = sum(accuracies.values()) / len(accuracies)
    shortcut_gap = iid_accuracy - transformed_accuracy
    capability_passed = (
        iid_accuracy >= minimum_iid_accuracy
        and transformed_accuracy >= minimum_transformed_accuracy
        and all(value >= minimum_per_transformation_accuracy for value in accuracies.values())
    )
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(),
        "accuracy_metric": "exact_verifier_reward",
        "model_checkpoint_hash": model_checkpoint_hash,
        "iid_example_count": len(examples),
        "transformed_case_count": len(cases),
        "iid_accuracy": iid_accuracy,
        "transformed_accuracy": transformed_accuracy,
        "transformation_accuracy": accuracies,
        "transformation_accuracy_wilson95_lower": lower_bounds,
        "shortcut_gap": shortcut_gap,
        "max_shortcut_gap": max_shortcut_gap,
        "minimum_iid_accuracy": minimum_iid_accuracy,
        "minimum_transformed_accuracy": minimum_transformed_accuracy,
        "minimum_per_transformation_accuracy": minimum_per_transformation_accuracy,
        "capability_passed": capability_passed,
        "passed": shortcut_gap <= max_shortcut_gap and capability_passed,
        "dataset_hash": dataset_hash,
        "code_commit": code_commit,
        "prereg_commit": prereg_commit,
        "suite_hash": sha256_value(
            [
                {
                    "source_example_id": case.source_example_id,
                    "transformation": case.transformation,
                    "semantic_hash": case.semantic_hash,
                    "prompt": case.prompt,
                }
                for case in cases
            ]
        ),
    }
    payload["sha256"] = sha256_value(payload)
    return payload
