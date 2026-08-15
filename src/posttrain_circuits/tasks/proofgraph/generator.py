"""Deterministic ProofGraph generation and semantic counterfactuals."""

from __future__ import annotations

import copy
import random
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.types import CounterfactualPair
from posttrain_circuits.tasks.proofgraph.parser import parse_response
from posttrain_circuits.tasks.proofgraph.renderer import (
    render_example,
    render_target,
)
from posttrain_circuits.tasks.proofgraph.schemas import (
    Literal,
    ProofStep,
    Rule,
    TaskExample,
)
from posttrain_circuits.tasks.proofgraph.verifier import (
    closure,
    verify_response,
)

CORRUPTIONS = {
    "fact_truth_flip",
    "necessary_fact_replacement",
    "critical_rule_consequent_replacement",
    "critical_rule_relocation",
    "query_flip",
    "distractor_replacement",
    "alternate_proof_path_activation",
}


def _canonical_derivation(example: TaskExample) -> list[ProofStep]:
    """Return a deterministic forward proof for a derivable non-fact query."""
    citations = {literal: fact_id for fact_id, literal in sorted(example.facts.items())}
    steps: list[ProofStep] = []
    remaining = dict(sorted(example.rules.items()))
    while remaining:
        progressed = False
        for rule_id, rule in list(remaining.items()):
            if rule.consequent in citations:
                remaining.pop(rule_id)
                continue
            if not all(item in citations for item in rule.antecedents):
                continue
            step_id = f"S{len(steps) + 1:02d}"
            step = ProofStep(
                step_id,
                rule_id,
                tuple(citations[item] for item in rule.antecedents),
                rule.consequent,
            )
            steps.append(step)
            citations[rule.consequent] = step_id
            remaining.pop(rule_id)
            progressed = True
            if rule.consequent == example.query:
                return steps
        if not progressed:
            break
    return []


def _add_rule(
    rules: dict[str, Rule],
    antecedents: tuple[Literal, ...],
    consequent: Literal,
) -> None:
    rule_id = f"R{len(rules) + 1:02d}"
    rules[rule_id] = Rule(rule_id, antecedents, consequent)


def _chain_rules(
    facts: dict[str, Literal],
    rules: dict[str, Rule],
    depth: int,
) -> None:
    current = facts["F01"]
    for level in range(1, depth + 1):
        consequent = Literal("Q" if level == depth else f"I{level:02d}")
        _add_rule(rules, (current,), consequent)
        current = consequent


def _branch_rules(
    facts: dict[str, Literal],
    rules: dict[str, Rule],
    depth: int,
) -> None:
    left = facts["F01"]
    right = facts["F02"]
    for level in range(1, depth):
        left_next = Literal(f"L{level:02d}")
        right_next = Literal(f"RPATH{level:02d}")
        _add_rule(rules, (left,), left_next)
        _add_rule(rules, (right,), right_next)
        left, right = left_next, right_next
    _add_rule(rules, (left, right), Literal("Q"))


def _converging_dag_rules(
    facts: dict[str, Literal],
    rules: dict[str, Rule],
    depth: int,
) -> None:
    left = Literal("I01")
    right = Literal("I02")
    _add_rule(rules, (facts["F01"],), left)
    _add_rule(rules, (facts["F01"], facts["F02"]), right)
    current = Literal("Q") if depth == 2 else Literal("I03")
    _add_rule(rules, (left, right), current)
    for level in range(3, depth + 1):
        consequent = Literal("Q" if level == depth else f"I{level + 1:02d}")
        _add_rule(rules, (current,), consequent)
        current = consequent


def _add_alternate_proof(
    facts: dict[str, Literal],
    rules: dict[str, Rule],
    depth: int,
) -> None:
    fact_id = f"F{len(facts) + 1:02d}"
    current = Literal("ALT00")
    facts[fact_id] = current
    for level in range(1, depth + 1):
        consequent = Literal("Q" if level == depth else f"ALT{level:02d}")
        _add_rule(rules, (current,), consequent)
        current = consequent


def _choice(
    config: dict[str, Any],
    singular: str,
    plural: str,
    default: Any,
    rng: random.Random,
) -> Any:
    if singular in config:
        return config[singular]
    values = config.get(plural)
    if values is None:
        return default
    if isinstance(values, list) and len(values) == 2 and all(isinstance(value, int) for value in values):
        return rng.randint(int(values[0]), int(values[1]))
    if isinstance(values, list) and values:
        return rng.choice(values)
    raise ValueError(f"{plural} must be a non-empty list")


class ProofGraphTask:
    """Generate, render, parse, verify, and corrupt exact symbolic proofs."""

    generator_version = "proofgraph-v2"

    def generate(
        self,
        seed: int,
        difficulty: dict[str, Any] | None = None,
    ) -> TaskExample:
        cfg = dict(difficulty or {})
        rng = random.Random(seed)
        depth = int(_choice(cfg, "depth", "depth_range", 2, rng))
        if not 1 <= depth <= 7:
            raise ValueError("depth must be between 1 and 7")
        structure = str(_choice(cfg, "structure", "structures", "chain", rng))
        if structure not in {"chain", "branch", "converging_dag"}:
            raise ValueError(f"unsupported structure: {structure}")
        if depth == 1 and structure != "chain":
            raise ValueError("branch and converging_dag require depth >= 2")
        distractors = int(_choice(cfg, "distractors", "distractor_range", 2, rng))
        positive = bool(cfg.get("positive", seed % 2 == 0))
        multiple_proofs = bool(cfg.get("multiple_valid_proofs", False))
        unique_proof = bool(cfg.get("unique_proof", not multiple_proofs))
        if multiple_proofs and unique_proof:
            raise ValueError("multiple_valid_proofs and unique_proof cannot both be true")

        atoms = [atom for atom in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if atom != "Q"]
        rng.shuffle(atoms)
        facts: dict[str, Literal] = {
            "F01": Literal(atoms[0]),
            "F02": Literal(atoms[1]),
        }
        rules: dict[str, Rule] = {}
        if structure == "chain":
            _chain_rules(facts, rules, depth)
        elif structure == "branch":
            _branch_rules(facts, rules, depth)
        else:
            _converging_dag_rules(facts, rules, depth)
        if multiple_proofs:
            _add_alternate_proof(facts, rules, depth)

        for offset in range(distractors):
            fact_id = f"F{len(facts) + 1:02d}"
            fact = Literal(f"D{offset:02d}", bool(rng.getrandbits(1)))
            facts[fact_id] = fact
            _add_rule(rules, (fact,), Literal(f"X{offset:02d}"))

        query = Literal("Q") if positive else Literal("UNPROVABLE")
        semantic = {
            "seed": seed,
            "depth": depth,
            "distractors": distractors,
            "structure": structure,
            "positive": positive,
            "proof_multiplicity": 2 if multiple_proofs else 1,
            "unique_proof": unique_proof,
        }
        example = TaskExample(
            example_id="",
            facts=facts,
            rules=rules,
            query=query,
            label=int(positive),
            canonical_proof=[],
            metadata=semantic,
        )
        if positive:
            example.canonical_proof = _canonical_derivation(example)
            if not example.canonical_proof:
                raise RuntimeError("generated positive graph has no proof")
        example.example_id = f"pg-{sha256_value(semantic)[:16]}"
        return example

    def render(self, example: TaskExample) -> str:
        return render_example(example)

    def parse_response(self, text: str):  # type: ignore[no-untyped-def]
        return parse_response(text)

    def verify(self, example: TaskExample, response):  # type: ignore[no-untyped-def]
        return verify_response(example, response)

    def canonical_target(self, example: TaskExample) -> str:
        return render_target(example)

    def make_counterfactual(
        self,
        example: TaskExample,
        corruption: str,
        seed: int,
    ) -> CounterfactualPair:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {corruption!r}; choose from {sorted(CORRUPTIONS)}")
        corrupt = copy.deepcopy(example)
        changed_field = ""
        critical_step = example.canonical_proof[0] if example.canonical_proof else None

        if corruption == "query_flip":
            corrupt.query = Literal("UNPROVABLE") if example.label else Literal("Q")
            changed_field = "query"
        elif corruption in {
            "fact_truth_flip",
            "necessary_fact_replacement",
        }:
            fact_id = critical_step.citations[0] if critical_step else next(iter(corrupt.facts))
            old = corrupt.facts[fact_id]
            corrupt.facts[fact_id] = (
                old.flipped() if corruption == "fact_truth_flip" else Literal(f"REPLACED_{seed}")
            )
            changed_field = f"facts.{fact_id}"
        elif corruption == "critical_rule_consequent_replacement":
            rule_id = critical_step.rule_id if critical_step else next(iter(corrupt.rules))
            rule = corrupt.rules[rule_id]
            corrupt.rules[rule_id] = Rule(
                rule_id,
                rule.antecedents,
                Literal(f"BROKEN_{seed}"),
            )
            changed_field = f"rules.{rule_id}.consequent"
        elif corruption == "critical_rule_relocation":
            items = list(corrupt.rules.items())
            items.reverse()
            corrupt.rules = dict(items)
            changed_field = "rule_order"
        elif corruption == "distractor_replacement":
            distractor_id = next(
                (key for key, value in corrupt.facts.items() if value.atom.startswith("D")),
                None,
            )
            if distractor_id is None:
                distractor_id = f"F{len(corrupt.facts) + 1:02d}"
            corrupt.facts[distractor_id] = Literal(f"DISTRACTOR_{seed}")
            changed_field = f"facts.{distractor_id}"
        else:
            new_fact_id = f"F{len(corrupt.facts) + 1:02d}"
            new_rule_id = f"R{len(corrupt.rules) + 1:02d}"
            corrupt.facts[new_fact_id] = Literal(f"ALT_{seed}")
            corrupt.rules[new_rule_id] = Rule(
                new_rule_id,
                (corrupt.facts[new_fact_id],),
                corrupt.query,
            )
            changed_field = "alternate_proof_path"

        derivable = corrupt.query in closure(corrupt)
        corrupt.label = int(derivable)
        if not derivable:
            corrupt.canonical_proof = []
        else:
            corrupt.canonical_proof = _canonical_derivation(corrupt)
        corrupt.metadata = {
            **corrupt.metadata,
            "corruption_type": corruption,
            "seed": seed,
        }
        corrupt.example_id = f"{example.example_id}-cf-{sha256_value(corrupt.metadata)[:10]}"
        pair = CounterfactualPair(
            pair_id=("pair-" + sha256_value([example.example_id, corruption, seed])[:16]),
            clean_example=example,
            corrupt_example=corrupt,
            clean_prompt=self.render(example),
            corrupt_prompt=self.render(corrupt),
            clean_target=self.canonical_target(example),
            corrupt_target=self.canonical_target(corrupt),
            corruption_type=corruption,
            changed_semantic_field=changed_field,
        )
        if (
            corruption
            in {
                "fact_truth_flip",
                "necessary_fact_replacement",
                "critical_rule_consequent_replacement",
                "query_flip",
            }
            and pair.clean_target == pair.corrupt_target
        ):
            raise ValueError(f"{corruption} failed to change the target")
        return pair
