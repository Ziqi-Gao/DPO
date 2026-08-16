"""Deterministic paired signed-entailment ProofGraph generation."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.scientific_versions import GENERATOR_VERSION, LABEL_SEMANTICS
from posttrain_circuits.core.types import CounterfactualPair
from posttrain_circuits.tasks.proofgraph.parser import parse_response
from posttrain_circuits.tasks.proofgraph.renderer import render_example, render_target
from posttrain_circuits.tasks.proofgraph.schemas import Literal, ProofStep, Rule, TaskExample
from posttrain_circuits.tasks.proofgraph.verifier import closure, verify_response

CORRUPTIONS = {
    "active_support_path_swap",
    "critical_support_swap",
    "fact_truth_flip",
    "necessary_fact_replacement",
    "critical_rule_consequent_replacement",
    "critical_rule_relocation",
    "query_flip",
    "distractor_replacement",
    "alternate_proof_path_activation",
}


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


def _symbol_mapping(rng: random.Random, count: int = 96) -> dict[str, str]:
    """Randomize fixed-width surface symbols independently for every pair."""

    symbols = [f"SYM_{index:03d}" for index in range(count)]
    rng.shuffle(symbols)
    roles = [
        "query",
        "positive_support",
        "negative_support",
        "positive_alt_support",
        "negative_alt_support",
        *[f"positive_intermediate_{index:02d}" for index in range(32)],
        *[f"negative_intermediate_{index:02d}" for index in range(32)],
    ]
    return dict(zip(roles, symbols, strict=False))


def _path_specs(
    *,
    support: Literal,
    target: Literal,
    polarity: str,
    depth: int,
    structure: str,
    role_to_symbol: dict[str, str],
    suffix: str = "",
) -> list[tuple[str, tuple[Literal, ...], Literal]]:
    """Return a symmetric semantic path before randomized rule-ID assignment."""

    prefix = f"{polarity}{suffix}"
    if structure == "chain":
        specs: list[tuple[str, tuple[Literal, ...], Literal]] = []
        current = support
        for level in range(1, depth + 1):
            consequent = (
                target
                if level == depth
                else Literal(role_to_symbol[f"{polarity}_intermediate_{level:02d}"] + suffix)
            )
            specs.append((f"{prefix}_step_{level:02d}", (current,), consequent))
            current = consequent
        return specs

    left = support
    right = support
    specs = []
    if structure == "branch":
        for level in range(1, depth):
            left_next = Literal(role_to_symbol[f"{polarity}_intermediate_{level:02d}"] + suffix + "_L")
            right_next = Literal(role_to_symbol[f"{polarity}_intermediate_{level:02d}"] + suffix + "_R")
            specs.extend(
                [
                    (f"{prefix}_left_{level:02d}", (left,), left_next),
                    (f"{prefix}_right_{level:02d}", (right,), right_next),
                ]
            )
            left, right = left_next, right_next
        specs.append((f"{prefix}_merge_{depth:02d}", (left, right), target))
        return specs

    # Converging DAG: the right branch depends on both the support and the
    # first left conclusion before the branches merge. This is topologically
    # distinct from the independent branch template.
    left = Literal(role_to_symbol[f"{polarity}_intermediate_01"] + suffix + "_L")
    right = Literal(role_to_symbol[f"{polarity}_intermediate_01"] + suffix + "_R")
    specs.extend(
        [
            (f"{prefix}_left_01", (support,), left),
            (f"{prefix}_right_01", (support, left), right),
        ]
    )
    current = target if depth == 2 else Literal(role_to_symbol[f"{polarity}_intermediate_02"] + suffix)
    specs.append((f"{prefix}_merge_02", (left, right), current))
    for level in range(3, depth + 1):
        consequent = (
            target
            if level == depth
            else Literal(role_to_symbol[f"{polarity}_intermediate_{level:02d}"] + suffix)
        )
        specs.append((f"{prefix}_step_{level:02d}", (current,), consequent))
        current = consequent
    return specs


def _canonical_derivation(
    example: TaskExample,
    preferred_rule_ids: set[str] | None = None,
) -> list[ProofStep]:
    """Build a deterministic nonempty proof of the example's derived polarity."""

    target = example.query if example.label == 1 else example.query.flipped()
    derivable = closure(example)
    if target not in derivable:
        return []
    citations: dict[Literal, str] = {}
    for fact_id, literal in sorted(example.facts.items()):
        citations.setdefault(literal, fact_id)
    steps: list[ProofStep] = []
    visiting: set[Literal] = set()

    def establish(literal: Literal) -> bool:
        if literal in citations:
            return True
        if literal in visiting:
            return False
        visiting.add(literal)
        candidates = [
            rule
            for _, rule in sorted(example.rules.items())
            if rule.consequent == literal and all(item in derivable for item in rule.antecedents)
        ]
        candidates.sort(
            key=lambda rule: (
                preferred_rule_ids is not None and rule.rule_id not in preferred_rule_ids,
                rule.rule_id,
            )
        )
        for rule in candidates:
            if not all(establish(item) for item in rule.antecedents):
                continue
            step_id = f"S{len(steps) + 1:02d}"
            step = ProofStep(
                step_id=step_id,
                rule_id=rule.rule_id,
                citations=tuple(citations[item] for item in rule.antecedents),
                conclusion=literal,
            )
            steps.append(step)
            citations[literal] = step_id
            visiting.remove(literal)
            return True
        visiting.remove(literal)
        return False

    return steps if establish(target) else []


def _rule_payload(rules: dict[str, Rule]) -> list[dict[str, Any]]:
    return [asdict(rule) for rule in rules.values()]


class ProofGraphTask:
    """Generate, render, parse, verify, and corrupt exact signed proofs."""

    generator_version = GENERATOR_VERSION
    label_semantics = LABEL_SEMANTICS
    paired_generation = True
    require_exactly_one_query_polarity = True

    def generate_pair(
        self,
        seed: int,
        difficulty: dict[str, Any] | None = None,
    ) -> tuple[TaskExample, TaskExample]:
        cfg = dict(difficulty or {})
        requested_semantics = str(cfg.get("label_semantics", LABEL_SEMANTICS))
        if requested_semantics != LABEL_SEMANTICS:
            raise ValueError(f"core ProofGraph requires label_semantics={LABEL_SEMANTICS}")
        if cfg.get("paired_generation", True) is not True:
            raise ValueError("core ProofGraph requires paired_generation=true")
        if cfg.get("require_exactly_one_query_polarity", True) is not True:
            raise ValueError("core ProofGraph requires exactly one derivable query polarity")

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
        multiple_proofs = bool(cfg.get("multiple_valid_proofs", False))
        unique_proof = bool(cfg.get("unique_proof", not multiple_proofs))
        if multiple_proofs and unique_proof:
            raise ValueError("multiple_valid_proofs and unique_proof cannot both be true")

        role_to_symbol = _symbol_mapping(rng)
        query = Literal(role_to_symbol["query"])
        positive_support = Literal(role_to_symbol["positive_support"])
        negative_support = Literal(role_to_symbol["negative_support"])
        semantic_specs = [
            *_path_specs(
                support=positive_support,
                target=query,
                polarity="positive",
                depth=depth,
                structure=structure,
                role_to_symbol=role_to_symbol,
            ),
            *_path_specs(
                support=negative_support,
                target=query.flipped(),
                polarity="negative",
                depth=depth,
                structure=structure,
                role_to_symbol=role_to_symbol,
            ),
        ]
        positive_facts = {"F01": positive_support}
        negative_facts = {"F01": negative_support}
        if multiple_proofs:
            positive_alt = Literal(role_to_symbol["positive_alt_support"])
            negative_alt = Literal(role_to_symbol["negative_alt_support"])
            positive_facts["F02"] = positive_alt
            negative_facts["F02"] = negative_alt
            semantic_specs.extend(
                _path_specs(
                    support=positive_alt,
                    target=query,
                    polarity="positive",
                    depth=depth,
                    structure="chain",
                    role_to_symbol=role_to_symbol,
                    suffix="_ALT",
                )
            )
            semantic_specs.extend(
                _path_specs(
                    support=negative_alt,
                    target=query.flipped(),
                    polarity="negative",
                    depth=depth,
                    structure="chain",
                    role_to_symbol=role_to_symbol,
                    suffix="_ALT",
                )
            )

        # Random assignment prevents a global R01/polarity convention. The same
        # assignment and insertion order are reused by both variants.
        rule_ids = [f"R{index + 1:02d}" for index in range(len(semantic_specs) + distractors)]
        rng.shuffle(rule_ids)
        assigned: list[tuple[str, str, tuple[Literal, ...], Literal]] = []
        for rule_id, (role, antecedents, consequent) in zip(
            rule_ids[: len(semantic_specs)], semantic_specs, strict=True
        ):
            assigned.append((rule_id, role, antecedents, consequent))

        distractor_facts: dict[str, Literal] = {}
        fact_offset = len(positive_facts)
        for index in range(distractors):
            fact_id = f"F{fact_offset + index + 1:02d}"
            literal = Literal(f"DST_{seed:08d}_{index:03d}", bool(rng.getrandbits(1)))
            distractor_facts[fact_id] = literal
            rule_id = rule_ids[len(semantic_specs) + index]
            assigned.append(
                (
                    rule_id,
                    f"distractor_{index:03d}",
                    (literal,),
                    Literal(f"DST_RESULT_{seed:08d}_{index:03d}"),
                )
            )
        positive_facts.update(distractor_facts)
        negative_facts.update(distractor_facts)
        assigned.sort(key=lambda value: value[0])
        rules = {
            rule_id: Rule(rule_id, antecedents, consequent)
            for rule_id, _, antecedents, consequent in assigned
        }
        rule_role_mapping = {role: rule_id for rule_id, role, _, _ in assigned}
        topology = [
            {"role": role, "antecedent_count": len(antecedents), "target_negated": consequent.negated}
            for _, role, antecedents, consequent in assigned
        ]
        pair_payload = {
            "seed": seed,
            "depth": depth,
            "distractors": distractors,
            "structure": structure,
            "multiple_proofs": multiple_proofs,
            "role_to_symbol": role_to_symbol,
            "rule_role_mapping": rule_role_mapping,
            "rules": _rule_payload(rules),
            "query": asdict(query),
        }
        pair_group_id = "pgpair-" + sha256_value(pair_payload)[:20]
        pair_config = {
            "depth": depth,
            "distractors": distractors,
            "structure": structure,
            "multiple_valid_proofs": multiple_proofs,
            "unique_proof": unique_proof,
            "label_semantics": LABEL_SEMANTICS,
            "paired_generation": True,
            "require_exactly_one_query_polarity": True,
        }

        def variant(label: int, facts: dict[str, Literal]) -> TaskExample:
            metadata = {
                "seed": seed,
                "pair_seed": seed,
                "pair_group_id": pair_group_id,
                "pair_variant": "positive" if label == 1 else "negative",
                "depth": depth,
                "proof_depth": depth,
                "distractors": distractors,
                "structure": structure,
                "positive": bool(label),
                "proof_multiplicity": 2 if multiple_proofs else 1,
                "unique_proof": unique_proof,
                "generator_version": GENERATOR_VERSION,
                "label_semantics": LABEL_SEMANTICS,
                "paired_generation": True,
                "require_exactly_one_query_polarity": True,
                "role_to_symbol": role_to_symbol,
                "rule_role_mapping": rule_role_mapping,
                "rule_set_hash": sha256_value(_rule_payload(rules)),
                "topology_hash": sha256_value(topology),
                "pair_generation_config": pair_config,
                "active_support_fact_ids": ["F01", *(["F02"] if multiple_proofs else [])],
            }
            example = TaskExample(
                example_id=f"{pair_group_id}-{'pos' if label == 1 else 'neg'}",
                facts=copy.deepcopy(facts),
                rules=copy.deepcopy(rules),
                query=query,
                label=label,
                canonical_proof=[],
                pair_group_id=pair_group_id,
                metadata=metadata,
            )
            positive_derivable = query in closure(example)
            negative_derivable = query.flipped() in closure(example)
            if positive_derivable == negative_derivable or int(positive_derivable) != label:
                raise RuntimeError("paired generator failed exactly-one signed entailment")
            primary_rule_ids = {
                rule_id
                for role, rule_id in rule_role_mapping.items()
                if role.startswith("positive" if label == 1 else "negative") and "ALT" not in role
            }
            example.canonical_proof = _canonical_derivation(example, primary_rule_ids)
            if not example.canonical_proof:
                raise RuntimeError("both signed labels require a nonempty canonical proof")
            return example

        positive = variant(1, positive_facts)
        negative = variant(0, negative_facts)
        if positive.query != negative.query or positive.rules != negative.rules:
            raise RuntimeError("paired variants must share query and rules")
        if len(positive.canonical_proof) != len(negative.canonical_proof):
            raise RuntimeError("paired variants must have equal canonical proof length")
        return positive, negative

    def generate(
        self,
        seed: int,
        difficulty: dict[str, Any] | None = None,
    ) -> TaskExample:
        cfg = dict(difficulty or {})
        positive, negative = self.generate_pair(seed, cfg)
        requested = cfg.get("positive")
        if requested is None:
            requested = seed % 2 == 0
        return positive if bool(requested) else negative

    def render(self, example: TaskExample) -> str:
        return render_example(example)

    def parse_response(self, text: str):  # type: ignore[no-untyped-def]
        return parse_response(text)

    def verify(self, example: TaskExample, response):  # type: ignore[no-untyped-def]
        return verify_response(example, response)

    def canonical_target(self, example: TaskExample) -> str:
        return render_target(example)

    def _paired_sibling(self, example: TaskExample) -> TaskExample:
        sibling = copy.deepcopy(example)
        roles = dict(example.metadata["role_to_symbol"])
        sibling.label = 1 - example.label
        polarity = "positive" if sibling.label == 1 else "negative"
        sibling.facts["F01"] = Literal(roles[f"{polarity}_support"])
        if "F02" in example.metadata.get("active_support_fact_ids", []):
            sibling.facts["F02"] = Literal(roles[f"{polarity}_alt_support"])
        sibling.metadata = {
            **sibling.metadata,
            "positive": bool(sibling.label),
            "pair_variant": polarity,
        }
        sibling.example_id = f"{example.pair_group_id}-{'pos' if sibling.label == 1 else 'neg'}"
        primary_rule_ids = {
            rule_id
            for role, rule_id in sibling.metadata["rule_role_mapping"].items()
            if role.startswith(polarity) and "ALT" not in role
        }
        sibling.canonical_proof = _canonical_derivation(sibling, primary_rule_ids)
        positive_derivable = sibling.query in closure(sibling)
        negative_derivable = sibling.query.flipped() in closure(sibling)
        if positive_derivable == negative_derivable or int(positive_derivable) != sibling.label:
            raise RuntimeError("paired sibling reconstruction violated signed entailment")
        return sibling

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
        target_changing = False

        if corruption in {
            "active_support_path_swap",
            "critical_support_swap",
            "fact_truth_flip",
            "necessary_fact_replacement",
        }:
            corrupt = self._paired_sibling(example)
            changed_field = "facts.active_support"
            target_changing = True
        elif corruption == "query_flip":
            corrupt.query = example.query.flipped()
            corrupt.label = int(corrupt.query in closure(corrupt))
            corrupt.canonical_proof = _canonical_derivation(corrupt)
            changed_field = "query"
            target_changing = True
        elif corruption == "critical_rule_consequent_replacement":
            terminal = example.canonical_proof[-1]
            rule = corrupt.rules[terminal.rule_id]
            corrupt.rules[terminal.rule_id] = Rule(
                terminal.rule_id,
                rule.antecedents,
                (example.query if example.label == 0 else example.query.flipped()),
            )
            corrupt.label = 1 - example.label
            corrupt.canonical_proof = _canonical_derivation(corrupt)
            changed_field = f"rules.{terminal.rule_id}.consequent"
            target_changing = True
        elif corruption == "critical_rule_relocation":
            corrupt.rules = dict(reversed(list(corrupt.rules.items())))
            changed_field = "rule_order"
        elif corruption == "distractor_replacement":
            active = set(example.metadata.get("active_support_fact_ids", []))
            distractor_id = next(key for key in corrupt.facts if key not in active)
            corrupt.facts[distractor_id] = Literal(f"DISTRACTOR_{seed:08d}")
            changed_field = f"facts.{distractor_id}"
        else:
            expected = example.query if example.label == 1 else example.query.flipped()
            new_fact_id = f"F{len(corrupt.facts) + 1:02d}"
            new_rule_id = f"R{len(corrupt.rules) + 1:02d}"
            corrupt.facts[new_fact_id] = Literal(f"ALT_{seed:08d}")
            corrupt.rules[new_rule_id] = Rule(new_rule_id, (corrupt.facts[new_fact_id],), expected)
            corrupt.canonical_proof = _canonical_derivation(corrupt)
            changed_field = "alternate_proof_path"

        positive_derivable = corrupt.query in closure(corrupt)
        negative_derivable = corrupt.query.flipped() in closure(corrupt)
        if positive_derivable == negative_derivable:
            raise ValueError(f"{corruption} produced an ambiguous signed graph")
        corrupt.label = int(positive_derivable)
        corrupt.canonical_proof = _canonical_derivation(corrupt)
        corrupt.metadata = {
            **corrupt.metadata,
            "corruption_type": corruption,
            "corruption_class": (
                "query_routing_only"
                if corruption == "query_flip"
                else "semantic_target_changing"
                if target_changing
                else "semantic_preserving"
            ),
            "corruption_seed": seed,
            "source_example_id": example.example_id,
        }
        corrupt.example_id = f"{example.example_id}-cf-{sha256_value([corruption, seed])[:10]}"
        pair = CounterfactualPair(
            pair_id="pair-" + sha256_value([example.example_id, corruption, seed])[:16],
            clean_example=example,
            corrupt_example=corrupt,
            clean_prompt=self.render(example),
            corrupt_prompt=self.render(corrupt),
            clean_target=self.canonical_target(example),
            corrupt_target=self.canonical_target(corrupt),
            corruption_type=corruption,
            changed_semantic_field=changed_field,
        )
        if target_changing and pair.clean_target == pair.corrupt_target:
            raise ValueError(f"{corruption} failed to change the target")
        return pair
