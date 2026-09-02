from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.datasets.circuit_probes.cohorts import (
    build_probe_cohort_manifest,
    family_probe_pairs,
    group_complete_pairs,
    load_probe_examples,
    validate_probe_cohort_manifest,
    write_probe_cohort_manifest,
)
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.manifests import write_dataset_family
from posttrain_circuits.datasets.proofgraph.splits import SPLITS, build_all_splits


def _ancestry() -> list[dict[str, str]]:
    return [
        {
            "calibration_checkpoint_sha256": "a" * 64,
            "calibration_run_manifest_sha256": "b" * 64,
            "calibration_run_id": "calibration-run",
            "experiment_binding_sha256": "c" * 64,
        }
    ]


class ProbeCohortContractTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        root = Path(self.temporary.name) / "family"
        splits = build_all_splits(
            ProofGraphTask(),
            split_sizes={split: 8 for split in SPLITS},
            base_seed=71,
            difficulty={},
        )
        write_dataset_family(root, splits)
        self.family = load_dataset_family(root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _scores(
        self,
        *,
        limit: int = 3,
    ) -> dict[str, dict[str, bool]]:
        scores: dict[str, dict[str, bool]] = {}
        states = ((True, True), (False, True), (False, False))
        for subset in ("discovery", "validation"):
            for index, pair in enumerate(
                family_probe_pairs(self.family, subset=subset, limit_pairs=limit)
            ):
                initial, learned = states[index]
                for example in pair.examples:
                    scores[example.example_id] = {
                        "initial_correct": initial,
                        "learnable_after_post_training": learned,
                    }
        return scores

    def _manifest(self) -> dict[str, object]:
        return build_probe_cohort_manifest(
            self.family,
            self._scores(),
            initial_student_checkpoint_hash="d" * 64,
            scoring_manifest_hash="e" * 64,
            eligibility_evidence_ancestry=_ancestry(),
            limit_pairs_per_split=3,
        )

    def test_pair_limit_population_exclusions_and_single_hash(self) -> None:
        manifest = self._manifest()
        for subset in ("discovery", "validation"):
            audit = manifest["candidate_selection"][subset]  # type: ignore[index]
            self.assertEqual(audit["candidate_pair_count"], 3)
            self.assertEqual(audit["selected_pair_count"], 2)
            self.assertEqual(audit["excluded_pair_count"], 1)
            self.assertEqual(len(audit["ordered_pair_population"]), 3)
            self.assertEqual(
                audit["exclusions"][0]["reason"],
                "initially_unsolved_and_not_learned_by_frozen_calibration",
            )
        self.assertEqual(manifest["confirmatory_training_ancestry"], [])
        self.assertEqual(manifest["eligibility_evidence_ancestry"], _ancestry())

        nested_sha_keys: list[str] = []

        def visit(value: object, *, root: bool = False) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "sha256" and not root:
                        nested_sha_keys.append(key)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(manifest, root=True)
        self.assertEqual(nested_sha_keys, [])

    def test_score_ids_must_equal_complete_limited_population(self) -> None:
        missing = self._scores()
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "exact candidate population"):
            build_probe_cohort_manifest(
                self.family,
                missing,
                initial_student_checkpoint_hash="d" * 64,
                scoring_manifest_hash="e" * 64,
                eligibility_evidence_ancestry=_ancestry(),
                limit_pairs_per_split=3,
            )
        extra = self._scores()
        extra["not-a-candidate"] = {
            "initial_correct": True,
            "learnable_after_post_training": True,
        }
        with self.assertRaisesRegex(ValueError, "extra"):
            build_probe_cohort_manifest(
                self.family,
                extra,
                initial_student_checkpoint_hash="d" * 64,
                scoring_manifest_hash="e" * 64,
                eligibility_evidence_ancestry=_ancestry(),
                limit_pairs_per_split=3,
            )

    def test_mixed_sibling_evidence_fails_closed(self) -> None:
        scores = self._scores()
        pair = family_probe_pairs(
            self.family,
            subset="discovery",
            limit_pairs=1,
        )[0]
        scores[pair.examples[1].example_id] = {
            "initial_correct": False,
            "learnable_after_post_training": True,
        }
        with self.assertRaisesRegex(ValueError, "mixed sibling evidence"):
            build_probe_cohort_manifest(
                self.family,
                scores,
                initial_student_checkpoint_hash="d" * 64,
                scoring_manifest_hash="e" * 64,
                eligibility_evidence_ancestry=_ancestry(),
                limit_pairs_per_split=3,
            )

    def test_incomplete_pair_and_empty_eligibility_ancestry_are_rejected(self) -> None:
        examples = list(self.family.examples("circuit_discovery"))
        with self.assertRaisesRegex(ValueError, "exactly one sibling"):
            group_complete_pairs(examples[:-1], subset="discovery")
        with self.assertRaisesRegex(ValueError, "eligibility_evidence_ancestry"):
            build_probe_cohort_manifest(
                self.family,
                self._scores(),
                initial_student_checkpoint_hash="d" * 64,
                scoring_manifest_hash="e" * 64,
                eligibility_evidence_ancestry=[],
                limit_pairs_per_split=3,
            )

    def test_writer_and_loader_preserve_complete_pairs(self) -> None:
        output = Path(self.temporary.name) / "cohort"
        manifest = self._manifest()
        write_probe_cohort_manifest(output, manifest)
        self.assertEqual([path.name for path in output.iterdir()], ["manifest.json"])
        examples, validated = load_probe_examples(
            output / "manifest.json",
            cohort="base_capable",
            subset="discovery",
            expected_manifest_hash=str(manifest["sha256"]),
        )
        self.assertEqual(validated["sha256"], manifest["sha256"])
        pairs = group_complete_pairs(examples, subset="discovery")
        self.assertEqual(len(pairs), 1)
        self.assertEqual({example.label for example in pairs[0].examples}, {0, 1})

        tampered = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        example = tampered["cohorts"]["base_capable"]["discovery"]["pairs"][0][
            "examples"
        ][0]
        example["example_id"] = str(example["example_id"]) + "-tampered"
        atomic_write_json(output / "manifest.json", tampered)
        with self.assertRaisesRegex(ValueError, "top-level manifest hash mismatch"):
            validate_probe_cohort_manifest(output / "manifest.json")

    def test_discovery_source_uses_complete_pairs_and_no_row_dedup_helper(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[2]
            / "src/posttrain_circuits/cli/discover_circuit.py"
        )
        source = source_path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("group_complete_pairs", source)
        self.assertNotIn("_all_unique_pair_examples", source)
        self.assertNotIn("deserialize_example", source)


if __name__ == "__main__":
    unittest.main()
