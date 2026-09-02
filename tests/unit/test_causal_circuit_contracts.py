from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.causal_circuits.contracts import (
    CircuitArtifact,
    CircuitMask,
    q_to_kv_head_mapping,
)
from posttrain_circuits.causal_circuits.dynamics import attribution_rank_stability
from posttrain_circuits.causal_circuits.interventions.masks import (
    layer_matched_random_mask,
    layer_size_activation_matched_random_masks,
)


def _artifact(**overrides: object) -> CircuitArtifact:
    values: dict[str, object] = {
        "run_id": "run",
        "checkpoint_id": "checkpoint",
        "task_manifest_hash": "task",
        "pair_manifest_hash": "pairs",
        "backend_version": "backend",
        "model_compatibility_hash": "compatibility",
        "node_or_edge_level": "node",
        "integrated_gradient_steps": 5,
        "ablation_baseline": "counterfactual_replacement",
        "scores": {"layer.0.a": 1.0, "layer.0.b": 0.5},
        "score_uncertainty": {"layer.0.a": 0.1, "layer.0.b": 0.2},
        "semantic_probe_manifest": {"probes": [{"probe_id": "semantic"}]},
        "tokenized_probe_manifest": {"probes": [{"probe_id": "tokenized"}]},
        "discovery_pair_manifest": {"pairs": [{"pair_id": "pair"}]},
        "probe_cohort_manifest": {"cohort": "base_capable"},
        "model_compatibility": {"passed": True},
    }
    values.update(overrides)
    return CircuitArtifact(**values)  # type: ignore[arg-type]


class CausalCircuitContractTests(unittest.TestCase):
    def test_discovery_artifact_embeds_structure_and_has_one_envelope_hash(self) -> None:
        artifact = _artifact()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "circuit.json"
            artifact.write(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        digest = payload.pop("sha256")
        self.assertEqual(digest, sha256_value(payload))
        self.assertEqual(payload["semantic_probe_manifest"]["probes"][0]["probe_id"], "semantic")
        self.assertEqual(payload["discovery_pair_manifest"]["pairs"][0]["pair_id"], "pair")
        self.assertNotIn("sha256", payload["model_compatibility"])

    def test_discovery_artifact_fails_closed_without_embedded_structure(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded structured inputs"):
            _artifact(tokenized_probe_manifest={})

    def test_spearman_uses_average_tie_ranks_and_rejects_constants(self) -> None:
        tied = attribution_rank_stability(
            {"a": 1.0, "b": 1.0, "c": 3.0},
            {"a": 1.0, "b": 2.0, "c": 3.0},
        )
        self.assertTrue(math.isclose(tied, math.sqrt(3.0) / 2.0, rel_tol=1e-12))
        with self.assertRaisesRegex(ValueError, "constant score vector"):
            attribution_rank_stability(
                {"a": 1.0, "b": 1.0},
                {"a": 2.0, "b": 3.0},
            )

    def test_gqa_mapping_is_total_divisible_and_range_checked(self) -> None:
        self.assertEqual(q_to_kv_head_mapping(8, 2), (0, 0, 0, 0, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            q_to_kv_head_mapping(7, 2)
        with self.assertRaises(ValueError):
            q_to_kv_head_mapping(0, 2)

    def test_random_controls_never_fall_back_across_layers(self) -> None:
        reference = CircuitMask(("layer.0.selected",), 0.5)
        universe = ("layer.0.selected", "layer.1.other")
        with self.assertRaisesRegex(ValueError, "layer layer.0"):
            layer_matched_random_mask(universe, reference, seed=3)
        statistics = {
            component: {"activation_size": 8.0, "activation_norm": 1.0}
            for component in universe
        }
        with self.assertRaisesRegex(ValueError, "layer layer.0"):
            layer_size_activation_matched_random_masks(
                universe,
                reference,
                statistics,
                repeats=2,
                seed=3,
            )


if __name__ == "__main__":
    unittest.main()
