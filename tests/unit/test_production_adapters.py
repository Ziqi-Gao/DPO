from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from posttrain_circuits.artifacts.hashing import sha256_file
from posttrain_circuits.causal_circuits.discovery.backends.mib_eap_ig import (
    GIT_EXECUTABLE,
    MIB_REVISION,
    MibEapIgAdapter,
    _validated_code_src_root,
    write_fixed_discovery_pairs,
)


class MibProductionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(os.environ.get("TMPDIR", "/scr/del6500/OPD/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mib_checkout_validation_uses_absolute_git(self) -> None:
        repository = self.root / "mib"
        repository.mkdir()
        (repository / "run_attribution.py").write_text("", encoding="utf-8")
        commands: list[list[str]] = []

        def fake_check_output(command, **_kwargs):  # type: ignore[no-untyped-def]
            commands.append(command)
            return MIB_REVISION + "\n"

        with mock.patch(
            "posttrain_circuits.causal_circuits.discovery.backends.mib_eap_ig.subprocess.check_output",
            side_effect=fake_check_output,
        ):
            MibEapIgAdapter(repository)
        self.assertTrue(commands)
        self.assertEqual(commands[0][0], GIT_EXECUTABLE)
        self.assertEqual(GIT_EXECUTABLE, "/usr/bin/git")

    def test_mib_execution_uses_project_src_without_pythonpath_injection(self) -> None:
        repository = self.root / "mib"
        repository.mkdir()
        adapter = MibEapIgAdapter.__new__(MibEapIgAdapter)
        adapter.repository = repository
        output = self.root / "output"
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        pairs = self.root / "pairs.json"
        write_fixed_discovery_pairs(
            pairs,
            [
                {
                    "pair_id": "p1",
                    "clean_prompt": "clean",
                    "corrupt_prompt": "corrupt",
                    "clean_target": "1",
                    "corrupt_target": "0",
                    "clean_input_ids": [3, 4],
                    "corrupt_input_ids": [5, 6],
                    "clean_target_ids": [4],
                    "corrupt_target_ids": [6],
                    "clean_metric_positions": [1],
                    "corrupt_metric_positions": [1],
                    "clean_intervention_positions": [0, 1],
                    "corrupt_intervention_positions": [0, 1],
                    "stage": "final_answer",
                    "semantic_pair_hash": "semantic-pair-hash",
                    "tokenized_pair_hash": "tokenized-pair-hash",
                    "semantic_manifest_hash": "semantic-manifest-hash",
                    "tokenizer_hash": "tokenizer-hash",
                }
            ],
        )
        calls: list[tuple[list[str], Path, bool]] = []
        expected_src_root = _validated_code_src_root()

        def fake_run(command, *, cwd, check):  # type: ignore[no-untyped-def]
            self.assertNotIn("PYTHONPATH", os.environ)
            calls.append((command, cwd, check))
            result_path = Path(command[command.index("--output") + 1])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "backend": "mib-eap-ig",
                        "backend_revision": adapter.backend_revision,
                        "method": adapter.method,
                        "level": "node",
                        "pair_count": 1,
                        "pair_manifest_hash": "pair-hash",
                        "bootstrap_replicates": 2,
                        "uncertainty_method": "prompt_bootstrap_standard_deviation",
                        "compatibility_hash": "compatibility-hash",
                        "graph": {
                            "nodes": {"layer.0.q": {"score": 1.5}},
                            "edges": {"input->layer.0.q": {"score": 0.4}},
                        },
                        "uncertainty": {"layer.0.q": 0.2},
                    }
                ),
                encoding="utf-8",
            )

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHONPATH", None)
            with mock.patch(
                "posttrain_circuits.causal_circuits.discovery.backends.mib_eap_ig._validated_code_src_root",
                return_value=expected_src_root,
            ), mock.patch(
                "posttrain_circuits.causal_circuits.discovery.backends.mib_eap_ig.subprocess.run",
                side_effect=fake_run,
            ):
                scores = adapter.run(
                    model="organization/model",
                    model_revision="resolved-commit",
                    checkpoint=checkpoint,
                    checkpoint_sha256=sha256_file(checkpoint),
                    level="node",
                    steps=10,
                    pair_manifest=pairs,
                    output_dir=output,
                    bootstrap_replicates=2,
                    seed=7,
                    parity_tolerance=0.02,
                )
        self.assertEqual(scores.scores, {"layer.0.q": 1.5})
        self.assertEqual(scores.node_scores, {"layer.0.q": 1.5})
        self.assertEqual(scores.edge_scores, {"input->layer.0.q": 0.4})
        self.assertEqual(scores.uncertainty, {"layer.0.q": 0.2})
        command = calls[0][0]
        self.assertEqual(
            command[1:3],
            ["-m", "posttrain_circuits.causal_circuits.model.runner"],
        )
        self.assertEqual(command[command.index("--pairs") + 1], str(pairs))
        self.assertEqual(
            command[command.index("--model-revision") + 1],
            "resolved-commit",
        )
        self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoint))
        self.assertEqual(calls[0][1:], (expected_src_root, True))


if __name__ == "__main__":
    unittest.main()
