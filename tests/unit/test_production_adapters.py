from __future__ import annotations

import json
from pathlib import Path

import pytest

from posttrain_circuits.circuits.mib_eap_ig import (
    MibEapIgAdapter,
    write_fixed_discovery_pairs,
)
from posttrain_circuits.core.hashing import sha256_file


@pytest.mark.unit
def test_mib_execution_uses_fixed_pairs_and_complete_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    repository = tmp_path / "mib"
    repository.mkdir()
    adapter = MibEapIgAdapter.__new__(MibEapIgAdapter)
    adapter.repository = repository
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    pairs = tmp_path / "pairs.json"
    write_fixed_discovery_pairs(
        pairs,
        [
            {
                "pair_id": "p1",
                "clean_prompt": "clean",
                "corrupt_prompt": "corrupt",
                "clean_target": "1",
                "corrupt_target": "0",
            }
        ],
    )
    calls = []

    def fake_run(command, *, cwd, check):  # type: ignore[no-untyped-def]
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
                    "uncertainty_method": ("prompt_bootstrap_standard_error"),
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

    monkeypatch.setattr(
        "posttrain_circuits.circuits.mib_eap_ig.subprocess.run",
        fake_run,
    )
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
    assert scores.scores == {"layer.0.q": 1.5}
    assert scores.node_scores == {"layer.0.q": 1.5}
    assert scores.edge_scores == {"input->layer.0.q": 0.4}
    assert scores.uncertainty == {"layer.0.q": 0.2}
    command = calls[0][0]
    assert command[1:3] == [
        "-m",
        "posttrain_circuits.circuits.mib_runner",
    ]
    assert command[command.index("--pairs") + 1] == str(pairs)
    assert command[command.index("--model-revision") + 1] == ("resolved-commit")
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert calls[0][1:] == (repository, True)
