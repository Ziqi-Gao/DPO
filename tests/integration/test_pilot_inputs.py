from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

from posttrain_circuits.cli.build_local_fork_inputs import main as build_local_fork_inputs
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.utils.smoke import build_grouped_fork_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


@pytest.mark.integration
def test_local_fork_inputs_are_mixed_multi_prompt_and_bank_hash_bound(
    monkeypatch,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    records = build_grouped_fork_bank(build_smoke_examples(2, seed=17), tokenizer, 19, group_size=4)
    for record in records:
        positions = len(record.response_ids)
        record.teacher_topk_ids = [[1, 2] for _ in range(positions)]
        record.teacher_topk_logprobs = [[-0.2, -1.7] for _ in range(positions)]
        record.teacher_topk_mass = [0.95 for _ in range(positions)]
        record.teacher_entropy = [0.4 for _ in range(positions)]
    store = TrajectoryStore(tmp_path / "bank")
    manifest = store.write(
        records,
        behavior_policy={"id": "initial", "revision": "pinned"},
        prompt_manifest_hash="prompts",
        sampling_configuration={"temperature": 1.0, "top_p": 1.0},
        verifier_version="proofgraph-exact-v1",
        teacher_version="teacher-pinned",
        top_k=2,
    )
    monkeypatch.setattr(
        "posttrain_circuits.cli.build_local_fork_inputs.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    output = tmp_path / "inputs"
    build_local_fork_inputs(
        [
            "model=tiny_qwen",
            "--trajectory-store",
            str(store.root),
            "--limit",
            "8",
            "--output",
            str(output),
        ]
    )
    prompts = json.loads((output / "prompts.json").read_text(encoding="utf-8"))
    probes = json.loads((output / "probe_set.json").read_text(encoding="utf-8"))
    assert len(prompts["prompt_ids"]) == 8
    assert len(set(prompts["prompt_ids"])) == 2
    assert prompts["trajectory_store_hash"] == manifest["sha256"]
    assert probes["trajectory_store_hash"] == manifest["sha256"]
    assert probes["prompt_ids"] == prompts["prompt_ids"]
    assert probes["trajectory_ids"] == prompts["trajectory_ids"]
    assert len(prompts["trajectory_ids"]) == len(set(prompts["trajectory_ids"])) == 8
    assert len(prompts["generation_groups"]) == 2
    assert probes["group_membership_hash"] == prompts["group_membership_hash"]
    assert len({len(row) for row in probes["input_ids"]}) == 1
