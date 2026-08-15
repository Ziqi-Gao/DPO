from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from posttrain_circuits.cli.build_state_source_fork import (
    main as build_state_source_fork_main,
)
from posttrain_circuits.cli.run_local_fork import (
    main as run_local_fork_main,
)
from posttrain_circuits.core.seeding import seed_everything
from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.supervision.hard_teacher import (
    HardTeacherSupervisor,
)
from posttrain_circuits.teacher.hf_scorer import (
    HuggingFaceTeacherScorer,
)
from posttrain_circuits.training.local_fork import (
    create_fork_bundle,
    load_fork_bundle,
    state_hash,
)
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    build_smoke_examples,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _create_populated_bundle(path: Path) -> Path:
    seed_everything(1729)
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen(7)
    records = build_fixed_bank(
        build_smoke_examples(2, seed=31),
        tokenizer,
        seed=41,
    )
    trajectories = HuggingFaceTeacherScorer(
        build_tiny_qwen(8),
        teacher_id="local/fork-teacher",
        teacher_revision="fork-teacher-v1",
        top_k=8,
        minimum_retained_mass=0.0,
    ).score(TrajectoryBatch(records, policy_version=0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    prepared = HardTeacherSupervisor(tokenizer.pad_token_id).prepare_targets(trajectories, None, None)
    output = HardTeacherSupervisor(tokenizer.pad_token_id).compute_loss(model, prepared)
    output.loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()
    probe_ids = torch.tensor([records[0].input_ids])
    with torch.no_grad():
        probe_outputs = model(input_ids=probe_ids).logits
    model.train(was_training)
    prompts = PromptBatch(
        tuple(record.prompt_id for record in records),
        tuple(record.prompt_text for record in records),
    )
    create_fork_bundle(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompts=prompts,
        trajectories=trajectories,
        pre_update_outputs=probe_outputs,
        manifest_hashes={
            "task": "proofgraph-smoke",
            "bank": "fixed-bank-smoke",
            "model_revision": "local-random-v1",
        },
        model_spec={
            "model_name_or_path": "local/tiny-qwen",
            "model_revision": "local-random-v1",
            "seed": 7,
        },
    )
    return path


@pytest.mark.integration
def test_local_fork_restores_identical_complete_state_for_all_branches(
    tmp_path: Path,
) -> None:
    bundle_path = _create_populated_bundle(tmp_path / "bundle.pt")
    payload = load_fork_bundle(bundle_path)
    assert payload["optimizer"]["state"]
    assert payload["manifest"]["optimizer_moment_hash"] == state_hash(payload["optimizer"]["state"])
    output_path = tmp_path / "results.json"
    run_local_fork_main(
        [
            "--bundle",
            str(bundle_path),
            "--output",
            str(output_path),
            "--horizons",
            "1",
        ]
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(artifact["results"]) == 4
    baseline = artifact["results"][0]["unmatched"]["initial_hashes"]
    expected = {
        "model": payload["manifest"]["checkpoint_hash"],
        "optimizer": payload["manifest"]["optimizer_hash"],
        "optimizer_moments": payload["manifest"]["optimizer_moment_hash"],
        "scheduler": payload["manifest"]["scheduler_hash"],
        "rng": payload["manifest"]["rng_hash"],
        "prompts": payload["manifest"]["prompt_hash"],
        "trajectories": payload["manifest"]["trajectory_hash"],
        "probe_inputs": payload["manifest"]["probe_input_hash"],
        "probe_outputs": payload["manifest"]["probe_output_hash"],
    }
    assert baseline == expected
    for result in artifact["results"]:
        assert result["unmatched"]["initial_hashes"] == baseline
        assert result["matched"]["initial_hashes"] == baseline
        assert len(result["unmatched"]["step_losses"]) == 1
        assert len(result["matched"]["step_losses"]) == 1
        assert result["unmatched"]["parameter_update_norm"] > 0
        assert math.isfinite(result["unmatched"]["probe_output_kl_new_to_fork"])
        assert result["primary_comparison_axis"] == "matched_output_kl_new_to_fork"
        assert result["secondary_comparison_axes"] == ["update_count", "parameter_update_norm"]
        assert result["calibration"]["learning_rate"] > 0
        assert math.isfinite(result["calibration"]["matched_relative_error"])
        for mode in ("unmatched", "matched"):
            pre = Path(result[mode]["pre_checkpoint"])
            post = Path(result[mode]["post_checkpoint"])
            assert pre.is_file()
            assert post.is_file()
            pre_payload = torch.load(
                pre,
                map_location="cpu",
                weights_only=False,
            )
            assert pre_payload["initial_hashes"] == baseline
            assert state_hash(pre_payload["optimizer"]["state"]) == baseline["optimizer_moments"]


def _write_source_store(
    root: Path,
    records: list,
    source_name: str,
) -> None:
    source_records = copy.deepcopy(records)
    for record in source_records:
        record.trajectory_id = f"{source_name}-{record.trajectory_id}"
    TrajectoryStore(root).write(
        source_records,
        behavior_policy={
            "id": source_name,
            "revision": "local-state-source-v1",
        },
        prompt_manifest_hash="shared-prompts-v1",
        sampling_configuration={"temperature": 1.0, "top_p": 1.0},
        verifier_version="proofgraph-exact-v1",
        teacher_version="fork-teacher-v1",
        top_k=8,
    )


@pytest.mark.integration
def test_state_source_fork_matches_all_four_independent_stores(
    tmp_path: Path,
) -> None:
    tokenizer = build_tiny_tokenizer()
    records = build_fixed_bank(
        build_smoke_examples(2, seed=71),
        tokenizer,
        seed=81,
    )
    records = (
        HuggingFaceTeacherScorer(
            build_tiny_qwen(9),
            teacher_id="local/fork-teacher",
            teacher_revision="fork-teacher-v1",
            top_k=8,
            minimum_retained_mass=0.0,
        )
        .score(TrajectoryBatch(records, policy_version=0))
        .records
    )
    source_names = (
        "common_behavior",
        "initial_student",
        "current_fork_checkpoint",
        "teacher_policy",
    )
    paths = {}
    for name in source_names:
        paths[name] = tmp_path / name
        _write_source_store(paths[name], records, name)
    output_path = tmp_path / "state-source-fork.json"
    build_state_source_fork_main(
        [
            "--common-behavior",
            str(paths["common_behavior"]),
            "--initial-student",
            str(paths["initial_student"]),
            "--current-fork-checkpoint",
            str(paths["current_fork_checkpoint"]),
            "--teacher-policy",
            str(paths["teacher_policy"]),
            "--output",
            str(output_path),
        ]
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["matched_count_per_source"] == len(records)
    assert set(artifact["selected_trajectory_ids"]) == set(source_names)
    assert set(artifact["sources"]) == set(source_names)
    assert artifact["matching_fields"] == [
        "prompt_id",
        "response_length_bin_8",
        "verifier_reward",
        "teacher_entropy_0.1",
    ]
