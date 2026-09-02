from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.cli.build_state_source_fork import (
    main as build_state_source_fork_main,
)
from posttrain_circuits.cli.run_local_fork import (
    main as run_local_fork_main,
)
from posttrain_circuits.core.seeding import seed_everything
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.learning.contracts import PromptBatch, TrajectoryBatch
from posttrain_circuits.learning.supervision.hard_teacher import (
    HardTeacherSupervisor,
)
from posttrain_circuits.learning.teacher.hf_scorer import (
    HuggingFaceTeacherScorer,
)
from posttrain_circuits.learning.training.local_fork import (
    _branch_checkpoint_state_hashes,
    _probe_kl,
    _validate_adamw_optimizer_contract,
    calibrate_learning_rate_for_output_kl,
    create_fork_bundle,
    load_fork_bundle,
    restore_bundle_fresh,
    state_hash,
    validate_branch_checkpoint_evidence,
    validate_branch_deterministic_replay,
    validate_checkpoint_probe_recomputation,
)
from posttrain_circuits.experiments.protocols.local_fork import validate_local_fork_report
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    build_grouped_fork_bank,
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
    records = build_grouped_fork_bank(
        build_smoke_examples(2, seed=31),
        tokenizer,
        seed=41,
        group_size=4,
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
    for record in trajectories.records:
        tokens = record.input_ids + record.response_ids
        with torch.no_grad():
            logprobs = model(input_ids=torch.tensor([tokens])).logits[0].float().log_softmax(-1)
        start = len(record.input_ids) - 1
        record.behavior_logprobs = [
            float(logprobs[start + index, token]) for index, token in enumerate(record.response_ids)
        ]
    was_training = model.training
    model.eval()
    probe_rows = [records[0].input_ids, records[4].input_ids]
    width = max(map(len, probe_rows))
    probe_ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in probe_rows])
    probe_attention_mask = torch.tensor(
        [[1] * len(row) + [0] * (width - len(row)) for row in probe_rows]
    )
    with torch.no_grad():
        probe_outputs = model(
            input_ids=probe_ids,
            attention_mask=probe_attention_mask,
        ).logits
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
        probe_input_ids=probe_ids,
        probe_attention_mask=probe_attention_mask,
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
def test_local_fork_probe_kl_is_padding_invariant_and_requires_valid_tokens() -> None:
    initial = torch.tensor([[[2.0, -1.0], [0.0, 0.0]]])
    current = torch.tensor([[[1.0, 0.0], [100.0, -100.0]]])
    mask = torch.tensor([[1, 0]])
    observed = _probe_kl(initial, current, mask)
    padded_initial = torch.cat((initial, torch.tensor([[[50.0, -50.0]]])), dim=1)
    padded_current = torch.cat((current, torch.tensor([[[-50.0, 50.0]]])), dim=1)
    padded_mask = torch.tensor([[1, 0, 0]])
    assert _probe_kl(padded_initial, padded_current, padded_mask) == pytest.approx(observed)
    with pytest.raises(ValueError, match="at least one non-padding"):
        _probe_kl(initial, current, torch.zeros_like(mask))


@pytest.mark.integration
def test_local_fork_bundle_rejects_mask_tampering_and_pre_v4_schema(tmp_path: Path) -> None:
    bundle_path = _create_populated_bundle(tmp_path / "bundle.pt")
    tampered = torch.load(bundle_path, map_location="cpu", weights_only=False)
    tampered["probe_attention_mask"][0, 0] = 0
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="payload changed"):
        load_fork_bundle(tampered_path)

    old = torch.load(bundle_path, map_location="cpu", weights_only=False)
    old["manifest"].pop("local_fork_spec_sha256")
    old_path = tmp_path / "old.pt"
    torch.save(old, old_path)
    with pytest.raises(ValueError, match="format v4"):
        load_fork_bundle(old_path)

    extra = torch.load(bundle_path, map_location="cpu", weights_only=False)
    extra["unexpected"] = True
    extra_path = tmp_path / "extra.pt"
    torch.save(extra, extra_path)
    with pytest.raises(ValueError, match="fields differ from schema"):
        load_fork_bundle(extra_path)

    restored = restore_bundle_fresh(
        load_fork_bundle(bundle_path),
        model_factory=lambda: build_tiny_qwen(7),
    )
    duplicate_records = copy.deepcopy(restored.trajectories.records)
    duplicate_records.append(copy.deepcopy(duplicate_records[0]))
    with pytest.raises(ValueError, match="duplicate/malformed trajectory_id"):
        create_fork_bundle(
            tmp_path / "duplicate-trajectory-bundle.pt",
            model=restored.model,
            optimizer=restored.optimizer,
            scheduler=restored.scheduler,
            prompts=restored.prompts,
            trajectories=TrajectoryBatch(
                duplicate_records,
                restored.trajectories.policy_version,
            ),
            probe_input_ids=restored.probe_input_ids,
            probe_attention_mask=restored.probe_attention_mask,
            pre_update_outputs=restored.pre_update_outputs,
            manifest_hashes={"bank": "duplicate-negative-test"},
        )


@pytest.mark.integration
def test_local_fork_optimizer_contract_rejects_missing_duplicate_extra_and_remapped_state(
    tmp_path: Path,
) -> None:
    payload = load_fork_bundle(_create_populated_bundle(tmp_path / "bundle.pt"))

    def validate(optimizer: dict, names: list[list[str]]) -> None:
        _validate_adamw_optimizer_contract(
            optimizer,
            names,
            model_state=payload["model"],
            model_parameter_names=payload["model_parameter_names"],
            trainable_parameter_names=payload["trainable_parameter_names"],
            context="negative-test",
        )

    optimizer = copy.deepcopy(payload["optimizer"])
    names = copy.deepcopy(payload["optimizer_parameter_names"])
    first_reference = optimizer["param_groups"][0]["params"][0]

    missing = copy.deepcopy(optimizer)
    missing["state"].pop(first_reference)
    with pytest.raises(ValueError, match="state keys differ"):
        validate(missing, names)

    extra = copy.deepcopy(optimizer)
    extra["state"][max(extra["state"]) + 1] = copy.deepcopy(
        extra["state"][first_reference]
    )
    with pytest.raises(ValueError, match="state keys differ"):
        validate(extra, names)

    duplicate = copy.deepcopy(optimizer)
    duplicate["param_groups"][0]["params"][1] = first_reference
    with pytest.raises(ValueError, match="not bijective"):
        validate(duplicate, names)

    reordered = copy.deepcopy(optimizer)
    reordered_names = copy.deepcopy(names)
    reordered["param_groups"][0]["params"][:2] = reversed(
        reordered["param_groups"][0]["params"][:2]
    )
    reordered_names[0][:2] = reversed(reordered_names[0][:2])
    with pytest.raises(ValueError, match="canonical model order"):
        validate(reordered, reordered_names)

    remapped_names = copy.deepcopy(names)
    remapped_names[0][:2] = reversed(remapped_names[0][:2])
    with pytest.raises(ValueError, match="canonical model order"):
        validate(optimizer, remapped_names)

    extra_adam_field = copy.deepcopy(optimizer)
    extra_adam_field["state"][first_reference]["unexpected"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="AdamW state fields"):
        validate(extra_adam_field, names)

    wrong_shape = copy.deepcopy(optimizer)
    wrong_shape["state"][first_reference]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="moment shape"):
        validate(wrong_shape, names)

    wrong_dtype = copy.deepcopy(optimizer)
    wrong_dtype["state"][first_reference]["exp_avg_sq"] = wrong_dtype["state"][
        first_reference
    ]["exp_avg_sq"].double()
    with pytest.raises(ValueError, match="moment dtypes"):
        validate(wrong_dtype, names)

    nonfinite = copy.deepcopy(optimizer)
    nonfinite["state"][first_reference]["exp_avg"].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate(nonfinite, names)

    wrong_step_dtype = copy.deepcopy(optimizer)
    wrong_step_dtype["state"][first_reference]["step"] = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="float32 scalar"):
        validate(wrong_step_dtype, names)


@pytest.mark.integration
def test_local_fork_bundle_publication_reuses_only_identical_state(tmp_path: Path) -> None:
    bundle_path = _create_populated_bundle(tmp_path / "bundle.pt")
    first = load_fork_bundle(bundle_path)
    _create_populated_bundle(bundle_path)
    assert load_fork_bundle(bundle_path)["manifest"] == first["manifest"]

    conflicting_path = tmp_path / "conflicting.pt"
    conflicting = copy.deepcopy(first)
    first_parameter = conflicting["model_parameter_names"][0]
    conflicting["model"][first_parameter].add_(1)
    torch.save(conflicting, conflicting_path)
    with pytest.raises(FileExistsError, match="conflicting local-fork bundle"):
        _create_populated_bundle(conflicting_path)


@pytest.mark.integration
def test_local_fork_calibration_uses_exact_registered_formula() -> None:
    expected = max(1e-8, 2e-4 * math.sqrt(0.25))
    assert calibrate_learning_rate_for_output_kl(1.0, 4.0, 2e-4) == expected
    with pytest.raises(ValueError, match="constants differ"):
        calibrate_learning_rate_for_output_kl(
            1.0,
            4.0,
            2e-4,
            maximum_scale=9.0,
        )


@pytest.mark.integration
def test_local_fork_bundle_rejects_two_sample_protocol_claim(tmp_path: Path) -> None:
    bundle_path = _create_populated_bundle(tmp_path / "bundle.pt")
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    payload["manifest"]["minimum_group_size"] = 2
    two_sample_path = tmp_path / "two-sample.pt"
    torch.save(payload, two_sample_path)
    with pytest.raises(ValueError, match="group size differs from the registered protocol"):
        load_fork_bundle(two_sample_path)


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
    first_report_bytes = output_path.read_bytes()
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
    assert output_path.read_bytes() == first_report_bytes
    with pytest.raises(ValueError, match="requires a model factory"):
        validate_local_fork_report(
            artifact,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
        )
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
        "probe_attention_mask": payload["manifest"]["probe_attention_mask_hash"],
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
            evidence = result[mode]["checkpoint_evidence"]
            pre = Path(evidence["pre"]["path"])
            post = Path(evidence["post"]["path"])
            assert pre.is_file()
            assert post.is_file()
            assert evidence["pre"]["sha256"]
            assert evidence["post"]["sha256"]
            pre_payload = torch.load(
                pre,
                map_location="cpu",
                weights_only=False,
            )
            assert pre_payload["initial_hashes"] == baseline
            assert state_hash(pre_payload["optimizer"]["state"]) == baseline["optimizer_moments"]

    post_path = Path(
        artifact["results"][0]["unmatched"]["checkpoint_evidence"]["post"]["path"]
    )
    pre_path = Path(
        artifact["results"][0]["unmatched"]["checkpoint_evidence"]["pre"]["path"]
    )
    pre_payload = torch.load(pre_path, map_location="cpu", weights_only=False)
    post_payload = torch.load(post_path, map_location="cpu", weights_only=False)
    validate_branch_deterministic_replay(
        pre_payload,
        post_payload,
        bundle_payload=payload,
        model_factory=lambda: build_tiny_qwen(7),
    )
    tampered_replayed_loss = copy.deepcopy(post_payload)
    tampered_replayed_loss["loss_evidence"]["step_losses"][0] += 1.0
    tampered_replayed_loss["loss_evidence"]["final_loss"] += 1.0
    with pytest.raises(ValueError, match="replay loss/step metrics differ"):
        validate_branch_deterministic_replay(
            pre_payload,
            tampered_replayed_loss,
            bundle_payload=payload,
            model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_group = copy.deepcopy(post_payload)
    tampered_group["optimizer"]["param_groups"][0]["weight_decay"] += 0.1
    with pytest.raises(ValueError, match="hyperparameter 'weight_decay' changed"):
        validate_branch_deterministic_replay(
            pre_payload,
            tampered_group,
            bundle_payload=payload,
            model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_moment_dtype = copy.deepcopy(post_payload)
    first_reference = tampered_moment_dtype["optimizer"]["param_groups"][0]["params"][0]
    first_state = tampered_moment_dtype["optimizer"]["state"][first_reference]
    first_state["exp_avg"] = first_state["exp_avg"].double()
    first_state["exp_avg_sq"] = first_state["exp_avg_sq"].double()
    if "max_exp_avg_sq" in first_state:
        first_state["max_exp_avg_sq"] = first_state["max_exp_avg_sq"].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_branch_deterministic_replay(
            pre_payload,
            tampered_moment_dtype,
            bundle_payload=payload,
            model_factory=lambda: build_tiny_qwen(7),
        )

    boolean_schema = copy.deepcopy(post_payload)
    boolean_schema["loss_evidence"]["schema_version"] = True
    boolean_schema["state_hashes"] = _branch_checkpoint_state_hashes(boolean_schema)
    boolean_root = tmp_path / "boolean-loss-schema"
    boolean_root.mkdir()
    boolean_path = boolean_root / "post.pt"
    torch.save(boolean_schema, boolean_path)
    boolean_evidence = {
        "path": str(boolean_path.resolve(strict=True)),
        "sha256": sha256_file(boolean_path),
        "payload_state_sha256": boolean_schema["state_hashes"]["payload"],
        "phase": "post",
        "state_hashes": boolean_schema["state_hashes"],
    }
    with pytest.raises(ValueError, match="loss evidence ancestry changed"):
        validate_branch_checkpoint_evidence(
            boolean_evidence,
            checkpoint_root=boolean_root,
            bundle_payload=payload,
            branch=str(boolean_schema["branch"]),
            horizon=int(boolean_schema["horizon"]),
            phase="post",
            comparison_mode=str(boolean_schema["comparison_mode"]),
            calibration_round=int(boolean_schema["calibration_round"]),
        )
    validate_checkpoint_probe_recomputation(
        post_payload,
        model_factory=lambda: build_tiny_qwen(7),
    )
    tampered_post_outputs = copy.deepcopy(post_payload)
    tampered_post_outputs["observed_probe_outputs"][0, 0, 0].add_(1)
    with pytest.raises(ValueError, match="recomputed probe logits differ"):
        validate_checkpoint_probe_recomputation(
            tampered_post_outputs,
            model_factory=lambda: build_tiny_qwen(7),
        )
    tampered_steps = copy.deepcopy(post_payload)
    first_step_name = next(iter(tampered_steps["optimizer_steps"]))
    tampered_steps["optimizer_steps"][first_step_name] += 1
    with pytest.raises(ValueError, match="optimizer-step evidence changed"):
        _branch_checkpoint_state_hashes(tampered_steps)

    report_with_extra = copy.deepcopy(artifact)
    report_with_extra["unexpected"] = True
    report_with_extra["sha256"] = sha256_value(
        {key: value for key, value in report_with_extra.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="fields differ from format v4"):
        validate_local_fork_report(
            report_with_extra,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_norm = copy.deepcopy(artifact)
    tampered_norm["results"][0]["unmatched"]["parameter_update_norm"] += 0.5
    tampered_norm["sha256"] = sha256_value(
        {key: value for key, value in tampered_norm.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="parameter update norm differs"):
        validate_local_fork_report(
            tampered_norm,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_kl = copy.deepcopy(artifact)
    tampered_kl["results"][0]["unmatched"]["probe_output_kl_new_to_fork"] += 0.5
    tampered_kl["sha256"] = sha256_value(
        {key: value for key, value in tampered_kl.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="probe KL differs"):
        validate_local_fork_report(
            tampered_kl,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_update_type = copy.deepcopy(artifact)
    tampered_update_type["results"][0]["unmatched"]["optimizer_updates"] = True
    tampered_update_type["sha256"] = sha256_value(
        {key: value for key, value in tampered_update_type.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="update count differs"):
        validate_local_fork_report(
            tampered_update_type,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    calibrated = next(row for row in artifact["results"] if row["calibration_trace"])
    tampered_calibration_lr = copy.deepcopy(artifact)
    tampered_cell = next(
        row
        for row in tampered_calibration_lr["results"]
        if row["branch"] == calibrated["branch"] and row["horizon"] == calibrated["horizon"]
    )
    tampered_cell["calibration_trace"][0]["learning_rate"] *= 1.5
    tampered_calibration_lr["sha256"] = sha256_value(
        {
            key: value
            for key, value in tampered_calibration_lr.items()
            if key != "sha256"
        }
    )
    with pytest.raises(ValueError, match="calibration learning rate differs"):
        validate_local_fork_report(
            tampered_calibration_lr,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_initial = copy.deepcopy(artifact)
    tampered_initial["results"][0]["matched"]["initial_parameter_hash"] = "0" * 64
    tampered_initial["sha256"] = sha256_value(
        {key: value for key, value in tampered_initial.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="initial-state evidence changed"):
        validate_local_fork_report(
            tampered_initial,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    tampered_metrics = copy.deepcopy(artifact)
    tampered_metrics["results"][0]["matched"]["step_metrics"][0]["tampered"] = float("nan")
    tampered_metrics["sha256"] = sha256_value(
        {key: value for key, value in tampered_metrics.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="loss differs|non-finite"):
        validate_local_fork_report(
            tampered_metrics,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    partial = copy.deepcopy(artifact)
    partial["results"][0]["matched"]["checkpoint_evidence"]["post"].pop("sha256")
    partial["sha256"] = sha256_value(
        {key: value for key, value in partial.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="partial or has extra fields"):
        validate_local_fork_report(
            partial,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )

    extra_checkpoint = output_path.parent / "checkpoints" / "stale-extra.pt"
    extra_checkpoint.write_bytes(b"stale")
    with pytest.raises(ValueError, match="checkpoint tree differs"):
        validate_local_fork_report(
            artifact,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )
    extra_checkpoint.unlink()

    extra_empty_directory = output_path.parent / "checkpoints" / "stale-empty-directory"
    extra_empty_directory.mkdir()
    with pytest.raises(ValueError, match="checkpoint tree differs"):
        validate_local_fork_report(
            artifact,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )
    extra_empty_directory.rmdir()

    original = artifact["results"][0]["matched"]["checkpoint_evidence"]["post"]
    replaced_path = Path(original["path"])
    replaced_payload = torch.load(replaced_path, map_location="cpu", weights_only=False)
    replaced_payload["model"][next(iter(replaced_payload["model"]))].add_(1)
    torch.save(replaced_payload, replaced_path)
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        validate_local_fork_report(
            artifact,
            checkpoint_root=output_path.parent / "checkpoints",
            bundle_path=bundle_path,
            source_model_factory=lambda: build_tiny_qwen(7),
        )


def _write_source_store(
    root: Path,
    records: list,
    source_name: str,
) -> None:
    source_records = copy.deepcopy(records)
    for record in source_records:
        record.behavior_policy_id = source_name
        record.behavior_policy_revision = "local-state-source-v1"
        record.trajectory_id = record.expected_trajectory_id
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
