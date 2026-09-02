from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    extend_checkpoint_ancestry,
    validate_native_trainer_checkpoint_files,
)
from posttrain_circuits.artifacts.io import publish_path_no_clobber
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.cli.finalize_pilot_training import _recompute_grpo_update_norm
from posttrain_circuits.cli.run_grpo import (
    _native_checkpoint_preflight_report,
    _native_trainer_checkpoint_path,
    _snapshot_parameter_state,
    _validate_distributed_training_consensus,
    _validate_grpo_resume_state_directory,
    _validate_native_checkpoint_preflight_reports,
    main as run_grpo_main,
)
from posttrain_circuits.learning.training.grpo_backend import (
    GrpoSettings,
    TrlGrpoBackend,
    require_pinned_trl_version,
)
from posttrain_circuits.learning.training.grpo_data import build_grpo_rows_and_reward
from posttrain_circuits.utils.smoke import build_smoke_examples


@pytest.mark.unit
def test_native_trainer_checkpoint_preflight_is_collective_and_no_clobber(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    run_root.mkdir()
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(run_root)))
    target = _native_trainer_checkpoint_path(
        trainer,
        output=run_root,
        global_step=3,
    )
    reports = [
        _native_checkpoint_preflight_report(
            target,
            expected_run_root=run_root,
            rank=rank,
        )
        for rank in range(2)
    ]
    target.mkdir()
    assert _validate_native_checkpoint_preflight_reports(
        reports,
        world_size=2,
    ) == target.absolute()
    staged = run_root / ".checkpoint-3-stage"
    staged.mkdir()
    (staged / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_path_no_clobber(staged, target)
    assert (target).is_dir()
    assert (staged / "trainer_state.json").is_file()

    metadata_parent = run_root / "checkpoints"
    metadata_parent.mkdir()
    metadata = metadata_parent / "step-00000005.pt"
    native = run_root / "checkpoint-5"
    two_target_reports = [
        _native_checkpoint_preflight_report(
            native,
            expected_run_root=run_root,
            rank=rank,
            metadata_path=metadata,
        )
        for rank in range(2)
    ]
    assert _validate_native_checkpoint_preflight_reports(
        two_target_reports,
        world_size=2,
    ) == {
        "metadata": metadata.absolute(),
        "native_state": native.absolute(),
    }
    metadata.write_bytes(b"injected-after-preflight")
    staged_metadata = metadata_parent / ".metadata-stage"
    staged_metadata.write_bytes(b"new-metadata")
    with pytest.raises(FileExistsError):
        publish_path_no_clobber(staged_metadata, metadata)
    assert metadata.read_bytes() == b"injected-after-preflight"

    existing = run_root / "checkpoint-4"
    existing.mkdir()
    existing_reports = [
        _native_checkpoint_preflight_report(
            existing,
            expected_run_root=run_root,
            rank=rank,
        )
        for rank in range(2)
    ]
    with pytest.raises(RuntimeError, match="preflight failed"):
        _validate_native_checkpoint_preflight_reports(
            existing_reports,
            world_size=2,
        )


@pytest.mark.unit
def test_native_trainer_inventory_is_exact_and_directory_components_cannot_spoof() -> None:
    digest = "a" * 64
    complete = {
        "trainer_state.json": digest,
        "model.safetensors": digest,
        "optimizer.pt": digest,
        "scheduler.pt": digest,
        "rng_state.pth": digest,
        "training_args.bin": digest,
    }
    assert validate_native_trainer_checkpoint_files(complete, world_size=1) == complete
    for missing in (
        "trainer_state.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        broken = dict(complete)
        broken.pop(missing)
        with pytest.raises(ValueError, match="incomplete|ambiguous"):
            validate_native_trainer_checkpoint_files(broken, world_size=1)

    with pytest.raises(ValueError, match="unexpected files"):
        validate_native_trainer_checkpoint_files(
            {**complete, "optimizer.pt/not-state.bin": digest},
            world_size=1,
        )
    with pytest.raises(ValueError, match="incomplete|ambiguous"):
        validate_native_trainer_checkpoint_files(
            {**complete, "optimizer.bin": digest},
            world_size=1,
        )
    with pytest.raises(ValueError, match="unexpected files"):
        validate_native_trainer_checkpoint_files(
            {**complete, "junk.bin": digest},
            world_size=1,
        )
    with pytest.raises(ValueError, match="canonical"):
        validate_native_trainer_checkpoint_files(
            {**complete, "./duplicate/../optimizer.pt": digest},
            world_size=1,
        )

    fsdp = {
        "trainer_state.json": digest,
        "pytorch_model_fsdp_0/.metadata": digest,
        "pytorch_model_fsdp_0/__0_0.distcp": digest,
        "optimizer_0/.metadata": digest,
        "optimizer_0/__0_0.distcp": digest,
        "scheduler.pt": digest,
        "rng_state.pth": digest,
    }
    assert validate_native_trainer_checkpoint_files(fsdp, world_size=1) == fsdp
    incomplete_fsdp = dict(fsdp)
    incomplete_fsdp.pop("optimizer_0/.metadata")
    with pytest.raises(ValueError, match="incomplete FSDP"):
        validate_native_trainer_checkpoint_files(incomplete_fsdp, world_size=1)

    full_fsdp = {
        "trainer_state.json": digest,
        "pytorch_model_fsdp.bin": digest,
        "optimizer.bin": digest,
        "scheduler.pt": digest,
        "rng_state.pth": digest,
    }
    assert validate_native_trainer_checkpoint_files(full_fsdp, world_size=1) == full_fsdp

    for alias in ("optimizer_999.pt", "optimizer_999.bin"):
        aliased = dict(complete)
        aliased.pop("optimizer.pt")
        aliased[alias] = digest
        with pytest.raises(ValueError, match="unexpected files"):
            validate_native_trainer_checkpoint_files(aliased, world_size=1)
    for source, alias in (
        ("pytorch_model_fsdp_0", "pytorch_model_fsdp_77"),
        ("optimizer_0", "optimizer_41"),
    ):
        aliased = {
            relative.replace(source, alias): value
            for relative, value in fsdp.items()
        }
        with pytest.raises(ValueError, match="unexpected files"):
            validate_native_trainer_checkpoint_files(aliased, world_size=1)


@pytest.mark.unit
def test_grpo_resume_ancestry_is_unique_content_addressed_and_cycle_free(tmp_path) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"resume")
    ancestry = extend_checkpoint_ancestry([f"sha256:{'a' * 64}"], checkpoint)
    assert ancestry[0] == f"sha256:{'a' * 64}"
    assert ancestry[-1].startswith("sha256:")
    with pytest.raises(ValueError, match="cycle"):
        extend_checkpoint_ancestry(
            [f"sha256:{'a' * 64}", f"sha256:{'a' * 64}"],
            checkpoint,
        )
    with pytest.raises(ValueError, match="content-addressed"):
        extend_checkpoint_ancestry(["resume.pt"], checkpoint)
    with pytest.raises(ValueError, match="repeated checkpoint"):
        extend_checkpoint_ancestry(ancestry, checkpoint)


@pytest.mark.unit
def test_grpo_distributed_consensus_rejects_one_rank_step_drift() -> None:
    shared = {
        "global_step": 7,
        "token_budget": {"consumed": 128},
        "experiment_binding_sha256": "a" * 64,
        "current_policy_contract": {"max_policy_lag": 0},
        "optimizer_boundary": "completed_optimizer_step",
    }
    assert len(
        _validate_distributed_training_consensus(
            [{"rank": 0, **shared}, {"rank": 1, **shared}],
            world_size=2,
        )
    ) == 2
    with pytest.raises(RuntimeError, match="ranks disagree"):
        _validate_distributed_training_consensus(
            [
                {"rank": 0, **shared},
                {"rank": 1, **{**shared, "global_step": 6}},
            ],
            world_size=2,
        )


@pytest.mark.unit
def test_grpo_resume_requires_content_bound_native_trainer_state(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    checkpoint_path = run_root / "checkpoints" / "step-00000003.pt"
    checkpoint_path.parent.mkdir(parents=True)
    state_root = run_root / "checkpoint-3"
    state_root.mkdir()
    (state_root / "trainer_state.json").write_text(
        json.dumps({"global_step": 3}),
        encoding="utf-8",
    )
    optimizer_path = state_root / "optimizer.pt"
    optimizer_path.write_bytes(b"optimizer-state")
    (state_root / "scheduler.pt").write_bytes(b"scheduler-state")
    (state_root / "rng_state.pth").write_bytes(b"rng-state")
    (state_root / "model.safetensors").write_bytes(b"model-state")
    files = accelerator_state_file_hashes(
        state_root,
        expected_run_root=run_root,
    )
    payload = {
        "global_step": 3,
        "world_size": 1,
        "optimizer": {
            "accelerate_state_path": str(state_root.resolve()),
            "files": files,
            "content_sha256": sha256_value(files),
        },
        "accelerate_state_path": str(state_root.resolve()),
        "accelerate_state_sha256": sha256_value(files),
    }
    assert _validate_grpo_resume_state_directory(
        payload,
        checkpoint_path=checkpoint_path,
    ) == state_root.resolve()
    optimizer_path.write_bytes(b"same-path-replacement")
    with pytest.raises(ValueError, match="content differs"):
        _validate_grpo_resume_state_directory(
            payload,
            checkpoint_path=checkpoint_path,
        )


@pytest.mark.unit
def test_grpo_update_norm_recomputation_rejects_snapshot_tampering(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    baseline_path = tmp_path / "baseline.pt"
    torch.save({"model": {"weight": torch.tensor([1.0, 2.0])}}, baseline_path)
    baseline_sha256 = sha256_file(baseline_path)
    snapshot = _snapshot_parameter_state(
        {"weight": torch.tensor([1.0, 2.0])},
        run_root / "initial_parameter_snapshot",
    )
    manifest_path = run_root / "initial_parameter_snapshot" / "manifest.json"
    evidence = {
        "initial_parameter_snapshot": snapshot,
        "initial_parameter_snapshot_manifest_path": str(manifest_path.resolve()),
        "initial_parameter_snapshot_manifest_file_sha256": (
            accelerator_state_file_hashes(
                run_root / "initial_parameter_snapshot",
                expected_run_root=run_root,
            )["manifest.json"]
        ),
        "update_norm_baseline_checkpoint_path": str(baseline_path.resolve()),
        "update_norm_baseline_checkpoint_sha256": baseline_sha256,
    }
    observed_norm, _ = _recompute_grpo_update_norm(
        run_root,
        evidence=evidence,
        checkpoint_payload={
            "model": {"weight": torch.tensor([2.0, 4.0])},
            "update_norm_baseline_checkpoint_path": str(baseline_path.resolve()),
            "update_norm_baseline_checkpoint_sha256": baseline_sha256,
        },
    )
    assert observed_norm == pytest.approx(5**0.5)
    tensor_path = next(
        path
        for path in (run_root / "initial_parameter_snapshot").iterdir()
        if path.suffix == ".pt"
    )
    tensor_path.write_bytes(b"same-path-tampering")
    with pytest.raises(ValueError, match="tensor bytes changed"):
        _recompute_grpo_update_norm(
            run_root,
            evidence=evidence,
            checkpoint_payload={
                "model": {"weight": torch.tensor([2.0, 4.0])},
                "update_norm_baseline_checkpoint_path": str(baseline_path.resolve()),
                "update_norm_baseline_checkpoint_sha256": baseline_sha256,
            },
        )


@pytest.mark.unit
def test_grpo_update_norm_rejects_late_snapshot_and_zero_update(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    baseline_path = tmp_path / "baseline.pt"
    baseline_model = {"weight": torch.tensor([1.0, 2.0])}
    torch.save({"model": baseline_model}, baseline_path)
    baseline_sha256 = sha256_file(baseline_path)

    def evidence_for(root, snapshot):  # type: ignore[no-untyped-def]
        manifest_path = root / "initial_parameter_snapshot" / "manifest.json"
        return {
            "initial_parameter_snapshot": snapshot,
            "initial_parameter_snapshot_manifest_path": str(manifest_path.resolve()),
            "initial_parameter_snapshot_manifest_file_sha256": sha256_file(manifest_path),
            "update_norm_baseline_checkpoint_path": str(baseline_path.resolve()),
            "update_norm_baseline_checkpoint_sha256": baseline_sha256,
        }

    late_root = tmp_path / "late-run"
    late_snapshot = _snapshot_parameter_state(
        {"weight": torch.tensor([1.5, 2.0])},
        late_root / "initial_parameter_snapshot",
    )
    bound_checkpoint_fields = {
        "update_norm_baseline_checkpoint_path": str(baseline_path.resolve()),
        "update_norm_baseline_checkpoint_sha256": baseline_sha256,
    }
    with pytest.raises(ValueError, match="differs from its bound baseline checkpoint"):
        _recompute_grpo_update_norm(
            late_root,
            evidence=evidence_for(late_root, late_snapshot),
            checkpoint_payload={
                "model": {"weight": torch.tensor([2.0, 4.0])},
                **bound_checkpoint_fields,
            },
        )

    zero_root = tmp_path / "zero-run"
    zero_snapshot = _snapshot_parameter_state(
        baseline_model,
        zero_root / "initial_parameter_snapshot",
    )
    with pytest.raises(ValueError, match="did not change"):
        _recompute_grpo_update_norm(
            zero_root,
            evidence=evidence_for(zero_root, zero_snapshot),
            checkpoint_payload={"model": baseline_model, **bound_checkpoint_fields},
        )


@pytest.mark.unit
def test_grpo_reward_adapter_supports_exact_format_and_matched_random() -> None:
    examples = build_smoke_examples(2)
    rows, exact = build_grpo_rows_and_reward(
        examples,
        reward_name="exact",
        seed=3,
    )
    completions = [
        "<proof>\n"
        + ("S01: R01(F01,F02) -> Q\n" if examples[0].label else "")
        + f"</proof>\n<answer>{examples[0].label}</answer>",
        "invalid",
    ]
    assert len(rows) == 2
    assert exact([row["prompt"] for row in rows], completions)[1] == 0.0
    _, format_reward = build_grpo_rows_and_reward(
        examples,
        reward_name="format_only",
        seed=3,
    )
    assert (
        format_reward(
            [row["prompt"] for row in rows],
            completions,
        )[1]
        == 0.0
    )
    _, random_reward = build_grpo_rows_and_reward(
        examples,
        reward_name="matched_random",
        seed=3,
        matched_positive_rate=0.5,
    )
    values = random_reward(
        [row["prompt"] for row in rows],
        completions,
    )
    assert sum(values) == 1.0
    assert values == random_reward(
        [row["prompt"] for row in rows],
        list(reversed(completions)),
    )


@pytest.mark.unit
def test_run_grpo_builds_official_backend_and_calls_train(
    monkeypatch,
    tmp_path,
    dataset_family_path,
) -> None:  # type: ignore[no-untyped-def]
    state = {"trained": False, "rows": 0}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):  # type: ignore[no-untyped-def]
            state["rows"] = len(rows)
            return rows

    class FakeTrainer:
        def __init__(self, model):  # type: ignore[no-untyped-def]
            self.model = model
            self.state = SimpleNamespace(global_step=0)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda _: 1.0,
            )

        def train(self):  # type: ignore[no-untyped-def]
            with torch.no_grad():
                next(self.model.parameters()).add_(0.01)
            self.state.global_step = 1
            state["trained"] = True
            return "ok"

    def fake_build(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["train_dataset"]
        assert callable(kwargs["reward_funcs"])
        return FakeTrainer(kwargs["model"])

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=FakeDataset),
    )
    monkeypatch.setattr(TrlGrpoBackend, "build", fake_build)
    monkeypatch.setattr(
        "posttrain_circuits.cli.run_grpo.require_pinned_trl_version",
        lambda: "0.22.2",
    )
    run_grpo_main(
        [
            "experiment=canonical_grpo",
            f"task.dataset_family_path={dataset_family_path}",
            "task.num_examples=2",
            "trainer.max_steps=1",
            "supervision.num_generations=2",
            "supervision.max_completion_length=4",
            "supervision.gradient_accumulation_steps=1",
            "--output",
            str(tmp_path / "grpo"),
        ]
    )
    assert state == {"trained": True, "rows": 2}
    evidence = json.loads(
        (tmp_path / "grpo" / "grpo_update_evidence.json").read_text(
            encoding="utf-8",
        )
    )
    assert evidence["optimizer_steps"] == 1
    assert evidence["parameter_update_norm"] > 0.0
    assert evidence["backend_version"] == "0.22.2"
    assert evidence["backend_batch_contract"] == "0.22.2-default-steps_per_generation"
    assert evidence["batch_contract"]["steps_per_generation"] == 1
    assert evidence["experiment_binding_sha256"]
    manifest = json.loads((tmp_path / "grpo" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_cell"] == "canonical_grpo"
    assert manifest["prereg_sha256"] != "unavailable"
    assert "prereg_git_commit" in manifest
    assert manifest["end_time"]


@pytest.mark.unit
def test_grpo_runtime_refuses_any_unpinned_trl_version(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "posttrain_circuits.learning.training.grpo_backend.importlib.metadata.version",
        lambda _: "0.22.1",
    )
    with pytest.raises(RuntimeError, match="trl==0.22.2"):
        require_pinned_trl_version()
    monkeypatch.setattr(
        "posttrain_circuits.learning.training.grpo_backend.importlib.metadata.version",
        lambda _: "0.22.2",
    )
    assert require_pinned_trl_version() == "0.22.2"


@pytest.mark.unit
def test_trl_config_preserves_registered_generation_batch_semantics(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["config"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def compute_loss(self, model, inputs, *args, **kwargs):  # type: ignore[no-untyped-def]
            del model, inputs, args, kwargs

    monkeypatch.setattr(
        "posttrain_circuits.learning.training.grpo_backend.require_pinned_trl_version",
        lambda: "0.22.2",
    )
    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(GRPOConfig=FakeConfig, GRPOTrainer=FakeTrainer),
    )
    settings = GrpoSettings(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,
        reserved_tokens_per_update=16,
    )
    trainer = TrlGrpoBackend(settings).build(
        model=object(),
        reward_funcs=lambda *_: 0.0,
        train_dataset=[{"prompt": "p"}],
        output_dir="unused",
    )
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["gradient_accumulation_steps"] == 2
    assert config["per_device_train_batch_size"] == 2
    assert config["num_generations"] == 4
    assert "steps_per_generation" not in config
    assert trainer.registered_max_steps == settings.max_steps
