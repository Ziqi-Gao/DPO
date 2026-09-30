"""Exercise the real BF16-load/FP32-master Factorial checkpoint boundary on CPU.

The single-rank FakeProcessGroup supplies communication only. Accelerator,
FSDP, model forwards/backwards, AdamW, and checkpoint publication are real.
This fixture does not establish GPU or multi-rank correctness.
"""

import copy
import json
from types import SimpleNamespace

import accelerate
import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import DistributedType
from torch.distributed.fsdp import MixedPrecision

from posttrain_circuits.artifacts.checkpoints import (
    adapted_student_model_update_evidence,
    checkpoint_runtime_state_hashes,
    model_update_evidence,
    torch_state_hash,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.cli.finalize_pilot_training import _validate_factorial_update_evidence
from posttrain_circuits.learning.primitives import SupervisionBatch
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    PromptScheduler,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

V5_CONFIG = {
    "protocol_amendment_path": "prereg/amendments/qwen3_adapted_student_calibration_v5.json",
    "adapted_teacher": {
        "student_protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v5.json"
    },
}


@pytest.fixture(params=[5, 6])
def reviewed_config(request):
    path = f"prereg/amendments/qwen3_adapted_student_calibration_v{request.param}.json"
    return {"protocol_amendment_path": path, "adapted_teacher": {"student_protocol_path": path}}


@pytest.fixture
def bf16_cpu_fsdp(monkeypatch):
    import torch.testing._internal.distributed.fake_pg

    assert not dist.is_initialized()
    assert not torch.cuda.is_initialized()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "ACCELERATE_USE_FSDP", "FSDP_SYNC_MODULE_STATES"):
        monkeypatch.delenv(name, raising=False)
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)

    def factory(**kwargs):
        value = Accelerator(cpu=True, mixed_precision="bf16", **kwargs)
        value.state.distributed_type = DistributedType.FSDP
        # Autocast re-reads AcceleratorState from PartialState, so both must
        # describe the same communication fixture as they do under a launcher.
        PartialState().distributed_type = DistributedType.FSDP
        value.state.fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=1,
            use_orig_params=False,
            sync_module_states=False,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
        return value

    monkeypatch.setattr(accelerate, "Accelerator", factory)
    try:
        yield factory
    finally:
        dist.destroy_process_group()
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        assert not torch.cuda.is_initialized()


class CheckpointTeacherSource(TeacherDemoStateSource):
    """Only source-state serialization is involved in this checkpoint fixture."""

    def __init__(self):
        self.cursor = {}

    def state_dict(self):
        return {
            "kind": "teacher_demo",
            "cursor_protocol_id": "teacher-demo-round-robin-v2-accepted-view",
            "cursor": dict(self.cursor),
            "attempt_ids": [f"p{i}:candidate-0000" for i in range(64)],
        }

    def load_state_dict(self, state):
        self.cursor = dict(state["cursor"])


def create_checkpoint_trainer(tmp_path, protocol_config):
    model = build_tiny_qwen3(123).to(dtype=torch.bfloat16)
    baseline_path = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict()}, baseline_path)
    baseline_sha = sha256_file(baseline_path)
    baseline = torch.load(baseline_path, weights_only=False)["model"]
    assert {tensor.dtype for tensor in baseline.values()} == {torch.bfloat16}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step / 100))
    prompts = PromptScheduler.for_allocation_neutral_rank(
        [f"p{i}" for i in range(64)],
        ["fixture"] * 64,
        rank=0,
        world_size=1,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    trainer = FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompts,
        state_source=CheckpointTeacherSource(),
        supervisor=CanonicalSFTSupervisor(0),
        config=TrainerConfig(
            max_steps=120,
            token_budget=2_000_000,
            backend="accelerate",
            checkpoint_every=20,
            evaluation_every=20,
            batch_partition_protocol=ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "run",
        resolved_config={
            **copy.deepcopy(protocol_config),
            "production_safety": {"initial_checkpoint_path": str(baseline_path)},
            "trainer": {
                "backend": "accelerate",
                "max_steps": 120,
                "max_model_input_length": 1536,
                "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
            },
        },
        manifest_hashes={"initial_checkpoint": baseline_sha},
    )
    assert {parameter.dtype for parameter in trainer.model.parameters()} == {torch.float32}
    return trainer, baseline_path, baseline_sha, baseline


def real_global64_update(trainer):
    batch = SupervisionBatch(
        input_ids=torch.tensor([[1, 2, 3, 4]] * 4),
        attention_mask=torch.ones(4, 4, dtype=torch.bool),
        response_mask=torch.tensor([[False, False, True, True]] * 4),
        rewards=torch.ones(4),
    )
    assert trainer.token_budget.reserve_optimizer_update(64 * 4) == (True, 64 * 4)
    synchronization = []
    for _ in range(16):
        prompts = trainer.prompt_scheduler.next_batch()
        for prompt_id in prompts.prompt_ids:
            trainer.state_source.cursor[prompt_id] = trainer.state_source.cursor.get(prompt_id, 0) + 1
        synchronization.append(trainer._training_micro_step(batch, expected_sequence_count=4))
    assert synchronization == [False] * 15 + [True]
    trainer.global_step += 1
    trainer.cumulative_counts["prompts_consumed"] += 64.0
    trainer.cumulative_counts["trajectories_generated"] += 64.0
    trainer.cumulative_counts["model_facing_input_tokens_processed"] = float(trainer.token_budget.consumed)
    trainer._validate_live_optimizer_scheduler_cadence()


@pytest.mark.unit
def test_real_bf16_baseline_fp32_fsdp_update_publishes_checkpoint(
    bf16_cpu_fsdp, tmp_path, reviewed_config
):
    trainer, baseline_path, baseline_sha, baseline = create_checkpoint_trainer(tmp_path, reviewed_config)
    real_global64_update(trainer)
    final_state = trainer._full_model_state_for_checkpoint()
    assert set(final_state) == set(baseline)
    assert all(final_state[name].shape == baseline[name].shape for name in final_state)
    assert {tensor.dtype for tensor in final_state.values()} == {torch.float32}
    expected_squared = sum(
        float(((final_state[name].double() - baseline[name].double()) ** 2).sum()) for name in final_state
    )
    assert expected_squared > 0
    checkpoint_path = trainer.run_dir / "checkpoints" / "step-00000001.pt"
    checkpoint_sha = trainer.save(checkpoint_path)
    assert checkpoint_sha == sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=False)
    assert payload["parameter_update_norm"] == pytest.approx(expected_squared**0.5)
    assert payload["final_model_state_hash"] == torch_state_hash(final_state)
    assert payload["update_norm_baseline_checkpoint_sha256"] == baseline_sha
    assert sha256_file(baseline_path) == baseline_sha
    assert {tensor.dtype for tensor in payload["model"].values()} == {torch.float32}
    assert checkpoint_path.with_suffix(".accelerate").is_dir()
    validate_actual_finalizer(trainer, checkpoint_path, payload, baseline_sha)

    # Restore through the real trainer boundary, preserving its ancestry,
    # optimizer/scheduler/RNG and source state checks, then update and save.
    saved_state = torch_state_hash(trainer._full_model_state_for_checkpoint())
    real_global64_update(trainer)
    expected_next_state = torch_state_hash(trainer._full_model_state_for_checkpoint())
    expected_next_optimizer = torch_state_hash(trainer.optimizer.state_dict())
    trainer.resume(checkpoint_path)
    assert trainer.global_step == 1
    assert trainer.resume_ancestry == [f"sha256:{checkpoint_sha}"]
    assert torch_state_hash(trainer._full_model_state_for_checkpoint()) == saved_state
    with pytest.raises(RuntimeError, match="no parameter change"):
        trainer._model_update_evidence(trainer._full_model_state_for_checkpoint())
    real_global64_update(trainer)
    assert torch_state_hash(trainer._full_model_state_for_checkpoint()) == expected_next_state
    assert torch_state_hash(trainer.optimizer.state_dict()) == expected_next_optimizer
    resumed_checkpoint = trainer.run_dir / "checkpoints" / "step-00000002.pt"
    trainer.save(resumed_checkpoint)
    resumed_payload = torch.load(resumed_checkpoint, weights_only=False)
    assert resumed_payload["parameter_update_norm"] > 0
    assert resumed_payload["update_norm_baseline_checkpoint_sha256"] == checkpoint_sha
    assert resumed_payload["resume_ancestry"] == [f"sha256:{checkpoint_sha}"]
    assert {tensor.dtype for tensor in resumed_payload["model"].values()} == {torch.float32}
    assert sha256_file(baseline_path) == baseline_sha
    validate_actual_finalizer(trainer, resumed_checkpoint, resumed_payload, baseline_sha)


def validate_actual_finalizer(trainer, checkpoint_path, payload, initial_sha):
    """Recompute final evidence through the actual independent finalizer."""
    root = trainer.run_dir
    metrics_path = root / "metrics.jsonl"
    rows = [{"step": payload["global_step"], "parameter_update_norm": payload["parameter_update_norm"]}]
    metrics_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    binding = SimpleNamespace(
        initial_checkpoint_sha256=initial_sha,
        scientific_sha256="a" * 64,
        factorial_design_sha256="b" * 64,
        method_spec_sha256="c" * 64,
        implementation_commit="d" * 40,
        implementation_dirty=False,
    )
    evidence = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metric_update_rows": rows,
        "metric_update_rows_sha256": sha256_value(rows),
        "global_step": payload["global_step"],
        "parameter_update_norm": payload["parameter_update_norm"],
        "final_model_state_hash": payload["final_model_state_hash"],
        "update_norm_baseline_checkpoint_path": payload["update_norm_baseline_checkpoint_path"],
        "update_norm_baseline_checkpoint_sha256": payload["update_norm_baseline_checkpoint_sha256"],
        "token_budget": payload["token_budget"],
        "resume_ancestry": payload["resume_ancestry"],
        "checkpoint_runtime_state_hashes": checkpoint_runtime_state_hashes(payload),
        "manifest_hashes": payload["manifest_hashes"],
        "experiment_binding_sha256": binding.scientific_sha256,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "method_spec_sha256": binding.method_spec_sha256,
        "git_commit": binding.implementation_commit,
        "implementation_dirty": binding.implementation_dirty,
    }
    evidence["sha256"] = sha256_value(evidence)
    (root / "factorial_update_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    budget = payload["token_budget"]
    manifest = {
        "resume_ancestry": payload["resume_ancestry"],
        "dataset_hashes": {"factorial_update_evidence": evidence["sha256"]},
        "token_budget": budget["budget"],
        "token_budget_unit": budget["unit"],
        "token_budget_consumed": budget["consumed"],
        "training_stop_reason": budget["stop_reason"],
    }
    assert (
        _validate_factorial_update_evidence(
            root,
            manifest=manifest,
            binding=binding,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=trainer.resolved_config,
        )
        == evidence["sha256"]
    )


@pytest.mark.unit
def test_v5_v6_promotion_alone_is_zero_and_does_not_modify_or_rehash_final(reviewed_config):
    baseline = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    final = {"weight": baseline["weight"].float()}
    before_hash = torch_state_hash(baseline)
    final_hash = torch_state_hash(final)
    norm, observed_hash = adapted_student_model_update_evidence(
        baseline, final, resolved_config=reviewed_config, resume_ancestry=[]
    )
    assert norm == 0.0
    assert observed_hash == final_hash
    assert torch_state_hash(baseline) == before_hash
    assert torch_state_hash(final) == final_hash
    final["weight"][0] += 1 / 1024
    norm, observed_hash = adapted_student_model_update_evidence(
        baseline, final, resolved_config=reviewed_config, resume_ancestry=[]
    )
    assert norm == 1 / 1024  # Would disappear if final values were rounded to BF16.
    assert observed_hash == torch_state_hash(final)


@pytest.mark.unit
@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 7])
def test_historical_and_unknown_protocols_keep_strict_metadata(version):
    baseline = {"weight": torch.tensor([1.0], dtype=torch.bfloat16)}
    final = {"weight": torch.tensor([1.01], dtype=torch.float32)}
    config = (
        {}
        if version is None
        else {
            "protocol_amendment_path": (
                f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
            ),
            "adapted_teacher": {
                "student_protocol_path": (
                    f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
                )
            }
        }
    )
    with pytest.raises(ValueError, match="metadata differs"):
        adapted_student_model_update_evidence(baseline, final, resolved_config=config, resume_ancestry=[])
    with pytest.raises(ValueError, match="metadata differs"):
        model_update_evidence(baseline, final)


@pytest.mark.unit
@pytest.mark.parametrize("selector", ["protocol_amendment_path", "student_protocol_path"])
@pytest.mark.parametrize("other", [None, "", "v4", "other_reviewed", "v7"])
def test_v5_v6_requires_both_matching_protocol_selectors(selector, other, reviewed_config):
    config = copy.deepcopy(reviewed_config)
    if other == "other_reviewed":
        other = "v6" if config == V5_CONFIG else "v5"
    changed = config if selector == "protocol_amendment_path" else config["adapted_teacher"]
    if other is None:
        del changed[selector]
    else:
        changed[selector] = (
            f"prereg/amendments/qwen3_adapted_student_calibration_{other}.json" if other else ""
        )
    # Even equal dtypes must fail, rather than silently dropping to the
    # historical comparator when either selector explicitly requests v5/v6.
    baseline = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    final = {"weight": torch.tensor([1.01], dtype=torch.float32)}
    with pytest.raises(ValueError, match="contradictory protocol selectors"):
        adapted_student_model_update_evidence(
            baseline, final, resolved_config=config, resume_ancestry=[]
        )


@pytest.mark.unit
@pytest.mark.parametrize("adapted_teacher", [None, {}, "invalid"])
def test_v5_v6_top_level_without_nested_protocol_fails_closed(adapted_teacher, reviewed_config):
    config = {"protocol_amendment_path": reviewed_config["protocol_amendment_path"]}
    if adapted_teacher is not None:
        config["adapted_teacher"] = adapted_teacher
    state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    with pytest.raises(ValueError, match="contradictory protocol selectors"):
        adapted_student_model_update_evidence(state, state, resolved_config=config, resume_ancestry=[])


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect",
    [
        "keys",
        "shape",
        "fp16_baseline",
        "fp32_fresh_baseline",
        "mixed_baseline",
        "bf16_final",
        "fp16_final",
        "fp64_final",
        "mixed_final",
        "non_tensor",
        "nonfinite_baseline",
        "nonfinite_final",
        "nonfloating_dtype",
        "nonfloating_value",
        "complex",
        "no_floating",
        "invalid_ancestry_type",
        "invalid_ancestry_hash",
        "duplicate_ancestry",
        "bf16_resumed_baseline",
    ],
)
def test_v5_v6_rejects_unreviewed_tensor_metadata_and_ancestry(defect, reviewed_config):
    baseline = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    final = {"weight": torch.tensor([1.01, 2.01], dtype=torch.float32)}
    ancestry = []
    if defect == "keys":
        final = {"other": final["weight"]}
    elif defect == "shape":
        final["weight"] = final["weight"].reshape(2, 1)
    elif defect.endswith("baseline") and defect.startswith(("fp16", "fp32")):
        baseline["weight"] = baseline["weight"].to(
            torch.float16 if defect.startswith("fp16") else torch.float32
        )
    elif defect in {"mixed_baseline", "mixed_final"}:
        baseline["other"] = torch.ones(
            1, dtype=torch.float32 if defect == "mixed_baseline" else torch.bfloat16
        )
        final["other"] = torch.ones(1, dtype=torch.bfloat16 if defect == "mixed_final" else torch.float32)
    elif defect in {"bf16_final", "fp16_final", "fp64_final"}:
        final["weight"] = final["weight"].to(
            {"bf16_final": torch.bfloat16, "fp16_final": torch.float16, "fp64_final": torch.float64}[defect]
        )
    elif defect == "non_tensor":
        final["weight"] = [1.01, 2.01]
    elif defect == "nonfinite_baseline":
        baseline["weight"][0] = float("inf")
    elif defect == "nonfinite_final":
        final["weight"][0] = float("nan")
    elif defect == "nonfloating_dtype":
        baseline["counter"] = torch.tensor([1], dtype=torch.int64)
        final["counter"] = torch.tensor([1], dtype=torch.int32)
    elif defect == "nonfloating_value":
        baseline["counter"] = torch.tensor([1], dtype=torch.int64)
        final["counter"] = torch.tensor([2], dtype=torch.int64)
    elif defect == "complex":
        baseline["counter"] = torch.tensor([1j])
        final["counter"] = torch.tensor([1j])
    elif defect == "no_floating":
        baseline = final = {"counter": torch.tensor([1], dtype=torch.int64)}
    elif defect == "invalid_ancestry_type":
        ancestry = "sha256:" + "a" * 64
    elif defect == "invalid_ancestry_hash":
        ancestry = ["sha256:invalid"]
    elif defect == "duplicate_ancestry":
        ancestry = ["sha256:" + "a" * 64] * 2
    elif defect == "bf16_resumed_baseline":
        ancestry = ["sha256:" + "a" * 64]
    with pytest.raises(ValueError):
        adapted_student_model_update_evidence(
            baseline, final, resolved_config=reviewed_config, resume_ancestry=ancestry
        )
