"""Real CPU FSDP/Accelerate parameter replacement; no GPU or network allocation.

PyTorch's single-rank FakeProcessGroup supplies only communication plumbing.
FSDP flattening, model forwards/backwards, AcceleratedOptimizer and AdamW are real.
This does not establish multi-rank collectives or GPU acceptance.
"""

import copy
from types import SimpleNamespace

import accelerate
import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DistributedType

from posttrain_circuits.artifacts.checkpoints import torch_state_hash
from posttrain_circuits.learning.primitives import SupervisionBatch
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.factorial_trainer import (
    FactorialTrainer,
    TrainerConfig,
    prepare_accelerate_model_optimizer_scheduler,
    validate_prepared_optimizer_binding,
)
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    PromptScheduler,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3


@pytest.fixture
def cpu_fsdp(monkeypatch):
    # Registration creates no group or sockets; one rank's NO_SHARD performs
    # exactly the parameter replacement relevant to both real failing ranks.
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
        value = Accelerator(cpu=True, mixed_precision="no", **kwargs)
        value.state.distributed_type = DistributedType.FSDP
        value.state.fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=1,
            use_orig_params=False,
            sync_module_states=False,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
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


class EmptyTeacherSource(TeacherDemoStateSource):
    def __init__(self):
        pass

    def state_dict(self):
        return {"kind": "teacher_demo"}


def create_trainer(tmp_path):
    model = build_tiny_qwen3(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
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
        state_source=EmptyTeacherSource(),
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
        run_dir=tmp_path,
    )
    return trainer, optimizer, scheduler


def optimizer_window(trainer):
    batch = SupervisionBatch(
        input_ids=torch.tensor([[1, 2, 3, 4]] * 4),
        attention_mask=torch.ones(4, 4, dtype=torch.bool),
        response_mask=torch.tensor([[False, False, True, True]] * 4),
        rewards=torch.ones(4),
    )
    sync = []
    for _ in range(16):
        sync.append(trainer._training_micro_step(batch, expected_sequence_count=4))
    assert sync == [False] * 15 + [True]
    trainer.global_step += 1
    trainer._validate_live_optimizer_scheduler_cadence()


@pytest.mark.unit
def test_actual_trainer_first_global64_update_creates_bound_adamw_state(cpu_fsdp, tmp_path):
    trainer, raw_optimizer, raw_scheduler = create_trainer(tmp_path)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    optimizer_window(trainer)
    assert trainer.optimizer.state_dict()["state"]
    assert raw_optimizer.state_dict()["state"]
    assert trainer.scheduler is raw_scheduler
    assert raw_scheduler.optimizer is raw_optimizer
    assert any(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, trainer.model.parameters(), strict=True)
    )
    evidence = validate_prepared_optimizer_binding(trainer.model, trainer.optimizer, raw_scheduler)
    assert evidence["exact_parameter_identity_match"] is True
    assert evidence["raw_scheduler_optimizer_identity_match"] is True
    assert evidence["model_trainable_parameter_tensors"] == evidence["optimizer_parameter_tensors"]
    assert len(raw_optimizer.state) == len(list(trainer.model.parameters()))
    assert {int(state["step"]) for state in raw_optimizer.state.values()} == {1}
    assert raw_scheduler.last_epoch == 1
    assert raw_optimizer.param_groups[0]["lr"] == pytest.approx(0.002 / 1.01)


@pytest.mark.unit
def test_preparation_preserves_raw_objects_options_scheduler_and_rng(cpu_fsdp):
    model = build_tiny_qwen3(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step / 100))
    options = {
        key: copy.deepcopy(value) for key, value in optimizer.param_groups[0].items() if key != "params"
    }
    defaults = copy.deepcopy(optimizer.defaults)
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    original_ids = {id(parameter) for parameter in model.parameters()}
    rng = torch.random.get_rng_state().clone()
    accelerator = cpu_fsdp(step_scheduler_with_optimizer=False)
    prepared, accelerated_optimizer, raw_scheduler = prepare_accelerate_model_optimizer_scheduler(
        accelerator,
        model,
        optimizer,
        scheduler,
    )
    assert accelerated_optimizer.optimizer is optimizer
    assert raw_scheduler is scheduler
    assert scheduler.optimizer is optimizer
    assert original_ids.isdisjoint({id(parameter) for parameter in prepared.parameters()})
    assert options == {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    assert optimizer.defaults == defaults
    assert scheduler.state_dict() == scheduler_state
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.unit
def test_real_accelerator_full_state_restore_reproduces_next_update(cpu_fsdp, tmp_path):
    trainer, raw_optimizer, scheduler = create_trainer(tmp_path)
    optimizer_window(trainer)
    state_dir = tmp_path / "saved-accelerator-state"
    trainer._accelerator.save_state(state_dir)
    saved_model = torch_state_hash(trainer.model.state_dict())
    saved_optimizer = torch_state_hash(raw_optimizer.state_dict())
    saved_scheduler = torch_state_hash(scheduler.state_dict())
    optimizer_window(trainer)
    expected = (
        torch_state_hash(trainer.model.state_dict()),
        torch_state_hash(raw_optimizer.state_dict()),
        torch_state_hash(scheduler.state_dict()),
    )
    trainer._accelerator.load_state(state_dir)
    trainer.global_step = 1
    assert torch_state_hash(trainer.model.state_dict()) == saved_model
    assert torch_state_hash(raw_optimizer.state_dict()) == saved_optimizer
    assert torch_state_hash(scheduler.state_dict()) == saved_scheduler
    validate_prepared_optimizer_binding(trainer.model, trainer.optimizer, scheduler)
    optimizer_window(trainer)
    assert (
        torch_state_hash(trainer.model.state_dict()),
        torch_state_hash(raw_optimizer.state_dict()),
        torch_state_hash(scheduler.state_dict()),
    ) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect",
    [
        "non_adamw",
        "multiple_groups",
        "existing_state",
        "existing_gradients",
        "wrong_scheduler",
        "advanced_scheduler",
        "named_parameters",
        "missing_parameter",
        "duplicate_parameter",
    ],
)
def test_unsupported_rebinding_fails_before_model_preparation(defect):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if defect == "non_adamw":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    elif defect == "multiple_groups":
        optimizer = torch.optim.AdamW([{"params": [model.weight]}, {"params": [model.bias]}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if defect == "existing_state":
        optimizer.state[model.weight] = {"step": torch.tensor(1.0)}
    elif defect == "existing_gradients":
        model.weight.grad = torch.ones_like(model.weight)
    elif defect == "wrong_scheduler":
        scheduler = torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(model.parameters()), lambda _: 1.0)
    elif defect == "advanced_scheduler":
        scheduler.last_epoch = 1
    elif defect == "named_parameters":
        optimizer.param_groups[0]["param_names"] = ["weight", "bias"]
    elif defect == "missing_parameter":
        optimizer.param_groups[0]["params"] = [model.weight]
    elif defect == "duplicate_parameter":
        optimizer.param_groups[0]["params"].append(model.weight)

    def forbidden_prepare(*args):
        raise AssertionError("unsupported input reached model preparation")

    accelerator = SimpleNamespace(
        distributed_type="FSDP",
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(fsdp_version=1, use_orig_params=False),
        ),
        prepare=forbidden_prepare,
    )
    with pytest.raises(ValueError):
        prepare_accelerate_model_optimizer_scheduler(accelerator, model, optimizer, scheduler)


@pytest.mark.unit
def test_prepared_binding_rejects_stale_parameters_and_detached_scheduler():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    other = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="exactly match"):
        validate_prepared_optimizer_binding(other, optimizer, scheduler)
    with pytest.raises(ValueError, match="not bound"):
        validate_prepared_optimizer_binding(model, torch.optim.AdamW(model.parameters()), scheduler)
