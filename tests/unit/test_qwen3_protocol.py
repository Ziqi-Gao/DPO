from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.runs import (
    RunManifest,
    run_manifest_payload,
    validate_run_manifest_payload,
)
from posttrain_circuits.cli.finalize_pilot import _hash_valid as pilot_hash_valid
from posttrain_circuits.cli.compare_distributed_resume import _validate_exact_checkpoint
from posttrain_circuits.cli.finalize_pilot_training import (
    _validate_factorial_rank_states,
)
from posttrain_circuits.cli.train import (
    _require_qwen3_store_binding,
    _validate_allocation_neutral_prompt_population,
)
from posttrain_circuits.core.config import compose_config, validate_config
from posttrain_circuits.learning.contracts import PromptBatch, SamplingCursor, SamplingRequest
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import (
    chat_template_sha256,
    format_model_prompt,
)
from posttrain_circuits.learning.state_sources.generation import HF_SAMPLING_PROTOCOL_ID, hf_generate_trajectories
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    ExactGlobalBatchPlan,
    PromptScheduler,
)
from posttrain_circuits.learning.training.token_budget import TOKEN_BUDGET_UNIT
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _protocol_tokenizer():  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    tokenizer.chat_template = "frozen-qwen3-template-v1"

    def apply_chat_template(
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    tokenizer.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    return tokenizer


def _protocol_config(tokenizer) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "messages": "single_user",
            "add_generation_prompt": True,
            "chat_template_sha256": chat_template_sha256(tokenizer),
        }
    }


@pytest.mark.unit
def test_qwen3_config_is_exact_and_never_falls_back_to_qwen25() -> None:
    config = compose_config(
        [
            "production=qwen3_primary",
            "experiment=offline_soft",
            "model=qwen3_1p7b",
            "teacher=qwen3_teacher_8b",
            "g0=qwen3_eap_separation",
            "pilot=qwen3_core",
        ],
        config_root=Path("configs"),
    )
    assert config["model"]["model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert config["teacher"]["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert config["model"]["gradient_checkpointing"] is True
    assert config["model"]["use_cache"] is False
    assert config["teacher"]["gradient_checkpointing"] is False
    assert config["teacher"]["use_cache"] is True
    assert config["protocol_track"] == "qwen3_v1"
    assert "qwen3-v1" in Path(config["output_root"]).parts
    assert config["state_source"]["top_k"] == 20
    assert config["state_source"]["min_p"] == 0.0
    assert config["supervision"]["top_k"] == 20
    assert config["supervision"]["min_p"] == 0.0

    stale = copy.deepcopy(config)
    stale["state_source"]["store_path"] = "outputs/rollout_banks/qwen25"
    with pytest.raises(ValueError, match="escape their namespace"):
        validate_config(stale)


@pytest.mark.unit
def test_non_thinking_formatter_hashes_raw_and_exact_model_facing_bytes() -> None:
    tokenizer = _protocol_tokenizer()
    raw = "FACTS F01 A\nQUERY A"
    formatted = format_model_prompt(raw, tokenizer, _protocol_config(tokenizer))
    assert formatted.raw_prompt == raw
    assert formatted.prompt_protocol == "qwen3_non_thinking_v1"
    assert formatted.enable_thinking is False
    assert formatted.model_facing_prompt.count("<|im_start|>user") == 1
    assert "<think>\n\n</think>" in formatted.model_facing_prompt
    assert formatted.raw_prompt_sha256 != formatted.model_facing_prompt_sha256

    changed = copy.deepcopy(_protocol_config(tokenizer))
    changed["prompt_protocol"]["chat_template_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="chat template differs"):
        format_model_prompt(raw, tokenizer, changed)


@pytest.mark.unit
def test_tokenizer_fingerprint_includes_chat_template() -> None:
    tokenizer = _protocol_tokenizer()
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.chat_template += "-tampered"
    assert tokenizer_fingerprint(tokenizer) != original


@pytest.mark.unit
def test_qwen3_sampling_parameters_reach_hf_generate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    tokenizer = _protocol_tokenizer()
    model = build_tiny_qwen(55).eval()

    def fake_generate(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.0
        prefix = kwargs["input_ids"]
        token = torch.tensor([[4]], dtype=torch.long, device=prefix.device)
        scores = (torch.zeros((1, model.config.vocab_size), device=prefix.device),)
        return SimpleNamespace(sequences=torch.cat((prefix, token), dim=1), scores=scores)

    monkeypatch.setattr(model, "generate", fake_generate)
    record = hf_generate_trajectories(
        model,
        tokenizer,
        PromptBatch(("p0",), ("FACTS F01 A",)),
        policy_version=0,
        sampling_request=SamplingRequest(
            sampling_request_seed=17,
            sampling_protocol_id=HF_SAMPLING_PROTOCOL_ID,
            cursors=(SamplingCursor(0, 0, "p0", 0),),
        ),
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        policy_id="Qwen/Qwen3-1.7B",
        policy_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        model_config=_protocol_config(tokenizer),
    )[0]
    assert record.top_k == 20
    assert record.min_p == 0.0
    assert record.raw_prompt_text == "FACTS F01 A"
    assert record.prompt_text != record.raw_prompt_text
    assert len(record.response_ids) == len(record.behavior_logprobs) == 1


@pytest.mark.unit
def test_four_rank_prompt_shards_are_disjoint_and_resume_is_rank_bound() -> None:
    ids = [f"p-{index}" for index in range(16)]
    prompts = [f"prompt-{index}" for index in range(16)]
    schedulers = [
        PromptScheduler.for_distributed_rank(ids, prompts, 2, rank=rank, world_size=4) for rank in range(4)
    ]
    shards = [set(scheduler.prompt_ids) for scheduler in schedulers]
    assert set.union(*shards) == set(ids)
    assert all(shards[left].isdisjoint(shards[right]) for left in range(4) for right in range(left + 1, 4))
    with pytest.raises(ValueError, match="different rank shard"):
        schedulers[1].load_state_dict(schedulers[0].state_dict())


@pytest.mark.unit
def test_allocation_neutral_prompt_population_is_unique_window_aligned_and_ordered() -> None:
    prompt_ids = [f"p-{index}" for index in range(256)]
    _validate_allocation_neutral_prompt_population(
        prompt_ids,
        global_batch_size=64,
        expected_prompt_ids=list(prompt_ids),
        expected_prompt_count=256,
    )

    duplicate = list(prompt_ids)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="unique global prompt IDs"):
        _validate_allocation_neutral_prompt_population(
            duplicate,
            global_batch_size=64,
            expected_prompt_ids=duplicate,
            expected_prompt_count=256,
        )
    with pytest.raises(ValueError, match="exact multiple"):
        _validate_allocation_neutral_prompt_population(
            prompt_ids[:-1],
            global_batch_size=64,
            expected_prompt_ids=prompt_ids[:-1],
        )
    out_of_order = list(prompt_ids)
    out_of_order[0], out_of_order[1] = out_of_order[1], out_of_order[0]
    with pytest.raises(ValueError, match="train-family order"):
        _validate_allocation_neutral_prompt_population(
            out_of_order,
            global_batch_size=64,
            expected_prompt_ids=prompt_ids,
            expected_prompt_count=256,
        )
    with pytest.raises(ValueError, match="exactly 256"):
        _validate_allocation_neutral_prompt_population(
            prompt_ids[:128],
            global_batch_size=64,
            expected_prompt_ids=prompt_ids[:128],
            expected_prompt_count=256,
        )


@pytest.mark.unit
def test_allocation_neutral_prompt_windows_preserve_global_order_for_every_world_size() -> None:
    prompt_ids = [f"p-{index}" for index in range(256)]
    prompt_texts = [f"prompt-{index}" for index in range(256)]

    def consume_population(world_size: int) -> list[str]:
        plans = [ExactGlobalBatchPlan(64, 4, rank, world_size) for rank in range(world_size)]
        schedulers = [
            PromptScheduler.for_allocation_neutral_rank(
                prompt_ids,
                prompt_texts,
                rank=rank,
                world_size=world_size,
                global_batch_size=64,
                max_microbatch_size=4,
            )
            for rank in range(world_size)
        ]
        assert {plan.microsteps_per_optimizer_update for plan in plans} == {
            {1: 16, 2: 8, 3: 6, 4: 4}[world_size]
        }
        assigned_slots: list[tuple[int, str]] = []
        for window_index in range(4):
            for microbatch_index in range(plans[0].microsteps_per_optimizer_update):
                for rank, scheduler in enumerate(schedulers):
                    batch = scheduler.next_batch()
                    slots = plans[rank].global_slots_for_microbatch(microbatch_index)
                    absolute_slots = [window_index * 64 + slot for slot in slots]
                    assigned_slots.extend(zip(absolute_slots, batch.prompt_ids, strict=True))
        assert all(scheduler.global_slot_cursor == 256 for scheduler in schedulers)
        assert all(scheduler.microbatch_index == 0 for scheduler in schedulers)
        return [prompt_id for _, prompt_id in sorted(assigned_slots)]

    assert [consume_population(world_size) for world_size in (1, 2, 3, 4)] == [
        prompt_ids,
        prompt_ids,
        prompt_ids,
        prompt_ids,
    ]


@pytest.mark.unit
def test_allocation_neutral_three_rank_tail_is_collective_safe() -> None:
    plans = [ExactGlobalBatchPlan(64, 4, rank, 3) for rank in range(3)]
    assert [plan.microbatch_sizes for plan in plans] == [
        (4, 4, 4, 4, 4, 2),
        (4, 4, 4, 4, 4, 1),
        (4, 4, 4, 4, 4, 1),
    ]
    assert sum(plan.local_sequence_count for plan in plans) == 64


def _exact_checkpoint_fixture(world_size: int, *, global_step: int = 1) -> dict[str, object]:
    prompt_ids = [f"p-{index:03d}" for index in range(64)]
    attempt_ids = [f"{prompt_id}:candidate-0000" for prompt_id in prompt_ids]
    plans = [ExactGlobalBatchPlan(64, 4, rank, world_size) for rank in range(world_size)]
    partition = {
        "protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
        "global_batch_size": 64,
        "max_microbatch_size": 4,
        "max_model_input_length": 1536,
        "world_size": world_size,
        "microsteps_per_optimizer_update": plans[0].microsteps_per_optimizer_update,
        "rank_local_sequence_counts": [plan.local_sequence_count for plan in plans],
        "microbatch_sizes_by_rank": [list(plan.microbatch_sizes) for plan in plans],
    }
    budget = {
        "budget": 2_000_000,
        "unit": TOKEN_BUDGET_UNIT,
        "consumed": global_step * 64,
        "accepted_optimizer_updates": global_step,
        "stop_reason": "max_steps_safety_limit",
    }
    prompt_states: list[dict[str, object]] = []
    source_states: list[dict[str, object]] = []
    trainer_states: list[dict[str, object]] = []
    for rank, plan in enumerate(plans):
        cursor: dict[str, int] = {}
        for step in range(global_step):
            for slot in range(rank, 64, world_size):
                prompt_id = prompt_ids[(step * 64 + slot) % len(prompt_ids)]
                cursor[prompt_id] = cursor.get(prompt_id, 0) + 1
        prompt_states.append(
            {
                "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
                "global_batch_size": 64,
                "max_microbatch_size": 4,
                "global_slot_cursor": global_step * 64,
                "microbatch_index": 0,
                "rank": rank,
                "world_size": world_size,
            }
        )
        source_states.append(
            {
                "kind": "teacher_demo",
                "cursor_protocol_id": "teacher-demo-round-robin-v2-accepted-view",
                "cursor": cursor,
                "attempt_ids": attempt_ids,
                "rank": rank,
                "world_size": world_size,
            }
        )
        local_sequences = global_step * plan.local_sequence_count
        trainer_states.append(
            {
                "cumulative_counts": {
                    "prompts_consumed": float(local_sequences),
                    "trajectories_generated": float(local_sequences),
                    "response_tokens_generated": float(local_sequences),
                    "supervised_response_tokens": float(local_sequences),
                    "model_facing_input_tokens_processed": float(budget["consumed"]),
                    "forward_backward_flop_estimate": float(local_sequences),
                },
                "accumulation_micro_step": 0,
                "token_budget": copy.deepcopy(budget),
                "batch_partition": copy.deepcopy(partition),
                "rank": rank,
                "world_size": world_size,
            }
        )
    return {
        "world_size": world_size,
        "global_step": global_step,
        "optimizer": {
            "state": {0: {"step": global_step}},
            "param_groups": [{"params": [0]}],
        },
        "scheduler": {
            "last_epoch": global_step,
            "_step_count": global_step + 1,
        },
        "policy_version": 0,
        "online_rollout_round": 0,
        "prompt_scheduler": prompt_states[0],
        "prompt_scheduler_by_rank": prompt_states,
        "state_source": source_states[0],
        "state_source_by_rank": source_states,
        "trainer_state": trainer_states[0],
        "trainer_state_by_rank": trainer_states,
        "token_budget": budget,
    }


@pytest.mark.unit
def test_exact_checkpoint_finalization_covers_all_world_sizes_and_rejects_tampering() -> None:
    for world_size in (1, 2, 3, 4):
        payload = _exact_checkpoint_fixture(world_size)
        assert _validate_factorial_rank_states(
            payload,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        ) == world_size
        assert (
            _validate_exact_checkpoint(
                payload,
                name="fixture",
                world_size=world_size,
                max_steps=120,
            )
            is payload
        )

        production = _exact_checkpoint_fixture(world_size)
        production["format"] = "accelerate_fsdp_full_export_v1"
        strategy_contract = {
            "requested_fsdp_sharding_strategy": "FULL_SHARD",
            "effective_fsdp_sharding_strategy": (
                "NO_SHARD" if world_size == 1 else "FULL_SHARD"
            ),
            "fsdp_wrapper_count": 29,
        }
        production["trainer_state"]["batch_partition"].update(strategy_contract)  # type: ignore[index,union-attr]
        for trainer_state in production["trainer_state_by_rank"]:  # type: ignore[union-attr]
            trainer_state["batch_partition"].update(strategy_contract)
        assert _validate_factorial_rank_states(
            production,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        ) == world_size

        wrong_strategy = copy.deepcopy(production)
        wrong_strategy["trainer_state"]["batch_partition"][  # type: ignore[index,union-attr]
            "effective_fsdp_sharding_strategy"
        ] = "FULL_SHARD" if world_size == 1 else "NO_SHARD"
        with pytest.raises(ValueError, match="exact batch partition"):
            _validate_factorial_rank_states(
                wrong_strategy,
                expected_max_steps=120,
                expected_max_model_input_length=1536,
            )

    tampered_cursor = _exact_checkpoint_fixture(3)
    tampered_cursor["state_source_by_rank"][1]["cursor"] = {"not-a-prompt": 21}  # type: ignore[index]
    with pytest.raises(ValueError, match="exact prompt schedule"):
        _validate_factorial_rank_states(
            tampered_cursor,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        )

    over_limit = _exact_checkpoint_fixture(3, global_step=121)
    with pytest.raises(ValueError, match="optimizer-step limit"):
        _validate_factorial_rank_states(
            over_limit,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        )

    downgraded = _exact_checkpoint_fixture(3)
    downgraded["trainer_state"].pop("batch_partition")  # type: ignore[union-attr]
    downgraded.pop("trainer_state_by_rank")
    with pytest.raises(ValueError, match="lacks its batch partition"):
        _validate_factorial_rank_states(
            downgraded,
            expected_batch_partition_protocol=ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        )

    nonzero_policy = _exact_checkpoint_fixture(3)
    nonzero_policy["policy_version"] = 1
    with pytest.raises(ValueError, match="nonzero policy or rollout"):
        _validate_factorial_rank_states(
            nonzero_policy,
            expected_max_steps=120,
            expected_max_model_input_length=1536,
        )


@pytest.mark.unit
def test_allocation_neutral_prompt_scheduler_checkpoint_is_rank_and_world_bound() -> None:
    ids = [f"p-{index}" for index in range(128)]
    texts = [f"prompt-{index}" for index in range(128)]
    scheduler = PromptScheduler.for_allocation_neutral_rank(
        ids,
        texts,
        rank=0,
        world_size=3,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    for _ in range(6):
        scheduler.next_batch()
    state = scheduler.state_dict()
    restored = PromptScheduler.for_allocation_neutral_rank(
        ids,
        texts,
        rank=0,
        world_size=3,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    restored.load_state_dict(state)
    assert restored.next_batch() == scheduler.next_batch()
    cross_world = PromptScheduler.for_allocation_neutral_rank(
        ids,
        texts,
        rank=0,
        world_size=2,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    with pytest.raises(ValueError, match="different allocation batch partition"):
        cross_world.load_state_dict(state)


@pytest.mark.unit
def test_qwen3_artifacts_and_run_manifests_fail_closed_on_cross_model_or_tamper() -> None:
    config = compose_config(
        ["production=qwen3_primary", "model=qwen3_1p7b", "teacher=qwen3_teacher_8b"],
        config_root=Path("configs"),
    )
    expected_manifest = {
        "protocol_track": "qwen3_v1",
        "artifact_namespace": "qwen3-v1",
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
        "tokenizer_hash": config["model"]["tokenizer_fingerprint"],
        "behavior_policy": {"id": "Qwen/Qwen3-1.7B"},
    }
    _require_qwen3_store_binding(
        expected_manifest,
        config=config,
        expected_behavior_policy="Qwen/Qwen3-1.7B",
    )
    stale = copy.deepcopy(expected_manifest)
    stale["behavior_policy"]["id"] = "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ValueError, match="stale/cross-model"):
        _require_qwen3_store_binding(
            stale,
            config=config,
            expected_behavior_policy="Qwen/Qwen3-1.7B",
        )

    config_binding = bind_config(
        {"seed": 42},
        input_artifact_hashes={},
        execution_context={"entrypoint": "unit-test"},
    )
    experiment_binding = {
        "fixture": "qwen3-run-binding-v3",
        "factorial_design_sha256": "f" * 64,
        "scientific_config_sha256": config_binding.scientific_config_sha256,
    }
    manifest = RunManifest(
        run_id="qwen3-run",
        experiment_cell="offline_soft",
        seed=42,
        model_id="Qwen/Qwen3-1.7B",
        model_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        tokenizer_id="Qwen/Qwen3-1.7B",
        tokenizer_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        resolved_model_commit="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        resolved_tokenizer_commit="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        dataset_hashes={"train": "hash"},
        rollout_bank_hash="bank",
        prompt_schedule_hash="schedule",
        experiment_binding=experiment_binding,
        experiment_binding_sha256=sha256_value(experiment_binding),
        factorial_design_sha256="f" * 64,
        config_binding=config_binding.as_dict(),
        resolved_config_sha256=config_binding.resolved_config_sha256,
        scientific_config_sha256=config_binding.scientific_config_sha256,
        execution_config_sha256=config_binding.execution_config_sha256,
        execution_context=config_binding.as_dict()["execution_context"],
        resolved_config_yaml_sha256="0" * 64,
        raw_prompt_schedule_hash="1" * 64,
        model_facing_prompt_schedule_hash="2" * 64,
        prompt_protocol="qwen3_non_thinking_v1",
        enable_thinking=False,
        chat_template_sha256="3" * 64,
        tokenizer_fingerprint="4" * 64,
        protocol_track="qwen3_v1",
        artifact_namespace="qwen3-v1",
        prereg_version="qwen3_v1",
        prereg_path="prereg/qwen3_v1.yaml",
    )
    payload = run_manifest_payload(manifest)
    validate_run_manifest_payload(payload)
    payload["model_revision"] = "tampered"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_run_manifest_payload(payload)
    assert pilot_hash_valid({"phase": "pilot-input", "sha256": "bad"}) is False


@pytest.mark.unit
def test_qwen3_launch_paths_require_explicit_configs_and_contain_no_qwen25_fallback() -> None:
    paths = (
        Path("scripts/production/run_qwen3_gpu_preflight.sh"),
        Path("scripts/production/run_qwen3_g0.sh"),
        Path("scripts/production/submit_qwen3_pilot.sh"),
        Path("scripts/slurm/qwen3_gpu_preflight.slurm"),
    )
    required = {
        "MODEL_CONFIG",
        "TEACHER_CONFIG",
        "PRODUCTION_CONFIG",
        "G0_CONFIG",
        "PILOT_CONFIG",
        "PROJECT_ROOT",
        "PYTHON_BIN",
        "ACCELERATE_BIN",
        "OUTPUT_ROOT",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "qwen25" not in text.lower()
        assert required <= {name for name in required if name in text}
