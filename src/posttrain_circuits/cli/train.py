"""Run one controlled factorial cell or canonical SFT."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.artifacts.checkpoints import checkpoint_runtime_state_hashes
from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import publish_json_once
from posttrain_circuits.artifacts.runs import (
    PROTOCOL_AMENDMENT_BINDING_FIELDS,
    RunManifest,
    finalize_run_directory,
    formal_artifact_binding,
    git_output,
    initialize_run_directory,
)
from posttrain_circuits.causal_circuits.model.runner import load_checkpoint_into_hf_model
from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.core.config import is_production_scale
from posttrain_circuits.core.readiness import require_factorial_prerequisites
from posttrain_circuits.core.seeding import seed_everything
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.datasets.teacher_demos.contracts import TeacherDemoAttempt
from posttrain_circuits.datasets.teacher_demos.store import read_teacher_demo_store
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.experiments.protocols.specs import (
    build_experiment_binding,
    validate_run_manifest_experiment_binding,
)
from posttrain_circuits.methods.registry import get_method_spec
from posttrain_circuits.methods.sft import CANONICAL_SFT_METHOD
from posttrain_circuits.methods.specs import FACTORIAL_TRAINING_BACKEND
from posttrain_circuits.models.loading import (
    LoadedModel,
    assert_tokenizer_compatible,
    load_model_and_tokenizer,
    move_model_to_local_cuda,
)
from posttrain_circuits.models.prompt_protocol import format_model_prompts, prompt_schedule_hashes
from posttrain_circuits.learning.state_sources.generation import build_proofgraph_hf_generator
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_example
from posttrain_circuits.learning.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.learning.training.evaluation import build_proofgraph_evaluator
from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
from posttrain_circuits.learning.training.factories import build_state_source, build_supervisor
from posttrain_circuits.learning.training.optimizer import build_adamw
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    LEGACY_BATCH_PARTITION_PROTOCOL,
    PromptScheduler,
)
from posttrain_circuits.learning.training.local_fork import state_hash
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    scripted_current_policy_generator,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


_QWEN3_V2_G0_PROMPT_POPULATION_SIZE = 256


def _validate_allocation_neutral_prompt_population(
    prompt_ids: list[str],
    *,
    global_batch_size: int,
    expected_prompt_ids: list[str],
    expected_prompt_count: int | None = None,
) -> None:
    """Fail closed when rank-local teacher cursors could change sample order.

    The accepted teacher-demo cursor is rank local.  Allocation-neutral sample
    order therefore additionally requires one unique global prompt population
    whose length is an exact multiple of the 64-slot optimizer window.  The
    production G0 request freezes that population at 256 prompts.
    """

    if type(global_batch_size) is not int or global_batch_size < 1:
        raise ValueError("allocation-neutral global batch size must be a positive integer")
    if not prompt_ids or any(
        not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids
    ):
        raise ValueError("allocation-neutral training requires non-empty string prompt IDs")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("allocation-neutral training requires unique global prompt IDs")
    if len(prompt_ids) % global_batch_size != 0:
        raise ValueError(
            "allocation-neutral prompt population must be an exact multiple of "
            "the global batch size"
        )
    if expected_prompt_count is not None and len(prompt_ids) != expected_prompt_count:
        raise ValueError(
            "production Qwen3-v2 G0 requires exactly "
            f"{expected_prompt_count} configured train prompts"
        )
    if prompt_ids != expected_prompt_ids:
        raise ValueError(
            "allocation-neutral accepted-view prompt order differs from the configured "
            "train-family order"
        )


def _trajectory_prompts(
    records: list[TrajectoryRecord],
) -> tuple[list[str], list[str]]:
    by_prompt: dict[str, str] = {}
    for record in records:
        by_prompt.setdefault(record.prompt_id, record.raw_prompt_text or record.prompt_text)
    if not by_prompt:
        raise ValueError("teacher-demo store has no prompts")
    return list(by_prompt), list(by_prompt.values())


def _teacher_demo_prompts(
    attempts: list[TeacherDemoAttempt],
) -> tuple[list[str], list[str]]:
    by_prompt: dict[str, str] = {}
    for attempt in attempts:
        by_prompt.setdefault(attempt.prompt_id, attempt.raw_prompt_text)
    if not by_prompt:
        raise ValueError("teacher-demo accepted view has no prompts")
    return list(by_prompt), list(by_prompt.values())


def _validate_teacher_demo_model_input_lengths(
    attempts: list[TeacherDemoAttempt],
    manifest: dict[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    max_model_input_length: int,
) -> dict[str, int | str]:
    """Bind the frozen demo store to the reviewed one-GPU sequence envelope.

    No truncation is scientifically acceptable here: prompt and response IDs
    are already part of each accepted demonstration's identity.  The complete
    store is checked before any training forward, while the trainer separately
    checks every materialized supervision tensor.
    """

    limits = (max_prompt_tokens, max_new_tokens, max_model_input_length)
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("teacher-demo model-input length limits must be positive integers")
    if max_prompt_tokens + max_new_tokens > max_model_input_length:
        raise ValueError("teacher-demo prompt/response bounds exceed the model-input limit")
    generation = manifest.get("teacher_demo_generation")
    if not isinstance(generation, dict) or (
        generation.get("max_prompt_tokens") != max_prompt_tokens
        or generation.get("max_new_tokens") != max_new_tokens
    ):
        raise ValueError(
            "teacher-demo generation length contract differs from the reviewed G0 config"
        )
    if not attempts:
        raise ValueError("teacher-demo accepted view is empty")
    observed_prompt = 0
    observed_response = 0
    observed_total = 0
    for attempt in attempts:
        prompt_tokens = len(attempt.input_ids)
        response_tokens = len(attempt.response_ids)
        total_tokens = prompt_tokens + response_tokens
        if prompt_tokens > max_prompt_tokens:
            raise ValueError(
                f"teacher-demo prompt exceeds max_prompt_tokens: {attempt.attempt_id}"
            )
        if response_tokens > max_new_tokens:
            raise ValueError(
                f"teacher-demo response exceeds max_new_tokens: {attempt.attempt_id}"
            )
        if total_tokens > max_model_input_length:
            raise ValueError(
                f"teacher-demo model input exceeds max_model_input_length: {attempt.attempt_id}"
            )
        observed_prompt = max(observed_prompt, prompt_tokens)
        observed_response = max(observed_response, response_tokens)
        observed_total = max(observed_total, total_tokens)
    return {
        "max_model_input_length": max_model_input_length,
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "observed_max_model_input_tokens": observed_total,
        "observed_max_prompt_tokens": observed_prompt,
        "observed_max_response_tokens": observed_response,
        "overlength_policy": "reject_without_truncation_before_any_training_forward",
    }


def _require_qwen3_store_binding(
    manifest: dict[str, Any],
    *,
    config: dict[str, Any],
    expected_behavior_policy: str,
) -> None:
    if not str(config.get("protocol_track", "")).startswith("qwen3_"):
        return
    expected = {
        "protocol_track": config["protocol_track"],
        "artifact_namespace": config["model"]["artifact_namespace"],
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
        "tokenizer_hash": config["model"]["tokenizer_fingerprint"],
    }
    if config.get("protocol_track") == "qwen3_v2":
        binding = formal_artifact_binding(config)
        expected.update(
            {
                "prereg_path": binding["prereg_path"],
                "prereg_version": binding["prereg_version"],
                "prereg_commit": binding["prereg_commit"],
                "prereg_sha256": binding["prereg_sha256"],
                "code_commit": binding["code_commit"],
                **{
                    key: binding[key]
                    for key in PROTOCOL_AMENDMENT_BINDING_FIELDS
                    if key in binding
                },
            }
        )
    protocol_bindings = manifest.get("protocol_bindings")
    observed_bindings = dict(manifest)
    if isinstance(protocol_bindings, dict):
        observed_bindings.update(protocol_bindings)
    mismatches = {
        key: {"expected": value, "observed": observed_bindings.get(key)}
        for key, value in expected.items()
        if observed_bindings.get(key) != value
    }
    observed_policy = str(manifest.get("behavior_policy", {}).get("id", ""))
    if observed_policy != expected_behavior_policy:
        mismatches["behavior_policy.id"] = {
            "expected": expected_behavior_policy,
            "observed": observed_policy,
        }
    if mismatches:
        raise ValueError(f"Qwen3 refused stale/cross-model trajectory artifact: {mismatches}")


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Train one controlled post-training cell", argv)
    cell = str(config["experiment"]["name"])
    output = args.output or Path(config["output_root"]) / "runs" / cell
    production_scale = is_production_scale(config)
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return
    method = get_method_spec(cell)
    if method.training_backend != FACTORIAL_TRAINING_BACKEND or not (
        method.role == "factorial_cell" or method is CANONICAL_SFT_METHOD
    ):
        raise ValueError(f"train CLI does not execute registered method {cell!r}")
    prerequisite_evidence: dict[str, Any] = {}
    if production_scale and (
        method.role == "factorial_cell"
        or (
            method is CANONICAL_SFT_METHOD
            and "pilot_profile" in config
        )
    ):
        prerequisite_evidence = require_factorial_prerequisites(config)
    seed = int(config["seed"])
    seed_everything(seed)
    local_model = str(config["model"]["model_name_or_path"]).startswith("local/")
    loaded_student: LoadedModel | None = None
    if local_model:
        tokenizer = build_tiny_tokenizer()
        model = build_tiny_qwen(seed)
    else:
        loaded_student = load_model_and_tokenizer(
            config["model"],
            for_training=True,
        )
        tokenizer = loaded_student.tokenizer
        model = loaded_student.model
    if not local_model:
        assert loaded_student is not None
    initial_checkpoint_hash: str | None = None
    if production_scale:
        initial_checkpoint_value = str(
            config.get("production_safety", {}).get("initial_checkpoint_path", "")
        ).strip()
        expected_initial_hash = str(
            config.get("production_safety", {}).get("initial_checkpoint_hash", "")
        ).strip()
        if not initial_checkpoint_value or len(expected_initial_hash) != 64:
            raise ValueError("production training requires initial_checkpoint_path and its SHA-256")
        initial_checkpoint = Path(initial_checkpoint_value)
        initial_checkpoint_hash = load_checkpoint_into_hf_model(
            model,
            initial_checkpoint,
            expected_sha256=expected_initial_hash,
        )
    else:
        initial_checkpoint_hash = state_hash(model.state_dict())
    state_source_name = str(config["state_source"]["name"])
    supervision_name = str(config["supervision"]["name"])
    trainer_settings = config["trainer"]
    batch_partition_protocol = str(
        trainer_settings.get("batch_partition_protocol", LEGACY_BATCH_PARTITION_PROTOCOL)
    )
    global_batch_size: int | None = None
    max_microbatch_size: int | None = None
    max_model_input_length: int | None = None
    if batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
        global_batch_size = trainer_settings.get("global_batch_size")
        max_microbatch_size = trainer_settings.get("max_microbatch_size")
        max_model_input_length = trainer_settings.get("max_model_input_length")
        if any(
            type(value) is not int
            for value in (
                global_batch_size,
                max_microbatch_size,
                max_model_input_length,
            )
        ):
            raise ValueError(
                "allocation-neutral training requires integer global_batch_size, "
                "max_microbatch_size, and max_model_input_length"
            )
        if state_source_name != "teacher_demo" or supervision_name != "canonical_sft":
            raise ValueError(
                "allocation-neutral exact-global batching is reviewed only for the deterministic "
                "teacher_demo/canonical_sft training path"
            )
        # The physical prompt batches are planned from the scheduler-owned
        # runtime world size below; there is no fixed per-rank batch size.
        batch_size = max_microbatch_size
    elif batch_partition_protocol == LEGACY_BATCH_PARTITION_PROTOCOL:
        batch_size = int(trainer_settings["batch_size"])
    else:
        raise ValueError(f"unsupported trainer batch_partition_protocol {batch_partition_protocol!r}")
    family_path = Path(str(config["task"].get("dataset_family_path", "")))
    family = load_dataset_family(family_path)
    family_train = family.examples("train")
    train_count = int(config["task"].get("num_examples", len(family_train)))
    if train_count < 1 or train_count > len(family_train):
        raise ValueError("task.num_examples is outside the frozen train family split")
    examples = family_train[:train_count]
    family_train_ids = {example.example_id for example in family_train}
    validation_examples = family.examples("validation")
    validation_limit = int(
        config["trainer"].get("validation_examples", len(validation_examples))
    )
    if validation_limit < 1 or validation_limit > len(validation_examples):
        raise ValueError("trainer.validation_examples is outside the frozen validation family split")
    validation_examples = validation_examples[:validation_limit]
    fixed_bank: list[TrajectoryRecord] | None = None
    fixed_bank_manifest: dict[str, Any] | None = None
    teacher_demos: list[TeacherDemoAttempt] | None = None
    teacher_demo_manifest: dict[str, Any] | None = None
    teacher_demo_length_contract: dict[str, int | str] | None = None
    current_generator = None
    if state_source_name == "teacher_demo":
        store_path = Path(str(config["state_source"]["store_path"]))
        teacher_demos, loaded_manifest = read_teacher_demo_store(
            store_path,
            require_formal=production_scale,
        )
        teacher_demo_manifest = dict(loaded_manifest)
        _require_qwen3_store_binding(
            teacher_demo_manifest,
            config=config,
            expected_behavior_policy=str(
                config["teacher"].get(
                    "model_name_or_path",
                    config["teacher"].get("teacher_id", ""),
                )
            ),
        )
        if (
            loaded_student is not None
            and teacher_demo_manifest.get("tokenizer_hash") != loaded_student.tokenizer_hash
        ):
            raise ValueError("teacher-demo tokenizer does not match the student tokenizer")
        expected_prompt_ids = [example.example_id for example in examples]
        observed_prompt_ids = {attempt.prompt_id for attempt in teacher_demos}
        if list(teacher_demo_manifest.get("ordered_prompt_ids", [])) != expected_prompt_ids:
            raise ValueError(
                "teacher-demo ordered prompt population differs from the configured train family"
            )
        if observed_prompt_ids != set(expected_prompt_ids):
            missing = set(expected_prompt_ids) - observed_prompt_ids
            extra = observed_prompt_ids - set(expected_prompt_ids)
            raise ValueError(
                "teacher-demo accepted view does not exactly cover configured train prompts: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
            assert max_model_input_length is not None
            teacher_demo_length_contract = _validate_teacher_demo_model_input_lengths(
                teacher_demos,
                teacher_demo_manifest,
                max_prompt_tokens=config["state_source"].get("max_prompt_tokens"),
                max_new_tokens=config["state_source"].get("max_new_tokens"),
                max_model_input_length=max_model_input_length,
            )
        prompt_ids, prompt_texts = _teacher_demo_prompts(teacher_demos)
    else:
        prompt_ids = [example.example_id for example in examples]
        prompt_texts = [render_example(example) for example in examples]
        if state_source_name == "fixed_bank":
            if local_model:
                smoke_store_path = Path(str(config["state_source"].get("store_path", "")))
                if (smoke_store_path / "manifest.json").is_file():
                    fixed_store = TrajectoryStore(smoke_store_path)
                    fixed_bank_manifest = fixed_store.check_integrity()
                    fixed_bank = fixed_store.read()
                    expected_tokenizer_hash = sha256_value(tokenizer.get_vocab())
                    if fixed_bank_manifest.get("tokenizer_hash") != expected_tokenizer_hash:
                        raise ValueError("smoke rollout-bank tokenizer does not match the student tokenizer")
                    unknown_prompt_ids = {
                        record.prompt_id for record in fixed_bank
                    } - family_train_ids
                    if unknown_prompt_ids:
                        raise ValueError(
                            "rollout-bank prompt IDs are outside the configured train family: "
                            f"{sorted(unknown_prompt_ids)}"
                        )
                    prompt_ids, prompt_texts = _trajectory_prompts(fixed_bank)
                else:
                    fixed_bank = build_fixed_bank(examples, tokenizer, seed)
            else:
                assert loaded_student is not None
                fixed_store = TrajectoryStore(Path(str(config["state_source"]["store_path"])))
                fixed_bank_manifest = fixed_store.check_integrity()
                fixed_bank = fixed_store.read()
                _require_qwen3_store_binding(
                    fixed_bank_manifest,
                    config=config,
                    expected_behavior_policy=str(config["model"]["model_name_or_path"]),
                )
                if fixed_bank_manifest.get("tokenizer_hash") != loaded_student.tokenizer_hash:
                    raise ValueError("rollout-bank tokenizer does not match the student tokenizer")
                unknown_prompt_ids = {
                    record.prompt_id for record in fixed_bank
                } - family_train_ids
                if unknown_prompt_ids:
                    raise ValueError(
                        "rollout-bank prompt IDs are outside the configured train family: "
                        f"{sorted(unknown_prompt_ids)}"
                    )
                prompt_ids, prompt_texts = _trajectory_prompts(fixed_bank)
        elif state_source_name == "current_policy":
            if local_model:
                current_generator = scripted_current_policy_generator(
                    examples,
                    tokenizer,
                )
            else:
                assert loaded_student is not None
                current_generator = build_proofgraph_hf_generator(
                    tokenizer=tokenizer,
                    examples_by_id={example.example_id: example for example in examples},
                    max_new_tokens=int(config["trainer"]["max_completion_length"]),
                    temperature=float(config["state_source"]["temperature"]),
                    top_p=float(config["state_source"]["top_p"]),
                    top_k=int(config["state_source"].get("top_k", 0)),
                    min_p=float(config["state_source"].get("min_p", 0.0)),
                    policy_id=loaded_student.model_id,
                    initial_policy_revision=(loaded_student.resolved_model_commit),
                    model_config=config["model"],
                )

    global_prompt_ids = list(prompt_ids)
    global_prompt_texts = list(prompt_texts)
    if batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
        assert global_batch_size is not None
        _validate_allocation_neutral_prompt_population(
            global_prompt_ids,
            global_batch_size=global_batch_size,
            expected_prompt_ids=[example.example_id for example in examples],
            expected_prompt_count=(
                _QWEN3_V2_G0_PROMPT_POPULATION_SIZE
                if production_scale and config.get("protocol_track") == "qwen3_v2"
                else None
            ),
        )
    launch_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
        assert global_batch_size is not None
        assert max_microbatch_size is not None
        prompt_scheduler = PromptScheduler.for_allocation_neutral_rank(
            global_prompt_ids,
            global_prompt_texts,
            rank=launch_rank,
            world_size=world_size,
            global_batch_size=global_batch_size,
            max_microbatch_size=max_microbatch_size,
        )
    else:
        prompt_scheduler = PromptScheduler.for_distributed_rank(
            global_prompt_ids,
            global_prompt_texts,
            min(batch_size, len(global_prompt_ids)),
            rank=launch_rank,
            world_size=world_size,
        )
    state_source = build_state_source(
        config["state_source"],
        fixed_bank=fixed_bank,
        current_generator=current_generator,
        teacher_demos=teacher_demos,
        seed=seed,
    )
    supervisor = build_supervisor(
        config["supervision"],
        pad_token_id=tokenizer.pad_token_id,
    )

    teacher: HuggingFaceTeacherScorer | None = None
    loaded_teacher: LoadedModel | None = None
    if supervision_name in {"hard_teacher", "soft_teacher"}:
        if local_model:
            teacher_model = build_tiny_qwen(seed + 1)
            teacher_id_value = "local/tiny-teacher"
            teacher_revision_value = "local-random-v1"
        elif (
            config.get("protocol_track") == "qwen3_v2"
            and world_size > 1
            and bool(config["teacher"].get("rank_zero_only_training_load"))
        ):
            if launch_rank == 0:
                loaded_teacher = load_model_and_tokenizer(
                    config["teacher"],
                    for_training=False,
                )
                assert_tokenizer_compatible(tokenizer, loaded_teacher.tokenizer)
                teacher_model = move_model_to_local_cuda(loaded_teacher.model)
                teacher_id_value = loaded_teacher.model_id
                teacher_revision_value = loaded_teacher.resolved_model_commit
            else:
                teacher_model = None
                teacher_id_value = str(config["teacher"]["model_name_or_path"])
                teacher_revision_value = str(config["teacher"]["model_revision"])
        else:
            loaded_teacher = load_model_and_tokenizer(
                config["teacher"],
                for_training=False,
            )
            assert_tokenizer_compatible(
                tokenizer,
                loaded_teacher.tokenizer,
            )
            teacher_model = move_model_to_local_cuda(loaded_teacher.model)
            teacher_id_value = loaded_teacher.model_id
            teacher_revision_value = loaded_teacher.resolved_model_commit
        teacher = HuggingFaceTeacherScorer(
            teacher_model,
            teacher_id=teacher_id_value,
            teacher_revision=teacher_revision_value,
            top_k=min(
                int(config["supervision"].get("teacher_top_k", 128)),
                model.config.vocab_size,
            ),
            minimum_retained_mass=float(config["supervision"].get("minimum_retained_mass", 0.0)),
            fail_below_mass=bool(
                config["supervision"].get("fail_below_minimum_retained_mass", False)
            ),
            distributed_rank_zero=(
                config.get("protocol_track") == "qwen3_v2"
                and world_size > 1
                and bool(config["teacher"].get("rank_zero_only_training_load"))
            ),
        )

    optimizer = build_adamw(
        model.parameters(),
        learning_rate=float(config["trainer"]["learning_rate"]),
        weight_decay=float(config["trainer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    run_id = f"{cell}-seed{seed}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    dataset_hashes = {
        "family_manifest": str(family.manifest["sha256"]),
        "train": str(family.boundary("train")["examples_file_sha256"]),
        "validation": str(family.boundary("validation")["examples_file_sha256"]),
    }
    if initial_checkpoint_hash is not None:
        dataset_hashes["initial_checkpoint"] = initial_checkpoint_hash
    for name, evidence in prerequisite_evidence.items():
        dataset_hashes[f"prerequisite_{name}"] = str(
            evidence.get("report_hash", evidence.get("manifest_hash"))
        )
    probe_manifest_hash = str(
        prerequisite_evidence.get("probe_cohorts", {}).get(
            "manifest_hash",
            sha256_value(
                {
                    "kind": "deterministic_smoke_probe",
                    "validation": dataset_hashes["validation"],
                    "seed": seed,
                }
            ),
        )
    )
    dataset_hashes["probe_manifest"] = probe_manifest_hash
    if method.requires_common_rollout_bank and fixed_bank_manifest is None:
        if not fixed_bank:
            raise ValueError("offline method has no rollout-bank content to bind")
        in_memory_content_hash = sha256_value([asdict(record) for record in fixed_bank])
        fixed_bank_manifest = {
            "sha256": sha256_value(
                {
                    "kind": "deterministic_smoke_rollout_bank",
                    "content": in_memory_content_hash,
                }
            ),
            "files": {"in-memory-records": in_memory_content_hash},
        }
    if fixed_bank_manifest is not None:
        source_hash = str(fixed_bank_manifest["sha256"])
    elif fixed_bank is not None:
        source_hash = sha256_value([asdict(record) for record in fixed_bank])
    elif teacher_demo_manifest is not None:
        source_hash = str(teacher_demo_manifest["sha256"])
    else:
        source_hash = sha256_value(
            {
                "state_source": config["state_source"],
                "generator": (
                    "scripted_current_policy_generator" if local_model else "hf_generate_trajectories"
                ),
                "seed": seed,
            }
        )
    model_config = config["model"]
    prompt_schedule_hash = sha256_value(
        {
            "prompt_ids": global_prompt_ids,
            "prompt_texts": global_prompt_texts,
        }
    )
    formatted_schedule = format_model_prompts(global_prompt_texts, tokenizer, model_config)
    raw_prompt_schedule_hash, model_facing_prompt_schedule_hash = prompt_schedule_hashes(
        formatted_schedule
    )

    teacher_config = config["teacher"]
    teacher_id = str(
        teacher_config.get("model_name_or_path", teacher_config.get("teacher_id", ""))
    )
    teacher_revision = str(
        teacher_config.get("model_revision", teacher_config.get("teacher_revision", ""))
    )
    resolved_teacher_commit = str(
        teacher_config.get(
            "model_revision",
            teacher_config.get("resolved_teacher_commit", teacher_revision),
        )
    )
    teacher_demo_generation: dict[str, Any] | None = None
    if loaded_teacher is not None:
        resolved_teacher_commit = loaded_teacher.resolved_model_commit
    elif teacher is not None:
        resolved_teacher_commit = teacher.teacher_revision
    elif teacher_demo_manifest is not None:
        teacher_demo_generation = dict(teacher_demo_manifest["teacher_demo_generation"])
        teacher_id = str(teacher_demo_generation["teacher_id"])
        teacher_revision = str(teacher_demo_generation["teacher_revision"])
        resolved_teacher_commit = str(teacher_demo_generation["resolved_teacher_commit"])

    model_revision = str(model_config["model_revision"])
    tokenizer_revision = str(model_config["tokenizer_revision"])
    resolved_model_commit_value = (
        loaded_student.resolved_model_commit if loaded_student is not None else model_revision
    )
    resolved_tokenizer_commit_value = (
        loaded_student.resolved_tokenizer_commit if loaded_student is not None else tokenizer_revision
    )
    tokenizer_fingerprint_value = (
        loaded_student.tokenizer_hash
        if loaded_student is not None
        else sha256_value(tokenizer.get_vocab())
    )
    full_parameter_training = all(parameter.requires_grad for parameter in model.parameters())
    offline_cursor = None
    if method.requires_common_rollout_bank:
        source_state = state_source.state_dict()
        offline_cursor = source_state.get("cursor")
        if not isinstance(offline_cursor, dict):
            raise ValueError("offline rollout-bank state source lacks its initial cursor")
    task_protocol = {
        key: value
        for key, value in config["task"].items()
        if not any(marker in key.lower() for marker in ("path", "directory", "output_root"))
    }
    task_protocol.update(
        generator_version=ProofGraphTask.generator_version,
        label_semantics=ProofGraphTask.label_semantics,
    )
    config_input_hashes = {
        "prereg_path": sha256_file(Path(str(config["prereg_path"]))),
        "task.dataset_family_path": str(family.manifest["sha256"]),
    }
    amendment_path = str(config.get("protocol_amendment_path", "")).strip()
    if amendment_path:
        config_input_hashes["protocol_amendment_path"] = sha256_file(
            Path(amendment_path)
        )
    initial_checkpoint_locator = str(
        config.get("production_safety", {}).get("initial_checkpoint_path", "")
    ).strip()
    if initial_checkpoint_locator:
        config_input_hashes["production_safety.initial_checkpoint_path"] = str(
            initial_checkpoint_hash
        )
    if fixed_bank_manifest is not None:
        config_input_hashes["state_source.store_path"] = str(
            fixed_bank_manifest["sha256"]
        )
    elif teacher_demo_manifest is not None:
        config_input_hashes["state_source.store_path"] = str(
            teacher_demo_manifest["sha256"]
        )
    prerequisite_locator_bindings = {
        "full_readiness": ("production_safety.readiness_report", "report_hash"),
        "anti_shortcut": ("anti_shortcut.report_path", "report_hash"),
        "probe_cohorts": (
            "production_safety.probe_cohort_manifest",
            "manifest_hash",
        ),
    }
    for evidence_name, (locator_name, hash_name) in prerequisite_locator_bindings.items():
        evidence = prerequisite_evidence.get(evidence_name)
        if isinstance(evidence, dict) and evidence.get(hash_name):
            config_input_hashes[locator_name] = str(evidence[hash_name])
    execution_context = {
        "entrypoint": "train",
        "output": str(Path(output).absolute()),
        "backend": str(config["trainer"]["backend"]),
        "world_size": world_size,
        "resume": str(args.resume.absolute()) if args.resume is not None else None,
    }
    config_binding = bind_config(
        config,
        input_artifact_hashes=config_input_hashes,
        execution_context=execution_context,
    )
    experiment_binding = build_experiment_binding(
        config=config,
        config_binding=config_binding,
        implementation_commit=git_output(["rev-parse", "HEAD"]) or "unavailable",
        implementation_dirty=bool(git_output(["status", "--porcelain"]) or ""),
        model_resolved_revision=resolved_model_commit_value,
        tokenizer_resolved_revision=resolved_tokenizer_commit_value,
        tokenizer_fingerprint=tokenizer_fingerprint_value,
        teacher_resolved_revision=resolved_teacher_commit,
        initial_checkpoint_sha256=str(initial_checkpoint_hash),
        train_dataset_sha256=dataset_hashes["train"],
        validation_dataset_sha256=dataset_hashes["validation"],
        model_facing_prompt_schedule_sha256=model_facing_prompt_schedule_hash,
        probe_manifest_sha256=probe_manifest_hash,
        task_protocol=task_protocol,
        optimizer_spec={
            "class": "torch.optim.AdamW",
            "learning_rate": optimizer.defaults["lr"],
            "weight_decay": optimizer.defaults["weight_decay"],
            "betas": optimizer.defaults["betas"],
            "eps": optimizer.defaults["eps"],
        },
        scheduler_spec={"class": "torch.optim.lr_scheduler.LambdaLR", "schedule": "constant_1.0"},
        resolved_batch_contract=(
            {
                "execution_backend": config["trainer"]["backend"],
                "batch_partition_protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
                "global_batch_size": global_batch_size,
                "max_microbatch_size": max_microbatch_size,
                "max_model_input_length": max_model_input_length,
                "teacher_demo_length_contract": teacher_demo_length_contract,
                "prompt_assignment": "global_logical_slots_strided_by_runtime_rank_v1",
                "accumulation_schedule": "derived_from_runtime_world_size_v1",
                "steps_per_round": config["trainer"]["steps_per_round"],
            }
            if batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1
            else {
                "execution_backend": config["trainer"]["backend"],
                "world_size": world_size,
                "per_rank_batch_size": batch_size,
                "gradient_accumulation_steps": config["trainer"]["gradient_accumulation_steps"],
                "steps_per_round": config["trainer"]["steps_per_round"],
            }
        ),
        full_parameter_training=full_parameter_training,
        offline_bank_manifest=fixed_bank_manifest,
        offline_bank_initial_cursor=offline_cursor,
        teacher_demo_manifest=teacher_demo_manifest,
    )
    manifest = RunManifest(
        run_id=run_id,
        experiment_cell=cell,
        seed=seed,
        model_id=str(model_config["model_name_or_path"]),
        model_revision=model_revision,
        tokenizer_id=str(model_config["tokenizer_name_or_path"]),
        tokenizer_revision=tokenizer_revision,
        resolved_model_commit=resolved_model_commit_value,
        resolved_tokenizer_commit=resolved_tokenizer_commit_value,
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        resolved_teacher_commit=resolved_teacher_commit,
        teacher_demo_generation=teacher_demo_generation,
        dataset_hashes=dataset_hashes,
        rollout_bank_hash=source_hash,
        prompt_schedule_hash=prompt_schedule_hash,
        experiment_binding=experiment_binding.to_payload(),
        experiment_binding_sha256=experiment_binding.scientific_sha256,
        factorial_design_sha256=experiment_binding.factorial_design_sha256,
        config_binding=config_binding.as_dict(),
        resolved_config_sha256=config_binding.resolved_config_sha256,
        scientific_config_sha256=config_binding.scientific_config_sha256,
        execution_config_sha256=config_binding.execution_config_sha256,
        execution_context=execution_context,
        raw_prompt_schedule_hash=raw_prompt_schedule_hash,
        model_facing_prompt_schedule_hash=model_facing_prompt_schedule_hash,
        prompt_protocol=experiment_binding.prompt_protocol,
        enable_thinking=experiment_binding.enable_thinking,
        chat_template_sha256=experiment_binding.chat_template_sha256,
        tokenizer_fingerprint=experiment_binding.tokenizer_fingerprint,
        protocol_track=str(config.get("protocol_track", "core_v2")),
        artifact_namespace=str(model_config.get("artifact_namespace", "legacy")),
        protocol_teacher_revision=str(config["teacher"].get("model_revision", "unbound")),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get(
                "token_budget_unit", "global_nonpadding_model_input_tokens_processed"
            )
        ),
    )
    validate_run_manifest_experiment_binding(asdict(manifest))
    if launch_rank == 0:
        initialize_run_directory(
            output,
            config,
            config_binding,
            manifest,
            require_git=production_scale,
        )
    trainer_config = TrainerConfig(
        max_steps=int(config["trainer"]["max_steps"]),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get(
                "token_budget_unit", "global_nonpadding_model_input_tokens_processed"
            )
        ),
        steps_per_round=int(config["trainer"]["steps_per_round"]),
        learning_rate=float(config["trainer"]["learning_rate"]),
        checkpoint_every=int(config["trainer"]["checkpoint_every"]),
        evaluation_every=int(config["trainer"].get("evaluation_every", 1)),
        backend=str(config["trainer"]["backend"]),
        gradient_accumulation_steps=int(config["trainer"].get("gradient_accumulation_steps", 1)),
        batch_partition_protocol=batch_partition_protocol,
        global_batch_size=global_batch_size,
        max_microbatch_size=max_microbatch_size,
        max_model_input_length=max_model_input_length,
        max_completion_length=int(config["trainer"]["max_completion_length"]),
        require_evaluation_metrics=production_scale,
    )
    evaluation_completion_length = (
        trainer_config.max_completion_length
        if production_scale
        else min(16, trainer_config.max_completion_length)
    )
    trainer = FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=state_source,
        supervisor=supervisor,
        config=trainer_config,
        run_dir=output,
        teacher=teacher,
        resolved_config=config,
        manifest_hashes={
            "dataset_train": dataset_hashes["train"],
            "dataset_validation": dataset_hashes["validation"],
            "initial_checkpoint": str(initial_checkpoint_hash),
            "state_source": source_hash,
            "prompt_schedule": prompt_schedule_hash,
            "model_facing_prompt_schedule": model_facing_prompt_schedule_hash,
            "probe_manifest": probe_manifest_hash,
            "experiment_binding": experiment_binding.scientific_sha256,
            "factorial_design": experiment_binding.factorial_design_sha256,
            "method_spec": experiment_binding.method_spec_sha256,
        },
        rank_shard_hash=sha256_value(
            {
                "rank": launch_rank,
                "world_size": world_size,
                "prompt_ids": prompt_scheduler.prompt_ids,
            }
        ),
        git_commit=manifest.git_commit,
        implementation_dirty=manifest.dirty_working_tree,
        dependency_versions=manifest.package_versions,
        resume_ancestry=manifest.resume_ancestry,
        probe_input_ids=torch.tensor(
            [
                tokenizer.encode(
                    formatted_schedule[0].model_facing_prompt,
                    add_special_tokens=False,
                )
            ],
            dtype=torch.long,
        ),
        evaluation_fn=build_proofgraph_evaluator(
            validation_examples,
            tokenizer,
            max_completion_length=evaluation_completion_length,
            model_config=config["model"],
        ),
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    history = trainer.train()
    if trainer.is_main_process:
        final_checkpoint = trainer.final_checkpoint_path
        if final_checkpoint is None or not final_checkpoint.is_file():
            raise RuntimeError("training completed without a final optimizer-boundary checkpoint")
        checkpoint_payload = torch.load(
            final_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise RuntimeError("final Factorial checkpoint payload is not a mapping")
        final_token_budget = trainer.token_budget.state_dict()
        if checkpoint_payload.get("token_budget") != final_token_budget:
            raise RuntimeError(
                "final Factorial checkpoint token budget differs from the training manifest state"
            )
        if checkpoint_payload.get("global_step") != trainer.global_step:
            raise RuntimeError(
                "final Factorial checkpoint optimizer step differs from the training manifest state"
            )
        manifest.token_budget_consumed = trainer.token_budget.consumed
        manifest.training_stop_reason = trainer.token_budget.stop_reason
        manifest.metrics_sha256 = sha256_file(output / "metrics.jsonl")
        manifest.final_checkpoint_path = str(final_checkpoint)
        manifest.final_checkpoint_sha256 = sha256_file(final_checkpoint)
        manifest.resume_ancestry = list(trainer.resume_ancestry)
        metric_rows = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        metric_update_rows = [
            {
                "step": row.get("step"),
                "parameter_update_norm": row.get("parameter_update_norm"),
            }
            for row in metric_rows
            if isinstance(row, dict) and row.get("parameter_update_norm") is not None
        ]
        update_evidence: dict[str, Any] = {
            "format_version": 1,
            "checkpoint": str(final_checkpoint.resolve()),
            "final_checkpoint_sha256": manifest.final_checkpoint_sha256,
            "metrics_sha256": manifest.metrics_sha256,
            "metric_update_rows": metric_update_rows,
            "metric_update_rows_sha256": sha256_value(metric_update_rows),
            "global_step": checkpoint_payload.get("global_step"),
            "parameter_update_norm": checkpoint_payload.get("parameter_update_norm"),
            "final_model_state_hash": checkpoint_payload.get("final_model_state_hash"),
            "update_norm_baseline_checkpoint_path": checkpoint_payload.get(
                "update_norm_baseline_checkpoint_path"
            ),
            "update_norm_baseline_checkpoint_sha256": checkpoint_payload.get(
                "update_norm_baseline_checkpoint_sha256"
            ),
            "token_budget": checkpoint_payload.get("token_budget"),
            "resume_ancestry": checkpoint_payload.get("resume_ancestry"),
            "checkpoint_runtime_state_hashes": checkpoint_runtime_state_hashes(
                checkpoint_payload
            ),
            "manifest_hashes": checkpoint_payload.get("manifest_hashes"),
            "experiment_binding_sha256": experiment_binding.scientific_sha256,
            "factorial_design_sha256": experiment_binding.factorial_design_sha256,
            "method_spec_sha256": experiment_binding.method_spec_sha256,
            "git_commit": experiment_binding.implementation_commit,
            "implementation_dirty": experiment_binding.implementation_dirty,
        }
        update_evidence["sha256"] = sha256_value(update_evidence)
        update_evidence_path = output / "factorial_update_evidence.json"
        publish_json_once(update_evidence_path, update_evidence)
        manifest.dataset_hashes["factorial_update_evidence"] = str(
            update_evidence["sha256"]
        )
        finalize_run_directory(output, manifest)
        print_json(
            {
                "run_id": run_id,
                "cell": cell,
                "steps": trainer.global_step,
                "final_metrics": history[-1] if history else {},
                "output": str(output),
                "backend": trainer_config.backend,
            }
        )


if __name__ == "__main__":
    main()
