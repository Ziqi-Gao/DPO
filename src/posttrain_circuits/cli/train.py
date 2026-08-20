"""Run one controlled factorial cell or canonical SFT."""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.circuits.mib_runner import load_checkpoint_into_hf_model
from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.core.config import is_production_scale
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.provenance import (
    RunManifest,
    finalize_run_directory,
    formal_artifact_binding,
    initialize_run_directory,
)
from posttrain_circuits.core.readiness import require_factorial_prerequisites
from posttrain_circuits.core.seeding import seed_everything
from posttrain_circuits.core.types import TrajectoryRecord
from posttrain_circuits.data.splits import build_split, load_frozen_split
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.models.loading import (
    LoadedModel,
    assert_tokenizer_compatible,
    load_model_and_tokenizer,
    move_model_to_local_cuda,
)
from posttrain_circuits.models.prompt_protocol import format_model_prompts
from posttrain_circuits.rollout.generation import build_proofgraph_hf_generator
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.renderer import render_example
from posttrain_circuits.teacher.demo_generation import read_teacher_demo_store
from posttrain_circuits.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.training.evaluation import build_proofgraph_evaluator
from posttrain_circuits.training.factorial_trainer import FactorialTrainer, TrainerConfig
from posttrain_circuits.training.factories import build_state_source, build_supervisor
from posttrain_circuits.training.optimizer import build_adamw
from posttrain_circuits.training.schedules import PromptScheduler
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    build_smoke_examples,
    scripted_current_policy_generator,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _teacher_demo_prompts(
    records: list[TrajectoryRecord],
) -> tuple[list[str], list[str]]:
    by_prompt: dict[str, str] = {}
    for record in records:
        by_prompt.setdefault(record.prompt_id, record.raw_prompt_text or record.prompt_text)
    if not by_prompt:
        raise ValueError("teacher-demo store has no prompts")
    return list(by_prompt), list(by_prompt.values())


def _production_examples(config: dict[str, Any], split: str, count: int):
    task_config = config["task"]
    return build_split(
        ProofGraphTask(),
        split,
        count,
        int(task_config.get("seed", config["seed"])),
        dict(task_config),
    )


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
            }
        )
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
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
    factorial_cells = {
        "offline_hard",
        "online_hard",
        "offline_soft",
        "online_soft_opd",
        "offline_verified_replay",
        "online_verified_replay",
    }
    prerequisite_evidence: dict[str, Any] = {}
    if production_scale and (
        cell in factorial_cells or (cell == "canonical_sft" and "pilot_profile" in config)
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
    state_source_name = str(config["state_source"]["name"])
    supervision_name = str(config["supervision"]["name"])
    batch_size = int(config["trainer"]["batch_size"])

    examples = (
        build_smoke_examples(batch_size, seed)
        if local_model
        else _production_examples(
            config,
            "train",
            int(config["task"]["num_examples"]),
        )
    )
    fixed_bank: list[TrajectoryRecord] | None = None
    fixed_bank_manifest: dict[str, Any] | None = None
    teacher_demos: list[TrajectoryRecord] | None = None
    teacher_demo_manifest: dict[str, Any] | None = None
    current_generator = None
    if state_source_name == "teacher_demo":
        store_path = Path(str(config["state_source"]["store_path"]))
        teacher_demos, loaded_manifest = read_teacher_demo_store(store_path)
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
                    prompt_ids, prompt_texts = _teacher_demo_prompts(fixed_bank)
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
                prompt_ids, prompt_texts = _teacher_demo_prompts(fixed_bank)
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
    launch_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
    validation_manifest: dict[str, Any] | None = None
    if production_scale:
        validation_examples, validation_manifest = load_frozen_split(
            Path(str(config["task"]["validation_split_path"])),
            expected_split="validation",
        )
        validation_limit = int(config["trainer"].get("validation_examples", len(validation_examples)))
        if validation_limit < 1 or validation_limit > len(validation_examples):
            raise ValueError("trainer.validation_examples is outside the frozen validation split")
        validation_examples = validation_examples[:validation_limit]
    else:
        validation_examples = build_smoke_examples(max(2, min(4, batch_size)), seed + 100_000)
    dataset_hashes = {
        "train": sha256_value(
            {
                "prompt_ids": global_prompt_ids,
                "prompt_texts": global_prompt_texts,
            }
        ),
        "validation": sha256_value([asdict(example) for example in validation_examples]),
    }
    if validation_manifest is not None:
        dataset_hashes["validation_manifest"] = str(validation_manifest["sha256"])
        dataset_hashes["validation_file"] = str(validation_manifest["examples_file_sha256"])
    if initial_checkpoint_hash is not None:
        dataset_hashes["initial_checkpoint"] = initial_checkpoint_hash
    for name, evidence in prerequisite_evidence.items():
        dataset_hashes[f"prerequisite_{name}"] = str(
            evidence.get("report_hash", evidence.get("manifest_hash"))
        )
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
    raw_prompt_schedule_hash = sha256_value([prompt.raw_prompt_sha256 for prompt in formatted_schedule])
    model_facing_prompt_schedule_hash = sha256_value(
        [prompt.model_facing_prompt_sha256 for prompt in formatted_schedule]
    )

    teacher_id: str | None = None
    teacher_revision: str | None = None
    resolved_teacher_commit: str | None = None
    teacher_demo_generation: dict[str, Any] | None = None
    if loaded_teacher is not None:
        teacher_id = loaded_teacher.model_id
        teacher_revision = loaded_teacher.requested_model_revision
        resolved_teacher_commit = loaded_teacher.resolved_model_commit
    elif teacher is not None:
        teacher_id = teacher.teacher_id
        teacher_revision = teacher.teacher_revision
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
        raw_prompt_schedule_hash=raw_prompt_schedule_hash,
        model_facing_prompt_schedule_hash=model_facing_prompt_schedule_hash,
        prompt_protocol=(loaded_student.prompt_protocol if loaded_student is not None else "legacy_raw_v1"),
        enable_thinking=False,
        chat_template_sha256=(
            loaded_student.chat_template_sha256 if loaded_student is not None else "legacy-unrecorded"
        ),
        tokenizer_fingerprint=(
            loaded_student.tokenizer_hash if loaded_student is not None else "legacy-unrecorded"
        ),
        protocol_track=str(config.get("protocol_track", "core_v2")),
        artifact_namespace=str(model_config.get("artifact_namespace", "legacy")),
        protocol_teacher_revision=str(config["teacher"].get("model_revision", "unbound")),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get("token_budget_unit", "global_nonpadding_model_input_tokens")
        ),
    )
    if launch_rank == 0:
        initialize_run_directory(
            output,
            config,
            manifest,
            require_git=production_scale,
        )
    trainer_config = TrainerConfig(
        max_steps=int(config["trainer"]["max_steps"]),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get("token_budget_unit", "global_nonpadding_model_input_tokens")
        ),
        steps_per_round=int(config["trainer"]["steps_per_round"]),
        learning_rate=float(config["trainer"]["learning_rate"]),
        checkpoint_every=int(config["trainer"]["checkpoint_every"]),
        evaluation_every=int(config["trainer"].get("evaluation_every", 1)),
        backend=str(config["trainer"]["backend"]),
        gradient_accumulation_steps=int(config["trainer"]["gradient_accumulation_steps"]),
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
            "state_source": source_hash,
            "prompt_schedule": prompt_schedule_hash,
            "prompt_shard": sha256_value(
                {
                    "rank": launch_rank,
                    "world_size": world_size,
                    "prompt_ids": prompt_scheduler.prompt_ids,
                }
            ),
        },
        git_commit=manifest.git_commit,
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
        final_checkpoint = output / "checkpoints" / f"step-{trainer.global_step:08d}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError("training completed without a final optimizer-boundary checkpoint")
        manifest.token_budget_consumed = trainer.token_budget.consumed
        manifest.training_stop_reason = trainer.token_budget.stop_reason
        manifest.metrics_sha256 = sha256_file(output / "metrics.jsonl")
        manifest.final_checkpoint_path = str(final_checkpoint)
        manifest.final_checkpoint_sha256 = sha256_file(final_checkpoint)
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
