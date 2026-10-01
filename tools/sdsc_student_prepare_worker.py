#!/usr/bin/env python3
"""Prepare one shared 1.7B initial model from disjoint symbolic canonical proofs.

This fixed-W2 successor reuses the production FactorialTrainer loss, FSDP
preparation, optimizer and accumulation primitives. Its source is symbolic
canonical data, never a teacher-demonstration store. Preparation is not G0.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCHEMA = "quest-sdsc-student-prepare-report-v1"
INPUT_SCHEMA = "quest-sdsc-student-prepare-inputs-v1"
LR = 5e-5
STEPS = (4, 8, 16, 32)
FLAGS = dict(
    student_accepted=False,
    g0_passed=False,
    pilot_passed=False,
    factorial_ready=False,
    formal_initial_accepted=False,
    execution_class_certified=False,
)


def sibling(name):
    spec = importlib.util.spec_from_file_location("_prepare_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


util = sibling("sdsc_student_lr_probe")
require, same, canonical, digest = util.require, util.same, util.canonical, util.digest
file_sha, atomic, document = util.file_sha, util.atomic, util.document


def append(path, value):
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def artifact(path, output):
    return dict(path=path.relative_to(output).as_posix(), size=path.stat().st_size, sha256=file_sha(path))


def raw_artifacts(output):
    return [
        artifact(path, output)
        for path in sorted(output.glob("prepare-*.json*"))
        if path.name != "prepare-report.json"
    ]


class CanonicalSource:
    """One deterministic symbolic proof per prompt, with no sampling or teacher."""

    def __init__(self, records):
        require(records and len({row.prompt_id for row in records}) == len(records), "canonical source IDs")
        self.records = {row.prompt_id: copy.deepcopy(row) for row in records}
        self.identity = digest(canonical([row.trajectory_id for row in records]))
        self.cursor = {}

    def get_batch(self, model, prompt_batch, step):
        from posttrain_circuits.learning.contracts import TrajectoryBatch

        del model, step
        records = []
        for key in prompt_batch.prompt_ids:
            require(
                key in self.records and self.cursor.get(key, 0) == 0, "canonical one-pass cursor repeated"
            )
            self.cursor[key] = 1
            records.append(copy.deepcopy(self.records[key]))
        return TrajectoryBatch(records, policy_version=0)

    def refresh_if_needed(self, model, step):
        del model, step

    def state_dict(self):
        return dict(kind="symbolic_canonical", identity_sha256=self.identity, cursor=dict(self.cursor))

    def load_state_dict(self, state):
        same(state.get("kind"), "symbolic_canonical", "canonical source kind changed")
        same(state.get("identity_sha256"), self.identity, "canonical source population changed")
        cursor = state.get("cursor")
        require(
            isinstance(cursor, dict)
            and all(
                key in self.records and type(value) is int and value == 1 for key, value in cursor.items()
            ),
            "canonical cursor invalid",
        )
        self.cursor = dict(cursor)


def canonical_records(examples, rows, tokenizer, model_config):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    task, result = ProofGraphTask(), []
    for example, row in zip(examples, rows, strict=True):
        prompt = format_model_prompt(task.render(example), tokenizer, model_config)
        prefix = row["prefix_length"]
        response_ids = row["input_ids"][prefix:]
        target = task.canonical_target(example)
        verification = task.verify(example, task.parse_response(target))
        require(verification.reward == 1.0, "symbolic target must pass original verifier")
        record = TrajectoryRecord(
            trajectory_id="",
            prompt_id=example.example_id,
            split="student_preparation_fit",
            prompt_text=prompt.model_facing_prompt,
            input_ids=row["input_ids"][:prefix],
            response_ids=response_ids,
            response_text=target,
            response_token_mask=[True] * len(response_ids),
            behavior_policy_id="symbolic-canonical-proof",
            behavior_policy_revision="v1",
            policy_version=0,
            sampling_request_seed=0,
            actual_sampling_seed=0,
            sampling_cursor_id=example.example_id,
            sampling_protocol_id="deterministic-canonical-no-sampling-v1",
            sampling_temperature=0.0,
            top_p=1.0,
            behavior_logprobs=[float("nan")] * len(response_ids),
            verifier_reward=1.0,
            verification_trace=asdict(verification),
            raw_prompt_text=prompt.raw_prompt,
            prompt_protocol=prompt.prompt_protocol,
            enable_thinking=prompt.enable_thinking,
            chat_template_sha256=prompt.chat_template_sha256,
            raw_prompt_sha256=prompt.raw_prompt_sha256,
            model_facing_prompt_sha256=prompt.model_facing_prompt_sha256,
            tokenizer_fingerprint=util.TOKENIZER_SHA,
        )
        record.trajectory_id = record.expected_trajectory_id
        record.validate()
        result.append(record)
    return result


def build_trainer(loaded, records, config, checkpoint, rank, output, *, max_steps, science_commit):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
    from posttrain_circuits.learning.training.fsdp_contract import validate_model_fsdp_sharding
    from posttrain_circuits.learning.training.optimizer import build_adamw
    from posttrain_circuits.learning.training.schedules import PromptScheduler

    ids = [row.prompt_id for row in records]
    scheduler = PromptScheduler.for_distributed_rank(
        ids, [row.raw_prompt_text for row in records], 4, rank=rank, world_size=2
    )
    source = CanonicalSource(records)
    optimizer = build_adamw(loaded.model.parameters(), learning_rate=LR, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    # The old experiment's resolved configuration remains immutable input.
    # This new identity is explicit, without its teacher/SFT selectors.
    resolved = dict(
        model=copy.deepcopy(config["model"]),
        experiment={"name": "common_student_preparation", "state_source": "symbolic_canonical"},
        production_safety={"initial_checkpoint_path": str(checkpoint)},
        preparation={"world_size": 2, "global_batch_size": 64, "learning_rate": LR},
    )
    trainer = FactorialTrainer(
        model=loaded.model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        prompt_scheduler=scheduler,
        state_source=source,
        supervisor=VerifiedReplaySupervisor(
            loaded.tokenizer.pad_token_id, normalization="sequence", minimum_positives=1, retry_limit=0
        ),
        config=TrainerConfig(
            max_steps=max_steps,
            token_budget=2_000_000,
            steps_per_round=1,
            learning_rate=LR,
            checkpoint_every=32,
            evaluation_every=32,
            backend="accelerate",
            gradient_accumulation_steps=8,
            max_completion_length=128,
            require_evaluation_metrics=False,
        ),
        run_dir=output / "training",
        resolved_config=resolved,
        manifest_hashes={"initial_checkpoint": util.INITIAL_SHA},
        git_commit=science_commit,
        implementation_dirty=False,
        probe_input_ids=None,
        evaluation_fn=None,
    )
    require(
        trainer._accelerator.num_processes == 2 and trainer._accelerator.mixed_precision == "bf16",
        "actual W2 precision differs",
    )
    require(isinstance(trainer.model, FSDP), "training root must be real FSDP")
    contract = validate_model_fsdp_sharding(trainer.model, world_size=2, fsdp_type=FSDP)
    same(
        contract,
        dict(
            requested_fsdp_sharding_strategy="FULL_SHARD",
            effective_fsdp_sharding_strategy="FULL_SHARD",
            fsdp_wrapper_count=29,
        ),
        "FSDP contract differs",
    )
    require(
        all(p.requires_grad and p.dtype == torch.float32 for p in trainer.model.parameters()),
        "full-parameter FP32 FSDP masters required",
    )
    require(not optimizer.state and trainer.global_step == 0 and not source.cursor, "training not fresh")
    return trainer, optimizer, contract


def train_window(trainer, ids, rows, rank, step, *, observer=None, world_size=2):
    """Actual production backward/optimizer path, with exact fixed-W2 admission."""
    started, batches, seen = time.perf_counter(), [], []
    for microstep in range(8):
        prompts = trainer.prompt_scheduler.next_batch()
        slots = list(
            range(
                (step - 1) * 64 + rank + microstep * 4 * world_size,
                (step - 1) * 64 + rank + (microstep + 1) * 4 * world_size,
                world_size,
            )
        )
        same(prompts.prompt_ids, [ids[slot] for slot in slots], "fixed-W2 slot assignment changed")
        trajectories, supervision, attempts = trainer._collect_trajectories(prompts)
        trainer._register_collection(prompts, trajectories, supervision, attempts)
        require(attempts == 1 and len(trajectories.records) == 4, "canonical collection count differs")
        require(supervision.input_ids.shape[1] <= 1536, "preparation batch would truncate")
        for trajectory, slot in zip(trajectories.records, slots, strict=True):
            same(
                trajectory.input_ids + trajectory.response_ids,
                rows[slot]["input_ids"],
                "collator target identity",
            )
        batches.append(supervision)
        seen.extend(slots)
    admitted, total = trainer.token_budget.reserve_optimizer_update(
        sum(int(batch.attention_mask.sum()) for batch in batches)
    )
    expected = sum(len(row["input_ids"]) for row in rows[(step - 1) * 64 : step * 64])
    require(admitted and total == expected, "global nonpadding token reservation differs")
    trainer._current_global_update_tokens = total
    syncs, calls = [], []
    for batch in batches:
        syncs.append(trainer._training_micro_step(batch, expected_sequence_count=4))
        if observer is not None:
            calls.append(observer.calls)
    same(syncs, [False] * 7 + [True], "optimizer accumulation boundary differs")
    if observer is not None:
        same(calls, [step - 1] * 7 + [step], "actual AdamW cadence differs")
    metric = trainer._finish_optimizer_update(started=started)
    require(trainer.global_step == trainer.scheduler.last_epoch == step, "scheduler cadence differs")
    require(metric["parameter_update_norm"] > 0, "zero parameter update")
    same(
        trainer.state_source.cursor,
        {ids[index]: 1 for index in range(rank, step * 64, world_size)},
        "one-pass canonical source cursor differs",
    )
    return dict(
        rank=rank,
        step=step,
        global_slots=seen,
        global_input_tokens=total,
        optimizer_calls=calls,
        metric=metric,
        token_budget=trainer.token_budget.state_dict(),
    )


class StepObserver:
    def __init__(self, optimizer):
        self.calls = 0
        self.handle = optimizer.register_step_post_hook(self._after)

    def _after(self, optimizer, args, kwargs):
        del args, kwargs
        self.calls += 1
        require(bool(optimizer.state), "AdamW state empty after step")
        require(
            all(int(state["step"]) == self.calls for state in optimizer.state.values()),
            "AdamW parameter cadence differs",
        )

    def close(self):
        self.handle.remove()


def tree_hash(value):
    """Hash actual tensor bytes and structure without a second full state copy."""
    import hashlib

    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash

    checksum = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            checksum.update(torch_state_hash({"tensor": item.detach().cpu()}).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda value: (type(value).__name__, str(value))):
                checksum.update(canonical([type(key).__name__, key]))
                visit(item[key])
        elif isinstance(item, tuple | list):
            checksum.update(canonical([type(item).__name__, len(item)]))
            for child in item:
                visit(child)
        else:
            checksum.update(canonical(item))

    visit(value)
    return checksum.hexdigest()


def runtime_state(trainer):
    from posttrain_circuits.core.seeding import RNGState

    return dict(
        global_step=trainer.global_step,
        source=trainer.state_source.state_dict(),
        prompts=trainer.prompt_scheduler.state_dict(),
        cumulative_counts=copy.deepcopy(trainer.cumulative_counts),
        token_budget=trainer.token_budget.state_dict(),
        scheduler=trainer.scheduler.state_dict(),
        rng=RNGState.capture().as_dict(),
    )


def restore_runtime(trainer, state):
    from posttrain_circuits.core.seeding import RNGState

    trainer.global_step = state["global_step"]
    trainer.state_source.load_state_dict(state["source"])
    trainer.prompt_scheduler.load_state_dict(state["prompts"])
    trainer.cumulative_counts = copy.deepcopy(state["cumulative_counts"])
    trainer.token_budget.load_state_dict(state["token_budget"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    RNGState(**state["rng"]).restore()


def full_state_restore(trainer, output, rank, *, world_size=2):
    """Save real Accelerate state, perturb, restore and compare every local state.

    These are same-allocation state restoration measurements. They are not the
    separate G0 two-run, step20-to33 resume-equivalence experiment.
    """
    import torch

    require(
        trainer._accumulation_micro_step == 0 and trainer._parameters_before_update is None,
        "resume test requires committed optimizer boundary",
    )
    before = runtime_state(trainer)
    parameter_hash = tree_hash(list(trainer.model.parameters()))
    optimizer_hash = tree_hash(trainer.optimizer.state_dict())
    path = output / "resume" / f"step-{trainer.global_step:08d}"
    util.local_phase(
        lambda: path.mkdir(parents=True, exist_ok=False) if rank == 0 else None,
        rank=rank,
        world_size=world_size,
    )
    trainer._accelerator.save_state(str(path))
    trainer._accelerator.wait_for_everyone()
    atomic(path / f"runtime-rank-{rank}.json", dict(world_size=world_size, rank=rank, state=before))
    with torch.no_grad():
        next(trainer.model.parameters()).add_(0.125)
        state = next(iter(trainer.optimizer.state.values()))
        state["exp_avg"].add_(0.25)
    trainer.scheduler.step()
    trainer.state_source.cursor = {}
    trainer.prompt_scheduler.position = 0
    trainer.cumulative_counts["prompts_consumed"] += 7
    torch.rand(3)
    torch.rand(3, device=next(trainer.model.parameters()).device)
    require(
        tree_hash(list(trainer.model.parameters())) != parameter_hash
        and tree_hash(trainer.optimizer.state_dict()) != optimizer_hash,
        "resume perturbation ineffective",
    )
    metadata = document(path / f"runtime-rank-{rank}.json")
    require(metadata["world_size"] == world_size and metadata["rank"] == rank, "resume world/rank changed")
    trainer._accelerator.load_state(str(path))
    restore_runtime(trainer, metadata["state"])
    same(runtime_state(trainer), before, "rank-local runtime/RNG restore differs")
    require(tree_hash(list(trainer.model.parameters())) == parameter_hash, "full model restore differs")
    require(tree_hash(trainer.optimizer.state_dict()) == optimizer_hash, "full AdamW restore differs")
    trainer._validate_live_optimizer_scheduler_cadence()
    return dict(
        rank=rank,
        passed=True,
        step=trainer.global_step,
        model_sha256=parameter_hash,
        optimizer_sha256=optimizer_hash,
        runtime_sha256=digest(canonical(before)),
        actual_accelerate_save_load=True,
        scope="same_world_full_state_restoration_not_complete_G0_resume",
    )


def save_dense(state, path, *, step, inputs, dataset_sha, protocol_sha):
    import torch

    require(not path.exists(), "dense checkpoint already exists")
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(
            dict(
                format="student_preparation_dense_v1",
                model=state,
                global_step=step,
                parent_checkpoint_sha256=util.INITIAL_SHA,
                dataset_sha256=dataset_sha,
                protocol_sha256=protocol_sha,
                protocol_artifact_sha256=inputs["protocol"]["sha256"],
                learning_rate=LR,
                scope="common_student_preparation_model_only",
                **FLAGS,
            ),
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint_and_development(
    trainer,
    model_config,
    tokenizer,
    examples,
    dev_rows,
    *,
    output,
    rank,
    step,
    inputs,
    dataset_sha,
    protocol_sha,
    dev_sha,
    evaluate,
):
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.loading import load_model_and_tokenizer
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    quality = sibling("sdsc_student_quality_probe")
    path = output / "checkpoints" / f"step-{step:08d}.pt"
    reference = []
    with util.measurement(trainer.model, trainer), torch.no_grad():
        for row in dev_rows[:2]:
            ids = torch.tensor(
                [row["input_ids"][: row["prefix_length"]]], device=next(trainer.model.parameters()).device
            )
            logits = trainer.model(
                input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False
            ).logits.float()
            reference.append(logits.detach().cpu())
        state = trainer._full_model_state_for_checkpoint()
        util.local_phase(
            lambda: save_dense(
                state, path, step=step, inputs=inputs, dataset_sha=dataset_sha, protocol_sha=protocol_sha
            )
            if rank == 0
            else None,
            rank=rank,
            world_size=2,
        )
        # Read the actual saved bytes on each rank; do not infer reload success
        # from the in-memory export used to write the file.
        state = None
        gc.collect()
        checkpoint_sha = file_sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        loaded = load_model_and_tokenizer(model_config, for_training=False)
        exact = util.strict_load_export(loaded.model, saved["model"])
        state_sha = torch_state_hash(saved["model"])
        saved = None
        model = loaded.model.to(device=next(trainer.model.parameters()).device, dtype=torch.bfloat16).eval()
        parity = []
        for row, expected in zip(dev_rows[:2], reference, strict=True):
            ids = torch.tensor(
                [row["input_ids"][: row["prefix_length"]]], device=next(model.parameters()).device
            )
            actual = (
                model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False)
                .logits.float()
                .cpu()
            )
            comparison = util.difference_statistics(expected, actual)
            require(torch.equal(expected, actual), "FSDP/export native BF16 logits differ")
            parity.append(comparison)
        counts = dict(answer_correct=0, proof_correct=0, format_valid=0, num_examples=0)
        generation_started = time.perf_counter()
        for_generation = list(range(rank, len(examples) if evaluate else 8, 2))
        if evaluate or step == 4:
            for ordinal in for_generation:
                example, row = examples[ordinal], dev_rows[ordinal]
                ids = torch.tensor(
                    [row["input_ids"][: row["prefix_length"]]], device=next(model.parameters()).device
                )
                require(
                    ids.shape[1] + 256 <= 1600 and ids.shape[1] + 256 <= model.config.max_position_embeddings,
                    "preparation development generation exceeds declared1600 input envelope",
                )
                seed_everything(42 + ordinal)
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids, dtype=torch.bool),
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                require(torch.equal(generated[0, : ids.shape[1]], ids[0]), "generation changed prefix")
                prompt = format_model_prompt(
                    ProofGraphTask().render(example), tokenizer, model_config
                ).model_facing_prompt
                record = quality.response_record(
                    example,
                    generated[0, ids.shape[1] :].tolist(),
                    tokenizer,
                    prompt_ids=ids[0].tolist(),
                    prompt_text=prompt,
                    cap=256,
                )
                record.update(
                    step=step,
                    ordinal=ordinal,
                    rank=rank,
                    checkpoint_sha256=checkpoint_sha,
                    parsed_trace=asdict(ProofGraphTask().parse_response(record["response_text"])),
                )
                record["cohort"] = "student_dev" if evaluate else "student_fit_preflight_first8"
                raw_name = "dev" if evaluate else "preflight"
                append(output / f"prepare-{raw_name}-records-rank-{rank}.jsonl", record)
                verification = record["verification"]
                counts["answer_correct"] += int(verification["answer_correct"])
                counts["proof_correct"] += int(verification["reward"] == 1.0)
                counts["format_valid"] += int(verification["parse_valid"])
                counts["num_examples"] += 1
                if counts["num_examples"] % 32 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="preparation_development",
                                rank=rank,
                                step=step,
                                completed=counts["num_examples"],
                            )
                        ),
                        flush=True,
                    )
        loaded, model, reference = None, None, []
        gc.collect()
        torch.cuda.empty_cache()
    local = dict(
        rank=rank,
        exact_saved_master_reload=exact,
        root_export_logits=parity,
        model_state_sha256=state_sha,
        counts=counts,
        generation_seconds=time.perf_counter() - generation_started,
    )
    results = util.gather(local)
    same(
        results[0]["model_state_sha256"],
        results[1]["model_state_sha256"],
        "ranks reloaded different checkpoint",
    )
    checkpoint = dict(
        **artifact(path, output),
        step=step,
        scope="common_student_preparation_model_only",
        reload_by_rank=results,
    )
    development = None
    if evaluate:
        sums = {key: sum(item["counts"][key] for item in results) for key in counts}
        require(sums["num_examples"] == 512, "incomplete development population")
        development = dict(step=step, checkpoint_sha256=checkpoint["sha256"], examples_sha256=dev_sha, **sums)
    return checkpoint, development


def staged_science(inputs, output):
    work = output.parent.resolve()
    require(
        set(inputs)
        == {
            "schema",
            "mode",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "original_science_files",
            "dataset_root",
            "hf_home",
            "protocol",
            "original_config",
            "initial_checkpoint",
        },
        "preparation input identity fields differ",
    )
    require(
        inputs.get("schema") == INPUT_SCHEMA and inputs.get("mode") in ("preflight", "fit"),
        "preparation input identity",
    )
    require(inputs["job_id"] == os.environ.get("SLURM_JOB_ID"), "job allocation differs")
    for name in ("source_code_sha256", "plan_sha256"):
        require(
            isinstance(inputs[name], str) and re.fullmatch("[a-f0-9]{64}", inputs[name]), "invalid input SHA"
        )
    science = util.confined(inputs["science_root"], work, directory=True)
    require(Path.cwd().resolve() == science, "worker cwd must equal verified science root")
    require(
        not any(
            name == "posttrain_circuits" or name.startswith("posttrain_circuits.") for name in sys.modules
        ),
        "science imported before root verification",
    )
    for name, expected in inputs["original_science_files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe scientific path")
        require(file_sha(science / name) == expected, "frozen scientific file changed: " + name)
    require(len(inputs["original_science_files"]) == 49, "frozen scientific inventory differs")
    require(
        str(util.confined(inputs["hf_home"], work, directory=True)) == os.environ.get("HF_HOME"),
        "offline cache differs",
    )
    sys.path.insert(0, str(science / "src"))
    return work


def load_data(inputs, work, output, rank):
    import yaml

    from posttrain_circuits.datasets.proofgraph import rendering
    from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
    from posttrain_circuits.experiments.protocols import student_preparation as protocol
    from posttrain_circuits.learning.teacher.adaptation_fit import FitPlan, make_fit_examples

    config_path = util.verified(inputs["original_config"], work, util.CONFIG_SHA, 7923)
    checkpoint = util.verified(inputs["initial_checkpoint"], work, util.INITIAL_SHA, util.INITIAL_SIZE)
    config = yaml.safe_load(config_path.read_text())
    require(
        set(inputs["protocol"]) == {"path", "size", "sha256", "protocol_sha256"},
        "preparation protocol input fields differ",
    )
    protocol_path = util.verified({key: inputs["protocol"][key] for key in ("path", "size", "sha256")}, work)
    require(
        digest(rendering.RESPONSE_FORMAT_INSTRUCTIONS.encode()) == util.INSTRUCTION_SHA,
        "original instruction differs",
    )
    # The successor's independent implementation/acceptance resolver is called
    # before any model load. It also freezes protocol numeric settings.
    binding = protocol.resolve_student_preparation_protocol(Path(inputs["science_root"]))
    require(binding.artifact_sha256 == file_sha(protocol_path), "resolved protocol artifact differs")
    require(
        binding.protocol_sha256 == inputs["protocol"]["protocol_sha256"], "resolved protocol core differs"
    )
    for name in (
        "sdsc_student_prepare_worker.py",
        "sdsc_student_lr_probe.py",
        "sdsc_student_quality_probe.py",
    ):
        require(
            file_sha(Path(__file__).with_name(name)) == binding.science_file_sha256["tools/" + name],
            "executing tool differs from independently accepted science: " + name,
        )
    root = util.confined(inputs["dataset_root"], work, directory=True)
    manifest_path = util.confined(str(root / "manifest.json"), work)
    require(file_sha(manifest_path) == util.FAMILY_SHA, "formal family manifest changed")
    manifest = document(manifest_path)

    def formal_rows():
        total = 0
        for name, entry in manifest["splits"].items():
            require(entry["path"] == name + "/examples.jsonl", "formal family path changed")
            path = util.confined(str(root / entry["path"]), work)
            require(file_sha(path) == entry["examples_file_sha256"], "formal family split changed")
            count = 0
            with path.open() as stream:
                for line in stream:
                    count += 1
                    yield deserialize_example(json.loads(line))
            require(count == entry["num_examples"], "formal family split count changed")
            total += count
        require(total == 144000, "formal family incomplete")

    splits = protocol.make_examples(config["task"])
    teacher = make_fit_examples(config["task"], FitPlan())
    isolation = protocol.audit_isolation(splits, formal_rows(), teacher)
    dataset = protocol.dataset_manifest(splits)
    if rank == 0:
        atomic(output / "prepare-dataset-manifest.json", dataset)
        atomic(output / "prepare-data-isolation.json", isolation)
    return config, checkpoint, splits, dataset, isolation, binding


def execute(inputs, output, work):
    import torch
    from accelerate import Accelerator

    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.experiments.protocols import student_preparation as protocol

    rank, world = int(os.environ.get("RANK", "-1")), int(os.environ.get("WORLD_SIZE", "0"))
    require(
        world == 2 and rank in (0, 1) and int(os.environ.get("LOCAL_RANK", "-1")) == rank,
        "preparation requires single-node W2",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 2
        and all("H100" in torch.cuda.get_device_name(index) for index in range(2)),
        "two assigned H100s required",
    )
    for name, value in dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        require(os.environ.get(name) == value, "execution environment differs: " + name)
    require(torch.__version__ == "2.8.0+cu128", "fixed Torch differs")
    for package, version in {
        "transformers": "4.56.2",
        "accelerate": "1.10.1",
        "tokenizers": "0.22.0",
    }.items():
        require(importlib.metadata.version(package) == version, "fixed runtime differs: " + package)
    torch.cuda.set_device(rank)
    torch.set_num_threads(12)
    bootstrap = Accelerator(gradient_accumulation_steps=8, step_scheduler_with_optimizer=False)
    require(
        bootstrap.num_processes == 2 and bootstrap.mixed_precision == "bf16",
        "launcher precision/world differs",
    )
    config, checkpoint, splits, dataset, isolation, binding = util.local_phase(
        lambda: load_data(inputs, work, output, rank), rank=rank, world_size=2
    )
    seed_everything(42)
    loaded, initial_loading = util.local_phase(
        lambda: util.load_training_initial(config["model"], checkpoint), rank=rank, world_size=2
    )
    encoded = {
        key: [protocol.encode_example(example, loaded.tokenizer, config["model"]) for example in examples]
        for key, examples in splits.items()
    }
    token_audit = protocol.validate_encoded_fit(encoded["student_fit"])
    # All token/mask audits finish before FSDP preparation or backward.
    records = canonical_records(
        splits["student_fit"], encoded["student_fit"], loaded.tokenizer, config["model"]
    )
    max_steps = 4 if inputs["mode"] == "preflight" else 32
    trainer, optimizer, fsdp = build_trainer(
        loaded,
        records,
        config,
        checkpoint,
        rank,
        output,
        max_steps=max_steps,
        science_commit=binding.implementation_commit,
    )
    observer = StepObserver(optimizer)
    dataset_sha = digest(canonical(dataset))
    dev_sha = digest(canonical([asdict(row) for row in splits["student_dev"]]))
    report = dict(
        schema=SCHEMA,
        mode=inputs["mode"],
        passed=False,
        execution_complete=False,
        preparation_complete=False,
        **FLAGS,
        **{key: inputs[key] for key in ("job_id", "run_id", "source_code_sha256", "plan_sha256")},
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        initial_checkpoint_sha256=util.INITIAL_SHA,
        initial_loading=initial_loading,
        global_batch_size=64,
        world_size=2,
        learning_rate=LR,
        source_kind="symbolic_canonical",
        teacher_data_required=False,
        optimizer_steps=0,
        data_audit=dict(passed=True, isolation=isolation, token_envelope=token_audit),
        fsdp=fsdp,
        development_generation=dict(
            max_new_tokens=256,
            max_model_input_length=1600,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            truncation=False,
        ),
        checkpoints=[],
        development=[],
        checkpoint_restore={"passed": False},
        dataset_sha256=dataset_sha,
        raw_artifacts=[],
        selected_checkpoint=None,
    )
    if rank == 0:
        atomic(output / "prepare-token-audit.json", token_audit)
        atomic(output / "prepare-report.json", report)
        if inputs["mode"] == "fit":
            for ordinal, example in enumerate(splits["student_dev"]):
                row = encoded["student_dev"][ordinal]
                append(
                    output / "prepare-dev-prompts.jsonl",
                    dict(
                        ordinal=ordinal,
                        example=asdict(example),
                        prompt_ids=row["input_ids"][: row["prefix_length"]],
                    ),
                )
    for step in range(1, max_steps + 1):
        require(not trainer._terminate, "allocation termination requested")
        update = train_window(
            trainer, [row.prompt_id for row in records], encoded["student_fit"], rank, step, observer=observer
        )
        updates = util.gather(update)
        if rank == 0:
            append(output / "prepare-updates.jsonl", dict(step=step, ranks=updates))
            report["optimizer_steps"] = step
            atomic(output / "prepare-report.json", report)
            print(
                json.dumps(dict(stage="preparation_update", step=step, tokens=trainer.token_budget.consumed)),
                flush=True,
            )
        if step == 4:
            restore = full_state_restore(trainer, output, rank)
            restored = util.gather(restore)
            report["checkpoint_restore"] = dict(passed=all(row["passed"] for row in restored), ranks=restored)
        if step in STEPS:
            dense, development = checkpoint_and_development(
                trainer,
                config["model"],
                loaded.tokenizer,
                splits["student_dev"] if inputs["mode"] == "fit" else splits["student_fit"][:8],
                encoded["student_dev"] if inputs["mode"] == "fit" else encoded["student_fit"][:8],
                output=output,
                rank=rank,
                step=step,
                inputs=inputs,
                dataset_sha=dataset_sha,
                protocol_sha=binding.protocol_sha256,
                dev_sha=dev_sha,
                evaluate=inputs["mode"] == "fit",
            )
            report["checkpoints"].append(dense)
            if development is not None:
                report["development"].append(development)
            if rank == 0:
                atomic(output / "prepare-report.json", report)
    observer.close()
    report.update(
        optimizer_steps=max_steps,
        execution_complete=True,
        preparation_complete=inputs["mode"] == "fit",
        training_input_tokens=trainer.token_budget.consumed,
    )
    if inputs["mode"] == "fit":
        selected = protocol.select_checkpoint(report["development"])
        report["selected_checkpoint"] = selected
        report["passed"] = selected is not None
    else:
        report["passed"] = True
    if rank == 0:
        report["raw_artifacts"] = raw_artifacts(output)
        atomic(output / "prepare-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(exist_ok=True)
    try:
        inputs = document(args.inputs_json)
        work = staged_science(inputs, args.output_dir)
        result = execute(inputs, args.output_dir, work)
        return 0 if result["passed"] else 2
    except Exception as error:
        rank = os.environ.get("RANK", "0")
        failure = dict(
            passed=False,
            execution_complete=False,
            preparation_complete=False,
            **FLAGS,
            failed_rank=rank,
            error=f"{type(error).__name__}: {error}",
        )
        atomic(args.output_dir / f"rank-{rank}-failure.json", failure)
        if rank == "0":
            path = args.output_dir / "prepare-report.json"
            try:
                report = document(path) if path.exists() else {"schema": SCHEMA}
            except (OSError, ValueError):
                report = {"schema": SCHEMA}
            report.update(failure, raw_artifacts=raw_artifacts(args.output_dir))
            atomic(path, report)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
