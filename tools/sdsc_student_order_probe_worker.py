#!/usr/bin/env python3
"""Diagnose joint-order coverage in two fresh native 1.7B training arms.

This fixed-W2 successor reuses the production FactorialTrainer loss, FSDP
preparation, optimizer and accumulation primitives. Its source is symbolic
canonical data, never a teacher-demonstration store. Completion is diagnostic
evidence only; no checkpoint is selected or accepted.
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

SCHEMA = "quest-sdsc-student-order-probe-report-v1"
INPUT_SCHEMA = "quest-sdsc-student-order-probe-inputs-v1"
LR = 5e-5
STEPS = (4, 6, 7, 8, 12)
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


base = sibling("sdsc_student_prepare_worker")
CanonicalSource = base.CanonicalSource
train_window = base.train_window
StepObserver = base.StepObserver
tree_hash = base.tree_hash
runtime_state = base.runtime_state
restore_runtime = base.restore_runtime
full_state_restore = base.full_state_restore


def canonical_records(views, rows, tokenizer, model_config):
    """Bind actual augmented prompts to original canonical response supervision."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    task, result = ProofGraphTask(), []
    for view, row in zip(views, rows, strict=True):
        example = view.example
        prompt = format_model_prompt(view.prompt, tokenizer, model_config)
        prefix = row["prefix_length"]
        response_ids = row["input_ids"][prefix:]
        same(
            tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False),
            row["input_ids"][:prefix],
            "augmented canonical prompt text/token prefix differ",
        )
        target = task.canonical_target(example)
        require(tokenizer.eos_token_id is not None, "canonical response EOS is missing")
        same(
            [*tokenizer.encode(target, add_special_tokens=False), tokenizer.eos_token_id],
            response_ids,
            "symbolic canonical response/token IDs differ",
        )
        verification = task.verify(example, task.parse_response(target))
        require(verification.reward == 1.0, "symbolic target must pass original verifier")
        record = TrajectoryRecord(
            trajectory_id="",
            prompt_id=example.example_id,
            split="student_order_probe_fit",
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


def specification():
    # Import only after staged_science has admitted the actual scientific root.
    # The independently accepted protocol owns the new population/budget/shape.
    from posttrain_circuits.experiments.protocols import student_order_probe as protocol

    value = protocol.proposed_student_order_probe_protocol()
    require(tuple(value["training"]["checkpoint_steps"]) == STEPS, "order checkpoint schedule differs")
    require(value["training"]["optimizer"]["learning_rate"] == LR, "order learning rate differs")
    return value


def checkpoint_steps(mode):
    require(mode == "probe", "diagnostic-only mode required")
    return STEPS


def build_trainer(loaded, records, config, checkpoint, rank, output, *, max_steps, science_commit, arm):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
    from posttrain_circuits.learning.training.fsdp_contract import validate_model_fsdp_sharding
    from posttrain_circuits.learning.training.optimizer import build_adamw
    from posttrain_circuits.learning.training.schedules import PromptScheduler

    require(arm in ("control", "treatment"), "unknown diagnostic arm")
    settings = specification()
    training = settings["training"]
    require(
        max_steps == training["optimizer_steps"] == 12,
        "unreviewed order training length",
    )
    require(len(records) == settings["data"]["fit_examples"], "order training population differs")
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
        experiment={"name": "student_order_probe", "state_source": "symbolic_canonical", "arm": arm},
        production_safety={"initial_checkpoint_path": str(checkpoint)},
        diagnostic={"world_size": 2, "global_batch_size": 64, "learning_rate": LR, "arm": arm},
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
            token_budget=training["input_token_budget"],
            steps_per_round=1,
            learning_rate=LR,
            checkpoint_every=training["optimizer_steps"],
            evaluation_every=training["optimizer_steps"],
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


def save_dense(state, path, *, step, inputs, dataset_sha, protocol_sha):
    import torch

    require(not path.exists(), "dense checkpoint already exists")
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(
            dict(
                format="student_order_probe_dense_v1",
                arm=inputs["arm"],
                mode="probe",
                model=state,
                global_step=step,
                parent_checkpoint_sha256=util.INITIAL_SHA,
                dataset_sha256=dataset_sha,
                protocol_sha256=protocol_sha,
                protocol_artifact_sha256=inputs["protocol"]["sha256"],
                learning_rate=LR,
                scope="student_order_probe_diagnostic_model_only",
                **FLAGS,
            ),
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_dense_payload(saved, *, step, arm, dataset_sha, protocol_sha, protocol_artifact_sha):
    require(
        isinstance(saved, dict)
        and set(saved)
        == {
            "format",
            "arm",
            "mode",
            "model",
            "global_step",
            "parent_checkpoint_sha256",
            "dataset_sha256",
            "protocol_sha256",
            "protocol_artifact_sha256",
            "learning_rate",
            "scope",
            *FLAGS,
        },
        "diagnostic dense fields differ",
    )
    expected = dict(
        format="student_order_probe_dense_v1",
        arm=arm,
        mode="probe",
        global_step=step,
        parent_checkpoint_sha256=util.INITIAL_SHA,
        dataset_sha256=dataset_sha,
        protocol_sha256=protocol_sha,
        protocol_artifact_sha256=protocol_artifact_sha,
        learning_rate=LR,
        scope="student_order_probe_diagnostic_model_only",
        **FLAGS,
    )
    require(
        arm in ("control", "treatment") and type(step) is int and step in STEPS,
        "diagnostic dense arm or checkpoint step differs",
    )
    same({key: saved[key] for key in expected}, expected, "diagnostic dense provenance differs")


def empty_counts():
    return dict(answer_correct=0, proof_correct=0, format_valid=0, num_examples=0)


def add_verification(counts, verification):
    counts["answer_correct"] += int(verification["answer_correct"])
    counts["proof_correct"] += int(verification["reward"] == 1.0)
    counts["format_valid"] += int(verification["parse_valid"])
    counts["num_examples"] += 1


def generation_ordinals(count, rank, *, evaluate):
    settings = specification()
    require(rank in (0, 1), "order generation requires W2 rank")
    require(evaluate is True, "diagnostic always evaluates the complete fixed development panel")
    require(
        count == settings["data"]["development_examples"] == 768,
        "order generation population differs",
    )
    # Keep all six views of one development base on one rank so both ranks
    # receive equal long-OOD and short-IID work before their final collective.
    views_per_base = len(settings["development"]["views"])
    return [ordinal for ordinal in range(count) if (ordinal // views_per_base) % 2 == rank]


def aggregate_development(results, *, step, checkpoint_sha, dev_sha):
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS

    names = ("identity", *TRANSFORMATIONS)
    structures = ("chain", "branch", "converging_dag")
    require(
        isinstance(results, list) and [row.get("rank") for row in results] == [0, 1],
        "both diagnostic development ranks required",
    )
    for row in results:
        require(
            set(row["per_view_counts"]) == set(names) and set(row["structure_counts"]) == set(structures),
            "diagnostic view/structure inventory differs",
        )
        for counts in [row["counts"], *row["per_view_counts"].values(), *row["structure_counts"].values()]:
            require(
                set(counts) == set(empty_counts()) and all(type(v) is int for v in counts.values()),
                "diagnostic raw counts must be integers",
            )
            require(
                0
                <= counts["proof_correct"]
                <= counts["answer_correct"]
                <= counts["format_valid"]
                <= counts["num_examples"],
                "diagnostic count ordering differs",
            )
        require(
            row["counts"]["num_examples"] == 384
            and all(v["num_examples"] == 64 for v in row["per_view_counts"].values()),
            "diagnostic rank population incomplete",
        )
        same(
            {key: sum(x[key] for x in row["per_view_counts"].values()) for key in empty_counts()},
            row["counts"],
            "diagnostic rank view totals differ",
        )
    per_view = {
        name: {key: sum(row["per_view_counts"][name][key] for row in results) for key in empty_counts()}
        for name in names
    }
    by_structure = {
        name: {key: sum(row["structure_counts"][name][key] for row in results) for key in empty_counts()}
        for name in structures
    }
    require(all(row["num_examples"] == 128 for row in per_view.values()), "incomplete development views")
    same(
        {key: sum(row[key] for row in by_structure.values()) for key in empty_counts()},
        per_view["identity"],
        "development structure totals differ",
    )
    return dict(
        step=step,
        checkpoint_sha256=checkpoint_sha,
        examples_sha256=dev_sha,
        **per_view["identity"],
        per_view_counts=per_view,
        structure_counts=by_structure,
    )


def inference_envelope_probe(model, *, enabled):
    """Exercise the new context on fixed valid tokens, without development data."""
    if not enabled:
        return None
    import torch

    require(
        all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()),
        "context probe requires native BF16 model",
    )
    length = specification()["development"]["max_model_input_tokens"]
    require(model.config.max_position_embeddings >= length, "model context shorter than new inference bound")
    require(model.config.vocab_size > 17, "fixed context-probe token unavailable")
    ids = torch.full((1, length), 17, dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        require(
            logits.shape[:2] == (1, length) and bool(torch.isfinite(logits).all()),
            "context probe invalid logits",
        )
    result = dict(
        passed=True,
        input_length=length,
        token_id=17,
        all_logits_finite=True,
        use_cache=False,
        native_bfloat16=True,
        scope="synthetic_diagnostic_context_no_development_content",
    )
    del ids, logits
    torch.cuda.empty_cache()
    return result


def checkpoint_and_development(
    trainer,
    model_config,
    tokenizer,
    views,
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
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.loading import load_model_and_tokenizer
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    require(evaluate is True and inputs["mode"] == "probe", "diagnostic-only checkpoint evaluation")
    quality = sibling("sdsc_student_quality_probe")
    settings = specification()
    completion_limit = settings["development"]["max_new_tokens"]
    context_limit = settings["development"]["max_model_input_tokens"]
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
        state = None
        gc.collect()
        checkpoint_sha = file_sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_dense_payload(
            saved,
            step=step,
            arm=inputs["arm"],
            dataset_sha=dataset_sha,
            protocol_sha=protocol_sha,
            protocol_artifact_sha=inputs["protocol"]["sha256"],
        )
        loaded = load_model_and_tokenizer(model_config, for_training=False)
        exact = util.strict_load_export(loaded.model, saved["model"])
        require(exact["key_count"] == 311, "diagnostic dense master tensor count differs")
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
        envelope_probe = inference_envelope_probe(model, enabled=True)
        counts = empty_counts()
        per_view = {name: empty_counts() for name in ("identity", *TRANSFORMATIONS)}
        structures = {name: empty_counts() for name in ("chain", "branch", "converging_dag")}
        generation_started = time.perf_counter()
        for ordinal in generation_ordinals(len(views), rank, evaluate=evaluate):
            view, row = views[ordinal], dev_rows[ordinal]
            example = view.example
            ids = torch.tensor(
                [row["input_ids"][: row["prefix_length"]]], device=next(model.parameters()).device
            )
            require(
                ids.shape[1] + completion_limit <= context_limit
                and ids.shape[1] + completion_limit <= model.config.max_position_embeddings,
                "order development generation exceeds declared envelope",
            )
            seed_everything(42 + ordinal)
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                max_new_tokens=completion_limit,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            require(torch.equal(generated[0, : ids.shape[1]], ids[0]), "generation changed prefix")
            prompt = format_model_prompt(view.prompt, tokenizer, model_config).model_facing_prompt
            same(
                tokenizer.encode(prompt, add_special_tokens=False),
                ids[0].tolist(),
                "view prompt encoding differs",
            )
            record = quality.response_record(
                example,
                generated[0, ids.shape[1] :].tolist(),
                tokenizer,
                prompt_ids=ids[0].tolist(),
                prompt_text=prompt,
                cap=completion_limit,
            )
            record.update(
                step=step,
                ordinal=ordinal,
                rank=rank,
                checkpoint_sha256=checkpoint_sha,
                parsed_trace=asdict(ProofGraphTask().parse_response(record["response_text"])),
                cohort="student_order_probe_dev",
                arm=inputs["arm"],
                source_example_id=view.source_example_id,
                view=view.view,
            )
            filename = f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl"
            append(output / filename, record)
            add_verification(counts, record["verification"])
            if evaluate:
                add_verification(per_view[view.view], record["verification"])
                if view.view == "identity":
                    add_verification(structures[example.metadata["structure"]], record["verification"])
            if counts["num_examples"] % 32 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="order_probe_development",
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
        per_view_counts=per_view,
        structure_counts=structures,
        inference_envelope_probe=envelope_probe,
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
        scope="student_order_probe_diagnostic_model_only",
        arm=inputs["arm"],
        reload_by_rank=results,
    )
    development = aggregate_development(
        results, step=step, checkpoint_sha=checkpoint["sha256"], dev_sha=dev_sha
    )
    return checkpoint, development


def staged_science(inputs, output):
    work = output.parent.resolve()
    require(
        set(inputs)
        == {
            "schema",
            "mode",
            "arm",
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
        inputs.get("schema") == INPUT_SCHEMA
        and inputs.get("mode") == "probe"
        and inputs.get("arm") in ("control", "treatment"),
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
    from posttrain_circuits.experiments.protocols import student_order_probe as protocol
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
    binding = protocol.resolve_student_order_probe_protocol(Path(inputs["science_root"]))
    require(binding.artifact_sha256 == file_sha(protocol_path), "resolved protocol artifact differs")
    require(
        binding.protocol_sha256 == inputs["protocol"]["protocol_sha256"], "resolved protocol core differs"
    )
    for name in (
        "sdsc_student_order_probe_worker.py",
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
    dataset = protocol.dataset_manifest(splits, arm=inputs["arm"])
    if rank == 0:
        atomic(output / "prepare-dataset-manifest.json", dataset)
        atomic(output / "prepare-data-isolation.json", isolation)
    return config, checkpoint, splits, dataset, isolation, binding


def execute(inputs, output, work):
    import torch
    from accelerate import Accelerator

    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.experiments.protocols import student_order_probe as protocol
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

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
    fit_views = protocol.fit_views(splits["student_fit"], arm=inputs["arm"])
    dev_views = protocol.development_views(splits["student_dev"])
    encoded_fit = [
        protocol.encode_view(view, loaded.tokenizer, config["model"], training=True) for view in fit_views
    ]
    encoded_dev = [
        protocol.encode_view(view, loaded.tokenizer, config["model"], training=False) for view in dev_views
    ]
    token_audit = protocol.validate_encoded_fit(encoded_fit)
    records = canonical_records(fit_views, encoded_fit, loaded.tokenizer, config["model"])
    settings = specification()
    same(binding.payload["training"], settings["training"], "accepted order training settings differ")
    max_steps = settings["training"]["optimizer_steps"]
    schedule = checkpoint_steps(inputs["mode"])
    trainer, optimizer, fsdp = build_trainer(
        loaded,
        records,
        config,
        checkpoint,
        rank,
        output,
        max_steps=max_steps,
        science_commit=binding.implementation_commit,
        arm=inputs["arm"],
    )
    observer = StepObserver(optimizer)
    dataset_sha = digest(canonical(dataset))
    dev_sha = digest(canonical([asdict(row) for row in splits["student_dev"]]))
    report = dict(
        schema=SCHEMA,
        mode=inputs["mode"],
        arm=inputs["arm"],
        passed=False,
        diagnostic_complete=False,
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
        source_kind="symbolic_canonical_order_probe",
        teacher_data_required=False,
        fit_base_examples=settings["data"]["fit_base_examples"],
        fit_training_views=settings["data"]["fit_examples"],
        consumed_fit_training_views=12 * 64,
        development_base_examples=settings["data"]["development_base_examples"],
        development_views=settings["data"]["development_examples"],
        optimizer_steps=0,
        data_audit=dict(passed=True, isolation=isolation, token_envelope=token_audit),
        fsdp=fsdp,
        development_generation=dict(
            max_new_tokens=settings["development"]["max_new_tokens"],
            max_model_input_length=settings["development"]["max_model_input_tokens"],
            max_training_model_input_length=settings["data"]["max_model_input_tokens"],
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
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
        if inputs["mode"] == "probe":
            for ordinal, view in enumerate(dev_views):
                row = encoded_dev[ordinal]
                append(
                    output / "prepare-dev-prompts.jsonl",
                    dict(
                        ordinal=ordinal,
                        example=asdict(view.example),
                        prompt_ids=row["input_ids"][: row["prefix_length"]],
                        source_example_id=view.source_example_id,
                        view=view.view,
                        prompt_text=format_model_prompt(
                            view.prompt, loaded.tokenizer, config["model"]
                        ).model_facing_prompt,
                    ),
                )
    for step in range(1, max_steps + 1):
        require(not trainer._terminate, "allocation termination requested")
        update = train_window(
            trainer, [row.prompt_id for row in records], encoded_fit, rank, step, observer=observer
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
        if step == settings["execution"]["full_state_restore_step"]:
            restore = full_state_restore(trainer, output, rank)
            restored = util.gather(restore)
            report["checkpoint_restore"] = dict(passed=all(row["passed"] for row in restored), ranks=restored)
        if step in schedule:
            dense, development = checkpoint_and_development(
                trainer,
                config["model"],
                loaded.tokenizer,
                dev_views,
                encoded_dev,
                output=output,
                rank=rank,
                step=step,
                inputs=inputs,
                dataset_sha=dataset_sha,
                protocol_sha=binding.protocol_sha256,
                dev_sha=dev_sha,
                evaluate=True,
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
        preparation_complete=False,
        diagnostic_complete=True,
        passed=True,
        training_input_tokens=trainer.token_budget.consumed,
    )
    require(
        len(report["checkpoints"]) == len(report["development"]) == len(STEPS),
        "diagnostic checkpoint/development evidence incomplete",
    )
    require(report["checkpoint_restore"]["passed"] is True, "full-state restore incomplete")
    require(report["selected_checkpoint"] is None, "diagnostic must never select a checkpoint")
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
            diagnostic_complete=False,
            preparation_complete=False,
            selected_checkpoint=None,
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
