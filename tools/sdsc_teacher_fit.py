#!/usr/bin/env python3
"""Four-H100 teacher-only fit/preflight; never a formal teacher acceptance claim."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path


def sibling(name):
    spec = importlib.util.spec_from_file_location("_fit_" + name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    sibling("sdsc_teacher_probe").progress_json(path, value)


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def runtime_and_allocation(args, contract, guards):
    guards.require(os.environ.get("SLURM_JOB_ID") == args.job_id, "Slurm job identity differs")
    guards.require(re.fullmatch(r"[1-9][0-9]*", args.job_id), "invalid actual Slurm job")
    guards.require(os.environ.get("SLURM_CPUS_PER_TASK") == "24", "fit requires 24 CPUs")
    guards.require(os.environ.get("SLURM_MEM_PER_NODE") == "196608", "fit requires 192 GiB")
    allocated_cpus = re.fullmatch(r"([1-9][0-9]*)(?:\(x1\))?", os.environ.get("SLURM_JOB_CPUS_PER_NODE", ""))
    guards.require(
        allocated_cpus is not None and int(allocated_cpus[1]) >= 24, "actual allocated CPU count is missing"
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    guards.require(
        len(devices) == len(set(devices)) == 4
        and all(value and value.strip() == value and value != "-1" for value in devices),
        "fit requires four distinct scheduler-assigned visible GPUs",
    )
    packages = {name: importlib.metadata.version(name) for name in contract.DEPENDENCIES}
    guards.require(packages == contract.DEPENDENCIES, "teacher fit runtime packages differ")
    expected = contract.FIXED_EXECUTION["runtime"]
    guards.require(platform.python_version() == expected["python"], "Python version differs")
    guards.require(file_hash(Path(sys.executable)) == expected["python_sha256"], "Python bytes differ")
    environment = sibling("sdsc_teacher_fit_memory").local_environment(args, guards)
    # The reviewed DDP policy is six threads on each of the four ranks,
    # before importing numerical code, even when Slurm allocates all 72 CPUs.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "6"
    return (
        {
            "job_id": args.job_id,
            "cpus": 24,
            "requested_cpus_per_task": 24,
            "allocated_cpus_on_node": int(allocated_cpus[1]),
            "allocated_cpus_slurm_value": os.environ["SLURM_JOB_CPUS_PER_NODE"],
            "memory_mib": 196608,
            "cuda_visible_devices": visible,
        },
        {"python": platform.python_version(), "executable": sys.executable, "packages": packages},
        environment,
    )


def actual_execution_plan(root, contract, guards):
    files = []
    for name in contract.KERNEL_PATHS:
        path = guards.real_path(root / name)
        raw = guards.file_bytes(path)
        files.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    return contract.execution_plan({"files": files})


def rank_observation(probe, adaptation_worker, output, rank, report):
    directory = output if rank == 0 else output / f"rank-{rank}-progress"
    directory.mkdir(parents=True, exist_ok=True)
    return adaptation_worker.adaptation_observation(probe, directory, report)


def validate_checkpoint_for_export(checkpoint, expected_manifest):
    from posttrain_circuits.learning.teacher.adaptation_fit import validate_fit_checkpoint

    observed = validate_fit_checkpoint(checkpoint, expected_manifest["metadata"])
    if observed["sha256"] != expected_manifest["sha256"]:
        raise ValueError("saved adapter checkpoint identity changed before export")


def export_checkpoint_collectively(
    checkpoint, state_manifest, dev_row, teacher, provenance, device, observe, *, rank, control
):
    import torch.distributed as dist

    from posttrain_circuits.learning.teacher.adaptation_fit import _shared_call

    _shared_call(
        lambda: validate_checkpoint_for_export(checkpoint, state_manifest),
        control,
        "saved adapter checkpoint verification",
    )
    exported = _shared_call(
        lambda: export_dense_checkpoint(checkpoint, dev_row, teacher, provenance, device, observe)
        if rank == 0
        else None,
        control,
        "fresh dense checkpoint export",
    )
    metadata = [exported[0] if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0, group=control)
    return exported, metadata[0]


def require_changed_world_rejection(checkpoint):
    from posttrain_circuits.learning.teacher.adaptation_fit import validate_fit_checkpoint

    try:
        validate_fit_checkpoint(checkpoint, {"world_size": 3})
    except ValueError as error:
        if str(error) != "teacher checkpoint identity differs: world_size":
            raise
        return True
    raise ValueError("changed-world checkpoint was accepted")


def checkpoint_memory_evidence(control, rank):
    import torch.distributed as dist

    from posttrain_circuits.learning.teacher.adaptation_fit import _shared_call

    usage = _shared_call(
        lambda: sibling("sdsc_teacher_fit_memory").memory_envelope(),
        control,
        "checkpoint cgroup headroom",
    )
    evidence = [None] * 4
    dist.all_gather_object(evidence, {"rank": rank, "cgroup_memory": usage}, group=control)
    return evidence


def export_dense_checkpoint(checkpoint, dev_row, teacher, provenance, device, observe):
    """Export a saved adapter through a fresh base, leaving training weights alone."""
    import torch
    from peft import PeftModel

    from posttrain_circuits.learning.teacher.adaptation import _tensor_digest, checkpoint_manifest, collate
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    observe.phase("checkpoint_dense_export", checkpoint=str(checkpoint))
    if (checkpoint / "merged").exists() or (checkpoint / "dense-manifest.json").exists():
        raise ValueError("dense checkpoint export destination already exists")
    loaded = load_model_and_tokenizer(teacher, for_training=False)
    base = loaded.model.to(device)
    adapted = PeftModel.from_pretrained(
        base, checkpoint / "adapter", is_trainable=False, local_files_only=True
    )
    adapted.eval()
    adapted.config.use_cache = True
    dense_modules = [
        (name, layer.base_layer.weight)
        for name, layer in adapted.get_base_model().named_modules()
        if hasattr(layer, "lora_A")
    ]
    dense_before = _tensor_digest(dense_modules)
    batch = collate([dev_row], loaded.tokenizer.pad_token_id, device)
    with torch.inference_mode():
        before = (
            adapted(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            .logits[:, -1]
            .float()
            .cpu()
        )
    merged = adapted.merge_and_unload(safe_merge=True).eval()
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise ValueError("export still contains adapter parameters")
    dense_after = _tensor_digest((name, merged.get_submodule(name).weight) for name, _ in dense_modules)
    if dense_after == dense_before:
        raise ValueError("adapter merge did not change actual dense teacher weights")
    del dense_modules
    merged.config._commit_hash = None
    merged.config.use_cache = True
    merged.save_pretrained(checkpoint / "merged", safe_serialization=True, max_shard_size="2GB")
    loaded.tokenizer.save_pretrained(checkpoint / "merged")
    with torch.inference_mode():
        expected = (
            merged(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            .logits[:, -1]
            .float()
            .cpu()
        )
    if not bool(torch.isfinite(before).all() and torch.isfinite(expected).all()):
        raise ValueError("nonfinite merged checkpoint logits")
    plan = provenance["actual_plan"]
    manifest = checkpoint_manifest(
        checkpoint,
        base_revision=teacher["model_revision"],
        plan=plan,
        training_provenance={key: value for key, value in provenance.items() if key != "actual_plan"},
    )
    write_json(checkpoint / "dense-manifest.json", manifest)
    metadata = {
        "manifest": manifest,
        "adapter_merge_max_logit_error": float((before - expected).abs().max()),
        "dense_weights_sha256_before": dense_before,
        "dense_weights_sha256_after": dense_after,
    }
    # All aliases of this temporary model are released before the verified
    # dense reload. The DDP training model remains untouched and resident.
    del adapted, merged, base, loaded, before, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "work-dir", "output-dir", "hf-home"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("run-id", "code-sha256", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("preflight", "full-fit"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    guards, contract = sibling("sdsc_teacher_prepare"), sibling("sdsc_teacher_fit_contract")
    guards.require(guards.SHA256.fullmatch(args.code_sha256), "invalid source snapshot identity")
    guards.real_path(args.science_root)
    allocation, runtime, environment = runtime_and_allocation(args, contract, guards)
    execution = actual_execution_plan(args.science_root, contract, guards)
    actual = contract.actual_plan(args.mode)
    if args.validate_only:
        print(
            json.dumps({"validated": True, "mode": args.mode, "formal_teacher_accepted": False}), flush=True
        )
        return 0
    rank, local_rank, world = (
        int(os.environ.get(name, "-1")) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    guards.require(world == 4 and rank == local_rank and 0 <= rank < 4, "unexpected torchrun rank layout")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.science_root / "src"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": contract.task_for_mode(args.mode),
        "mode": args.mode,
        "artifact_kind": "teacher_adaptation_fit_execution",
        "job_id": args.job_id,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "passed": False,
        "exit_code": 1,
        "accepted_science": False,
        "formal_teacher_accepted": False,
        "full_teacher_ready": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "student_training_started": False,
        "readiness_artifact_produced": False,
        "resumable": False,
        "checkpoint_resume_supported": True,
        "automatic_resume": False,
        "teacher_training_started": False,
        "training_started": False,
        "world_size": 4,
        "allocation": allocation,
        "runtime": runtime,
        "environment": environment,
        "execution_plan": execution,
        "execution_plan_sha256": contract.sha256_value(execution),
        "actual_plan": actual,
        "actual_plan_sha256": contract.sha256_value(actual),
        "passed_means": "bounded teacher fitting execution, not formal teacher acceptance",
        "original_128_token_readiness_pass_claim": False,
    }
    probe, adaptation_worker = sibling("sdsc_teacher_probe"), sibling("sdsc_teacher_adapt")
    observe = rank_observation(probe, adaptation_worker, args.output_dir, rank, report)
    started = time.monotonic()
    try:
        observe.phase("scientific_imports", rank=rank)
        import torch
        import torch.distributed as dist

        from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
        from posttrain_circuits.core.config import compose_config
        from posttrain_circuits.learning.teacher.adaptation import dataset_manifest
        from posttrain_circuits.learning.teacher.adaptation_evaluation import (
            build_adaptation_probes,
            combine_development_shards,
            measure_development_shard,
        )
        from posttrain_circuits.learning.teacher.adaptation_fit import (
            FitPlan,
            _shared_call,
            encode_fit_example,
            make_fit_examples,
            run_adapter_fit,
        )
        from posttrain_circuits.learning.teacher.evaluation import TeacherReadinessThresholds
        from posttrain_circuits.models.adapted_teacher import load_adapted_teacher
        from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda

        torch.set_num_threads(6)
        torch.cuda.set_device(local_rank)
        guards.require(torch.cuda.device_count() == 4, "actual visible CUDA count differs")
        device = torch.device("cuda", local_rank)
        guards.require("H100" in torch.cuda.get_device_name(device), "teacher fitting requires H100")
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        control = dist.group.WORLD
        training = dist.new_group(backend="nccl", timeout=timedelta(minutes=20))
        value = torch.ones((), device=device)
        work = dist.all_reduce(value, group=training, async_op=True)
        work.wait(timeout=timedelta(seconds=120))
        guards.require(float(value) == 4, "four-rank NCCL scalar probe failed")
        guards.require(
            torch.version.cuda == "12.8" and tuple(torch.cuda.nccl.version()) == (2, 27, 3),
            "CUDA/NCCL versions differ",
        )
        report["nccl_probe_passed"] = True
        report["rank"] = rank
        report["local_rank"] = local_rank
        config = compose_config(list(guards.OVERRIDES), config_root=args.science_root / "configs")
        teacher = config["teacher"]
        guards.require(teacher["model_revision"] == contract.BASE_REVISION, "teacher origin differs")
        thresholds = asdict(TeacherReadinessThresholds())
        report["teacher_readiness_thresholds"] = thresholds
        report["teacher_configuration"] = teacher
        report["task_configuration"] = config["task"]
        observe.phase("independent_dataset_generation", rank=rank)
        fit_plan = FitPlan.preflight() if args.mode == "preflight" else FitPlan()
        splits = _shared_call(lambda: make_fit_examples(config["task"], fit_plan), control, "data generation")
        report["adaptation_dataset"] = dataset_manifest(splits)
        report["adaptation_dataset_sha256"] = sha256_value(report["adaptation_dataset"])
        _shared_call(
            lambda: guards.require(
                report["adaptation_dataset_sha256"] == actual["dataset_manifest_sha256"],
                "generated fit/development data differ from the reviewed population",
            ),
            control,
            "frozen dataset identity",
        )
        isolated = _shared_call(
            lambda: adaptation_worker.formal_isolation(splits, guards) if rank == 0 else None,
            control,
            "full formal-family isolation",
        )
        isolated_values = [isolated]
        dist.broadcast_object_list(isolated_values, src=0, group=control)
        report["formal_isolation"] = isolated_values[0]
        if rank == 0:
            write_json(args.output_dir / "data-isolation.json", isolated_values[0])
        observe.phase("pinned_teacher_load", rank=rank)
        loaded = _shared_call(
            lambda: load_model_and_tokenizer(teacher, for_training=False), control, "base load"
        )
        base = _shared_call(lambda: move_model_to_local_cuda(loaded.model), control, "base placement")
        rows = [encode_fit_example(e, loaded.tokenizer, teacher) for e in splits["teacher_fit"]]
        dev_rows = [encode_fit_example(e, loaded.tokenizer, teacher) for e in splits["teacher_dev"]]
        loaded = replace(loaded, model=None)
        report["trainer_identity"] = {
            "base_revision": contract.BASE_REVISION,
            "tokenizer_sha256": loaded.tokenizer_hash,
            "fit_dataset_sha256": report["adaptation_dataset"]["teacher_fit"]["examples_sha256"],
            "protocol_sha256": sha256_value(
                {
                    "actual_plan": actual,
                    "execution_plan_sha256": report["execution_plan_sha256"],
                    "teacher_readiness_thresholds": thresholds,
                }
            ),
        }
        semantic, specifications, tokenized, skipped = build_adaptation_probes(
            splits["teacher_dev"], loaded, teacher
        )
        checkpoint_records = []

        def checkpoint_hook(checkpoint, state_manifest):
            step = state_manifest["metadata"]["completed_updates"]
            provenance = {
                "actual_plan": actual,
                "run_id": args.run_id,
                "job_id": args.job_id,
                "code_sha256": args.code_sha256,
                "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
                "train_metrics_sha256": state_manifest["metadata"]["train_metrics_sha256"],
                "optimizer_steps": step,
                "consumed_tokens": state_manifest["metadata"]["consumed_tokens"],
            }
            export, export_metadata = export_checkpoint_collectively(
                checkpoint,
                state_manifest,
                dev_rows[0],
                teacher,
                provenance,
                device,
                observe,
                rank=rank,
                control=control,
            )
            shared_export = [export_metadata]
            dense = _shared_call(
                lambda: load_adapted_teacher(
                    checkpoint, shared_export[0]["manifest"]["sha256"], teacher, device=device
                ),
                control,
                "content-verified dense checkpoint reload",
            )

            def verify_reload(dense=dense):
                if rank != 0:
                    return None
                from posttrain_circuits.learning.teacher.adaptation import collate

                batch = collate([dev_rows[0]], loaded.tokenizer.pad_token_id, device)
                with torch.inference_mode():
                    observed = (
                        dense.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        .logits[:, -1]
                        .float()
                        .cpu()
                    )
                torch.testing.assert_close(observed, export[1], atol=1e-5, rtol=1e-5)
                return float((observed - export[1]).abs().max())

            reload_error = _shared_call(verify_reload, control, "dense reload equivalence")
            measured = _shared_call(
                lambda dense=dense: measure_development_shard(
                    dense.model,
                    dense,
                    teacher,
                    splits["teacher_dev"],
                    specifications,
                    checkpoint / "eval" / f"rank-{rank}",
                    rank=rank,
                    world_size=4,
                    observe=observe,
                ),
                control,
                "merged development measurements",
            )
            measurements = [None] * 4
            dist.all_gather_object(measurements, measured, group=control)
            development = combine_development_shards(splits["teacher_dev"], specifications, measurements)
            # The verification function's default also owns the evaluation
            # model; release that reference before collecting memory evidence.
            del verify_reload, dense
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint_memory = checkpoint_memory_evidence(control, rank)
            if rank == 0:
                development.pop("prefix_scores")
                development["adapted_teacher_sha256"] = shared_export[0]["manifest"]["sha256"]
                development["semantic_prefix_manifest_sha256"] = semantic["sha256"]
                development["tokenized_prefix_manifest_sha256"] = tokenized["sha256"]
                development["alignment_skips"] = skipped
                write_json(checkpoint / "dev-capability.json", development)
                write_json(checkpoint / "eval" / "semantic-prefix-manifest.json", semantic)
                write_json(checkpoint / "eval" / "tokenized-prefix-manifest.json", tokenized)
                checkpoint_records.append(
                    {
                        "step": step,
                        "optimizer_steps": step,
                        "directory": checkpoint.relative_to(args.output_dir).as_posix(),
                        "consumed_tokens": provenance["consumed_tokens"],
                        "adapted_teacher_sha256": shared_export[0]["manifest"]["sha256"],
                        "dense_manifest_sha256": sha256_file(checkpoint / "dense-manifest.json"),
                        "trainer_state_manifest_sha256": sha256_file(checkpoint / "manifest.json"),
                        "training_metrics_sha256": provenance["train_metrics_sha256"],
                        "dev_metrics_sha256": sha256_file(checkpoint / "dev-capability.json"),
                        "dev_metrics_passed": development["metrics_passed"],
                        "adapter_merge_max_logit_error": shared_export[0]["adapter_merge_max_logit_error"],
                        "dense_reload_max_logit_error": reload_error,
                        "cgroup_memory_by_rank": checkpoint_memory,
                    }
                )

        model, summary = run_adapter_fit(
            base,
            loaded.tokenizer,
            rows,
            args.output_dir,
            identity=report["trainer_identity"],
            observe=observe,
            plan=fit_plan,
            control_group=control,
            training_group=training,
            checkpoint_hook=checkpoint_hook,
        )
        # Exercise changed-world metadata rejection on the real saved checkpoint
        # before any tensor state can be loaded, in both operational modes.
        final_checkpoint = args.output_dir / "checkpoints" / f"step-{actual['optimizer_steps']:06d}"
        changed_world_rejected = _shared_call(
            lambda: require_changed_world_rejection(final_checkpoint), control, "changed-world rejection"
        )
        usage = _shared_call(
            lambda: sibling("sdsc_teacher_fit_memory").memory_envelope(), control, "cgroup headroom"
        )
        summaries = [None] * 4
        dist.all_gather_object(
            summaries, {"rank": rank, "training": summary, "cgroup_memory": usage}, group=control
        )
        _shared_call(
            lambda: finish_report(
                args, report, checkpoint_records, summaries, changed_world_rejected, contract, started
            )
            if rank == 0
            else None,
            control,
            "final execution evidence",
        )
        del model, base
        observe.phase("teacher_fit_execution_complete", rank=rank)
        dist.destroy_process_group()
        return 0
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["failure_stage"] = observe.stage
        observe.failed(error)
        raise
    finally:
        if rank == 0:
            write_json(args.output_dir / "teacher-fit.json", report)
        observe.close()


def finish_report(args, report, checkpoints, ranks, changed_world_rejected, contract, started):
    """Bind actual completed measurements; execution PASS remains nonaccepting."""
    from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value

    actual = report["actual_plan"]
    if [row["step"] for row in checkpoints] != actual["checkpoint_steps"]:
        raise ValueError("scheduled checkpoint measurements are incomplete")
    if len(ranks) != 4 or {row["rank"] for row in ranks} != {0, 1, 2, 3}:
        raise ValueError("actual rank summaries are incomplete")
    if changed_world_rejected is not True:
        raise ValueError("changed-world checkpoint rejection was not verified")
    training = ranks[0]["training"]
    if any(
        row["training"]["completed_updates"] != actual["optimizer_steps"]
        or row["training"]["consumed_tokens"] != training["consumed_tokens"]
        for row in ranks
    ):
        raise ValueError("rank update or token summaries differ")
    if any(row["cgroup_memory"].get("passed") is not True for row in ranks):
        raise ValueError("not every rank verified cgroup headroom")
    if any(
        type(row["dev_metrics_passed"]) is not bool
        or not math.isfinite(row["dense_reload_max_logit_error"])
        or not math.isfinite(row["adapter_merge_max_logit_error"])
        for row in checkpoints
    ):
        raise ValueError("checkpoint measurements are invalid")
    for checkpoint in checkpoints:
        memory = checkpoint.get("cgroup_memory_by_rank", [])
        if (
            len(memory) != 4
            or {row["rank"] for row in memory} != {0, 1, 2, 3}
            or any(row["cgroup_memory"].get("passed") is not True for row in memory)
        ):
            raise ValueError("checkpoint cgroup headroom evidence is incomplete")
    if args.mode == "preflight":
        if not all(row["training"]["same_world_resume_verified"] for row in ranks):
            raise ValueError("preflight did not verify real same-world resume on all ranks")
        report["same_world_resume_performed_this_attempt"] = True
        report["same_world_resume_prerequisite_job_id"] = None
    else:
        prerequisites = json.loads((args.work_dir / "prerequisites.json").read_bytes())
        report["same_world_resume_performed_this_attempt"] = False
        report["same_world_resume_prerequisite_job_id"] = prerequisites["preflight_job_id"]
    selected = (
        next((row for row in checkpoints if row["dev_metrics_passed"]), None)
        if args.mode == "full-fit"
        else None
    )
    report["selected_checkpoint_sha256"] = selected["adapted_teacher_sha256"] if selected else None
    write_json(
        args.output_dir / "checkpoint-selection.json",
        {
            "selection_rule": actual["checkpoint_selection"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "selected_step": selected["step"] if selected else None,
            "formal_teacher_accepted": False,
            "original_128_token_readiness_pass_claim": False,
            "evaluation_max_new_tokens": 256,
        },
    )
    # Progress observers keep writing diagnostics until final close; they are
    # excluded from this immutable checkpoint inventory and separately covered
    # by the stopped-worker publication receipt.
    files = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("output contains a symlink")
        relative = path.relative_to(args.output_dir)
        if path.is_file() and relative.parts[0] in {
            "checkpoints",
            "checkpoint-selection.json",
            "train-metrics.jsonl",
            "data-isolation.json",
        }:
            files.append(
                {"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    manifest = {
        "artifact_kind": "teacher_adaptation_checkpoint_set",
        "inventory_scope": "checkpoint_and_training_artifacts",
        "formal_teacher_accepted": False,
        "identity": {
            name: report[name]
            for name in ("run_id", "job_id", "code_sha256", "execution_plan_sha256", "actual_plan_sha256")
        },
        "base_revision": contract.BASE_REVISION,
        "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
        "checkpoints": checkpoints,
        "files": files,
    }
    manifest["sha256"] = sha256_value(manifest)
    write_json(args.output_dir / "checkpoint-manifest.json", manifest)
    report.update(
        checkpoint_manifest_sha256=sha256_file(args.output_dir / "checkpoint-manifest.json"),
        checkpoint_set_sha256=manifest["sha256"],
        optimizer_steps=actual["optimizer_steps"],
        consumed_tokens=training["consumed_tokens"],
        consumed_sequences=actual["optimizer_steps"] * 64,
        completed_epochs=actual["epochs"],
        rank_summaries=ranks,
        selection_does_not_stop_training=True,
        checks={
            name: True
            for name in (
                "isolated_data",
                "finite_loss_and_gradients",
                "exact_optimizer_steps",
                "exact_global_batch_and_tokens",
                "nonzero_adapter_update",
                "frozen_base_before_merge",
                "complete_module_coverage",
                "all_four_ranks",
                "same_world_resume",
                "dense_export_reload",
                "all_scheduled_merged_dev_evaluations",
                "cgroup_headroom",
            )
        },
        passed=True,
        exit_code=0,
        elapsed_seconds=time.monotonic() - started,
    )
    report["checks"]["changed_world_resume_rejected"] = changed_world_rejected


if __name__ == "__main__":
    raise SystemExit(main())
