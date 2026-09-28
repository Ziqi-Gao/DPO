#!/usr/bin/env python3
"""One fixed teacher-adaptation execution preflight, not teacher acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import stat
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

TASK = "qwen3-v2-teacher-adapt"
DATASET = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/"
    "20260918T021516Z-82d6fc51200a-2cf5f753/f98dca38ea9d4d9bbe4f3917e151a796/artifacts/dataset"
)
DATASET_MANIFEST_SHA = "bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189"
PEFT_VERSION = "0.17.1"


def sibling(name):
    spec = importlib.util.spec_from_file_location("_adapt_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adaptation_observation(probe, directory, report):
    """Reuse durable diagnostics with honest teacher-only training state."""

    class AdaptationObservation(probe.ProbeObservation):
        def phase(self, name, **context):
            if name == "teacher_training_start":
                report["training_started"] = report["teacher_training_started"] = True
            if name == "teacher_optimizer_window_committed":
                report["completed_optimizer_steps"] = context["completed_optimizer_steps"]
            super().phase(name, **context)

        def snapshot(self):
            value = super().snapshot()
            value.update(
                artifact_kind="teacher_adaptation_progress",
                training_started=bool(report.get("teacher_training_started", False)),
                teacher_training_started=bool(report.get("teacher_training_started", False)),
                student_training_started=False,
                readiness_artifact_produced=False,
                completed_optimizer_steps=report.get("completed_optimizer_steps", 0),
            )
            return value

    return AdaptationObservation(directory, report)


def formal_isolation(splits, guards):
    """Inspect identities only; no formal example enters training/evaluation."""
    from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
    from posttrain_circuits.datasets.proofgraph.splits import SPLITS
    from posttrain_circuits.learning.teacher.adaptation import isolation_inventory, reject_formal_overlap

    raw = guards.file_bytes(DATASET / "manifest.json")
    guards.require(hashlib.sha256(raw).hexdigest() == DATASET_MANIFEST_SHA, "formal family manifest changed")
    manifest = json.loads(raw)
    guards.require(set(manifest["splits"]) == set(SPLITS), "formal family splits differ")
    inventory = isolation_inventory(splits)
    evidence = {}
    for name in SPLITS:
        boundary = manifest["splits"][name]
        guards.require(boundary["path"] == name + "/examples.jsonl", "formal split path differs")
        path = guards.real_path(DATASET / boundary["path"])
        checksum = hashlib.sha256()
        count = size = 0
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            before = os.fstat(stream.fileno())
            guards.require(
                stat.S_ISREG(before.st_mode) and before.st_size < 1024**3, "invalid formal split file"
            )
            while line := stream.readline(128 * 1024 + 1):
                guards.require(len(line) <= 128 * 1024, "formal row exceeds bound")
                checksum.update(line)
                size += len(line)
                count += 1
                guards.require(size <= 1024**3 and count <= 144000, "formal split exceeds bounds")
                reject_formal_overlap(inventory, deserialize_example(json.loads(line)))
            after = os.fstat(stream.fileno())
        guards.require(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and count == boundary["num_examples"]
            and checksum.hexdigest() == boundary["examples_file_sha256"],
            "formal split changed",
        )
        evidence[name] = {"rows": count, "bytes": size, "sha256": checksum.hexdigest()}
    guards.require(sum(value["rows"] for value in evidence.values()) == 144000, "formal family count differs")
    return {
        "passed": True,
        "formal_manifest_sha256": DATASET_MANIFEST_SHA,
        "formal_splits": evidence,
        "overlaps": {key: 0 for key in inventory},
        "formal_records_used_for_optimization_or_development_scoring": 0,
    }


def evaluate_dev(model, loaded, teacher, examples, output, observe):
    from posttrain_circuits.cli.evaluate_teacher_readiness import _prefix_scores
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator
    from posttrain_circuits.learning.teacher.evaluation import TeacherReadinessThresholds

    capability = sibling("sdsc_teacher_capability")
    generator = HfTeacherCandidateGenerator(model, loaded.tokenizer, max_new_tokens=128, model_config=teacher)
    generated = {}
    task = ProofGraphTask()
    with (output / "dev-attempts.jsonl").open("x") as stream:
        for index, example in enumerate(examples):
            observe.phase("merged_teacher_dev_generation", example_id=example.example_id, index=index)
            result = generator(
                example=example,
                candidate_index=0,
                actual_sampling_seed=42 + index,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
            )
            generated[example.example_id] = result.response_text
            row = {
                "example_id": example.example_id,
                **asdict(result),
                "verification": asdict(task.verify(example, task.parse_response(result.response_text))),
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    observe.phase("merged_teacher_dev_prefixes")
    semantic, tokenized, tokenized_manifest, skipped = capability.prefix_probes(
        examples, task, loaded, teacher
    )
    scores = [
        score
        for spec in tokenized
        if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
        for score in _prefix_scores(model, spec, top_k=128)
    ]
    metrics = capability.capability_metrics(examples, generated, scores, TeacherReadinessThresholds())
    return {
        **metrics,
        "scope": "independent teacher-dev; not original readiness or held-out confirmation",
        "formal_teacher_accepted": False,
        "prefix_scores": [asdict(score) for score in scores],
        "semantic_prefix_manifest": semantic,
        "tokenized_prefix_manifest": tokenized_manifest,
        "skipped_prefix_pairs": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "work-dir", "output-dir", "hf-home"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("run-id", "code-sha256", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    guards = sibling("sdsc_teacher_prepare")
    probe = sibling("sdsc_teacher_probe")
    guards.real_path(args.science_root)
    guards.require(guards.SHA256.fullmatch(args.code_sha256), "invalid snapshot identity")
    allocation, runtime, environment = (
        guards.allocation(args),
        guards.runtime_identity(),
        guards.local_environment(args),
    )
    guards.require(
        importlib.metadata.version("peft") == PEFT_VERSION, "adaptation requires pinned PEFT 0.17.1"
    )
    runtime["packages"]["peft"] = PEFT_VERSION
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.science_root / "src"))
    if args.validate_only:
        print(json.dumps({"validated": True, "exploratory": True, "accepted_science": False}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "task": TASK,
        "artifact_kind": "teacher_adaptation_preflight",
        "adaptation_preflight": True,
        "job_id": args.job_id,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "passed": False,
        "exit_code": 1,
        "exploratory": True,
        "accepted_science": False,
        "full_teacher_ready": False,
        "training_started": False,
        "student_training_started": False,
        "teacher_training_started": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "resumable": False,
        "readiness_artifact_produced": False,
        "allocation": allocation,
        "runtime": runtime,
        "environment": environment,
        "passed_means": "eight-step teacher adaptation execution only",
    }
    observe = adaptation_observation(probe, args.output_dir, report)
    started = time.monotonic()
    try:
        observe.phase("scientific_imports")
        import numpy as np
        import torch

        from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
        from posttrain_circuits.core.config import compose_config
        from posttrain_circuits.learning.teacher.adaptation import (
            PREFLIGHT,
            dataset_manifest,
            encode_example,
            make_examples,
            run_adapter_preflight,
        )
        from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda

        config = compose_config(list(guards.OVERRIDES), config_root=args.science_root / "configs")
        teacher = config["teacher"]
        report["task_configuration"] = config["task"]
        report["teacher_configuration"] = teacher
        guards.require(
            teacher["model_revision"] == PREFLIGHT["base_revision"], "teacher base revision differs"
        )
        guards.require(torch.cuda.device_count() == 1, "preflight requires one actual visible CUDA device")
        torch.set_num_threads(24)
        random.seed(PREFLIGHT["seed"])
        np.random.seed(PREFLIGHT["seed"])
        torch.manual_seed(PREFLIGHT["seed"])
        torch.cuda.manual_seed_all(PREFLIGHT["seed"])
        report["gpu"] = guards.gpu_identity()
        report["adaptation_plan"] = dict(PREFLIGHT)
        report["adaptation_plan_sha256"] = sha256_value(PREFLIGHT)
        observe.phase("independent_dataset_generation")
        splits = make_examples(config["task"])
        report["adaptation_dataset"] = dataset_manifest(splits)
        observe.phase("formal_family_isolation")
        isolation = formal_isolation(splits, guards)
        probe.progress_json(args.output_dir / "data-isolation.json", isolation)
        observe.phase("pinned_teacher_load")
        loaded = load_model_and_tokenizer(teacher, for_training=False)
        base = move_model_to_local_cuda(loaded.model)
        guards.require(
            loaded.resolved_model_commit == PREFLIGHT["base_revision"], "loaded teacher revision differs"
        )
        rows = [encode_example(example, loaded.tokenizer, teacher) for example in splits["teacher_fit"]]
        dev_rows = [encode_example(example, loaded.tokenizer, teacher) for example in splits["teacher_dev"]]
        report["token_envelope"] = {
            "max_input": max(len(row["input_ids"]) for row in rows + dev_rows),
            "max_prefix": max(row["prefix_length"] for row in rows + dev_rows),
            "max_response": max(row["response_length"] for row in rows + dev_rows),
        }
        report["base_generation_config"] = base.generation_config.to_dict()
        loaded = replace(loaded, model=None)
        reloaded, reloaded_tokenizer, training_metrics, manifest = run_adapter_preflight(
            base,
            loaded.tokenizer,
            rows,
            dev_rows,
            args.output_dir,
            observe=observe,
            require=guards.require,
            training_provenance={
                "run_id": args.run_id,
                "job_id": args.job_id,
                "code_sha256": args.code_sha256,
                "dataset_manifest_sha256": sha256_value(report["adaptation_dataset"]),
            },
        )
        del base
        loaded = replace(loaded, tokenizer=reloaded_tokenizer)
        report.update(training_metrics)
        probe.progress_json(args.output_dir / "checkpoint-manifest.json", manifest)
        report["checkpoint_manifest_sha256"] = sha256_file(args.output_dir / "checkpoint-manifest.json")
        report["adapted_teacher_sha256"] = manifest["sha256"]
        metrics = evaluate_dev(reloaded, loaded, teacher, splits["teacher_dev"], args.output_dir, observe)
        probe.progress_json(args.output_dir / "dev-capability.json", metrics)
        report["dev_metrics"] = metrics["metrics"]
        report["dev_checks"] = metrics["checks"]
        training_guards = sibling("sdsc_training_preflight")
        report["cgroup_memory"] = training_guards.memory_envelope()
        report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["checks"] = {
            "isolated_data": True,
            "finite_loss_and_gradients": True,
            "eight_real_optimizer_steps": True,
            "nonzero_adapter_update": True,
            "frozen_base_before_merge": True,
            "complete_module_coverage": True,
            "dense_export_reload": True,
            "merged_dev_evaluated": True,
            "cgroup_headroom": True,
        }
        report.update(
            passed=True,
            exit_code=0,
            optimizer_steps=PREFLIGHT["optimizer_steps"],
            elapsed_seconds=time.monotonic() - started,
        )
        observe.phase("preflight_complete")
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["failure_stage"] = observe.stage
        observe.failed(error)
        raise
    finally:
        probe.progress_json(args.output_dir / "teacher-adapt.json", report)
        observe.close()
    print(
        json.dumps(
            {"passed": True, "teacher_accepted": False, "report": str(args.output_dir / "teacher-adapt.json")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
