#!/usr/bin/env python3
"""Bounded, exploratory frozen-validation teacher capability measurement.

This is not the formal readiness CLI and never publishes a readiness artifact.
Generation, prefix construction/scoring, and all thresholds match that CLI;
only its pure metric reduction is repeated here to avoid inventing formal Git
bindings for a diagnostic source snapshot. CPU fixtures compare that reduction
directly with the scientific implementation, including rejection cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from dataclasses import asdict
from pathlib import Path

TASK = "qwen3-v2-teacher-capability-probe"
DATASET = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/"
    "20260918T021516Z-82d6fc51200a-2cf5f753/f98dca38ea9d4d9bbe4f3917e151a796/"
    "artifacts/dataset/validation/examples.jsonl"
)
DATASET_SHA256 = "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3"
POPULATION = 128
SPLIT_ROWS = 10000
MAX_DATASET_BYTES = 128 * 1024**2
MAX_RECORD_BYTES = 128 * 1024
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
THRESHOLDS = {
    "minimum_teacher_answer_accuracy": 0.90,
    "minimum_teacher_exact_proof_accuracy": 0.85,
    "minimum_teacher_first_rule_top1_accuracy": 0.80,
    "minimum_teacher_intermediate_top1_accuracy": 0.80,
    "minimum_teacher_topk_mass": 0.90,
    "minimum_teacher_topk_target_coverage": 0.90,
    "minimum_corrupted_prefix_recovery_accuracy": 0.70,
    "minimum_causal_shift_logprob": 0.0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population(path=DATASET, expected_sha256=DATASET_SHA256):
    """Authenticate the complete frozen split, retaining its fixed first 128."""
    require(not any(p.is_symlink() for p in (path, *path.parents)), "dataset has a symlink")
    rows, checksum = [], hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= MAX_DATASET_BYTES,
            "invalid validation split file",
        )
        count = total = 0
        while raw := stream.readline(MAX_RECORD_BYTES + 1):
            count += 1
            total += len(raw)
            require(
                len(raw) <= MAX_RECORD_BYTES and total <= MAX_DATASET_BYTES and count <= SPLIT_ROWS,
                "validation split exceeds bounded record or population size",
            )
            checksum.update(raw)
            if count <= POPULATION:
                rows.append(json.loads(raw))
        after = os.fstat(stream.fileno())
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        and checksum.hexdigest() == expected_sha256,
        "persistent validation split differs from the verified publication",
    )
    require(count == SPLIT_ROWS and len(rows) == POPULATION, "incomplete frozen validation population")
    require(len({row["example_id"] for row in rows}) == POPULATION, "duplicate example identity")
    return rows


def capability_metrics(examples, generated, prefix_scores, thresholds):
    """Pure reduction equivalent to learning.teacher.evaluation, without formal bindings.

    Keep the canonical *step equality* requirement, parser-gated answer accuracy,
    minimum (not mean) top-k mass, and every prefix condition. Equivalence fixtures
    must change if the authoritative scientific reduction ever changes.
    """
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    require(bool(examples), "teacher capability requires a nonempty frozen validation set")
    require(
        set(generated) == {example.example_id for example in examples},
        "teacher responses do not exactly cover validation examples",
    )
    task = ProofGraphTask()
    answer_correct = exact_proof_correct = format_valid = 0
    rows = []
    for example in examples:
        parsed = task.parse_response(generated[example.example_id])
        verified = task.verify(example, parsed)
        format_valid += int(parsed.parse_valid)
        derived_answer_correct = bool(parsed.parse_valid and parsed.answer == example.label)
        answer_correct += int(derived_answer_correct)
        exact_proof = bool(
            verified.proof_valid and verified.answer_correct and parsed.steps == example.canonical_proof
        )
        exact_proof_correct += int(exact_proof)
        rows.append(
            {
                "example_id": example.example_id,
                "format_valid": parsed.parse_valid,
                "answer_correct": derived_answer_correct,
                "exact_proof_correct": exact_proof,
                "verification_reward": verified.reward,
                "verification_error": verified.error_code,
            }
        )
    canonical = [score for score in prefix_scores if score.prefix_kind == "canonical"]
    corrupted = [score for score in prefix_scores if score.prefix_kind != "canonical"]

    def stage_accuracy(stage):
        selected = [score for score in canonical if score.stage == stage]
        return sum(score.top1_correct for score in selected) / len(selected) if selected else 0.0

    metrics = {
        "answer_accuracy": answer_correct / len(examples),
        "exact_proof_accuracy": exact_proof_correct / len(examples),
        "format_validity": format_valid / len(examples),
        "first_rule_top1_accuracy": stage_accuracy("first_rule_selection"),
        "intermediate_top1_accuracy": stage_accuracy("intermediate_conclusion"),
        "topk_target_coverage": (
            sum(score.target_in_topk for score in prefix_scores) / len(prefix_scores)
            if prefix_scores
            else 0.0
        ),
        "minimum_topk_mass": min((score.minimum_topk_mass for score in prefix_scores), default=0.0),
        "corrupted_prefix_recovery_accuracy": (
            sum(score.top1_correct for score in corrupted) / len(corrupted) if corrupted else 0.0
        ),
        "minimum_causal_shift_logprob": min(
            (score.causal_shift_logprob for score in prefix_scores), default=float("-inf")
        ),
    }
    checks = {
        "answer_accuracy": metrics["answer_accuracy"] >= thresholds.minimum_teacher_answer_accuracy,
        "exact_proof_accuracy": metrics["exact_proof_accuracy"]
        >= thresholds.minimum_teacher_exact_proof_accuracy,
        "first_rule_top1_accuracy": (
            metrics["first_rule_top1_accuracy"] >= thresholds.minimum_teacher_first_rule_top1_accuracy
        ),
        "intermediate_top1_accuracy": (
            metrics["intermediate_top1_accuracy"] >= thresholds.minimum_teacher_intermediate_top1_accuracy
        ),
        "topk_mass": metrics["minimum_topk_mass"] >= thresholds.minimum_teacher_topk_mass,
        "topk_target_coverage": metrics["topk_target_coverage"]
        >= thresholds.minimum_teacher_topk_target_coverage,
        "corrupted_prefix_recovery": (
            metrics["corrupted_prefix_recovery_accuracy"]
            >= thresholds.minimum_corrupted_prefix_recovery_accuracy
        ),
        "causal_shift": bool(prefix_scores)
        and all(score.causal_shift_valid for score in prefix_scores)
        and metrics["minimum_causal_shift_logprob"] >= thresholds.minimum_causal_shift_logprob,
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "full_generation_rows": rows,
        "metrics_passed": all(checks.values()),
    }


def prefix_probes(examples, task, loaded, teacher):
    """Mirror the original readiness CLI's fixed, tokenizer-aligned probe selection."""
    from posttrain_circuits.causal_circuits.metrics.probes import (
        build_semantic_probe_specs,
        semantic_probe_manifest,
        tokenize_probe_specs,
        tokenized_probe_manifest,
    )

    def tokenize(semantic):
        return tokenize_probe_specs(
            semantic,
            loaded.tokenizer,
            tokenizer_id=loaded.tokenizer_id,
            tokenizer_revision=loaded.requested_tokenizer_revision,
            model_config=teacher,
        )

    pairs, skipped, seen = [], [], set()
    for example in examples:
        if example.pair_group_id in seen or len(example.canonical_proof) < 2:
            continue
        pair = task.make_counterfactual(example, "active_support_path_swap", 42)
        try:
            tokenize(semantic_probe_manifest(build_semantic_probe_specs([pair], subset="validation")))
        except ValueError as error:
            skipped.append({"example_id": example.example_id, "reason": str(error)})
            continue
        pairs.append(pair)
        seen.add(example.pair_group_id)
    require(bool(pairs), "no tokenizer-aligned depth>1 teacher prefix probes were available")
    semantic = semantic_probe_manifest(build_semantic_probe_specs(pairs, subset="validation"))
    tokenized = tokenize(semantic)
    require(0 < len(tokenized) <= POPULATION * 3, "unexpected diagnostic prefix population")
    manifest = tokenized_probe_manifest(tokenized, semantic_manifest_hash=semantic["sha256"])
    return semantic, tokenized, manifest, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "work-dir", "output-dir", "hf-home"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("run-id", "code-sha256", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    specification = importlib.util.spec_from_file_location(
        "sdsc_teacher_guards", Path(__file__).with_name("sdsc_teacher_prepare.py")
    )
    guards = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(guards)
    guards.real_path(args.science_root)
    require((args.science_root / "src").is_dir(), "snapshot science source missing")
    require(guards.SHA256.fullmatch(args.code_sha256 or ""), "invalid snapshot identity")
    allocation = guards.allocation(args)
    runtime = guards.runtime_identity()
    environment = guards.local_environment(args)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.science_root / "src"))
    from posttrain_circuits.core.config import compose_config
    from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS

    config = compose_config(list(guards.OVERRIDES), config_root=args.science_root / "configs")
    teacher = config["teacher"]
    require(
        config["seed"] == 42
        and config["trainer"]["max_completion_length"] == 128
        and teacher["model_revision"] == teacher["tokenizer_revision"] == TEACHER_REVISION
        and teacher["model_name_or_path"] == "Qwen/Qwen3-8B"
        and teacher["prompt_protocol"]["enable_thinking"] is False
        and config["teacher_readiness"] == THRESHOLDS,
        "frozen teacher capability protocol differs",
    )
    selected_rows = population()
    if args.validate_only:
        print(json.dumps({"validated": True, "exploratory": True, "accepted_science": False}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "task": TASK,
        "artifact_kind": "teacher_capability_diagnostic",
        "job_id": args.job_id,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "passed": False,
        "exit_code": 1,
        "exploratory": True,
        "accepted_science": False,
        "full_teacher_ready": False,
        "readiness": False,
        "readiness_artifact_produced": False,
        "g0_passed": False,
        "training_started": False,
        "execution_class_certified": False,
        "resumable": False,
        "metrics_passed": False,
        "population_sha256": DATASET_SHA256,
        "response_format_instructions_sha256": hashlib.sha256(
            RESPONSE_FORMAT_INSTRUCTIONS.encode()
        ).hexdigest(),
        "allocation": allocation,
        "runtime": runtime,
        "environment": environment,
        "scope": "fixed-first-128-frozen-validation; one greedy128-token generation; unchanged prefix gates",
        "passed_means": "diagnostic completed and preserved; not scientific readiness",
        "generation_score_semantics": (
            "Unchanged generator transition logprobs: log_softmax of returned generation scores "
            "at selected tokens. Greedy argmax selection is deterministic; these are not "
            "probabilities of a stochastic behavior policy."
        ),
        "prefix_score_semantics": "Original readiness helper's unfiltered full-vocabulary log_softmax",
        "generation": {
            "seed_base": 42,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "prefix_top_k": 128,
        },
    }
    started = time.monotonic()
    try:
        report["gpu"] = guards.gpu_identity()
        from posttrain_circuits.artifacts.hashing import sha256_value
        from posttrain_circuits.cli.evaluate_teacher_readiness import _prefix_scores
        from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
        from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
        from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator
        from posttrain_circuits.learning.teacher.evaluation import TeacherReadinessThresholds
        from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
        from posttrain_circuits.models.prompt_protocol import format_model_prompt

        examples = [deserialize_example(row) for row in selected_rows]
        report["ordered_prompt_ids"] = [example.example_id for example in examples]
        report["ordered_examples_sha256"] = sha256_value([asdict(example) for example in examples])
        report["prompt_envelope"] = guards.validate_prompts(config, examples)
        loaded = load_model_and_tokenizer(teacher, for_training=False)
        require(
            loaded.resolved_model_commit == loaded.resolved_tokenizer_commit == TEACHER_REVISION,
            "loaded teacher/tokenizer revision differs",
        )
        model = move_model_to_local_cuda(loaded.model)
        report["teacher"] = {
            "model_id": loaded.model_id,
            "model_revision": loaded.resolved_model_commit,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.resolved_tokenizer_commit,
            "tokenizer_hash": loaded.tokenizer_hash,
        }
        task = ProofGraphTask()
        generator = HfTeacherCandidateGenerator(
            model, loaded.tokenizer, max_new_tokens=128, model_config=teacher
        )
        generated = {}
        with (args.output_dir / "attempts.jsonl").open("x") as stream:
            for index, example in enumerate(examples):
                raw = task.render(example)
                formatted = format_model_prompt(raw, loaded.tokenizer, teacher)
                inputs = loaded.tokenizer.encode(formatted.model_facing_prompt, add_special_tokens=False)
                require(0 < len(inputs) <= 1246, "validation prompt exceeds unchanged prefix bound")
                output = generator(
                    example=example,
                    candidate_index=0,
                    actual_sampling_seed=42 + index,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                )
                output.validate()
                require(
                    output.logprob_status == "available"
                    and output.response_ids is not None
                    and len(output.response_ids) <= 128,
                    "diagnostic requires actual bounded response IDs and generator transition logprobs",
                )
                verification = task.verify(example, task.parse_response(output.response_text))
                record = {
                    "prompt_id": example.example_id,
                    "prompt_identity_sha256": sha256_value(asdict(example)),
                    "candidate_index": 0,
                    "actual_sampling_seed": 42 + index,
                    "raw_prompt_text": raw,
                    "model_facing_prompt_text": formatted.model_facing_prompt,
                    "input_ids": inputs,
                    **asdict(output),
                    "verification_trace": asdict(verification),
                    "accepted": verification.reward == 1.0,
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                generated[example.example_id] = output.response_text
                print(
                    json.dumps(
                        {
                            "stage": "generation",
                            "completed": index + 1,
                            "total": POPULATION,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            os.fsync(stream.fileno())
        semantic, specs, tokenized, skipped = prefix_probes(examples, task, loaded, teacher)
        guards.publish(args.output_dir / "semantic-prefix-manifest.json", semantic)
        guards.publish(args.output_dir / "tokenized-prefix-manifest.json", tokenized)
        report["prefix_probe_hash"] = tokenized["sha256"]
        report["alignment_skips"] = skipped
        selected_specs = [
            spec for spec in specs if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
        ]
        require(bool(selected_specs), "no required prefix stages are available")
        prefix_scores = []
        with (args.output_dir / "prefix-scores.jsonl").open("x") as stream:
            for index, spec in enumerate(selected_specs):
                scores = _prefix_scores(model, spec, top_k=128)
                require(len(scores) == 2, "prefix scoring did not preserve clean and corrupted sides")
                for score in scores:
                    stream.write(json.dumps(asdict(score), sort_keys=True, allow_nan=False) + "\n")
                    prefix_scores.append(score)
                stream.flush()
                print(
                    json.dumps(
                        {
                            "stage": "prefix_scoring",
                            "completed": index + 1,
                            "total": len(selected_specs),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            os.fsync(stream.fileno())
        report.update(
            capability_metrics(examples, generated, prefix_scores, TeacherReadinessThresholds(**THRESHOLDS))
        )
        report["prefix_score_count"] = len(prefix_scores)
        report["usage"] = guards.final_usage()
        report.update(passed=True, exit_code=0, elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        guards.publish(args.output_dir / "teacher-capability-probe.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
