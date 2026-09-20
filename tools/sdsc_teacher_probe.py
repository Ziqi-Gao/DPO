#!/usr/bin/env python3
"""One bounded, exploratory prompt comparison; never a formal teacher store.

Both arms call the unchanged production generator/verifier. The historical arm
uses its exact old instruction bytes; the candidate uses this snapshot's
renderer. Prompt selection is fixed before generation, independent of outcomes.
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
from collections import Counter
from dataclasses import asdict
from pathlib import Path

TASK = "qwen3-v2-teacher-prompt-probe"
DATASET = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/"
    "20260918T021516Z-82d6fc51200a-2cf5f753/f98dca38ea9d4d9bbe4f3917e151a796/"
    "artifacts/dataset/train/examples.jsonl"
)
DATASET_SHA256 = "377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b"
BASELINE_INSTRUCTIONS = (
    "Prove QUERY (1) or its negation (0); stop there. "
    "Only <proof>...</proof><answer>1 or 0</answer>. "
    "Line: step: rule(citations) -> actual consequent. "
    "Number steps S01,S02,...; use actual rule IDs and comma-separated "
    "fact/earlier-step IDs matching every antecedent. No unrelated steps or prose."
)
BASELINE_PROTOCOL = "qwen3-v2-g0-candidate-e-seed-42-prompt-v4"
POPULATION = 256
PROBE_PROMPTS = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population(path=DATASET, expected_sha256=DATASET_SHA256):
    """Read and hash the existing train split once; retain only 256 small rows."""
    require(not any(p.is_symlink() for p in (path, *path.parents)), "dataset has a symlink")
    checksum = hashlib.sha256()
    rows = []
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 1024**3, "invalid train split file")
        index = 0
        total = 0
        while raw := stream.readline(128 * 1024 + 1):
            total += len(raw)
            require(len(raw) <= 128 * 1024 and total <= 1024**3, "example exceeds bounded record size")
            checksum.update(raw)
            if index < POPULATION:
                rows.append(json.loads(raw))
            index += 1
        after = os.fstat(stream.fileno())
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        and checksum.hexdigest() == expected_sha256,
        "persistent train split differs from the failed job's verified publication",
    )
    require(len(rows) == POPULATION, "probe requires the original ordered 256-prompt population")
    require(len({row["example_id"] for row in rows}) == POPULATION, "duplicate example identity")
    return rows


def summary(records):
    accepted = [record for record in records if record["accepted"]]
    return {
        "attempt_count": len(records),
        "accepted_count": len(accepted),
        "prompts_with_success": len({record["prompt_id"] for record in accepted}),
        "total_prompts": len({record["prompt_id"] for record in records}),
        "errors": dict(
            Counter(record["verification_trace"]["error_code"] or "accepted" for record in records)
        ),
        "finish_reasons": dict(Counter(record["finish_reason"] for record in records)),
        "literal_TRUE_X": sum("TRUE X" in record["response_text"] for record in records),
        "exact_schema_line": sum(
            "S01: R01(F01,F02) -> TRUE X" in record["response_text"] for record in records
        ),
    }


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
    guards.require(guards.SHA256.fullmatch(args.code_sha256 or ""), "invalid snapshot identity")
    allocation = guards.allocation(args)
    runtime = guards.runtime_identity()
    environment = guards.local_environment(args)
    # The wrapper verifies every snapshot byte before this process starts. No
    # accepted protocol or committed-clean scientific lineage is claimed here.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.science_root / "src"))
    from posttrain_circuits.core.config import compose_config

    config = compose_config(list(guards.OVERRIDES), config_root=args.science_root / "configs")
    require(
        config["teacher"]["generation_seed"] == 31415
        and config["state_source"]["num_candidates"] == 8
        and config["state_source"]["max_prompt_tokens"] == 1246
        and config["state_source"]["max_new_tokens"] == 256,
        "probe sampling envelope differs",
    )
    require(DATASET.is_file(), "verified failed-job population is unavailable")
    if args.validate_only:
        print(json.dumps({"validated": True, "exploratory": True, "accepted_science": False}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "task": TASK,
        "job_id": args.job_id,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "passed": False,
        "exit_code": 1,
        "exploratory": True,
        "accepted_science": False,
        "full_teacher_ready": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "resumable": False,
        "sampling_request_seed": 31415,
        "population_sha256": DATASET_SHA256,
        "allocation": allocation,
        "runtime": runtime,
        "environment": environment,
        "scope": "fixed-first-32-training-prompts; baseline candidate 0; candidate indices 0-7",
        "baseline_protocol": BASELINE_PROTOCOL,
        "passed_means": "diagnostic completed and preserved; not scientific readiness",
    }
    try:
        report["gpu"] = guards.gpu_identity()
        from posttrain_circuits.artifacts.hashing import sha256_value
        from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
        from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS
        from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
        from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator
        from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed
        from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
        from posttrain_circuits.models.prompt_protocol import format_model_prompt

        examples = [deserialize_example(row) for row in population()]
        report["prompt_envelope"] = guards.validate_prompts(config, examples)
        selected = examples[:PROBE_PROMPTS]
        report["ordered_prompt_ids"] = [example.example_id for example in selected]
        report["candidate_instructions_sha256"] = hashlib.sha256(
            RESPONSE_FORMAT_INSTRUCTIONS.encode()
        ).hexdigest()
        report["baseline_instructions_sha256"] = hashlib.sha256(BASELINE_INSTRUCTIONS.encode()).hexdigest()
        teacher = config["teacher"]
        loaded = load_model_and_tokenizer(teacher, for_training=False)
        model = move_model_to_local_cuda(loaded.model)
        require(loaded.resolved_model_commit == teacher["model_revision"], "teacher revision differs")
        task = ProofGraphTask()

        class BaselineTask:
            def render(self, example):
                current = task.render(example)
                require(current.endswith(RESPONSE_FORMAT_INSTRUCTIONS), "unexpected renderer boundary")
                return current[: -len(RESPONSE_FORMAT_INSTRUCTIONS)] + BASELINE_INSTRUCTIONS

        records = {"baseline": [], "candidate": []}
        generator = HfTeacherCandidateGenerator(
            model, loaded.tokenizer, max_new_tokens=256, model_config=teacher
        )
        started = time.monotonic()
        with (args.output_dir / "attempts.jsonl").open("x") as stream:
            for variant, count, renderer in (("baseline", 1, BaselineTask()), ("candidate", 8, task)):
                generator.task = renderer
                for example in selected:
                    prompt = renderer.render(example)
                    formatted = format_model_prompt(prompt, loaded.tokenizer, teacher)
                    input_ids = loaded.tokenizer.encode(
                        formatted.model_facing_prompt, add_special_tokens=False
                    )
                    require(len(input_ids) <= 1246, "probe prefix exceeds unchanged bound")
                    identity = sha256_value(asdict(example))
                    for candidate_index in range(count):
                        seed = teacher_candidate_seed(31415, identity, candidate_index)
                        output = generator(
                            example=example,
                            candidate_index=candidate_index,
                            actual_sampling_seed=seed,
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20,
                            min_p=0.0,
                        )
                        verification = task.verify(example, task.parse_response(output.response_text))
                        record = {
                            "variant": variant,
                            "prompt_id": example.example_id,
                            "prompt_identity_sha256": identity,
                            "candidate_index": candidate_index,
                            "actual_sampling_seed": seed,
                            "raw_prompt_text": prompt,
                            "model_facing_prompt_text": formatted.model_facing_prompt,
                            "input_ids": input_ids,
                            **asdict(output),
                            "verification_trace": asdict(verification),
                            "accepted": verification.reward == 1.0,
                        }
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        records[variant].append(record)
                    print(
                        json.dumps(
                            {
                                "variant": variant,
                                "prompts_done": len(records[variant]) // count,
                                "accepted_count": sum(item["accepted"] for item in records[variant]),
                                "elapsed_seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            os.fsync(stream.fileno())
        require(
            len(records["baseline"]) == 32 and len(records["candidate"]) == 256,
            "incomplete diagnostic population",
        )
        report["arms"] = {name: summary(values) for name, values in records.items()}
        report["paired_candidate_zero"] = summary(
            [row for row in records["candidate"] if row["candidate_index"] == 0]
        )
        report["usage"] = guards.final_usage()
        report.update(passed=True, exit_code=0, elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        guards.publish(args.output_dir / "teacher-prompt-probe.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
