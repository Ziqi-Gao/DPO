#!/usr/bin/env python3
"""One bounded, exploratory prompt comparison; never a formal teacher store.

Both arms call the unchanged production generator/verifier. The historical arm
uses its exact old instruction bytes; the candidate uses this snapshot's
renderer. Prompt selection is fixed before generation, independent of outcomes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import sys
import tempfile
import time
import traceback
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
    "Output only:\n<proof>\nSnn: Rnn(citations) -> consequent\n</proof>\n"
    "<answer>0 or 1</answer>\nReplace placeholders. Number S01,S02,...; "
    "attach ( to actual rule ID. Cite comma-separated fact/earlier-step IDs "
    "matching all antecedents, never symbols. Use rule consequent. "
    "Stop at QUERY (1) or its negation (0)."
)
BASELINE_PROTOCOL = "qwen3-v2-g0-candidate-e-seed-42-prompt-v5"
CANDIDATE_PROTOCOL = "qwen3-v2-g0-candidate-e-seed-42-prompt-v7"
POPULATION = 256
PROBE_PROMPTS = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProbeInterrupted(SystemExit):
    """A catchable termination that still exits nonzero after publication."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(128 + signum)


def progress_json(path, value):
    """Replace only this invocation's small local progress snapshot atomically."""
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only mkstemp's owned path is removed. A TERM between fsync and
        # replace must not leave a fixed-name file that blocks failed(). A
        # cleanup error must also never replace the original interruption.
        active_error = sys.exc_info()[0] is not None
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not active_error:
                raise


class ProbeObservation:
    """Diagnostic I/O only; never changes generation, RNG, or proof acceptance."""

    def __init__(self, directory, report):
        self.directory = directory
        self.identity = {key: report[key] for key in ("task", "job_id", "run_id", "code_sha256")}
        self.started = self.stage_started = time.monotonic()
        self.stage = "worker_start"
        self.stage_started_at = dt.datetime.now(dt.UTC).isoformat()
        self.context = {}
        self.completed = {"baseline": 0, "candidate": 0}
        self.committed_bytes = 0
        self.interrupted_signal = None
        self.timeline = (directory / "progress.jsonl").open("x", encoding="utf-8")
        self.stacks = (directory / "worker-stacks.log").open("x", encoding="utf-8", buffering=1)
        self.previous_handlers = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self.interrupt)
        # Python cleanup runs only when the interpreter regains control. Do not
        # walk native frames asynchronously: the periodic faulthandler timer
        # reproduced a SIGSEGV during CPU Torch forward. A native hang followed
        # by SIGKILL leaves only already fsynced progress and ledger evidence.
        self.phase("worker_start")

    def interrupt(self, signum, _frame):
        if self.interrupted_signal is None:
            self.interrupted_signal = signum
            raise ProbeInterrupted(signum)

    def snapshot(self):
        return {
            **self.identity, "artifact_kind": "teacher_probe_progress", "not_a_completion_report": True,
            "passed": False, "accepted_science": False, "full_teacher_ready": False,
            "g0_passed": False, "training_started": False,
            "stage": self.stage, "stage_started_at": self.stage_started_at,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "stage_elapsed_seconds": time.monotonic() - self.stage_started,
            "completed_attempt_counts": dict(self.completed), "ledger_committed_bytes": self.committed_bytes,
            "interrupted_signal": self.interrupted_signal, **self.context,
        }

    def phase(self, name, **context):
        previous_stage, previous_elapsed = self.stage, time.monotonic() - self.stage_started
        self.stage = name
        self.stage_started = time.monotonic()
        self.stage_started_at = dt.datetime.now(dt.UTC).isoformat()
        self.context = context
        value = self.snapshot()
        value.update(previous_stage=previous_stage, previous_stage_elapsed_seconds=previous_elapsed)
        self.timeline.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self.timeline.flush()
        os.fsync(self.timeline.fileno())
        progress_json(self.directory / "progress.json", value)
        print(json.dumps({"event": "probe_phase", **value}, sort_keys=True, allow_nan=False), flush=True)

    def committed(self, variant, stream):
        # The ledger is already flushed; fsync before recording a completed
        # candidate so partial evidence never claims an uncommitted response.
        os.fsync(stream.fileno())
        self.completed[variant] += 1
        self.committed_bytes = os.fstat(stream.fileno()).st_size
        self.phase("candidate_committed", **self.context)

    def failed(self, error):
        traceback.print_exception(error, file=self.stacks)
        self.stacks.flush()
        os.fsync(self.stacks.fileno())
        failure_stage = self.stage
        self.phase("interrupted" if self.interrupted_signal is not None else "failed",
                   failure_stage=failure_stage, error=type(error).__name__ + ": " + str(error))

    def close(self):
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)
        self.stacks.flush()
        os.fsync(self.stacks.fileno())
        self.stacks.close()
        self.timeline.close()


def probe_config(args, guards):
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
    return config


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


def output_contract_diagnostic(parsed):
    """Observe TRUE spelling without repairing text or changing proof validity.

    Bare positive literals remain legal to the original parser/verifier. These
    counts only measure alignment with the canonical prefix-target rendering;
    no count here is an acceptance or readiness criterion.
    """
    result = {
        "parse_valid": parsed.parse_valid,
        "raw_TRUE_step_lines": len(
            re.findall(r"(?:^|\n)\s*S\d+:\s*R\d+\([^)]*\)\s*->\s*TRUE ", parsed.raw_text)
        ),
        "parsed_positive_conclusions": None,
        "parsed_positive_conclusions_with_TRUE": None,
        "parsed_positive_conclusions_without_TRUE": None,
        "all_positive_conclusions_use_TRUE": None,
    }
    if parsed.parse_valid:
        # The anchored production parser permits closing-tag text inside a
        # literal; its outer proof ends at the last closing tag before answer.
        # Preserve such rejected proof bytes instead of splitting at an inner
        # tag and turning an observational count into an execution failure.
        proof = parsed.raw_text.partition("<proof>")[2].rpartition("</proof>")[0].strip()
        lines = proof.splitlines() if proof else []
        positives = [
            # Citation text can contain an arrow under the original syntax;
            # only the arrow after the rule call starts the conclusion.
            line.partition(")")[2].partition("->")[2].strip()
            for step, line in zip(parsed.steps, lines, strict=True)
            if not step.conclusion.negated
        ]
        marked = sum(literal.startswith("TRUE ") for literal in positives)
        result.update(
            parsed_positive_conclusions=len(positives),
            parsed_positive_conclusions_with_TRUE=marked,
            parsed_positive_conclusions_without_TRUE=len(positives) - marked,
            all_positive_conclusions_use_TRUE=(marked == len(positives)) if positives else None,
        )
    return result


def summary(records):
    accepted = [record for record in records if record["accepted"]]
    diagnostics = [record["output_contract_diagnostic"] for record in records]
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
        "output_contract_diagnostic": {
            "parsed_outputs": sum(item["parse_valid"] for item in diagnostics),
            **{
                name: sum(item[name] or 0 for item in diagnostics)
                for name in (
                    "raw_TRUE_step_lines", "parsed_positive_conclusions",
                    "parsed_positive_conclusions_with_TRUE", "parsed_positive_conclusions_without_TRUE",
                )
            },
            "outputs_with_all_positive_conclusions_using_TRUE": sum(
                item["all_positive_conclusions_use_TRUE"] is True for item in diagnostics
            ),
            "outputs_with_unmarked_positive_conclusions": sum(
                item["all_positive_conclusions_use_TRUE"] is False for item in diagnostics
            ),
        },
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
    if args.validate_only:
        probe_config(args, guards)
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
        "training_started": False,
        "readiness_artifact_produced": False,
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
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "candidate_protocol_review": "proposed",
        "output_contract_diagnostic_means": "rendering observation only; not proof validity or readiness",
        "passed_means": "diagnostic completed and preserved; not scientific readiness",
    }
    observation = None
    try:
        observation = ProbeObservation(args.output_dir, report)
        observation.phase("config_composition")
        config = probe_config(args, guards)
        observation.phase("gpu_identity")
        report["gpu"] = guards.gpu_identity()
        observation.phase("scientific_imports")
        from posttrain_circuits.artifacts.hashing import sha256_value
        from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
        from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS
        from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
        from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator
        from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed
        from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
        from posttrain_circuits.models.prompt_protocol import format_model_prompt

        observation.phase("dataset_read_and_hash")
        examples = [deserialize_example(row) for row in population()]
        observation.phase("prompt_validation")
        report["prompt_envelope"] = guards.validate_prompts(config, examples)
        selected = examples[:PROBE_PROMPTS]
        report["ordered_prompt_ids"] = [example.example_id for example in selected]
        report["candidate_instructions_sha256"] = hashlib.sha256(
            RESPONSE_FORMAT_INSTRUCTIONS.encode()
        ).hexdigest()
        report["baseline_instructions_sha256"] = hashlib.sha256(BASELINE_INSTRUCTIONS.encode()).hexdigest()
        teacher = config["teacher"]
        observation.phase("model_load")
        loaded = load_model_and_tokenizer(teacher, for_training=False)
        observation.phase("model_device_transfer")
        model = move_model_to_local_cuda(loaded.model)
        require(loaded.resolved_model_commit == teacher["model_revision"], "teacher revision differs")
        task = ProofGraphTask()

        class BaselineTask:
            def render(self, example):
                current = task.render(example)
                require(current.endswith(RESPONSE_FORMAT_INSTRUCTIONS), "unexpected renderer boundary")
                return current[: -len(RESPONSE_FORMAT_INSTRUCTIONS)] + BASELINE_INSTRUCTIONS

        records = {"baseline": [], "candidate": []}
        observation.phase("generator_setup")
        generator = HfTeacherCandidateGenerator(
            model, loaded.tokenizer, max_new_tokens=256, model_config=teacher
        )
        started = time.monotonic()
        with (args.output_dir / "attempts.jsonl").open("x") as stream:
            for variant, count, renderer in (("baseline", 1, BaselineTask()), ("candidate", 8, task)):
                generator.task = renderer
                for example in selected:
                    observation.phase("prompt_encoding", variant=variant, prompt_id=example.example_id)
                    prompt = renderer.render(example)
                    formatted = format_model_prompt(prompt, loaded.tokenizer, teacher)
                    input_ids = loaded.tokenizer.encode(
                        formatted.model_facing_prompt, add_special_tokens=False
                    )
                    require(len(input_ids) <= 1246, "probe prefix exceeds unchanged bound")
                    identity = sha256_value(asdict(example))
                    for candidate_index in range(count):
                        seed = teacher_candidate_seed(31415, identity, candidate_index)
                        observation.phase("candidate_generation", variant=variant,
                                          prompt_id=example.example_id, candidate_index=candidate_index)
                        output = generator(
                            example=example,
                            candidate_index=candidate_index,
                            actual_sampling_seed=seed,
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20,
                            min_p=0.0,
                        )
                        observation.phase("candidate_verification", variant=variant,
                                          prompt_id=example.example_id, candidate_index=candidate_index)
                        parsed = task.parse_response(output.response_text)
                        verification = task.verify(example, parsed)
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
                            "output_contract_diagnostic": output_contract_diagnostic(parsed),
                        }
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        observation.committed(variant, stream)
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
        observation.phase("final_usage")
        report["usage"] = guards.final_usage()
        report.update(passed=True, exit_code=0, elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(passed=False, exit_code=1)
        report["error"] = type(error).__name__ + ": " + str(error)
        if observation is not None:
            report["failure_stage"] = observation.stage
            observation.failed(error)
            report["partial_results"] = {
                "not_a_quality_estimate": True,
                "completed_attempt_counts": dict(observation.completed),
                "ledger_committed_bytes": observation.committed_bytes,
                "in_flight_response_is_not_a_completed_candidate": True,
            }
        if isinstance(error, ProbeInterrupted):
            report["exit_code"] = error.code
            report["interrupted_signal"] = signal.Signals(error.signum).name
        raise
    finally:
        try:
            if observation is not None:
                report["execution_diagnostics"] = observation.snapshot()
            guards.publish(args.output_dir / "teacher-prompt-probe.json", report)
        finally:
            if observation is not None:
                observation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
