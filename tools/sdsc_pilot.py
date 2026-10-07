#!/usr/bin/env python3
"""Fixed seed-42 pilot stages; no scheduler, submission, or arbitrary commands.

Every invocation runs inside the same private, node-local container namespace.
The caller restores verified immutable inputs and persists outputs. Training
uses a real eight-element Slurm array with concurrency one, never fabricated
array accounting. This worker preserves the original four-rank pilot science.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
from typing import NamedTuple

SCIENCE_HEAD = "0215c356355b29b5e2b407978a207db2156719e1"
CELLS = (
    "offline_hard",
    "online_hard",
    "offline_soft",
    "online_soft_opd",
    "offline_verified_replay",
    "online_verified_replay",
    "canonical_sft",
    "canonical_grpo",
)
PROBE_STAGES = (("process", "first_rule_selection"), ("final_answer", "final_answer"))
COHORTS = ("base_capable", "challenge")
STAGES = (
    "prepare",
    "teacher-shard",
    "pilot-inputs",
    "train-cell",
    "bind-training",
    "initial-circuits",
    "final-circuits",
    "local-fork",
    "resume",
    "dynamics",
    "finalize",
)
TERMINAL_STAGES = ("training", "initial_circuits", "final_circuits", "local_fork", "resume", "dynamics")
GPUS = {
    "prepare": 1,
    "teacher-shard": 1,
    "pilot-inputs": 1,
    "train-cell": 4,
    "bind-training": 1,
    "initial-circuits": 1,
    "final-circuits": 1,
    "local-fork": 1,
    "resume": 4,
    "dynamics": 1,
    "finalize": 1,
}
SHA = re.compile(r"[a-f0-9]{64}\Z")
JOB = re.compile(r"[1-9][0-9]*\Z")
LEGACY_CORE = frozenset(
    {
        "manifest_hashes",
        "model",
        "optimizer",
        "prompt_scheduler_by_rank",
        "rank_shard_hashes",
        "rng",
        "scaler",
        "scheduler",
        "state_source_by_rank",
        "token_budget",
        "trainer_state",
    }
)


class Step(NamedTuple):
    name: str
    module: str
    arguments: tuple[str, ...]
    distributed: bool = False
    outputs: tuple[str, ...] = ()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("pilot_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(value):
    return helper("sdsc_science_binding").real_path(value)


def file_hash(value):
    return helper("sdsc_science_binding").file_identity(value, limit=128 * 1024**3)["sha256"]


def read_json(value, expected=None):
    result, identity = helper("sdsc_science_binding").json_document(value)
    require(expected is None or identity["sha256"] == expected, "input JSON byte hash differs")
    return result


def publish(value, payload):
    helper("sdsc_teacher_prepare").publish(value, payload)


def self_hashed(value):
    require(
        isinstance(value, dict)
        and value.get("sha256") == sha(canonical({k: v for k, v in value.items() if k != "sha256"})),
        "scientific artifact self-hash differs",
    )
    return value


def quoted(name, value):
    return name + "=" + json.dumps(str(value))


def common_overrides(args, inputs, cell="canonical_sft"):
    """Storage locators change; original pilot scientific values remain fixed."""
    result = [
        "experiment=" + cell,
        "pilot=qwen3_v2_core",
        "seed=42",
        quoted("output_root", args.pilot_root),
        quoted("task.dataset_family_path", inputs["dataset"]),
        quoted("anti_shortcut.report_path", inputs["anti_shortcut"]),
        quoted("production_safety.probe_cohort_manifest", inputs["probes"]),
        quoted("production_safety.readiness_report", inputs["readiness"]),
        quoted("production_safety.initial_checkpoint_path", inputs["initial_checkpoint"]),
        "production_safety.initial_checkpoint_hash=" + inputs["initial_checkpoint_sha256"],
    ]
    if cell.startswith("offline_"):
        result.append(quoted("state_source.store_path", inputs["scored_bank"]))
    elif cell == "canonical_sft":
        result.append(quoted("state_source.store_path", inputs["teacher_demos"]))
    if cell == "canonical_grpo":
        result.append("trainer.batch_size=8")
    return result


def matrix(final=False):
    return tuple(
        dict(cell=cell, stage_label=label, probe_stage=stage, cohort=cohort)
        for cell in (CELLS if final else (None,))
        for label, stage in PROBE_STAGES
        for cohort in COHORTS
    )


def cell_root(args, cell):
    require(cell in CELLS, "unknown pilot method")
    return args.pilot_root / "runs" / cell / "seed-42"


def final_checkpoint(root):
    manifest = self_hashed(read_json(root / "manifest.json"))
    checkpoint = path(manifest["final_checkpoint_path"])
    require(root / "checkpoints" in checkpoint.parents, "checkpoint escapes its method directory")
    require(file_hash(checkpoint) == manifest["final_checkpoint_sha256"], "final checkpoint bytes differ")
    return checkpoint


def stage_plan(args, inputs, *, index=None, checkpoints=None):
    """An explicit fixed stage plan; accepts no caller-provided command vector."""
    require(args.stage in STAGES, "unknown pilot stage")
    base = common_overrides(args, inputs)
    root = args.pilot_root
    if args.stage == "prepare":
        return (
            Step(
                "prepare",
                "prepare_pilot",
                tuple(
                    [*base, "--g0", inputs["scientific_g0"], "--output", str(root / "pilot_manifest.json")]
                ),
                outputs=("pilot_manifest.json",),
            ),
        )
    if args.stage == "teacher-shard":
        require(type(index) is int and 0 <= index < 16, "teacher shard needs real array index 0..15")
        return ()  # Uses the original prompt-identity RNG domain function, below.
    if args.stage == "pilot-inputs":
        bank = root / "inputs/rollout_bank"
        return (
            Step(
                "build-pilot-bank",
                "build_rollout_bank",
                tuple(
                    [
                        *common_overrides(args, inputs, "offline_hard"),
                        "--output",
                        str(bank),
                        "--confirm-production",
                    ]
                ),
                outputs=(str(bank / "manifest.json"),),
            ),
            Step(
                "score-pilot-bank",
                "score_teacher",
                tuple(
                    [
                        *common_overrides(args, inputs, "offline_soft"),
                        "--bank",
                        str(bank),
                        "--output",
                        inputs["scored_bank"],
                        "--confirm-production",
                    ]
                ),
                outputs=(str(Path(inputs["scored_bank"]) / "manifest.json"),),
            ),
        )
    if args.stage == "train-cell":
        require(type(index) is int and 0 <= index < len(CELLS), "training needs a real array index 0..7")
        cell = CELLS[index]
        return (
            Step(
                "train-" + cell,
                "run_grpo" if cell == "canonical_grpo" else "train",
                tuple(
                    [
                        *common_overrides(args, inputs, cell),
                        "--output",
                        str(cell_root(args, cell)),
                        "--confirm-production",
                    ]
                ),
                True,
                (str(cell_root(args, cell) / "manifest.json"),),
            ),
        )
    if args.stage == "bind-training":
        return (
            Step(
                "bind-training",
                "finalize_pilot_training",
                tuple(
                    [
                        *base,
                        "--run-dir",
                        str(root),
                        "--terminal",
                        str(root / "terminal-training.txt"),
                        "--job-id",
                        args.training_job_id,
                        "--output",
                        str(root / "training_artifact_chain.json"),
                    ]
                ),
                outputs=("training_artifact_chain.json",),
            ),
        )
    if args.stage in {"initial-circuits", "final-circuits"}:
        final = args.stage == "final-circuits"
        steps = []
        for row in matrix(final):
            circuit_root = root / "circuits" / ("final" if final else "initial")
            if final:
                circuit_root /= row["cell"]
            circuit_root = circuit_root / row["stage_label"] / row["cohort"]
            checkpoint = checkpoints[row["cell"]] if final else inputs["initial_checkpoint"]
            shared = [
                *base,
                "--checkpoint",
                str(checkpoint),
                "--initial-checkpoint",
                inputs["initial_checkpoint"],
                "--probe-cohort-manifest",
                inputs["probes"],
                "--cohort",
                row["cohort"],
            ]
            slot = "-".join(str(row[k]) for k in ("cell", "stage_label", "cohort"))
            steps.append(
                Step(
                    "discover-" + slot,
                    "discover_circuit",
                    tuple(
                        [
                            *shared,
                            "--stage",
                            row["probe_stage"],
                            "--output",
                            str(circuit_root / "circuit.json"),
                            "--confirm-production",
                        ]
                    ),
                    outputs=(str(circuit_root / "circuit.json"),),
                )
            )
            extra = (
                [
                    "--transfer-source-circuit",
                    str(root / "circuits/initial" / row["stage_label"] / row["cohort"] / "circuit.json"),
                ]
                if final
                else []
            )
            steps.append(
                Step(
                    "evaluate-" + slot,
                    "evaluate_circuit",
                    tuple(
                        [
                            *shared,
                            "--circuit-artifact",
                            str(circuit_root / "circuit.json"),
                            *extra,
                            "--output",
                            str(circuit_root / "exact_patching.json"),
                            "--confirm-production",
                        ]
                    ),
                    outputs=(str(circuit_root / "exact_patching.json"),),
                )
            )
        return tuple(steps)
    if args.stage == "local-fork":
        source = cell_root(args, "canonical_sft")
        fork = root / "local_fork"
        return (
            Step(
                "fork-inputs",
                "build_local_fork_inputs",
                tuple([*base, "--trajectory-store", inputs["scored_bank"], "--output", str(fork / "inputs")]),
                outputs=("local_fork/inputs/prompts.json", "local_fork/inputs/probe_set.json"),
            ),
            Step(
                "fork-bundle",
                "create_fork_bundle",
                tuple(
                    [
                        *common_overrides(args, inputs, "local_fork"),
                        "--checkpoint",
                        str(checkpoints["canonical_sft"]),
                        "--source-manifest",
                        str(source / "manifest.json"),
                        "--prompts",
                        str(fork / "inputs/prompts.json"),
                        "--trajectory-store",
                        inputs["scored_bank"],
                        "--probe-set",
                        str(fork / "inputs/probe_set.json"),
                        "--output",
                        str(fork / "bundle.pt"),
                        "--confirm-production",
                    ]
                ),
                outputs=("local_fork/bundle.pt",),
            ),
            Step(
                "local-fork",
                "run_local_fork",
                (
                    "--bundle",
                    str(fork / "bundle.pt"),
                    "--output",
                    str(fork / "results.json"),
                    "--horizons",
                    "1",
                    "5",
                    "20",
                ),
                outputs=("local_fork/results.json",),
            ),
        )
    if args.stage == "resume":
        source = cell_root(args, "canonical_sft") / "checkpoints/step-00000020.pt"
        return tuple(
            Step(
                "resume-" + replica,
                "train",
                tuple(
                    [
                        *base,
                        "--resume",
                        str(source),
                        "--output",
                        str(root / ("resume-" + replica)),
                        "--confirm-production",
                    ]
                ),
                True,
                ("resume-" + replica + "/manifest.json",),
            )
            for replica in ("a", "b")
        )
    if args.stage == "dynamics":
        steps = []
        for row in matrix(True):
            initial = root / "circuits/initial" / row["stage_label"] / row["cohort"]
            final = root / "circuits/final" / row["cell"] / row["stage_label"] / row["cohort"]
            output = root / "dynamics" / row["cell"] / row["stage_label"] / (row["cohort"] + ".json")
            steps.append(
                Step(
                    "dynamics-" + "-".join((row["cell"], row["stage_label"], row["cohort"])),
                    "analyze_circuit_dynamics",
                    tuple(
                        [
                            *base,
                            "--circuits",
                            str(initial / "circuit.json"),
                            str(final / "circuit.json"),
                            "--evaluations",
                            str(initial / "exact_patching.json"),
                            str(final / "exact_patching.json"),
                            "--transfers",
                            str(final / "exact_patching.json"),
                            "--activation-threshold",
                            "0.0",
                            "--output",
                            str(output),
                        ]
                    ),
                    outputs=(str(output),),
                )
            )
        return tuple(steps)
    return ()  # The explicit pilot adapter below preserves the shared validators.


def command(args, step):
    # Preserve the explicitly admitted interpreter spelling: resolving a venv
    # symlink can silently select the host Python's package environment.
    python = str(args.python)
    if step.distributed:
        return [
            python,
            "-I",
            "-B",
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(args.science_root / "configs/accelerate/fsdp.yaml"),
            "--num_processes",
            "4",
            "--num_machines",
            "1",
            "--main_process_port",
            "0",
            "-m",
            "posttrain_circuits.cli." + step.module,
            *step.arguments,
        ]
    return [python, "-B", "-m", "posttrain_circuits.cli." + step.module, *step.arguments]


def terminal_records(raw, job_id, *, training=False, count=None):
    """Accept actual sacct records; never synthesize array-task completion."""
    require(isinstance(job_id, str) and JOB.fullmatch(job_id), "invalid Slurm job ID")
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split("|")]
        require(len(fields) >= 3, "malformed Slurm accounting record")
        identifier, state, code = fields[:3]
        match = re.fullmatch(re.escape(job_id) + r"(?:_([0-9]+))?(?:\.[A-Za-z0-9_-]+)?", identifier)
        require(match is not None, "unrelated accounting record")
        array_count = 8 if training and count is None else count
        require(
            match[1] is None or (array_count is not None and 0 <= int(match[1]) < array_count),
            "unexpected array task accounting record",
        )
        require(state == "COMPLETED" and code == "0:0", "Slurm allocation/step did not complete successfully")
        rows.append(fields)
    require(rows and len({row[0] for row in rows}) == len(rows), "empty/duplicate accounting records")
    if training or count is not None:
        count = 8 if count is None else count
        allocations = {row[0] for row in rows if "." not in row[0] and row[0] != job_id}
        require(
            allocations == {job_id + "_" + str(i) for i in range(count)},
            "real array must contain exactly its registered tasks",
        )
    else:
        require(sum(row[0] == job_id for row in rows) == 1, "missing allocation accounting row")
    return rows


def stage_evidence(args, admission, name, *, count=None):
    record = admission["stages"][name]
    job = record["job_id"]
    status = record["status"]
    require(
        status.get("success") is True
        and status.get("state") == "COMPLETED"
        and str(status.get("job_id")) == job,
        "prior stage lacks verified Slurm success",
    )
    require(
        status["queue"]["returncode"] == 0
        and not status["queue"]["stdout"].strip()
        and status["accounting"]["returncode"] == 0
        and status["result"].get("verified") is True,
        "prior stage accounting/publication is not verified",
    )
    terminal = path(record["terminal_path"])
    require(file_hash(terminal) == record["terminal_sha256"], "prior terminal bytes differ")
    raw = terminal.read_text()
    require(raw == status["accounting"]["stdout"], "terminal file differs from the actual accounting reply")
    terminal_records(raw, job, training=name == "training", count=count)
    stage_publications(args, admission, record, name, raw, count=count)
    if name not in TERMINAL_STAGES:
        return record
    destination = args.pilot_root / ("terminal-" + name + ".txt")
    if destination.exists():
        require(destination.read_bytes() == terminal.read_bytes(), "persisted terminal evidence changed")
    else:
        with destination.open("xb") as stream:
            stream.write(terminal.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    return record


def stage_publications(args, admission, record, name, raw, *, count=None):
    receipt = read_json(record["submission_receipt_path"], record["submission_receipt_sha256"])
    expected_stage = {
        "teacher_shards": "teacher-shard",
        "pilot_inputs": "pilot-inputs",
        "training": "train-cell",
        "initial_circuits": "initial-circuits",
        "final_circuits": "final-circuits",
        "local_fork": "local-fork",
    }.get(name, name)
    require(
        str(receipt["job_id"]) == record["job_id"]
        and receipt["flow_id"] == admission["flow_id"]
        and receipt["stage"] == expected_stage,
        "upstream submission receipt identity differs",
    )
    count = 8 if name == "training" else count
    expected_jobs = (
        {record["job_id"] + "_" + str(index) for index in range(count)} if count else {record["job_id"]}
    )
    publications = {
        read_json(item["path"], item["sha256"])["job_id"]: (item, read_json(item["path"], item["sha256"]))
        for item in record["publications"]
    }
    reports = {
        read_json(item["path"], item["sha256"])["job_id"]: (item, read_json(item["path"], item["sha256"]))
        for item in record["reports"]
    }
    require(
        set(publications) == set(reports) == expected_jobs
        and len(record["publications"]) == len(expected_jobs)
        and len(record["reports"]) == len(expected_jobs),
        "upstream publication/report task inventory differs",
    )
    accounting = {parts[0]: parts for parts in (line.split("|") for line in raw.splitlines() if line)}
    for job in sorted(expected_jobs):
        _, publication = publications[job]
        report_identity, report = reports[job]
        for key, expected in {
            "flow_id": admission["flow_id"],
            "stage": expected_stage,
            "job_id": job,
            "run_id": receipt["run_id"],
            "code_sha256": receipt["code_sha256"],
        }.items():
            require(
                publication.get(key) == report.get(key) == expected,
                "upstream publication identity differs: " + key,
            )
        require(
            publication.get("passed") is True
            and publication.get("persisted") is True
            and publication.get("persistent_read_back_verified") is True
            and publication.get("result_sha256") == report_identity["sha256"]
            and report.get("passed") is True,
            "upstream stage was not successfully published and read back",
        )
        require(
            len(accounting[job]) >= 6 and str(report["allocation_job_id"]) == accounting[job][5],
            "wrapper allocation ID differs from actual sacct JobIDRaw",
        )
        delta = publication.get("delta")
        require(isinstance(delta, dict) and delta, "upstream publication contains no artifact inventory")
        inner = report["scientific_report"]
        require(
            isinstance(inner.get("value"), dict) and inner["value"].get("passed") is True,
            "upstream scientific stage did not pass",
        )
        logical = "pilot/" + Path(inner["path"]).relative_to(args.pilot_root).as_posix()
        require(
            delta.get(logical, {}).get("sha256") == inner["sha256"],
            "inner scientific report is not published",
        )
        actual = path(inner["path"])
        require(read_json(actual, inner["sha256"]) == inner["value"], "inner scientific report bytes changed")
        require(
            str(inner["value"].get("job_id")) == str(report["allocation_job_id"])
            and inner["value"].get("run_id") == receipt["run_id"]
            and inner["value"].get("code_sha256") == receipt["code_sha256"]
            and (
                inner["value"].get("task") == "qwen3-v2-pilot-preflight"
                if name == "preflight4"
                else inner["value"].get("stage") == expected_stage
            ),
            "scientific stage identity differs",
        )


def science_api(args):
    teacher = helper("sdsc_teacher_prepare")
    provenance = teacher.validate_checkout(args)
    require(provenance["science_git_head"] == SCIENCE_HEAD, "pilot science HEAD is not the reviewed source")
    helper("sdsc_science_binding").verify_source_bytes(args.science_root, SCIENCE_HEAD)
    api = teacher.scientific_api(args.science_root)
    return api, provenance


def frozen_config(args, inputs, api, cell="canonical_sft"):
    config = api.compose(common_overrides(args, inputs, cell), config_root=args.science_root / "configs")
    require(
        config["seed"] == 42
        and config["task"]["num_examples"] == 4096
        and config["trainer"]["max_steps"] == 120
        and config["trainer"]["token_budget"] == 2000000
        and config["trainer"]["gradient_accumulation_steps"] == 4
        and config["trainer"]["batch_size"] == (8 if cell == "canonical_grpo" else 4)
        and config["pilot"]["full_parameter_training"] is True,
        "pilot configuration differs from the original registered science",
    )
    require(
        config["trainer"].get("batch_partition_protocol", "legacy")
        != "allocation_neutral_exact_global_batch_v1",
        "G0-only allocation-neutral protocol must not replace original pilot batching",
    )
    return config


def load_inputs(args):
    admission = read_json(args.inputs, args.inputs_sha256)
    require(
        admission.get("schema") == "quest-sdsc-pilot-inputs-v1" and admission.get("stage") == args.stage,
        "wrong pilot admission schema/stage",
    )
    require(
        admission.get("target") == {"run_id": args.run_id, "code_sha256": args.code_sha256},
        "admission target differs",
    )
    require(
        admission.get("pilot_root") == str(args.pilot_root) and admission.get("g0_root") == str(args.g0_root),
        "stable namespace changed",
    )
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", admission.get("flow_id", "")),
        "invalid pipeline identity",
    )
    namespace = Path("/opd-pipeline") / admission["flow_id"]
    require(
        args.work_dir == namespace
        and args.pilot_root == namespace / "qwen3-v2/pilot"
        and args.science_root == namespace / "science"
        and args.hf_home == namespace / "huggingface"
        and args.mib_repository == namespace / "mib",
        "private stable namespace layout differs",
    )
    for value in (
        args.work_dir,
        args.pilot_root,
        args.g0_root,
        args.science_root,
        args.hf_home,
        args.mib_repository,
    ):
        path(value)
    require(
        args.python.is_absolute()
        and ".." not in args.python.parts
        and args.python.resolve(strict=True).is_file()
        and os.access(args.python, os.X_OK),
        "explicit runtime interpreter is invalid",
    )
    preflight = admission["preflight4"]
    report = read_json(preflight["report_path"], preflight["report_sha256"])
    validate_preflight_report(report)
    status = preflight["status"]
    require(
        status.get("success") is True
        and status.get("state") == "COMPLETED"
        and status["result"].get("verified") is True
        and status["result"].get("result_sha256") == preflight["report_sha256"],
        "four-H100 preflight publication is unverified",
    )
    require(
        status["queue"]["returncode"] == 0
        and not status["queue"]["stdout"].strip()
        and status["accounting"]["returncode"] == 0,
        "four-H100 preflight is not terminal",
    )
    terminal_records(status["accounting"]["stdout"], str(report["job_id"]))
    require(str(status["job_id"]) == str(report["job_id"]), "four-H100 preflight job identity differs")
    stage = stage_evidence(args, admission, "preflight4")
    require(
        stage["publications"]
        == [
            {"path": preflight["publication_receipt_path"], "sha256": preflight["publication_receipt_sha256"]}
        ]
        and stage["reports"]
        == [{"path": preflight["wrapper_report_path"], "sha256": preflight["wrapper_report_sha256"]}],
        "preflight proof paths differ from registered stage",
    )
    wrapper = read_json(preflight["wrapper_report_path"], preflight["wrapper_report_sha256"])
    require(
        wrapper["scientific_report"]["path"] == preflight["report_path"]
        and wrapper["scientific_report"]["sha256"] == preflight["report_sha256"]
        and report["run_id"] == args.run_id
        and report["code_sha256"] == args.code_sha256,
        "four-H100 preflight runtime/release/report binding differs",
    )
    g0 = admission["g0"]
    report = read_json(args.g0_root / "g0.json", g0["report_sha256"])
    require(
        report.get("schema") == "quest-sdsc-g0-v1"
        and report.get("passed") is True
        and report.get("git_commit") == SCIENCE_HEAD,
        "a real complete SDSC G0 result is required",
    )
    inputs = {
        "scientific_g0": str(args.g0_root / "g0.json"),
        "dataset": str(args.g0_root / "dataset"),
        "teacher_demos": str(args.pilot_root / "inputs/teacher_demos"),
        "scored_bank": str(args.pilot_root / "inputs/teacher_scores"),
        "probes": str(args.g0_root / "probes/manifest.json"),
        "anti_shortcut": str(args.g0_root / "anti_shortcut.json"),
        "readiness": str(args.g0_root / "readiness/readiness.json"),
        "initial_checkpoint": str(args.g0_root / "initial_checkpoint.pt"),
        "initial_checkpoint_sha256": g0["replay"]["expected_initial_sha256"],
    }
    return admission, inputs


def validate_preflight_report(report):
    require(
        report.get("kind") == "sdsc_four_h100_pilot_preflight_v1"
        and report.get("task") == "qwen3-v2-pilot-preflight"
        and report.get("passed") is True
        and type(report.get("world_size")) is int
        and report["world_size"] == 4
        and type(report.get("exit_code")) is int
        and report["exit_code"] == 0,
        "pilot requires a real four-H100 preflight report",
    )
    for field in ("pilot_passed", "g0_passed", "execution_class_certified", "uses_blackwell_certificate"):
        require(report.get(field) is False, "preflight must not claim scientific/other-backend certification")
    ranks = report.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == 4, "preflight lacks four rank reports")
    for index, rank in enumerate(ranks):
        require(
            rank.get("passed") is True
            and rank.get("run_id") == report["run_id"]
            and rank.get("code_sha256") == report["code_sha256"]
            and rank.get("job_id") == report["job_id"],
            "preflight rank identity differs",
        )
        allocation = rank["allocation"]
        require(
            allocation["rank"] == allocation["local_rank"] == index
            and allocation["world_size"] == 4
            and allocation["job_id"] == report["job_id"]
            and allocation["threads_per_rank"] == 6,
            "preflight allocation/rank identity differs",
        )
        require(
            rank["local_global_slots"] == list(range(index, 64, 4))
            and rank["reserved_global_nonpadding_tokens"] == 98304
            and rank["full_state_optimizer_scheduler_rng_restore"] is True,
            "preflight batch/restore proof differs",
        )
        restore = rank["state_restore_sha256"]
        require(
            restore["all_exact"] is True
            and restore["every_category_perturbed"] is True
            and set(restore["saved"]) == {"model", "optimizer", "scheduler", "cpu_rng", "cuda_rng"}
            and restore["saved"] == restore["restored"]
            and set(restore["perturbed"]) == set(restore["saved"])
            and all(
                SHA.fullmatch(value) and value != restore["perturbed"][name]
                for name, value in restore["saved"].items()
            ),
            "preflight state perturb/restore proof differs",
        )
        require(
            rank["runtime"]["python"] == "3.12.13"
            and rank["runtime"]["packages"] == helper("sdsc_pilot_preflight").DEPENDENCIES,
            "preflight runtime differs from full pinned pilot environment",
        )
        gpu = rank["gpu"]
        require(
            "H100" in gpu["name"]
            and gpu["compute_capability"] == [9, 0]
            and gpu["logical_device"] == index
            and gpu["total_memory_bytes"] >= 75 * 1024**3
            and 0 < rank["peak_gpu_reserved_bytes"] < gpu["total_memory_bytes"],
            "preflight GPU evidence differs",
        )
        require(
            rank["nccl_all_reduce"] is True
            and rank["finite_gradients"] is True
            and rank["parameter_update_nonzero"] is True
            and SHA.fullmatch(rank["local_parameter_sha256_before"])
            and SHA.fullmatch(rank["local_parameter_sha256_after"])
            and rank["local_parameter_sha256_before"] != rank["local_parameter_sha256_after"],
            "preflight lacks finite real distributed optimizer update",
        )
        require(
            isinstance(rank["losses"], list)
            and rank["losses"]
            and all(type(value) in (int, float) and math.isfinite(value) for value in rank["losses"]),
            "preflight losses are missing or nonfinite",
        )
        fsdp = rank["fsdp"]
        require(
            fsdp["requested_fsdp_sharding_strategy"]
            == fsdp["effective_fsdp_sharding_strategy"]
            == "FULL_SHARD"
            and type(fsdp["fsdp_wrapper_count"]) is int
            and fsdp["fsdp_wrapper_count"] == 29,
            "preflight model FSDP wrapper tree differs",
        )
        checkpoint = rank["checkpoint"]
        records = checkpoint["files"]
        expected_names = {
            "model-full.pt",
            "optimizer-full.pt",
            *(f"rank-{number}-runtime.pt" for number in range(4)),
        }
        require(
            checkpoint["all_files_verified_before_restore"] is True
            and isinstance(records, list)
            and len(records) == 6
            and {entry["path"] for entry in records} == expected_names
            and all(
                type(entry["size"]) is int and entry["size"] > 0 and SHA.fullmatch(entry["sha256"])
                for entry in records
            ),
            "preflight full checkpoint inventory differs",
        )
        for field in ("cgroup_memory", "initial_cgroup_memory"):
            memory = rank[field]
            require(
                memory["passed"] is True
                and memory["limit_bytes"] == 192 * 1024**3
                and 0 < memory["current_bytes"] <= memory["peak_bytes"] <= memory["limit_bytes"]
                and memory["headroom_bytes"] == memory["limit_bytes"] - memory["peak_bytes"]
                and memory["minimum_headroom_bytes"] == max(32 * 1024**3, math.ceil(0.2 * 192 * 1024**3))
                and memory["headroom_bytes"] >= memory["minimum_headroom_bytes"],
                "preflight lacks actual 192GiB memory headroom",
            )
    require(
        len({row["allocation"]["cuda_visible_devices"] for row in ranks}) == 1,
        "preflight CUDA visibility differs between ranks",
    )


def replay_g0(args, admission):
    replay = admission["g0"]["replay"]
    argv = [
        str(args.python),
        "-I",
        "-B",
        str(Path(__file__).with_name("sdsc_finalize_g0.py")),
        "--science-root",
        str(args.science_root),
        "--workspace",
        str(args.g0_root),
        "--expected-science-head",
        SCIENCE_HEAD,
        "--replay",
        str(args.g0_root / "g0.json"),
    ]
    for key in (
        "runtime_config",
        "execution_evidence",
        "execution_evidence_sha256",
        "expected_protocol_sha256",
        "expected_initial_sha256",
        "expected_adapter_sha256",
    ):
        require(isinstance(replay.get(key), str) and replay[key], "missing G0 replay identity " + key)
        argv.extend(("--" + key.replace("_", "-"), replay[key]))
    result = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=1800,
        env=child_environment(args, 1),
        check=False,
    )
    require(
        result.returncode == 0, "full G0 artifact replay rejected pilot admission: " + result.stderr[-4000:]
    )
    return {
        "validated": True,
        "scientific_g0_sha256": admission["g0"]["report_sha256"],
        "replay_stdout_sha256": sha(result.stdout.encode()),
    }


def child_environment(args, gpu_count=None):
    gpu_count = GPUS[args.stage] if gpu_count is None else gpu_count
    cache = args.pilot_root / ".cache" / args.stage / os.environ.get("SLURM_ARRAY_TASK_ID", "single")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "SERVER_SCHEDULER_")) and key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        PYTHONPATH=str(args.science_root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(args.hf_home),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        MIB_REPOSITORY=str(args.mib_repository),
        MPLCONFIGDIR=str(cache / "matplotlib"),
        TORCH_HOME=str(cache / "torch"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        XDG_CACHE_HOME=str(cache),
        OMP_NUM_THREADS=str(24 // max(gpu_count, 1)),
        MKL_NUM_THREADS=str(24 // max(gpu_count, 1)),
    )
    return environment


def allocation(args):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(JOB.fullmatch(job), "real Slurm job required")
    require(
        os.environ.get("SLURM_CPUS_PER_TASK") == "24" and os.environ.get("SLURM_MEM_PER_NODE") == "196608",
        "pilot worker requires its fixed 24 CPU / 192 GiB allocation",
    )
    index = None
    if args.stage in {"train-cell", "teacher-shard"}:
        total = 8 if args.stage == "train-cell" else 16
        master = os.environ.get("SLURM_ARRAY_JOB_ID", "")
        raw = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        require(
            JOB.fullmatch(master)
            and raw.isdigit()
            and 0 <= int(raw) < total
            and os.environ.get("SLURM_ARRAY_TASK_COUNT") == str(total),
            "registered real Slurm array required",
        )
        index = int(raw)
    if GPUS[args.stage]:
        import torch

        require(torch.cuda.device_count() == GPUS[args.stage], "wrong visible GPU count for pilot stage")
        for device in range(torch.cuda.device_count()):
            info = torch.cuda.get_device_properties(device)
            require("H100" in info.name and info.total_memory >= 79 * 1024**3, "pilot requires H100 80GB")
    return {
        "job_id": job,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_index": index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": GPUS[args.stage],
        "cpus": 24,
        "memory_mib": 196608,
    }


def runtime(args):
    value = helper("sdsc_teacher_prepare").runtime_identity()
    lock = read_json(args.science_root / "deployments/qwen3_v2_g0/dependency-lock.json")
    required = {item["name"]: item["version"] for item in lock["packages"]}
    required["trl"] = "0.22.2"
    actual = {name: importlib.metadata.version(name) for name in required}
    require(
        actual == required and Path(sys.executable).resolve() == args.python.resolve(),
        "pilot runtime pins/executable differ",
    )
    value["packages"] = actual
    return value


def node_storage(args):
    for target in (args.work_dir, args.pilot_root, args.g0_root, args.hf_home):
        require(path(target).is_dir(), "missing staged directory")
        mount = json.loads(
            subprocess.run(
                ["findmnt", "--json", "--target", str(target), "--output", "TARGET,SOURCE,FSTYPE"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        )["filesystems"][0]
        require(
            mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"},
            "pilot inputs/work are not node-local",
        )
    guards = helper("sdsc_training_preflight")
    return {"memory": guards.memory_envelope(), "cache": guards.pinned_cache(args.hf_home)}


def scientific(name):
    return importlib.import_module("posttrain_circuits." + name)


def tree_hashes(root):
    require(path(root).is_dir(), "artifact directory is missing")
    result = {}
    for entry in sorted(root.rglob("*")):
        require(not entry.is_symlink(), "artifact tree contains a symlink")
        if entry.is_file():
            result[str(entry.relative_to(root))] = file_hash(entry)
        else:
            require(entry.is_dir(), "artifact tree contains a special file")
    require(result, "artifact tree is empty")
    return result


def dataset_inputs(inputs):
    family = scientific("datasets.proofgraph.family").load_dataset_family(Path(inputs["dataset"]))
    examples = family.examples("train")[:4096]
    require(
        len(examples) == 4096 and len({row.example_id for row in examples}) == 4096,
        "pilot requires 4096 distinct original training prompts",
    )
    return family, examples


def teacher_generation(config, loaded):
    state, teacher = config["state_source"], config["teacher"]
    cls = scientific("learning.teacher.demo_generation").TeacherDemoGenerationConfig
    return cls(
        teacher_id=loaded.model_id,
        teacher_revision=loaded.requested_model_revision,
        resolved_teacher_commit=loaded.resolved_model_commit,
        sampling_request_seed=int(teacher["generation_seed"]),
        temperature=float(teacher["temperature"]),
        top_p=float(teacher["top_p"]),
        top_k=int(teacher.get("top_k", 0)),
        min_p=float(teacher.get("min_p", 0)),
        candidates_per_prompt=int(state["num_candidates"]),
        max_prompt_tokens=int(state["max_prompt_tokens"]),
        max_new_tokens=int(state["max_new_tokens"]),
    )


def teacher_shard(args, inputs, api, index):
    config = frozen_config(args, inputs, api)
    family, examples = dataset_inputs(inputs)
    selected = examples[index * 256 : (index + 1) * 256]
    target = args.pilot_root / "inputs/teacher_shards" / f"{index:02d}"
    require(not target.exists(), "teacher shard already exists; implicit retry is forbidden")
    loading = scientific("models.loading")
    generation = scientific("learning.teacher.demo_generation")
    loaded = loading.load_model_and_tokenizer(config["teacher"], for_training=False)
    generator = generation.HfTeacherCandidateGenerator(
        loading.move_model_to_local_cuda(loaded.model),
        loaded.tokenizer,
        max_new_tokens=int(config["state_source"]["max_new_tokens"]),
        model_config=config["teacher"],
    )
    specification = teacher_generation(config, loaded)
    require(
        specification.candidates_per_prompt == 8 and specification.sampling_request_seed == 31415,
        "teacher sampling population/seed differs",
    )
    result = generation.generate_teacher_demonstrations(
        selected, loaded.tokenizer, generator, specification, model_config=config["teacher"]
    )
    bindings = {
        **api.binding(config),
        "dataset_family_sha256": family.manifest["sha256"],
        "train_examples_file_sha256": family.boundary("train")["examples_file_sha256"],
    }
    store = scientific("datasets.teacher_demos.store")
    store.write_teacher_demo_store(
        target,
        result.attempts,
        ordered_prompt_ids=result.ordered_prompt_ids,
        prompt_manifest_hash=result.prompt_manifest_hash,
        tokenizer_hash=result.tokenizer_hash,
        generation=asdict(result.config),
        protocol_bindings=bindings,
    )
    _, manifest = store.read_teacher_demo_store(target, require_formal=True)
    return {
        "shard": index,
        "prompt_count": 256,
        "candidate_count": 2048,
        "manifest_sha256": manifest["sha256"],
        "files": tree_hashes(target),
        "partial_attempt_ledger_guaranteed": False,
    }


def merge_teacher_shards(args, inputs, api):
    """Merge original complete ledgers, including rejected candidates, in original order."""
    family, examples = dataset_inputs(inputs)
    store = scientific("datasets.teacher_demos.store")
    ledger = scientific("datasets.teacher_demos.ledger")
    expected_bindings = {
        **api.binding(frozen_config(args, inputs, api)),
        "dataset_family_sha256": family.manifest["sha256"],
        "train_examples_file_sha256": family.boundary("train")["examples_file_sha256"],
    }
    attempts, manifests, inventories = [], [], {}
    for index in range(16):
        root = args.pilot_root / "inputs/teacher_shards" / f"{index:02d}"
        _, manifest = store.read_teacher_demo_store(root, require_formal=True)
        expected_ids = [row.example_id for row in examples[index * 256 : (index + 1) * 256]]
        require(
            manifest["ordered_prompt_ids"] == expected_ids and manifest["attempt_count"] == 2048,
            "teacher shard population/order differs",
        )
        require(
            canonical(manifest["protocol_bindings"]) == canonical(expected_bindings),
            "teacher shard scientific binding differs",
        )
        if manifests:
            require(
                manifest["teacher_demo_generation"] == manifests[0]["teacher_demo_generation"]
                and manifest["tokenizer_hash"] == manifests[0]["tokenizer_hash"],
                "teacher shards use different generation protocols",
            )
        attempts.extend(
            ledger.validate_attempt_ledger(store._strict_json(root / "ledger.json", name="attempt ledger"))
        )
        manifests.append(manifest)
        inventories[str(index)] = tree_hashes(root)
    generation = {key: manifests[0]["teacher_demo_generation"][key] for key in store.GENERATION_INPUT_FIELDS}
    require(
        generation["candidates_per_prompt"] == 8 and generation["sampling_request_seed"] == 31415,
        "merged teacher population/seed differs",
    )
    target = Path(inputs["teacher_demos"])
    require(not target.exists(), "merged teacher store already exists")
    store.write_teacher_demo_store(
        target,
        attempts,
        ordered_prompt_ids=[row.example_id for row in examples],
        prompt_manifest_hash=sha(canonical([asdict(row) for row in examples])),
        tokenizer_hash=manifests[0]["tokenizer_hash"],
        generation=generation,
        protocol_bindings=expected_bindings,
    )
    store.read_teacher_demo_store(target, require_formal=True)
    return inventories


def validate_population(inputs, api, config):
    family, examples = dataset_inputs(inputs)
    expected_ids = [row.example_id for row in examples]
    teacher = scientific("datasets.teacher_demos.store")
    _, demos = teacher.read_teacher_demo_store(Path(inputs["teacher_demos"]), require_formal=True)
    require(
        demos["ordered_prompt_ids"] == expected_ids and demos["attempt_count"] == 4096 * 8,
        "pilot teacher store must cover the full 4096-by-8 population",
    )
    require(
        demos["prompt_manifest_hash"] == sha(canonical([asdict(row) for row in examples])),
        "teacher prompt bytes differ",
    )
    expected = {
        **api.binding(config),
        "dataset_family_sha256": family.manifest["sha256"],
        "train_examples_file_sha256": family.boundary("train")["examples_file_sha256"],
    }
    require(
        canonical(demos["protocol_bindings"]) == canonical(expected), "pilot teacher formal binding differs"
    )
    trajectories = scientific("datasets.trajectories.store").TrajectoryStore(Path(inputs["scored_bank"]))
    bank = trajectories.check_integrity()
    records = trajectories.read()
    require(
        {row.prompt_id for row in records} == set(expected_ids),
        "pilot common bank does not cover exactly 4096 prompts",
    )
    counts = {identifier: 0 for identifier in expected_ids}
    for row in records:
        counts[row.prompt_id] += 1
    # Original offline state source defines exactly four samples per prompt.
    require(set(counts.values()) == {4}, "pilot bank candidate counts differ")
    require(
        all(
            row.teacher_topk_ids
            and row.teacher_topk_logprobs
            and row.teacher_topk_mass
            and row.teacher_entropy
            for row in records
        ),
        "pilot bank contains unscored trajectories",
    )
    scientific("core.readiness").require_formal_prerequisite_binding(
        bank, api.binding(config), name="pilot bank"
    )
    require(
        0 < int(bank["reward_distribution"]["positive"]) < int(bank["total_trajectories"]),
        "pilot bank lacks both reward classes",
    )
    return {
        "teacher_manifest_sha256": demos["sha256"],
        "bank_manifest_sha256": bank["sha256"],
        "prompt_count": 4096,
        "candidate_count": 32768,
        "dataset_manifest_sha256": file_hash(Path(inputs["dataset"]) / "manifest.json"),
        "teacher_files": tree_hashes(Path(inputs["teacher_demos"])),
        "bank_files": tree_hashes(Path(inputs["scored_bank"])),
    }


def input_admission(args, inputs, api, *, create=False, shard_evidence=None):
    target = args.pilot_root / "pilot-input-admission.json"
    observed = validate_population(inputs, api, frozen_config(args, inputs, api))
    content = {
        "schema": "quest-sdsc-pilot-population-v1",
        "science_git_head": SCIENCE_HEAD,
        "g0_path": inputs["scientific_g0"],
        "g0_report_sha256": file_hash(Path(inputs["scientific_g0"])),
        "population": observed,
        "g0_passed": True,
        "pilot_passed": False,
        "difference_from_g0": "independent_4096_prompt_teacher_and_common_bank_not_the_256_prompt_g0_bank",
    }
    if create:
        require(shard_evidence is not None, "missing real teacher array evidence")
        content["teacher_array"] = shard_evidence
        content["sha256"] = sha(canonical(content))
        publish(target, content)
    else:
        prior = self_hashed(read_json(target))
        require(
            all(canonical(prior.get(key)) == canonical(value) for key, value in content.items()),
            "pilot input admission changed",
        )
        require(
            isinstance(prior.get("teacher_array"), dict), "pilot input admission lacks teacher array evidence"
        )
    return {"path": str(target), "sha256": file_hash(target), "population": observed}


def run_steps(args, steps):
    """Foreground execution; fail immediately and forward termination to all ranks."""
    journal = []
    log_root = (
        args.pilot_root
        / "stage-logs"
        / args.stage
        / (args.run_id + "-" + os.environ.get("SLURM_ARRAY_TASK_ID", "single"))
    )
    log_root.mkdir(parents=True, exist_ok=False)
    for number, step in enumerate(steps):
        argv = command(args, step)
        log = log_root / f"{number:03d}-{step.name}.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(
                argv,
                cwd=args.science_root,
                env=child_environment(args),
                stdout=stream,
                stderr=subprocess.STDOUT,
                # All ranks remain in the outer wrapper's process group so its
                # final kill/wait covers every writer before publication.
                start_new_session=False,
            )
            previous = {}
            interrupted = []

            def forward(signum, _frame, process=process, interrupted=interrupted):
                interrupted.append(signum)
                with contextlib.suppress(ProcessLookupError):
                    process.send_signal(signum)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    process.wait(timeout=5)

            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, forward)
            try:
                code = process.wait()
            finally:
                for signum, old in previous.items():
                    signal.signal(signum, old)
                stream.flush()
                os.fsync(stream.fileno())
        record = {
            "name": step.name,
            "argv": argv,
            "returncode": code,
            "log": str(log),
            # A successful original scientific CLI may deliberately be silent.
            # The storage identity checker safely hashes empty regular logs too.
            "log_sha256": helper("sdsc_pipeline_storage").identity(log)["sha256"],
        }
        journal.append(record)
        publish(log_root / f"{number:03d}.json", record)
        require(not interrupted, "scientific stage interrupted; implicit continuation is forbidden")
        require(code == 0, "scientific stage failed: " + step.name)
        for output in step.outputs:
            target = Path(output)
            target = target if target.is_absolute() else args.pilot_root / target
            file_hash(target)
    return journal


def validate_run(args, inputs, api, cell, root, *, ancestry=None):
    """Use the original full-checkpoint, update, rank-state and config validators."""
    import yaml

    validator = scientific("cli.finalize_pilot_training")
    manifest = scientific("artifacts.runs").validate_run_manifest_payload(read_json(root / "manifest.json"))
    binding = scientific("experiments.protocols.specs").validate_run_manifest_experiment_binding(manifest)
    resolved = yaml.load(
        (root / "resolved_config.yaml").read_text(),
        Loader=scientific("cli.factorial_run_validation")._UniqueKeyLoader,
    )
    expected = frozen_config(args, inputs, api, cell)
    require(
        canonical(resolved) == canonical(expected), "run resolved scientific/runtime configuration differs"
    )
    validator._validate_resolved_config_yaml_sha256(root, manifest=manifest)
    validator._load_run_config_binding(
        root, manifest=manifest, resolved_config=resolved, experiment_binding=binding
    )
    require(
        manifest["experiment_cell"] == cell
        and type(manifest["seed"]) is int
        and manifest["seed"] == 42
        and manifest["git_commit"] == SCIENCE_HEAD
        and manifest["dirty_working_tree"] is False
        and manifest["prereg_dirty"] is False,
        "training identity differs",
    )
    checkpoint = final_checkpoint(root)
    payload = validator._validate_checkpoint_scientific_binding(checkpoint, binding=binding)
    require(
        payload.get("world_size") == 4 and type(payload["world_size"]) is int,
        "pilot training was not four-rank",
    )
    require(file_hash(root / "metrics.jsonl") == manifest["metrics_sha256"], "training metrics bytes differ")
    if cell == "canonical_grpo":
        validator._validate_grpo_update_evidence(
            root, manifest=manifest, binding=binding, checkpoint_path=checkpoint, checkpoint_payload=payload
        )
    else:
        validator._validate_factorial_update_evidence(
            root,
            manifest=manifest,
            binding=binding,
            checkpoint_path=checkpoint,
            checkpoint_payload=payload,
            resolved_config=resolved,
        )
    if ancestry is not None:
        require(payload.get("resume_ancestry") == ancestry, "checkpoint resume ancestry differs")
    return {
        "root": root,
        "manifest": manifest,
        "binding": binding,
        "config": resolved,
        "checkpoint": checkpoint,
        "payload": payload,
    }


def validate_resume_source(args, inputs, api):
    validator = scientific("cli.finalize_pilot_training")
    root = cell_root(args, "canonical_sft")
    manifest = scientific("artifacts.runs").validate_run_manifest_payload(read_json(root / "manifest.json"))
    binding = scientific("experiments.protocols.specs").validate_run_manifest_experiment_binding(manifest)
    source = path(root / "checkpoints/step-00000020.pt")
    digest = file_hash(source)
    payload = validator._validate_checkpoint_scientific_binding(source, binding=binding)
    require(
        payload.get("world_size") == 4
        and payload.get("global_step") == 20
        and payload.get("resume_ancestry") == []
        and payload.get("token_budget", {}).get("stop_reason") is None,
        "resume source is not the original four-rank optimizer boundary 20",
    )
    config = frozen_config(args, inputs, api)
    validator._validate_factorial_accelerator_state(
        root, checkpoint_path=source, checkpoint_payload=payload, resolved_config=config
    )
    require(
        validator._validate_factorial_rank_states(
            payload,
            expected_batch_partition_protocol=config["trainer"].get(
                "batch_partition_protocol", "rank_local_round_robin_v1"
            ),
            expected_max_steps=120,
        )
        == 4,
        "resume source rank inventory differs",
    )
    return {
        "path": str(source),
        "sha256": digest,
        "world_size": 4,
        "global_step": 20,
        "accelerator_state_files": payload["accelerate_state_files"],
    }


def resume_comparison(args, inputs, api):
    source = validate_resume_source(args, inputs, api)
    ancestry = ["sha256:" + source["sha256"]]
    compare = scientific("cli.compare_distributed_resume")
    # Retain only hashes/metadata, not three full model+optimizer states in RAM.
    evidence = {}
    for label, root, ancestors in (
        ("reference", cell_root(args, "canonical_sft"), []),
        ("left", args.pilot_root / "resume-a", ancestry),
        ("right", args.pilot_root / "resume-b", ancestry),
    ):
        value = validate_run(args, inputs, api, "canonical_sft", root, ancestry=ancestors)
        payload = value["payload"]
        metric = compare._last_metric(root / "metrics.jsonl")
        hashes = legacy_runtime_hashes(payload)
        require(payload["format"] == "accelerate_fsdp_full_export_v1", "resume export format differs")
        require(
            type(payload["global_step"]) is int
            and 20 < payload["global_step"] <= 120
            and compare._metric_step(metric) == payload["global_step"],
            "resume terminal optimizer/metric step differs",
        )
        require(
            payload["policy_version"] == payload["online_rollout_round"] == 0,
            "canonical SFT rollout state differs",
        )
        if ancestors:
            require(
                payload["update_norm_baseline_checkpoint_sha256"] == source["sha256"],
                "resume update baseline differs",
            )
        evidence[label] = {
            "root": str(root),
            "checkpoint_sha256": file_hash(value["checkpoint"]),
            "manifest_sha256": file_hash(root / "manifest.json"),
            "metrics_sha256": file_hash(root / "metrics.jsonl"),
            "runtime_state_hashes": {key: hashes[key] for key in sorted(LEGACY_CORE)},
            "accelerator_rank_rng_sha256": native_rank_rng(payload),
            "global_step": payload["global_step"],
            "loss": compare._objective_loss(metric),
            "optimizer_state_key_type": payload["optimizer_state_key_type"],
            "launcher_sha256": file_hash(args.science_root / "configs/accelerate/fsdp.yaml"),
            "stop_reason": value["manifest"]["training_stop_reason"],
        }
        del value, payload
    return compare_resume_evidence(evidence, source, api.binding(frozen_config(args, inputs, api)))


def compare_resume_evidence(evidence, source, formal):
    require(set(evidence) == {"reference", "left", "right"}, "resume comparison lacks independent runs")
    reference = evidence["reference"]
    for value in evidence.values():
        for key in (
            "runtime_state_hashes",
            "accelerator_rank_rng_sha256",
            "global_step",
            "optimizer_state_key_type",
            "launcher_sha256",
            "stop_reason",
        ):
            require(canonical(value[key]) == canonical(reference[key]), "resume state differs: " + key)
        require(type(value["loss"]) in (int, float) and math.isfinite(value["loss"]), "nonfinite resume loss")
    require(len({row["root"] for row in evidence.values()}) == 3, "resume roots are not independent")
    errors = {
        left + "_vs_" + right: abs(evidence[left]["loss"] - evidence[right]["loss"])
        for left, right in (("left", "reference"), ("right", "reference"), ("left", "right"))
    }
    require(all(error <= 1e-6 for error in errors.values()), "resume loss differs beyond fixed tolerance")
    content = {
        "schema": "quest-sdsc-pilot-resume-v1",
        "checkpoint_protocol": "original_rank_local_pilot_without_G0_only_sharding_fields",
        "passed": True,
        "world_size": 4,
        "source": source,
        "runs": evidence,
        "tolerance": 1e-6,
        "next_loss_absolute_errors": errors,
        "checks": {
            "original_checkpoint_config_rank_update_validators": True,
            "all_core_states_equal_uninterrupted_reference": True,
            "two_independent_explicit_resumes": True,
        },
        **formal,
    }
    content["sha256"] = sha(canonical(content))
    return content


def native_rank_rng(payload):
    """Compare every real Accelerate rank RNG payload, not pickle container bytes."""
    import numpy as np
    import torch

    names = {name for name in payload["accelerate_state_files"] if name.startswith("random_states_")}
    require(names == {f"random_states_{rank}.pkl" for rank in range(4)}, "Accelerate RNG inventory differs")

    def normalized(value):
        if isinstance(value, np.ndarray):
            return {
                "numpy_dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes_sha256": sha(value.tobytes()),
            }
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return type(value)(normalized(item) for item in value)
        return value

    result = {}
    for name in sorted(names):
        target = path(Path(payload["accelerate_state_dir"]) / name)
        require(file_hash(target) == payload["accelerate_state_files"][name], "Accelerate RNG bytes changed")
        value = torch.load(target, map_location="cpu", weights_only=False)
        require(
            isinstance(value, dict)
            and {
                "step",
                "random_state",
                "numpy_random_seed",
                "torch_manual_seed",
                "torch_cuda_manual_seed",
            }.issubset(value),
            "Accelerate CUDA/CPU RNG state is incomplete",
        )
        result[name] = scientific("artifacts.checkpoints").torch_state_hash(normalized(value))
    return result


def legacy_runtime_hashes(payload):
    hashes = scientific("artifacts.checkpoints").checkpoint_runtime_state_hashes(payload)
    require(
        LEGACY_CORE.issubset(hashes) and "trainer_state_by_rank" not in payload,
        "legacy resume runtime hash inventory differs",
    )
    return {key: hashes[key] for key in sorted(LEGACY_CORE)}


def validate_resume(args, inputs, api, *, create=False):
    observed = resume_comparison(args, inputs, api)
    destination = args.pilot_root / "distributed_resume.json"
    if create:
        publish(destination, observed)
    else:
        require(canonical(read_json(destination)) == canonical(observed), "resume proof does not replay")
    return {"path": str(destination), "sha256": file_hash(destination)}


def adapted_finalizer(original, bank_path, bank_sha256, input_proof):
    """Isolate exactly one reviewed migration: independently bound 4096-prompt bank.

    No source/global module is mutated. All other original main bytecode and
    validators remain intact. The resulting report is explicitly SDSC-adapted.
    """
    require(
        input_proof.get("population", {}).get("prompt_count") == 4096, "incomplete pilot population proof"
    )
    original_binding = original.__globals__["_require_g0_file_binding"]

    def binding(g0, target, *, name):
        if name == "rollout bank":
            require(
                Path(target) == bank_path and file_hash(target) == bank_sha256,
                "pilot bank changed after admission",
            )
            proof = self_hashed(read_json(input_proof["path"], input_proof["sha256"]))
            require(
                proof["g0_report_sha256"] == file_hash(Path(proof["g0_path"]))
                and self_hashed(read_json(proof["g0_path"])) == g0,
                "bank admission is detached from actual G0",
            )
            require(
                proof["population"]["bank_files"]["manifest.json"] == bank_sha256,
                "bank admission bytes differ",
            )
            return
        original_binding(g0, target, name=name)

    namespace = dict(original.__globals__)
    namespace["_require_g0_file_binding"] = binding
    return FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )


def finalize(args, admission, inputs, api):
    proof = input_admission(args, inputs, api)
    validate_resume(args, inputs, api)
    jobs = {}
    for name in TERMINAL_STAGES:
        jobs[name] = stage_evidence(args, admission, name)["job_id"]
    job_path = args.pilot_root / "job_ids.env"
    content = "".join(f"{name}={jobs[name]}\n" for name in TERMINAL_STAGES).encode()
    if job_path.exists():
        require(job_path.read_bytes() == content, "pilot job registry differs")
    else:
        with job_path.open("xb") as stream:
            stream.write(content)
    bank = Path(inputs["scored_bank"]) / "manifest.json"
    original = scientific("cli.finalize_pilot").main
    output = args.pilot_root / "pilot_scientific_report.json"
    require(not output.exists(), "scientific finalization output already exists")
    runner = adapted_finalizer(original, bank, file_hash(bank), proof)
    runner(
        [
            *common_overrides(args, inputs),
            "--run-dir",
            str(args.pilot_root),
            "--g0",
            inputs["scientific_g0"],
            "--bank-manifest",
            str(bank),
            "--probe-manifest",
            inputs["probes"],
            "--validation-manifest",
            str(Path(inputs["dataset"]) / "manifest.json"),
            "--job-ids",
            str(job_path),
            "--output",
            str(output),
        ]
    )
    scientific_report = self_hashed(read_json(output))
    require(
        scientific_report.get("passed") is True
        and all(item is True for item in scientific_report["checks"].values()),
        "pilot scientific gate failed",
    )
    result = {
        "schema": "quest-sdsc-pilot-v1",
        "passed": True,
        "pilot_passed": True,
        "science_git_head": SCIENCE_HEAD,
        "input_admission": proof,
        "scientific_report": {"path": str(output), "sha256": file_hash(output)},
        "adaptation": (
            "original main with only rollout-bank G0 binding replaced by full4096 independent input admission"
        ),
        "original_source_modified": False,
        "claim_scope": scientific_report["claim_scope"],
        "full_three_seed_factorial_launched": False,
        "execution_class_certified": False,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    result["sha256"] = sha(canonical(result))
    publish(args.pilot_root / "pilot_report.json", result)
    return result


def validate_mib(args):
    teacher = helper("sdsc_teacher_prepare")
    root = path(args.mib_repository)
    lock = read_json(args.science_root / "deployments/qwen3_v2_g0/dependency-lock.json")
    record = next(item for item in lock["external_sources"] if item["name"] == "MIB-circuit-track")
    revision = teacher.git(root, "rev-parse", "HEAD").decode().strip()
    require(
        revision == record["revision"] and (root / "run_attribution.py").is_file(),
        "MIB revision/entrypoint differs",
    )
    require(
        not teacher.git(root, "status", "--porcelain=v1", "--untracked-files=all"), "MIB checkout is dirty"
    )
    actual = teacher.git(root, "submodule", "status", "--recursive").decode().splitlines()
    require(all(line.startswith(" ") for line in actual), "MIB submodule is missing/changed")
    return {"revision": revision, "submodules": actual}


def prerequisites(args, admission, inputs, api):
    if args.stage == "prepare":
        require(not (args.pilot_root / "pilot_manifest.json").exists(), "pilot was already prepared")
        return {}
    manifest = self_hashed(read_json(args.pilot_root / "pilot_manifest.json"))
    require(
        manifest.get("seed") == 42
        and manifest.get("git_commit") == SCIENCE_HEAD
        and manifest.get("g0_path") == inputs["scientific_g0"]
        and manifest.get("g0_sha256") == admission["g0"]["report_sha256"]
        and manifest.get("cells") == list(CELLS[:6])
        and manifest.get("anchors") == list(CELLS[6:]),
        "pilot preparation identity/population differs",
    )
    if args.stage == "teacher-shard":
        stage_evidence(args, admission, "prepare")
        return {}
    if args.stage == "pilot-inputs":
        return {"teacher_array": stage_evidence(args, admission, "teacher_shards", count=16)}
    stage_evidence(args, admission, "pilot_inputs")
    proof = input_admission(args, inputs, api)
    if args.stage == "train-cell":
        return {"inputs": proof}
    training = stage_evidence(args, admission, "training")
    args.training_job_id = training["job_id"]
    required = {
        "bind-training": (),
        "initial-circuits": (),
        "final-circuits": ("initial_circuits",),
        "local-fork": ("initial_circuits", "final_circuits"),
        "resume": ("initial_circuits", "final_circuits", "local_fork"),
        "dynamics": ("initial_circuits", "final_circuits", "local_fork", "resume"),
        "finalize": ("initial_circuits", "final_circuits", "local_fork", "resume", "dynamics"),
    }[args.stage]
    for name in required:
        stage_evidence(args, admission, name)
    return {"inputs": proof, "training_job_id": args.training_job_id}


def validate_stage_artifacts(args, inputs, api):
    finalizer = scientific("cli.finalize_pilot")
    expected = api.binding(frozen_config(args, inputs, api))
    artifacts = {}
    if args.stage in {"initial-circuits", "final-circuits", "dynamics"}:
        final = args.stage != "initial-circuits"
        for row in matrix(final):
            base = args.pilot_root / "circuits" / ("final" if final else "initial")
            if final:
                base /= row["cell"]
            base = base / row["stage_label"] / row["cohort"]
            checkpoint_hash = (
                file_hash(final_checkpoint(cell_root(args, row["cell"])))
                if final
                else inputs["initial_checkpoint_sha256"]
            )
            if args.stage == "dynamics":
                target = (
                    args.pilot_root
                    / "dynamics"
                    / row["cell"]
                    / row["stage_label"]
                    / (row["cohort"] + ".json")
                )
                artifact = self_hashed(read_json(target))
                finalizer._require_dynamics_slot(
                    artifact=artifact,
                    expected_initial_sha256=inputs["initial_checkpoint_sha256"],
                    expected_final_sha256=checkpoint_hash,
                    cohort=row["cohort"],
                    probe_stage=row["probe_stage"],
                )
                selected = ((target, artifact),)
            else:
                circuit, exact = (
                    self_hashed(read_json(base / name)) for name in ("circuit.json", "exact_patching.json")
                )
                finalizer._require_circuit_slot(
                    circuit=circuit,
                    exact=exact,
                    circuit_path=base / "circuit.json",
                    expected_checkpoint_sha256=checkpoint_hash,
                    cohort=row["cohort"],
                    probe_stage=row["probe_stage"],
                )
                selected = ((base / "circuit.json", circuit), (base / "exact_patching.json", exact))
            for target, artifact in selected:
                finalizer.require_formal_prerequisite_binding(artifact, expected, name="pilot stage artifact")
                require(finalizer._finite(artifact), "non-finite pilot stage artifact")
                artifacts[str(target)] = file_hash(target)
    elif args.stage == "local-fork":
        import torch

        config = frozen_config(args, inputs, api)
        target = args.pilot_root / "local_fork/results.json"
        artifact = self_hashed(read_json(target))
        source = cell_root(args, "canonical_sft") / "manifest.json"
        source_binding = finalizer.validate_run_manifest_experiment_binding(read_json(source))

        def source_model_factory():
            return finalizer.load_model_and_tokenizer(config["model"], for_training=True).model.to(
                torch.device("cuda")
            )

        finalizer.validate_local_fork_report(
            artifact,
            checkpoint_root=target.parent / "checkpoints",
            bundle_path=target.parent / "bundle.pt",
            require_full_protocol=True,
            require_source_binding=True,
            expected_source_manifest_path=source,
            expected_source_manifest_sha256=file_hash(source),
            expected_source_experiment_binding_sha256=source_binding.scientific_sha256,
            source_model_factory=source_model_factory,
            expected_trajectory_bank_manifest_path=Path(inputs["scored_bank"]) / "manifest.json",
            expected_prompt_manifest_path=target.parent / "inputs/prompts.json",
            expected_probe_manifest_path=target.parent / "inputs/probe_set.json",
        )
        finalizer.require_formal_prerequisite_binding(artifact, expected, name="pilot local fork")
        artifacts[str(target)] = file_hash(target)
    return artifacts


def execute_stage(args, admission, inputs, api, index, gates):
    if args.stage == "teacher-shard":
        return teacher_shard(args, inputs, api, index)
    if args.stage == "finalize":
        return finalize(args, admission, inputs, api)
    prefix = []
    if args.stage == "pilot-inputs":
        prefix.append({"teacher_shards": merge_teacher_shards(args, inputs, api)})
    if args.stage == "initial-circuits":
        # A fresh initial-circuits job performs the heavy checkpoint validation,
        # using real array accounting. Never run this work on a login node.
        require(
            not (args.pilot_root / "training_artifact_chain.json").exists(),
            "training already bound in this pipeline branch",
        )
        original = args.stage
        args.stage = "bind-training"
        try:
            prefix.extend(run_steps(args, stage_plan(args, inputs)))
        finally:
            args.stage = original
    checkpoints = None
    if args.stage in {"final-circuits", "local-fork"}:
        checkpoints = {cell: final_checkpoint(cell_root(args, cell)) for cell in CELLS}
    if args.stage == "resume":
        validate_resume_source(args, inputs, api)
    steps = stage_plan(args, inputs, index=index, checkpoints=checkpoints)
    result = {"commands": prefix + run_steps(args, steps)}
    if args.stage == "pilot-inputs":
        result["input_admission"] = input_admission(
            args, inputs, api, create=True, shard_evidence=gates["teacher_array"]
        )
    elif args.stage == "train-cell":
        value = validate_run(args, inputs, api, CELLS[index], cell_root(args, CELLS[index]), ancestry=[])
        result["training"] = {
            "cell": CELLS[index],
            "manifest_sha256": file_hash(value["root"] / "manifest.json"),
            "checkpoint_sha256": file_hash(value["checkpoint"]),
        }
    elif args.stage == "resume":
        result["resume"] = validate_resume(args, inputs, api, create=True)
    result["artifact_validation"] = validate_stage_artifacts(args, inputs, api)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=STAGES, required=True)
    for name in (
        "science-root",
        "work-dir",
        "pilot-root",
        "g0-root",
        "hf-home",
        "python",
        "mib-repository",
        "provenance-manifest",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--inputs", "--admission", dest="inputs", type=Path, required=True)
    result.add_argument("--inputs-sha256", "--admission-sha256", dest="inputs_sha256", required=True)
    for name in ("run-id", "code-sha256", "provenance-manifest-sha256"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--validate-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run identity")
    for value in (args.code_sha256, args.inputs_sha256, args.provenance_manifest_sha256):
        require(SHA.fullmatch(value), "invalid hash identity")
    admission, inputs = load_inputs(args)
    identity = runtime(args)
    # Set only fixed offline/thread environment values, preserving the assigned
    # CUDA visibility; no scheduler protocol or allocation is fabricated.
    environment = child_environment(args)
    os.environ.clear()
    os.environ.update(environment)
    os.chdir(args.science_root)
    api, provenance = science_api(args)
    frozen_config(args, inputs, api)
    mib = validate_mib(args)
    g0 = replay_g0(args, admission)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "validation_only": True,
                    "runtime": identity,
                    "provenance": provenance,
                    "mib": mib,
                    "g0": g0,
                    "submitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    resources = allocation(args)
    storage = node_storage(args)
    index = resources["array_index"]
    suffix = "single" if index is None else str(index)
    report = args.pilot_root / ("sdsc-stage-" + args.stage + "-" + suffix + ".json")
    require(not report.exists(), "stage output already exists; no implicit retry")
    claim = args.pilot_root / (".sdsc-claim-" + args.stage + "-" + suffix)
    with claim.open("xb") as stream:
        stream.write(canonical({"run_id": args.run_id, "job_id": resources["job_id"]}))
        stream.flush()
        os.fsync(stream.fileno())
    content = {
        "schema": "quest-sdsc-pilot-stage-v1",
        "task": "qwen3-v2-pilot",
        "stage": args.stage,
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "job_id": resources["job_id"],
        "provenance_manifest_sha256": args.provenance_manifest_sha256,
        "science_git_head": SCIENCE_HEAD,
        "admission_sha256": args.inputs_sha256,
        "runtime": identity,
        "allocation": resources,
        "storage": storage,
        "mib": mib,
        "g0_replay": g0,
        "pilot_passed": False,
        "execution_class_certified": False,
        "passed": False,
        "exit_code": 1,
    }
    try:
        gates = prerequisites(args, admission, inputs, api)
        result = execute_stage(args, admission, inputs, api, index, gates)
        content.update(result=result, passed=True, exit_code=0, pilot_passed=args.stage == "finalize")
    except (Exception, SystemExit) as error:
        content["error"] = type(error).__name__ + ": " + str(error)[:2000]
        publish(report, content)
        raise
    publish(report, content)
    print(
        json.dumps(
            {"stage": args.stage, "passed": True, "report": str(report), "report_sha256": file_hash(report)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("SDSC pilot stopped: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
