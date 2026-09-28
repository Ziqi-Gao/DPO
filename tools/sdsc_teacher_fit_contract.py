"""Stdlib-only identity contract for the separate four-H100 teacher fit.

This task-specific execution plan is not a Blackwell execution certificate or
formal teacher acceptance. Deployment run IDs and unrelated source files do
not enter its safety identity.
"""

import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

TASK_MODES = {
    "qwen3-v2-teacher-fit-preflight": "preflight",
    "qwen3-v2-teacher-fit": "full-fit",
}
KERNEL_PATHS = (
    "tools/sdsc_teacher_fit.py",
    "tools/sdsc_teacher_fit_job.sh",
    "tools/sdsc_teacher_fit_contract.py",
    "tools/sdsc_teacher_fit_memory.py",
    "tools/sdsc_teacher_prepare.py",
    "tools/sdsc_training_preflight.py",
    "tools/sdsc_teacher_adapt.py",
    "tools/sdsc_teacher_probe.py",
    "src/posttrain_circuits/learning/teacher/adaptation.py",
    "src/posttrain_circuits/learning/teacher/adaptation_fit.py",
    "src/posttrain_circuits/learning/teacher/adaptation_evaluation.py",
    "src/posttrain_circuits/learning/teacher/evaluation.py",
    "src/posttrain_circuits/learning/teacher/demo_generation.py",
    "src/posttrain_circuits/cli/evaluate_teacher_readiness.py",
    "src/posttrain_circuits/learning/supervision/losses.py",
    "src/posttrain_circuits/models/loading.py",
    "src/posttrain_circuits/models/prompt_protocol.py",
    "src/posttrain_circuits/models/adapted_teacher.py",
    "src/posttrain_circuits/causal_circuits/metrics/probes.py",
)
DEPENDENCIES = {
    "torch": "2.8.0+cu128",
    "transformers": "4.56.2",
    "tokenizers": "0.22.0",
    "safetensors": "0.5.3",
    "huggingface-hub": "0.36.2",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pydantic": "2.11.7",
    "PyYAML": "6.0.2",
    "accelerate": "1.10.1",
    "peft": "0.17.1",
}
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
EXPECTED_DATASETS = {
    "preflight": {
        "dataset_manifest_sha256": "208f279053605acfed08b4bbd027b23b23b3e39337ecd70dbc87455cd9814368",
        "teacher_fit_examples_sha256": "1f5acafb7d70e0927df0a669d8b31e90ea84db0c39ecd9080515f9b275510070",
        "teacher_dev_examples_sha256": "39a27aef84df9d60edcb8778e1d817a99e1772d3376cb42faa40ac7d5ec9f707",
    },
    "full-fit": {
        "dataset_manifest_sha256": "742b62a1ee328c8d8f660106265e4a342458fe5145243ecb08368eaa745f502d",
        "teacher_fit_examples_sha256": "a85ac939cba53e675d1737c75d9ba77d2516ed0f2cb0def642c97774cc09522f",
        "teacher_dev_examples_sha256": "3bd34a2e3a743c41a5215317f220c0ae99dba66841776f81565c61768f16e4c5",
    },
}
READINESS_THRESHOLDS = {
    "minimum_teacher_answer_accuracy": 0.90,
    "minimum_teacher_exact_proof_accuracy": 0.85,
    "minimum_teacher_first_rule_top1_accuracy": 0.80,
    "minimum_teacher_intermediate_top1_accuracy": 0.80,
    "minimum_teacher_topk_mass": 0.90,
    "minimum_teacher_topk_target_coverage": 0.90,
    "minimum_corrupted_prefix_recovery_accuracy": 0.70,
    "minimum_causal_shift_logprob": 0.0,
}
LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXECUTION_CHECKS = {
    "isolated_data",
    "finite_loss_and_gradients",
    "exact_optimizer_steps",
    "exact_global_batch_and_tokens",
    "nonzero_adapter_update",
    "frozen_base_before_merge",
    "complete_module_coverage",
    "all_four_ranks",
    "same_world_resume",
    "changed_world_resume_rejected",
    "dense_export_reload",
    "all_scheduled_merged_dev_evaluations",
    "cgroup_headroom",
}
MAX_OUTPUT_BYTES = 100 * 1024**3
FIXED_EXECUTION = {
    "version": "qwen3-v2-teacher-fit-execution-v2",
    "scope": "separate_h100_teacher_lora_ddp_not_blackwell_certificate",
    "model": {
        "name": "Qwen/Qwen3-8B",
        "base_revision": BASE_REVISION,
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
        "quantized": False,
        "attention_implementation": "sdpa",
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "use_cache_during_training": False,
        "tokenizer_revision": BASE_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    "runtime": {
        "python": "3.12.13",
        "python_sha256": "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "packages": DEPENDENCIES,
        "cuda": "12.8",
        "nccl": "2.27.3",
    },
    "allocation": {
        "nodes": 1,
        "world_size": 4,
        "gpu_type": "h100",
        "cpus": 24,
        "requested_cpus_per_task": 24,
        "allocated_cpus_on_node": "record_actual_exclusive_node_allocation",
        "threads_per_rank": 6,
        "host_memory_bytes": 192 * 1024**3,
        "minimum_host_headroom_bytes": max(32 * 1024**3, math.ceil(0.20 * 192 * 1024**3)),
        "memory_policy": "exclusive_node_workload_budget_v1",
        "host_memory_scope": "workload_budget_not_claimed_kernel_cap",
        "effective_kernel_limit": "finite_own_job_or_step_boundary_at_least_workload_budget",
        "memory_peak_scope": "maximum_own_job_and_batch_step_real_cgroup_peak",
    },
    "batch": {
        "global_sequences": 64,
        "microbatch_sequences_per_rank": 1,
        "sequences_per_rank_per_update": 16,
        "loss": "mean_of_response_token_means_per_sequence",
        "gradient_reduction": "ddp_rank_mean_after_16_sequence_local_mean",
        "token_accounting": "exact_nonpadding_model_input_tokens_all_four_ranks",
    },
    "distributed": {
        "wrapper": "DistributedDataParallel",
        "control_backend": "gloo",
        "training_backend": "nccl",
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "synchronize_only_last_accumulation_microstep": True,
        "initial_nccl_scalar_timeout_seconds": 120,
    },
    "token_envelope": {"max_sequence_length": 1536, "max_fit_prefix": 1280, "max_response_with_eos": 256},
    "lora": {"rank": 32, "alpha": 64, "dropout": 0.0, "modules": list(LORA_MODULES)},
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "max_grad_norm": 1.0,
    },
    "checkpoint": {"optimizer_boundary_only": True, "same_world_resume": True, "changed_world_resume": False},
    "development": {"max_new_tokens": 256, "prefix_top_k": 128, "formal_validation_used": False},
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha256_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def task_for_mode(mode):
    for task, candidate in TASK_MODES.items():
        if mode == candidate:
            return task
    raise ValueError("unsupported teacher fit mode")


def actual_plan(mode):
    task_for_mode(mode)
    preflight = mode == "preflight"
    return {
        "version": "qwen3-v2-teacher-fit-actual-v1",
        "mode": mode,
        "dataset_manifest_sha256": EXPECTED_DATASETS[mode]["dataset_manifest_sha256"],
        "seed": 271828,
        "raw_fit_seed_start": 70_000_042,
        "raw_dev_seed_start": 80_000_042,
        "train_examples": 512 if preflight else 8192,
        "dev_examples": 32 if preflight else 512,
        "optimizer_steps": 8 if preflight else 512,
        "epochs": 1 if preflight else 4,
        "global_batch_size": 64,
        "microbatch_size": 1,
        "learning_rate": 1e-4,
        "schedule": "constant" if preflight else "cosine",
        "warmup_steps": 0 if preflight else 16,
        "checkpoint_steps": [8] if preflight else [128, 256, 384, 512],
        "development_max_new_tokens": 256,
        "checkpoint_selection": "none_preflight"
        if preflight
        else "first_scheduled_all_eight_development_gates",
        "complete_all_scheduled_optimizer_steps": True,
        "formal_teacher_accepted": False,
    }


def execution_plan(manifest):
    """Bind only named safety surfaces from an already verified release manifest."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("teacher fit requires a release file manifest")
    files = {}
    for row in manifest["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in files:
            raise ValueError("invalid or duplicate release file record")
        files[row["path"]] = row
    hashes = {}
    for path in KERNEL_PATHS:
        row = files.get(path)
        if row is None or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))):
            raise ValueError("missing or invalid teacher fit safety surface: " + path)
        hashes[path] = row["sha256"]
    return {**copy.deepcopy(FIXED_EXECUTION), "kernel_sha256": hashes}


def trainer_identity(mode, execution_plan_sha256, fit_dataset_sha256):
    """Expected task adaptation identity, not a frozen preregistration claim."""
    for value in (execution_plan_sha256, fit_dataset_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("invalid teacher fit content identity")
    return {
        "base_revision": BASE_REVISION,
        "tokenizer_sha256": TOKENIZER_FINGERPRINT,
        "fit_dataset_sha256": fit_dataset_sha256,
        "protocol_sha256": sha256_value(
            {
                "actual_plan": actual_plan(mode),
                "execution_plan_sha256": execution_plan_sha256,
                "teacher_readiness_thresholds": READINESS_THRESHOLDS,
            }
        ),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_memory(value, job_id=None):
    budget = FIXED_EXECUTION["allocation"]["host_memory_bytes"]
    required = FIXED_EXECUTION["allocation"]["minimum_host_headroom_bytes"]
    require(
        isinstance(value, dict)
        and value.get("passed") is True
        and value.get("policy") == FIXED_EXECUTION["allocation"]["memory_policy"]
        and all(
            type(value.get(key)) is int
            for key in (
                "budget_bytes",
                "effective_limit_bytes",
                "current_bytes",
                "peak_bytes",
                "headroom_bytes",
                "minimum_headroom_bytes",
                "kernel_headroom_bytes",
            )
        )
        and value["budget_bytes"] == budget
        and value["effective_limit_bytes"] >= budget
        and 0 < value["current_bytes"] <= value["peak_bytes"] <= budget
        and value["headroom_bytes"] == budget - value["peak_bytes"] >= required
        and value["minimum_headroom_bytes"] == required
        and value["kernel_headroom_bytes"] == value["effective_limit_bytes"] - value["peak_bytes"]
        and value.get("kernel_cap_equals_budget") is (value["effective_limit_bytes"] == budget)
        and value.get("software_budget_only") is (value["effective_limit_bytes"] > budget),
        "teacher fit cgroup memory/headroom evidence differs",
    )
    require(
        isinstance(value.get("job_id"), str)
        and re.fullmatch(r"[1-9][0-9]*", value["job_id"])
        and (job_id is None or value["job_id"] == job_id),
        "teacher fit memory job identity differs",
    )
    path, step = value.get("path"), value.get("step_path")
    require(
        isinstance(path, str)
        and path.startswith("/")
        and str(PurePosixPath(path)) == path
        and ".." not in PurePosixPath(path).parts
        and not any(ord(c) < 32 for c in path)
        and path.endswith("/job_" + value["job_id"])
        and step == path + "/step_batch",
        "teacher fit memory must bind its own job and batch step",
    )
    ancestors = value.get("ancestors")
    require(isinstance(ancestors, list) and ancestors, "teacher fit memory ancestry is missing")
    by_path = {}
    finite = []
    for row in ancestors:
        require(isinstance(row, dict) and isinstance(row.get("path"), str), "invalid memory ancestor")
        name = row["path"]
        location = PurePosixPath(name)
        require(
            name not in by_path
            and location.is_absolute()
            and str(location) == name
            and ".." not in location.parts
            and not any(ord(c) < 32 for c in name)
            and (
                name in (path, step)
                or location in PurePosixPath(step).parents
                or PurePosixPath(step) in location.parents
            ),
            "memory ancestor is outside the job batch lineage",
        )
        by_path[name] = row
        limit = row.get("limit_bytes")
        require(limit is None or (type(limit) is int and 0 < limit < 2**60), "invalid effective memory limit")
        if limit is not None:
            finite.append(limit)
    require(path in by_path and step in by_path, "own job/step memory counters are missing")
    workload = [by_path[path], by_path[step]]
    require(
        any(row.get("limit_bytes") is not None for row in workload)
        and finite
        and min(finite) == value["effective_limit_bytes"]
        and all(type(row.get(key)) is int for row in workload for key in ("current_bytes", "peak_bytes"))
        and all(
            0 < row["current_bytes"] <= row["peak_bytes"] <= value["effective_limit_bytes"]
            for row in workload
        )
        and value["current_bytes"] == max(row["current_bytes"] for row in workload)
        and value["peak_bytes"] == max(row["peak_bytes"] for row in workload),
        "teacher fit own-job workload counters or finite kernel boundary differ",
    )


def validate_rank_memory(rows, job_id=None):
    require(
        isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id),
        "teacher fit rank memory lacks the report job identity",
    )
    require(
        isinstance(rows, list)
        and len(rows) == 4
        and all(isinstance(row, dict) and type(row.get("rank")) is int for row in rows)
        and {row["rank"] for row in rows} == {0, 1, 2, 3},
        "teacher fit memory evidence omits a rank",
    )
    for row in rows:
        validate_memory(row.get("cgroup_memory"), job_id)


def validate_report(report, expected_execution_plan, expected_task):
    mode = TASK_MODES[expected_task]
    plan = actual_plan(mode)
    expected_sha = sha256_value(expected_execution_plan)
    require(
        report.get("task") == expected_task
        and report.get("mode") == mode
        and report.get("artifact_kind") == "teacher_adaptation_fit_execution",
        "teacher fit report task, mode or artifact differs",
    )
    require(
        all(report.get(key) is True for key in ("passed", "training_started", "teacher_training_started"))
        and all(
            report.get(key) is False
            for key in (
                "accepted_science",
                "formal_teacher_accepted",
                "full_teacher_ready",
                "g0_passed",
                "execution_class_certified",
                "student_training_started",
                "readiness_artifact_produced",
            )
        )
        and type(report.get("exit_code")) is int
        and report["exit_code"] == 0,
        "teacher fit execution is incomplete or claims formal acceptance",
    )
    require(
        canonical(report.get("execution_plan")) == canonical(expected_execution_plan)
        and report.get("execution_plan_sha256") == expected_sha
        and canonical(report.get("actual_plan")) == canonical(plan)
        and report.get("actual_plan_sha256") == sha256_value(plan),
        "teacher fit report does not match the intended plans",
    )
    allocation = report.get("allocation")
    require(isinstance(allocation, dict), "teacher fit allocation evidence is missing")
    raw_cpus = re.fullmatch(
        r"([1-9][0-9]*)(?:\(x1\))?", str(allocation.get("allocated_cpus_slurm_value", ""))
    )
    require(isinstance(allocation.get("cuda_visible_devices"), str), "missing actual CUDA visibility")
    visible = allocation["cuda_visible_devices"].split(",")
    require(
        allocation.get("job_id") == report.get("job_id")
        and type(allocation.get("cpus")) is int
        and allocation["cpus"] == 24
        and type(allocation.get("requested_cpus_per_task")) is int
        and allocation["requested_cpus_per_task"] == 24
        and type(allocation.get("allocated_cpus_on_node")) is int
        and allocation["allocated_cpus_on_node"] >= 24
        and raw_cpus is not None
        and int(raw_cpus[1]) == allocation["allocated_cpus_on_node"]
        and type(allocation.get("memory_mib")) is int
        and allocation["memory_mib"] == 196608
        and len(visible) == len(set(visible)) == 4
        and all(value and value.strip() == value and value != "-1" for value in visible),
        "teacher fit actual CPU/GPU allocation differs from requested workload",
    )
    require(
        type(report.get("world_size")) is int
        and report["world_size"] == 4
        and type(report.get("optimizer_steps")) is int
        and report["optimizer_steps"] == plan["optimizer_steps"]
        and type(report.get("consumed_tokens")) is int
        and report["consumed_tokens"] > 0
        and type(report.get("consumed_sequences")) is int
        and report.get("consumed_sequences") == plan["optimizer_steps"] * 64
        and type(report.get("completed_epochs")) is int
        and report.get("completed_epochs") == plan["epochs"]
        and report.get("selection_does_not_stop_training") is True
        and report.get("original_128_token_readiness_pass_claim") is False,
        "teacher fit actual world, updates or tokens differ",
    )
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and set(checks) == EXECUTION_CHECKS
        and all(value is True for value in checks.values()),
        "teacher fit execution checks differ",
    )
    dataset = report.get("adaptation_dataset")
    require(
        isinstance(dataset, dict)
        and set(dataset) == {"teacher_fit", "teacher_dev"}
        and dataset["teacher_fit"].get("count") == plan["train_examples"]
        and dataset["teacher_dev"].get("count") == plan["dev_examples"]
        and report.get("adaptation_dataset_sha256")
        == sha256_value(dataset)
        == plan["dataset_manifest_sha256"]
        and all(
            dataset[role].get("examples_sha256") == EXPECTED_DATASETS[mode][role + "_examples_sha256"]
            for role in ("teacher_fit", "teacher_dev")
        ),
        "teacher fit dataset manifest differs",
    )
    expected_trainer = trainer_identity(mode, expected_sha, dataset["teacher_fit"].get("examples_sha256"))
    require(
        canonical(report.get("trainer_identity")) == canonical(expected_trainer)
        and canonical(report.get("teacher_readiness_thresholds")) == canonical(READINESS_THRESHOLDS),
        "teacher fit trainer model, tokenizer, data or protocol differs",
    )
    require(
        report.get("same_world_resume_performed_this_attempt") is (mode == "preflight"),
        "teacher fit must distinguish real resume testing from inherited evidence",
    )
    if mode == "full-fit":
        require(
            re.fullmatch(r"[1-9][0-9]*", str(report.get("same_world_resume_prerequisite_job_id", ""))),
            "full teacher fit lacks a concrete resume prerequisite",
        )
    else:
        require(
            report.get("selected_checkpoint_sha256") is None,
            "eight-step preflight cannot select a formal-fit teacher",
        )
    ranks = report.get("rank_summaries")
    validate_rank_memory(ranks, report.get("job_id"))
    for row in ranks:
        training = row.get("training", {})
        require(
            training.get("completed_updates") == plan["optimizer_steps"]
            and training.get("completed_samples") == plan["optimizer_steps"] * 64
            and training.get("consumed_tokens") == report["consumed_tokens"]
            and training.get("world_size") == 4
            and training.get("same_world_resume_verified") is (mode == "preflight"),
            "teacher fit rank training or actual resume evidence differs",
        )


def safe_regular(root, relative):
    require(isinstance(relative, str), "teacher fit artifact path must be text")
    name = PurePosixPath(relative)
    require(
        isinstance(relative, str)
        and relative
        and not name.is_absolute()
        and ".." not in name.parts
        and str(name) == relative
        and not any(ord(char) < 32 for char in relative),
        "unsafe teacher fit artifact path",
    )
    path = root / relative
    require(
        not any(value.is_symlink() for value in (path, *path.parents))
        and path.is_file()
        and stat.S_ISREG(path.stat().st_mode),
        "teacher fit artifact is not regular",
    )
    return path


def small_bytes(root, relative, limit=1024**2):
    path = safe_regular(root, relative)
    require(path.stat().st_size <= limit, "teacher fit metadata exceeds its bound")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "teacher fit metadata grew beyond its bound")
    return raw


def checkpoint_records(root, report):
    """Check bounded manifests and file metadata; publisher/consumer hashes weights."""
    root = Path(root)
    raw = small_bytes(root, "checkpoint-manifest.json")
    require(
        hashlib.sha256(raw).hexdigest() == report.get("checkpoint_manifest_sha256"),
        "teacher fit checkpoint manifest physical hash differs",
    )
    manifest = json.loads(raw)
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    require(
        manifest.get("sha256") == sha256_value(content) == report.get("checkpoint_set_sha256")
        and manifest.get("artifact_kind") == "teacher_adaptation_checkpoint_set"
        and manifest.get("formal_teacher_accepted") is False
        and manifest.get("inventory_scope") == "checkpoint_and_training_artifacts",
        "teacher fit checkpoint set identity differs",
    )
    for key in ("run_id", "job_id", "code_sha256", "execution_plan_sha256", "actual_plan_sha256"):
        require(
            manifest.get("identity", {}).get(key) == report.get(key),
            "teacher fit checkpoint set binding differs: " + key,
        )
    require(
        manifest.get("base_revision") == BASE_REVISION
        and manifest.get("dataset_manifest_sha256") == report.get("adaptation_dataset_sha256"),
        "teacher fit checkpoint model/data differs",
    )
    records = {}
    require(isinstance(manifest.get("files"), list), "teacher fit checkpoint inventory missing")
    for row in manifest["files"]:
        require(isinstance(row, dict) and isinstance(row.get("path"), str), "invalid fit file record")
        name = row["path"]
        require(
            name not in records
            and name not in {"teacher-fit.json", "checkpoint-manifest.json"}
            and type(row.get("size")) is int
            and row["size"] >= 0
            and re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))),
            "invalid fit file metadata",
        )
        path = safe_regular(root, name)
        require(path.stat().st_size == row["size"], "teacher fit artifact size differs")
        records[name] = row
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "teacher fit output contains a symlink")
        if not path.is_dir() and path.relative_to(root).parts[0] in {
            "checkpoints",
            "checkpoint-selection.json",
            "train-metrics.jsonl",
            "data-isolation.json",
        }:
            actual.add(path.relative_to(root).as_posix())
    require(
        set(records) == actual and sum(row["size"] for row in records.values()) <= MAX_OUTPUT_BYTES,
        "teacher fit output inventory is incomplete or exceeds budget",
    )
    for required in ("train-metrics.jsonl", "checkpoint-selection.json", "data-isolation.json"):
        value = small_bytes(root, required)
        require(
            required in records and hashlib.sha256(value).hexdigest() == records[required]["sha256"],
            "teacher fit small evidence hash differs",
        )
    checkpoints = manifest.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and [row.get("step") for row in checkpoints] == actual_plan(report["mode"])["checkpoint_steps"],
        "teacher fit scheduled checkpoints differ",
    )
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        directory = f"checkpoints/step-{step:06d}"
        validate_rank_memory(checkpoint.get("cgroup_memory_by_rank"), report.get("job_id"))
        require(
            checkpoint.get("directory") == directory
            and checkpoint.get("optimizer_steps") == step
            and type(checkpoint.get("consumed_tokens")) is int
            and checkpoint["consumed_tokens"] > 0,
            "teacher fit checkpoint step/cumulative tokens differ",
        )
        for name, key in (
            ("manifest.json", "trainer_state_manifest_sha256"),
            ("dense-manifest.json", "dense_manifest_sha256"),
            ("dev-capability.json", "dev_metrics_sha256"),
            ("train-metrics.jsonl", "training_metrics_sha256"),
        ):
            relative = directory + "/" + name
            value = small_bytes(root, relative)
            require(
                relative in records
                and hashlib.sha256(value).hexdigest() == checkpoint.get(key) == records[relative]["sha256"],
                "teacher fit per-checkpoint evidence hash differs",
            )
        state = json.loads(small_bytes(root, directory + "/manifest.json"))
        require(
            state.get("sha256")
            == sha256_value({key: value for key, value in state.items() if key != "sha256"})
            and state.get("artifact_kind") == "teacher_adapter_fit_checkpoint"
            and state.get("format_version") == 1
            and state.get("accepted_science") is False
            and state.get("formal_teacher_accepted") is False,
            "teacher fit trainer state manifest differs",
        )
        metadata = state.get("metadata", {})
        actual_plan_value = actual_plan(report["mode"])
        expected_trainer_plan = {
            "mode": "preflight" if report["mode"] == "preflight" else "fit",
            "world_size": 4,
            "train_examples": actual_plan_value["train_examples"],
            "epochs": actual_plan_value["epochs"],
            "global_batch_size": 64,
            "microbatch_size": 1,
            "seed": 271828,
            "lora_rank": 32,
            "lora_alpha": 64,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_steps": actual_plan_value["warmup_steps"],
            "checkpoint_steps": actual_plan_value["checkpoint_steps"],
        }
        require(
            canonical(metadata.get("identity")) == canonical(report.get("trainer_identity"))
            and canonical(metadata.get("plan")) == canonical(expected_trainer_plan)
            and metadata.get("plan_sha256") == sha256_value(expected_trainer_plan)
            and metadata.get("total_updates") == actual_plan_value["optimizer_steps"]
            and metadata.get("train_examples") == actual_plan_value["train_examples"]
            and metadata.get("world_size") == 4
            and metadata.get("completed_updates") == step
            and metadata.get("completed_samples") == step * 64
            and metadata.get("consumed_tokens") == checkpoint["consumed_tokens"]
            and metadata.get("train_metrics_sha256") == checkpoint["training_metrics_sha256"],
            "teacher fit trainer checkpoint progress or identity differs",
        )
        for row in state.get("files", []):
            relative = directory + "/" + row["path"]
            require(
                records.get(relative) == dict(row, path=relative), "teacher fit trainer file binding differs"
            )
        required_state = {"trainer.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
        required_state.update(f"rng-rank{rank}.pt" for rank in range(4))
        require(
            required_state <= {row["path"] for row in state.get("files", [])},
            "teacher fit checkpoint lacks four-rank optimizer/RNG/adapter state",
        )
        dense = json.loads(small_bytes(root, directory + "/dense-manifest.json"))
        require(
            dense.get("sha256")
            == sha256_value({key: value for key, value in dense.items() if key != "sha256"})
            == checkpoint.get("adapted_teacher_sha256")
            and dense.get("artifact_kind") == "adapted_dense_teacher"
            and dense.get("base_revision") == BASE_REVISION
            and dense.get("formal_teacher_accepted") is False,
            "teacher fit dense checkpoint identity differs",
        )
        dense_provenance = {
            **{key: report.get(key) for key in ("run_id", "job_id", "code_sha256")},
            "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
            "train_metrics_sha256": checkpoint["training_metrics_sha256"],
            "optimizer_steps": step,
            "consumed_tokens": checkpoint["consumed_tokens"],
        }
        require(
            dense.get("adaptation_plan_sha256") == report["actual_plan_sha256"]
            and canonical(dense.get("training_provenance")) == canonical(dense_provenance),
            "teacher fit dense checkpoint training provenance differs",
        )
        dense_files = dense.get("files", [])
        for row in dense_files:
            relative = directory + "/" + row["path"]
            require(
                records.get(relative) == dict(row, path=relative), "teacher fit dense file binding differs"
            )
        require(
            any(
                row["path"].startswith("merged/") and row["path"].endswith(".safetensors")
                for row in dense_files
            )
            and {
                "merged/config.json",
                "merged/tokenizer.json",
                "merged/tokenizer_config.json",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
            }
            <= {row["path"] for row in dense_files},
            "teacher fit checkpoint lacks merged dense weights/tokenizer",
        )
        development = json.loads(small_bytes(root, directory + "/dev-capability.json"))
        checks = development.get("checks")
        require(
            isinstance(checks, dict)
            and set(checks)
            == {
                "answer_accuracy",
                "exact_proof_accuracy",
                "first_rule_top1_accuracy",
                "intermediate_top1_accuracy",
                "topk_mass",
                "topk_target_coverage",
                "corrupted_prefix_recovery",
                "causal_shift",
            }
            and all(type(value) is bool for value in checks.values())
            and development.get("metrics_passed") is all(checks.values())
            and checkpoint.get("dev_metrics_passed") is development["metrics_passed"]
            and canonical(development.get("thresholds")) == canonical(READINESS_THRESHOLDS)
            and development.get("adapted_teacher_sha256") == checkpoint["adapted_teacher_sha256"]
            and development.get("evaluation_max_new_tokens") == 256
            and development.get("formal_teacher_accepted") is False
            and development.get("original_128_token_readiness_pass_claim") is False,
            "teacher fit checkpoint development evidence differs",
        )
    require(
        checkpoints[-1]["consumed_tokens"] == report.get("consumed_tokens"),
        "teacher fit final token count differs from last checkpoint",
    )
    selection = json.loads(small_bytes(root, "checkpoint-selection.json"))
    selected = (
        next((row for row in checkpoints if row["dev_metrics_passed"]), None)
        if report["mode"] == "full-fit"
        else None
    )
    require(
        selection.get("selection_rule") == actual_plan(report["mode"])["checkpoint_selection"]
        and selection.get("selected_checkpoint_sha256")
        == report.get("selected_checkpoint_sha256")
        == (selected["adapted_teacher_sha256"] if selected else None)
        and selection.get("selected_step") == (selected["step"] if selected else None)
        and selection.get("formal_teacher_accepted") is False
        and selection.get("original_128_token_readiness_pass_claim") is False
        and selection.get("evaluation_max_new_tokens") == 256,
        "teacher fit checkpoint selection differs from the first scheduled passing development checkpoint",
    )
    records["checkpoint-manifest.json"] = {
        "path": "checkpoint-manifest.json",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return records
