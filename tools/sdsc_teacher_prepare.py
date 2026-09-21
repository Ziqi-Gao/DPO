#!/usr/bin/env python3
"""Prepare seed-42 teacher data from an unchanged reviewed scientific checkout.

This is an independent one-H100 data task, never a G0 decision or certification
of H100 execution under the historical Blackwell contract. --validate-only
performs metadata/provenance/configuration checks without generating data or
loading a tokenizer/model. The caller owns durable publication of all outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = "prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_prompt_v3.yaml"
AMENDMENT = "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml"
OVERRIDES = (
    "g0=qwen3_v2_eap_separation",
    "experiment=canonical_sft",
    "task.num_examples=256",
    "state_source.num_candidates=8",
    "seed=42",
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
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def real_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "path must be absolute without parent traversal")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "path traverses a symlink")
    return path


def file_bytes(path, maximum=4 * 1024 * 1024):
    path = real_path(path)
    require(stat.S_ISREG(path.lstat().st_mode), "expected a regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "file exceeds audit budget")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) <= maximum and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "file changed while reading",
    )
    return raw


def git(root, *arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_NO_LAZY_FETCH="1",
    )
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "protocol.allow=never",
            "-C",
            str(root),
            *arguments,
        ],
        capture_output=True,
        env=environment,
        timeout=60,
    )
    require(result.returncode == 0, "scientific Git provenance query failed")
    return result.stdout


def validate_checkout(args):
    root = real_path(args.science_root)
    require(root.is_dir() and (root / ".git").is_dir(), "science root must be a restored real Git checkout")
    real_path(root / ".git")
    raw = file_bytes(args.provenance_manifest)
    require(digest(raw) == args.provenance_manifest_sha256, "provenance manifest SHA mismatch")
    provenance = json.loads(raw)
    require(provenance.get("schema") == "quest-sdsc-git-provenance-v1", "wrong provenance schema")
    require(provenance["wrapper"]["code_sha256"] == args.code_sha256, "wrapper snapshot binding differs")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    require(head == provenance["git_head"], "science HEAD differs from bundled provenance")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all"), "science checkout is dirty")
    require(not git(root, "ls-files", "--others", "-z"), "science checkout contains ignored/untracked files")
    records = provenance.get("scientific_files")
    require(isinstance(records, list) and records, "missing scientific byte inventory")
    for record in records:
        relative = Path(record["path"])
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parts[0] not in {".git", ".opd-git", ".ssh", ".codex"},
            "unsafe scientific path",
        )
        path = root / relative
        content = file_bytes(path)
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        require(
            (len(content), digest(content), mode) == (record["size"], record["sha256"], record["mode"]),
            "scientific file bytes differ",
        )
    return {
        "science_git_head": head,
        "bundle_sha256": provenance["bundle"]["sha256"],
        "provenance_manifest_sha256": args.provenance_manifest_sha256,
    }


def runtime_identity():
    require(platform.python_version() == "3.12.13", "teacher task requires Python 3.12.13")
    packages = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    require(packages == DEPENDENCIES, "teacher runtime dependency versions differ")
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages}


def scientific_api(root):
    require(
        not any(
            name == "posttrain_circuits" or name.startswith("posttrain_circuits.") for name in sys.modules
        ),
        "scientific modules were imported before checkout validation",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    config = importlib.import_module("posttrain_circuits.core.config")
    protocols = importlib.import_module("posttrain_circuits.artifacts.execution_science_protocol")
    runs = importlib.import_module("posttrain_circuits.artifacts.runs")
    return SimpleNamespace(
        compose=config.compose_config,
        resolve=protocols.resolve_accepted_execution_science_protocol,
        config_sha=protocols.canonical_science_config_sha256,
        binding=runs.formal_artifact_binding,
    )


def validate_binding(args, api, provenance):
    resolved = api.resolve(
        code_root=args.science_root, configured_path=PROTOCOL, expected_head=provenance["science_git_head"]
    )
    require(tuple(resolved.binding.hydra_override_vector) == OVERRIDES, "reviewed science overrides differ")
    overrides = [*OVERRIDES, f"protocol_amendment_path={AMENDMENT}"]
    for key, path in (
        ("output_root", args.output_dir),
        ("task.dataset_family_path", args.output_dir / "dataset"),
        ("state_source.store_path", args.output_dir / "teacher_demos"),
    ):
        overrides.append(key + "=" + json.dumps(str(path)))
    config = api.compose(overrides, config_root=args.science_root / "configs")
    science_hash = api.config_sha(config)
    require(
        science_hash == resolved.binding.storage_neutral_resolved_config_sha256,
        "resolved science configuration differs from the accepted prompt-v3 protocol",
    )
    require(
        config["seed"] == 42
        and config["task"]["num_examples"] == 256
        and config["teacher"]["generation_seed"] == 31415
        and config["state_source"]["num_candidates"] == 8
        and config["state_source"]["max_prompt_tokens"] == 1246
        and config["state_source"]["max_new_tokens"] == 256,
        "teacher population, seeds, or generation envelope changed",
    )
    require(
        sum(config["task"]["split_sizes"].values()) == 144000,
        "the complete frozen 144000-example dataset family is required",
    )
    binding = api.binding(config)
    require(binding["code_commit"] == provenance["science_git_head"], "formal artifact HEAD differs")
    return (
        config,
        overrides,
        binding,
        {
            "path": PROTOCOL,
            "sha256": resolved.sha256,
            "acceptance_commit": resolved.acceptance_commit,
            "reviewed_implementation_commit": resolved.reviewed_implementation_commit,
            "storage_neutral_resolved_config_sha256": science_hash,
            "protocol_id": resolved.binding.protocol_id,
        },
    )


def allocation(args):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job), "actual Slurm job ID is required")
    require(args.job_id in (None, job), "supplied job ID differs from the allocation")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "24", "teacher task requires 24 allocated CPUs")
    require(os.environ.get("SLURM_MEM_PER_NODE") == "196608", "teacher task requires 192 GiB host memory")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(
        visible and "," not in visible and visible.strip() == visible,
        "teacher task requires exactly one assigned visible GPU",
    )
    require(os.environ.get("WORLD_SIZE", "1") == "1", "teacher generation is a single-process task")
    require(os.environ.get("LOCAL_RANK", "0") == "0", "teacher generation uses logical GPU zero")
    return {"job_id": job, "cpus": 24, "memory_mib": 196608, "cuda_visible_devices": visible}


def local_environment(args):
    work, output, home = map(real_path, (args.work_dir, args.output_dir, args.hf_home))
    require(work.is_dir() and work.stat().st_uid == os.getuid(), "work directory is missing or not owned")
    require(work != Path.home().resolve() and Path.home().resolve() not in work.parents, "work is in HOME")
    require(
        work in output.parents and work in home.parents and home.is_dir(),
        "outputs and staged HF_HOME must be inside node-local work",
    )
    require(
        args.science_root not in output.parents and output != args.science_root,
        "output must not modify the scientific checkout",
    )
    mount = json.loads(
        subprocess.run(
            ["findmnt", "--json", "--target", str(work), "--output", "TARGET,SOURCE,FSTYPE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    )["filesystems"][0]
    require(mount["fstype"] in {"ext4", "xfs", "ext3", "ext2", "btrfs"}, "work is not node-local disk")
    # Reuse the stdlib-only cgroup/cache checks, never the two-GPU training allocation check.
    spec = importlib.util.spec_from_file_location(
        "_sdsc_training_guards", Path(__file__).with_name("sdsc_training_preflight.py")
    )
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    memory = guards.memory_envelope()
    cache = guards.pinned_cache(home)
    os.environ.update(
        HF_HOME=str(home),
        HF_HUB_CACHE=str(home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="24",
        MKL_NUM_THREADS="24",
        TOKENIZERS_PARALLELISM="false",
    )
    return {"node_local_mount": mount, "cgroup_memory": memory, "pinned_cache": cache}


def final_usage():
    import torch

    spec = importlib.util.spec_from_file_location(
        "_sdsc_teacher_final_memory", Path(__file__).with_name("sdsc_training_preflight.py")
    )
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    torch.cuda.synchronize(0)
    return {
        "cgroup_memory": guards.memory_envelope(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def gpu_identity():
    import torch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible CUDA GPU required")
    properties = torch.cuda.get_device_properties(0)
    require(
        "H100" in properties.name
        and (properties.major, properties.minor) == (9, 0)
        and properties.total_memory >= 75 * 1024**3,
        "teacher task requires an H100 80GB",
    )
    require(torch.version.cuda == "12.8", "teacher CUDA runtime differs")
    torch.set_num_threads(24)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [9, 0],
        "logical_device": 0,
    }


def prepare_dataset(args, overrides, config):
    build = importlib.import_module("posttrain_circuits.cli.build_splits")
    family_api = importlib.import_module("posttrain_circuits.datasets.proofgraph.family")
    dataset = args.output_dir / "dataset"
    require(not dataset.exists(), "dataset output was already used")
    build.main([*overrides, "--output", str(dataset), "--confirm-production"])
    family = family_api.load_dataset_family(dataset)
    expected_sizes = config["task"]["split_sizes"]
    require(
        all(len(family.examples(split)) == size for split, size in expected_sizes.items()),
        "generated family split population differs",
    )
    examples = family.examples("train")[:256]
    ids = [example.example_id for example in examples]
    require(len(ids) == len(set(ids)) == 256, "teacher prompt population is not exactly 256 unique IDs")
    return (
        family,
        examples,
        {
            "dataset_family_sha256": family.manifest["sha256"],
            "train_examples_file_sha256": family.boundary("train")["examples_file_sha256"],
            "ordered_prompt_ids": ids,
            "split_sizes": expected_sizes,
        },
    )


def validate_prompts(config, examples):
    from transformers import AutoTokenizer

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.loading import tokenizer_fingerprint
    from posttrain_circuits.models.prompt_protocol import chat_template_sha256, format_model_prompt

    teacher = config["teacher"]
    tokenizer = AutoTokenizer.from_pretrained(
        teacher["tokenizer_name_or_path"],
        revision=teacher["tokenizer_revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        require(tokenizer.eos_token_id is not None, "tokenizer has no pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    fingerprint = tokenizer_fingerprint(tokenizer)
    template = chat_template_sha256(tokenizer)
    require(
        fingerprint == teacher["tokenizer_fingerprint"] == config["model"]["tokenizer_fingerprint"],
        "teacher/student tokenizer fingerprint differs",
    )
    require(template == teacher["prompt_protocol"]["chat_template_sha256"], "chat template differs")
    task = ProofGraphTask()
    lengths = []
    prompt_hash = hashlib.sha256()
    for example in examples:
        prompt = format_model_prompt(task.render(example), tokenizer, teacher).model_facing_prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        require(0 < len(tokens) <= 1246, "teacher prompt exceeds reviewed token bound")
        prompt_hash.update(json.dumps([example.example_id, tokens], separators=(",", ":")).encode() + b"\n")
        lengths.append(len(tokens))
    return {
        "tokenizer_fingerprint": fingerprint,
        "chat_template_sha256": template,
        "tokenized_prompt_audit_sha256": prompt_hash.hexdigest(),
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
    }


def generate_teacher(args, overrides):
    module = importlib.import_module("posttrain_circuits.cli.build_teacher_demos")
    module.main(
        [*overrides, "--output", str(args.output_dir / "teacher_demos"), "--confirm-production"],
        diagnostics_with_output=True,
    )


def validate_store(args, formal_binding, dataset, prompt):
    from posttrain_circuits.datasets.teacher_demos.store import read_teacher_demo_store

    store = args.output_dir / "teacher_demos"
    accepted, manifest = read_teacher_demo_store(store, require_formal=True)
    expected = {
        **formal_binding,
        "dataset_family_sha256": dataset["dataset_family_sha256"],
        "train_examples_file_sha256": dataset["train_examples_file_sha256"],
    }
    require(manifest["protocol_bindings"] == expected, "teacher store formal/dataset bindings differ")
    require(
        manifest["ordered_prompt_ids"] == dataset["ordered_prompt_ids"]
        and manifest["attempt_count"] == 2048
        and manifest["tokenizer_hash"] == prompt["tokenizer_fingerprint"],
        "teacher store population or tokenizer differs",
    )
    generation = manifest["teacher_demo_generation"]
    require(
        generation["sampling_request_seed"] == 31415
        and generation["candidates_per_prompt"] == 8
        and generation["max_prompt_tokens"] == 1246
        and generation["max_new_tokens"] == 256
        and generation["temperature"] == 0.7
        and generation["top_p"] == 0.8
        and generation["top_k"] == 20
        and generation["min_p"] == 0.0,
        "teacher generation identity changed",
    )
    require(
        manifest["behavior_policy"]
        == {
            "id": "Qwen/Qwen3-8B",
            "revision": formal_binding["teacher_revision"],
            "resolved_commit": formal_binding["teacher_revision"],
        },
        "teacher policy revision differs from the reviewed binding",
    )
    return {
        "manifest_sha256": digest(file_bytes(store / "manifest.json")),
        "semantic_sha256": manifest["sha256"],
        "ledger_sha256": manifest["ledger_file_sha256"],
        "accepted_view_sha256": manifest["accepted_view_file_sha256"],
        "accepted_count": len(accepted),
        "attempt_count": manifest["attempt_count"],
        "ready_for_formal_sft": True,
    }


def publish(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "work-dir", "output-dir", "hf-home", "provenance-manifest"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--provenance-manifest-sha256", "--provenance-sha256", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--code-sha256", required=True)
    result.add_argument("--job-id")
    result.add_argument("--validate-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run ID")
    require(
        SHA256.fullmatch(args.code_sha256) and SHA256.fullmatch(args.provenance_manifest_sha256),
        "invalid provenance/snapshot SHA",
    )
    for name in ("science_root", "work_dir", "output_dir", "hf_home", "provenance_manifest"):
        real_path(getattr(args, name))
    require(
        args.work_dir in args.output_dir.parents
        and args.science_root not in args.output_dir.parents
        and args.output_dir != args.science_root,
        "output must be outside science and below work",
    )
    runtime = runtime_identity()
    provenance = validate_checkout(args)
    previous_directory = Path.cwd()
    previous_signals = {}
    output_owned = False
    started = time.monotonic()
    report = {
        "kind": "sdsc_teacher_prepare_v1",
        "task": "qwen3-v2-teacher-prepare",
        "passed": False,
        "exit_code": 1,
        "g0_passed": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate_for_execution": False,
        "partial_attempt_ledger_guaranteed": False,
        "persistent_storage_publication": "caller_required",
        "run_id": args.run_id,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code_sha256": args.code_sha256,
        **provenance,
        "runtime": runtime,
    }

    def interrupted(number, _frame):
        raise InterruptedError("teacher task interrupted by signal " + str(number))

    try:
        os.chdir(args.science_root)
        api = scientific_api(args.science_root)
        config, overrides, binding, reviewed = validate_binding(args, api, provenance)
        report.update(formal_binding=binding, accepted_science=reviewed)
        if args.validate_only:
            report.update(
                validation_only=True,
                binding_validated=True,
                model_loaded=False,
                dataset_generated=False,
                exit_code=0,
            )
            print(json.dumps(report, sort_keys=True))
            return 0
        report["allocation"] = allocation(args)
        report.update(local_environment(args))
        require(not args.output_dir.exists(), "teacher output directory was already used")
        args.output_dir.mkdir(parents=True)
        output_owned = True
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_signals[number] = signal.signal(number, interrupted)
        publish(args.output_dir / "teacher-prepare-start.json", report)
        report["gpu"] = gpu_identity()
        _family, examples, dataset = prepare_dataset(args, overrides, config)
        report["dataset"] = dataset
        report["prompt_preflight"] = validate_prompts(config, examples)
        require(api.binding(config) == binding, "scientific binding changed before model generation")
        generate_teacher(args, overrides)
        require(api.binding(config) == binding, "scientific binding changed during generation")
        require(validate_checkout(args) == provenance, "scientific checkout changed during generation")
        report["teacher_store"] = validate_store(args, binding, dataset, report["prompt_preflight"])
        report["resource_usage"] = final_usage()
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == report["allocation"]["cuda_visible_devices"],
            "assigned CUDA visibility changed",
        )
        report.update(passed=True, exit_code=0, elapsed_seconds=time.monotonic() - started)
        publish(args.output_dir / "teacher-prepare.json", report)
        print(json.dumps({"passed": True, "g0_passed": False, "run_id": args.run_id}, sort_keys=True))
        return 0
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        if output_owned and args.output_dir.is_dir():
            report["ledger_written"] = (args.output_dir / "teacher_demos/ledger.json").is_file()
            with contextlib.suppress(OSError):
                publish(args.output_dir / "teacher-prepare.json", report)
        raise
    finally:
        os.chdir(previous_directory)
        for number, handler in previous_signals.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
