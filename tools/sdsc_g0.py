#!/usr/bin/env python3
"""Finish the accepted G0 science in the private calibration-path namespace.

The separate host wrapper first performs two real checkpoint continuations with
sdsc_resume.py. This worker retains original CLI scientific arguments and runs
the explicitly reviewed SDSC completion adapter; no scheduler certificate is
manufactured and no reference training is repeated.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdsc_pipeline_storage as store


def stage_plan(root, overrides, final_checkpoint):
    """Original handler stage order, omitting already verified producer stages."""
    dataset = root / "dataset"
    initial = root / "initial_checkpoint.pt"
    probes = root / "probes"
    scores = root / "probe_scores.json"
    circuits = root / "circuits"
    common = list(overrides)
    offline = [("experiment=offline_soft" if v.startswith("experiment=") else v) for v in common]
    result = [
        (
            "audit_label_leakage",
            [
                *common,
                "--split-root",
                str(dataset),
                "--split",
                "validation",
                "--output",
                str(root / "label_leakage.json"),
            ],
        ),
        (
            "evaluate_teacher_readiness",
            [
                *common,
                "--validation-split",
                str(dataset / "validation"),
                "--limit",
                "128",
                "--top-k",
                "128",
                "--output",
                str(root / "teacher_readiness.json"),
                "--confirm-production",
            ],
        ),
        (
            "score_probe_candidates",
            [
                *common,
                "--initial-checkpoint",
                str(initial),
                "--calibration-checkpoint",
                str(final_checkpoint),
                "--calibration-run-manifest",
                str(root / "canonical_sft/manifest.json"),
                "--world-size",
                "2",
                "--dataset-family",
                str(dataset),
                "--probe-limit-per-split",
                "128",
                "--task-validation-limit",
                "128",
                "--output",
                str(scores),
            ],
        ),
        (
            "build_probe_cohorts",
            [
                "--dataset-family",
                str(dataset),
                "--scores",
                str(scores),
                "--limit-per-split",
                "128",
                "--output",
                str(probes),
            ],
        ),
        ("build_rollout_bank", [*common, "--output", str(root / "rollout_bank"), "--confirm-production"]),
        (
            "score_teacher",
            [
                *offline,
                "--bank",
                str(root / "rollout_bank"),
                "--output",
                str(root / "teacher_scores"),
                "--confirm-production",
            ],
        ),
        (
            "evaluate_anti_shortcut",
            [*common, "--output", str(root / "anti_shortcut.json"), "--confirm-production"],
        ),
    ]
    for stage in ("final_answer", "first_rule_selection"):
        folder = circuits / stage
        args = [
            *common,
            "--checkpoint",
            str(initial),
            "--initial-checkpoint",
            str(initial),
            "--probe-cohort-manifest",
            str(probes / "manifest.json"),
            "--cohort",
            "base_capable",
        ]
        result.extend(
            [
                (
                    "discover_circuit",
                    [
                        *args,
                        "--stage",
                        stage,
                        "--output",
                        str(folder / "circuit.json"),
                        "--confirm-production",
                    ],
                ),
                (
                    "evaluate_circuit",
                    [
                        *args,
                        "--circuit-artifact",
                        str(folder / "circuit.json"),
                        "--output",
                        str(folder / "exact_patching.json"),
                        "--confirm-production",
                    ],
                ),
            ]
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    request = store.document(args.request, args.request_sha256)
    root = store.safe(request["g0_root"])
    science = store.safe(request["science_root"])
    report = store.document(request["reference_report"], request["reference_report_sha256"])
    store.require(
        report.get("passed") is True and report.get("training_executed") is True,
        "reference calibration failed",
    )
    store.require(
        str(root) == str(Path(report["training_artifact_root"]).parent), "original calibration path changed"
    )
    os.chdir(science)
    sys.path.insert(0, str(science / "src"))
    binding = store.helper("sdsc_science_binding")
    base, projection, accepted = binding.reviewed_science(
        science, request["science_head"], request["protocol_sha256"]
    )
    compose = importlib.import_module("posttrain_circuits.core.config").compose_config
    config = compose(report["plan"]["training_overrides"], config_root=science / "configs")
    store.require(
        store.sha(store.canonical(config)) == report["runtime_config_sha256"],
        "actual calibration config differs",
    )
    binding.verify_runtime_binding(
        base,
        config,
        expected_base_science_sha256=accepted.binding.storage_neutral_resolved_config_sha256,
        expected_checkpoint_sha256=report["initial_checkpoint"]["sha256"],
        checkpoint_path=root / "initial_checkpoint.pt",
        science_projection=projection,
    )
    calibration = store.helper("sdsc_g0_calibration")
    guards = store.helper("sdsc_training_preflight")
    calibration.allocation_identity()
    calibration.gpu_identity()
    guards.runtime_identity()
    guards.pinned_cache(Path(request["hf_home"]))
    store.require(
        (root / "distributed_resume.json").is_file(), "real full-state resume comparison is required"
    )
    runtime_config = root / "sdsc-runtime-config.json"
    store.atomic(runtime_config, config)
    store.copy(science / binding.PROTOCOL, root / "execution_science_protocol.yaml")
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.update(
        PYTHONPATH=str(science / "src"),
        MIB_REPOSITORY=request["mib_repository"],
        HF_HOME=request["hf_home"],
        HF_HUB_CACHE=str(Path(request["hf_home"]) / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
    )
    steps = []
    manifest = store.document(root / "canonical_sft/manifest.json")
    final_checkpoint = store.safe(manifest["final_checkpoint_path"])
    store.require(
        root / "canonical_sft" in final_checkpoint.parents
        and store.identity(final_checkpoint)["sha256"] == manifest["final_checkpoint_sha256"],
        "reference final checkpoint path/hash differs",
    )
    for index, (module, arguments) in enumerate(
        stage_plan(root, report["plan"]["training_overrides"], final_checkpoint)
    ):
        command = [request["python"], "-B", "-m", "posttrain_circuits.cli." + module, *arguments]
        print(store.canonical({"phase": module, "index": index, "started": time.time()}).decode(), flush=True)
        before = time.monotonic()
        with (root / f"sdsc-stage-{index:02d}-{module}.log").open("xb") as log:
            # Inherit the outer process group so its timeout handler can stop every descendant.
            result = subprocess.run(
                command,
                cwd=science,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        steps.append(
            {
                "module": module,
                "argv": command,
                "seconds": time.monotonic() - before,
                "exit_code": result.returncode,
            }
        )
        store.atomic(root / "sdsc-g0-progress.json", {"steps": steps, "complete": False}, replace=True)
        store.require(result.returncode == 0, "scientific stage failed: " + module)
    import torch

    lock = store.document(science / "deployments/qwen3_v2_g0/dependency-lock.json")
    runtime = {
        "python": platform.python_version(),
        "packages": {p["name"]: importlib.metadata.version(p["name"]) for p in lock["packages"]},
    }
    evidence = {
        "schema": "quest-sdsc-g0-execution-v1",
        "target": {**request["target"], "science_git_head": request["science_head"]},
        "allocation": {
            "account": "nwu181",
            "partition": "nairr-gpu-shared",
            "qos": "nairr-gpu-shared-normal",
            "nodes": 1,
            "gpus": 2,
            "gpu_type": "h100",
            "cpus": 24,
            "mem_gib": 192,
            "world_size": 2,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "runtime": runtime,
        "gpus": [
            {
                "logical_device": i,
                "name": torch.cuda.get_device_properties(i).name,
                "compute_capability": list(torch.cuda.get_device_capability(i)),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(2)
        ],
        "cgroup_memory": guards.memory_envelope(),
        "adapter_review": request["adapter_review"],
        "provenance": {"path": "provenance-manifest.json", "sha256": request["provenance_sha256"]},
        "preflight": {
            "path": "preflight.json",
            "sha256": request["preflight"]["report_sha256"],
            "source_manifest_path": "preflight-source-manifest.json",
            "source_manifest_sha256": request["preflight"]["source_manifest_sha256"],
            "status": request["preflight"]["status"],
        },
    }
    evidence_sha = store.atomic(root / "sdsc-execution.json", evidence)
    command = [
        request["python"],
        "-I",
        "-B",
        str(Path(__file__).with_name("sdsc_finalize_g0.py")),
        "--science-root",
        str(science),
        "--workspace",
        str(root),
        "--runtime-config",
        str(runtime_config),
        "--execution-evidence",
        str(root / "sdsc-execution.json"),
        "--execution-evidence-sha256",
        evidence_sha,
        "--expected-science-head",
        request["science_head"],
        "--expected-protocol-sha256",
        request["protocol_sha256"],
        "--expected-initial-sha256",
        report["initial_checkpoint"]["sha256"],
        "--expected-adapter-sha256",
        request["adapter_review"]["files"]["sdsc_finalize_g0.py"],
    ]
    subprocess.run(command, cwd=science, env=environment, check=True)
    subprocess.run([*command, "--replay", str(root / "g0.json")], cwd=science, env=environment, check=True)
    final = store.document(root / "g0.json")
    store.require(final.get("passed") is True, "full G0 scientific decision failed")
    store.atomic(
        root / "sdsc-g0-progress.json",
        {"steps": steps, "complete": True, "g0_sha256": store.sha(store.read(root / "g0.json"))},
        replace=True,
    )


if __name__ == "__main__":
    main()
