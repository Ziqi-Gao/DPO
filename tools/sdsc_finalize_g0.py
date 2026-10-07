#!/usr/bin/env python3
"""SDSC single-invocation G0 scientific completion and exact replay.

This explicitly new backend adapter keeps the real accepted scientific checkout.
It uses original domain validators and context-free finalizer helpers; it neither
constructs the historical ServerScheduler invocation context nor uses the
Blackwell execution-class certificate. All 26 scientific checks are retained;
the two historical backend/certificate checks are replaced by named, externally
hash-bound H100 invocation and adapter-review checks. It submits no jobs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "quest-sdsc-g0-v1"
EXECUTION_SCHEMA = "quest-sdsc-g0-execution-v1"
SCIENCE_HEAD = "0215c356355b29b5e2b407978a207db2156719e1"
PROTOCOL_SHA256 = "752fa685d795335527c639fb2b7f6cc3e94aa60ffb9a16d4329bf13099b0888e"
BACKEND_CHECKS = {"sdsc_single_invocation_execution", "sdsc_adapter_identity"}
OLD_BACKEND_CHECKS = {"execution_safety_descriptor", "execution_safety_certification"}
REQUIRED_ADAPTERS = {
    "sdsc_finalize_g0.py",
    "sdsc_science_binding.py",
    "sdsc_resume.py",
    "sdsc_g0_calibration.py",
    "sdsc_provenance.py",
    "sdsc_training_preflight.py",
    "sdsc_teacher_prepare.py",
}
REVIEWABLE_ADAPTERS = REQUIRED_ADAPTERS | {
    "sdsc_g0.py",
    "sdsc_pipeline_job.py",
    "sdsc_pipeline_storage.py",
    "sdsc_pipeline_inputs.py",
    "sdsc_pipeline_remote.py",
    "sdsc_pipeline.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper(name):
    spec = importlib.util.spec_from_file_location("_g0_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binding = helper("sdsc_science_binding")


def canonical(value):
    return binding.canonical(value)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def checked_json(path, expected=None):
    value, identity = binding.json_document(path)
    require(expected is None or identity["sha256"] == expected, "external JSON SHA-256 differs")
    return value, identity


def contained(workspace, relative):
    require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        "artifact locator must be workspace-relative",
    )
    require(
        ".." not in Path(relative).parts and str(Path(relative)) == relative,
        "artifact locator is not canonical",
    )
    path = binding.real_path(workspace / relative)
    require(workspace in path.parents, "artifact escaped workspace")
    return path


def validate_adapter_review(review, expected_adapter_sha256):
    require(
        isinstance(review, dict) and review.get("scope") == "single_invocation_h100_g0",
        "adapter review has the wrong explicit backend scope",
    )
    require(
        isinstance(review.get("reviewer"), str) and review["reviewer"].strip(),
        "independent adapter reviewer must be identified",
    )
    files = review.get("files")
    require(
        isinstance(files, dict) and REQUIRED_ADAPTERS <= set(files) <= REVIEWABLE_ADAPTERS,
        "adapter review omits required files or adds unsupported code",
    )
    require(
        binding.SHA256.fullmatch(expected_adapter_sha256 or "") is not None,
        "external reviewed adapter SHA is required",
    )
    require(
        files["sdsc_finalize_g0.py"] == expected_adapter_sha256,
        "adapter review differs from externally selected adapter",
    )
    for name, expected in files.items():
        require(binding.SHA256.fullmatch(expected or "") is not None, "adapter file hash is invalid")
        require(
            binding.file_identity(Path(__file__).with_name(name))["sha256"] == expected,
            "reviewed adapter bytes differ: " + name,
        )
    return files


def validate_execution(evidence, workspace, config, science_root, expected_adapter_sha256, api):
    require(evidence.get("schema") == EXECUTION_SCHEMA, "wrong SDSC execution evidence schema")
    target = evidence.get("target", {})
    require(target.get("science_git_head") == SCIENCE_HEAD, "SDSC execution used a different scientific HEAD")
    require(
        re.fullmatch(r"[1-9][0-9]*", str(target.get("job_id", ""))) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(target.get("run_id", "")))
        and binding.SHA256.fullmatch(target.get("code_sha256", "")),
        "SDSC invocation identity is incomplete",
    )
    allocation = evidence.get("allocation", {})
    expected = {
        "account": "nwu181",
        "partition": "nairr-gpu-shared",
        "qos": "nairr-gpu-shared-normal",
        "nodes": 1,
        "gpus": 2,
        "gpu_type": "h100",
        "cpus": 24,
        "mem_gib": 192,
        "world_size": 2,
    }
    require(
        canonical({key: allocation.get(key) for key in expected}) == canonical(expected),
        "SDSC allocation differs from the reviewed two-H100 invocation",
    )
    visible = allocation.get("cuda_visible_devices", "")
    require(
        isinstance(visible, str)
        and len(visible.split(",")) == 2
        and len(set(visible.split(","))) == 2
        and all(value and value.strip() == value for value in visible.split(",")),
        "CUDA visibility is not the exact two-device allocation",
    )
    calibration = helper("sdsc_g0_calibration")
    lock, _ = checked_json(science_root / "deployments/qwen3_v2_g0/dependency-lock.json")
    packages = {item["name"]: item["version"] for item in lock["packages"]}
    runtime = evidence.get("runtime", {})
    require(
        runtime.get("python") == "3.12.13" and canonical(runtime.get("packages")) == canonical(packages),
        "SDSC runtime differs from the pinned scientific runtime",
    )
    gpus = evidence.get("gpus")
    require(isinstance(gpus, list) and len(gpus) == 2, "both observed H100 devices are required")
    for rank, gpu in enumerate(gpus):
        require(
            type(gpu.get("logical_device")) is int
            and gpu["logical_device"] == rank
            and "H100" in gpu.get("name", "")
            and gpu.get("compute_capability") == [9, 0]
            and type(gpu.get("total_memory_bytes")) is int
            and gpu["total_memory_bytes"] >= 75 * 1024**3,
            "actual GPU observation is not the reviewed H100 allocation",
        )
    calibration.validate_memory(evidence.get("cgroup_memory", {}))
    review = validate_adapter_review(evidence.get("adapter_review"), expected_adapter_sha256)
    provenance_ref, preflight_ref = evidence.get("provenance", {}), evidence.get("preflight", {})
    require(
        provenance_ref.get("path") == "provenance-manifest.json"
        and preflight_ref.get("path") == "preflight.json"
        and preflight_ref.get("source_manifest_path") == "preflight-source-manifest.json",
        "execution proof locators differ from fixed names",
    )
    for expected_sha in (
        provenance_ref.get("sha256"),
        preflight_ref.get("sha256"),
        preflight_ref.get("source_manifest_sha256"),
    ):
        require(binding.SHA256.fullmatch(expected_sha or ""), "execution proof hash is missing")
    provenance, _ = checked_json(contained(workspace, provenance_ref["path"]), provenance_ref["sha256"])
    preflight, _ = checked_json(contained(workspace, preflight_ref["path"]), preflight_ref["sha256"])
    snapshot, _ = checked_json(
        contained(workspace, preflight_ref["source_manifest_path"]), preflight_ref["source_manifest_sha256"]
    )
    require(
        provenance.get("git_head") == SCIENCE_HEAD
        and provenance.get("wrapper", {}).get("run_id") == target["run_id"]
        and provenance.get("wrapper", {}).get("code_sha256") == target["code_sha256"],
        "provenance does not bind this invocation and wrapper",
    )
    provenance_api = helper("sdsc_provenance")
    records, _ = provenance_api.tree_records(science_root, SCIENCE_HEAD)
    require(
        canonical(provenance.get("scientific_files"))
        == canonical([record for record in records if provenance_api.is_science(record["path"])]),
        "provenance scientific bytes differ from the actual clean checkout",
    )
    calibration.validate_snapshot(snapshot, preflight, provenance)
    calibration.validate_preflight(preflight, api._batch_token_contract(config, 2))
    validate_preflight_status(preflight_ref.get("status", {}), preflight, preflight_ref["sha256"])
    require(
        str(preflight["job_id"]) != str(target["job_id"]),
        "preflight must be a separately completed invocation",
    )
    return {
        "target": target,
        "allocation": allocation,
        "reviewed_adapter_files": review,
        "preflight_job_id": preflight["job_id"],
        "provenance_sha256": provenance_ref["sha256"],
        "preflight_sha256": preflight_ref["sha256"],
    }


def validate_preflight_status(status, report, report_sha256):
    require(
        status.get("success") is True
        and status.get("state") == "COMPLETED"
        and status.get("job_id") == report["job_id"]
        and status.get("run_id") == report["run_id"]
        and status.get("result", {}).get("verified") is True
        and status.get("result", {}).get("result_sha256") == report_sha256,
        "preflight semantic terminal status differs",
    )
    queue, accounting = status.get("queue", {}), status.get("accounting", {})
    require(
        queue.get("returncode") == 0
        and not queue.get("stdout", "").strip()
        and accounting.get("returncode") == 0,
        "preflight status query is inconclusive",
    )
    rows = [line.strip().split("|") for line in accounting.get("stdout", "").splitlines() if line.strip()]
    job = str(report["job_id"])
    exact = [row for row in rows if row[0] == job]
    related = [row for row in rows if row[0] == job or row[0].startswith(job + ".")]
    require(
        len(exact) == 1 and all(len(row) >= 3 and row[1:3] == ["COMPLETED", "0:0"] for row in related),
        "preflight or one of its Slurm steps did not complete successfully",
    )


def scientific_decision(workspace, config, formal_binding, api):
    """The same reconstructed scientific decision is used by live and replay."""
    paths = {
        "initial_checkpoint": workspace / "initial_checkpoint.pt",
        "execution_science_protocol": workspace / "execution_science_protocol.yaml",
        "base_scores": workspace / "probe_scores.json",
        "teacher_store_manifest": workspace / "teacher_scores" / "manifest.json",
        "teacher_readiness": workspace / "teacher_readiness.json",
        "label_leakage": workspace / "label_leakage.json",
        "anti_shortcut": workspace / "anti_shortcut.json",
        "probe_manifest": workspace / "probes" / "manifest.json",
        "final_circuit": workspace / "circuits" / "final_answer" / "circuit.json",
        "final_exact_patching": workspace / "circuits" / "final_answer" / "exact_patching.json",
        "process_circuit": workspace / "circuits" / "first_rule_selection" / "circuit.json",
        "process_exact_patching": workspace / "circuits" / "first_rule_selection" / "exact_patching.json",
        "final_compatibility": workspace / "circuits" / "final_answer" / "mib_raw" / "compatibility.json",
        "process_compatibility": workspace
        / "circuits"
        / "first_rule_selection"
        / "mib_raw"
        / "compatibility.json",
        "distributed_resume": workspace / "distributed_resume.json",
    }
    for path in paths.values():
        if path.suffix == ".json":
            checked_json(path)
    prereg_version = str(config["prereg_version"])
    base = api.validate_probe_score_artifact(paths["base_scores"])
    teacher = api.TrajectoryStore(paths["teacher_store_manifest"].parent).check_integrity()
    api.require_scientific_artifact(teacher, expected_prereg_version=prereg_version)
    initial_checkpoint_hash = api.sha256_file(paths["initial_checkpoint"])
    label_leakage = api.validate_label_leakage_artifact(api._read(paths["label_leakage"]))
    teacher_readiness_raw = api._read(paths["teacher_readiness"])
    tokenized_prefix = teacher_readiness_raw.get("tokenized_prefix_manifest")
    if not isinstance(tokenized_prefix, dict) or tokenized_prefix.get("sha256") != api.sha256_value(
        {key: value for key, value in tokenized_prefix.items() if key != "sha256"}
    ):
        raise ValueError("teacher-readiness tokenized prefix manifest is invalid")
    readiness_bindings = {
        **formal_binding,
        "teacher_model_id": str(config["teacher"]["model_name_or_path"]),
        "teacher_model_revision": str(config["teacher"]["model_revision"]),
        "student_model_id": str(config["model"]["model_name_or_path"]),
        "student_model_revision": str(config["model"]["model_revision"]),
        "student_tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "teacher_tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "dataset_hash": str(label_leakage["dataset_hash"]),
        "prefix_probe_hash": str(tokenized_prefix["sha256"]),
    }
    readiness_bindings.pop("teacher_revision")
    readiness_bindings.pop("model_revision")
    teacher_readiness = api.validate_teacher_readiness_artifact(
        teacher_readiness_raw,
        expected_bindings=readiness_bindings,
    )
    anti = api.validate_anti_shortcut_report(
        paths["anti_shortcut"],
        max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
        expected_model_checkpoint_hash=initial_checkpoint_hash,
    )
    probes = api.validate_probe_cohort_manifest(
        paths["probe_manifest"],
        expected_initial_checkpoint_hash=initial_checkpoint_hash,
    )
    final_circuit = api._read(paths["final_circuit"])
    final_exact = api._read(paths["final_exact_patching"])
    process_circuit = api._read(paths["process_circuit"])
    process_exact = api._read(paths["process_exact_patching"])
    for artifact in (final_circuit, final_exact, process_circuit, process_exact):
        api.require_scientific_artifact(
            artifact,
            expected_prereg_version=prereg_version,
            require_circuit_schema=True,
            require_hash=True,
        )
    final_compatibility = api._read(paths["final_compatibility"])
    process_compatibility = api._read(paths["process_compatibility"])
    resume = helper("sdsc_resume").validate_resume_report(
        paths["distributed_resume"],
        config=config,
        expected_world_size=2,
        expected_formal_binding=formal_binding,
    )
    teacher_readiness_formal = api._teacher_readiness_formal_binding(teacher_readiness)
    bound_artifacts = {
        "base_scores": base,
        "teacher_store": teacher,
        "teacher_readiness": teacher_readiness_formal,
        "label_leakage": label_leakage,
        "anti_shortcut": anti,
        "probe_manifest": probes,
        "final_circuit": final_circuit,
        "final_exact_patching": final_exact,
        "process_circuit": process_circuit,
        "process_exact_patching": process_exact,
        "final_compatibility": final_compatibility,
        "process_compatibility": process_compatibility,
        "distributed_resume": resume,
    }
    for artifact_name, artifact in bound_artifacts.items():
        api._require_formal_binding(artifact, formal_binding, name=artifact_name)
        # Canonical bytes additionally reject Python bool/int equality aliases.
        binding.producer_origin(
            [artifact],
            expected_science_head=formal_binding["code_commit"],
            expected_formal_binding=formal_binding,
        )

    final_noise = api.estimate_estimator_noise_floor(
        final_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    process_noise = api.estimate_estimator_noise_floor(
        process_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    metrics = base.get("initial_validation_metrics", {})
    calibrated_metrics = base.get("calibrated_validation_metrics", {})
    teacher_mass = teacher.get("teacher_topk_mass", {})
    bank_total = int(teacher.get("total_trajectories", 0))
    bank_positive = int(teacher.get("reward_distribution", {}).get("positive", 0))
    checks: dict[str, bool] = {
        "base_task_accuracy": float(metrics.get("answer_accuracy", 0.0))
        >= float(config["anti_shortcut"]["minimum_iid_accuracy"]),
        "teacher_topk_mass": float(teacher_mass.get("minimum", 0.0)) >= 0.90,
        "teacher_correctness": teacher_readiness.get("passed") is True,
        "label_leakage": label_leakage.get("passed") is True,
        "fixed_bank_mixed_rewards": 0 < bank_positive < bank_total,
        "calibration_anchor_improves_accuracy": float(calibrated_metrics.get("answer_accuracy", 0.0))
        > float(metrics.get("answer_accuracy", 0.0)),
        "anti_shortcut": anti.get("passed") is True,
        "base_capable_probes": probes["cohorts"]["base_capable"]["discovery"]["num_examples"] > 0
        and probes["cohorts"]["base_capable"]["validation"]["num_examples"] > 0,
        "probe_scoring_binding": base.get("initial_checkpoint_sha256") == initial_checkpoint_hash
        and probes.get("scoring_manifest_hash") == base.get("sha256")
        and probes.get("eligibility_evidence_ancestry") == base.get("eligibility_evidence_ancestry"),
        "hf_transformerlens_gqa_parity": api._stage_compatibility_passes(
            final_compatibility,
            final_circuit,
        )
        and api._stage_compatibility_passes(
            process_compatibility,
            process_circuit,
        ),
        "bootstrap_stability": min(
            float(final_noise["within_checkpoint_full_score_spearman"]),
            float(process_noise["within_checkpoint_full_score_spearman"]),
        )
        >= float(config["g0"]["minimum_bootstrap_spearman"]),
        "final_stage_eap_beats_matched_random": float(
            final_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "process_stage_eap_beats_matched_random": float(
            process_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "distinct_stage_manifests": final_circuit.get("probe_stage") == "final_answer"
        and process_circuit.get("probe_stage") in {"first_rule_selection", "intermediate_conclusion"}
        and final_circuit.get("stage_target_manifest_hash")
        != process_circuit.get("stage_target_manifest_hash"),
        "identity_sanity": all(
            artifact.get("sanity_checks", {}).get("identity_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "full_corruption_sanity": all(
            artifact.get("sanity_checks", {}).get("full_corruption_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "attribution_exact_calibration": all(
            float(artifact.get("attribution_patching_spearman", float("-inf")))
            >= float(config["g0"]["minimum_attribution_exact_spearman"])
            and float(artifact.get("attribution_patching_spearman_ci", {}).get("lower", float("-inf")))
            >= float(config["g0"]["minimum_spearman_bootstrap_lower_bound"])
            for artifact in (final_exact, process_exact)
        ),
        "split_probe_isolation": probes.get("frozen_before_training") is True,
        "artifact_reconstruction": all(
            circuit.get("checkpoint_sha256") == initial_checkpoint_hash
            and exact.get("artifacts", {}).get("checkpoint_sha256") == initial_checkpoint_hash
            for circuit, exact in (
                (final_circuit, final_exact),
                (process_circuit, process_exact),
            )
        ),
        "execution_science_protocol": True,
    }
    qwen3_bindings = {
        "protocol_track": config["protocol_track"],
        "artifact_namespace": config["model"]["artifact_namespace"],
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
        "tokenizer_fingerprint": config["model"]["tokenizer_fingerprint"],
    }
    checks["qwen3_protocol_bindings"] = all(
        all(artifact.get(key) == value for key, value in qwen3_bindings.items())
        for artifact in bound_artifacts.values()
    )
    checks["qwen3_qk_norm_hook_semantics"] = api._qwen3_qk_norm_hooks_pass(
        final_compatibility
    ) and api._qwen3_qk_norm_hooks_pass(process_compatibility)
    batch_token_contract = api._batch_token_contract(config, 2)
    checks["protocol_amendment"] = (
        formal_binding.get("protocol_amendment_id") == api._AMENDMENT_ID
        and formal_binding.get("protocol_amendment_sha256") == formal_binding["protocol_amendment_sha256"]
        and formal_binding.get("reviewed_implementation_commit")
        == formal_binding["reviewed_implementation_commit"]
        and formal_binding.get("reviewed_implementation_commit") != formal_binding["code_commit"]
    )
    checks["batch_token_invariants"] = batch_token_contract == {
        "batch_partition_protocol": api._BATCH_PARTITION_PROTOCOL,
        "requested_fsdp_sharding_strategy": api.REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": api.effective_fsdp_sharding_strategy(2),
        "global_logical_batch_size": api._GLOBAL_BATCH_SIZE,
        "max_per_rank_microbatch_size": api._MAX_MICROBATCH_SIZE,
        "max_model_input_length": 1536,
        "full_parameter_training": True,
        "prompt_population_size": int(config["task"]["num_examples"]),
        "prompt_ids_unique": True,
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "microbatch_schedule_by_rank": api._MICROBATCHES_BY_WORLD_SIZE[2],
        "optimizer_microsteps": len(api._MICROBATCHES_BY_WORLD_SIZE[2][0]),
        "samples_by_rank": api._SAMPLES_BY_WORLD_SIZE[2],
        "world_size": 2,
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "max_optimizer_steps": 120,
    }
    checks.update(
        api._runtime_execution_class_checks(
            resume,
            config=config,
            world_size=2,
            certified_fsdp_wrapper_count=29,
        )
    )
    expected_metrics = {
        "base_accuracy": metrics.get("answer_accuracy"),
        "calibrated_accuracy": calibrated_metrics.get("answer_accuracy"),
        "teacher_topk_mass_minimum": teacher_mass.get("minimum"),
        "shortcut_gap": anti.get("shortcut_gap"),
        "iid_accuracy": anti.get("iid_accuracy"),
        "transformed_accuracy": anti.get("transformed_accuracy"),
        "label_leakage": label_leakage.get("metrics"),
        "teacher_readiness": teacher_readiness.get("metrics"),
        "stages": {
            "final_answer": {
                "bootstrap_spearman": final_noise["within_checkpoint_full_score_spearman"],
                "selected_vs_random_cpr_margin": final_exact.get("selected_vs_matched_random_cpr_margin"),
                "attribution_exact_spearman": final_exact.get("attribution_patching_spearman"),
            },
            str(process_circuit.get("probe_stage")): {
                "bootstrap_spearman": process_noise["within_checkpoint_full_score_spearman"],
                "selected_vs_random_cpr_margin": process_exact.get("selected_vs_matched_random_cpr_margin"),
                "attribution_exact_spearman": process_exact.get("attribution_patching_spearman"),
            },
        },
    }
    expected_names = set(api.G0_CHECK_NAMES) - OLD_BACKEND_CHECKS
    require(set(checks) == expected_names, "scientific G0 check schema drifted")
    require(all(type(value) is bool for value in checks.values()), "non-boolean scientific result")
    return checks, expected_metrics, paths, bound_artifacts


def validate_teacher_inputs(workspace, config, formal):
    """Re-read the full 144k family and formal 256-prompt teacher population."""
    family_api = importlib.import_module("posttrain_circuits.datasets.proofgraph.family")
    store_api = importlib.import_module("posttrain_circuits.datasets.teacher_demos.store")
    train = importlib.import_module("posttrain_circuits.cli.train")
    family = family_api.load_dataset_family(workspace / "dataset")
    sizes = config["task"]["split_sizes"]
    require(
        sum(sizes.values()) == 144000
        and all(len(family.examples(split)) == size for split, size in sizes.items()),
        "dataset population differs from the full frozen family",
    )
    accepted, manifest = store_api.read_teacher_demo_store(workspace / "teacher_demos", require_formal=True)
    train._require_qwen3_store_binding(manifest, config=config, expected_behavior_policy="Qwen/Qwen3-8B")
    # Original store validation checks all formal fields and valid behavior logprobs.
    prompt_ids, _ = train._teacher_demo_prompts(accepted)
    ordered = [example.example_id for example in family.examples("train")[:256]]
    train._validate_allocation_neutral_prompt_population(
        prompt_ids,
        global_batch_size=64,
        expected_prompt_ids=ordered,
        expected_prompt_count=256,
    )
    train._validate_teacher_demo_model_input_lengths(
        accepted,
        manifest,
        expected_prompt_count=256,
        candidates_per_prompt=8,
        max_prompt_tokens=1246,
        max_new_tokens=256,
        max_model_input_length=1536,
    )
    require(
        config["task"]["num_examples"] == 256 and config["seed"] == 42, "G0 prompt/seed population differs"
    )
    return {
        "dataset_family_sha256": family.manifest["sha256"],
        "ordered_prompt_ids_sha256": digest(canonical(ordered)),
        "teacher_manifest_sha256": binding.file_identity(workspace / "teacher_demos/manifest.json")["sha256"],
    }


def readiness_evidence(checks, metrics, paths):
    """Exactly the original readiness gate names and scientific predicates."""
    mapping = {
        "base_task_accuracy_nontrivial": ["base_task_accuracy"],
        "pilot_improves_accuracy": ["calibration_anchor_improves_accuracy"],
        "verifier_deterministic": ["anti_shortcut"],
        "fixed_bank_mixed_rewards": ["fixed_bank_mixed_rewards"],
        "teacher_topk_mass_acceptable": ["teacher_topk_mass"],
        "hf_circuit_logit_parity": ["hf_transformerlens_gqa_parity"],
        "eap_ig_beats_random": [
            "final_stage_eap_beats_matched_random",
            "process_stage_eap_beats_matched_random",
        ],
        "exact_patching_distinguishes_groups": ["distinct_stage_manifests"],
        "attribution_bootstrap_stable": ["bootstrap_stability"],
        "checkpoint_resume_verified": ["distributed_checkpoint_resume"],
        "split_leakage_absent": ["split_probe_isolation"],
        "anti_shortcut_gap": ["anti_shortcut"],
        "probe_cohorts_frozen": ["base_capable_probes"],
        "teacher_correctness": ["teacher_correctness"],
        "label_leakage": ["label_leakage"],
    }
    evidence_hash = digest(canonical({"checks": checks, "metrics": metrics, "artifacts": paths}))
    return {
        name: (
            all(checks[key] is True for key in keys),
            "reconstructed SDSC G0 scientific evidence sha256:" + evidence_hash,
        )
        for name, keys in mapping.items()
    }


def require_safe_tree(workspace):
    require(workspace.is_dir(), "G0 workspace does not exist")
    import stat

    identities = {}
    for path in workspace.rglob("*"):
        observed = path.lstat()
        mode = observed.st_mode
        require(
            stat.S_ISREG(mode) or stat.S_ISDIR(mode),
            "G0 workspace contains a symlink or special file: " + str(path),
        )
        identities[str(path.relative_to(workspace))] = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
            (
                binding.file_identity(path)["sha256"]
                if stat.S_ISREG(mode) and 0 < observed.st_size <= 8 * 1024**2
                else None
            ),
        )
    return identities


def evaluate_workspace(args):
    """Reconstruct the decision once; callers compare or publish its result.

    Invoke in a fresh process. Original scientific modules are imported only
    after their genuine clean Git and accepted scientific review are validated.
    """
    science_root, workspace = binding.real_path(args.science_root), binding.real_path(args.workspace)
    require(
        args.expected_science_head == SCIENCE_HEAD and args.expected_protocol_sha256 == PROTOCOL_SHA256,
        "this adapter supports only the actual accepted prompt-v3 scientific lineage",
    )
    workspace_identity = require_safe_tree(workspace)
    for path in (args.runtime_config, args.execution_evidence):
        require(workspace in binding.real_path(path).parents, "G0 input is outside its workspace")
    config, config_identity = checked_json(args.runtime_config)
    execution, execution_identity = checked_json(args.execution_evidence, args.execution_evidence_sha256)
    previous = Path.cwd()
    try:
        os.chdir(science_root)
        base, projection, reviewed = binding.reviewed_science(
            science_root,
            args.expected_science_head,
            args.expected_protocol_sha256,
        )
        api = importlib.import_module("posttrain_circuits.cli.finalize_g0")
        formal = api.formal_artifact_binding(base)
        require(
            canonical(formal) == canonical(api.formal_artifact_binding(config)),
            "runtime formal scientific binding differs from the accepted base",
        )
        require(formal["code_commit"] == SCIENCE_HEAD, "formal HEAD differs")
        runtime_binding = binding.verify_runtime_binding(
            base,
            config,
            expected_base_science_sha256=binding.FROZEN_BASE_SHA256,
            expected_checkpoint_sha256=args.expected_initial_sha256,
            checkpoint_path=workspace / "initial_checkpoint.pt",
            science_projection=projection,
            relocated_workspace=workspace,
        )
        initial = helper("sdsc_g0_calibration").initial_checkpoint_identity(
            workspace / "initial_checkpoint.pt", config
        )
        require(initial["sha256"] == args.expected_initial_sha256, "initial checkpoint identity changed")
        invocation = validate_execution(
            execution, workspace, config, science_root, args.expected_adapter_sha256, api
        )
        resume_publication, _ = checked_json(workspace / "distributed_resume.json")
        expected_resume_target = {
            key: execution["target"][key] for key in ("run_id", "job_id", "code_sha256")
        }
        require(
            canonical(resume_publication.get("target")) == canonical(expected_resume_target)
            and resume_publication.get("science_git_head") == SCIENCE_HEAD,
            "resume evidence belongs to a different Slurm invocation or scientific checkout",
        )
        teacher_inputs = validate_teacher_inputs(workspace, config, formal)
        checks, metrics, paths, artifacts = scientific_decision(workspace, config, formal, api)
        checks.update(dict.fromkeys(BACKEND_CHECKS, True))
        require(
            set(checks) == (set(api.G0_CHECK_NAMES) - OLD_BACKEND_CHECKS) | BACKEND_CHECKS,
            "SDSC G0 complete check schema drifted",
        )
        require(
            paths["execution_science_protocol"].read_bytes()
            == (science_root / binding.PROTOCOL).read_bytes(),
            "bundled accepted execution-science protocol differs from real Git",
        )
        source_workspace = binding.real_path(config["output_root"])
        paths.update(
            dataset_manifest=workspace / "dataset/manifest.json",
            teacher_demo_manifest=workspace / "teacher_demos/manifest.json",
            runtime_config=args.runtime_config,
            execution_evidence=args.execution_evidence,
            provenance_manifest=workspace / "provenance-manifest.json",
            preflight=workspace / "preflight.json",
            preflight_source_manifest=workspace / "preflight-source-manifest.json",
        )
        # The original pilot validator expects actual absolute file-name keys.
        roles = {name: str(source_workspace / path.relative_to(workspace)) for name, path in paths.items()}
        artifact_hashes = {roles[name]: binding.file_identity(path)["sha256"] for name, path in paths.items()}
        require(
            artifact_hashes[roles["runtime_config"]] == config_identity["sha256"]
            and artifact_hashes[roles["execution_evidence"]] == execution_identity["sha256"],
            "decision inputs changed during validation",
        )
        readiness = api.build_readiness_report(
            readiness_evidence(checks, metrics, artifact_hashes),
            bindings={
                **formal,
                "initial_checkpoint_hash": args.expected_initial_sha256,
                "dataset_hash": str(artifacts["anti_shortcut"]["dataset_hash"]),
                "suite_hash": str(artifacts["anti_shortcut"]["suite_hash"]),
            },
        )
        payload = {
            "schema": SCHEMA,
            "format_version": 2,
            **api.scientific_compatibility_fields(str(config["prereg_version"])),
            **formal,
            "phase": "G0",
            "backend": "sdsc_expanse_slurm_h100_single_invocation",
            "passed": all(value is True for value in checks.values()) and readiness.ready,
            "g0_passed": all(value is True for value in checks.values()) and readiness.ready,
            "pilot_passed": False,
            "factorial_ready": False,
            "execution_class_certified": False,
            "uses_blackwell_certificate": False,
            "uses_serverscheduler_context": False,
            "scientific_protocol_compatible": True,
            "compatibility_basis": {
                "science_head": SCIENCE_HEAD,
                "accepted_protocol_path": binding.PROTOCOL,
                "accepted_protocol_sha256": reviewed.sha256,
                "accepted_protocol_commit": reviewed.acceptance_commit,
                "accepted_implementation_commit": reviewed.reviewed_implementation_commit,
                "base_science_config_sha256": binding.FROZEN_BASE_SHA256,
                "runtime_science_config_sha256": runtime_binding["runtime_science_config_sha256"],
                "runtime_derived_field": "production_safety.initial_checkpoint_hash",
                "initial_checkpoint_sha256": args.expected_initial_sha256,
                "old_backend_checks_replaced": sorted(OLD_BACKEND_CHECKS),
                "new_backend_checks": sorted(BACKEND_CHECKS),
                "scientific_check_names": sorted(set(api.G0_CHECK_NAMES) - OLD_BACKEND_CHECKS),
            },
            "checks": checks,
            "metrics": metrics,
            "git_commit": SCIENCE_HEAD,
            "protocol_prereg_version": str(config["protocol_track"]),
            "batch_token_contract": api._batch_token_contract(config, 2),
            "world_size": 2,
            "resolved_config_sha256": digest(canonical(config)),
            "job_ids": [str(execution["target"]["job_id"])],
            "execution_science_protocol_git_commit": reviewed.acceptance_commit,
            "execution_science_protocol_id": reviewed.binding.protocol_id,
            "execution_science_protocol_reviewed_implementation_commit": (
                reviewed.reviewed_implementation_commit
            ),
            "execution_science_protocol_sha256": reviewed.sha256,
            "source_workspace": str(source_workspace),
            "execution": invocation,
            "execution_evidence_sha256": execution_identity["sha256"],
            "teacher_inputs": teacher_inputs,
            "artifact_roles": roles,
            "artifact_hashes": artifact_hashes,
        }
        require(
            not binding.git(science_root, "status", "--porcelain=v1", "--untracked-files=all"),
            "scientific checkout changed while evaluating G0",
        )
        binding.verify_source_bytes(science_root, SCIENCE_HEAD)
        require(
            require_safe_tree(workspace) == workspace_identity, "G0 workspace changed during reconstruction"
        )
        return payload, readiness
    finally:
        os.chdir(previous)


def validate_replay(report, reconstructed):
    """Reject extra/missing fields, altered decisions, and re-signed false claims."""
    require(set(report) == set(reconstructed) | {"created_at", "sha256"}, "SDSC G0 report schema differs")
    require(isinstance(report.get("created_at"), str) and report["created_at"], "G0 timestamp is absent")
    require(
        report.get("sha256")
        == digest(canonical({key: value for key, value in report.items() if key != "sha256"})),
        "G0 report hash differs",
    )
    require(
        canonical({key: report[key] for key in reconstructed}) == canonical(reconstructed),
        "G0 report differs from reconstructed scientific and execution evidence",
    )
    require(reconstructed["passed"] is True, "G0 did not pass every required check")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "workspace", "runtime-config", "execution-evidence"):
        result.add_argument("--" + name, required=True, type=Path)
    for name in (
        "execution-evidence-sha256",
        "expected-science-head",
        "expected-protocol-sha256",
        "expected-initial-sha256",
        "expected-adapter-sha256",
    ):
        result.add_argument("--" + name, required=True)
    result.add_argument(
        "--replay", type=Path, help="Reconstruct and validate an existing G0 report; never writes"
    )
    result.add_argument("--output", type=Path, help="Fresh output, fixed to workspace/g0.json")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    output = binding.real_path(args.output or args.workspace / "g0.json")
    require(output == args.workspace / "g0.json", "G0 output must be the fixed workspace/g0.json")
    require(not (args.replay and args.output), "replay is read-only")
    require(
        args.replay is None or binding.real_path(args.replay) == args.workspace / "g0.json",
        "replay report must be workspace/g0.json",
    )
    if args.replay is None:
        require(
            not output.exists() and not (args.workspace / "readiness").exists(),
            "G0/readiness publication exists; use replay instead of overwriting",
        )
    reconstructed, readiness = evaluate_workspace(args)
    if args.replay is not None:
        report, _ = checked_json(args.replay)
        validate_replay(report, reconstructed)
        # Verify the separately consumed readiness document against the same decision.
        actual, _ = checked_json(args.workspace / "readiness/readiness.json")
        content = {key: value for key, value in actual.items() if key != "sha256"}
        require(actual.get("sha256") == digest(canonical(content)), "readiness hash differs")
        require(
            actual.get("ready") is True
            and canonical(actual.get("bindings")) == canonical(readiness.bindings),
            "readiness bindings/decision differ",
        )
        import dataclasses

        require(
            canonical(actual.get("checks"))
            == canonical([dataclasses.asdict(check) for check in readiness.checks]),
            "readiness checks differ from reconstructed decision",
        )
        print(
            json.dumps(
                {
                    "validated": True,
                    "schema": SCHEMA,
                    "g0_sha256": report["sha256"],
                    "passed": True,
                    "execution_class_certified": False,
                },
                sort_keys=True,
            )
        )
        return 0
    reconstructed["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    reconstructed["sha256"] = digest(canonical(reconstructed))
    readiness.write(args.workspace / "readiness")
    api = importlib.import_module("posttrain_circuits.artifacts.io")
    api.atomic_write_json(output, reconstructed)
    print(
        json.dumps(
            {"path": str(output), "passed": reconstructed["passed"], "sha256": reconstructed["sha256"]},
            sort_keys=True,
        )
    )
    return 0 if reconstructed["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"error": str(error), "passed": False}), file=sys.stderr)
        raise SystemExit(2) from None
