"""Finalize the seed-42 feasibility pilot from a complete hash chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from posttrain_circuits.analysis.factorial import match_validation_accuracy
from posttrain_circuits.circuits.pilot_scope import PILOT_CELLS, resolve_pilot_circuit_scope
from posttrain_circuits.circuits.probe_cohorts import validate_probe_cohort_manifest
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import (
    formal_artifact_binding,
    require_git_output,
    validate_run_manifest_payload,
)
from posttrain_circuits.core.readiness import (
    require_formal_prerequisite_binding,
    validate_readiness_report,
)
from posttrain_circuits.data.splits import load_frozen_split
from posttrain_circuits.data.trajectory_store import TrajectoryStore

MATCHED_PAIRS = (
    ("offline_soft", "online_soft_opd"),
    ("online_hard", "online_soft_opd"),
    ("online_verified_replay", "canonical_grpo"),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"pilot artifact is not a mapping: {path}")
    return payload


def _hash_valid(payload: dict[str, Any]) -> bool:
    content = dict(payload)
    expected = content.pop("sha256", None)
    return expected == sha256_value(content)


def _require_g0_file_binding(g0: dict[str, Any], path: Path, *, name: str) -> None:
    artifact_hashes = g0.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("G0 artifact has no bound input-file index")
    resolved = path.resolve()
    digest = sha256_file(path)
    matches = [
        raw_path
        for raw_path, raw_digest in artifact_hashes.items()
        if Path(str(raw_path)).resolve() == resolved and raw_digest == digest
    ]
    if len(matches) != 1:
        raise ValueError(f"pilot {name} is not the exact file bound by G0: {path}")


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite(child) for child in value)
    return True


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"pilot metrics are empty or invalid: {path}")
    return rows


def _curve(rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    selected = [(float(row["step"]), float(row[metric])) for row in rows if row.get(metric) is not None]
    return [row[0] for row in selected], [row[1] for row in selected]


def _matched_summary(rows_by_cell: dict[str, list[dict[str, Any]]], validation_hash: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for left, right in MATCHED_PAIRS:
        left_steps, left_values = _curve(rows_by_cell[left], "validation_accuracy")
        right_steps, right_values = _curve(rows_by_cell[right], "validation_accuracy")
        if len(left_values) < 2 or len(right_values) < 2:
            results[f"{left}__{right}"] = {
                "valid": False,
                "reason": "both cells require at least two formal-validation checkpoints",
            }
            continue
        lower = max(min(left_values), min(right_values))
        upper = min(max(left_values), max(right_values))
        if lower > upper:
            results[f"{left}__{right}"] = {
                "valid": False,
                "reason": "validation-accuracy ranges do not overlap; extrapolation forbidden",
            }
            continue
        target = (lower + upper) / 2.0
        results[f"{left}__{right}"] = {
            "valid": True,
            "target_validation_accuracy": target,
            left: match_validation_accuracy(
                left_steps, left_values, target, validation_artifact_hash=validation_hash
            ).__dict__,
            right: match_validation_accuracy(
                right_steps, right_values, target, validation_artifact_hash=validation_hash
            ).__dict__,
        }
    return results


def _require_cell_chain(
    *,
    root: Path,
    cell: str,
    expected: dict[str, Any],
    terminal_hash: str,
    training_job_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    manifest_path = root / "manifest.json"
    manifest = validate_run_manifest_payload(_read(manifest_path))
    binding = manifest.get("pilot_terminal_binding")
    if not isinstance(binding, dict) or not _hash_valid(binding):
        raise ValueError(f"pilot terminal binding is absent or invalid: {cell}")
    metrics = root / "metrics.jsonl"
    checkpoint = Path(str(manifest.get("final_checkpoint_path", "")))
    resolved = root / "resolved_config.yaml"
    if not checkpoint.resolve().is_relative_to((root / "checkpoints").resolve()):
        raise ValueError(f"pilot final checkpoint is outside its cell directory: {cell}")
    required = {
        "experiment_cell": cell,
        "seed": 42,
        "protocol_track": expected["protocol_track"],
        "artifact_namespace": expected["artifact_namespace"],
        "model_revision": expected["model_revision"],
        "protocol_teacher_revision": expected["teacher_revision"],
        "tokenizer_revision": expected["tokenizer_revision"],
        "tokenizer_fingerprint": expected["tokenizer_fingerprint"],
        "chat_template_sha256": expected["chat_template_sha256"],
        "prompt_protocol": expected["prompt_protocol"],
        "enable_thinking": expected["enable_thinking"],
        "git_commit": expected["code_commit"],
        "prereg_path": expected["prereg_path"],
        "prereg_version": expected["prereg_version"],
        "prereg_git_commit": expected["prereg_commit"],
        "prereg_sha256": expected["prereg_sha256"],
        "slurm_terminal_evidence_sha256": terminal_hash,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"pilot cell manifest binding mismatch for {cell}: {mismatches}")
    file_bindings = {
        metrics: str(manifest.get("metrics_sha256", "")),
        checkpoint: str(manifest.get("final_checkpoint_sha256", "")),
        resolved: str(binding.get("resolved_config_sha256", "")),
    }
    for path, digest in file_bindings.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"pilot cell file changed after binding: {cell}: {path}")
    if binding.get("metrics_sha256") != manifest.get("metrics_sha256"):
        raise ValueError(f"pilot metrics chain is inconsistent: {cell}")
    if binding.get("final_checkpoint_sha256") != manifest.get("final_checkpoint_sha256"):
        raise ValueError(f"pilot checkpoint chain is inconsistent: {cell}")
    binding_required = {
        "cell": cell,
        "seed": 42,
        "slurm_terminal_evidence_sha256": terminal_hash,
        "token_budget": manifest.get("token_budget"),
        "token_budget_consumed": manifest.get("token_budget_consumed"),
        "token_budget_unit": manifest.get("token_budget_unit"),
        "training_stop_reason": manifest.get("training_stop_reason"),
        "dataset_hashes_sha256": sha256_value(manifest.get("dataset_hashes", {})),
        "probe_manifest_hashes": sorted(
            str(value)
            for key, value in manifest.get("dataset_hashes", {}).items()
            if key.startswith("prerequisite_probe")
        ),
        "initial_checkpoint_sha256": manifest.get("dataset_hashes", {}).get("initial_checkpoint"),
        "state_source_artifact_sha256": manifest.get("rollout_bank_hash"),
        "slurm": {
            "job_id_raw": f"{training_job_id}_{PILOT_CELLS.index(cell)}",
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
    }
    binding_mismatches = {
        key: {"expected": value, "observed": binding.get(key)}
        for key, value in binding_required.items()
        if binding.get(key) != value
    }
    if binding_mismatches:
        raise ValueError(f"pilot terminal binding mismatch for {cell}: {binding_mismatches}")
    if int(manifest.get("token_budget_consumed", -1)) > int(manifest.get("token_budget", -2)):
        raise ValueError(f"pilot cell exceeded its token budget: {cell}")
    return manifest, _rows(metrics), [manifest_path, metrics, checkpoint, resolved]


def _require_circuit_slot(
    *,
    circuit: dict[str, Any],
    exact: dict[str, Any],
    circuit_path: Path,
    expected_checkpoint_sha256: str,
    cohort: str,
    probe_stage: str,
) -> None:
    if not _hash_valid(circuit) or not _hash_valid(exact):
        raise ValueError(f"circuit slot contains an invalid artifact hash: {circuit_path}")
    circuit_required = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "probe_cohort": cohort,
        "probe_stage": probe_stage,
    }
    exact_required = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "probe_cohort": cohort,
        "probe_stage": probe_stage,
    }
    for kind, artifact, required in (
        ("circuit", circuit, circuit_required),
        ("exact-patching", exact, exact_required),
    ):
        mismatches = {
            key: {"expected": value, "observed": artifact.get(key)}
            for key, value in required.items()
            if artifact.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{kind} artifact is in the wrong pilot matrix slot: {mismatches}")
    checkpoint_path = Path(str(circuit.get("checkpoint_path", "")))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError(f"circuit slot checkpoint bytes are absent or changed: {circuit_path}")
    exact_inputs = exact.get("artifacts", {})
    if not isinstance(exact_inputs, dict):
        raise ValueError(f"exact-patching artifact has no input binding: {circuit_path}")
    if Path(str(exact_inputs.get("circuit_artifact", ""))).resolve() != circuit_path.resolve():
        raise ValueError(f"exact-patching artifact is bound to a different circuit: {circuit_path}")


def _require_dynamics_slot(
    *,
    artifact: dict[str, Any],
    expected_initial_sha256: str,
    expected_final_sha256: str,
    cohort: str,
    probe_stage: str,
) -> None:
    required = {
        "probe_cohort": cohort,
        "probe_stage": probe_stage,
    }
    mismatches = {
        key: {"expected": value, "observed": artifact.get(key)}
        for key, value in required.items()
        if artifact.get(key) != value
    }
    transitions = artifact.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != 1:
        mismatches["transitions"] = {"expected": "one initial-to-final transition", "observed": transitions}
    else:
        transition = transitions[0]
        for key, value in (
            ("source_checkpoint_sha256", expected_initial_sha256),
            ("target_checkpoint_sha256", expected_final_sha256),
        ):
            if not isinstance(transition, dict) or transition.get(key) != value:
                mismatches[f"transitions[0].{key}"] = {
                    "expected": value,
                    "observed": transition.get(key) if isinstance(transition, dict) else None,
                }
    if mismatches:
        raise ValueError(f"circuit dynamics artifact is in the wrong pilot matrix slot: {mismatches}")


def _terminal_paths(run_dir: Path, job_ids_path: Path) -> tuple[dict[str, str], list[Path]]:
    jobs: dict[str, str] = {}
    for line in job_ids_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            raise ValueError("pilot job-id manifest contains an invalid row")
        stage, job_id = line.split("=", 1)
        if stage in jobs or not job_id.isdigit():
            raise ValueError("pilot job-id manifest contains a duplicate or invalid job id")
        jobs[stage] = job_id
    expected = {"training", "initial_circuits", "final_circuits", "local_fork", "resume", "dynamics"}
    if set(jobs) != expected:
        raise ValueError(f"pilot job stages differ from the registered stages: {set(jobs)}")
    paths = [run_dir / f"terminal-{stage}.txt" for stage in sorted(expected)]
    for stage, path in zip(sorted(expected), paths, strict=True):
        rows = [line.split("|") for line in path.read_text(encoding="utf-8").splitlines() if line]
        job_id = jobs[stage]
        if not rows or any(
            len(row) < 3
            or not (row[0] == job_id or row[0].startswith(f"{job_id}_") or row[0].startswith(f"{job_id}."))
            or row[1] != "COMPLETED"
            or row[2] != "0:0"
            for row in rows
        ):
            raise ValueError(f"pilot stage does not have complete success evidence: {stage}")
    return jobs, paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finalize the gated seed-42 Qwen pilot")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--job-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    expected = formal_artifact_binding(config)
    scope = resolve_pilot_circuit_scope(config)
    g0 = _read(args.g0)
    g0_binding_keys = (
        "protocol_track",
        "artifact_namespace",
        "model_revision",
        "teacher_revision",
        "tokenizer_revision",
        "tokenizer_fingerprint",
        "chat_template_sha256",
        "prompt_protocol",
        "enable_thinking",
        "code_commit",
        "prereg_path",
        "prereg_version",
        "prereg_commit",
        "prereg_sha256",
    )
    g0_mismatches = {
        key: {"expected": expected[key], "observed": g0.get(key)}
        for key in g0_binding_keys
        if g0.get(key) != expected[key]
    }
    if g0_mismatches:
        raise ValueError(f"pilot refused a stale or cross-protocol G0 artifact: {g0_mismatches}")
    bank = TrajectoryStore(args.bank_manifest.parent).check_integrity()
    probes = validate_probe_cohort_manifest(args.probe_manifest)
    require_formal_prerequisite_binding(bank, expected, name="pilot rollout bank")
    require_formal_prerequisite_binding(probes, expected, name="pilot probe cohorts")
    _require_g0_file_binding(g0, args.bank_manifest, name="rollout bank")
    _require_g0_file_binding(g0, args.probe_manifest, name="probe manifest")
    _, validation = load_frozen_split(args.validation_manifest.parent, expected_split="validation")
    jobs, terminal_paths = _terminal_paths(args.run_dir, args.job_ids)
    training_terminal = args.run_dir / "terminal-training.txt"
    terminal_hash = sha256_file(training_terminal)
    chain_path = args.run_dir / "training_artifact_chain.json"
    chain = _read(chain_path)
    if (
        not _hash_valid(chain)
        or chain.get("terminal_sha256") != terminal_hash
        or chain.get("job_id") != jobs["training"]
    ):
        raise ValueError("pilot training artifact-chain index is invalid")

    manifests: dict[str, dict[str, Any]] = {}
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    cell_paths: list[Path] = []
    for cell in PILOT_CELLS:
        manifest, rows, paths = _require_cell_chain(
            root=args.run_dir / "runs" / cell / "seed-42",
            cell=cell,
            expected=expected,
            terminal_hash=terminal_hash,
            training_job_id=jobs["training"],
        )
        manifests[cell] = manifest
        rows_by_cell[cell] = rows
        cell_paths.extend(paths)
        if chain.get("cells", {}).get(cell, {}).get("manifest_sha256") != sha256_file(paths[0]):
            raise ValueError(f"pilot cell manifest differs from the terminal artifact-chain: {cell}")

    expected_initial_hash = str(probes.get("initial_student_checkpoint_hash", ""))
    initial_circuit_paths: list[Path] = []
    for row in scope["initial_matrix"]:
        root = args.run_dir / "circuits" / "initial" / row["stage_label"] / row["cohort"]
        initial_circuit_paths.extend((root / "circuit.json", root / "exact_patching.json"))
    final_circuit_paths: list[Path] = []
    for row in scope["final_matrix"]:
        root = args.run_dir / "circuits" / "final" / row["cell"] / row["stage_label"] / row["cohort"]
        final_circuit_paths.extend((root / "circuit.json", root / "exact_patching.json"))
    dynamics_paths = [
        args.run_dir / "dynamics" / row["cell"] / row["stage_label"] / f"{row['cohort']}.json"
        for row in scope["final_matrix"]
    ]
    circuit_artifacts = [_read(path) for path in (*initial_circuit_paths, *final_circuit_paths)]
    dynamics = [_read(path) for path in dynamics_paths]
    exact = [
        artifact
        for path, artifact in zip(
            (*initial_circuit_paths, *final_circuit_paths), circuit_artifacts, strict=True
        )
        if path.name == "exact_patching.json"
    ]
    for index, row in enumerate(scope["initial_matrix"]):
        circuit_path = initial_circuit_paths[index * 2]
        _require_circuit_slot(
            circuit=circuit_artifacts[index * 2],
            exact=circuit_artifacts[index * 2 + 1],
            circuit_path=circuit_path,
            expected_checkpoint_sha256=expected_initial_hash,
            cohort=str(row["cohort"]),
            probe_stage=str(row["probe_stage"]),
        )
    final_offset = len(initial_circuit_paths)
    for index, row in enumerate(scope["final_matrix"]):
        artifact_index = final_offset + index * 2
        cell = str(row["cell"])
        circuit_path = final_circuit_paths[index * 2]
        final_hash = str(manifests[cell].get("final_checkpoint_sha256", ""))
        _require_circuit_slot(
            circuit=circuit_artifacts[artifact_index],
            exact=circuit_artifacts[artifact_index + 1],
            circuit_path=circuit_path,
            expected_checkpoint_sha256=final_hash,
            cohort=str(row["cohort"]),
            probe_stage=str(row["probe_stage"]),
        )
        _require_dynamics_slot(
            artifact=dynamics[index],
            expected_initial_sha256=expected_initial_hash,
            expected_final_sha256=final_hash,
            cohort=str(row["cohort"]),
            probe_stage=str(row["probe_stage"]),
        )
    local_fork_path = args.run_dir / "local_fork" / "results.json"
    resume_path = args.run_dir / "distributed_resume.json"
    local_fork = _read(local_fork_path)
    resume = _read(resume_path)
    auxiliary = [local_fork, resume, *circuit_artifacts, *dynamics]
    readiness_path = args.g0.parent / "readiness" / "readiness.json"
    readiness = validate_readiness_report(
        readiness_path, expected_initial_checkpoint_hash=expected_initial_hash
    )
    require_formal_prerequisite_binding(readiness, expected, name="pilot readiness", nested=True)
    if readiness.get("bindings", {}).get("dataset_hash") != validation.get("sha256"):
        raise ValueError("pilot validation manifest differs from the dataset bound by readiness")
    for name, artifact in (
        ("local fork", local_fork),
        ("distributed resume", resume),
        *(("circuit or exact-patching", row) for row in circuit_artifacts),
        *(("circuit dynamics", row) for row in dynamics),
    ):
        require_formal_prerequisite_binding(artifact, expected, name=name)
    offline_hashes = {
        manifests[cell].get("rollout_bank_hash")
        for cell in ("offline_hard", "offline_soft", "offline_verified_replay")
    }
    validation_hash = str(validation.get("sha256", ""))
    matched = _matched_summary(rows_by_cell, validation_hash)
    transition_rows = [row for artifact in dynamics for row in artifact.get("transitions", [])]
    checks = {
        "g0_passed_hash_and_commit_bound": g0.get("passed") is True
        and _hash_valid(g0)
        and g0.get("git_commit") == require_git_output(["rev-parse", "HEAD"]),
        "g0_all_registered_checks": bool(g0.get("checks"))
        and all(value is True for value in g0["checks"].values()),
        "full_readiness_report": readiness.get("ready") is True,
        "all_training_cells_present": len(manifests) == len(PILOT_CELLS),
        "cell_artifact_chains_complete": all(
            manifest.get("pilot_terminal_binding", {}).get("slurm", {}).get("state") == "COMPLETED"
            for manifest in manifests.values()
        ),
        "all_input_artifacts_hash_valid": all(_hash_valid(row) for row in auxiliary),
        "all_artifacts_finite": all(_finite(rows) for rows in rows_by_cell.values())
        and all(_finite(row) for row in auxiliary),
        "single_seed_42": all(int(row.get("seed", -1)) == 42 for row in manifests.values()),
        "offline_bank_shared": len(offline_hashes) == 1
        and next(iter(offline_hashes), None) == bank.get("sha256"),
        "bank_reward_mixture": 0
        < int(bank.get("reward_distribution", {}).get("positive", 0))
        < int(bank.get("total_trajectories", 0)),
        "local_fork_output_kl_matched": local_fork.get("valid_for_primary_analysis") is True,
        "distributed_resume": resume.get("passed") is True and int(resume.get("world_size", 0)) == 4,
        "full_registered_circuit_matrix": len(initial_circuit_paths) == 2 * int(scope["initial_count"])
        and len(final_circuit_paths) == 2 * int(scope["final_count"])
        and len(dynamics_paths) == int(scope["final_count"]),
        "circuit_noise_floor_protocol_complete": bool(transition_rows)
        and all(
            all(
                key in row
                for key in (
                    "excess_churn",
                    "full_score_spearman_stability",
                    "weighted_overlap",
                    "cross_checkpoint_mask_transfer",
                )
            )
            for row in transition_rows
        ),
        "exact_patching_protocol_complete": bool(exact)
        and all(
            "heldout_exact_patching_effects" in row
            and "selected_vs_matched_random_cpr_margin" in row
            and "attribution_patching_spearman" in row
            for row in exact
        ),
        "probe_cohorts_frozen": probes.get("frozen_before_training") is True,
        "formal_validation_bound": bool(validation_hash)
        and all(
            row.get("dataset_hashes", {}).get("validation_manifest") == validation_hash
            for row in manifests.values()
        ),
        "exact_initial_checkpoint_bound": len(expected_initial_hash) == 64
        and all(
            row.get("dataset_hashes", {}).get("initial_checkpoint") == expected_initial_hash
            for row in manifests.values()
        ),
        "matched_accuracy_summaries_complete": len(matched) == len(MATCHED_PAIRS)
        and all(bool(row.get("valid")) or bool(row.get("reason")) for row in matched.values()),
        "qwen3_v2_protocol_bound": all(
            row.get("protocol_track") == "qwen3_v2"
            and row.get("artifact_namespace") == "qwen3-v2"
            and row.get("prereg_version") == "qwen3_v2"
            for row in manifests.values()
        ),
        "observed_effect_direction_not_used_as_gate": True,
    }
    output_kl = {
        cell: _curve(rows, "output_kl_from_initial")
        for cell, rows in rows_by_cell.items()
        if _curve(rows, "output_kl_from_initial")[1]
    }
    all_paths = (
        args.g0,
        args.bank_manifest,
        args.probe_manifest,
        args.validation_manifest,
        readiness_path,
        args.job_ids,
        chain_path,
        *terminal_paths,
        *cell_paths,
        local_fork_path,
        resume_path,
        *initial_circuit_paths,
        *final_circuit_paths,
        *dynamics_paths,
    )
    payload: dict[str, Any] = {
        "phase": "qwen3_v2_seed42_full_pipeline_feasibility",
        "claim_scope": config["pilot"]["claim_scope"],
        "passed": all(checks.values()),
        "readiness": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "seed": 42,
        "full_three_seed_factorial_launched": False,
        "gemma_replication_launched": False,
        "matched_validation_accuracy": matched,
        "output_kl_curves": output_kl,
        "circuit_scope": scope,
        "git_commit": expected["code_commit"],
        "prereg_commit": expected["prereg_commit"],
        "prereg_sha256": expected["prereg_sha256"],
        "artifact_hashes": {str(path): sha256_file(path) for path in all_paths},
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    args.output.with_suffix(".md").write_text(
        "# Qwen3-v2 seed-42 pipeline feasibility report\n\n"
        f"Readiness: **{payload['readiness']}**\n\n"
        "This pilot can establish pipeline and artifact-integrity feasibility only; it cannot "
        "establish confirmatory primary endpoints or population inference.\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("pilot completed with readiness NO-GO")


if __name__ == "__main__":
    main()
