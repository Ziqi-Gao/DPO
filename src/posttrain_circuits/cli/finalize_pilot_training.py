"""Bind completed pilot cell outputs to Slurm terminal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from posttrain_circuits.circuits.pilot_scope import PILOT_CELLS
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import (
    formal_artifact_binding,
    validate_run_manifest_payload,
)
from posttrain_circuits.training.token_budget import TOKEN_BUDGET_UNIT


def _terminal_tasks(path: Path, job_id: str) -> dict[int, dict[str, str]]:
    tasks: dict[int, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("|")
        if len(fields) < 3 or "_" not in fields[0]:
            continue
        raw_id, state, exit_code = fields[:3]
        parent, raw_index = raw_id.split("_", 1)
        if parent != job_id or not raw_index.isdigit():
            continue
        index = int(raw_index)
        if index in tasks:
            raise ValueError(f"duplicate Slurm terminal record for array task {index}")
        tasks[index] = {"job_id_raw": raw_id, "state": state, "exit_code": exit_code}
    if set(tasks) != set(range(len(PILOT_CELLS))):
        raise ValueError("Slurm terminal evidence does not contain exactly the eight pilot tasks")
    if any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in tasks.values()):
        raise ValueError("pilot training array contains a non-success terminal task")
    return tasks


def bind_pilot_training_outputs(
    *, run_dir: Path, terminal: Path, job_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    expected = formal_artifact_binding(config)
    tasks = _terminal_tasks(terminal, job_id)
    terminal_hash = sha256_file(terminal)
    budget = int(config["trainer"]["token_budget"])
    initial_hash = str(config["production_safety"]["initial_checkpoint_hash"])
    bound: dict[str, Any] = {}
    for index, cell in enumerate(PILOT_CELLS):
        root = run_dir / "runs" / cell / "seed-42"
        manifest_path = root / "manifest.json"
        manifest = validate_run_manifest_payload(json.loads(manifest_path.read_text(encoding="utf-8")))
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
            "token_budget": budget,
            "token_budget_unit": TOKEN_BUDGET_UNIT,
            "dirty_working_tree": False,
            "prereg_dirty": False,
        }
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in required.items()
            if manifest.get(key) != value
        }
        if manifest.get("dataset_hashes", {}).get("initial_checkpoint") != initial_hash:
            mismatches["dataset_hashes.initial_checkpoint"] = {
                "expected": initial_hash,
                "observed": manifest.get("dataset_hashes", {}).get("initial_checkpoint"),
            }
        if mismatches:
            raise ValueError(f"pilot cell provenance mismatch for {cell}: {mismatches}")
        if not any(key.startswith("prerequisite_probe") for key in manifest["dataset_hashes"]):
            raise ValueError(f"pilot cell {cell} did not bind its frozen probe manifest")
        metrics = root / "metrics.jsonl"
        checkpoint = Path(str(manifest.get("final_checkpoint_path", "")))
        if not checkpoint.resolve().is_relative_to((root / "checkpoints").resolve()):
            raise ValueError(f"pilot final checkpoint is outside its cell directory: {cell}")
        if sha256_file(metrics) != manifest.get("metrics_sha256"):
            raise ValueError(f"pilot metrics changed before terminal binding: {cell}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("final_checkpoint_sha256"):
            raise ValueError(f"pilot final checkpoint changed before terminal binding: {cell}")
        consumed = int(manifest.get("token_budget_consumed", -1))
        if not 0 <= consumed <= budget or not manifest.get("training_stop_reason"):
            raise ValueError(f"pilot token budget evidence is incomplete for {cell}")
        resolved = root / "resolved_config.yaml"
        resolved_payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(resolved_payload, dict):
            raise ValueError(f"pilot resolved config is not a mapping: {cell}")
        resolved_required = {
            "experiment.name": cell,
            "seed": 42,
            "protocol_track": expected["protocol_track"],
            "prereg_path": expected["prereg_path"],
            "prereg_version": expected["prereg_version"],
            "model.model_revision": expected["model_revision"],
            "teacher.model_revision": expected["teacher_revision"],
            "trainer.token_budget": budget,
            "trainer.token_budget_unit": TOKEN_BUDGET_UNIT,
            "production_safety.initial_checkpoint_hash": initial_hash,
        }

        def nested_value(dotted: str, payload: dict[str, Any] = resolved_payload) -> Any:
            value: Any = payload
            for part in dotted.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value

        resolved_mismatches = {
            key: {"expected": value, "observed": nested_value(key)}
            for key, value in resolved_required.items()
            if nested_value(key) != value
        }
        if resolved_mismatches:
            raise ValueError(f"pilot resolved config mismatch for {cell}: {resolved_mismatches}")
        dataset_hashes = manifest.get("dataset_hashes", {})
        probe_hashes = sorted(
            str(value) for key, value in dataset_hashes.items() if key.startswith("prerequisite_probe")
        )
        binding = {
            "cell": cell,
            "seed": 42,
            "resolved_config_sha256": sha256_file(resolved),
            "metrics_sha256": sha256_file(metrics),
            "final_checkpoint_sha256": sha256_file(checkpoint),
            "slurm_terminal_evidence_sha256": terminal_hash,
            "slurm": tasks[index],
            "token_budget": budget,
            "token_budget_consumed": consumed,
            "token_budget_unit": TOKEN_BUDGET_UNIT,
            "training_stop_reason": manifest["training_stop_reason"],
            "dataset_hashes_sha256": sha256_value(dataset_hashes),
            "probe_manifest_hashes": probe_hashes,
            "initial_checkpoint_sha256": initial_hash,
            "state_source_artifact_sha256": str(manifest.get("rollout_bank_hash", "")),
        }
        binding["sha256"] = sha256_value(binding)
        manifest["pilot_terminal_binding"] = binding
        manifest["slurm_terminal_evidence_sha256"] = terminal_hash
        manifest["sha256"] = sha256_value({key: value for key, value in manifest.items() if key != "sha256"})
        atomic_write_json(manifest_path, manifest)
        bound[cell] = {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}
    result: dict[str, Any] = {"job_id": job_id, "terminal_sha256": terminal_hash, "cells": bound}
    result["sha256"] = sha256_value(result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = bind_pilot_training_outputs(
        run_dir=args.run_dir,
        terminal=args.terminal,
        job_id=args.job_id,
        config=compose_config(args.overrides),
    )
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
