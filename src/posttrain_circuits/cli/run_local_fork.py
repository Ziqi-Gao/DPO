"""Run identical-state signal branches at configured update horizons."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from posttrain_circuits.artifacts.compatibility import scientific_compatibility_fields
from posttrain_circuits.artifacts.hashing import canonical_json, sha256_file, sha256_value
from posttrain_circuits.artifacts.io import publish_json_once, publish_path_no_clobber
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.experiments.protocols.local_fork import (
    LOCAL_FORK_SPEC,
    validate_local_fork_report,
)
from posttrain_circuits.models.loading import load_model_and_tokenizer
from posttrain_circuits.learning.training.local_fork import (
    calibrate_learning_rate_for_output_kl,
    load_fork_bundle,
    output_kl_match_status,
    run_branch,
    state_hash,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _checkpoint_tree_state_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f"local-fork checkpoint staging tree has invalid entry: {path}")
        relative = str(path.relative_to(root))
        if path.is_dir():
            index[f"directory:{relative}"] = "directory"
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            index[f"file:{relative}"] = state_hash(payload)
    return index


def _publish_checkpoint_tree_once(staging: Path, destination: Path) -> None:
    expected = _checkpoint_tree_state_index(staging)
    try:
        publish_path_no_clobber(staging, destination)
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or _checkpoint_tree_state_index(destination) != expected
        ):
            raise FileExistsError(
                f"conflicting local-fork checkpoint tree already exists: {destination}"
            )
        shutil.rmtree(staging)


def _retarget_checkpoint_evidence(
    value: object,
    *,
    staging: Path,
    final: Path,
    _visited: set[int] | None = None,
) -> None:
    visited = set() if _visited is None else _visited
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "payload_state_sha256", "phase", "state_hashes"}:
            current = Path(str(value["path"]))
            relative = current.resolve(strict=False).relative_to(staging.resolve(strict=False))
            target = (final / relative).resolve(strict=True)
            value["path"] = str(target)
            value["sha256"] = sha256_file(target)
        for child in value.values():
            _retarget_checkpoint_evidence(
                child,
                staging=staging,
                final=final,
                _visited=visited,
            )
    elif isinstance(value, list):
        for child in value:
            _retarget_checkpoint_evidence(
                child,
                staging=staging,
                final=final,
                _visited=visited,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local shared-state fork branches")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/local_fork/results.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(LOCAL_FORK_SPEC.horizons),
    )
    parser.add_argument(
        "--output-kl-relative-tolerance",
        type=float,
        default=LOCAL_FORK_SPEC.output_kl_relative_tolerance,
    )
    parser.add_argument(
        "--max-calibration-rounds",
        type=int,
        default=LOCAL_FORK_SPEC.maximum_calibration_rounds,
    )
    args = parser.parse_args(argv)
    if len(set(args.horizons)) != len(args.horizons) or any(
        horizon not in LOCAL_FORK_SPEC.horizons for horizon in args.horizons
    ):
        raise ValueError("local-fork horizons must be a unique subset of the registered horizons")
    if args.horizons != [
        horizon for horizon in LOCAL_FORK_SPEC.horizons if horizon in set(args.horizons)
    ]:
        raise ValueError("local-fork horizons must preserve registered order")
    if args.output_kl_relative_tolerance != LOCAL_FORK_SPEC.output_kl_relative_tolerance:
        raise ValueError("local-fork output-KL tolerance differs from LocalForkSpec")
    if args.max_calibration_rounds != LOCAL_FORK_SPEC.maximum_calibration_rounds:
        raise ValueError("local-fork calibration-round limit differs from LocalForkSpec")
    payload = load_fork_bundle(args.bundle)
    model_spec = dict(payload.get("model_spec", {}))
    formal_binding = dict(payload["manifest"].get("formal_binding", {}))
    if str(model_spec.get("protocol_track", "")).startswith("qwen3_") and not formal_binding:
        raise ValueError("Qwen3 local-fork bundle lacks its formal protocol binding")
    if str(model_spec.get("model_name_or_path", "local/")).startswith("local/"):
        tokenizer = build_tiny_tokenizer()

        def model_factory():  # type: ignore[no-untyped-def]
            return build_tiny_qwen(int(model_spec.get("seed", args.seed)))

    else:
        tokenizer = load_model_and_tokenizer(
            model_spec,
            for_training=False,
        ).tokenizer

        def model_factory():  # type: ignore[no-untyped-def]
            model = load_model_and_tokenizer(
                model_spec,
                for_training=True,
            ).model
            if torch.cuda.is_available():
                model.to(torch.device("cuda"))
            return model

    base_learning_rate = float(payload["optimizer"]["param_groups"][0]["lr"])
    results = []
    invalid_cells: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint_root = args.output.parent / "checkpoints"
    checkpoint_root = Path(
        tempfile.mkdtemp(
            prefix=".local-fork-checkpoints.stage-",
            dir=args.output.parent,
        )
    )
    for horizon in args.horizons:
        unmatched = {}
        for branch in LOCAL_FORK_SPEC.branch_ids:
            unmatched[branch] = run_branch(
                bundle_payload=payload,
                model_factory=model_factory,
                branch=branch,
                horizon=horizon,
                pad_token_id=tokenizer.pad_token_id,
                checkpoint_root=(checkpoint_root / f"horizon-{horizon}" / branch / "unmatched"),
                comparison_mode="unmatched",
                calibration_round=0,
            )
        target = float(unmatched["hard_teacher"]["probe_output_kl_new_to_fork"])
        if target <= 0:
            raise RuntimeError("hard-teacher reference produced zero output KL; matching is undefined")
        for branch in LOCAL_FORK_SPEC.branch_ids:
            observed = float(unmatched[branch]["probe_output_kl_new_to_fork"])
            current_lr = base_learning_rate
            matched = unmatched[branch]
            status = output_kl_match_status(
                target,
                observed,
                relative_tolerance=args.output_kl_relative_tolerance,
            )
            rounds = 0
            calibration_trace: list[dict[str, object]] = []
            while rounds < args.max_calibration_rounds and not bool(status["within_tolerance"]):
                current_lr = calibrate_learning_rate_for_output_kl(target, observed, current_lr)
                rounds += 1
                matched = run_branch(
                    bundle_payload=payload,
                    model_factory=model_factory,
                    branch=branch,
                    horizon=horizon,
                    pad_token_id=tokenizer.pad_token_id,
                    checkpoint_root=(
                        checkpoint_root
                        / f"horizon-{horizon}"
                        / branch
                        / "matched-output-kl"
                        / f"round-{rounds}"
                    ),
                    comparison_mode="matched",
                    calibration_round=rounds,
                    learning_rate_override=current_lr,
                )
                observed = float(matched["probe_output_kl_new_to_fork"])
                status = output_kl_match_status(
                    target,
                    observed,
                    relative_tolerance=args.output_kl_relative_tolerance,
                )
                calibration_trace.append(
                    {
                        "round": rounds,
                        "learning_rate": current_lr,
                        "target_output_kl_new_to_fork": target,
                        "observed_output_kl_new_to_fork": observed,
                        "absolute_error": status["absolute_error"],
                        "relative_error": status["relative_error"],
                        "within_tolerance": status["within_tolerance"],
                        "result": matched,
                    }
                )
            results.append(
                {
                    "branch": branch,
                    "horizon": horizon,
                    "primary_comparison_axis": LOCAL_FORK_SPEC.primary_comparison_axis,
                    "secondary_comparison_axes": list(
                        LOCAL_FORK_SPEC.secondary_comparison_axes
                    ),
                    "unmatched": unmatched[branch],
                    "matched": matched,
                    "calibration_trace": calibration_trace,
                    "calibration": {
                        "target_output_kl_new_to_fork": target,
                        "observed_unmatched_output_kl": float(
                            unmatched[branch]["probe_output_kl_new_to_fork"]
                        ),
                        "matched_output_kl": observed,
                        "matched_absolute_error": status["absolute_error"],
                        "matched_relative_error": status["relative_error"],
                        "within_tolerance": status["within_tolerance"],
                        "relative_tolerance": args.output_kl_relative_tolerance,
                        "calibration_rounds": rounds,
                        "learning_rate": current_lr,
                        "matched_parameter_update_norm": matched["parameter_update_norm"],
                    },
                }
            )
            if not bool(status["within_tolerance"]):
                invalid_cells.append(
                    {
                        "branch": branch,
                        "horizon": horizon,
                        "target_output_kl": target,
                        "observed_output_kl": observed,
                        "relative_error": status["relative_error"],
                    }
                )
    _publish_checkpoint_tree_once(checkpoint_root, final_checkpoint_root)
    _retarget_checkpoint_evidence(
        results,
        staging=checkpoint_root,
        final=final_checkpoint_root,
    )
    checkpoint_root = final_checkpoint_root
    report = {
        "format_version": 4,
        **scientific_compatibility_fields(str(formal_binding.get("prereg_version", "core_v2"))),
        **formal_binding,
        "bundle": payload["manifest"],
        "bundle_file": {
            "path": str(args.bundle.resolve(strict=True)),
            "sha256": sha256_file(args.bundle),
        },
        "bundle_id": payload["manifest"]["bundle_id"],
        "bundle_manifest_sha256": payload["manifest"]["manifest_sha256"],
        "local_fork_protocol_id": payload["manifest"]["protocol_id"],
        "local_fork_spec_sha256": payload["manifest"]["local_fork_spec_sha256"],
        "requested_horizons": list(args.horizons),
        "requested_output_kl_relative_tolerance": args.output_kl_relative_tolerance,
        "requested_maximum_calibration_rounds": args.max_calibration_rounds,
        "full_protocol_conformant": (
            tuple(args.horizons) == LOCAL_FORK_SPEC.horizons
            and args.output_kl_relative_tolerance
            == LOCAL_FORK_SPEC.output_kl_relative_tolerance
            and args.max_calibration_rounds == LOCAL_FORK_SPEC.maximum_calibration_rounds
            and not invalid_cells
        ),
        "results": results,
        "valid_for_primary_analysis": not invalid_cells,
        "invalid_cells": invalid_cells,
    }
    report = json.loads(canonical_json(report))
    report["sha256"] = sha256_value(report)
    validate_local_fork_report(
        report,
        checkpoint_root=checkpoint_root,
        bundle_path=args.bundle,
        require_source_binding=False,
        source_model_factory=model_factory,
    )
    publish_json_once(args.output, report)
    if invalid_cells:
        raise RuntimeError(
            "local-fork output-KL matching remained outside tolerance after calibration; "
            f"invalid_cells={invalid_cells}"
        )
    print_json(
        {
            "output": str(args.output),
            "branches_completed": len(results),
        }
    )


if __name__ == "__main__":
    main()
