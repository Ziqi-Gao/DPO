"""Run identical-state signal branches at configured update horizons."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields
from posttrain_circuits.models.loading import load_model_and_tokenizer
from posttrain_circuits.training.local_fork import (
    calibrate_learning_rate_for_output_kl,
    load_fork_bundle,
    output_kl_match_status,
    run_branch,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)

_BRANCHES = (
    "hard_teacher",
    "soft_teacher",
    "verified_replay",
    "centered_policy_gradient",
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
        default=[1, 5, 20],
    )
    parser.add_argument("--output-kl-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--max-calibration-rounds", type=int, default=2)
    args = parser.parse_args(argv)
    payload = load_fork_bundle(args.bundle)
    model_spec = dict(payload.get("model_spec", {}))
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
    checkpoint_root = args.output.parent / "checkpoints"
    for horizon in args.horizons:
        unmatched = {}
        for branch in _BRANCHES:
            unmatched[branch] = run_branch(
                bundle_payload=payload,
                model_factory=model_factory,
                branch=branch,
                horizon=horizon,
                pad_token_id=tokenizer.pad_token_id,
                checkpoint_root=(checkpoint_root / f"horizon-{horizon}" / branch / "unmatched"),
            )
        target = float(unmatched["hard_teacher"]["probe_output_kl_new_to_fork"])
        if target <= 0:
            raise RuntimeError("hard-teacher reference produced zero output KL; matching is undefined")
        for branch in _BRANCHES:
            observed = float(unmatched[branch]["probe_output_kl_new_to_fork"])
            current_lr = base_learning_rate
            matched = unmatched[branch]
            status = output_kl_match_status(
                target,
                observed,
                relative_tolerance=args.output_kl_relative_tolerance,
            )
            rounds = 0
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
                    learning_rate_override=current_lr,
                )
                observed = float(matched["probe_output_kl_new_to_fork"])
                status = output_kl_match_status(
                    target,
                    observed,
                    relative_tolerance=args.output_kl_relative_tolerance,
                )
            results.append(
                {
                    "branch": branch,
                    "horizon": horizon,
                    "primary_comparison_axis": "matched_output_kl_new_to_fork",
                    "secondary_comparison_axes": ["update_count", "parameter_update_norm"],
                    "unmatched": unmatched[branch],
                    "matched": matched,
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
    report = {
        "format_version": 2,
        **scientific_compatibility_fields(),
        "bundle": payload["manifest"],
        "results": results,
        "valid_for_primary_analysis": not invalid_cells,
        "invalid_cells": invalid_cells,
    }
    report["sha256"] = sha256_value(report)
    atomic_write_json(args.output, report)
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
