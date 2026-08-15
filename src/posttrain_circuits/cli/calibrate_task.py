"""Calibrate fixture success into the configured nontrivial range."""

from __future__ import annotations

from pathlib import Path

from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.utils.smoke import build_fixed_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Calibrate ProofGraph success", argv)
    output = args.output or Path(config["output_root"]) / "calibration" / "proofgraph.json"
    if not enforce_production_guard(
        config, dry_run=args.dry_run, confirm_production=args.confirm_production, output=output
    ):
        return
    tokenizer = build_tiny_tokenizer()
    records = build_fixed_bank(build_smoke_examples(20, int(config["seed"])), tokenizer, int(config["seed"]))
    rate = sum(record.verifier_reward == 1.0 for record in records) / len(records)
    target = config["state_source"].get("target_success_range", [0.20, 0.60])
    print_json(
        {
            "success_rate": rate,
            "target_success_range": target,
            "accepted": float(target[0]) <= rate <= float(target[1]),
            "backend": "deterministic_smoke_fixture",
        }
    )


if __name__ == "__main__":
    main()
