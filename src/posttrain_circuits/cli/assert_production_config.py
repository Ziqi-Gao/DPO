"""Resolve and reject production training configurations with inherited smoke defaults."""

from __future__ import annotations

import argparse

from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.config import compose_config, validate_production_training_config
from posttrain_circuits.core.hashing import sha256_value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assert a fully resolved production configuration")
    parser.add_argument("overrides", nargs="+")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    validate_production_training_config(config)
    print_json(
        {
            "passed": True,
            "resolved_config_sha256": sha256_value(config),
            "model": config["model"]["model_name_or_path"],
            "teacher": config["teacher"]["model_name_or_path"],
            "task": config["task"]["name"],
            "backend": config["trainer"]["backend"],
            "max_steps": config["trainer"]["max_steps"],
            "token_budget": config["trainer"]["token_budget"],
            "validation_split_path": config["task"]["validation_split_path"],
        }
    )


if __name__ == "__main__":
    main()
