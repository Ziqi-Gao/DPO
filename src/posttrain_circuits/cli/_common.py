"""CLI parsing and production guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from posttrain_circuits.core.config import compose_config, is_production_scale


def parse_cli(description: str, argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    return args, config


def dry_run_report(config: dict[str, Any], output: Path) -> dict[str, Any]:
    trainer = config.get("trainer", {})
    task = config.get("task", {})
    report = {
        "resolved_configuration": config,
        "expected_model": config.get("model", {}).get("model_name_or_path"),
        "expected_dataset": task.get("name"),
        "estimated_prompts": int(trainer.get("max_steps", 0)) * int(trainer.get("batch_size", 1)),
        "estimated_tokens": int(trainer.get("token_budget", 0)),
        "output_directory": str(output),
        "production_scale": is_production_scale(config),
    }
    print(yaml.safe_dump(report, sort_keys=False))
    return report


def enforce_production_guard(
    config: dict[str, Any], *, dry_run: bool, confirm_production: bool, output: Path
) -> bool:
    production = is_production_scale(config)
    if dry_run:
        dry_run_report(config, output)
        return False
    if production and not confirm_production:
        raise SystemExit(
            "production-scale run refused: inspect --dry-run, then pass --confirm-production explicitly"
        )
    return True


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))
