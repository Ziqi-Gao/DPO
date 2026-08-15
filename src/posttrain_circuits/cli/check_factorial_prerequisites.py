"""Read-only production factorial preflight gate."""

from __future__ import annotations

import argparse

from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.readiness import require_factorial_prerequisites


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check frozen pre-training factorial evidence")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    evidence = require_factorial_prerequisites(config)
    print_json({"passed": True, "evidence": evidence})


if __name__ == "__main__":
    main()
