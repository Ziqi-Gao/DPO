"""Resolve one preregistered pilot circuit array row."""

from __future__ import annotations

import argparse

from posttrain_circuits.circuits.pilot_scope import resolve_pilot_circuit_scope
from posttrain_circuits.core.config import compose_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--matrix", choices=("initial", "final"), required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    scope = resolve_pilot_circuit_scope(compose_config(args.overrides))
    rows = scope[f"{args.matrix}_matrix"]
    if not 0 <= args.index < len(rows):
        raise ValueError("pilot circuit array index is outside the frozen scope")
    row = rows[args.index]
    fields = (
        ("cell", "stage_label", "probe_stage", "cohort")
        if args.matrix == "final"
        else ("stage_label", "probe_stage", "cohort")
    )
    print("\t".join(str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
