"""Build all seven ProofGraph splits and globally reject leakage."""

from __future__ import annotations

from pathlib import Path

from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.manifests import write_dataset_family
from posttrain_circuits.datasets.proofgraph.splits import (
    SPLITS,
    build_all_splits,
)


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Build all isolated ProofGraph splits", argv)
    output = args.output or Path(config["output_root"]) / "datasets" / "proofgraph"
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return
    task_config = dict(config["task"])
    default_size = int(task_config["num_examples"])
    configured_sizes = task_config.get("split_sizes", {})
    split_sizes = {split: int(configured_sizes.get(split, default_size)) for split in SPLITS}
    task = ProofGraphTask()
    splits = build_all_splits(
        task,
        split_sizes=split_sizes,
        base_seed=int(task_config.get("seed", config["seed"])),
        difficulty=task_config,
    )
    manifest = write_dataset_family(output, splits)
    print_json({"output": str(output), "manifest": manifest})


if __name__ == "__main__":
    main()
