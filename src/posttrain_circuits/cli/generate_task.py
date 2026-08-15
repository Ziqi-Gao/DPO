"""Generate an immutable ProofGraph split."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.core.manifests import DatasetManifest
from posttrain_circuits.data.splits import (
    build_split,
    difficulty_distribution,
    serialize_examples,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Generate a deterministic ProofGraph split", argv)
    task_config = config["task"]
    split = str(config.get("split", "train"))
    count = int(task_config.get("split_sizes", {}).get(split, task_config.get("num_examples", 32)))
    output = args.output or Path(config["output_root"]) / "datasets" / f"proofgraph-{split}"
    if not enforce_production_guard(
        config, dry_run=args.dry_run, confirm_production=args.confirm_production, output=output
    ):
        return
    seed = int(task_config.get("seed", config["seed"]))
    task = ProofGraphTask()
    examples = build_split(
        task,
        split,
        count,
        seed,
        dict(task_config),
    )
    output.mkdir(parents=True, exist_ok=True)
    serialized = serialize_examples(examples)
    with (output / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in serialized:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    seeds = [int(example.metadata["seed"]) for example in examples]
    manifest = DatasetManifest(
        dataset_id=f"proofgraph-{split}-{min(seeds)}",
        generator_version=task.generator_version,
        git_commit=(
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if not str(config["model"]["model_name_or_path"]).startswith("local/")
            else "smoke-unversioned"
        ),
        task_config=dict(task_config),
        split_name=split,
        seed_range=(min(seeds), max(seeds)),
        num_examples=count,
        difficulty_distribution=difficulty_distribution(examples),
    ).finalize(serialized)
    manifest.write(output / "manifest.json")
    print_json({"output": str(output), "manifest": asdict(manifest)})


if __name__ == "__main__":
    main()
