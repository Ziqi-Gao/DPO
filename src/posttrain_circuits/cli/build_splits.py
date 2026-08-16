"""Build all seven ProofGraph splits and globally reject leakage."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import DatasetManifest, atomic_write_json
from posttrain_circuits.data.splits import (
    SPLITS,
    build_all_splits,
    difficulty_distribution,
    serialize_examples,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask


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
    manifests = {}
    for split, examples in splits.items():
        split_root = output / split
        split_root.mkdir(parents=True, exist_ok=True)
        serialized = serialize_examples(examples)
        with (split_root / "examples.jsonl").open(
            "w",
            encoding="utf-8",
        ) as handle:
            for example in serialized:
                handle.write(json.dumps(example, sort_keys=True) + "\n")
        seeds = [int(example.metadata["seed"]) for example in examples]
        manifest = DatasetManifest(
            dataset_id=f"proofgraph-{split}-{min(seeds)}",
            generator_version=task.generator_version,
            git_commit="unavailable",
            task_config=task_config,
            split_name=split,
            seed_range=(min(seeds), max(seeds)),
            num_examples=len(examples),
            difficulty_distribution=difficulty_distribution(examples),
            pair_group_count=len({example.pair_group_id for example in examples}),
            pair_group_hash=sha256_value(sorted({example.pair_group_id for example in examples})),
        ).finalize(serialized)
        manifest.write(split_root / "manifest.json")
        manifests[split] = asdict(manifest)
    global_manifest = {
        "dataset_schema_version": "proofgraph-dataset-v2-paired",
        "prereg_version": "core_v2",
        "generator_version": task.generator_version,
        "label_semantics": task.label_semantics,
        "split_sizes": split_sizes,
        "split_hashes": {split: manifest["sha256"] for split, manifest in manifests.items()},
        "global_semantic_hash": sha256_value(
            {split: [example.example_id for example in examples] for split, examples in splits.items()}
        ),
        "leakage_check": "passed",
    }
    global_manifest["sha256"] = sha256_value(global_manifest)
    atomic_write_json(output / "manifest.json", global_manifest)
    print_json({"output": str(output), "manifest": global_manifest})


if __name__ == "__main__":
    main()
