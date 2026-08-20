"""Run the hash-bound ProofGraph surface-label leakage audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import (
    formal_artifact_binding,
    require_git_output,
    resolve_preregistration,
)
from posttrain_circuits.data.splits import load_frozen_split
from posttrain_circuits.tasks.proofgraph.label_leakage import audit_label_leakage


def _git(args: list[str]) -> str:
    return require_git_output(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit paired ProofGraph label leakage")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-query-only-accuracy", type=float, default=0.55)
    parser.add_argument("--maximum-surface-feature-accuracy", type=float, default=0.55)
    parser.add_argument("--maximum-bow-accuracy", type=float, default=0.60)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    examples, manifest = load_frozen_split(args.split_root, expected_split=args.split)
    prereg_commit = resolve_preregistration(config).git_commit
    result = audit_label_leakage(
        examples,
        maximum_query_only_accuracy=args.maximum_query_only_accuracy,
        maximum_surface_feature_accuracy=args.maximum_surface_feature_accuracy,
        maximum_bow_accuracy=args.maximum_bow_accuracy,
        dataset_hash=str(manifest["sha256"]),
        code_commit=_git(["rev-parse", "HEAD"]),
        prereg_commit=prereg_commit,
    )
    result.update(formal_artifact_binding(config))
    result["sha256"] = sha256_value({key: value for key, value in result.items() if key != "sha256"})
    atomic_write_json(args.output, result)
    if result["passed"] is not True:
        raise SystemExit("ProofGraph label-leakage audit failed")


if __name__ == "__main__":
    main()
