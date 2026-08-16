"""Run the hash-bound ProofGraph surface-label leakage audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import PREREG_PATH, require_git_output
from posttrain_circuits.data.splits import load_frozen_split
from posttrain_circuits.tasks.proofgraph.label_leakage import audit_label_leakage


def _git(args: list[str]) -> str:
    return require_git_output(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit paired ProofGraph label leakage")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-query-only-accuracy", type=float, default=0.55)
    parser.add_argument("--maximum-surface-feature-accuracy", type=float, default=0.55)
    parser.add_argument("--maximum-bow-accuracy", type=float, default=0.60)
    args = parser.parse_args(argv)
    examples, manifest = load_frozen_split(args.split_root, expected_split=args.split)
    prereg_commit = _git(["log", "-n", "1", "--format=%H", "--", str(PREREG_PATH)])
    result = audit_label_leakage(
        examples,
        maximum_query_only_accuracy=args.maximum_query_only_accuracy,
        maximum_surface_feature_accuracy=args.maximum_surface_feature_accuracy,
        maximum_bow_accuracy=args.maximum_bow_accuracy,
        dataset_hash=str(manifest["sha256"]),
        code_commit=_git(["rev-parse", "HEAD"]),
        prereg_commit=prereg_commit,
    )
    atomic_write_json(args.output, result)
    if result["passed"] is not True:
        raise SystemExit("ProofGraph label-leakage audit failed")


if __name__ == "__main__":
    main()
