"""Produce JSON and Markdown production go/no-go reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from posttrain_circuits.circuits.model_adapter import check_hf_identity_compatibility
from posttrain_circuits.circuits.probe_cohorts import validate_probe_cohort_manifest
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.readiness import build_readiness_report, validate_anti_shortcut_report
from posttrain_circuits.data.splits import assert_split_isolation, build_split
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.utils.smoke import build_fixed_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate production go/no-go evidence")
    parser.add_argument("--output", type=Path, default=Path("outputs/readiness"))
    parser.add_argument("--anti-shortcut-report", type=Path)
    parser.add_argument("--probe-cohort-manifest", type=Path)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Hash-bound production evidence bundle; omitted mode is diagnostic NO-GO only.",
    )
    args = parser.parse_args(argv)
    if args.evidence is not None:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        checks = payload.get("checks")
        bindings = payload.get("bindings")
        if not isinstance(checks, dict) or not isinstance(bindings, dict):
            raise ValueError("production readiness evidence requires checks and bindings mappings")
        evidence = {}
        for name, row in checks.items():
            if not isinstance(row, dict) or set(row) < {"passed", "evidence"}:
                raise ValueError(f"malformed readiness evidence for {name}")
            evidence[str(name)] = (bool(row["passed"]), str(row["evidence"]))
        report = build_readiness_report(
            evidence,
            bindings={str(key): str(value) for key, value in bindings.items()},
        )
        report.write(args.output)
        print_json({"ready": report.ready, "output": str(args.output), "mode": "production"})
        return
    task = ProofGraphTask()
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen(42).eval()
    example = task.generate(2, {"positive": True})
    target = task.canonical_target(example)
    deterministic = (
        task.verify(example, task.parse_response(target)).reward
        == task.verify(example, task.parse_response(target)).reward
    )
    bank = build_fixed_bank(build_smoke_examples(8), tokenizer, 42)
    rewards = {float(record.verifier_reward) for record in bank if record.verifier_reward is not None}
    discovery = build_split(task, "circuit_discovery", 4, 42)
    validation = build_split(task, "circuit_validation", 4, 42)
    try:
        assert_split_isolation({"discovery": discovery, "validation": validation})
        isolated = True
    except ValueError:
        isolated = False
    ids = torch.tensor([tokenizer.encode(task.render(example), add_special_tokens=False)])
    compatibility = check_hf_identity_compatibility(model, ids)
    anti_path = args.anti_shortcut_report or (args.output / "anti_shortcut.json")
    anti_evidence = (False, f"no anti-shortcut report at {anti_path}")
    if anti_path.is_file():
        raw_anti = json.loads(anti_path.read_text(encoding="utf-8"))
        try:
            anti = validate_anti_shortcut_report(
                anti_path,
                max_shortcut_gap=float(raw_anti["max_shortcut_gap"]),
                expected_model_checkpoint_hash=str(raw_anti["model_checkpoint_hash"]),
            )
            anti_evidence = (
                True,
                (
                    f"shortcut_gap={float(anti['shortcut_gap']):.6f} <= "
                    f"{float(anti['max_shortcut_gap']):.6f}; report={anti['sha256']}"
                ),
            )
        except (ValueError, RuntimeError, KeyError) as error:
            anti_evidence = (False, str(error))
    probe_path = args.probe_cohort_manifest or Path("outputs/probes/proofgraph/manifest.json")
    probe_evidence = (False, f"no frozen probe cohort manifest at {probe_path}")
    if probe_path.is_file():
        try:
            probes = validate_probe_cohort_manifest(probe_path)
            probe_evidence = (
                True,
                (f"base_capable/challenge discovery/validation frozen; manifest={probes['sha256']}"),
            )
        except (ValueError, KeyError) as error:
            probe_evidence = (False, str(error))
    evidence = {
        "verifier_deterministic": (deterministic, "exact verifier repeated identically on Phase-0"),
        "fixed_bank_mixed_rewards": (rewards == {0.0, 1.0}, f"observed rewards={sorted(rewards)}"),
        "hf_circuit_logit_parity": (
            False,
            "tiny HF identity hooks pass, but production HF/TransformerLens parity is not yet measured",
        ),
        "checkpoint_resume_verified": (
            False,
            "run the checkpoint deterministic-resume test in the target distributed environment",
        ),
        "split_leakage_absent": (isolated, "discovery and validation semantic hashes are disjoint"),
        "anti_shortcut_gap": anti_evidence,
        "probe_cohorts_frozen": probe_evidence,
    }
    report = build_readiness_report(
        evidence,
        bindings={
            "initial_checkpoint_hash": "diagnostic-tiny-not-production",
            "dataset_hash": "diagnostic-tiny-not-production",
            "suite_hash": "diagnostic-tiny-not-production",
            "code_commit": "unavailable",
            "prereg_commit": "unavailable",
        },
    )
    report.write(args.output)
    print_json(
        {
            "ready": report.ready,
            "output": str(args.output),
            "compatibility_identity_passed": compatibility.passed,
        }
    )


if __name__ == "__main__":
    main()
