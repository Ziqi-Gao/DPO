"""Score an existing trajectory store with a pinned teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import ROLLOUT_GENERATION_VERSION
from posttrain_circuits.artifacts.runs import (
    PROTOCOL_AMENDMENT_BINDING_FIELDS,
    formal_artifact_binding,
)
from posttrain_circuits.cli._common import (
    dry_run_report,
    print_json,
)
from posttrain_circuits.core.config import compose_config, is_production_scale
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.learning.contracts import TrajectoryBatch
from posttrain_circuits.models.loading import (
    load_model_and_tokenizer,
    move_model_to_local_cuda,
    tokenizer_fingerprint,
)
from posttrain_circuits.learning.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.utils.smoke import build_fixed_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


_FORMAL_SOURCE_BINDING_FIELDS = (
    "protocol_track",
    "artifact_namespace",
    "model_revision",
    "teacher_revision",
    "tokenizer_revision",
    "tokenizer_fingerprint",
    "chat_template_sha256",
    "prompt_protocol",
    "enable_thinking",
    "code_commit",
    "prereg_path",
    "prereg_version",
    "prereg_commit",
    "prereg_sha256",
    *PROTOCOL_AMENDMENT_BINDING_FIELDS,
)


def _override_key(override: str) -> str:
    return override.split("=", 1)[0]


def _compose_score_teacher_config(overrides: list[str]) -> dict[str, Any]:
    """Ignore the generation-only candidate count for a fixed rollout bank."""

    candidate_key = "state_source.num_candidates"
    filtered = [override for override in overrides if _override_key(override) != candidate_key]
    if len(filtered) == len(overrides):
        return compose_config(overrides)
    candidate = compose_config(filtered)
    if candidate.get("state_source", {}).get("name") == "fixed_bank":
        return candidate
    # Non-fixed sources own this field, so compose the original request and
    # preserve normal unknown-key validation.
    return compose_config(overrides)


def _source_formal_metadata(
    source_manifest: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and propagate the complete producer binding when applicable."""

    if expected is not None:
        mismatches = {
            key: {"expected": value, "observed": source_manifest.get(key)}
            for key, value in expected.items()
            if source_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"rollout-bank formal binding mismatch: {mismatches}")
    return {
        key: source_manifest[key]
        for key in _FORMAL_SOURCE_BINDING_FIELDS
        if key in source_manifest
    }


def _score(
    records: list[Any],
    model: Any,
    *,
    teacher_id: str,
    teacher_revision: str,
    top_k: int,
    minimum_retained_mass: float,
) -> list[Any]:
    scorer = HuggingFaceTeacherScorer(
        model,
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        top_k=top_k,
        minimum_retained_mass=minimum_retained_mass,
    )
    return scorer.score(TrajectoryBatch(records, 0)).records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cache teacher top-k next-token distributions")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--bank", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/teacher_scores"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    args = parser.parse_args(argv)
    config = _compose_score_teacher_config(args.overrides)
    production = is_production_scale(config)
    if args.dry_run:
        dry_run_report(config, args.output)
        return
    if production and not args.confirm_production:
        raise SystemExit(
            "production teacher scoring refused: inspect --dry-run, then pass --confirm-production"
        )

    if production:
        if args.bank is None:
            raise ValueError("production teacher scoring requires --bank")
        source_store = TrajectoryStore(args.bank)
        source_manifest = source_store.check_integrity()
        expected_formal = None
        if str(config.get("protocol_track", "")).startswith("qwen3_"):
            expected_formal = formal_artifact_binding(config)
            expected_source = {
                "protocol_track": config["protocol_track"],
                "artifact_namespace": config["model"]["artifact_namespace"],
                "prompt_protocol": "qwen3_non_thinking_v1",
                "enable_thinking": False,
                "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
                "tokenizer_hash": config["model"]["tokenizer_fingerprint"],
            }
            mismatches = {
                key: {"expected": value, "observed": source_manifest.get(key)}
                for key, value in expected_source.items()
                if source_manifest.get(key) != value
            }
            if source_manifest.get("behavior_policy", {}).get("id") != config["model"]["model_name_or_path"]:
                mismatches["behavior_policy.id"] = {
                    "expected": config["model"]["model_name_or_path"],
                    "observed": source_manifest.get("behavior_policy", {}).get("id"),
                }
            if mismatches:
                raise ValueError(f"Qwen3 refused stale/cross-model rollout bank: {mismatches}")
        source_formal_metadata = _source_formal_metadata(
            source_manifest,
            expected=expected_formal,
        )
        records = source_store.read()
        loaded = load_model_and_tokenizer(
            config["teacher"],
            for_training=False,
        )
        source_tokenizer_hash = source_manifest.get("tokenizer_hash")
        if source_tokenizer_hash is not None and source_tokenizer_hash != loaded.tokenizer_hash:
            raise ValueError("teacher tokenizer is incompatible with rollout-bank token IDs")
        teacher_id = loaded.model_id
        teacher_revision = loaded.resolved_model_commit
        teacher_commit = loaded.resolved_model_commit
        tokenizer_hash = loaded.tokenizer_hash
        model = move_model_to_local_cuda(loaded.model)
    else:
        tokenizer = build_tiny_tokenizer()
        records = build_fixed_bank(build_smoke_examples(4), tokenizer, 42)
        source_manifest = {
            "behavior_policy": {
                "id": "common_mu_smoke",
                "revision": "local-smoke-v1",
            },
            "prompt_manifest_hash": "smoke-prompts-v1",
            "sampling_configuration": {"temperature": 1.0, "top_p": 1.0},
            "verifier_version": "proofgraph-exact-v1",
            "sha256": "local-smoke-bank",
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
            "protocol_track": "core_v2",
            "artifact_namespace": "smoke",
            "prompt_protocol": records[0].prompt_protocol,
            "enable_thinking": records[0].enable_thinking,
            "chat_template_sha256": records[0].chat_template_sha256,
        }
        model = build_tiny_qwen(43)
        teacher_id = "local/tiny-teacher"
        teacher_revision = "local-random-v1"
        teacher_commit = teacher_revision
        tokenizer_hash = tokenizer_fingerprint(tokenizer)
        source_formal_metadata = _source_formal_metadata(source_manifest)

    supervision = config["supervision"]
    top_k = min(
        int(supervision.get("teacher_top_k", 128)),
        int(model.config.vocab_size),
    )
    scored = _score(
        records,
        model,
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        top_k=top_k,
        minimum_retained_mass=float(supervision.get("minimum_retained_mass", 0.0)),
    )
    retained_mass = [value for record in scored for value in record.teacher_topk_mass]
    if not retained_mass:
        raise RuntimeError("teacher scoring produced no retained-mass measurements")
    manifest = TrajectoryStore(args.output).write(
        scored,
        behavior_policy=dict(source_manifest["behavior_policy"]),
        prompt_manifest_hash=str(source_manifest["prompt_manifest_hash"]),
        sampling_configuration=dict(source_manifest["sampling_configuration"]),
        verifier_version=str(source_manifest["verifier_version"]),
        teacher_version=teacher_revision,
        top_k=top_k,
        extra_metadata={
            "store_kind": "teacher_scored_rollout_bank",
            "rollout_generation_version": str(source_manifest["rollout_generation_version"]),
            "source_bank_hash": str(source_manifest["sha256"]),
            "teacher_id": teacher_id,
            "teacher_revision": teacher_revision,
            "resolved_teacher_commit": teacher_commit,
            "tokenizer_hash": tokenizer_hash,
            "tokenizer_fingerprint": tokenizer_hash,
            "teacher_topk_mass": {
                "minimum": min(retained_mass),
                "mean": sum(retained_mass) / len(retained_mass),
                "positions": len(retained_mass),
            },
            "protocol_track": str(source_manifest["protocol_track"]),
            "artifact_namespace": str(source_manifest["artifact_namespace"]),
            "prompt_protocol": str(source_manifest["prompt_protocol"]),
            "enable_thinking": bool(source_manifest["enable_thinking"]),
            "chat_template_sha256": str(source_manifest["chat_template_sha256"]),
            **source_formal_metadata,
        },
    )
    print_json(
        {
            "output": str(args.output),
            "trajectories": len(scored),
            "manifest": manifest,
        }
    )


if __name__ == "__main__":
    main()
