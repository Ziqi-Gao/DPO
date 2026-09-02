"""Freeze production local-fork prompts and multi-prompt KL probes from one scored bank."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from posttrain_circuits.artifacts.compatibility import (
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
    ROLLOUT_GENERATION_VERSION,
)
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import publish_json_once
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.experiments.protocols.local_fork import LOCAL_FORK_SPEC
from posttrain_circuits.models.prompt_protocol import prompt_protocol_name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hash-pinned production local-fork inputs")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--trajectory-store", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--minimum-group-size",
        type=int,
        default=LOCAL_FORK_SPEC.minimum_group_size,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.minimum_group_size < LOCAL_FORK_SPEC.minimum_group_size:
        raise ValueError("local-fork input group size is below LocalForkSpec")
    if args.limit < args.minimum_group_size:
        raise ValueError("local-fork input limit must fit at least one complete prompt group")
    config = compose_config(args.overrides)
    LOCAL_FORK_SPEC.validate_experiment_config(config["experiment"])
    store = TrajectoryStore(args.trajectory_store)
    bank_manifest = store.check_integrity()
    compatibility_expected = {
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
    }
    compatibility_mismatches = {
        key: {"expected": value, "observed": bank_manifest.get(key)}
        for key, value in compatibility_expected.items()
        if bank_manifest.get(key) != value
    }
    if compatibility_mismatches:
        raise ValueError(
            "local-fork trajectory bank compatibility changed: "
            f"{compatibility_mismatches}"
        )
    eligible = [
        record
        for record in store.read()
        if record.behavior_logprobs
        and record.teacher_topk_ids
        and record.teacher_topk_logprobs
        and record.verifier_reward is not None
    ]
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in eligible:
        if record.generation_group_id:
            grouped[record.generation_group_id].append(record)
    valid_groups: list[list[TrajectoryRecord]] = []
    for group_id, records in sorted(grouped.items()):
        records.sort(key=lambda record: record.generation_group_index)
        if len(records) < args.minimum_group_size:
            continue
        if len({record.prompt_id for record in records}) != 1:
            raise ValueError(f"generation group {group_id} mixes prompts")
        if len({float(record.verifier_reward or 0.0) for record in records}) < 2:
            continue
        if any(record.prompt_group_size != len(records) for record in records):
            raise ValueError(f"generation group {group_id} has inconsistent size metadata")
        valid_groups.append(records)
    selected: list[TrajectoryRecord] = []
    for records in valid_groups:
        if selected and len(selected) + len(records) > args.limit:
            break
        selected.extend(records)
    if not selected:
        raise ValueError("local-fork bank has no complete same-prompt generation group with reward variance")
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["tokenizer_name_or_path"]),
        revision=str(model_config["tokenizer_revision"]),
        local_files_only=True,
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    width = max(len(record.input_ids) for record in selected)
    probe_rows = [
        list(record.input_ids) + [int(tokenizer.pad_token_id)] * (width - len(record.input_ids))
        for record in selected
    ]
    probe_masks = [
        [1] * len(record.input_ids) + [0] * (width - len(record.input_ids))
        for record in selected
    ]
    prompt_content = {
        "prompt_ids": [record.prompt_id for record in selected],
        "prompt_texts": [record.prompt_text for record in selected],
        "trajectory_ids": [record.trajectory_id for record in selected],
        "generation_groups": {
            group_id: [record.trajectory_id for record in selected if record.generation_group_id == group_id]
            for group_id in sorted({record.generation_group_id for record in selected})
        },
    }
    compatibility = {
        "format_version": 1,
        "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
        "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
        "prompt_protocol": prompt_protocol_name(model_config),
    }
    prompt_content = {
        **compatibility,
        **prompt_content,
        "trajectory_store_hash": bank_manifest["sha256"],
        "group_membership_hash": sha256_value(prompt_content["generation_groups"]),
    }
    prompt_payload = {**prompt_content, "sha256": sha256_value(prompt_content)}
    probe_content = {
        **compatibility,
        "input_ids": probe_rows,
        "attention_mask": probe_masks,
        "prompt_ids": prompt_content["prompt_ids"],
        "trajectory_ids": prompt_content["trajectory_ids"],
        "trajectory_store_hash": bank_manifest["sha256"],
        "tokenizer_revision": str(model_config["tokenizer_revision"]),
        "group_membership_hash": prompt_payload["group_membership_hash"],
        "probe_kl_mask_protocol": LOCAL_FORK_SPEC.probe_kl_mask_protocol,
    }
    probe_payload = {**probe_content, "sha256": sha256_value(probe_content)}
    args.output.mkdir(parents=True, exist_ok=True)
    publish_json_once(args.output / "prompts.json", prompt_payload)
    publish_json_once(args.output / "probe_set.json", probe_payload)


if __name__ == "__main__":
    main()
