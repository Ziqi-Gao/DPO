"""Freeze production local-fork prompts and multi-prompt KL probes from one scored bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.data.trajectory_store import TrajectoryStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hash-pinned production local-fork inputs")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--trajectory-store", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 2:
        raise ValueError("local-fork input set requires at least two prompts")
    config = compose_config(args.overrides)
    store = TrajectoryStore(args.trajectory_store)
    bank_manifest = store.check_integrity()
    eligible = [
        record
        for record in store.read()
        if record.behavior_logprobs
        and record.teacher_topk_ids
        and record.teacher_topk_logprobs
        and record.verifier_reward is not None
    ]
    selected = []
    seen: set[str] = set()
    for reward_class in (True, False):
        for record in eligible:
            if (
                record.prompt_id not in seen
                and bool(float(record.verifier_reward or 0.0) > 0) is reward_class
            ):
                selected.append(record)
                seen.add(record.prompt_id)
                break
    for record in eligible:
        if len(selected) >= args.limit:
            break
        if record.prompt_id not in seen:
            selected.append(record)
            seen.add(record.prompt_id)
    if len(selected) < 2 or len({float(record.verifier_reward or 0.0) > 0 for record in selected}) < 2:
        raise ValueError("local-fork bank selection requires distinct prompts with mixed verifier rewards")
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
    prompt_content = {
        "prompt_ids": [record.prompt_id for record in selected],
        "prompt_texts": [record.prompt_text for record in selected],
        "trajectory_ids": [record.trajectory_id for record in selected],
    }
    prompt_payload = {
        **prompt_content,
        "trajectory_store_hash": bank_manifest["sha256"],
        "sha256": sha256_value(prompt_content),
    }
    probe_payload = {
        "input_ids": probe_rows,
        "prompt_ids": prompt_content["prompt_ids"],
        "trajectory_ids": prompt_content["trajectory_ids"],
        "trajectory_store_hash": bank_manifest["sha256"],
        "tokenizer_revision": str(model_config["tokenizer_revision"]),
        "sha256": sha256_value(probe_rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "prompts.json", prompt_payload)
    atomic_write_json(args.output / "probe_set.json", probe_payload)


if __name__ == "__main__":
    main()
