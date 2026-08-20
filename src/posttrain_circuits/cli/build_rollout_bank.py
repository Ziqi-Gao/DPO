"""Build an immutable rollout bank from a tiny fixture or pinned behavior model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.scientific_versions import ROLLOUT_GENERATION_VERSION
from posttrain_circuits.core.types import PromptBatch, TrajectoryRecord
from posttrain_circuits.data.splits import build_split
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.rollout.generation import hf_generate_trajectories
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.utils.smoke import build_grouped_fork_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


def _verify_records(
    records: list[TrajectoryRecord],
    examples_by_id: dict[str, Any],
) -> None:
    task = ProofGraphTask()
    for record in records:
        result = task.verify(
            examples_by_id[record.prompt_id],
            task.parse_response(record.response_text),
        )
        record.verifier_reward = result.reward
        record.verification_trace = asdict(result)


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Build an immutable common rollout bank", argv)
    output = args.output or (Path(config["output_root"]) / "rollout_banks" / "common-smoke")
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return

    seed = int(config["seed"])
    model_config = config["model"]
    production = not str(model_config["model_name_or_path"]).startswith("local/")
    generations_per_prompt = int(config["state_source"].get("num_generations_per_prompt", 4))
    if generations_per_prompt < 1:
        raise ValueError("num_generations_per_prompt must be positive")
    if not production:
        tokenizer = build_tiny_tokenizer()
        records = build_grouped_fork_bank(
            build_smoke_examples(8, seed),
            tokenizer,
            seed,
            group_size=max(4, generations_per_prompt),
        )
        behavior_policy = {
            "id": "common_mu_smoke",
            "revision": "local-smoke-v1",
            "resolved_commit": "local-smoke-v1",
        }
        prompt_manifest_hash = "smoke-prompts-v1"
        tokenizer_hash = sha256_value(tokenizer.get_vocab())
        resolved_tokenizer_commit = str(model_config["tokenizer_revision"])
    else:
        loaded = load_model_and_tokenizer(model_config, for_training=False)
        policy_model = move_model_to_local_cuda(loaded.model)
        tokenizer = loaded.tokenizer
        task_config = config["task"]
        task = ProofGraphTask()
        examples = build_split(
            task,
            "train",
            int(task_config["num_examples"]),
            int(task_config.get("seed", seed)),
            dict(task_config),
        )
        examples_by_id = {example.example_id: example for example in examples}
        batch_size = int(config["trainer"]["batch_size"])
        records = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = PromptBatch(
                tuple(example.example_id for example in batch for _ in range(generations_per_prompt)),
                tuple(task.render(example) for example in batch for _ in range(generations_per_prompt)),
            )
            records.extend(
                hf_generate_trajectories(
                    policy_model,
                    tokenizer,
                    prompts,
                    policy_version=0,
                    seed=seed + start,
                    max_new_tokens=int(config["trainer"]["max_completion_length"]),
                    temperature=float(config["state_source"]["temperature"]),
                    top_p=float(config["state_source"]["top_p"]),
                    top_k=int(config["state_source"].get("top_k", 0)),
                    min_p=float(config["state_source"].get("min_p", 0.0)),
                    policy_id=loaded.model_id,
                    policy_revision=loaded.resolved_model_commit,
                    model_config=model_config,
                )
            )
        _verify_records(records, examples_by_id)
        behavior_policy = {
            "id": loaded.model_id,
            "revision": loaded.requested_model_revision,
            "resolved_commit": loaded.resolved_model_commit,
        }
        prompt_manifest_hash = sha256_value([asdict(example) for example in examples])
        tokenizer_hash = loaded.tokenizer_hash
        resolved_tokenizer_commit = loaded.resolved_tokenizer_commit

    manifest = TrajectoryStore(output).write(
        records,
        behavior_policy=behavior_policy,
        prompt_manifest_hash=prompt_manifest_hash,
        sampling_configuration={
            "temperature": float(config["state_source"].get("temperature", 1.0)),
            "top_p": float(config["state_source"].get("top_p", 1.0)),
            "top_k": int(config["state_source"].get("top_k", 0)),
            "min_p": float(config["state_source"].get("min_p", 0.0)),
            "max_new_tokens": int(config["trainer"]["max_completion_length"]),
            "num_generations_per_prompt": generations_per_prompt,
        },
        verifier_version="proofgraph-exact-v1",
        teacher_version=None,
        top_k=0,
        extra_metadata={
            "store_kind": "rollout_bank",
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
            "tokenizer_hash": tokenizer_hash,
            "tokenizer_fingerprint": tokenizer_hash,
            "resolved_tokenizer_commit": resolved_tokenizer_commit,
            "protocol_track": str(config.get("protocol_track", "core_v2")),
            "artifact_namespace": str(model_config.get("artifact_namespace", "legacy")),
            "prompt_protocol": str(
                model_config.get("prompt_protocol", {}).get("name", "legacy_raw_v1")
                if isinstance(model_config.get("prompt_protocol"), dict)
                else model_config.get("prompt_protocol", "legacy_raw_v1")
            ),
            "chat_template_sha256": str(
                model_config.get("prompt_protocol", {}).get("chat_template_sha256", "legacy-unrecorded")
                if isinstance(model_config.get("prompt_protocol"), dict)
                else "legacy-unrecorded"
            ),
        },
    )
    print_json({"output": str(output), "manifest": manifest})


if __name__ == "__main__":
    main()
