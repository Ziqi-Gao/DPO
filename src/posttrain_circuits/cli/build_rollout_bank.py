"""Build an immutable rollout bank from a tiny fixture or pinned behavior model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import ROLLOUT_GENERATION_VERSION
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import PromptBatch, SamplingCursor, SamplingRequest
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.learning.state_sources.generation import HF_SAMPLING_PROTOCOL_ID, hf_generate_trajectories
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    LEGACY_BATCH_PARTITION_PROTOCOL,
    ExactGlobalBatchPlan,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.utils.smoke import build_grouped_fork_bank
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


def _rollout_generation_batch_contract(
    config: dict[str, Any],
    *,
    example_count: int,
    generations_per_prompt: int,
) -> tuple[int, dict[str, Any]]:
    """Resolve the scientific generation chunk without inventing placement hints."""

    if type(example_count) is not int or example_count < 1:
        raise ValueError("rollout generation requires a positive example count")
    if type(generations_per_prompt) is not int or generations_per_prompt < 1:
        raise ValueError("rollout generation requires a positive generations-per-prompt count")
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("rollout generation requires trainer configuration")
    protocol = str(
        trainer.get("batch_partition_protocol", LEGACY_BATCH_PARTITION_PROTOCOL)
    )
    if protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
        global_batch_size = trainer.get("global_batch_size")
        max_microbatch_size = trainer.get("max_microbatch_size")
        if type(global_batch_size) is not int or type(max_microbatch_size) is not int:
            raise ValueError(
                "allocation-neutral rollout generation requires integer global_batch_size "
                "and max_microbatch_size"
            )
        # Reuse the reviewed science contract to reject a silently changed
        # global window or microbatch ceiling.  This is not a resource request.
        ExactGlobalBatchPlan(
            global_batch_size=global_batch_size,
            max_microbatch_size=max_microbatch_size,
            rank=0,
            world_size=1,
        )
        if example_count % global_batch_size:
            raise ValueError(
                "allocation-neutral rollout population must be an exact multiple of "
                "the global logical batch"
            )
        state_source = config.get("state_source")
        if not isinstance(state_source, dict):
            raise ValueError(
                "allocation-neutral rollout generation requires state_source configuration"
            )
        temperature = state_source.get("temperature")
        if type(temperature) not in (int, float) or float(temperature) <= 0.0:
            raise ValueError(
                "allocation-neutral rollout generation requires sampled serial generation"
            )
        return max_microbatch_size, {
            "batch_partition_protocol": protocol,
            "global_logical_batch_size": global_batch_size,
            "max_generation_prompt_chunk_size": max_microbatch_size,
            "expanded_generation_sequences_per_full_chunk": (
                max_microbatch_size * generations_per_prompt
            ),
            "max_simultaneous_generation_sequences": 1,
        }
    if protocol != LEGACY_BATCH_PARTITION_PROTOCOL:
        raise ValueError(f"unsupported rollout batch_partition_protocol {protocol!r}")
    batch_size = trainer.get("batch_size")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("legacy rollout generation requires a positive trainer.batch_size")
    return batch_size, {
        "batch_partition_protocol": protocol,
        "generation_batch_size": batch_size,
    }


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
    task_config = config["task"]
    family = load_dataset_family(
        Path(str(task_config.get("dataset_family_path", "")))
    )
    family_train = family.examples("train")
    train_count = int(task_config.get("num_examples", len(family_train)))
    if train_count < 1 or train_count > len(family_train):
        raise ValueError("task.num_examples is outside the frozen train family split")
    examples = family_train[:train_count]
    family_train_ids = {example.example_id for example in family_train}
    model_config = config["model"]
    production = not str(model_config["model_name_or_path"]).startswith("local/")
    generations_per_prompt = int(config["state_source"].get("num_generations_per_prompt", 4))
    if generations_per_prompt < 1:
        raise ValueError("num_generations_per_prompt must be positive")
    generation_batch_size, generation_batch_contract = _rollout_generation_batch_contract(
        config,
        example_count=len(examples),
        generations_per_prompt=generations_per_prompt,
    )
    if not production:
        tokenizer = build_tiny_tokenizer()
        records = build_grouped_fork_bank(
            examples,
            tokenizer,
            seed,
            group_size=max(4, generations_per_prompt),
        )
        behavior_policy = {
            "id": "common_mu_smoke",
            "revision": "local-smoke-v1",
            "resolved_commit": "local-smoke-v1",
        }
        prompt_manifest_hash = str(
            family.boundary("train")["examples_file_sha256"]
        )
        tokenizer_hash = sha256_value(tokenizer.get_vocab())
        resolved_tokenizer_commit = str(model_config["tokenizer_revision"])
    else:
        loaded = load_model_and_tokenizer(model_config, for_training=False)
        policy_model = move_model_to_local_cuda(loaded.model)
        tokenizer = loaded.tokenizer
        task = ProofGraphTask()
        examples_by_id = {example.example_id: example for example in examples}
        records = []
        for start in range(0, len(examples), generation_batch_size):
            batch = examples[start : start + generation_batch_size]
            prompts = PromptBatch(
                tuple(example.example_id for example in batch for _ in range(generations_per_prompt)),
                tuple(task.render(example) for example in batch for _ in range(generations_per_prompt)),
            )
            sampling_request = SamplingRequest(
                sampling_request_seed=seed,
                sampling_protocol_id=HF_SAMPLING_PROTOCOL_ID,
                cursors=tuple(
                    SamplingCursor(
                        optimizer_step=0,
                        retry_index=0,
                        prompt_id=example.example_id,
                        generation_index=generation_index,
                    )
                    for example in batch
                    for generation_index in range(generations_per_prompt)
                ),
            )
            records.extend(
                hf_generate_trajectories(
                    policy_model,
                    tokenizer,
                    prompts,
                    policy_version=0,
                    sampling_request=sampling_request,
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
        prompt_manifest_hash = str(
            family.boundary("train")["examples_file_sha256"]
        )
        tokenizer_hash = loaded.tokenizer_hash
        resolved_tokenizer_commit = loaded.resolved_tokenizer_commit

    unknown_prompt_ids = {record.prompt_id for record in records} - family_train_ids
    if unknown_prompt_ids:
        raise ValueError(
            "rollout-bank prompt IDs are outside the configured train family: "
            f"{sorted(unknown_prompt_ids)}"
        )
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
            **generation_batch_contract,
        },
        verifier_version="proofgraph-exact-v1",
        teacher_version=None,
        top_k=0,
        extra_metadata={
            **formal_artifact_binding(config),
            "store_kind": "rollout_bank",
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
            "tokenizer_hash": tokenizer_hash,
            "tokenizer_fingerprint": tokenizer_hash,
            "resolved_tokenizer_commit": resolved_tokenizer_commit,
            "dataset_family_sha256": str(family.manifest["sha256"]),
            "train_examples_file_sha256": str(
                family.boundary("train")["examples_file_sha256"]
            ),
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
