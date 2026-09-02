"""Create a first-class shared-state fork bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from posttrain_circuits.artifacts.compatibility import (
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
    ROLLOUT_GENERATION_VERSION,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.cli._common import dry_run_report, print_json
from posttrain_circuits.core.config import compose_config, is_production_scale
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.experiments.protocols.local_fork import LOCAL_FORK_SPEC
from posttrain_circuits.learning.contracts import PromptBatch, TrajectoryBatch
from posttrain_circuits.models.loading import (
    load_model_and_tokenizer,
)
from posttrain_circuits.learning.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.learning.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.learning.training.local_fork import (
    _PROBE_MANIFEST_FIELDS,
    _PROMPT_MANIFEST_FIELDS,
    _strict_json_object,
    _validate_frozen_input_manifest,
    create_fork_bundle,
    state_hash,
)
from posttrain_circuits.learning.training.local_fork import build_local_fork_source_binding
from posttrain_circuits.utils.smoke import (
    build_grouped_fork_bank,
    build_smoke_examples,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _load_portable_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    checkpoint_payload: dict[str, object],
) -> None:
    state = checkpoint_payload["optimizer"]
    if not isinstance(state, dict):
        raise ValueError("fork checkpoint optimizer state is malformed")
    key_type = checkpoint_payload.get("optimizer_state_key_type", "parameter_id")
    if key_type == "parameter_name":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType

        state = FSDP.rekey_optim_state_dict(
            state,
            OptimStateKeyType.PARAM_ID,
            model,
            optim=optimizer,
        )
    elif key_type != "parameter_id":
        raise ValueError(f"unsupported optimizer state key type: {key_type}")
    optimizer.load_state_dict(state)


@torch.no_grad()
def _bind_smoke_behavior_logprobs(
    model: torch.nn.Module,
    records: list[TrajectoryRecord],
) -> None:
    """Make the tiny frozen behavior policy exactly equal to the fork model."""

    device = next(model.parameters()).device
    for record in records:
        tokens = record.input_ids + record.response_ids
        tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logprobs = model(input_ids=tensor).logits[0].float().log_softmax(-1)
        start = len(record.input_ids) - 1
        record.behavior_logprobs = [
            float(logprobs[start + index, token].cpu()) for index, token in enumerate(record.response_ids)
        ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a shared-state fork bundle")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--trajectory-store", type=Path)
    parser.add_argument("--probe-set", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/local_fork/bundle.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    LOCAL_FORK_SPEC.validate_experiment_config(config["experiment"])
    production = is_production_scale(config)
    if args.dry_run:
        dry_run_report(config, args.output)
        return
    if production and not args.confirm_production:
        raise SystemExit(
            "production fork creation refused: inspect --dry-run, then pass --confirm-production"
        )
    if production and args.source_manifest is None:
        raise ValueError("production fork creation requires --source-manifest")
    if production and args.seed != int(config["seed"]):
        raise ValueError("production fork seed must equal the composed experiment seed")

    model_config = config["model"]
    local_model = str(model_config["model_name_or_path"]).startswith("local/")
    loaded_student = None
    if local_model:
        tokenizer = build_tiny_tokenizer()
        model = build_tiny_qwen(args.seed)
        teacher_model = build_tiny_qwen(args.seed + 1)
        teacher_id = "local/tiny-teacher"
        teacher_revision = "local-random-v1"
    else:
        loaded_student = load_model_and_tokenizer(
            model_config,
            for_training=True,
        )
        tokenizer = loaded_student.tokenizer
        model = loaded_student.model
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
    checkpoint_payload = None
    if args.checkpoint is not None:
        checkpoint_payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise ValueError("fork checkpoint payload must be a mapping")
        required_checkpoint_fields = {
            "model",
            "optimizer",
            "scheduler",
            "rng",
        }
        missing = required_checkpoint_fields - set(checkpoint_payload)
        if missing:
            raise ValueError(f"fork checkpoint is incomplete; missing {sorted(missing)}")
        model.load_state_dict(checkpoint_payload["model"])
    elif production:
        raise ValueError("production fork creation requires --checkpoint")

    input_hashes: dict[str, str]
    if production:
        assert args.checkpoint is not None
        if args.prompts is None or args.trajectory_store is None or args.probe_set is None:
            raise ValueError(
                "production fork creation requires --prompts, --trajectory-store, and --probe-set"
            )
        prompt_payload = _validate_frozen_input_manifest(
            _strict_json_object(args.prompts, name="local-fork prompt manifest"),
            expected_fields=_PROMPT_MANIFEST_FIELDS,
            name="local-fork prompt manifest",
        )
        prompt_content = {
            key: value for key, value in prompt_payload.items() if key != "sha256"
        }
        if prompt_payload.get("sha256") != sha256_value(prompt_content):
            raise ValueError("frozen fork prompt manifest hash mismatch")
        prompts = PromptBatch(
            tuple(str(value) for value in prompt_content["prompt_ids"]),
            tuple(str(value) for value in prompt_content["prompt_texts"]),
        )
        store = TrajectoryStore(args.trajectory_store)
        bank_manifest = store.check_integrity()
        compatibility_expected = {
            "generator_version": GENERATOR_VERSION,
            "label_semantics": LABEL_SEMANTICS,
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
        }
        for key, value in compatibility_expected.items():
            if bank_manifest.get(key) != value or prompt_payload.get(key) != value:
                raise ValueError(f"fork input compatibility changed: {key}")
        if (
            prompt_payload.get("format_version") != 1
            or prompt_payload.get("local_fork_spec_sha256")
            != LOCAL_FORK_SPEC.scientific_sha256
            or prompt_payload.get("trajectory_store_hash") != bank_manifest.get("sha256")
        ):
            raise ValueError("fork prompt manifest protocol/bank binding changed")
        if (
            loaded_student is not None
            and bank_manifest.get("tokenizer_hash") != loaded_student.tokenizer_hash
        ):
            raise ValueError("fork trajectory store tokenizer differs from student tokenizer")
        records = store.read()
        trajectory_ids = [record.trajectory_id for record in records]
        if any(not value for value in trajectory_ids) or len(trajectory_ids) != len(
            set(trajectory_ids)
        ):
            raise ValueError(
                "fork trajectory store contains duplicate/malformed trajectory_id values"
            )
        by_trajectory = {record.trajectory_id: record for record in records}
        try:
            selected = [
                by_trajectory[str(trajectory_id)] for trajectory_id in prompt_content["trajectory_ids"]
            ]
        except KeyError as error:
            raise ValueError(f"fork manifest has no exact frozen trajectory: {error}") from error
        if len(selected) != len(prompts.prompt_ids) or any(
            record.prompt_id != prompt_id
            for record, prompt_id in zip(selected, prompts.prompt_ids, strict=True)
        ):
            raise ValueError("fork trajectory IDs are not aligned with the frozen prompts")
        trajectories = TrajectoryBatch(selected, policy_version=selected[0].policy_version)
        trajectories.validate(max_policy_lag=0)
        if any(record.policy_version != trajectories.policy_version for record in selected):
            raise ValueError("fork trajectory store mixes policy versions")
        observed_groups = {
            group_id: [record.trajectory_id for record in selected if record.generation_group_id == group_id]
            for group_id in sorted({record.generation_group_id for record in selected})
        }
        if observed_groups != prompt_content["generation_groups"]:
            raise ValueError("fork prompt manifest generation-group membership mismatch")
        if prompt_payload.get("group_membership_hash") != sha256_value(observed_groups):
            raise ValueError("fork prompt manifest group-membership hash mismatch")
        probe_payload = _validate_frozen_input_manifest(
            _strict_json_object(args.probe_set, name="local-fork probe manifest"),
            expected_fields=_PROBE_MANIFEST_FIELDS,
            name="local-fork probe manifest",
        )
        probe_rows = probe_payload.get("input_ids")
        probe_masks = probe_payload.get("attention_mask")
        probe_content = {key: value for key, value in probe_payload.items() if key != "sha256"}
        if probe_payload.get("sha256") != sha256_value(probe_content):
            raise ValueError("fixed fork probe-set hash mismatch")
        if probe_payload.get("probe_kl_mask_protocol") != LOCAL_FORK_SPEC.probe_kl_mask_protocol:
            raise ValueError("fixed fork probe-set mask protocol differs from LocalForkSpec")
        expected_prompt_protocol = loaded_student.prompt_protocol
        expected_tokenizer_revision = str(
            model_config.get("tokenizer_revision", model_config["model_revision"])
        )
        if (
            prompt_payload.get("prompt_protocol") != expected_prompt_protocol
            or probe_payload.get("prompt_protocol") != expected_prompt_protocol
            or probe_payload.get("tokenizer_revision") != expected_tokenizer_revision
            or probe_payload.get("trajectory_store_hash") != bank_manifest.get("sha256")
            or probe_payload.get("group_membership_hash")
            != prompt_payload.get("group_membership_hash")
            or probe_payload.get("prompt_ids") != prompt_content["prompt_ids"]
            or probe_payload.get("trajectory_ids") != prompt_content["trajectory_ids"]
        ):
            raise ValueError("fork probe set is bound to different frozen trajectories")
        if (
            not isinstance(probe_rows, list)
            or not probe_rows
            or not all(
                isinstance(row, list)
                and row
                and all(type(value) is int and value >= 0 for value in row)
                for row in probe_rows
            )
            or not isinstance(probe_masks, list)
            or len(probe_masks) != len(probe_rows)
            or not all(
                isinstance(mask, list)
                and len(mask) == len(row)
                and all(type(value) is int and value in {0, 1} for value in mask)
                for row, mask in zip(probe_rows, probe_masks, strict=True)
            )
        ):
            raise ValueError("fixed fork probe-set inputs/masks are malformed")
        probe = torch.tensor(probe_rows, dtype=torch.long, device=next(model.parameters()).device)
        probe_attention_mask = torch.tensor(
            probe_masks,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        if probe.ndim != 2 or probe.shape[0] < 2 or probe_attention_mask.shape != probe.shape:
            raise ValueError("production fork probe set requires at least two fixed prompts")
        input_hashes = {
            "task": sha256_file(args.prompts),
            "bank": str(bank_manifest["sha256"]),
            "probe": sha256_file(args.probe_set),
            "checkpoint_bytes": sha256_file(args.checkpoint),
            "teacher_targets": sha256_value(
                [[record.teacher_topk_ids, record.teacher_topk_logprobs] for record in selected]
            ),
            "verifier_rewards": sha256_value([record.verifier_reward for record in selected]),
            "behavior_logprobs": sha256_value([record.behavior_logprobs for record in selected]),
            "generation_groups": sha256_value(observed_groups),
        }
        input_evidence = {
            "schema_version": 1,
            "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
            "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
            "generator_version": GENERATOR_VERSION,
            "label_semantics": LABEL_SEMANTICS,
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
            "probe_kl_mask_protocol": LOCAL_FORK_SPEC.probe_kl_mask_protocol,
            "trajectory_bank_manifest_path": str(
                (args.trajectory_store / "manifest.json").resolve(strict=True)
            ),
            "trajectory_bank_manifest_file_sha256": sha256_file(
                args.trajectory_store / "manifest.json"
            ),
            "trajectory_bank_content_sha256": str(bank_manifest["sha256"]),
            "prompt_manifest_path": str(args.prompts.resolve(strict=True)),
            "prompt_manifest_file_sha256": sha256_file(args.prompts),
            "prompt_manifest_payload_sha256": str(prompt_payload["sha256"]),
            "probe_manifest_path": str(args.probe_set.resolve(strict=True)),
            "probe_manifest_file_sha256": sha256_file(args.probe_set),
            "probe_manifest_payload_sha256": str(probe_payload["sha256"]),
            "selected_trajectory_ids": [record.trajectory_id for record in selected],
            "selected_trajectory_hashes": {
                record.trajectory_id: sha256_value(asdict(record)) for record in selected
            },
            "selected_trajectories_sha256": sha256_value(
                [asdict(record) for record in selected]
            ),
            "probe_input_hash": "pending",
            "probe_attention_mask_hash": "pending",
            "pre_update_output_hash": "pending",
        }
    else:
        teacher = HuggingFaceTeacherScorer(
            teacher_model,
            teacher_id=teacher_id,
            teacher_revision=teacher_revision,
            top_k=min(128, int(model.config.vocab_size)),
            minimum_retained_mass=0.0,
        )
        examples = build_smoke_examples(2, args.seed)
        bank = build_grouped_fork_bank(examples, tokenizer, args.seed, group_size=4)
        prompts = PromptBatch(
            tuple(record.prompt_id for record in bank),
            tuple(record.prompt_text for record in bank),
        )
        records = bank
        trajectories = teacher.score(TrajectoryBatch(records, policy_version=0))
        _bind_smoke_behavior_logprobs(model, trajectories.records)
        probe_rows = [records[0].input_ids, records[4].input_ids]
        width = max(map(len, probe_rows))
        probe = torch.tensor(
            [row + [int(tokenizer.pad_token_id)] * (width - len(row)) for row in probe_rows],
            device=next(model.parameters()).device,
        )
        probe_attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in probe_rows],
            device=next(model.parameters()).device,
        )
        input_hashes = {
            "task": "smoke-task-v1",
            "bank": "smoke-bank-v1",
            "probe": sha256_value(probe.tolist()),
        }
        input_evidence = None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["trainer"]["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    if checkpoint_payload is not None:
        _load_portable_optimizer_state(optimizer, model, checkpoint_payload)
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
    else:
        prepared = HardTeacherSupervisor(int(tokenizer.pad_token_id)).prepare_targets(
            trajectories,
            None,
            None,
        )
        first_loss = HardTeacherSupervisor(int(tokenizer.pad_token_id)).compute_loss(
            model,
            prepared,
        )
        first_loss.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        _bind_smoke_behavior_logprobs(model, trajectories.records)
    formal_binding: dict[str, object] = {}
    if production:
        assert args.checkpoint is not None
        assert args.source_manifest is not None
        assert checkpoint_payload is not None
        assert loaded_student is not None
        artifact_binding = formal_artifact_binding(config)
        source_binding = build_local_fork_source_binding(
            source_manifest_path=args.source_manifest,
            checkpoint_path=args.checkpoint,
            checkpoint_payload=checkpoint_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_seed=args.seed,
            expected_protocol_track=str(
                config.get(
                    "protocol_track",
                    model_config.get("protocol_track", config.get("prereg_version", "")),
                )
            ),
            expected_preregistration_sha256=str(artifact_binding["prereg_sha256"]),
            expected_implementation_commit=str(artifact_binding["code_commit"]),
            expected_model_id=str(model_config["model_name_or_path"]),
            expected_model_revision=str(model_config["model_revision"]),
            expected_model_resolved_revision=loaded_student.resolved_model_commit,
            expected_tokenizer_id=str(
                model_config.get("tokenizer_name_or_path", model_config["model_name_or_path"])
            ),
            expected_tokenizer_revision=str(
                model_config.get("tokenizer_revision", model_config["model_revision"])
            ),
            expected_tokenizer_resolved_revision=loaded_student.resolved_tokenizer_commit,
            expected_tokenizer_fingerprint=loaded_student.tokenizer_hash,
            expected_prompt_protocol=loaded_student.prompt_protocol,
            expected_chat_template_sha256=loaded_student.chat_template_sha256,
            trajectory_bank_sha256=str(input_hashes["bank"]),
        )
        formal_binding = {
            **artifact_binding,
            "local_fork_source": source_binding,
        }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=probe,
            attention_mask=probe_attention_mask,
        ).logits
    model.train(was_training)
    if checkpoint_payload is not None:
        RNGState(**checkpoint_payload["rng"]).restore()
    manifest = create_fork_bundle(
        args.output,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompts=prompts,
        trajectories=trajectories,
        probe_input_ids=probe,
        probe_attention_mask=probe_attention_mask,
        pre_update_outputs=outputs,
        manifest_hashes={**input_hashes, "model_revision": str(model_config["model_revision"])},
        input_evidence=(
            {
                **input_evidence,
                "probe_input_hash": state_hash(probe.detach().cpu()),
                "probe_attention_mask_hash": state_hash(probe_attention_mask.detach().cpu()),
                "pre_update_output_hash": state_hash(outputs.detach().cpu()),
            }
            if input_evidence is not None
            else None
        ),
        formal_binding=formal_binding,
        model_spec={
            **model_config,
            "tokenizer_name_or_path": str(
                model_config.get(
                    "tokenizer_name_or_path",
                    model_config["model_name_or_path"],
                )
            ),
            "tokenizer_revision": str(
                model_config.get("tokenizer_revision", model_config["model_revision"])
            ),
            "seed": args.seed,
            "architecture": type(model).__name__,
            "protocol_track": str(
                config.get("protocol_track", model_config.get("protocol_track", "core_v2"))
            ),
            "resolved_model_commit": (
                loaded_student.resolved_model_commit
                if loaded_student is not None
                else str(model_config["model_revision"])
            ),
            "resolved_tokenizer_commit": (
                loaded_student.resolved_tokenizer_commit
                if loaded_student is not None
                else str(model_config.get("tokenizer_revision", model_config["model_revision"]))
            ),
            "resolved_tokenizer_fingerprint": (
                loaded_student.tokenizer_hash if loaded_student is not None else "smoke-unbound"
            ),
            "resolved_prompt_protocol": (
                loaded_student.prompt_protocol if loaded_student is not None else "smoke-unbound"
            ),
            "resolved_chat_template_sha256": (
                loaded_student.chat_template_sha256
                if loaded_student is not None
                else "smoke-unbound"
            ),
        },
    )
    print_json({"output": str(args.output), "manifest": manifest.__dict__})


if __name__ == "__main__":
    main()
