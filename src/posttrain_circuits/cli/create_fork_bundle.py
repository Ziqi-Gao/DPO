"""Create a first-class shared-state fork bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from posttrain_circuits.cli._common import dry_run_report, print_json
from posttrain_circuits.core.config import compose_config, is_production_scale
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.models.loading import (
    load_model_and_tokenizer,
)
from posttrain_circuits.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.training.local_fork import create_fork_bundle
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a shared-state fork bundle")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", type=Path)
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
    production = is_production_scale(config)
    if args.dry_run:
        dry_run_report(config, args.output)
        return
    if production and not args.confirm_production:
        raise SystemExit(
            "production fork creation refused: inspect --dry-run, then pass --confirm-production"
        )

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
        prompt_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
        prompt_content = {
            "prompt_ids": prompt_payload.get("prompt_ids"),
            "prompt_texts": prompt_payload.get("prompt_texts"),
            "trajectory_ids": prompt_payload.get("trajectory_ids"),
        }
        if prompt_payload.get("sha256") != sha256_value(prompt_content):
            raise ValueError("frozen fork prompt manifest hash mismatch")
        prompts = PromptBatch(
            tuple(str(value) for value in prompt_content["prompt_ids"]),
            tuple(str(value) for value in prompt_content["prompt_texts"]),
        )
        store = TrajectoryStore(args.trajectory_store)
        bank_manifest = store.check_integrity()
        if (
            loaded_student is not None
            and bank_manifest.get("tokenizer_hash") != loaded_student.tokenizer_hash
        ):
            raise ValueError("fork trajectory store tokenizer differs from student tokenizer")
        records = store.read()
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
        probe_payload = json.loads(args.probe_set.read_text(encoding="utf-8"))
        probe_rows = probe_payload.get("input_ids")
        if probe_payload.get("sha256") != sha256_value(probe_rows):
            raise ValueError("fixed fork probe-set hash mismatch")
        if probe_payload.get("trajectory_ids") != prompt_content["trajectory_ids"]:
            raise ValueError("fork probe set is bound to different frozen trajectories")
        probe = torch.tensor(probe_rows, dtype=torch.long, device=next(model.parameters()).device)
        if probe.ndim != 2 or probe.shape[0] < 2:
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
        }
    else:
        teacher = HuggingFaceTeacherScorer(
            teacher_model,
            teacher_id=teacher_id,
            teacher_revision=teacher_revision,
            top_k=min(128, int(model.config.vocab_size)),
            minimum_retained_mass=0.0,
        )
        examples = build_smoke_examples(4, args.seed)
        bank = build_fixed_bank(examples, tokenizer, args.seed)
        prompts = PromptBatch(
            tuple(example.example_id for example in examples),
            tuple(record.prompt_text for record in bank[::2]),
        )
        records = [record for index, record in enumerate(bank) if index % 2 == 0]
        if all(record.verifier_reward == records[0].verifier_reward for record in records):
            records[1] = build_fixed_bank(examples[1:2], tokenizer, args.seed + 100)[1]
        trajectories = teacher.score(TrajectoryBatch(records, policy_version=0))
        probe = torch.tensor([records[0].input_ids], device=next(model.parameters()).device)
        input_hashes = {
            "task": "smoke-task-v1",
            "bank": "smoke-bank-v1",
            "probe": sha256_value(probe.tolist()),
        }
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
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=probe).logits
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
        pre_update_outputs=outputs,
        manifest_hashes={**input_hashes, "model_revision": str(model_config["model_revision"])},
        model_spec={
            **model_config,
            "seed": args.seed,
            "architecture": type(model).__name__,
        },
    )
    print_json({"output": str(args.output), "manifest": manifest.__dict__})


if __name__ == "__main__":
    main()
