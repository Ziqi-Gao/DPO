"""Separate task adaptation of the teacher; never student-method training.

The original production loader still rejects adapters. This experiment trains
an adapter against independently generated, exact canonical proofs and exports
a new dense teacher whose actual weights, rather than its base revision, are
the checkpoint identity.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.datasets.proofgraph.splits import (
    assert_split_isolation,
    build_split,
    canonical_semantic_key,
)

BASE_SEED = 70_000_042
LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PREFLIGHT = {
    "version": "qwen3-v2-teacher-adapt-preflight-v1",
    "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "seed": 271828,
    "train_examples": 256,
    "dev_examples": 32,
    "optimizer_steps": 8,
    "global_batch_size": 32,
    "microbatch_size": 1,
    "max_sequence_length": 1536,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_modules": list(LORA_MODULES),
    "loss": "mean_of_response_token_means_per_sequence",
    "teacher_readiness_claim": False,
}


def make_examples(difficulty: dict[str, Any]) -> dict[str, list[TaskExample]]:
    """Disjoint seed namespaces, balanced pairs, no outcome-based selection."""
    task = ProofGraphTask()
    splits = {
        "teacher_fit": build_split(task, "train", PREFLIGHT["train_examples"], BASE_SEED, difficulty),
        "teacher_dev": build_split(task, "validation", PREFLIGHT["dev_examples"], BASE_SEED, difficulty),
    }
    assert_split_isolation(splits)
    for examples in splits.values():
        for example in examples:
            result = task.verify(example, task.parse_response(render_target(example)))
            if result.reward != 1.0:
                raise ValueError("teacher adaptation requires an exact verified canonical target")
    return splits


def isolation_inventory(splits: dict[str, list[TaskExample]]) -> dict[str, set[Any]]:
    assert_split_isolation(splits)
    examples = [example for rows in splits.values() for example in rows]
    return {
        "semantic": {canonical_semantic_key(example) for example in examples},
        "example_id": {example.example_id for example in examples},
        "pair_group_id": {example.pair_group_id for example in examples},
        "pair_seed": {example.metadata["pair_seed"] for example in examples},
    }


def reject_formal_overlap(inventory: dict[str, set[Any]], example: TaskExample) -> None:
    observed = {
        "semantic": canonical_semantic_key(example),
        "example_id": example.example_id,
        "pair_group_id": example.pair_group_id,
        "pair_seed": example.metadata["pair_seed"],
    }
    if any(value in inventory[key] for key, value in observed.items()):
        raise ValueError("teacher adaptation overlaps the original formal dataset family")


def encode_example(example: TaskExample, tokenizer: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    prompt = format_model_prompt(ProofGraphTask().render(example), tokenizer, model_config)
    prefix = tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False)
    target_text = render_target(example)
    target = tokenizer.encode(target_text, add_special_tokens=False)
    if tokenizer.eos_token_id is None or not prefix or not target:
        raise ValueError("teacher adaptation requires nonempty prefix/target and EOS")
    # Preserve the generation path's explicit prefix/response boundary. No
    # truncation, concatenated-string retokenization, or prompt-token loss.
    target = [*target, tokenizer.eos_token_id]
    tokens = [*prefix, *target]
    if len(tokens) > PREFLIGHT["max_sequence_length"] or len(prefix) > 1246 or len(target) > 256:
        raise ValueError("teacher adaptation example exceeds the unchanged token envelope")
    return {
        "example_id": example.example_id,
        "input_ids": tokens,
        "labels": [-100] * len(prefix) + target,
        "prefix_length": len(prefix),
        "response_length": len(target),
        "prompt_sha256": prompt.model_facing_prompt_sha256,
        "target_sha256": sha256_value(target_text),
    }


def collate(rows: list[dict[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    if not rows:
        raise ValueError("empty teacher adaptation microbatch")
    length = max(len(row["input_ids"]) for row in rows)
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for index, row in enumerate(rows):
        count = len(row["input_ids"])
        inputs[index, :count] = torch.tensor(row["input_ids"], device=device)
        labels[index, :count] = torch.tensor(row["labels"], device=device)
        mask[index, :count] = 1
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


def response_sequence_loss(logits: Any, labels: Any) -> Any:
    """Canonical CE with every sequence equally weighted, regardless of length."""
    import torch

    from posttrain_circuits.learning.supervision.losses import verified_replay_loss

    shifted_labels = labels[:, 1:]
    counts = (shifted_labels != -100).sum(-1)
    if bool((counts == 0).any()):
        raise ValueError("teacher adaptation sequence has no response targets")
    response_mask = labels != -100
    return verified_replay_loss(
        logits.float(),
        labels.masked_fill(~response_mask, 0),
        response_mask,
        torch.ones(labels.shape[0], device=labels.device),
        normalization="sequence",
    )


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor_digest(values: Any) -> str:
    import torch

    result = hashlib.sha256()
    for name, tensor in values:
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def run_adapter_preflight(
    model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    output: Path,
    *,
    observe: Any,
    require: Any = _require,
    plan: dict[str, Any] | None = None,
    training_provenance: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Train once, export dense weights, and verify a fresh offline reload.

    The SDSC worker always uses PREFLIGHT. An explicit smaller plan makes the
    same path testable on a tiny CPU model. This consumes/mutates the supplied
    model; the returned dense model is its independently reloaded successor.
    No training on dev rows or teacher-acceptance claim occurs here.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from posttrain_circuits.models.loading import tokenizer_fingerprint

    plan = dict(PREFLIGHT if plan is None else plan)
    require(
        plan["microbatch_size"] == 1
        and plan["optimizer_steps"] > 0
        and plan["global_batch_size"] > 0
        and plan["optimizer_steps"] * plan["global_batch_size"] == len(train_rows)
        and len(train_rows) == plan["train_examples"]
        and len(dev_rows) == plan["dev_examples"] > 0,
        "adaptation requires complete single-pass optimizer windows and nonempty dev rows",
    )
    require(
        plan["lora_modules"] == list(LORA_MODULES)
        and plan["lora_dropout"] == 0.0
        and plan["loss"] == PREFLIGHT["loss"]
        and plan["teacher_readiness_claim"] is False
        and 0 < plan["max_sequence_length"] <= PREFLIGHT["max_sequence_length"],
        "adaptation plan changes the module, loss, or nonaccepting contract",
    )
    identities = [row["example_id"] for row in train_rows + dev_rows]
    require(len(identities) == len(set(identities)), "duplicate or overlapping adaptation example IDs")
    for row in train_rows + dev_rows:
        tokens, labels = row["input_ids"], row["labels"]
        prefix = next((index for index, value in enumerate(labels) if value != -100), len(labels))
        require(
            len(tokens) == len(labels) <= plan["max_sequence_length"]
            and 0 < prefix <= 1246
            and 0 < len(labels) - prefix <= 256
            and labels[prefix:] == tokens[prefix:]
            and all(value >= 0 for value in labels[prefix:])
            and tokens[-1] == tokenizer.eos_token_id,
            "adaptation row has invalid response labels or token envelope",
        )
    require(
        output.is_dir()
        and not any((output / name).exists() for name in ("adapter", "merged", "train-metrics.jsonl")),
        "adaptation output is missing or already contains training artifacts",
    )
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    require(device.type in {"cpu", "cuda"}, "unsupported adaptation model device")
    attention = model.config._attn_implementation
    started = time.monotonic()
    metrics: dict[str, Any] = {}
    observe.phase("adapter_setup")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"],
            target_modules=list(LORA_MODULES),
            lora_dropout=plan["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    require(trainable and all("lora_" in name for name, _ in trainable), "unexpected trainable parameters")
    modules = [name for name, value in model.named_modules() if hasattr(value, "lora_A")]
    require(
        len(modules) == int(model.config.num_hidden_layers) * len(LORA_MODULES)
        and {name.rsplit(".", 1)[-1] for name in modules} == set(LORA_MODULES),
        "adapter module coverage differs",
    )
    metrics["trainable_parameter_count"] = sum(value.numel() for _, value in trainable)
    metrics["adapter_modules"] = modules
    before = {name: value.detach().cpu().clone() for name, value in trainable}
    observe.phase("frozen_base_identity_before")
    frozen_before = _tensor_digest(
        (name, value) for name, value in model.named_parameters() if not value.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable],
        lr=plan["learning_rate"],
        weight_decay=plan["weight_decay"],
    )
    metrics["optimizer_defaults"] = dict(optimizer.defaults)
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(plan["seed"])).tolist()
    model.train()
    observe.phase("teacher_training_start")
    consumed = 0
    consumed_ids = []
    with (output / "train-metrics.jsonl").open("x") as stream:
        for step in range(plan["optimizer_steps"]):
            observe.phase("teacher_optimizer_window", step=step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset in range(plan["global_batch_size"]):
                index = order[step * plan["global_batch_size"] + offset]
                batch = collate([train_rows[index]], tokenizer.pad_token_id, device)
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                loss = response_sequence_loss(logits, batch["labels"])
                require(bool(torch.isfinite(loss)), "nonfinite teacher loss")
                (loss / plan["global_batch_size"]).backward()
                losses.append(float(loss.detach().cpu()))
                consumed += len(train_rows[index]["input_ids"])
                consumed_ids.append(train_rows[index]["example_id"])
                del logits, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(
                [value for _, value in trainable],
                plan["max_grad_norm"],
                error_if_nonfinite=True,
            )
            require(float(norm) > 0, "zero teacher gradient")
            optimizer.step()
            row = {
                "step": step + 1,
                "loss": sum(losses) / len(losses),
                "grad_norm": float(norm),
                "samples": (step + 1) * plan["global_batch_size"],
                "tokens": consumed,
                "elapsed_seconds": time.monotonic() - started,
                "example_ids": consumed_ids[step * plan["global_batch_size"] :],
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            observe.phase("teacher_optimizer_window_committed", completed_optimizer_steps=step + 1)
            print(json.dumps({"event": "teacher_update", **row}), flush=True)
    observe.phase("parameter_update_validation")
    update_sq = sum(
        float((value.detach().cpu() - before[name]).float().square().sum()) for name, value in trainable
    )
    require(0 < update_sq < float("inf"), "adapter update is zero or nonfinite")
    require(
        _tensor_digest((name, value) for name, value in model.named_parameters() if not value.requires_grad)
        == frozen_before,
        "frozen base weights changed before merge",
    )
    require(
        len(consumed_ids) == len(set(consumed_ids)) == len(train_rows),
        "fit rows were not consumed exactly once",
    )
    metrics.update(
        adapter_update_l2=update_sq**0.5,
        frozen_base_sha256_before_and_after=frozen_before,
        optimizer_steps=plan["optimizer_steps"],
        consumed_tokens=consumed,
        ordered_consumed_example_ids=consumed_ids,
    )
    del optimizer, before, trainable
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    observe.phase("adapter_and_dense_export")
    probe_batch = collate([dev_rows[0]], tokenizer.pad_token_id, device)
    with torch.inference_mode():
        adapter_logits = (
            model(
                input_ids=probe_batch["input_ids"],
                attention_mask=probe_batch["attention_mask"],
            )
            .logits[:, -1]
            .float()
            .cpu()
        )
    adapted_modules = [
        (name, layer.base_layer.weight)
        for name, layer in model.get_base_model().named_modules()
        if hasattr(layer, "lora_A")
    ]
    dense_before = _tensor_digest(adapted_modules)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.eval()
    require(
        not any("lora_" in name for name, _ in merged.named_parameters()), "adapter remains in dense model"
    )
    dense_after = _tensor_digest((name, merged.get_submodule(name).weight) for name, _ in adapted_modules)
    require(dense_after != dense_before, "adapter merge did not change actual dense weights")
    metrics.update(
        dense_adapted_modules_sha256_before=dense_before, dense_adapted_modules_sha256_after=dense_after
    )
    del adapted_modules
    merged.config._commit_hash = None
    merged.save_pretrained(output / "merged", safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output / "merged")
    with torch.inference_mode():
        expected = (
            merged(**{key: probe_batch[key] for key in ("input_ids", "attention_mask")})
            .logits[:, -1]
            .float()
            .cpu()
        )
    require(
        bool(torch.isfinite(adapter_logits).all() and torch.isfinite(expected).all()),
        "nonfinite adapter or merged logits",
    )
    # Low-precision merge rounds the base+delta weights. Record the difference;
    # dev quality is measured on the exported dense teacher, never the adapter.
    metrics["adapter_merge_max_logit_error"] = float((adapter_logits - expected).abs().max())
    # A caller may still hold a reference to the original base. Explicitly move
    # it off CUDA after export so that reload never duplicates GPU residency.
    if device.type == "cuda":
        merged.to("cpu")
    del model, merged
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    observe.phase("dense_checkpoint_reload")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        output / "merged",
        trust_remote_code=False,
        local_files_only=True,
    )
    require(
        tokenizer_fingerprint(reloaded_tokenizer) == tokenizer_fingerprint(tokenizer)
        and reloaded_tokenizer.chat_template == tokenizer.chat_template,
        "reloaded teacher tokenizer identity or chat template differs",
    )
    metrics["reloaded_tokenizer_sha256"] = tokenizer_fingerprint(reloaded_tokenizer)
    reloaded = (
        AutoModelForCausalLM.from_pretrained(
            output / "merged",
            torch_dtype=dtype,
            attn_implementation=attention,
            trust_remote_code=False,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    with torch.inference_mode():
        actual = (
            reloaded(**{key: probe_batch[key] for key in ("input_ids", "attention_mask")})
            .logits[:, -1]
            .float()
            .cpu()
        )
    require(bool(torch.isfinite(actual).all()), "nonfinite reloaded model logits")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    metrics["dense_reload_max_logit_error"] = float((actual - expected).abs().max())
    observe.phase("checkpoint_manifest")
    provenance = (
        None
        if training_provenance is None
        else {
            **training_provenance,
            "train_metrics_sha256": sha256_file(output / "train-metrics.jsonl"),
            "optimizer_steps": plan["optimizer_steps"],
            "consumed_tokens": consumed,
        }
    )
    manifest = checkpoint_manifest(
        output, base_revision=plan["base_revision"], plan=plan, training_provenance=provenance
    )
    metrics["adapted_teacher_sha256"] = manifest["sha256"]
    return reloaded, reloaded_tokenizer, metrics, manifest


def checkpoint_manifest(
    root: Path,
    *,
    base_revision: str,
    plan: dict[str, Any],
    training_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Identify saved dense bytes; never label adapted weights as the HF base."""
    files = []
    paths = []
    for folder in ("merged", "adapter"):
        if (root / folder).is_symlink() or not (root / folder).is_dir():
            raise ValueError("teacher checkpoint is missing a real merged/adapter directory")
        paths.extend((root / folder).rglob("*"))
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError("teacher checkpoint must not contain symlinks")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not any(row["path"].startswith("merged/") and row["path"].endswith(".safetensors") for row in files):
        raise ValueError("teacher checkpoint contains no dense safe tensors")
    payload = {
        "artifact_kind": "adapted_dense_teacher",
        "base_revision": base_revision,
        "adaptation_plan_sha256": sha256_value(plan),
        "files": files,
        "formal_teacher_accepted": False,
        "training_provenance": training_provenance,
    }
    return {**payload, "sha256": sha256_value(payload)}


def dataset_manifest(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    return {
        role: {
            "count": len(rows),
            "examples_sha256": sha256_value([asdict(row) for row in rows]),
            "ordered_ids": [row.example_id for row in rows],
            "pair_seeds": sorted({row.metadata["pair_seed"] for row in rows}),
        }
        for role, rows in splits.items()
    }
