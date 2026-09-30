#!/usr/bin/env python3
"""Read-only, checkpoint-bound student generation diagnosis; never acceptance.

Run inside a reviewed SDSC allocation, after the wrapper stages verified inputs.
Original 128-token measurements and the exploratory 256-token comparison remain
separate. No optimizer, checkpoint write, threshold change or model selection.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def response_record(example, response_ids, tokenizer, *, prompt_ids, prompt_text, cap):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    task = ProofGraphTask()
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    parsed = task.parse_response(text)
    verified = task.verify(example, parsed)
    eos = tokenizer.eos_token_id
    ended = bool(response_ids and response_ids[-1] == eos)
    tags = re.findall(r"<answer>\s*([01])\s*</answer>", text)
    return {
        "example_id": example.example_id,
        "pair_group_id": example.pair_group_id,
        "expected_label": example.label,
        "prompt_ids": list(prompt_ids),
        "prompt_token_sha256": hashlib.sha256(canonical(list(prompt_ids))).hexdigest(),
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "response_ids": list(response_ids),
        "response_text": text,
        "response_tokens": len(response_ids),
        "max_new_tokens": cap,
        "stop_reason": "eos" if ended else "length" if len(response_ids) >= cap else "other",
        "parse_error": parsed.error_code,
        "verification": asdict(verified),
        "diagnostic_answer_tag_present": len(tags) == 1,
        "diagnostic_answer_tag_correct": len(tags) == 1 and int(tags[0]) == example.label,
    }


def summarize(records):
    if not records:
        raise ValueError("empty diagnostic arm")
    count = len(records)
    return {
        "count": count,
        "validation_accuracy": sum(r["verification"]["answer_correct"] for r in records) / count,
        "exact_proof_accuracy": sum(r["verification"]["reward"] for r in records) / count,
        "format_validity": sum(r["verification"]["parse_valid"] for r in records) / count,
        "diagnostic_answer_tag_accuracy": sum(r["diagnostic_answer_tag_correct"] for r in records) / count,
        "stop_counts": dict(collections.Counter(r["stop_reason"] for r in records)),
        "error_counts": dict(
            collections.Counter(r["verification"]["error_code"] or "accepted" for r in records)
        ),
    }


def evaluate_arm(model, tokenizer, examples, model_config, *, cap, on_record=None):
    import torch

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    if cap not in (128, 256):
        raise ValueError("only the original 128 and exploratory 256 token arms are allowed")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    records = []
    try:
        with torch.no_grad():
            for example in examples:
                text = format_model_prompt(
                    ProofGraphTask().render(example), tokenizer, model_config
                ).model_facing_prompt
                encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                positions = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0))
                limit = min(cap, 1536 - encoded.shape[1], positions - encoded.shape[1] if positions else cap)
                if limit < 1:
                    raise ValueError("prompt exceeds reviewed model-input envelope")
                generated = model.generate(
                    input_ids=encoded,
                    max_new_tokens=limit,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
                record = response_record(
                    example,
                    generated[0, encoded.shape[1] :].tolist(),
                    tokenizer,
                    prompt_ids=encoded[0].tolist(),
                    prompt_text=text,
                    cap=limit,
                )
                records.append(record)
                if on_record is not None:
                    on_record(record)
    finally:
        model.train(was_training)
    return summarize(records), records


def assert_local_file(value, work):
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or work not in path.resolve().parents
    ):
        raise ValueError("probe input must be a staged node-local regular file")
    return path


def teacher_forced_metrics(model, tokenizer, examples, model_config):
    """Canonical-target diagnostic only; not the teacher-demo training loss."""
    import torch

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for example in examples:
            prompt = format_model_prompt(
                ProofGraphTask().render(example), tokenizer, model_config
            ).model_facing_prompt
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            target = [
                *tokenizer.encode(render_target(example), add_special_tokens=False),
                tokenizer.eos_token_id,
            ]
            if len(prefix) + len(target) > 1536:
                raise ValueError("canonical diagnostic target exceeds model-input envelope")
            ids = torch.tensor([prefix + target], device=next(model.parameters()).device)
            logits = model(input_ids=ids).logits[:, len(prefix) - 1 : -1].float()
            labels = torch.tensor([target], device=ids.device)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic canonical NLL")
            total_loss += float(loss)
            correct += int((logits.argmax(-1) == labels).sum())
            count += len(target)
    return {
        "examples": len(examples),
        "response_tokens_including_eos": count,
        "canonical_response_nll": total_loss / count,
        "canonical_response_token_accuracy": correct / count,
    }


def load_checkpoint_for_diagnosis(model_config, checkpoint_path, *, label):
    """Load preserved state exactly before the separate BF16 inference copy."""
    import torch

    from posttrain_circuits.artifacts.checkpoints import load_checkpoint_model_state
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    if label not in {"initial", "step20", "step33"}:
        raise ValueError("unsupported diagnostic checkpoint label")
    state = load_checkpoint_model_state(checkpoint_path)
    expected = torch.bfloat16 if label == "initial" else torch.float32
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or tensor.is_complex():
            raise ValueError("checkpoint contains an unsupported state entry")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("checkpoint contains nonfinite state: " + name)
    dtypes = {tensor.dtype for tensor in state.values() if tensor.is_floating_point()}
    if dtypes != {expected}:
        raise ValueError("saved checkpoint precision differs")

    # Preserve the controlled BF16/SDPA configuration through the real loader.
    # Training checkpoints contain FP32 master weights: promote on CPU before
    # copying them, so load_state_dict cannot silently round them through BF16.
    loaded = load_model_and_tokenizer(model_config, for_training=False)
    loaded.model.to(device="cpu", dtype=expected)
    model_state = loaded.model.state_dict()
    if model_state.keys() != state.keys():
        raise ValueError("checkpoint state keys differ from the pinned model")
    for name, tensor in model_state.items():
        if tensor.shape != state[name].shape or tensor.dtype != state[name].dtype:
            raise ValueError("checkpoint state shape/dtype differs: " + name)
    loaded.model.load_state_dict(state, strict=True)
    for name, tensor in loaded.model.state_dict().items():
        if tensor.dtype != state[name].dtype or not torch.equal(tensor.cpu(), state[name]):
            raise ValueError("checkpoint did not load exactly: " + name)
    return loaded


def execute(inputs, output):
    import torch
    import yaml

    from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    if inputs.get("schema") != "quest-sdsc-student-quality-inputs-v1":
        raise ValueError("unsupported diagnostic input schema")
    if inputs.get("parent_job_id") != "54548846" or str(inputs.get("job_id")) != os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise ValueError("probe parent/allocation differs from reviewed diagnosis")
    if torch.cuda.device_count() != 1:
        raise ValueError("quality diagnosis requires exactly one assigned CUDA device")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name:
        raise ValueError("assigned diagnostic GPU is not the reviewed H100")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("diagnostic model loading must be offline")
    work = output.parent.resolve()
    config = yaml.safe_load(assert_local_file(inputs["resolved_config"], work).read_text())
    if (
        config["trainer"]["max_completion_length"] != 128
        or config["trainer"]["validation_examples"] != 128
        or config["trainer"]["max_model_input_length"] != 1536
    ):
        raise ValueError("parent student evaluation settings changed")
    if config["model"]["model_revision"] != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise ValueError("student base revision differs")
    dataset_root = Path(inputs["dataset_root"]).resolve()
    if work not in dataset_root.parents:
        raise ValueError("dataset must be staged on the compute node")
    family = load_dataset_family(dataset_root)
    cohorts = {
        "validation_exposed": family.examples("validation")[:128],
        "train_diagnostic": family.examples("train")[:32],
    }
    if [s["label"] for s in inputs["checkpoints"]] != ["initial", "step20", "step33"]:
        raise ValueError("all three preselected checkpoints required in order")
    torch.set_num_threads(24)
    torch.manual_seed(42)
    report = {
        "schema": "quest-sdsc-student-quality-probe-v1",
        "passed": False,
        "diagnostic_complete": False,
        "student_accepted": False,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
        "parent_job_id": inputs["parent_job_id"],
        "job_id": inputs["job_id"],
        "run_id": inputs["run_id"],
        "source_code_sha256": inputs["source_code_sha256"],
        "gpu": {"name": gpu_name, "logical_index": 0, "torch": torch.__version__},
        "dataset_manifest_sha256": file_sha(dataset_root / "manifest.json"),
        "cohorts": {
            k: {"count": len(v), "ordered_ids": [e.example_id for e in v]} for k, v in cohorts.items()
        },
        "precision_policy": (
            "Load exact saved dtype; diagnostic BF16 parameter copy for forwards. "
            "This single-GPU inference is not an exact replay of two-rank FSDP arithmetic."
        ),
        "selection_policy": "Preselected initial/step20/step33; no holdout selection or acceptance.",
        "generation": {
            "do_sample": False,
            "use_cache": False,
            "caps": [128, 256],
            "max_model_input_length": 1536,
        },
        "arms": [],
        "checkpoint_evidence": [],
    }
    raw_path = output / "quality-records.jsonl"
    raw_path.touch(exist_ok=False)
    prompts_path = output / "quality-prompts.jsonl"
    raw_count = 0
    for spec in inputs["checkpoints"]:
        path = assert_local_file(spec["path"], work)
        if path.stat().st_size != spec["size"] or file_sha(path) != spec["sha256"]:
            raise ValueError("checkpoint does not match actual parent publication")
        print(json.dumps({"stage": "load_checkpoint", "label": spec["label"]}), flush=True)
        expected = "torch.bfloat16" if spec["label"] == "initial" else "torch.float32"
        loaded = load_checkpoint_for_diagnosis(config["model"], path, label=spec["label"])
        gc.collect()
        model = loaded.model.to(device="cuda:0", dtype=torch.bfloat16)
        if spec["label"] == "initial":
            with prompts_path.open("xb") as stream:
                for cohort, examples in cohorts.items():
                    for example in examples:
                        text = format_model_prompt(
                            ProofGraphTask().render(example), loaded.tokenizer, config["model"]
                        ).model_facing_prompt
                        ids = loaded.tokenizer.encode(text, add_special_tokens=False)
                        stream.write(
                            canonical(
                                {
                                    "cohort": cohort,
                                    "example": asdict(example),
                                    "prompt_text": text,
                                    "prompt_ids": ids,
                                    "prompt_token_sha256": hashlib.sha256(canonical(ids)).hexdigest(),
                                    "canonical_target": render_target(example),
                                }
                            )
                            + b"\n"
                        )
                stream.flush()
                os.fsync(stream.fileno())
        report["checkpoint_evidence"].append(
            {
                **{k: spec[k] for k in ("label", "size", "sha256")},
                "saved_dtype": expected,
                "loaded_exactly": True,
                "forward_parameter_dtype": str(next(model.parameters()).dtype),
            }
        )
        for split, examples in cohorts.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_forced = teacher_forced_metrics(model, loaded.tokenizer, examples, config["model"])
            previous = None
            for cap in (128, 256):
                arm = f"{spec['label']}-{split}-{cap}"
                with raw_path.open("ab") as stream, torch.autocast("cuda", dtype=torch.bfloat16):

                    def record(row, *, stream=stream, label=spec["label"], cohort=split, cap=cap):
                        # Unique prompts/graphs are retained separately, never truncated.
                        compact = {k: v for k, v in row.items() if k != "prompt_ids"}
                        stream.write(
                            canonical({"checkpoint": label, "cohort": cohort, "cap": cap, **compact}) + b"\n"
                        )
                        stream.flush()

                    summary, records = evaluate_arm(
                        model, loaded.tokenizer, examples, config["model"], cap=cap, on_record=record
                    )
                    os.fsync(stream.fileno())
                if previous is not None:
                    summary["paired_prefix_equal_count"] = sum(
                        a["response_ids"] == b["response_ids"][: len(a["response_ids"])]
                        for a, b in zip(previous, records, strict=True)
                    )
                previous = records
                report["arms"].append(
                    {
                        "checkpoint": spec["label"],
                        "cohort": split,
                        "cap": cap,
                        "metrics": summary,
                        "teacher_forced": teacher_forced,
                        "raw_file": raw_path.name,
                        "record_start": raw_count,
                        "record_count": len(records),
                    }
                )
                raw_count += len(records)
                atomic_json(output / "progress.json", report)
                print(json.dumps({"stage": "arm_complete", "arm": arm, "metrics": summary}), flush=True)
        del model, loaded
        gc.collect()
        torch.cuda.empty_cache()
    report["raw_artifacts"] = [
        {"path": path.name, "sha256": file_sha(path), "size": path.stat().st_size}
        for path in (raw_path, prompts_path)
    ]
    report["raw_record_count"] = raw_count
    report.update(passed=True, diagnostic_complete=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "quality-probe.json").exists():
        raise ValueError("refusing to overwrite an existing diagnostic")
    started = time.time()
    try:
        report = execute(json.loads(args.inputs_json.read_text()), args.output_dir)
        code = 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        report = {}
        try:
            partial = json.loads((args.output_dir / "progress.json").read_text())
            if isinstance(partial, dict) and partial.get("schema") == "quest-sdsc-student-quality-probe-v1":
                report = partial
        except (OSError, ValueError):
            pass
        report.update(
            schema="quest-sdsc-student-quality-probe-v1",
            passed=False,
            diagnostic_complete=False,
            student_accepted=False,
            g0_passed=False,
            pilot_passed=False,
            factorial_ready=False,
            error=str(exc),
        )
        code = 1
    report.update(elapsed_seconds=time.time() - started)
    atomic_json(args.output_dir / "quality-probe.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
