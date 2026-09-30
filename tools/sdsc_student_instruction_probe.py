#!/usr/bin/env python3
"""One frozen training-only instruction comparison; never scientific acceptance.

The allocation wrapper stages the original parent inputs. This worker changes
only the renderer's uniform instruction constant inside a restoring context.
There is no optimizer, validation selection, output repair, or prompt truncation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCHEMA = "quest-sdsc-student-instruction-probe-v1"
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"
INITIAL_SIZE = 3441276375
CONFIG_SHA = "05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe"
FAMILY_SHA = "bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189"
TRAIN_SHA = "377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b"
BASELINE_SHA = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
CANDIDATE_SHA = "8c44dc8a1bc86167e3787cdcd84e20537db69585e3d7ee30b920447072f03c1e"
CANDIDATE = (
    "Prove QUERY (1) or its negation (0); omit unrelated rules.\n"
    "Output only:\n<proof>\nSnn: Rnn(citations) -> literal\n</proof>\n<answer>0 or 1</answer>\n"
    "Replace placeholders; number S01,S02,... . Apply a rule only when all its premises are established. "
    "Cite those premises by comma-separated Fnn or earlier Snn IDs, not literals. "
    "Copy its consequent: positive TRUE <atom>, negative NOT <atom>; no literal parentheses. "
    "Stop at the target."
)
FALSE_CLAIMS = {
    key: False
    for key in (
        "student_accepted",
        "g0_passed",
        "teacher_accepted",
        "teacher_accepted_under_candidate",
        "pilot_passed",
        "factorial_ready",
        "accepted_science",
        "formal_prompt_accepted",
    )
}


def quality_tools():
    path = Path(__file__).with_name("sdsc_student_quality_probe.py")
    spec = importlib.util.spec_from_file_location("_instruction_quality_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = quality_tools()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def candidate_identity():
    from posttrain_circuits.datasets.proofgraph import rendering

    if (
        text_sha(rendering.RESPONSE_FORMAT_INSTRUCTIONS) != BASELINE_SHA
        or text_sha(CANDIDATE) != CANDIDATE_SHA
    ):
        raise ValueError("single frozen instruction candidate/baseline identity differs")


@contextlib.contextmanager
def instruction(arm):
    from posttrain_circuits.datasets.proofgraph import rendering

    original = rendering.RESPONSE_FORMAT_INSTRUCTIONS
    if text_sha(original) != BASELINE_SHA or arm not in {"baseline", "candidate"}:
        raise ValueError("instruction context requires the original baseline and one known arm")
    try:
        rendering.RESPONSE_FORMAT_INSTRUCTIONS = CANDIDATE if arm == "candidate" else original
        yield
    finally:
        rendering.RESPONSE_FORMAT_INSTRUCTIONS = original


def prepare_prompts(examples, tokenizer, model_config, *, max_positions):
    """Only facts/rules/query reach rendering; no labels, targets, IDs or metadata."""
    from posttrain_circuits.datasets.proofgraph import rendering
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    baseline = rendering.RESPONSE_FORMAT_INSTRUCTIONS
    rows = []
    for example in examples:
        row = {}
        for arm in ("baseline", "candidate"):
            with instruction(arm):
                raw = rendering.render_example(example)
                text = format_model_prompt(raw, tokenizer, model_config).model_facing_prompt
            suffix = baseline if arm == "baseline" else CANDIDATE
            if not raw.endswith(suffix):
                raise ValueError("renderer did not preserve the uniform instruction suffix")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids or len(ids) + 256 > min(1536, max_positions):
                raise ValueError("prompt plus complete 256-token allowance exceeds the model-input envelope")
            row[arm] = {
                "raw_prompt": raw,
                "prompt_text": text,
                "prompt_ids": ids,
                "prompt_sha256": text_sha(text),
                "prompt_token_sha256": hashlib.sha256(Q.canonical(ids)).hexdigest(),
                "graph_text_sha256": text_sha(raw[: -len(suffix)]),
                "instruction_sha256": BASELINE_SHA if arm == "baseline" else CANDIDATE_SHA,
            }
        if (
            row["baseline"]["raw_prompt"][: -len(baseline)]
            != row["candidate"]["raw_prompt"][: -len(CANDIDATE)]
        ):
            raise ValueError("Facts/Rules/Query bytes changed between arms")
        # Check the complete chat prompt too: its only byte delta is this constant.
        before = row["baseline"]["prompt_text"]
        if (
            before.count(baseline) != 1
            or before.replace(baseline, CANDIDATE, 1) != row["candidate"]["prompt_text"]
        ):
            raise ValueError("model-facing prompt changed outside the instruction")
        rows.append(row)
    return rows


def embedding_tying(model):
    left = model.get_input_embeddings().weight
    right = model.get_output_embeddings().weight
    return {
        "configured_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "same_parameter_object": left is right,
        "same_storage": left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr(),
        "input_shape": list(left.shape),
        "output_shape": list(right.shape),
        "input_dtype": str(left.dtype),
        "output_dtype": str(right.dtype),
    }


def compare_pristine_then_load(model, state):
    """Compare every saved tensor BEFORE any state replacement can hide a mismatch."""
    import torch

    pristine = model.state_dict()
    if not isinstance(state, dict) or state.keys() != pristine.keys():
        raise ValueError("pristine/initial checkpoint keys differ")
    count, elements = 0, 0
    for name, current in pristine.items():
        saved = state[name]
        if (
            not isinstance(saved, torch.Tensor)
            or saved.is_complex()
            or saved.dtype != current.dtype
            or saved.shape != current.shape
            or current.device.type != "cpu"
            or saved.device.type != "cpu"
            or (saved.is_floating_point() and saved.dtype != torch.bfloat16)
            or (saved.is_floating_point() and not torch.isfinite(saved).all())
            or not torch.equal(current, saved)
        ):
            raise ValueError("pristine/initial checkpoint tensor differs: " + name)
        count += 1
        elements += saved.numel()
    before = embedding_tying(model)
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(value, state[name]) for name, value in model.state_dict().items()):
        raise ValueError("initial checkpoint changed after strict loading")
    after = embedding_tying(model)
    if before != after:
        raise ValueError("embedding tying changed during initial checkpoint loading")
    return {
        "pristine_comparison": {"all_tensors_exact": True, "key_count": count, "elements": elements},
        "comparison_performed_before_load": True,
        "embedding_tying_before": before,
        "embedding_tying_after": after,
        "scope": "Equality to original pinned offline HF loader state; no independent Hub download audit.",
    }


def load_pristine_initial(model_config, checkpoint_path):
    from posttrain_circuits.artifacts.checkpoints import load_checkpoint_model_state
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(model_config, for_training=False)
    identity = compare_pristine_then_load(loaded.model, load_checkpoint_model_state(checkpoint_path))
    return loaded, identity


def parameter_digest(model):
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(Q.canonical([name, list(parameter.shape), str(parameter.dtype)]))
        raw = parameter.detach().contiguous().view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 8 * 1024 * 1024):
            digest.update(raw[start : start + 8 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def training_screen_viable(arms):
    if [row["arm"] for row in arms] != ["baseline", "candidate"]:
        return False
    before, after = (row["metrics"] for row in arms)
    return all(
        after[key] >= 4 / 32 and after[key] > before[key]
        for key in ("answer_accuracy", "exact_proof_accuracy")
    )


def verified_file(record, work):
    path = Q.assert_local_file(record["path"], work)
    if path.stat().st_size != record["size"] or Q.file_sha(path) != record["sha256"]:
        raise ValueError("staged input file identity differs")
    return path


def training_examples(dataset_root, work):
    """Verify pinned parent bytes; decode only the first 32 training rows."""
    from posttrain_circuits.datasets.proofgraph.serialization import _strict_json_object, deserialize_example

    manifest_path = Q.assert_local_file(str(dataset_root / "manifest.json"), work)
    if Q.file_sha(manifest_path) != FAMILY_SHA:
        raise ValueError("dataset family differs from the frozen parent")
    manifest = json.loads(manifest_path.read_text())
    train_path = Q.assert_local_file(str(dataset_root / "train/examples.jsonl"), work)
    if (
        manifest["splits"]["train"]["examples_file_sha256"] != TRAIN_SHA
        or Q.file_sha(train_path) != TRAIN_SHA
    ):
        raise ValueError("training split differs from the frozen parent")
    with train_path.open() as stream:
        examples = [
            deserialize_example(_strict_json_object(line, context="training prefix"))
            for line in itertools.islice(stream, 32)
        ]
    if len(examples) != 32 or len({example.example_id for example in examples}) != 32:
        raise ValueError("fixed first32 training cohort is incomplete")
    return examples


def generate_arm(model, tokenizer, examples, prompts, arm, *, on_record):
    import torch

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    records = []
    model.eval()
    with torch.no_grad():
        for ordinal, (example, pair) in enumerate(zip(examples, prompts, strict=True)):
            prompt = pair[arm]
            ids = torch.tensor([prompt["prompt_ids"]], device=next(model.parameters()).device)
            if ids.shape[1] + 256 > min(1536, model.config.max_position_embeddings):
                raise ValueError("generation cannot fit the complete fixed 256-token allowance")
            output = model.generate(
                input_ids=ids,
                max_new_tokens=256,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            record = Q.response_record(
                example,
                output[0, ids.shape[1] :].tolist(),
                tokenizer,
                prompt_ids=prompt["prompt_ids"],
                prompt_text=prompt["prompt_text"],
                cap=256,
            )
            record.update(
                arm=arm,
                ordinal=ordinal,
                prompt_text=prompt["prompt_text"],
                parsed_trace=asdict(ProofGraphTask().parse_response(record["response_text"])),
            )
            records.append(record)
            on_record(record)
    metrics = Q.summarize(records)
    metrics["answer_accuracy"] = metrics.pop("validation_accuracy")
    return metrics


def raw_artifacts(output, *, required=False):
    records = []
    for path in (output / "instruction-prompts.jsonl", output / "instruction-records.jsonl"):
        if not path.exists() and not path.is_symlink() and not required:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("raw diagnostic artifact must be a regular non-symlink file: " + path.name)
        records.append({"path": path.name, "sha256": Q.file_sha(path), "size": path.stat().st_size})
    return records


def execute(inputs, output):
    import torch
    import yaml

    if (
        inputs.get("schema") != "quest-sdsc-student-instruction-inputs-v1"
        or inputs.get("parent_job_id") != "54548846"
        or str(inputs.get("job_id")) != os.environ.get("SLURM_JOB_ID")
        or inputs.get("candidate_sha256") != CANDIDATE_SHA
        or inputs.get("dataset_manifest_sha256") != FAMILY_SHA
    ):
        raise ValueError("instruction probe inputs differ from the frozen diagnostic")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("instruction diagnosis requires one assigned H100")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("instruction diagnosis requires offline model loading")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise ValueError("instruction diagnosis requires the reviewed eight CPUs")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    work = output.parent.resolve()
    config_path = verified_file(inputs["resolved_config"], work)
    if inputs["resolved_config"]["sha256"] != CONFIG_SHA:
        raise ValueError("resolved configuration differs from original parent")
    config = yaml.safe_load(config_path.read_text())
    candidate_identity()
    initial = inputs["initial_checkpoint"]
    if initial["sha256"] != INITIAL_SHA or initial["size"] != INITIAL_SIZE:
        raise ValueError("initial checkpoint differs from original parent")
    initial_path = verified_file(initial, work)
    examples = training_examples(Path(inputs["dataset_root"]), work)
    loaded, identity = load_pristine_initial(config["model"], initial_path)
    initial_parameters = parameter_digest(loaded.model)
    prompts = prepare_prompts(
        examples,
        loaded.tokenizer,
        config["model"],
        max_positions=loaded.model.config.max_position_embeddings,
    )
    # Canonical targets are accessed only for scoring, after all inference prompts are fixed.
    model = loaded.model.to(device="cuda:0")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("instruction inference parameters must remain BF16")
    report = {
        "schema": SCHEMA,
        "passed": False,
        "diagnostic_complete": False,
        **FALSE_CLAIMS,
        **{key: inputs[key] for key in ("parent_job_id", "job_id", "run_id", "source_code_sha256")},
        "scope": (
            "One new scientific prompt candidate screened only on fixed first32 training rows; "
            "not acceptance."
        ),
        "candidate_sha256": CANDIDATE_SHA,
        "baseline_sha256": BASELINE_SHA,
        "initial_identity": {**identity, "checkpoint_sha256": INITIAL_SHA, "size": INITIAL_SIZE},
        "dataset_manifest_sha256": FAMILY_SHA,
        "train_file_sha256": TRAIN_SHA,
        "ordered_train_ids": [example.example_id for example in examples],
        "precision": "Original pinned BF16 parameters and SDPA; no explicit autocast, unlike quality-v2.",
        "generation": {
            "max_new_tokens": 256,
            "max_model_input_length": 1536,
            "do_sample": False,
            "use_cache": False,
            "truncation": False,
        },
        "gpu": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "only_instruction_changed": True,
        "arms": [],
        "raw_record_count": 0,
        "model_parameters_sha256_before": initial_parameters,
        "training_screen_viable": False,
    }
    with (output / "instruction-prompts.jsonl").open("xb") as stream:
        for ordinal, (example, pair) in enumerate(zip(examples, prompts, strict=True)):
            stream.write(Q.canonical({"ordinal": ordinal, "example": asdict(example), **pair}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    with (output / "instruction-records.jsonl").open("xb") as stream:
        for arm in ("baseline", "candidate"):
            with instruction(arm):
                forced = Q.teacher_forced_metrics(model, loaded.tokenizer, examples, config["model"])
            start = report["raw_record_count"]
            report["current_arm"] = {"arm": arm, "teacher_forced": forced, "completed_responses": 0}
            Q.atomic_json(output / "progress.json", report)

            def record(row):
                stream.write(Q.canonical(row) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                report["raw_record_count"] += 1
                report["current_arm"]["completed_responses"] += 1
                Q.atomic_json(output / "progress.json", report)

            metrics = generate_arm(model, loaded.tokenizer, examples, prompts, arm, on_record=record)
            report["arms"].append(
                {
                    "arm": arm,
                    "metrics": metrics,
                    "teacher_forced": forced,
                    "record_start": start,
                    "record_count": 32,
                }
            )
            report.pop("current_arm")
            Q.atomic_json(output / "progress.json", report)
            print(json.dumps({"stage": "arm_complete", "arm": arm, "metrics": metrics}), flush=True)
    final_parameters = parameter_digest(model)
    if initial_parameters != final_parameters:
        raise ValueError("model parameters changed during inference-only instruction comparison")
    report.update(
        passed=True,
        diagnostic_complete=True,
        raw_artifacts=raw_artifacts(output, required=True),
        model_parameters_sha256_after=final_parameters,
        training_screen_viable=training_screen_viable(report["arms"]),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("instruction probe output must be fresh")
    started = time.time()
    try:
        inputs = json.loads(args.inputs_json.read_text())
        science = Path(inputs["science_root"])
        work = args.output_dir.parent.resolve()
        if (
            not science.is_absolute()
            or not science.is_dir()
            or science.is_symlink()
            or work not in science.resolve().parents
            or not (science / "src").is_dir()
            or (science / "src").is_symlink()
        ):
            raise ValueError("science_root must be the restored parent source inside the node workspace")
        sys.path.insert(0, str(science.resolve() / "src"))
        report = execute(inputs, args.output_dir)
        code = 0
    except Exception as error:
        import traceback

        traceback.print_exc()
        report = {}
        with contextlib.suppress(OSError, ValueError):
            partial = json.loads((args.output_dir / "progress.json").read_text())
            if isinstance(partial, dict) and partial.get("schema") == SCHEMA:
                report = partial
        report.update(
            schema=SCHEMA,
            passed=False,
            diagnostic_complete=False,
            training_screen_viable=False,
            error=str(error),
            raw_artifacts=[],
        )
        try:
            report["raw_artifacts"] = raw_artifacts(args.output_dir)
        except (OSError, ValueError) as raw_error:
            report["raw_artifact_error"] = str(raw_error)
        code = 1
    report.update(FALSE_CLAIMS)
    report["elapsed_seconds"] = time.time() - started
    Q.atomic_json(args.output_dir / "instruction-probe.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
