#!/usr/bin/env python3
"""Fixed two-model, two-rank inference diagnosis of structural naming cues.

No optimizer, training, new checkpoint, checkpoint choice or model acceptance.
The original LR-probe FP32 masters are independently restored twice; current
native-BF16 duplicate-load parity is not original FSDP/trainer parity evidence.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path

TASK = "qwen3-v2-student-name-cue-probe-v1"
SCHEMA = "quest-sdsc-student-name-cue-probe-report-v1"
INPUT_SCHEMA = "quest-sdsc-student-name-cue-probe-inputs-v1"
ARMS = ("control", "treatment")
CONDITIONS = ("preserve", "break_pairs", "break_paths", "break_both")
STRUCTURES = ("branch", "chain")
BLOCKS = (0, 1)
METRICS = ("proof_correct", "answer_correct", "format_valid", "length")
FLAGS = dict.fromkeys(
    (
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "formal_initial_accepted",
        "execution_class_certified",
    ),
    False,
)
SOURCE_TENSOR_COUNT = 311
INPUT_KEYS = {
    "schema",
    "job_id",
    "run_id",
    "source_code_sha256",
    "plan_sha256",
    "science_root",
    "original_science_files",
    "dataset_root",
    "hf_home",
    "protocol",
    "original_config",
    "checkpoints",
    "source_dataset_sha256",
    "source_protocol_artifact_sha256",
}


def sibling(name):
    spec = importlib.util.spec_from_file_location("_name_cue_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


util = sibling("sdsc_student_lr_probe")
require, same, canonical, digest = util.require, util.same, util.canonical, util.digest
file_sha, atomic, document = util.file_sha, util.atomic, util.document


def specification():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    require(
        tuple(p.ARMS) == ARMS and tuple(p.CONDITIONS) == CONDITIONS and tuple(p.BLOCKS) == BLOCKS,
        "name cue intervention inventory differs",
    )
    require(
        (p.BASE_COUNT, p.VIEW_COUNT, p.TOTAL_RESPONSE_COUNT) == (64, 512, 1024), "name cue population differs"
    )
    return p


def source_descriptor(arm):
    require(arm in ARMS, "unknown source model")
    return specification().SOURCE_CHECKPOINTS[arm]


def ordinals(rank):
    require(type(rank) is int and rank in (0, 1), "logical rank differs")
    return [i for i in range(512) if (i // 16 + (i // 4) % 2 + i % 4) % 2 == rank]


def counts(rows):
    return dict(
        num_examples=len(rows),
        proof_correct=sum(r["verification"]["reward"] == 1.0 for r in rows),
        answer_correct=sum(r["verification"]["answer_correct"] for r in rows),
        format_valid=sum(r["verification"]["parse_valid"] for r in rows),
        length=sum(r["stop_reason"] == "length" for r in rows),
    )


def metric(row, key):
    return (
        (row["stop_reason"] == "length")
        if key == "length"
        else (row["verification"]["reward"] == 1.0)
        if key == "proof_correct"
        else row["verification"]["answer_correct" if key == "answer_correct" else "parse_valid"]
    )


def reduce_records(records):
    """All map/condition/structure cells and paired factorial contrasts, no pooling."""
    require(len(records) == 512, "all512 model responses required")
    records = sorted(records, key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in records] == list(range(512)), "model response ordinal coverage differs")
    cells, contrasts = {}, {}
    pairs = (
        ("pair_cue_paths_present", "preserve", "break_pairs"),
        ("pair_cue_paths_broken", "break_paths", "break_both"),
        ("path_cue_pairs_present", "preserve", "break_paths"),
        ("path_cue_pairs_broken", "break_pairs", "break_both"),
    )
    for block in BLOCKS:
        cells[str(block)], contrasts[str(block)] = {}, {}
        for condition in CONDITIONS:
            chosen = [r for r in records if r["block"] == block and r["condition"] == condition]
            require(
                len(chosen) == 64 and len({r["source_example_id"] for r in chosen}) == 64,
                "cell population differs",
            )
            cells[str(block)][condition] = {
                s: counts([r for r in chosen if r["structure"] == s]) for s in STRUCTURES
            }
        for structure in STRUCTURES:
            by_condition = {
                c: {
                    r["source_example_id"]: r
                    for r in records
                    if r["block"] == block and r["condition"] == c and r["structure"] == structure
                }
                for c in CONDITIONS
            }
            keys = set(by_condition["preserve"])
            require(
                len(keys) == (48 if structure == "branch" else 16)
                and all(set(x) == keys for x in by_condition.values()),
                "paired structural population differs",
            )
            panel = {}
            for name, left, right in pairs:
                panel[name] = {}
                for key in METRICS:
                    a, b = (
                        [metric(by_condition[left][i], key) for i in sorted(keys)],
                        [metric(by_condition[right][i], key) for i in sorted(keys)],
                    )
                    panel[name][key] = dict(
                        present_condition=left,
                        absent_condition=right,
                        num_examples=len(keys),
                        present_only=sum(x and not y for x, y in zip(a, b, strict=True)),
                        absent_only=sum(y and not x for x, y in zip(a, b, strict=True)),
                        both=sum(x and y for x, y in zip(a, b, strict=True)),
                        neither=sum(not x and not y for x, y in zip(a, b, strict=True)),
                        difference_numerator=sum(a) - sum(b),
                        denominator=len(keys),
                    )
            panel["interaction"] = {
                k: dict(
                    numerator=sum(
                        metric(by_condition["preserve"][i], k)
                        - metric(by_condition["break_pairs"][i], k)
                        - metric(by_condition["break_paths"][i], k)
                        + metric(by_condition["break_both"][i], k)
                        for i in keys
                    ),
                    denominator=len(keys),
                    definition="preserve-break_pairs-break_paths+break_both",
                )
                for k in METRICS
            }
            contrasts[str(block)][structure] = panel
    return dict(cells=cells, contrasts=contrasts)


def artifact(path, output):
    return dict(path=path.relative_to(output).as_posix(), size=path.stat().st_size, sha256=file_sha(path))


def raw_artifacts(output):
    names = [
        "name-cue-dataset-manifest.json",
        "name-cue-data-isolation.json",
        "name-cue-token-audit.json",
        "name-cue-prompts.jsonl",
    ] + [f"name-cue-records-{arm}-rank-{rank}.jsonl" for arm in ARMS for rank in (0, 1)]
    return [artifact(output / name, output) for name in sorted(names) if (output / name).is_file()]


def append(path, value):
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_progress(output, inputs, rank, arm, completed):
    atomic(
        output / f"name-cue-progress-rank-{rank}.json",
        dict(
            schema="quest-sdsc-student-name-cue-probe-progress-v1",
            scope="progress_only_not_completion",
            rank=rank,
            arm=arm,
            completed=completed,
            total=256,
            completion_claim=False,
            **{k: inputs[k] for k in ("job_id", "run_id", "plan_sha256", "source_code_sha256")},
            **FLAGS,
        ),
    )


def staged_science(inputs, output):
    work = output.parent.resolve()
    require(
        isinstance(inputs, dict) and set(inputs) == INPUT_KEYS and inputs["schema"] == INPUT_SCHEMA,
        "name cue input fields differ",
    )
    require(
        isinstance(inputs["job_id"], str)
        and re.fullmatch("[1-9][0-9]*", inputs["job_id"])
        and inputs["job_id"] == os.environ.get("SLURM_JOB_ID"),
        "job allocation differs",
    )
    for name in (
        "source_code_sha256",
        "plan_sha256",
        "source_dataset_sha256",
        "source_protocol_artifact_sha256",
    ):
        require(
            isinstance(inputs[name], str) and re.fullmatch("[a-f0-9]{64}", inputs[name]), "invalid input SHA"
        )
    science = util.confined(inputs["science_root"], work, directory=True)
    require(Path.cwd().resolve() == science, "worker cwd must equal verified science root")
    require(
        not any(n == "posttrain_circuits" or n.startswith("posttrain_circuits.") for n in sys.modules),
        "science imported before root verification",
    )
    originals = inputs["original_science_files"]
    require(isinstance(originals, dict) and len(originals) == 49, "frozen scientific inventory differs")
    for name, expected in originals.items():
        require(
            isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe scientific path",
        )
        require(file_sha(science / name) == expected, "frozen scientific file changed: " + name)
    require(
        str(util.confined(inputs["hf_home"], work, directory=True)) == os.environ.get("HF_HOME"),
        "offline cache differs",
    )
    sys.path.insert(0, str(science / "src"))
    return work


def load_data(inputs, work, output, rank):
    import yaml

    from posttrain_circuits.datasets.proofgraph import rendering
    from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example
    from posttrain_circuits.learning.teacher.adaptation_fit import FitPlan, make_fit_examples

    p = specification()
    config_path = util.verified(inputs["original_config"], work, util.CONFIG_SHA, 7923)
    config = yaml.safe_load(config_path.read_text())
    require(
        set(inputs["protocol"]) == {"path", "size", "sha256", "protocol_sha256"},
        "protocol input fields differ",
    )
    protocol_path = util.verified({k: inputs["protocol"][k] for k in ("path", "size", "sha256")}, work)
    require(
        digest(rendering.RESPONSE_FORMAT_INSTRUCTIONS.encode()) == util.INSTRUCTION_SHA,
        "original instructions changed",
    )
    binding = p.resolve_student_name_cue_probe_protocol(Path(inputs["science_root"]))
    require(
        binding.artifact_sha256 == file_sha(protocol_path)
        and binding.protocol_sha256 == inputs["protocol"]["protocol_sha256"],
        "resolved protocol differs",
    )
    for name in (
        "sdsc_student_name_cue_probe_worker.py",
        "sdsc_student_lr_probe.py",
        "sdsc_student_quality_probe.py",
    ):
        require(
            file_sha(Path(__file__).with_name(name)) == binding.science_file_sha256["tools/" + name],
            "executing tool differs from accepted science: " + name,
        )
    require(set(inputs["checkpoints"]) == set(ARMS), "both prespecified checkpoints required")
    checkpoints = {}
    for arm in ARMS:
        r = inputs["checkpoints"][arm]
        d = source_descriptor(arm)
        require(
            set(r) == {"path", "size", "sha256", "job_id", "step", "learning_rate"},
            "checkpoint input fields differ",
        )
        same(
            {k: r[k] for k in ("size", "sha256", "job_id", "step", "learning_rate")},
            {k: d[k] for k in ("size", "sha256", "job_id", "step", "learning_rate")},
            "source checkpoint identity differs",
        )
        checkpoints[arm] = util.verified(
            {k: r[k] for k in ("path", "size", "sha256")}, work, d["sha256"], d["size"]
        )
        require(
            inputs["source_dataset_sha256"] == d["dataset_sha256"]
            and inputs["source_protocol_artifact_sha256"] == d["protocol_artifact_sha256"],
            "source diagnostic identity differs",
        )
    root = util.confined(inputs["dataset_root"], work, directory=True)
    require(file_sha(root / "manifest.json") == util.FAMILY_SHA, "original family manifest changed")
    manifest = document(root / "manifest.json")

    def formal_rows():
        total = 0
        for name, entry in manifest["splits"].items():
            require(entry["path"] == name + "/examples.jsonl", "formal family path changed")
            path = util.confined(str(root / entry["path"]), work)
            require(file_sha(path) == entry["examples_file_sha256"], "original split changed")
            count = 0
            with path.open() as stream:
                for line in stream:
                    count += 1
                    yield deserialize_example(json.loads(line))
            require(count == entry["num_examples"], "original split count differs")
            total += count
        require(total == 144000, "original family incomplete")

    splits = p.make_examples()
    isolation = p.audit_isolation(splits, formal_rows(), make_fit_examples(config["task"], FitPlan()))
    dataset = p.dataset_manifest(splits)
    if rank == 0:
        atomic(output / "name-cue-dataset-manifest.json", dataset)
        atomic(output / "name-cue-data-isolation.json", isolation)
    return config, checkpoints, splits, dataset, isolation, binding


def runtime():
    import torch

    rank, world = int(os.environ.get("RANK", "-1")), int(os.environ.get("WORLD_SIZE", "0"))
    require(
        world == 2 and rank in (0, 1) and int(os.environ.get("LOCAL_RANK", "-1")) == rank,
        "single-node W2 required",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 2
        and all("H100" in torch.cuda.get_device_name(i) for i in (0, 1)),
        "two assigned H100s required",
    )
    require(
        platform.python_version() == "3.12.13" and torch.__version__ == "2.8.0+cu128", "fixed runtime differs"
    )
    for package, version in {
        "transformers": "4.56.2",
        "accelerate": "1.10.1",
        "tokenizers": "0.22.0",
    }.items():
        require(importlib.metadata.version(package) == version, "fixed runtime package differs")
    for key, value in dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        require(os.environ.get(key) == value, "execution environment differs: " + key)
    require(
        not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"), "autocast prohibited"
    )
    torch.cuda.set_device(rank)
    torch.set_num_threads(12)
    torch.distributed.init_process_group(backend="gloo", timeout=timedelta(seconds=600))
    return rank, torch.device("cuda", rank)


def validate_dense_payload(saved, arm):
    d = source_descriptor(arm)
    keys = {
        "format",
        "arm",
        "mode",
        "model",
        "global_step",
        "parent_checkpoint_sha256",
        "dataset_sha256",
        "protocol_sha256",
        "protocol_artifact_sha256",
        "learning_rate",
        "scope",
        *FLAGS,
    }
    require(isinstance(saved, dict) and set(saved) == keys, "source dense fields differ")
    expected = dict(
        format="student_focus_lr_probe_dense_v1",
        arm=arm,
        mode="probe",
        global_step=32,
        parent_checkpoint_sha256=util.INITIAL_SHA,
        dataset_sha256=d["dataset_sha256"],
        protocol_sha256=d["protocol_sha256"],
        protocol_artifact_sha256=d["protocol_artifact_sha256"],
        learning_rate=d["learning_rate"],
        scope="student_focus_lr_probe_model_only",
        **FLAGS,
    )
    same({k: saved[k] for k in expected}, expected, "source dense provenance differs")


def load_prepared(config, checkpoint, arm, device):
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    d = source_descriptor(arm)
    require(
        file_sha(checkpoint) == d["sha256"] and checkpoint.stat().st_size == d["size"],
        "source checkpoint bytes changed",
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_dense_payload(saved, arm)
    before = canonical(config)
    loaded = load_model_and_tokenizer(config, for_training=False)
    require(
        loaded.resolved_model_commit == specification().BASE_REVISION
        and loaded.resolved_tokenizer_commit == specification().BASE_REVISION
        and loaded.tokenizer_hash == util.TOKENIZER_SHA
        and loaded.chat_template_sha256 == specification().CHAT_TEMPLATE_SHA256
        and loaded.prompt_protocol == "qwen3_non_thinking_v1",
        "model/tokenizer identity differs",
    )
    exact = util.strict_load_export(loaded.model, saved["model"])
    require(exact["key_count"] == SOURCE_TENSOR_COUNT, "master tensor count differs")
    master_sha = torch_state_hash(saved["model"])
    require(master_sha == d["model_state_sha256"], "source master state differs")
    del saved
    gc.collect()
    loaded.model.to(device=device, dtype=torch.bfloat16).eval()
    require(
        all(v.dtype == torch.bfloat16 and not v.requires_grad for v in loaded.model.parameters())
        and not loaded.model.is_gradient_checkpointing
        and loaded.model.config.use_cache is False,
        "native frozen inference state differs",
    )
    require(canonical(config) == before, "loader mutated model config")
    return loaded, dict(exact_saved_master_reload=exact, model_state_sha256=master_sha)


def prepare_views(views, tokenizer, config):
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    p = specification()
    for rank in (0, 1):
        same(p.generation_ordinals(rank), ordinals(rank), "protocol rank partition differs")
    rows = [p.encode_view(v, tokenizer, config) for v in views]
    token_audit = p.validate_encoded_views(rows, views)
    prompts = []
    for i, (v, r) in enumerate(zip(views, rows, strict=True)):
        require(
            r["response_length"] <= 256 and r["prefix_length"] + 256 <= 2454,
            "view exceeds fixed token envelope",
        )
        prompt = format_model_prompt(v.prompt, tokenizer, config).model_facing_prompt
        same(
            tokenizer.encode(prompt, add_special_tokens=False),
            r["input_ids"][: r["prefix_length"]],
            "prompt prefix differs",
        )
        prompts.append(
            dict(
                ordinal=i,
                source_example_id=v.source_example_id,
                view=v.view,
                block=v.block,
                condition=v.condition,
                example=asdict(v.example),
                prompt_text=prompt,
                prompt_ids=r["input_ids"][: r["prefix_length"]],
                canonical_response_tokens=r["response_length"],
            )
        )
    require(len(prompts) == 512, "view population incomplete")
    return rows, prompts, token_audit


def duplicate_reload_parity(loaded, checkpoint, arm, rows, device):
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash

    reference = []
    with torch.inference_mode():
        for row in rows[:2]:
            ids = torch.tensor([row["input_ids"][: row["prefix_length"]]], device=device)
            reference.append(
                loaded.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False
                )
                .logits.float()
                .cpu()
            )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_dense_payload(saved, arm)
    exact = util.strict_load_export(loaded.model, saved["model"])
    require(
        exact["key_count"] == SOURCE_TENSOR_COUNT
        and torch_state_hash(saved["model"]) == source_descriptor(arm)["model_state_sha256"],
        "second source restore differs",
    )
    del saved
    gc.collect()
    loaded.model.to(device=device, dtype=torch.bfloat16).eval()
    comparisons = []
    with torch.inference_mode():
        for row, expected in zip(rows[:2], reference, strict=True):
            ids = torch.tensor([row["input_ids"][: row["prefix_length"]]], device=device)
            actual = (
                loaded.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False
                )
                .logits.float()
                .cpu()
            )
            comparison = util.difference_statistics(expected, actual)
            require(torch.equal(expected, actual), "native BF16 duplicate-load logits differ")
            comparisons.append(comparison)
        ids = torch.full((1, 2454), 17, dtype=torch.long, device=device)
        logits = loaded.model(
            input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False
        ).logits
        require(bool(torch.isfinite(logits).all()), "nonfinite2454-token context logits")
    return (
        exact,
        comparisons,
        dict(
            passed=True,
            input_length=2454,
            token_id=17,
            all_logits_finite=True,
            native_bfloat16=True,
            use_cache=False,
            scope="synthetic_diagnostic_context_no_development_content",
        ),
    )


def generate_records(model, tokenizer, views, rows, prompts, *, arm, rank, output, inputs):
    import torch

    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    quality = sibling("sdsc_student_quality_probe")
    records = []
    d = source_descriptor(arm)
    path = output / f"name-cue-records-{arm}-rank-{rank}.jsonl"
    require(not path.exists(), "raw response file already exists")
    publish_progress(output, inputs, rank, arm, 0)
    with torch.inference_mode():
        for i in ordinals(rank):
            v, row, prompt = views[i], rows[i], prompts[i]
            ids = torch.tensor(
                [row["input_ids"][: row["prefix_length"]]], device=next(model.parameters()).device
            )
            require(
                ids.shape[1] + 256 <= 2454 and ids.shape[1] + 256 <= model.config.max_position_embeddings,
                "generation exceeds fixed context",
            )
            seed_everything(42 + i)
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                max_new_tokens=256,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            require(torch.equal(generated[0, : ids.shape[1]], ids[0]), "generation changed prefix")
            tokens = generated[0, ids.shape[1] :].tolist()
            require(
                0 < len(tokens) <= 256
                and (tokens[-1] == tokenizer.eos_token_id or len(tokens) == 256)
                and tokenizer.eos_token_id not in tokens[:-1],
                "generation termination differs",
            )
            record = quality.response_record(
                v.example,
                tokens,
                tokenizer,
                prompt_ids=ids[0].tolist(),
                prompt_text=prompt["prompt_text"],
                cap=256,
            )
            record.update(
                arm=arm,
                source_job_id=d["job_id"],
                source_step=32,
                learning_rate=d["learning_rate"],
                checkpoint_sha256=d["sha256"],
                ordinal=i,
                rank=rank,
                cohort="student_name_cue_probe_dev",
                source_example_id=v.source_example_id,
                view=v.view,
                block=v.block,
                condition=v.condition,
                structure=v.example.metadata["structure"],
                parsed_trace=asdict(ProofGraphTask().parse_response(record["response_text"])),
            )
            append(path, record)
            records.append(record)
            if len(records) % 16 == 0:
                publish_progress(output, inputs, rank, arm, len(records))
    return records


def validate_contrasts(readouts):
    cells, contrasts = readouts["cells"], readouts["contrasts"]
    pairs = (
        ("pair_cue_paths_present", "preserve", "break_pairs"),
        ("pair_cue_paths_broken", "break_paths", "break_both"),
        ("path_cue_pairs_present", "preserve", "break_paths"),
        ("path_cue_pairs_broken", "break_pairs", "break_both"),
    )
    require(set(contrasts) == {"0", "1"}, "contrast maps differ")
    for block in ("0", "1"):
        require(set(contrasts[block]) == set(STRUCTURES), "contrast structures differ")
        for structure in STRUCTURES:
            n = 48 if structure == "branch" else 16
            panel = contrasts[block][structure]
            require(set(panel) == {x[0] for x in pairs} | {"interaction"}, "contrast inventory differs")
            for name, left, right in pairs:
                require(set(panel[name]) == set(METRICS), "contrast metrics differ")
                for key in METRICS:
                    row = panel[name][key]
                    a, b = cells[block][left][structure][key], cells[block][right][structure][key]
                    both = row["both"]
                    require(
                        type(both) is int and max(0, a + b - n) <= both <= min(a, b), "paired overlap differs"
                    )
                    same(
                        row,
                        dict(
                            present_condition=left,
                            absent_condition=right,
                            num_examples=n,
                            present_only=a - both,
                            absent_only=b - both,
                            both=both,
                            neither=n - a - b + both,
                            difference_numerator=a - b,
                            denominator=n,
                        ),
                        "paired contrast differs",
                    )
            same(
                panel["interaction"],
                {
                    key: dict(
                        numerator=cells[block]["preserve"][structure][key]
                        - cells[block]["break_pairs"][structure][key]
                        - cells[block]["break_paths"][structure][key]
                        + cells[block]["break_both"][structure][key],
                        denominator=n,
                        definition="preserve-break_pairs-break_paths+break_both",
                    )
                    for key in METRICS
                },
                "factorial interaction differs",
            )


def validate_report(report, *, protocol, checkpoints):
    """Pure execution/count validator shared with transport; raw audit is separate."""
    expected = dict(
        schema=SCHEMA,
        task=TASK,
        mode="probe",
        passed=True,
        diagnostic_complete=True,
        execution_complete=True,
        preparation_complete=False,
        inference_only=True,
        optimizer_steps=0,
        training_input_tokens=0,
        world_size=2,
        base_examples=64,
        views_per_model=512,
        total_responses=1024,
        selected_checkpoint=None,
        **FLAGS,
    )
    require(
        set(report)
        == set(expected)
        | {
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "protocol_sha256",
            "protocol_artifact_sha256",
            "dataset_sha256",
            "data_audit",
            "models",
            "raw_artifacts",
        },
        "report fields differ",
    )
    for k, v in expected.items():
        same(report.get(k), v, "report identity/completion differs: " + k)
    require(protocol["protocol_id"] == "qwen3-student-name-cue-probe-v1", "protocol identity differs")
    require(
        set(checkpoints) == set(ARMS) and set(report["models"]) == set(ARMS), "both fixed models required"
    )
    data_audit = report["data_audit"]
    require(
        set(data_audit) == {"passed", "isolation", "token_audit"} and data_audit["passed"] is True,
        "data audit incomplete",
    )
    isolation, token = data_audit["isolation"], data_audit["token_audit"]
    require(
        isolation.get("passed") is True
        and isolation.get("scientific_acceptance") is False
        and isolation.get("formal_generated_responses_or_scores_read") is False,
        "isolation scope differs",
    )
    same(isolation["dataset_manifest_sha256"], report["dataset_sha256"], "isolation dataset differs")
    for key, expected_key in (
        ("counts", "isolation_population_counts"),
        ("excluded_view_counts", "isolation_excluded_view_counts"),
        ("dimensions", "isolation_dimensions"),
    ):
        same(isolation[key], protocol["data"][expected_key], "isolation population differs")
    require(
        token.get("passed") is True
        and token.get("scientific_acceptance") is False
        and token.get("matched_prefix_and_canonical_EOS_lengths") is True,
        "token audit scope differs",
    )
    same([token["views"], token["bases"]], [512, 64], "token population differs")
    require(
        type(token["maximum_prefix_tokens"]) is int
        and 0 < token["maximum_prefix_tokens"] <= 2198
        and type(token["maximum_canonical_response_tokens"]) is int
        and 0 < token["maximum_canonical_response_tokens"] <= 256
        and token["maximum_prefix_plus_cap"] == token["maximum_prefix_tokens"] + 256,
        "token audit envelope differs",
    )
    for arm in ARMS:
        r = report["models"][arm]
        d = checkpoints[arm]
        expected_model = dict(
            arm=arm,
            source_job_id=d["job_id"],
            source_step=32,
            learning_rate=d["learning_rate"],
            checkpoint_sha256=d["sha256"],
            checkpoint_size=d["size"],
        )
        require(
            set(r) == set(expected_model) | {"model_state_sha256", "ranks", "readouts"},
            "model report fields differ",
        )
        same({k: r.get(k) for k in expected_model}, expected_model, "model checkpoint binding differs")
        require(
            isinstance(r["model_state_sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", r["model_state_sha256"]),
            "master state hash absent",
        )
        require(len(r["ranks"]) == 2, "rank evidence incomplete")
        same(
            r["model_state_sha256"],
            protocol["model"]["sources"][arm]["model_state_sha256"],
            "published master state differs",
        )
        total = {k: 0 for k in ("num_examples", *METRICS)}
        for rank, z in enumerate(r["ranks"]):
            require(
                set(z)
                == {
                    "rank",
                    "reference_saved_master_reload",
                    "exact_saved_master_reload",
                    "model_state_sha256",
                    "native_bf16_reload_logits",
                    "inference_envelope_probe",
                    "counts",
                    "completed_ordinals",
                    "generation_seconds",
                },
                "rank evidence fields differ",
            )
            same(z["rank"], rank, "rank order differs")
            same(z["completed_ordinals"], ordinals(rank), "rank ordinal coverage differs")
            same(z["model_state_sha256"], r["model_state_sha256"], "rank master state differs")
            for k in ("exact_saved_master_reload", "reference_saved_master_reload"):
                same(
                    z[k],
                    dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True),
                    "master reload evidence differs",
                )
            require(len(z["native_bf16_reload_logits"]) == 2, "duplicate reload parity incomplete")
            for comp in z["native_bf16_reload_logits"]:
                require(
                    comp["bitwise_equal"] is True
                    and comp["formal_parity_gate_applied"] is False
                    and comp["argmax_mismatches"] == 0
                    and comp["max_abs_error"] == 0
                    and comp["rms_error"] == 0,
                    "duplicate-load parity differs",
                )
                require(
                    comp["shape"][0] == 1
                    and comp["shape"][2] == 151936
                    and comp["shape"][1] > 0
                    and comp["compared_positions"] == comp["shape"][1]
                    and comp["elements"] == comp["shape"][1] * 151936,
                    "parity shape differs",
                )
            same(
                z["inference_envelope_probe"],
                dict(
                    passed=True,
                    input_length=2454,
                    token_id=17,
                    all_logits_finite=True,
                    native_bfloat16=True,
                    use_cache=False,
                    scope="synthetic_diagnostic_context_no_development_content",
                ),
                "context evidence differs",
            )
            c = z["counts"]
            require(set(c) == set(total) and c["num_examples"] == 256, "rank counts differ")
            for k in total:
                require(type(c[k]) is int and 0 <= c[k] <= 256, "rank numeric counts differ")
                total[k] += c[k]
            require(
                c["proof_correct"] <= c["answer_correct"] <= c["format_valid"],
                "rank correctness ordering differs",
            )
            require(
                type(z["generation_seconds"]) in (int, float) and 0 <= z["generation_seconds"] < 4800,
                "generation time differs",
            )
        require(set(r["readouts"]) == {"cells", "contrasts"}, "readout fields differ")
        cells = r["readouts"]["cells"]
        require(set(cells) == {"0", "1"}, "map readouts incomplete")
        reduced = {k: 0 for k in total}
        for block in ("0", "1"):
            require(set(cells[block]) == set(CONDITIONS), "condition readouts incomplete")
            for condition in CONDITIONS:
                require(set(cells[block][condition]) == set(STRUCTURES), "structure readouts incomplete")
                for structure in STRUCTURES:
                    c = cells[block][condition][structure]
                    n = 48 if structure == "branch" else 16
                    require(set(c) == set(total) and c["num_examples"] == n, "cell counts differ")
                    for k in total:
                        require(type(c[k]) is int and 0 <= c[k] <= n, "cell numeric counts differ")
                        reduced[k] += c[k]
        same(reduced, total, "rank/cell counts disagree")
        validate_contrasts(r["readouts"])
    names = [
        "name-cue-dataset-manifest.json",
        "name-cue-data-isolation.json",
        "name-cue-token-audit.json",
        "name-cue-prompts.jsonl",
    ] + [f"name-cue-records-{arm}-rank-{rank}.jsonl" for arm in ARMS for rank in (0, 1)]
    require(
        sorted(x["path"] for x in report["raw_artifacts"]) == sorted(names), "raw artifact inventory differs"
    )
    for x in report["raw_artifacts"]:
        require(
            set(x) == {"path", "size", "sha256"}
            and type(x["size"]) is int
            and 0 < x["size"] <= 32 * 1024**2
            and isinstance(x["sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", x["sha256"]),
            "raw artifact metadata differs",
        )
    require(len(report["raw_artifacts"]) == 8, "all raw artifacts required")
    return report


def execute(inputs, output, work):
    import torch

    from posttrain_circuits.core.seeding import seed_everything

    rank, device = runtime()
    p = specification()
    config, checkpoints, splits, dataset, isolation, binding = util.local_phase(
        lambda: load_data(inputs, work, output, rank), rank=rank, world_size=2
    )
    views = p.development_views(splits["student_dev"])
    models = {}
    reference_prompts = None
    token_audit = None
    for arm in ARMS:
        seed_everything(42)
        loaded, load_evidence = util.local_phase(
            partial(load_prepared, config["model"], checkpoints[arm], arm, device), rank=rank, world_size=2
        )
        rows, prompts, token_audit = util.local_phase(
            partial(prepare_views, views, loaded.tokenizer, config["model"]), rank=rank, world_size=2
        )
        if reference_prompts is None:
            reference_prompts = prompts
            if rank == 0:
                atomic(output / "name-cue-token-audit.json", token_audit)
                path = output / "name-cue-prompts.jsonl"
                require(not path.exists(), "prompt artifact exists")
                for row in prompts:
                    append(path, row)
        else:
            same(prompts, reference_prompts, "two models consumed different prompts")
        exact, parity, context = util.local_phase(
            partial(duplicate_reload_parity, loaded, checkpoints[arm], arm, rows, device),
            rank=rank,
            world_size=2,
        )
        started = time.monotonic()
        records = util.local_phase(
            partial(
                generate_records,
                loaded.model,
                loaded.tokenizer,
                views,
                rows,
                prompts,
                arm=arm,
                rank=rank,
                output=output,
                inputs=inputs,
            ),
            rank=rank,
            world_size=2,
        )
        local = dict(
            rank=rank,
            reference_saved_master_reload=load_evidence["exact_saved_master_reload"],
            exact_saved_master_reload=exact,
            model_state_sha256=load_evidence["model_state_sha256"],
            native_bf16_reload_logits=parity,
            inference_envelope_probe=context,
            counts=counts(records),
            completed_ordinals=[r["ordinal"] for r in records],
            generation_seconds=time.monotonic() - started,
        )
        ranks = util.gather(local)
        all_records = [row for group in util.gather(records) for row in group]
        d = source_descriptor(arm)
        models[arm] = dict(
            arm=arm,
            source_job_id=d["job_id"],
            source_step=32,
            learning_rate=d["learning_rate"],
            checkpoint_sha256=d["sha256"],
            checkpoint_size=d["size"],
            model_state_sha256=d["model_state_sha256"],
            ranks=ranks,
            readouts=reduce_records(all_records),
        )
        loaded = None
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(
        schema=SCHEMA,
        task=TASK,
        mode="probe",
        **{k: inputs[k] for k in ("job_id", "run_id", "source_code_sha256", "plan_sha256")},
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        dataset_sha256=digest(canonical(dataset)),
        passed=True,
        diagnostic_complete=True,
        execution_complete=True,
        preparation_complete=False,
        inference_only=True,
        optimizer_steps=0,
        training_input_tokens=0,
        world_size=2,
        base_examples=64,
        views_per_model=512,
        total_responses=1024,
        selected_checkpoint=None,
        **FLAGS,
        data_audit=dict(passed=True, isolation=isolation, token_audit=token_audit),
        models=models,
        raw_artifacts=raw_artifacts(output),
    )
    validate_report(report, protocol=binding.payload, checkpoints=inputs["checkpoints"])
    if rank == 0:
        atomic(output / "name-cue-report.json", report)
    torch.distributed.barrier()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(exist_ok=True)
    try:
        inputs = document(args.inputs_json)
        work = staged_science(inputs, args.output_dir)
        result = execute(inputs, args.output_dir, work)
        return 0 if result["passed"] else 2
    except Exception as error:
        rank = os.environ.get("RANK", "0")
        failure = dict(
            schema=SCHEMA,
            passed=False,
            diagnostic_complete=False,
            execution_complete=False,
            preparation_complete=False,
            inference_only=True,
            selected_checkpoint=None,
            **FLAGS,
            failed_rank=rank,
            error=f"{type(error).__name__}: {error}",
        )
        atomic(args.output_dir / f"rank-{rank}-failure.json", failure)
        if rank == "0":
            atomic(args.output_dir / "name-cue-report.json", failure)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
