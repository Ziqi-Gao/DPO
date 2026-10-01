#!/usr/bin/env python3
"""One native-BF16 original initial/validation128 check; never full G0 acceptance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

SCHEMA = "quest-sdsc-student-initial-probe-v1"
INPUT_SCHEMA = "quest-sdsc-student-initial-inputs-v1"
TASK = "qwen3-v2-student-initial-native-v1"
PARENT_HEAD = "6c04f804b302184b8ff95d00fab404e0531ed8d6"
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"
INITIAL_SIZE = 3441276375
CONFIG_SHA = "05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe"
FAMILY_SHA = "bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189"
VALIDATION_SHA = "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3"
VALIDATION_SIZE = 75081250
INSTRUCTION_SHA = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TOKENIZER_SHA = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
TEMPLATE_SHA = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PROTOCOL_SHA = "c701dde9691210dcdd06f4a1076299941fdb6ad8d609280a13e80d5d7a4333f7"
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "teacher_accepted",
    "teacher_accepted_under_candidate",
    "accepted_science",
    "formal_prompt_accepted",
)
FALSE = dict.fromkeys(FLAGS, False)
RAW_NAMES = ("initial-prompts.jsonl", "initial-records.jsonl")
SOURCE_PINS = {
    "src/posttrain_circuits/cli/score_probe_candidates.py": (
        "cfb9eb51e8bd9437666be65841c03dbaa5fe94ce09c1e2c36ea57304267cc9f3"
    ),
    "src/posttrain_circuits/models/loading.py": (
        "441573c2db622e0e70e055f7aea788da72334b77070de878b261ed1f7f7e0c31"
    ),
    "src/posttrain_circuits/causal_circuits/model/runner.py": (
        "dabbed884323eaa4ad3a8979ab43bdbd2d5fb5c4716e518428aae440ea652442"
    ),
    "src/posttrain_circuits/models/prompt_protocol.py": (
        "82291fa5a78a8e2e76ae8ad948eceb06e4ee2e4f8115f19697541db5d686037a"
    ),
    "src/posttrain_circuits/datasets/proofgraph/rendering.py": (
        "7908a33b79fa2e423997c55c3338bd17d9ac70c41610ca6cd32a415a0ccbd6d8"
    ),
    "src/posttrain_circuits/datasets/proofgraph/parsing.py": (
        "f77fcfc9c74c8003ac59ec9831b6dc9fc2843c905ae93b1bc46eb6cc4f27fe6f"
    ),
    "src/posttrain_circuits/datasets/proofgraph/verification.py": (
        "f1ffcd38fe38c18bddf698a3239967d60d793f2d03433850f236f8c653ce1157"
    ),
    "src/posttrain_circuits/datasets/proofgraph/metrics.py": (
        "f225223ce4a50c8717b65a58ff515ccac5094459d21e16cd4a33f9debf47fa28"
    ),
    "src/posttrain_circuits/datasets/proofgraph/serialization.py": (
        "151d41a3ad9057697756068289c9e9bae349edfdb6ebe937047af79da6b57939"
    ),
}
GENERATION = dict(
    max_new_tokens=256,
    max_model_input_length=1536,
    do_sample=False,
    use_cache=False,
    truncation=False,
    explicit_attention_mask=False,
    autocast_enabled=False,
    precision="native_bfloat16",
    scorer="posttrain_circuits.cli.score_probe_candidates._score_examples",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    require(path.stat().st_size <= 512 * 1024, "JSON metadata is too large")
    result = json.loads(path.read_bytes(), object_pairs_hook=unique)
    require(isinstance(result, dict), "JSON object required")
    canonical(result)
    return result


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute non-aliased path required")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink rejected")
    return path


def confined(path, work, *, directory=False):
    path = safe(path)
    require(work in path.resolve().parents, "input must remain within node-local work")
    require(path.is_dir() if directory else path.is_file(), "input is not a regular file/directory")
    return path


def verified(record, work, sha, size):
    require(isinstance(record, dict) and set(record) == {"path", "size", "sha256"}, "file record fields")
    path = confined(record["path"], work)
    require(
        type(record["size"]) is int and record["size"] == size == path.stat().st_size, "input size differs"
    )
    require(record["sha256"] == sha == file_sha(path), "input SHA differs")
    return path


def original_sources(science):
    """Check restored deployment bytes before imports; not a new science fingerprint."""
    metadata = science / (".git" if (science / ".git/HEAD").is_file() else ".opd-git")
    confined(metadata, science.parent, directory=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_OPTIONAL_LOCKS="0",
    )

    def git(*args, data=None):
        proc = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(metadata),
                "--work-tree",
                str(science),
                "-C",
                str(science),
                *args,
            ],
            input=data,
            capture_output=True,
            env=environment,
            timeout=30,
            check=False,
        )
        require(proc.returncode == 0, "genuine source Git query failed")
        return proc.stdout

    require(git("rev-parse", "HEAD^{commit}").decode().strip() == PARENT_HEAD, "scientific HEAD differs")
    require(
        git("rev-parse", "--is-shallow-repository").strip() == b"false", "complete source history required"
    )
    require(not (metadata / "info/grafts").exists(), "grafted history rejected")
    entries = []
    for entry in git("ls-tree", "-r", "-z", PARENT_HEAD, "src").split(b"\0"):
        if not entry:
            continue
        fields, name = entry.split(b"\t", 1)
        mode, kind, object_id = fields.split()
        require(mode in (b"100644", b"100755") and kind == b"blob", "nonregular scientific source")
        relative = name.decode()
        if relative.endswith(".py"):
            entries.append((relative, object_id))
    require(entries, "empty scientific source tree")
    blobs = git("cat-file", "--batch", data=b"".join(object_id + b"\n" for _, object_id in entries))
    offset, matched = 0, {}
    for relative, object_id in entries:
        end = blobs.index(b"\n", offset)
        observed, kind, size = blobs[offset:end].split()
        require(observed == object_id and kind == b"blob", "source blob framing differs")
        size = int(size)
        raw = blobs[end + 1 : end + 1 + size]
        require(
            confined(science / relative, science.parent).read_bytes() == raw,
            "restored source differs: " + relative,
        )
        if relative in SOURCE_PINS:
            require(digest(raw) == SOURCE_PINS[relative], "explicit source pin differs")
            matched[relative] = digest(raw)
        offset = end + size + 2
    require(offset == len(blobs) and matched == SOURCE_PINS, "source deployment incomplete")
    require(git("rev-parse", "HEAD^{commit}").decode().strip() == PARENT_HEAD, "source HEAD changed")
    return dict(
        head=PARENT_HEAD,
        source_pins=matched,
        deployment_python_files_verified=len(entries),
        deployment_integrity_only=True,
    )


def staged_science(inputs, output):
    work = safe(output).parent.resolve()
    require(
        set(inputs)
        == {
            "schema",
            "parent_job_id",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "resolved_config",
            "initial_checkpoint",
            "dataset_root",
            "dataset_manifest_sha256",
            "hf_home",
        },
        "unexpected input fields",
    )
    require(inputs["schema"] == INPUT_SCHEMA and inputs["parent_job_id"] == "54548846", "input identity")
    require(
        isinstance(inputs["job_id"], str)
        and re.fullmatch(r"[0-9]+", inputs["job_id"])
        and inputs["job_id"] == os.environ.get("SLURM_JOB_ID"),
        "job allocation differs",
    )
    require(
        isinstance(inputs["run_id"], str) and re.fullmatch(r"[A-Za-z0-9-]+", inputs["run_id"]), "run identity"
    )
    for key in ("source_code_sha256", "plan_sha256"):
        require(isinstance(inputs[key], str) and re.fullmatch(r"[a-f0-9]{64}", inputs[key]), "invalid SHA")
    require(inputs["dataset_manifest_sha256"] == FAMILY_SHA, "dataset identity differs")
    cache = confined(inputs["hf_home"], work, directory=True)
    require(str(cache) == os.environ.get("HF_HOME"), "offline cache differs")
    science = confined(inputs["science_root"], work, directory=True)
    require(Path.cwd().resolve() == science.resolve(), "worker cwd must equal verified science root")
    require(
        not any(key == "posttrain_circuits" or key.startswith("posttrain_circuits.") for key in sys.modules),
        "science imported before verified source",
    )
    sources = original_sources(science)
    sys.path.insert(0, str(science / "src"))
    return work, sources


def load_inputs(inputs, work, sources):
    import yaml

    from posttrain_circuits.artifacts.adapted_student_protocol import validate_student_protocol
    from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
    from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS

    config_path = verified(inputs["resolved_config"], work, CONFIG_SHA, 7923)
    config = yaml.safe_load(config_path.read_text())
    binding = validate_student_protocol(config)
    require(
        binding.head == binding.git_commit == PARENT_HEAD and binding.sha256 == PROTOCOL_SHA,
        "original accepted student protocol differs",
    )
    require(len(binding.science_file_sha256) == 49, "original named scientific binding differs")
    require(
        config["model"]["torch_dtype"] == "bfloat16"
        and config["model"]["attn_implementation"] == "sdpa"
        and config["model"]["model_revision"] == REVISION
        and config["model"]["use_cache"] is False,
        "original native model configuration differs",
    )
    require(
        config["anti_shortcut"]["max_completion_length"] == 256
        and config["anti_shortcut"]["minimum_iid_accuracy"] == 0.10
        and config["g0"]["base_accuracy_examples"] == 128
        and config["trainer"]["max_model_input_length"] == 1536,
        "original base check contract differs",
    )
    require(digest(RESPONSE_FORMAT_INSTRUCTIONS.encode()) == INSTRUCTION_SHA, "original instruction differs")
    checkpoint = verified(inputs["initial_checkpoint"], work, INITIAL_SHA, INITIAL_SIZE)
    dataset = confined(inputs["dataset_root"], work, directory=True)
    manifest = confined(dataset / "manifest.json", work)
    validation = confined(dataset / "validation/examples.jsonl", work)
    require(file_sha(manifest) == FAMILY_SHA and manifest.stat().st_size == 1722, "family manifest differs")
    require(
        file_sha(validation) == VALIDATION_SHA and validation.stat().st_size == VALIDATION_SIZE,
        "original validation split differs",
    )
    family = load_dataset_family(dataset)
    examples = family.examples("validation")[:128]
    require(
        len(examples) == len({row.example_id for row in examples}) == 128, "fixed validation128 incomplete"
    )
    sources.update(
        protocol_artifact_sha256=binding.sha256,
        protocol_sha256=binding.protocol_sha256,
        acceptance_commit=binding.git_commit,
        implementation_commit=binding.reviewed_implementation_commit,
        science_file_sha256=binding.science_file_sha256,
    )
    return config, checkpoint, examples, family.manifest["sha256"]


def runtime():
    import torch

    require(platform.python_version() == "3.12.13" and torch.__version__ == "2.8.0+cu128", "runtime differs")
    for package, expected in {
        "transformers": "4.56.2",
        "accelerate": "1.10.1",
        "tokenizers": "0.22.0",
    }.items():
        require(importlib.metadata.version(package) == expected, "runtime dependency differs: " + package)
    for name, expected in dict(
        SLURM_CPUS_PER_TASK="8",
        SLURM_MEM_PER_NODE="65536",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        require(os.environ.get(name) == expected, "execution environment differs: " + name)
    require(os.environ.get("LOCAL_RANK", "0") == "0", "one logical GPU required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one assigned GPU required")
    require("H100" in torch.cuda.get_device_name(0), "one H100 required")
    require(
        not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"), "autocast prohibited"
    )
    torch.set_num_threads(8)
    return dict(
        name=torch.cuda.get_device_name(0),
        logical_index=0,
        torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_threads=8,
    )


def load_original_initial(model_config, checkpoint, *, expected_sha256=INITIAL_SHA):
    """The exact formal loader/cuda placement/strict checkpoint path; no promotion or autocast."""
    import torch

    from posttrain_circuits.causal_circuits.model.runner import load_checkpoint_into_hf_model
    from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda

    require(
        not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"), "autocast prohibited"
    )
    before = canonical(model_config)
    loaded = load_model_and_tokenizer(model_config, for_training=False)
    model = move_model_to_local_cuda(loaded.model)
    require(model is loaded.model, "inference model identity changed")
    load_checkpoint_into_hf_model(model, checkpoint, expected_sha256=expected_sha256)
    require(canonical(model_config) == before, "loader mutated model configuration")
    require(
        all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.parameters()),
        "initial inference parameters must remain native BF16 and frozen",
    )
    require(
        not model.training and not model.is_gradient_checkpointing and model.config.use_cache is False,
        "formal inference mode/cache differs",
    )
    return loaded


def prepare_prompts(examples, tokenizer, model_config, *, max_positions):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    prompts = []
    for ordinal, example in enumerate(examples):
        formatted = format_model_prompt(ProofGraphTask().render(example), tokenizer, model_config)
        ids = tokenizer.encode(formatted.model_facing_prompt, add_special_tokens=False)
        require(
            ids and len(ids) + 256 <= min(1536, max_positions), "full original 256-token allowance cannot fit"
        )
        prompts.append(
            dict(
                ordinal=ordinal,
                cohort="validation_exposed",
                example=asdict(example),
                raw_prompt=formatted.raw_prompt,
                prompt_text=formatted.model_facing_prompt,
                raw_prompt_sha256=digest(formatted.raw_prompt.encode()),
                prompt_text_sha256=digest(formatted.model_facing_prompt.encode()),
                prompt_ids=ids,
                prompt_token_sha256=digest(canonical(ids)),
            )
        )
    return prompts


@contextlib.contextmanager
def recording_generate(model, tokenizer, examples, prompts, on_record):
    """Observe the original bound method, preserving its kwargs, tensors and return object."""
    import torch

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    original = model.generate
    had_local = "generate" in model.__dict__
    local_value = model.__dict__.get("generate")
    count = 0

    def recorded(*args, **kwargs):
        nonlocal count
        require(
            not args
            and set(kwargs)
            == {"input_ids", "max_new_tokens", "do_sample", "pad_token_id", "eos_token_id", "use_cache"},
            "formal generate arguments differ",
        )
        require(count < len(examples), "extra generation")
        require(
            not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"),
            "autocast prohibited",
        )
        require(not torch.is_grad_enabled() and not model.training, "original no-grad/eval context absent")
        ids = kwargs["input_ids"]
        prompt = prompts[count]
        require(
            ids.dtype == torch.long
            and ids.ndim == 2
            and ids.shape[0] == 1
            and ids[0].tolist() == prompt["prompt_ids"],
            "actual prompt differs",
        )
        require(
            kwargs["max_new_tokens"] == 256
            and kwargs["do_sample"] is False
            and kwargs["use_cache"] is False
            and kwargs["pad_token_id"] == tokenizer.pad_token_id
            and kwargs["eos_token_id"] == tokenizer.eos_token_id,
            "original generation policy differs",
        )
        output = original(*args, **kwargs)
        require(
            isinstance(output, torch.Tensor)
            and output.dtype == torch.long
            and output.ndim == 2
            and output.shape[0] == 1,
            "generation tensor shape",
        )
        require(torch.equal(output[0, : ids.shape[1]], ids[0]), "generation changed prompt prefix")
        response_ids = output[0, ids.shape[1] :].tolist()
        require(0 < len(response_ids) <= 256, "response token count outside original cap")
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        task = ProofGraphTask()
        parsed = task.parse_response(response)
        verification = task.verify(examples[count], parsed)
        record = dict(
            ordinal=count,
            example_id=examples[count].example_id,
            cohort="validation_exposed",
            pair_group_id=examples[count].pair_group_id,
            prompt_token_sha256=prompt["prompt_token_sha256"],
            prompt_text_sha256=prompt["prompt_text_sha256"],
            response_ids=response_ids,
            response_text=response,
            response_tokens=len(response_ids),
            max_new_tokens=256,
            stop_reason="eos"
            if response_ids[-1] == tokenizer.eos_token_id
            else "length"
            if len(response_ids) == 256
            else "other",
            parsed_trace=asdict(parsed),
            verification=asdict(verification),
        )
        on_record(record)
        count += 1
        return output

    model.generate = recorded
    try:
        yield
        require(count == len(examples), "incomplete original scoring")
    finally:
        if had_local:
            model.generate = local_value
        else:
            del model.generate


def score_original(model, tokenizer, examples, prompts, model_config, on_record):
    from posttrain_circuits.cli.score_probe_candidates import _score_examples
    from posttrain_circuits.datasets.proofgraph.metrics import aggregate_verification

    records = []

    def capture(record):
        records.append(record)
        on_record(record)

    with recording_generate(model, tokenizer, examples, prompts, capture):
        scores, results = _score_examples(
            model, tokenizer, examples, max_new_tokens=256, model_config=model_config
        )
    require(len(results) == len(records) == len(examples), "original scorer incomplete")
    for example, result, record in zip(examples, results, records, strict=True):
        require(
            asdict(result) == record["verification"] and scores[example.example_id] == (result.reward == 1.0),
            "original verification differs from recorded trace",
        )
    return aggregate_verification(results), sum(result.answer_correct for result in results), records


def base_gate(answer_count, total):
    require(
        type(answer_count) is int and type(total) is int and total == 128 and 0 <= answer_count <= total,
        "base gate needs all 128 original results",
    )
    return answer_count >= 13


def raw_artifacts(output, *, required=False):
    rows = []
    for name in RAW_NAMES:
        path = output / name
        if not path.exists() and not required:
            continue
        require(path.is_file() and not path.is_symlink(), "raw artifact missing or not regular")
        rows.append(dict(path=name, size=path.stat().st_size, sha256=file_sha(path)))
    return rows


def progress(output, phase, *, completed=0):
    atomic(
        output / "progress.json",
        dict(
            schema="quest-sdsc-student-initial-progress-v1",
            phase="initial_native_scoring",
            completed_examples=completed,
            total_examples=128,
            passed=False,
            diagnostic_complete=False,
            **FALSE,
        ),
    )


def execute(inputs, output, work, sources):
    config, checkpoint, examples, family_hash = load_inputs(inputs, work, sources)
    gpu = runtime()
    report = dict(
        schema=SCHEMA,
        task=TASK,
        passed=False,
        diagnostic_complete=False,
        **FALSE,
        **{
            key: inputs[key]
            for key in ("parent_job_id", "job_id", "run_id", "source_code_sha256", "plan_sha256")
        },
        scope="original initial native-BF16 validation128 base criterion only; never full G0",
        initial_base_gate_satisfied=False,
        threshold=0.10,
        required_correct=13,
        evaluated_count=0,
        initial_checkpoint_sha256=INITIAL_SHA,
        resolved_config_sha256=CONFIG_SHA,
        dataset_manifest_sha256=FAMILY_SHA,
        dataset_family_semantic_sha256=family_hash,
        validation_file_sha256=VALIDATION_SHA,
        original_instruction_sha256=INSTRUCTION_SHA,
        scientific_binding=sources,
        generation=GENERATION,
        gpu=gpu,
        raw_record_count=0,
        new_test_holdout=False,
    )
    atomic(output / "initial-probe.json", report)
    progress(output, "original_science_and_inputs_verified")
    loaded = load_original_initial(config["model"], checkpoint)
    require(
        loaded.resolved_model_commit == REVISION
        and loaded.resolved_tokenizer_commit == REVISION
        and loaded.tokenizer_hash == TOKENIZER_SHA
        and loaded.chat_template_sha256 == TEMPLATE_SHA
        and loaded.prompt_protocol == "qwen3_non_thinking_v1",
        "actual model/tokenizer identity differs",
    )
    prompts = prepare_prompts(
        examples, loaded.tokenizer, config["model"], max_positions=loaded.model.config.max_position_embeddings
    )
    with (output / "initial-prompts.jsonl").open("xb") as stream:
        for row in prompts:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    report.update(
        tokenizer_fingerprint=loaded.tokenizer_hash,
        chat_template_sha256=loaded.chat_template_sha256,
        model_revision=loaded.resolved_model_commit,
        forward_parameter_dtype="torch.bfloat16",
        prompt_count=len(prompts),
        ordered_example_ids=[row.example_id for row in examples],
        maximum_prompt_tokens=max(len(row["prompt_ids"]) for row in prompts),
    )
    atomic(output / "initial-probe.json", report)
    progress(output, "native_initial_loaded")
    try:
        with (output / "initial-records.jsonl").open("xb") as stream:

            def captured(row):
                stream.write(canonical(row) + b"\n")
                stream.flush()
                report["raw_record_count"] += 1
                if report["raw_record_count"] % 16 == 0:
                    os.fsync(stream.fileno())
                    atomic(output / "initial-probe.json", report)
                    progress(output, "original_scoring", completed=report["raw_record_count"])

            metrics, correct, records = score_original(
                loaded.model, loaded.tokenizer, examples, prompts, config["model"], captured
            )
            os.fsync(stream.fileno())
    finally:
        # Keep the completed raw count even when the next generation raises.
        atomic(output / "initial-probe.json", report)
    require(len(records) == report["raw_record_count"] == 128, "all original responses required")
    require(metrics["answer_accuracy"] == correct / 128, "original aggregate answer denominator differs")
    report.update(
        passed=True,
        diagnostic_complete=True,
        metrics=metrics,
        answer_correct_count=correct,
        evaluated_count=128,
        initial_base_gate_satisfied=base_gate(correct, 128),
        raw_artifacts=raw_artifacts(output, required=True),
        limitations=[
            "This is one original G0 base criterion, not full G0 or student acceptance.",
            "Previously exposed fixed validation128; no new holdout or prompt/model selection.",
            "No trained checkpoint, anti-shortcut suite or circuit eligibility is evaluated.",
        ],
    )
    atomic(output / "initial-probe.json", report)
    progress(output, "original_scoring_complete", completed=128)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = safe(args.output_dir)
    output.mkdir(parents=False, exist_ok=True)
    try:
        inputs = document(safe(args.inputs_json))
        work, sources = staged_science(inputs, output)
        report = execute(inputs, output, work, sources)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("passed", "initial_base_gate_satisfied", "answer_correct_count")
                }
            )
        )
        return 0
    except Exception as exc:
        report = {}
        prior = output / "initial-probe.json"
        if prior.is_file() and not prior.is_symlink():
            with contextlib.suppress(ValueError, OSError):
                report = document(prior)
        artifacts = []
        try:
            artifacts = raw_artifacts(output)
        except (ValueError, OSError) as raw_error:
            report["raw_artifact_error"] = str(raw_error)
        report.update(
            schema=SCHEMA,
            task=TASK,
            passed=False,
            diagnostic_complete=False,
            initial_base_gate_satisfied=False,
            **FALSE,
            error={"type": type(exc).__name__, "message": str(exc)},
            raw_artifacts=artifacts,
        )
        atomic(output / "initial-probe.json", report)
        print(json.dumps({"passed": False, "error": report["error"]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
