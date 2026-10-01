#!/usr/bin/env python3
"""Independent raw-response replay for prepared-student candidates; no GPU inference.

This auditor reconstructs prompt encodings, response decoding, verifier traces,
development counts and prospective checkpoint selection. It does not replay GPU
optimization or accept a model/G0. Accounting and execution acceptance remain
separate prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
METADATA_SHA = "03fd8f3cbbe38aa95f84eb55a023bf8a7238ee16aa7f47f611f3bc32b0a38b20"
MAX_BYTES = 32 * 1024**2
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def read(path, maximum=MAX_BYTES):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink input")
    require(path.is_file() and 0 < path.stat().st_size <= maximum, "input type/size")
    raw = path.read_bytes()
    require(0 < len(raw) <= maximum, "input exceeds bound")
    return raw


def document(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(value, dict), "JSON object required")
    canonical(value)
    return value


def lines(raw, count):
    rows = raw.splitlines()
    require(len(rows) == count and all(row.strip() for row in rows), "raw population count differs")
    return [document(row) for row in rows]


def load_tokenizer():
    from transformers import AutoTokenizer

    from posttrain_circuits.experiments.protocols.student_preparation import (
        CHAT_TEMPLATE_SHA256,
        TOKENIZER_FINGERPRINT,
    )
    from posttrain_circuits.models.loading import tokenizer_fingerprint

    token_raw = read(ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json", 12 * 1024**2)
    metadata_raw = read(ROOT / ".sdsc/diagnostics/student-quality-v1/pinned-tokenizer-metadata.json")
    require(
        sha(token_raw) == TOKENIZER_SHA and sha(metadata_raw) == METADATA_SHA,
        "original tokenizer bytes differ",
    )
    items = {row["name"]: row for row in document(metadata_raw)["items"]}
    with tempfile.TemporaryDirectory(prefix="student-preparation-tokenizer-") as temporary:
        directory = Path(temporary)
        (directory / "tokenizer.json").write_bytes(token_raw)
        for name in ("config.json", "tokenizer_config.json"):
            (directory / name).write_text(items[name]["text"])
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    require(
        tokenizer.eos_token_id == 151645 and tokenizer.clean_up_tokenization_spaces is False,
        "tokenizer termination/decoding changed",
    )
    require(
        sha(tokenizer.chat_template.encode()) == CHAT_TEMPLATE_SHA256
        and tokenizer_fingerprint(tokenizer) == TOKENIZER_FINGERPRINT,
        "tokenizer identity differs",
    )
    return tokenizer


def replay(report, prompts, rank_records, tokenizer):
    """No producer scoring/selection functions are used to recompute outcomes."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_preparation as protocol
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    require(report.get("mode") in ("preflight", "fit"), "unknown preparation mode")
    require(all(report.get(key) is False for key in FLAGS), "producer overclaims acceptance")
    fit = report["mode"] == "fit"
    examples = protocol.make_examples(protocol.DIFFICULTY)["student_dev" if fit else "student_fit"]
    examples = examples if fit else examples[:8]
    steps = (4, 8, 16, 32) if fit else (4,)
    require(len(rank_records) == 2, "both rank records required")
    require(len(report["checkpoints"]) == len(steps), "checkpoint schedule differs")
    checkpoints = {}
    for step, row in zip(steps, report["checkpoints"], strict=True):
        require(
            row["step"] == step and re.fullmatch("[a-f0-9]{64}", row["sha256"]), "checkpoint identity differs"
        )
        require(row["sha256"] not in checkpoints.values(), "duplicate checkpoint bytes")
        checkpoints[step] = row["sha256"]
    if fit:
        require(len(prompts) == len(examples), "development prompts missing")
    else:
        require(prompts == [], "preflight must not publish development prompts")
    task, expected = ProofGraphTask(), {}
    model_config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": protocol.CHAT_TEMPLATE_SHA256,
        }
    }
    for ordinal, example in enumerate(examples):
        text = format_model_prompt(task.render(example), tokenizer, model_config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(len(ids) + 256 <= 1600, "preparation generation envelope differs")
        if fit:
            same(
                prompts[ordinal],
                dict(ordinal=ordinal, example=asdict(example), prompt_ids=ids),
                "prompt population/encoding differs",
            )
        expected[ordinal] = (example, text, ids)
    by_step = {
        step: dict(
            step=step,
            checkpoint_sha256=checkpoints[step],
            examples_sha256=protocol.EXAMPLES_SHA256["student_dev"],
            num_examples=0,
            answer_correct=0,
            proof_correct=0,
            format_valid=0,
        )
        for step in steps
    }
    seen = set()
    vocabulary_size = len(tokenizer)
    for rank, records in enumerate(rank_records):
        schedule = [(step, ordinal) for step in steps for ordinal in range(rank, len(examples), 2)]
        require(len(records) == len(schedule), "rank response count differs")
        for record, (step, ordinal) in zip(records, schedule, strict=True):
            same(
                [record["rank"], record["step"], record["ordinal"]],
                [rank, step, ordinal],
                "rank/step/ordinal order differs",
            )
            require((step, ordinal) not in seen, "duplicate response")
            seen.add((step, ordinal))
            example, text, ids = expected[ordinal]
            same(
                [record["example_id"], record["pair_group_id"], record["expected_label"]],
                [example.example_id, example.pair_group_id, example.label],
                "response population differs",
            )
            require(
                record["cohort"] == ("student_dev" if fit else "student_fit_preflight_first8")
                and record["checkpoint_sha256"] == checkpoints[step],
                "response checkpoint/cohort differs",
            )
            same(record["prompt_ids"], ids, "response prefix changed")
            require(
                record["prompt_token_sha256"] == sha(canonical(ids))
                and record["prompt_text_sha256"] == sha(text.encode()),
                "prompt hashes differ",
            )
            tokens = record["response_ids"]
            require(
                isinstance(tokens, list)
                and 0 < len(tokens) <= 256
                and all(type(token) is int and 0 <= token < vocabulary_size for token in tokens),
                "invalid response token IDs",
            )
            require(tokenizer.eos_token_id not in tokens[:-1], "tokens after EOS")
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            require(
                record["response_text"] == response
                and record["response_tokens"] == len(tokens)
                and record["max_new_tokens"] == 256,
                "response decoding/budget differs",
            )
            stop = (
                "eos" if tokens[-1] == tokenizer.eos_token_id else "length" if len(tokens) == 256 else "other"
            )
            require(record["stop_reason"] == stop and stop != "other", "generation termination differs")
            parsed = task.parse_response(response)
            verified = task.verify(example, parsed)
            same(record["parsed_trace"], asdict(parsed), "parser trace differs")
            same(record["verification"], asdict(verified), "verifier trace differs")
            require(record["parse_error"] == parsed.error_code, "parse error differs")
            tags = re.findall(r"<answer>\s*([01])\s*</answer>", response)
            require(
                record["diagnostic_answer_tag_present"] is (len(tags) == 1)
                and record["diagnostic_answer_tag_correct"]
                is (len(tags) == 1 and int(tags[0]) == example.label),
                "diagnostic answer tags differ",
            )
            counts = by_step[step]
            counts["num_examples"] += 1
            counts["answer_correct"] += int(verified.answer_correct)
            counts["proof_correct"] += int(verified.reward == 1.0)
            counts["format_valid"] += int(verified.parse_valid)
    development = [by_step[step] for step in steps] if fit else []
    same(report["development"], development, "development reduction differs")
    selected = next((row for row in development if 52 <= row["answer_correct"] <= 307), None)
    same(report["selected_checkpoint"], selected, "earliest preparation selection differs")
    require(report["passed"] is (selected is not None if fit else True), "preparation outcome differs")
    return dict(
        raw_replay_passed=True,
        mode=report["mode"],
        prompts_replayed=len(examples),
        responses_replayed=len(seen),
        development=development,
        selected_checkpoint=selected,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
        **dict.fromkeys(FLAGS, False),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--fetch-dir", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from posttrain_circuits.experiments.protocols.student_preparation import (
        resolve_student_preparation_protocol,
    )

    plan = document(read(args.plan))
    git_dir = ROOT / (".opd-git" if (ROOT / ".opd-git").is_dir() else ".git")
    binding = resolve_student_preparation_protocol(ROOT, git_dir=git_dir)
    same(
        plan["protocol"]["science_file_sha256"],
        binding.science_file_sha256,
        "reviewed source binding differs",
    )
    same(
        [plan["protocol"][key] for key in ("implementation_commit", "acceptance_commit", "protocol_sha256")],
        [binding.implementation_commit, binding.acceptance_commit, binding.protocol_sha256],
        "reviewed protocol lineage differs",
    )
    raw = read(args.fetch_dir / "receipt.json")
    require(sha(raw) == args.receipt_sha256, "externally bound publication hash differs")
    publication = document(raw)
    require(publication["plan_sha256"] == sha(canonical(plan)), "publication plan differs")
    rows = publication["files"]
    require(len({row["path"] for row in rows}) == len(rows), "duplicate published file")
    files = {}
    for row in rows:
        name = row["path"]
        require(isinstance(name, str) and Path(name).name == name, "unsafe fetched filename")
        data = read(args.fetch_dir / name)
        require(len(data) == row["size"] and sha(data) == row["sha256"], "published file hash differs")
        files[name] = data
    report = document(files["prepare-report.json"])
    same(
        [report["mode"], report["protocol_sha256"], report["protocol_artifact_sha256"]],
        [plan["mode"], binding.protocol_sha256, binding.artifact_sha256],
        "report protocol/mode differs",
    )
    same(
        [report["job_id"], report["run_id"], report["plan_sha256"], report["source_code_sha256"]],
        [publication["job_id"], plan["run_id"], sha(canonical(plan)), plan["code_sha256"]],
        "report binding differs",
    )
    inventory = {row["path"]: row for row in publication["large_files"]}
    require(len(inventory) == len(publication["large_files"]), "duplicate persistent checkpoint")
    for checkpoint in report["checkpoints"]:
        row = {key: checkpoint[key] for key in ("path", "size", "sha256")}
        same(inventory.get(row["path"]), row, "checkpoint not bound to persistent publication")
    fit = report["mode"] == "fit"
    count, prefix = (1024, "dev") if fit else (4, "preflight")
    prompts = lines(files["prepare-dev-prompts.jsonl"], 512) if fit else []
    records = [lines(files[f"prepare-{prefix}-records-rank-{rank}.jsonl"], count) for rank in range(2)]
    result = replay(report, prompts, records, load_tokenizer())
    result.update(
        job_id=publication["job_id"],
        publication_sha256=args.receipt_sha256,
        plan_sha256=sha(canonical(plan)),
        auditor_sha256=sha(Path(__file__).read_bytes()),
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
