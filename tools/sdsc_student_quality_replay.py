#!/usr/bin/env python3
"""Bounded, local text/verifier replay of diagnostic 54557365, never acceptance.

Input receipt and plan SHA-256 values are external trust anchors from the verified
fetch/operational review. No SSH, tokenizer download, model, GPU, or scheduler.
Saved text is hash-bound, not independently decoded from saved token IDs. NLL
cannot be recomputed without logits; its finite/range/cross-cap checks are named
separately. The validation cohort was already exposed, not a new test holdout.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = "54557365"
PARENT = "54548846"
TASK = "qwen3-v2-student-quality-diagnostic-v1"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MAX_FILE = 8 * 1024**2
MAX_TOTAL = 16 * 1024**2
NAMES = (
    "quality-probe.json",
    "quality-records.jsonl",
    "quality-prompts.jsonl",
    "node-result.json",
    "memory.json",
)
FALSE_FLAGS = ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready")
COHORTS = (("validation_exposed", 128), ("train_diagnostic", 32))
LABELS = ("initial", "step20", "step33")
METADATA_SHA = {
    "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same(left, right, message):
    require(canonical(left) == canonical(right), message)


def read(path, maximum=MAX_FILE):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink input rejected")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, "input size/type")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) == before.st_size == after.st_size
        and (before.st_mtime_ns, before.st_ctime_ns) == (after.st_mtime_ns, after.st_ctime_ns)
        and len(raw) <= maximum,
        "input changed or oversized",
    )
    return raw


def document(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(result, dict), "JSON object required")
    canonical(result)  # Reject exponent overflow too, not only NaN/Infinity literals.
    return result


def jsonlines(raw, expected):
    lines = raw.splitlines()
    require(len(lines) == expected and all(line.strip() for line in lines), "JSONL row count/blank")
    return [document(line) for line in lines]


def no_acceptance(value):
    require(all(value.get(key) is False for key in FALSE_FLAGS), "diagnostic acceptance overclaim")


def token_ids(value):
    require(isinstance(value, list) and all(type(x) is int and x >= 0 for x in value), "invalid token IDs")
    return value


def metadata_contract(raw):
    value = document(raw)
    require(value.get("revision") == REVISION, "tokenizer revision differs")
    items = value.get("items")
    require(isinstance(items, list), "tokenizer metadata inventory missing")
    names = [item.get("name") for item in items]
    require(len(names) == len(set(names)) and set(METADATA_SHA) <= set(names), "tokenizer metadata names")
    decoded = {}
    for item in items:
        text = item["text"].encode()
        require(len(text) == item["size"] and sha(text) == item["sha256"], "tokenizer metadata bytes")
        if item["name"] in METADATA_SHA:
            require(item["sha256"] == METADATA_SHA[item["name"]], "unpinned tokenizer metadata")
            require(
                f"/snapshots/{REVISION}/" in item["path"] and item["path"].endswith("/" + item["name"]),
                "tokenizer source revision/path differs",
            )
            decoded[item["name"]] = document(text)
    config, tokenizer = decoded["config.json"], decoded["tokenizer_config.json"]
    eos = config["eos_token_id"]
    require(type(eos) is int and eos == 151645, "unexpected EOS metadata")
    require(
        tokenizer["eos_token"] == "<|im_end|>"
        and tokenizer["added_tokens_decoder"][str(eos)]["content"] == "<|im_end|>"
        and config["max_position_embeddings"] == 40960,
        "EOS/position metadata differs",
    )
    return eos, config["max_position_embeddings"]


def checkpoint_contract(report, plan):
    evidence = report.get("checkpoint_evidence")
    require(isinstance(evidence, list) and len(evidence) == 3, "checkpoint evidence count")
    require([x["label"] for x in plan["checkpoints"]] == list(LABELS), "plan checkpoint order")
    for label, expected, actual in zip(LABELS, plan["checkpoints"], evidence, strict=True):
        same(
            actual,
            {
                **{key: expected[key] for key in ("label", "size", "sha256")},
                "saved_dtype": "torch.bfloat16" if label == "initial" else "torch.float32",
                "loaded_exactly": True,
                "forward_parameter_dtype": "torch.bfloat16",
            },
            "checkpoint/precision evidence differs",
        )


def summary(rows):
    count = len(rows)
    return {
        "count": count,
        "validation_accuracy": sum(row["verification"]["answer_correct"] for row in rows) / count,
        "exact_proof_accuracy": sum(row["verification"]["reward"] for row in rows) / count,
        "format_validity": sum(row["verification"]["parse_valid"] for row in rows) / count,
        "diagnostic_answer_tag_accuracy": sum(row["diagnostic_answer_tag_correct"] for row in rows) / count,
        "stop_counts": dict(collections.Counter(row["stop_reason"] for row in rows)),
        "error_counts": dict(
            collections.Counter(row["verification"]["error_code"] or "accepted" for row in rows)
        ),
    }


def teacher_forced_integrity(value, count):
    require(
        set(value)
        == {
            "examples",
            "response_tokens_including_eos",
            "canonical_response_nll",
            "canonical_response_token_accuracy",
        },
        "teacher-forced metric fields differ",
    )
    require(type(value["examples"]) is int and value["examples"] == count, "teacher-forced cohort count")
    tokens = value["response_tokens_including_eos"]
    require(type(tokens) is int and count <= tokens <= count * 1536, "teacher-forced token count")
    for key in ("canonical_response_nll", "canonical_response_token_accuracy"):
        number = value[key]
        require(
            type(number) in (int, float) and math.isfinite(number) and number >= 0,
            "nonfinite/invalid teacher-forced metric",
        )
    require(value["canonical_response_token_accuracy"] <= 1, "teacher-forced accuracy range")


def replay_rows(report, prompts, records, *, eos, positions):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example

    task = ProofGraphTask()
    require(len(prompts) == 160 and len(records) == 960, "expected 160 prompts and 960 responses")
    require(
        report.get("raw_record_count") == 960 and type(report["raw_record_count"]) is int, "report raw count"
    )
    cohorts, cursor, all_ids = {}, 0, set()
    for cohort, count in COHORTS:
        rows = prompts[cursor : cursor + count]
        cursor += count
        examples = []
        for row in rows:
            require(
                set(row)
                == {
                    "cohort",
                    "example",
                    "prompt_text",
                    "prompt_ids",
                    "prompt_token_sha256",
                    "canonical_target",
                },
                "prompt fields differ",
            )
            require(row["cohort"] == cohort, "prompt cohort order differs")
            example = deserialize_example(row["example"])
            same(asdict(example), row["example"], "lossy/invalid ProofGraph example")
            require(example.example_id and example.example_id not in all_ids, "duplicate prompt ID")
            all_ids.add(example.example_id)
            ids = token_ids(row["prompt_ids"])
            require(0 < len(ids) < 1536, "prompt token envelope")
            require(sha(canonical(ids)) == row["prompt_token_sha256"], "prompt token hash differs")
            require(
                isinstance(row["prompt_text"], str) and task.render(example) in row["prompt_text"],
                "prompt graph text differs",
            )
            require(row["canonical_target"] == render_target(example), "canonical target differs")
            examples.append((example, row))
        cohorts[cohort] = examples
    same(
        report["cohorts"],
        {
            cohort: {"count": count, "ordered_ids": [x.example_id for x, _ in cohorts[cohort]]}
            for cohort, count in COHORTS
        },
        "cohort IDs/order/count differ",
    )
    require(isinstance(report.get("arms"), list) and len(report["arms"]) == 12, "expected 12 arms")
    index, offset, rebuilt = 0, 0, []
    record_fields = {
        "checkpoint",
        "cohort",
        "cap",
        "example_id",
        "pair_group_id",
        "expected_label",
        "prompt_token_sha256",
        "prompt_text_sha256",
        "response_ids",
        "response_text",
        "response_tokens",
        "max_new_tokens",
        "stop_reason",
        "parse_error",
        "verification",
        "diagnostic_answer_tag_present",
        "diagnostic_answer_tag_correct",
    }
    for label in LABELS:
        for cohort, count in COHORTS:
            previous, previous_nll = None, None
            for cap in (128, 256):
                arm = report["arms"][index]
                index += 1
                require(
                    set(arm)
                    == {
                        "checkpoint",
                        "cohort",
                        "cap",
                        "metrics",
                        "teacher_forced",
                        "raw_file",
                        "record_start",
                        "record_count",
                    },
                    "arm fields differ",
                )
                same(
                    {
                        k: arm[k]
                        for k in ("checkpoint", "cohort", "cap", "raw_file", "record_start", "record_count")
                    },
                    {
                        "checkpoint": label,
                        "cohort": cohort,
                        "cap": cap,
                        "raw_file": "quality-records.jsonl",
                        "record_start": offset,
                        "record_count": count,
                    },
                    "arm ordering/range differs",
                )
                rows = records[offset : offset + count]
                offset += count
                for row, (example, prompt) in zip(rows, cohorts[cohort], strict=True):
                    require(set(row) == record_fields, "response fields differ")
                    expected = {
                        "checkpoint": label,
                        "cohort": cohort,
                        "cap": cap,
                        "example_id": example.example_id,
                        "pair_group_id": example.pair_group_id,
                        "expected_label": example.label,
                        "prompt_token_sha256": prompt["prompt_token_sha256"],
                        "prompt_text_sha256": sha(prompt["prompt_text"].encode()),
                    }
                    same({k: row[k] for k in expected}, expected, "record identity/order/prompt differs")
                    ids = token_ids(row["response_ids"])
                    limit = min(cap, 1536 - len(prompt["prompt_ids"]), positions - len(prompt["prompt_ids"]))
                    same(
                        [row["response_tokens"], row["max_new_tokens"]],
                        [len(ids), limit],
                        "response length/cap differs",
                    )
                    require(0 < len(ids) <= limit, "response outside generation envelope")
                    ended = ids[-1] == eos
                    require(eos not in ids[:-1], "EOS precedes final generated token")
                    stop = "eos" if ended else "length" if len(ids) >= limit else "other"
                    require(row["stop_reason"] == stop, "stop reason differs")
                    text = row["response_text"]
                    require(isinstance(text, str), "response text must be a string")
                    parsed = task.parse_response(text)
                    verified = task.verify(example, parsed)
                    same(row["parse_error"], parsed.error_code, "parse error differs")
                    same(row["verification"], asdict(verified), "verification trace differs")
                    tags = re.findall(r"<answer>\s*([01])\s*</answer>", text)
                    same(
                        [row["diagnostic_answer_tag_present"], row["diagnostic_answer_tag_correct"]],
                        [len(tags) == 1, len(tags) == 1 and int(tags[0]) == example.label],
                        "answer-tag diagnostic differs",
                    )
                metrics = summary(rows)
                if previous is not None:
                    metrics["paired_prefix_equal_count"] = sum(
                        left["response_ids"] == right["response_ids"][: len(left["response_ids"])]
                        for left, right in zip(previous, rows, strict=True)
                    )
                    same(arm["teacher_forced"], previous_nll, "teacher-forced values differ across caps")
                teacher_forced_integrity(arm["teacher_forced"], count)
                same(arm["metrics"], metrics, "aggregate metrics differ")
                rebuilt.append({"checkpoint": label, "cohort": cohort, "cap": cap, "metrics": metrics})
                previous, previous_nll = rows, arm["teacher_forced"]
    return rebuilt


def replay(fetch_dir, plan_path, *, plan_sha256, receipt_sha256, tokenizer_metadata):
    plan_raw = read(plan_path)
    require(sha(plan_raw) == plan_sha256, "plan SHA differs from external anchor")
    plan = document(plan_raw)
    require(sha(canonical(plan)) == plan_sha256, "plan canonical SHA differs")
    require(
        plan["schema"] == "quest-sdsc-student-quality-plan-v1"
        and plan["task"] == TASK
        and plan["scope"] == "inference_diagnostic_only"
        and plan["parent"]["receipt"]["job_id"] == PARENT,
        "plan scope/parent differs",
    )
    no_acceptance(plan)
    fetch = Path(fetch_dir)
    receipt_raw = read(fetch / "receipt.json")
    require(sha(receipt_raw) == receipt_sha256, "receipt SHA differs from external anchor")
    receipt = document(receipt_raw)
    no_acceptance(receipt)
    require(
        receipt.get("passed") is True
        and receipt.get("diagnostic_complete") is True
        and receipt.get("persistent_read_back_verified") is True,
        "publication incomplete",
    )
    for key, expected in {
        "task": TASK,
        "job_id": JOB,
        "run_id": plan["run_id"],
        "intent_id": plan["intent_id"],
        "plan_sha256": plan_sha256,
        "code_sha256": plan["code_sha256"],
    }.items():
        same(receipt.get(key), expected, "publication identity differs: " + key)
    rows = receipt["files"]
    require(isinstance(rows, list) and len(rows) == len(NAMES), "publication inventory count")
    inventory = {}
    for row in rows:
        require(
            set(row) == {"path", "size", "sha256"} and row["path"] in NAMES and row["path"] not in inventory,
            "publication inventory entry",
        )
        inventory[row["path"]] = row
    raw_files = {name: read(fetch / name) for name in NAMES}
    require(sum(map(len, raw_files.values())) <= MAX_TOTAL, "publication too large")
    for name, raw in raw_files.items():
        same(
            inventory[name],
            {"path": name, "size": len(raw), "sha256": sha(raw)},
            "published file hash/size differs: " + name,
        )
    report = document(raw_files["quality-probe.json"])
    node = document(raw_files["node-result.json"])
    document(raw_files["memory.json"])
    for value in (report, node):
        no_acceptance(value)
        require(value.get("diagnostic_complete") is True, "incomplete node/worker")
        for key, expected in {
            "job_id": JOB,
            "parent_job_id": PARENT,
            "run_id": plan["run_id"],
            "source_code_sha256": plan["code_sha256"],
        }.items():
            same(value.get(key), expected, "node/worker identity differs: " + key)
    require(
        report.get("schema") == "quest-sdsc-student-quality-probe-v1" and report.get("passed") is True,
        "worker schema/success differs",
    )
    same(node.get("plan_sha256"), plan_sha256, "node plan differs")
    same(node.get("exit_code"), 0, "node exit differs")
    same(
        report["raw_artifacts"],
        [inventory[name] for name in ("quality-records.jsonl", "quality-prompts.jsonl")],
        "raw report hashes differ",
    )
    manifest = [row for row in plan["dataset_inputs"] if row["path"] == "manifest.json"]
    require(len(manifest) == 1, "plan dataset manifest missing/duplicate")
    same(report["dataset_manifest_sha256"], manifest[0]["sha256"], "dataset manifest identity differs")
    same(
        report["generation"],
        {"do_sample": False, "use_cache": False, "caps": [128, 256], "max_model_input_length": 1536},
        "generation policy differs",
    )
    checkpoint_contract(report, plan)
    metadata_raw = read(tokenizer_metadata)
    eos, positions = metadata_contract(metadata_raw)
    arms = replay_rows(
        report,
        jsonlines(raw_files["quality-prompts.jsonl"], 160),
        jsonlines(raw_files["quality-records.jsonl"], 960),
        eos=eos,
        positions=positions,
    )
    parser_sources = {}
    for name in ("contracts", "generation", "parsing", "verification", "rendering", "serialization"):
        path = f"src/posttrain_circuits/datasets/proofgraph/{name}.py"
        parser_sources[path] = sha(read(ROOT / path))
    return {
        "schema": "quest-sdsc-student-quality-local-replay-v1",
        "job_id": JOB,
        "parent_job_id": PARENT,
        "passed": True,
        "diagnostic_replay_complete": True,
        **{key: False for key in FALSE_FLAGS},
        "new_test_holdout": False,
        "plan_sha256": plan_sha256,
        "publication_receipt_sha256": receipt_sha256,
        "source_code_sha256": plan["code_sha256"],
        "files": inventory,
        "tokenizer_metadata_sha256": sha(metadata_raw),
        "pinned_metadata_files": METADATA_SHA,
        "eos_token_id": eos,
        "max_position_embeddings": positions,
        "parser_source_sha256": parser_sources,
        "prompt_count": 160,
        "response_count": 960,
        "arms": arms,
        "checks": {
            key: True
            for key in (
                "external_plan_and_receipt_hashes",
                "complete_publication_file_hashes",
                "checkpoint_and_dataset_binding",
                "unique_ordered_prompts_and_12_arms",
                "canonical_targets_and_prompt_hashes",
                "full_text_parser_and_verifier_traces",
                "generation_aggregates_and_answer_tags",
                "token_stop_reasons_and_256_prefixes",
                "teacher_forced_finite_range_and_cross_cap_consistency",
            )
        },
        "limitations": [
            "Saved response text is receipt-hash-bound; "
            "response IDs were not decoded with the full tokenizer.",
            "Teacher-forced NLL/token accuracy were not recomputed: logits are absent.",
            "Original full dataset splits were not reread; cohort membership is bound "
            "to published examples, ordered IDs and dataset-manifest SHA.",
            "This is the exposed validation cohort plus training diagnostics, "
            "not a new test holdout or a model/G0 acceptance.",
            "No accounting query, GPU inference, checkpoint rehash, or memory-envelope replay occurs here.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--tokenizer-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    result = replay(
        args.fetch_dir,
        args.plan,
        plan_sha256=args.plan_sha256,
        receipt_sha256=args.receipt_sha256,
        tokenizer_metadata=args.tokenizer_metadata,
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(args.output.absolute()),
                "job_id": JOB,
                "prompts": 160,
                "responses": 960,
                "student_accepted": False,
                "g0_passed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
