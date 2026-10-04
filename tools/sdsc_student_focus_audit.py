#!/usr/bin/env python3
"""Independent raw replay for the prospective common-student branch-focused fit.

Binds genuine reviewed sources and persistent publications, reconstructs all
views and token boundaries, then recomputes original verification, independent
view/structure reductions and the prospective earliest checkpoint choice. It
neither replays GPU optimization nor accepts a model, G0 or execution class.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = 224 * 1024**2
STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
VIEWS = (
    "identity",
    "entity_symbol_renaming",
    "fact_order_permutation",
    "rule_order_permutation",
    "surface_template_paraphrase",
    "distractor_count_ood",
)
STRUCTURES = ("chain", "branch", "converging_dag")
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"


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
    # This historical helper verifies actual externally pinned tokenizer bytes;
    # it performs no producer scoring or checkpoint selection.
    spec = importlib.util.spec_from_file_location(
        "focus_tokenizer", ROOT / "tools/sdsc_student_prepare_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tokenizer()


def population(tokenizer, mode):
    require(mode in ("preflight", "fit"), "unknown focus mode")
    from posttrain_circuits.experiments.protocols import student_focus_preparation as protocol
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    require(tuple(protocol.CHECKPOINT_STEPS) == STEPS, "focus checkpoint schedule differs")
    require(
        (protocol.FIT_BASE_COUNT, protocol.FIT_COUNT, protocol.DEV_BASE_COUNT, protocol.DEV_COUNT)
        == (256, 2048, 256, 1536),
        "focus preparation population differs",
    )
    splits = protocol.make_examples(protocol.DIFFICULTY)
    fit = mode == "fit"
    views = (
        protocol.development_views(splits["student_dev"])
        if fit
        else protocol.fit_views(splits["student_fit"])[:8]
    )
    expected_count = 1536 if fit else 8
    require(len(views) == expected_count, "reconstructed view population differs")
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": protocol.CHAT_TEMPLATE_SHA256,
        }
    }
    prompts = []
    for ordinal, view in enumerate(views):
        text = format_model_prompt(view.prompt, tokenizer, config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(len(ids) > 0 and len(ids) + 256 <= 2454, "generation envelope differs")
        prompts.append(
            dict(
                ordinal=ordinal,
                example=asdict(view.example),
                prompt_ids=ids,
                source_example_id=view.source_example_id,
                view=view.view,
                prompt_text=text,
            )
        )
    digest = sha(canonical([asdict(row) for row in splits["student_dev"]]))
    return prompts, [row.example for row in views], digest


def empty_counts():
    return dict(num_examples=0, answer_correct=0, proof_correct=0, format_valid=0)


def add_counts(counts, result):
    counts["num_examples"] += 1
    counts["answer_correct"] += int(result.answer_correct)
    counts["proof_correct"] += int(result.reward == 1.0)
    counts["format_valid"] += int(result.parse_valid)


def eligible(row):
    """Independent exact-integer interpretation of the frozen selection gates."""
    by_view = row["per_view_counts"]
    iid = by_view["identity"]
    others = [by_view[name] for name in VIEWS[1:]]
    require(
        iid["num_examples"] == 256 and all(x["num_examples"] == 256 for x in others),
        "selection population differs",
    )
    total = sum(x["proof_correct"] for x in others)
    # Multiplication preserves the signed gap; stronger transformed results are
    # allowed. No floating-point boundary approximation or producer helper.
    return (
        26 <= iid["answer_correct"] <= 153
        and iid["proof_correct"] * 10 >= 256
        and total * 100 >= 8 * 1280
        and all(x["proof_correct"] * 20 >= 256 for x in others)
        and (5 * iid["proof_correct"] - total) * 20 <= 1280
        and all((iid["proof_correct"] - x["proof_correct"]) * 20 <= 256 for x in others)
        and all(
            x["num_examples"] > 0 and x["proof_correct"] * 10 >= x["num_examples"]
            for x in row["structure_counts"].values()
        )
    )


def replay(report, prompts, step_records, tokenizer):
    """Recompute all records without importing a producer reducer or selector."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    require(report.get("mode") in ("preflight", "fit"), "unknown focus mode")
    require(report.get("schema") == "quest-sdsc-student-focus-report-v1", "report schema differs")
    require(all(report.get(key) is False for key in FLAGS), "producer overclaims acceptance")
    require(
        "arm" not in report and "diagnostic_complete" not in report, "diagnostic report identity forbidden"
    )
    fit = report["mode"] == "fit"
    require(report.get("execution_complete") is True, "full preparation execution incomplete")
    require(report.get("preparation_complete") is fit, "worker preparation completion differs")
    same(
        [
            report.get(k)
            for k in (
                "optimizer_steps",
                "fit_base_examples",
                "fit_training_views",
                "consumed_fit_training_views",
                "development_base_examples",
                "development_views",
                "source_kind",
            )
        ],
        [
            32 if fit else 4,
            256,
            2048,
            2048 if fit else 256,
            256,
            1536,
            "symbolic_canonical_focus_preparation",
        ],
        "full preparation population or updates differ",
    )
    same(
        report.get("development_generation"),
        dict(
            max_new_tokens=256,
            max_model_input_length=2454,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
            truncation=False,
        ),
        "declared inference contract differs",
    )
    fit = report["mode"] == "fit"
    schedule = STEPS if fit else (4,)
    require(set(step_records) == set(schedule), "raw checkpoint schedule differs")
    expected, examples, dev_sha = population(tokenizer, report["mode"])
    same(prompts, expected if fit else [], "prompt population/encoding differs")
    require(len(report["checkpoints"]) == len(schedule), "checkpoint schedule differs")
    checkpoints = {}
    for step, row in zip(schedule, report["checkpoints"], strict=True):
        require(
            type(row["step"]) is int
            and row["step"] == step
            and isinstance(row["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
            "checkpoint identity differs",
        )
        require(row["sha256"] not in checkpoints.values(), "duplicate checkpoint bytes")
        checkpoints[step] = row["sha256"]
    task, seen, development = ProofGraphTask(), set(), []
    vocabulary_size = len(tokenizer)
    for step in schedule:
        ranks = step_records[step]
        require(isinstance(ranks, list) and len(ranks) == 2, "both rank records required")
        groups = {name: empty_counts() for name in VIEWS}
        structures = {name: empty_counts() for name in STRUCTURES}
        for rank, records in enumerate(ranks):
            ordinals = [
                ordinal
                for ordinal in range(len(examples))
                if ((ordinal // 6) % 2 if fit else ordinal % 2) == rank
            ]
            require(len(records) == len(ordinals), "rank response count differs")
            for record, ordinal in zip(records, ordinals, strict=True):
                same(
                    [record["rank"], record["step"], record["ordinal"]],
                    [rank, step, ordinal],
                    "rank/step/ordinal order differs",
                )
                require((step, ordinal) not in seen, "duplicate response")
                seen.add((step, ordinal))
                example, prompt = examples[ordinal], expected[ordinal]
                same(
                    [
                        record["example_id"],
                        record["pair_group_id"],
                        record["expected_label"],
                        record["source_example_id"],
                        record["view"],
                    ],
                    [
                        example.example_id,
                        example.pair_group_id,
                        example.label,
                        prompt["source_example_id"],
                        prompt["view"],
                    ],
                    "response population/view differs",
                )
                require(
                    record["checkpoint_sha256"] == checkpoints[step]
                    and record["cohort"]
                    == ("student_focus_dev" if fit else "student_focus_fit_preflight_first8"),
                    "response checkpoint/cohort differs",
                )
                ids, text = prompt["prompt_ids"], prompt["prompt_text"]
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
                    and all(type(x) is int and 0 <= x < vocabulary_size for x in tokens),
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
                    "eos"
                    if tokens[-1] == tokenizer.eos_token_id
                    else "length"
                    if len(tokens) == 256
                    else "other"
                )
                require(record["stop_reason"] == stop and stop != "other", "termination differs")
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
                    "diagnostic tags differ",
                )
                if fit:
                    require(prompt["view"] in groups, "unknown development view")
                    add_counts(groups[prompt["view"]], verified)
                    if prompt["view"] == "identity":
                        add_counts(structures[example.metadata["structure"]], verified)
        if fit:
            require(all(x["num_examples"] == 256 for x in groups.values()), "incomplete development views")
            development.append(
                dict(
                    step=step,
                    checkpoint_sha256=checkpoints[step],
                    examples_sha256=dev_sha,
                    **groups["identity"],
                    per_view_counts=groups,
                    structure_counts=structures,
                )
            )
    same(report["development"], development, "development reduction differs")
    selected = next((row for row in development if eligible(row)), None)
    same(report["selected_checkpoint"], selected, "earliest focus selection differs")
    require(report["passed"] is (selected is not None if fit else True), "preparation outcome differs")
    return dict(
        schema="quest-sdsc-student-focus-raw-audit-v1",
        preparation_complete=fit,
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


def verify_producer_head(plan, binding, git_dir):
    from posttrain_circuits.artifacts.git_provenance import _git_environment

    head = plan["protocol"]["head"]
    require(isinstance(head, str) and re.fullmatch(r"[a-f0-9]{40}", head), "invalid producer head")
    same(head, plan["provenance"]["head"], "provenance producer head differs")
    for older, newer in ((binding.acceptance_commit, head), (head, binding.head)):
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(git_dir),
                "--work-tree",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                older,
                newer,
            ],
            capture_output=True,
            check=False,
            timeout=30,
            env=_git_environment(),
        )
        require(result.returncode == 0, "producer head outside accepted genuine lineage")


def validate_publication(plan, publication_raw, receipt_sha256, files, binding):
    require(sha(publication_raw) == receipt_sha256, "externally bound publication hash differs")
    publication = document(publication_raw)
    require(
        plan.get("schema") == "quest-sdsc-student-focus-plan-v1"
        and plan.get("task") == "qwen3-v2-student-focus-preparation-v5"
        and plan.get("mode") in ("preflight", "fit"),
        "plan identity differs",
    )
    require("arm" not in plan and "diagnostic_complete" not in plan, "diagnostic plan identity forbidden")
    require(
        set(publication)
        == {
            "schema",
            "task",
            "mode",
            "job_id",
            "run_id",
            "intent_id",
            "plan_sha256",
            "code_sha256",
            "passed",
            "stage_complete",
            "preparation_complete",
            "persistent_read_back_verified",
            "large_files_read_back_verified",
            "files",
            "large_files",
            *FLAGS,
        },
        "flat preparation publication fields differ",
    )
    require(
        type(publication.get("stage_complete")) is bool
        and publication.get("passed") is publication["stage_complete"]
        and publication.get("preparation_complete")
        is (publication["stage_complete"] and plan["mode"] == "fit"),
        "publication completion flags differ",
    )
    plan_sha = sha(canonical(plan))
    for key, value in dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=plan["task"],
        mode=plan["mode"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=plan_sha,
        code_sha256=plan["code_sha256"],
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
    ).items():
        same(publication.get(key), value, "publication identity differs: " + key)
    require(re.fullmatch(r"[1-9][0-9]*", str(publication.get("job_id", ""))), "publication job missing")
    require(
        all(plan.get(k) is False and publication.get(k) is False for k in FLAGS),
        "plan/publication overclaims acceptance",
    )
    same(plan["protocol"]["science_file_sha256"], binding.science_file_sha256, "reviewed source differs")
    for key in ("implementation_commit", "acceptance_commit", "protocol_sha256", "artifact_sha256"):
        same(plan["protocol"][key], getattr(binding, key), "reviewed protocol lineage differs: " + key)
    same(
        plan["control_sha256"]["tools/sdsc_student_focus_audit.py"],
        sha(Path(__file__).read_bytes()),
        "auditor is not the pinned implementation",
    )
    rows = publication["files"]
    require(
        isinstance(rows, list) and 0 < len(rows) <= 48 and len({x["path"] for x in rows}) == len(rows),
        "duplicate/invalid published files",
    )
    require(sum(x["size"] for x in rows) <= MAX_TOTAL_BYTES, "publication exceeds bounded fetch")
    for row in rows:
        name = row["path"]
        require(
            isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
            "unsafe published filename",
        )
        require(
            set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 < row["size"] <= (MAX_BYTES if name == "prepare-dev-prompts.jsonl" else 16 * 1024**2)
            and re.fullmatch(r"[a-f0-9]{64}", str(row["sha256"])),
            "invalid published file identity",
        )
        require(
            name in files and len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
            "published file hash differs",
        )
    require(set(files) == {x["path"] for x in rows}, "unbound fetched publication files")
    report = document(files["prepare-report.json"])
    for key, value in dict(
        mode=plan["mode"],
        job_id=publication["job_id"],
        run_id=plan["run_id"],
        plan_sha256=plan_sha,
        source_code_sha256=plan["code_sha256"],
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        initial_checkpoint_sha256=INITIAL_SHA,
    ).items():
        same(report.get(key), value, "report binding differs: " + key)
    require(all(report.get(k) is False for k in FLAGS), "report overclaims acceptance")
    require(
        not publication["stage_complete"]
        or (report.get("passed") is True and report.get("execution_complete") is True),
        "positive publication contradicts worker outcome",
    )
    large = publication["large_files"]
    require(
        isinstance(large, list) and 0 < len(large) <= 32 and len({x["path"] for x in large}) == len(large),
        "duplicate/invalid persistent checkpoint",
    )
    inventory = {}
    for row in large:
        name = row["path"]
        require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and name.startswith(("checkpoints/", "resume/")),
            "unsafe persistent checkpoint path",
        )
        require(
            set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 < row["size"] <= 32 * 1024**3
            and re.fullmatch(r"[a-f0-9]{64}", str(row["sha256"])),
            "invalid checkpoint identity",
        )
        inventory[name] = row
    require(sum(x["size"] for x in large) <= 128 * 1024**3, "checkpoint publication exceeds bound")
    for checkpoint in report["checkpoints"]:
        row = {k: checkpoint[k] for k in ("path", "size", "sha256")}
        require(
            row["path"] == f"checkpoints/step-{checkpoint['step']:08d}.pt", "checkpoint step path differs"
        )
        same(inventory.get(row["path"]), row, "checkpoint absent from persistent publication")
    return report, publication


def validate_dataset(report, files):
    """Bind the actual published full population to the reviewed fresh inventory."""
    from posttrain_circuits.experiments.protocols import student_focus_preparation as protocol

    splits = protocol.make_examples(protocol.DIFFICULTY)
    expected = protocol.dataset_manifest(splits)
    same(document(files["prepare-dataset-manifest.json"]), expected, "published training population differs")
    dataset_sha = sha(canonical(expected))
    same(report.get("dataset_sha256"), dataset_sha, "report training population digest differs")
    return dataset_sha


def validate_isolation(report, files, dataset_sha):
    """Cross-bind retained producer isolation evidence; do not rerun its 144k scan."""
    from posttrain_circuits.experiments.protocols import student_focus_preparation as protocol

    data = protocol.proposed_student_focus_preparation_protocol()["data"]
    isolation = document(files["prepare-data-isolation.json"])
    same(report.get("data_audit", {}).get("isolation"), isolation, "report isolation evidence differs")
    require(
        report.get("data_audit", {}).get("passed") is True
        and isolation.get("passed") is True
        and isolation.get("scientific_acceptance") is False
        and isolation.get("formal_generated_responses_or_scores_read") is False,
        "isolation flags differ",
    )
    same(isolation.get("dataset_manifest_sha256"), dataset_sha, "isolation population digest differs")
    same(isolation.get("counts"), data["isolation_population_counts"], "isolation population counts differ")
    same(
        isolation.get("excluded_view_counts"),
        data["isolation_excluded_view_counts"],
        "isolation view counts differ",
    )
    same(isolation.get("dimensions"), data["isolation_dimensions"], "isolation dimensions differ")
    for key, counts_key in (
        ("population_stream_sha256", "counts"),
        ("excluded_view_stream_sha256", "excluded_view_counts"),
    ):
        digests = isolation.get(key)
        require(
            isinstance(digests, dict)
            and set(digests) == set(isolation[counts_key])
            and all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in digests.values()),
            "isolation stream digests differ",
        )
    return dict(
        producer_evidence_sha256=sha(files["prepare-data-isolation.json"]),
        dataset_manifest_sha256=dataset_sha,
        producer_isolation_consistency_verified=True,
        historical_isolation_independently_recomputed=False,
    )


def training_population(tokenizer):
    """Reconstruct the fixed signed-pair row order and independently encode every row."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_focus_preparation as protocol
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    splits = protocol.make_examples(protocol.DIFFICULTY)
    bases = splits["student_fit"]
    rows = protocol.fit_views(bases)
    require(len(bases) == 256 and len(rows) == 2048, "full training population differs")
    names = tuple(protocol.FIT_VIEW_NAMES)
    require(len(names) == len(set(names)) == 8, "training view population differs")
    lookup = {(row.source_example_id, row.view): row for row in rows}
    require(len(lookup) == 2048, "duplicate scientific training row")
    pair_ids = set()
    for index in range(128):
        left, right = bases[2 * index : 2 * index + 2]
        window, slot = divmod(index, 4)
        structure = "branch" if slot < 3 else "chain" if window % 2 == 0 else "converging_dag"
        require(
            left.pair_group_id == right.pair_group_id
            and left.pair_group_id not in pair_ids
            and {left.label, right.label} == {0, 1}
            and left.metadata["structure"] == right.metadata["structure"] == structure,
            "three branch pairs and alternating nonbranch pair required per original window",
        )
        pair_ids.add(left.pair_group_id)
    expected = []
    for source in bases:
        order = names if source.label == 1 else names[1:] + names[:1]
        for name in order:
            row = lookup[(source.example_id, name)]
            require(
                row.example.label == source.label and row.example.pair_group_id == source.pair_group_id,
                "training row source identity differs",
            )
            expected.append(row)
    same(
        [asdict(row) for row in rows],
        [asdict(row) for row in expected],
        "original complete-pair row order differs",
    )
    require(len({canonical(asdict(row)) for row in rows}) == 2048, "duplicate full scientific row")
    config = {
        "prompt_protocol": dict(
            name="qwen3_non_thinking_v1",
            enable_thinking=False,
            chat_template_sha256=protocol.CHAT_TEMPLATE_SHA256,
        )
    }
    task, lengths = ProofGraphTask(), []
    for view in rows:
        row = protocol.encode_view(view, tokenizer, config)
        text = format_model_prompt(view.prompt, tokenizer, config).model_facing_prompt
        prefix = tokenizer.encode(text, add_special_tokens=False)
        response = [
            *tokenizer.encode(task.canonical_target(view.example), add_special_tokens=False),
            tokenizer.eos_token_id,
        ]
        require(
            len(prefix) > 0 and 0 < len(response) <= 256 and len(prefix) + len(response) <= 1536,
            "training token envelope differs",
        )
        same(row["input_ids"], prefix + response, "independent model-facing training tokens differ")
        same(row["labels"], [-100] * len(prefix) + response, "independent response loss mask differs")
        same(row["prefix_length"], len(prefix), "independent training prefix length differs")
        same(row["example_id"], view.example.example_id, "encoded training identity differs")
        lengths.append(len(prefix) + len(response))
    windows = [sum(lengths[start : start + 64]) for start in range(0, 2048, 64)]
    require(sum(windows) <= 2_000_000, "complete training population exceeds original token budget")
    return lengths, dict(
        fit_rows=2048,
        optimizer_windows=32,
        window_input_tokens=windows,
        total_input_tokens=sum(windows),
        token_budget=2_000_000,
    )


def validate_training_evidence(report, files, tokenizer):
    """Bind actual four/full32 optimizer windows to independently reconstructed rows."""
    require(report.get("mode") in ("preflight", "fit"), "unknown training evidence mode")
    fit = report["mode"] == "fit"
    steps = 32 if fit else 4
    lengths, expected = training_population(tokenizer)
    same(document(files["prepare-token-audit.json"]), expected, "full32 training token audit differs")
    same(
        report.get("data_audit", {}).get("token_envelope"), expected, "report training token envelope differs"
    )
    same(
        [
            report.get(k)
            for k in (
                "optimizer_steps",
                "consumed_fit_training_views",
                "world_size",
                "global_batch_size",
                "training_input_tokens",
            )
        ],
        [steps, steps * 64, 2, 64, sum(expected["window_input_tokens"][:steps])],
        "actual training report differs",
    )
    updates = lines(files["prepare-updates.jsonl"], steps)
    cumulative = 0
    for step, update in enumerate(updates, 1):
        require(type(update.get("step")) is int and update["step"] == step, "training update order differs")
        ranks = update.get("ranks")
        require(isinstance(ranks, list) and len(ranks) == 2, "training rank evidence missing")
        window_tokens = expected["window_input_tokens"][step - 1]
        cumulative += window_tokens
        for rank, row in enumerate(ranks):
            slots = list(range((step - 1) * 64 + rank, step * 64, 2))
            same([row.get("rank"), row.get("step")], [rank, step], "training rank/step differs")
            same(row.get("global_slots"), slots, "training slots differ")
            same(row.get("optimizer_calls"), [step - 1] * 7 + [step], "training accumulation differs")
            same(row.get("global_input_tokens"), window_tokens, "training global window tokens differ")
            same(
                row.get("token_budget"),
                dict(
                    accepted_optimizer_updates=step,
                    budget=2_000_000,
                    consumed=cumulative,
                    stop_reason=None,
                    unit="global_nonpadding_model_input_tokens_processed",
                ),
                "training budget cursor differs",
            )
            metric = row.get("metric", {})
            for key, value in dict(
                step=float(step),
                optimizer_updates=float(step),
                model_facing_input_tokens_processed=float(cumulative),
                model_facing_input_tokens_this_update=window_tokens,
                local_model_facing_input_tokens_this_update=sum(lengths[index] for index in slots),
                token_budget_consumed=cumulative,
                token_budget=2_000_000,
                token_budget_remaining=2_000_000 - cumulative,
            ).items():
                same(metric.get(key), value, "training metric differs: " + key)
    return dict(
        training_order_reconstructed=True,
        training_rows_reconstructed=2048,
        consumed_training_rows=steps * 64,
        optimizer_windows_replayed=steps,
        full_population_input_tokens=expected["total_input_tokens"],
        training_input_tokens=cumulative,
        checkpoint_cumulative_input_tokens={
            str(step): sum(expected["window_input_tokens"][:step]) for step in (STEPS if fit else (4,))
        },
        gpu_training_independently_recomputed=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from posttrain_circuits.experiments.protocols.student_focus_preparation import (
        resolve_student_focus_preparation_protocol,
    )

    plan = document(read(args.plan))
    metadata = ROOT / (".opd-git" if (ROOT / ".opd-git").is_dir() else ".git")
    binding = resolve_student_focus_preparation_protocol(ROOT, git_dir=metadata)
    verify_producer_head(plan, binding, metadata)
    raw = read(args.fetch_dir / "receipt.json")
    require(sha(raw) == args.receipt_sha256, "externally bound publication hash differs")
    publication = document(raw)
    files = {}
    for row in publication["files"]:
        name = row["path"]
        require(
            isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
            "unsafe fetched filename",
        )
        require(name not in files, "duplicate published file")
        files[name] = read(args.fetch_dir / name)
        require(sum(map(len, files.values())) <= MAX_TOTAL_BYTES, "fetched files exceed bound")
    report, publication = validate_publication(plan, raw, args.receipt_sha256, files, binding)
    fit = report["mode"] == "fit"
    steps = STEPS if fit else (4,)
    prompts = lines(files["prepare-dev-prompts.jsonl"], 1536) if fit else []
    records = {
        step: [
            lines(
                files[
                    f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl"
                    if fit
                    else f"prepare-preflight-records-rank-{rank}.jsonl"
                ],
                768 if fit else 4,
            )
            for rank in (0, 1)
        ]
        for step in steps
    }
    tokenizer = load_tokenizer()
    dataset_sha = validate_dataset(report, files)
    isolation_evidence = validate_isolation(report, files, dataset_sha)
    training_evidence = validate_training_evidence(report, files, tokenizer)
    result = replay(report, prompts, records, tokenizer)
    result["training_evidence"] = training_evidence
    result["isolation_evidence"] = isolation_evidence
    result["dataset_sha256"] = dataset_sha
    result.update(
        job_id=publication["job_id"],
        publication_sha256=args.receipt_sha256,
        plan_sha256=sha(canonical(plan)),
        result_sha256=sha(files["prepare-report.json"]),
        auditor_sha256=sha(Path(__file__).read_bytes()),
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        implementation_commit=binding.implementation_commit,
        acceptance_commit=binding.acceptance_commit,
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
