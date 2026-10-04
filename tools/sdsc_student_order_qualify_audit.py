#!/usr/bin/env python3
"""Independently replay the fixed prepared-student qualification responses.

Reconstructs every original validation/IID/transformation prompt and its exact
tokenization, parsing, verification and numerical gates. It does not rerun GPU
inference, approve G0 or select another preparation checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 48 * 1024**2
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
RAW_NAMES = ("qualification-report.json", "qualification-prompts.jsonl", "qualification-records.jsonl")
VALIDATION_SHA256 = "ceecd7470309dbb78351f8ecba850b98b6f35bf749aa89ac101498ba4ce921e8"
IID_SHA256 = "ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3"


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

    result = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(result, dict), "JSON object required")
    canonical(result)
    return result


def lines(raw, count):
    rows = raw.splitlines()
    require(len(rows) == count and all(row.strip() for row in rows), "raw population count differs")
    return [document(row) for row in rows]


def load_tokenizer():
    # This helper and its original tokenizer inputs are independently pinned by
    # the qualification protocol. No producer scoring code is imported here.
    spec = importlib.util.spec_from_file_location(
        "qualification_pinned_tokenizer", ROOT / "tools/sdsc_student_prepare_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tokenizer()


def population(tokenizer):
    """Rebuild the unscreened populations from original scientific primitives."""
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import build_anti_shortcut_suite
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.splits import build_split
    from posttrain_circuits.experiments.protocols.student_preparation import (
        CHAT_TEMPLATE_SHA256,
        DIFFICULTY,
    )
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    task = ProofGraphTask()
    validation = build_split(task, "validation", 128, 42, DIFFICULTY)
    iid = build_split(task, "iid_test", 128, 42, DIFFICULTY)
    require(
        sha(canonical([asdict(x) for x in validation])) == VALIDATION_SHA256, "validation population changed"
    )
    require(sha(canonical([asdict(x) for x in iid])) == IID_SHA256, "IID population changed")
    cases = build_anti_shortcut_suite(iid, seed=42, distractor_ood_count=32)
    cohort_rows = (
        [("validation_exposed", None, x.example_id, x, task.render(x)) for x in validation]
        + [("anti_shortcut_iid", None, x.example_id, x, task.render(x)) for x in iid]
        + [
            ("anti_shortcut_transformed", x.transformation, x.source_example_id, x.example, x.prompt)
            for x in cases
        ]
    )
    model_config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        }
    }
    prompts, examples = [], []
    for ordinal, (cohort, transformation, source, example, raw) in enumerate(cohort_rows):
        text = format_model_prompt(raw, tokenizer, model_config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        limit = 1536 if cohort == "validation_exposed" else 2244
        require(len(ids) + 256 <= limit, "qualification inference envelope changed")
        prompts.append(
            dict(
                ordinal=ordinal,
                cohort=cohort,
                transformation=transformation,
                source_example_id=source,
                example=asdict(example),
                raw_prompt=raw,
                prompt_text=text,
                prompt_ids=ids,
                raw_prompt_sha256=sha(raw.encode()),
                prompt_text_sha256=sha(text.encode()),
                prompt_token_sha256=sha(canonical(ids)),
            )
        )
        examples.append(example)
    return prompts, examples, iid, cases


def _wilson_lower(successes, count):
    p, z = successes / count, 1.959963984540054
    center = p + z * z / (2.0 * count)
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count))
    return max(0.0, (center - radius) / (1.0 + z * z / count))


def replay(report, prompts, records, tokenizer, *, candidate):
    """Recompute outcomes without producer scoring, summary or gate functions."""
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols.student_order_qualification import (
        validate_prepared_initial,
    )

    candidate = validate_prepared_initial(candidate, require_bound=True)
    checkpoint_sha = candidate["checkpoint_sha256"]
    same(report.get("prepared_initial"), candidate, "frozen candidate differs")

    require(
        report.get("schema") == "quest-sdsc-student-order-qualification-report-v1", "report schema differs"
    )
    require(report.get("execution_complete") is True, "qualification execution incomplete")
    require(report.get("parameters_unchanged") is True, "qualification parameters changed")
    require(all(report.get(key) is False for key in FLAGS), "producer overclaims formal acceptance")
    require(report.get("prepared_checkpoint_sha256") == checkpoint_sha, "prepared checkpoint differs")
    loading = report.get("loading", {})
    same(
        loading.get("exact_saved_master_reload"),
        dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True),
        "exact master reload evidence differs",
    )
    require(
        loading.get("prepared_checkpoint_sha256") == checkpoint_sha
        and loading.get("master_model_state_sha256") == candidate["master_model_state_sha256"]
        and loading.get("forward_parameter_dtype") == "torch.bfloat16"
        and loading.get("all_parameters_frozen") is True
        and loading.get("gradient_checkpointing") is False
        and loading.get("use_cache") is False
        and re.fullmatch(r"[0-9a-f]{64}", str(loading.get("forward_model_state_sha256", ""))),
        "selected master/native-BF16 inference evidence differs",
    )
    same(
        report.get("generation"),
        dict(
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
            explicit_attention_mask=False,
            explicit_autocast=False,
            precision="native_bfloat16",
            truncation=False,
            seed=42,
            validation_max_model_input_length=1536,
            anti_shortcut_max_model_input_length=2244,
        ),
        "original generation contract differs",
    )
    expected, examples, iid, cases = population(tokenizer)
    require(len(prompts) == len(records) == len(expected) == 896, "complete 896 population required")
    task, results, stops = ProofGraphTask(), [], {"eos": 0, "length": 0}
    vocabulary_size = len(tokenizer)
    for prompt, record, original, example in zip(prompts, records, expected, examples, strict=True):
        same(prompt, original, "original prompt population/tokenization differs")
        for key in (
            "ordinal",
            "cohort",
            "transformation",
            "source_example_id",
            "prompt_ids",
            "prompt_text_sha256",
            "prompt_token_sha256",
        ):
            same(record.get(key), original[key], "response identity/prompt differs: " + key)
        require(record.get("example_id") == example.example_id, "response example differs")
        require(record.get("prepared_checkpoint_sha256") == checkpoint_sha, "response checkpoint differs")
        if "raw_prompt_sha256" in record:
            require(record["raw_prompt_sha256"] == original["raw_prompt_sha256"], "raw prompt hash differs")
        tokens = record.get("response_ids")
        require(
            isinstance(tokens, list)
            and 0 < len(tokens) <= 256
            and all(type(token) is int and 0 <= token < vocabulary_size for token in tokens),
            "invalid response token IDs",
        )
        require(tokenizer.eos_token_id not in tokens[:-1], "tokens after EOS")
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        require(record.get("response_text") == response, "response decoding differs")
        require(
            record.get("response_tokens") == len(tokens) and record.get("max_new_tokens") == 256,
            "response generation budget differs",
        )
        stop = "eos" if tokens[-1] == tokenizer.eos_token_id else "length" if len(tokens) == 256 else "other"
        require(stop != "other" and record.get("stop_reason") == stop, "generation termination differs")
        stops[stop] += 1
        parsed = task.parse_response(response)
        verified = task.verify(example, parsed)
        same(record.get("parsed_trace"), asdict(parsed), "parser trace differs")
        same(record.get("verification"), asdict(verified), "verifier trace differs")
        require(record.get("parse_error") == parsed.error_code, "parse error differs")
        for key, value in (("pair_group_id", example.pair_group_id), ("expected_label", example.label)):
            if key in record:
                same(record[key], value, "example metadata differs")
        tags = re.findall(r"<answer>\s*([01])\s*</answer>", response)
        for key, value in (
            ("diagnostic_answer_tag_present", len(tags) == 1),
            ("diagnostic_answer_tag_correct", len(tags) == 1 and int(tags[0]) == example.label),
        ):
            if key in record:
                require(record[key] is value, "diagnostic answer tags differ")
        results.append(verified)
    values = results[:128]
    answer = sum(x.answer_correct for x in values)
    proof = sum(x.reward == 1.0 for x in values)
    valid = sum(x.parse_valid for x in values)
    validation = dict(
        num_examples=128,
        answer_correct=answer,
        proof_correct=proof,
        format_valid=valid,
        answer_accuracy=answer / 128,
        exact_proof_accuracy=proof / 128,
        format_validity=valid / 128,
        passed=answer >= 13,
    )
    same(report.get("validation"), validation, "validation summary differs")
    iid_count = sum(x.reward == 1.0 for x in results[128:256])
    counts = {name: 0 for name in TRANSFORMATIONS}
    for result, case in zip(results[256:], cases, strict=True):
        counts[case.transformation] += int(result.reward == 1.0)
    accuracies = {name: count / 128 for name, count in counts.items()}
    iid_accuracy = iid_count / 128
    transformed = sum(accuracies.values()) / 5
    gap = iid_accuracy - transformed
    capable = iid_accuracy >= 0.10 and transformed >= 0.08 and all(v >= 0.05 for v in accuracies.values())
    anti = dict(
        format_version=2,
        accuracy_metric="exact_verifier_reward",
        model_checkpoint_hash=checkpoint_sha,
        iid_example_count=128,
        transformed_case_count=640,
        iid_accuracy=iid_accuracy,
        transformed_accuracy=transformed,
        transformation_accuracy=accuracies,
        transformation_accuracy_wilson95_lower={
            name: _wilson_lower(count, 128) for name, count in counts.items()
        },
        shortcut_gap=gap,
        max_shortcut_gap=0.05,
        minimum_iid_accuracy=0.10,
        minimum_transformed_accuracy=0.08,
        minimum_per_transformation_accuracy=0.05,
        capability_passed=capable,
        passed=capable and gap <= 0.05,
        dataset_hash=sha(canonical([asdict(x) for x in iid])),
        suite_hash=sha(
            canonical(
                [
                    dict(
                        source_example_id=x.source_example_id,
                        transformation=x.transformation,
                        semantic_hash=x.semantic_hash,
                        prompt=x.prompt,
                    )
                    for x in cases
                ]
            )
        ),
    )
    observed = report.get("anti_shortcut")
    require(isinstance(observed, dict), "anti-shortcut report missing")
    for key, value in anti.items():
        same(observed.get(key), value, "anti-shortcut summary differs: " + key)
    from posttrain_circuits.artifacts.compatibility import scientific_compatibility_fields

    for key, value in scientific_compatibility_fields("qwen3_v2").items():
        same(observed.get(key), value, "anti-shortcut compatibility differs: " + key)
    source = report.get("scientific_binding")
    require(isinstance(source, dict), "scientific source binding missing")
    require(
        re.fullmatch(r"[0-9a-f]{40}", str(source.get("head", "")))
        and re.fullmatch(r"[0-9a-f]{40}", str(source.get("prereg_commit", ""))),
        "scientific source commit missing",
    )
    require(
        observed.get("code_commit") == source["head"]
        and observed.get("prereg_commit") == source["prereg_commit"],
        "anti-shortcut source binding differs",
    )
    require(
        observed.get("sha256") == sha(canonical({k: v for k, v in observed.items() if k != "sha256"})),
        "anti-shortcut report hash differs",
    )
    passed = validation["passed"] and anti["passed"]
    require(
        report.get("qualification_passed") is passed and report.get("passed") is passed,
        "qualification outcome differs",
    )
    return dict(
        raw_replay_passed=True,
        prompts_replayed=896,
        responses_replayed=896,
        validation=validation,
        anti_shortcut=anti,
        qualification_passed=passed,
        prepared_checkpoint_sha256=checkpoint_sha,
        termination_counts=stops,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
        **dict.fromkeys(FLAGS, False),
    )


def prereg_commit_at(root, metadata, head, binding):
    """Resolve producer-time provenance, allowing later unrelated local commits."""
    from posttrain_circuits.artifacts.git_provenance import _git_environment

    require(re.fullmatch(r"[0-9a-f]{40}", str(head)), "invalid producer source head")

    def git(*args):
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(metadata),
                "--work-tree",
                str(root),
                "-C",
                str(root),
                *args,
            ],
            capture_output=True,
            check=True,
            timeout=30,
            env=_git_environment(),
        )
        return result.stdout.decode("utf-8", errors="strict").strip()

    require(
        git("merge-base", binding.acceptance_commit, head) == binding.acceptance_commit,
        "producer source predates accepted qualification",
    )
    require(
        git("merge-base", head, binding.head) == head, "producer source is outside current accepted lineage"
    )
    result = git("log", "-n", "1", "--format=%H", head, "--", "prereg/qwen3_v2.yaml")
    require(re.fullmatch(r"[0-9a-f]{40}", result), "original prereg commit missing")
    return result


def validate_fit_audit_record(record, candidate, parent):
    """Independently bind the reviewed parent raw-audit artifact, not producer reducers."""
    require(
        isinstance(record, dict) and set(record) == {"sha256", "size", "base64"},
        "parent audit record fields differ",
    )
    require(
        type(record["size"]) is int
        and 0 < record["size"] <= 65536
        and record["sha256"] == candidate["independent_raw_audit_sha256"]
        and isinstance(record["base64"], str)
        and len(record["base64"]) <= 4 * ((65536 + 2) // 3),
        "parent audit envelope differs",
    )
    raw = base64.b64decode(record["base64"], validate=True)
    require(len(raw) == record["size"] and sha(raw) == record["sha256"], "parent audit bytes differ")
    audit = document(raw)
    require(raw == canonical(audit) + b"\n", "parent audit encoding differs")
    expected = dict(
        mode="fit",
        job_id=candidate["fit_job_id"],
        plan_sha256=candidate["fit_plan_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        auditor_sha256=parent["parent_auditor_sha256"],
        protocol_sha256=parent["parent_protocol_core_sha256"],
        protocol_artifact_sha256=parent["parent_student_protocol_sha256"],
        implementation_commit=parent["parent_implementation_commit"],
        acceptance_commit=parent["parent_acceptance_commit"],
        raw_replay_passed=True,
        prompts_replayed=1536,
        responses_replayed=18432,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
    )
    for key, value in expected.items():
        same(audit.get(key), value, "parent audit binding differs: " + key)
    require(all(audit.get(key) is False for key in FLAGS), "parent audit overclaims acceptance")
    selected = audit.get("selected_checkpoint")
    require(
        isinstance(selected, dict)
        and type(selected.get("step")) is int
        and selected["step"] == candidate["checkpoint_step"]
        and selected.get("checkpoint_sha256") == candidate["checkpoint_sha256"],
        "parent selection differs",
    )
    return audit


RECOVERY_NAMES = ("execution_plan", "verified_status", "execution_publication", "science_publication")
MAX_RECOVERY_BYTES = 4 * 1024**2


def decode_recovery_records(records, candidate, fit):
    """Offline cross-binding; fresh accounting/CUDA verification remains controller-owned."""
    from posttrain_circuits.experiments.protocols.student_order_qualification import (
        validate_parent_recovery_evidence,
    )

    require(
        isinstance(records, dict) and set(records) == set(RECOVERY_NAMES), "recovery record inventory differs"
    )
    raw, documents, total = {}, {}, 0
    for name in RECOVERY_NAMES:
        record = records[name]
        require(
            isinstance(record, dict) and set(record) == {"size", "sha256", "base64"},
            "recovery record fields differ",
        )
        require(
            type(record["size"]) is int
            and 0 < record["size"] <= MAX_RECOVERY_BYTES
            and isinstance(record["sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", record["sha256"])
            and isinstance(record["base64"], str)
            and len(record["base64"]) <= 4 * ((MAX_RECOVERY_BYTES + 2) // 3),
            "recovery record envelope differs",
        )
        total += record["size"]
        require(total <= MAX_RECOVERY_BYTES, "recovery bundle exceeds total bound")
        raw[name] = base64.b64decode(record["base64"], validate=True)
        require(
            len(raw[name]) == record["size"] and sha(raw[name]) == record["sha256"],
            "recovery record bytes differ",
        )
        documents[name] = document(raw[name])
        require(canonical(documents[name]) == raw[name], "recovery encoding differs")
    same(documents["execution_plan"].get("science_plan"), fit, "recovery science plan differs")
    validate_parent_recovery_evidence(
        candidate,
        execution_plan=documents["execution_plan"],
        status=documents["verified_status"],
        execution_publication_raw=raw["execution_publication"],
        science_publication_raw=raw["science_publication"],
    )
    return raw


def validate_publication(plan, publication_raw, receipt_sha256, files, binding, prereg_commit):
    """Bind externally supplied publication identity before accepting raw bytes."""
    require(sha(publication_raw) == receipt_sha256, "externally bound publication hash differs")
    publication = document(publication_raw)
    plan_sha = sha(canonical(plan))
    task = "qwen3-v2-student-order-qualification-v1"
    require(
        plan.get("schema") == "quest-sdsc-student-order-qualification-plan-v1" and plan.get("task") == task,
        "qualification plan identity differs",
    )
    for key, value in dict(
        task=task,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=plan_sha,
        code_sha256=plan["code_sha256"],
    ).items():
        same(publication.get(key), value, "publication identity differs: " + key)
    require(re.fullmatch(r"[0-9]+", str(publication.get("job_id", ""))), "publication job missing")
    require(publication.get("persistent_read_back_verified") is True, "persistent publication not verified")
    require(
        all(plan.get(key) is False and publication.get(key) is False for key in FLAGS),
        "plan/publication overclaims formal acceptance",
    )
    same(plan["protocol"]["head"], plan["provenance"]["head"], "plan scientific head differs")
    same(
        plan["protocol"]["science_file_sha256"],
        binding.science_file_sha256,
        "reviewed source binding differs",
    )
    for key in ("implementation_commit", "acceptance_commit", "protocol_sha256", "artifact_sha256"):
        same(plan["protocol"][key], getattr(binding, key), "reviewed protocol lineage differs: " + key)
    from posttrain_circuits.experiments.protocols.student_order_qualification import (
        validate_prepared_initial,
    )

    candidate = validate_prepared_initial(binding.payload["prepared_initial"], require_bound=True)
    same(
        plan["prepared_checkpoint"],
        {
            "path": candidate["checkpoint_path"],
            "size": candidate["checkpoint_size"],
            "sha256": candidate["checkpoint_sha256"],
        },
        "plan selected checkpoint differs",
    )
    fit = plan["fit"]
    require(
        isinstance(fit, dict)
        and set(fit)
        == {"plan", "plan_sha256", "job_id", "publication_sha256", "report_sha256", "audit", "recovery"},
        "parent fit fields differ",
    )
    recovery = decode_recovery_records(fit["recovery"], candidate, fit["plan"])
    for name, key in (
        ("job_id", "fit_job_id"),
        ("plan_sha256", "fit_plan_sha256"),
        ("publication_sha256", "fit_publication_sha256"),
        ("report_sha256", "fit_report_sha256"),
    ):
        same(fit[name], candidate[key], "plan parent candidate differs: " + name)
    parent = binding.payload["preserved_base"]
    parent_audit = validate_fit_audit_record(fit["audit"], candidate, parent)
    same(sha(canonical(fit["plan"])), candidate["fit_plan_sha256"], "embedded parent plan differs")
    same(fit["plan"]["mode"], "fit", "parent is not fit")
    same(fit["plan"]["task"], "qwen3-v2-student-order-preparation-v4", "parent task differs")
    for key, field in (
        ("protocol_sha256", "parent_protocol_core_sha256"),
        ("artifact_sha256", "parent_student_protocol_sha256"),
        ("implementation_commit", "parent_implementation_commit"),
        ("acceptance_commit", "parent_acceptance_commit"),
    ):
        same(fit["plan"]["protocol"][key], parent[field], "parent protocol differs: " + key)
    rows = publication.get("files")
    require(
        isinstance(rows, list) and len({x["path"] for x in rows}) == len(rows), "duplicate published file"
    )
    require(set(RAW_NAMES) <= {x["path"] for x in rows}, "required qualification files missing")
    for item in rows:
        name = item["path"]
        require(isinstance(name, str) and Path(name).name == name, "unsafe fetched filename")
        require(
            set(item) == {"path", "size", "sha256"}
            and type(item["size"]) is int
            and 0 < item["size"] <= MAX_BYTES
            and re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])),
            "invalid published file identity",
        )
        raw = files[name]
        require(len(raw) == item["size"] and sha(raw) == item["sha256"], "published file hash differs")
    require(set(files) == {x["path"] for x in rows}, "unbound fetched files")
    require(sum(x["size"] for x in rows) <= MAX_TOTAL_BYTES, "qualification publication exceeds total bound")
    report = document(files[RAW_NAMES[0]])
    same(report.get("prepared_initial"), candidate, "report candidate differs from accepted protocol")
    same(
        report.get("preparation_protocol_sha256"),
        parent["parent_protocol_core_sha256"],
        "report parent core differs",
    )
    same(
        report.get("preparation_protocol_artifact_sha256"),
        parent["parent_student_protocol_sha256"],
        "report parent artifact differs",
    )
    evidence = report.get("fit_evidence", {})
    same(
        evidence.get("recovery"),
        {name: dict(size=len(raw), sha256=sha(raw)) for name, raw in recovery.items()},
        "report recovery evidence differs",
    )
    for name, key in (
        ("fit_job_id", "fit_job_id"),
        ("publication_sha256", "fit_publication_sha256"),
        ("report_sha256", "fit_report_sha256"),
        ("independent_audit_sha256", "independent_raw_audit_sha256"),
        ("master_model_state_sha256", "master_model_state_sha256"),
        ("checkpoint_path", "checkpoint_path"),
        ("checkpoint_size", "checkpoint_size"),
    ):
        same(evidence.get(name), candidate[key], "report fit evidence differs: " + name)
    same(
        evidence.get("selected_checkpoint"),
        parent_audit["selected_checkpoint"],
        "report parent selection differs",
    )
    same(
        report.get("prepared_checkpoint_sha256"), candidate["checkpoint_sha256"], "report checkpoint differs"
    )
    for key, value in {
        "task": task,
        "job_id": publication["job_id"],
        "run_id": plan["run_id"],
        "plan_sha256": plan_sha,
        "source_code_sha256": plan["code_sha256"],
        "protocol_sha256": binding.protocol_sha256,
        "protocol_artifact_sha256": binding.artifact_sha256,
    }.items():
        same(report.get(key), value, "report binding differs: " + key)
    require(all(report.get(key) is False for key in FLAGS), "report overclaims formal acceptance")
    source = report.get("scientific_binding")
    require(isinstance(source, dict), "report scientific source binding missing")
    for key in ("head", "implementation_commit", "acceptance_commit", "science_file_sha256"):
        same(source.get(key), plan["protocol"][key], "report scientific source differs: " + key)
    same(source.get("prereg_commit"), prereg_commit, "original prereg commit differs")
    return report, publication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from posttrain_circuits.experiments.protocols.student_order_qualification import (
        resolve_student_order_qualification_protocol,
    )

    metadata = ROOT / (".opd-git" if (ROOT / ".opd-git").is_dir() else ".git")
    binding = resolve_student_order_qualification_protocol(ROOT, git_dir=metadata)
    plan = document(read(args.plan))
    raw = read(args.fetch_dir / "receipt.json")
    require(sha(raw) == args.receipt_sha256, "externally bound publication hash differs")
    publication = document(raw)
    files = {}
    for item in publication["files"]:
        name = item["path"]
        require(isinstance(name, str) and Path(name).name == name, "unsafe fetched filename")
        require(name not in files, "duplicate fetched filename")
        files[name] = read(args.fetch_dir / name)
        require(sum(map(len, files.values())) <= MAX_TOTAL_BYTES, "fetched files exceed total bound")
    prereg = prereg_commit_at(ROOT, metadata, plan["protocol"]["head"], binding)
    report, publication = validate_publication(plan, raw, args.receipt_sha256, files, binding, prereg)
    result = replay(
        report,
        lines(files[RAW_NAMES[1]], 896),
        lines(files[RAW_NAMES[2]], 896),
        load_tokenizer(),
        candidate=binding.payload["prepared_initial"],
    )
    result.update(
        job_id=publication["job_id"],
        publication_sha256=args.receipt_sha256,
        plan_sha256=sha(canonical(plan)),
        auditor_sha256=sha(Path(__file__).read_bytes()),
        result_sha256=sha(files[RAW_NAMES[0]]),
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        implementation_commit=binding.implementation_commit,
        acceptance_commit=binding.acceptance_commit,
        prepared_initial=binding.payload["prepared_initial"],
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
