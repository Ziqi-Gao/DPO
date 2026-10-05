#!/usr/bin/env python3
"""Independent original-tokenizer/verifier replay of the fixed naming-cue probe.

Reconstructs all512 prompts and1024 outputs, model/condition/map/rank identity,
paired readouts and retained isolation bytes. No GPU, new score, checkpoint
choice, qualification or promotion. Failed-prefix diagnosis never repairs scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "treatment")
CONDITIONS = ("preserve", "break_pairs", "break_paths", "break_both")
STRUCTURES = ("branch", "chain")
METRICS = ("proof_correct", "answer_correct", "format_valid", "length")
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
TASK = "qwen3-v2-student-name-cue-probe-v1"
MAX_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2


def sibling(name):
    spec = importlib.util.spec_from_file_location("_cue_audit_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = sibling("sdsc_student_focus_lr_probe_audit")
require, canonical, sha, same = frozen.require, frozen.canonical, frozen.sha, frozen.same
read, document, lines = frozen.read, frozen.document, frozen.lines
load_tokenizer, verify_producer_head = frozen.load_tokenizer, frozen.verify_producer_head


def population(tokenizer):
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    splits = p.make_examples()
    views = p.development_views(splits["student_dev"])
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    rows = [p.encode_view(v, tokenizer, config) for v in views]
    token_audit = p.validate_encoded_views(rows, views)
    prompts = []
    for i, (v, row) in enumerate(zip(views, rows, strict=True)):
        text = format_model_prompt(v.prompt, tokenizer, config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        same(ids, row["input_ids"][: row["prefix_length"]], "reconstructed prefix differs")
        require(
            len(ids) + 256 <= 2454 and row["response_length"] <= 256, "reconstructed token envelope differs"
        )
        prompts.append(
            dict(
                ordinal=i,
                source_example_id=v.source_example_id,
                view=v.view,
                block=v.block,
                condition=v.condition,
                example=asdict(v.example),
                prompt_text=text,
                prompt_ids=ids,
                canonical_response_tokens=row["response_length"],
            )
        )
    return prompts, [v.example for v in views], token_audit, p.dataset_manifest(splits)


def reduce_independently(records):
    """Source-paired factorial counts; independent of worker reduction code."""
    require(len(records) == 512, "complete model records required")
    index = {}
    for r in records:
        key = (r["block"], r["condition"], r["structure"], r["source_example_id"])
        require(key not in index, "duplicate scientific response")
        index[key] = r

    def bits(r):
        v = r["verification"]
        return dict(
            proof_correct=int(v["reward"] == 1),
            answer_correct=int(v["answer_correct"]),
            format_valid=int(v["parse_valid"]),
            length=int(r["stop_reason"] == "length"),
        )

    cells, contrasts = {}, {}
    comparisons = {
        "pair_cue_paths_present": ("preserve", "break_pairs"),
        "pair_cue_paths_broken": ("break_paths", "break_both"),
        "path_cue_pairs_present": ("preserve", "break_paths"),
        "path_cue_pairs_broken": ("break_pairs", "break_both"),
    }
    for block in (0, 1):
        bs = str(block)
        cells[bs] = {}
        contrasts[bs] = {}
        for condition in CONDITIONS:
            cells[bs][condition] = {}
            for structure in STRUCTURES:
                rows = [
                    bits(r) for (b, c, s, _), r in index.items() if (b, c, s) == (block, condition, structure)
                ]
                require(
                    len(rows) == (48 if structure == "branch" else 16), "condition structure count differs"
                )
                cells[bs][condition][structure] = {
                    "num_examples": len(rows),
                    **{m: sum(r[m] for r in rows) for m in METRICS},
                }
        for structure in STRUCTURES:
            ids = sorted(k[3] for k in index if k[:3] == (block, "preserve", structure))
            panel = {}
            for name, (left, right) in comparisons.items():
                panel[name] = {}
                for metric in METRICS:
                    pairs = Counter(
                        (
                            bits(index[block, left, structure, i])[metric],
                            bits(index[block, right, structure, i])[metric],
                        )
                        for i in ids
                    )
                    panel[name][metric] = dict(
                        present_condition=left,
                        absent_condition=right,
                        num_examples=len(ids),
                        present_only=pairs[1, 0],
                        absent_only=pairs[0, 1],
                        both=pairs[1, 1],
                        neither=pairs[0, 0],
                        difference_numerator=pairs[1, 0] - pairs[0, 1],
                        denominator=len(ids),
                    )
            panel["interaction"] = {
                m: dict(
                    numerator=sum(
                        coefficient * cells[bs][condition][structure][m]
                        for coefficient, condition in zip((1, -1, -1, 1), CONDITIONS, strict=True)
                    ),
                    denominator=len(ids),
                    definition="preserve-break_pairs-break_paths+break_both",
                )
                for m in METRICS
            }
            contrasts[bs][structure] = panel
    return dict(cells=cells, contrasts=contrasts)


def prefix_diagnosis(example, record, parsed, verified):
    """Explain first invalid step; synthetic closure never substitutes a score."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    if verified.reward == 1:
        return None
    synthetic = False
    task = ProofGraphTask()
    if (
        not parsed.parse_valid
        and record["stop_reason"] == "length"
        and record["response_text"].startswith("<proof>\n")
    ):
        accepted = []
        candidate = None
        for line in record["response_text"].splitlines()[1:]:
            trial = task.parse_response(
                "<proof>\n" + "\n".join([*accepted, line]) + "\n</proof>\n<answer>0</answer>"
            )
            if not trial.parse_valid:
                break
            accepted.append(line)
            candidate = trial
        if candidate is not None:
            parsed = candidate
            verified = task.verify(example, parsed)
            synthetic = True
    first = next((i for i, z in enumerate(verified.step_results) if not z.valid), None)
    result = dict(
        original_failure=record["verification"]["error_code"],
        synthetic_closure_used=synthetic,
        original_score_unchanged=True,
        first_invalid_step=None,
        valid_prefix_steps=first,
        error_category=record["verification"]["error_code"],
    )
    if first is None:
        return result
    known = dict(example.facts)
    for step in parsed.steps[:first]:
        known[step.step_id] = step.conclusion
    step = parsed.steps[first]
    rule = example.rules.get(step.rule_id)
    result["first_invalid_step"] = step.step_id
    result["error_category"] = verified.step_results[first].error_code
    if rule is not None:
        cited = [known[x] for x in step.citations if x in known]
        missing = list((Counter(rule.antecedents) - Counter(cited)).elements())
        unavailable = [x for x in missing if x not in known.values()]
        if unavailable:
            result["error_category"] = (
                "branch_required_premise_not_established"
                if example.metadata["structure"] == "branch" and len(rule.antecedents) > 1
                else "other_required_premise_not_established"
            )
        elif missing:
            result["error_category"] = "required_premise_available_but_not_cited"
    return result


def replay(report, prompts, arm_records, tokenizer):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    require(
        report.get("schema") == "quest-sdsc-student-name-cue-probe-report-v1"
        and report.get("task") == TASK
        and report.get("mode") == "probe",
        "report identity differs",
    )
    for k, v in dict(
        passed=True,
        execution_complete=True,
        diagnostic_complete=True,
        preparation_complete=False,
        inference_only=True,
        optimizer_steps=0,
        training_input_tokens=0,
        world_size=2,
        base_examples=64,
        views_per_model=512,
        total_responses=1024,
        selected_checkpoint=None,
        **dict.fromkeys(FLAGS, False),
    ).items():
        same(report.get(k), v, "report scope differs: " + k)
    expected, examples, token_audit, dataset = population(tokenizer)
    same(prompts, expected, "prompt population differs")
    same(report["dataset_sha256"], sha(canonical(dataset)), "dataset identity differs")
    same(report["data_audit"]["token_audit"], token_audit, "producer token audit differs")
    require(
        set(arm_records) == set(ARMS) and set(report["models"]) == set(ARMS), "fixed model inventory differs"
    )
    results = {}
    diagnostics = {}
    task = ProofGraphTask()
    vocab = len(tokenizer)
    keys = {
        "example_id",
        "pair_group_id",
        "expected_label",
        "prompt_ids",
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
        "arm",
        "source_job_id",
        "source_step",
        "learning_rate",
        "checkpoint_sha256",
        "ordinal",
        "rank",
        "cohort",
        "source_example_id",
        "view",
        "block",
        "condition",
        "structure",
        "parsed_trace",
    }
    for arm in ARMS:
        d = p.SOURCE_CHECKPOINTS[arm]
        records = []
        detail = []
        for k, v in dict(
            arm=arm,
            source_job_id=d["job_id"],
            source_step=32,
            learning_rate=d["learning_rate"],
            checkpoint_sha256=d["sha256"],
            checkpoint_size=d["size"],
            model_state_sha256=d["model_state_sha256"],
        ).items():
            same(report["models"][arm].get(k), v, "model source differs")
        require(len(arm_records[arm]) == 2, "two rank shards required")
        for rank, rows in enumerate(arm_records[arm]):
            ordinals = [i for i in range(512) if (i // 16 + (i // 4) % 2 + i % 4) % 2 == rank]
            same(p.generation_ordinals(rank), ordinals, "frozen rank mapping differs")
            require(len(rows) == len(ordinals) == 256, "rank row coverage differs")
            for r, i in zip(rows, ordinals, strict=True):
                require(set(r) == keys, "raw response fields differ")
                prompt = expected[i]
                e = examples[i]
                wanted = dict(
                    arm=arm,
                    source_job_id=d["job_id"],
                    source_step=32,
                    learning_rate=d["learning_rate"],
                    checkpoint_sha256=d["sha256"],
                    ordinal=i,
                    rank=rank,
                    cohort="student_name_cue_probe_dev",
                    source_example_id=prompt["source_example_id"],
                    view=prompt["view"],
                    block=prompt["block"],
                    condition=prompt["condition"],
                    structure=e.metadata["structure"],
                    example_id=e.example_id,
                    pair_group_id=e.pair_group_id,
                    expected_label=e.label,
                    prompt_ids=prompt["prompt_ids"],
                    prompt_token_sha256=sha(canonical(prompt["prompt_ids"])),
                    prompt_text_sha256=sha(prompt["prompt_text"].encode()),
                    max_new_tokens=256,
                )
                same({k: r[k] for k in wanted}, wanted, "response model/source/view/rank identity differs")
                tokens = r["response_ids"]
                require(
                    isinstance(tokens, list)
                    and 0 < len(tokens) <= 256
                    and all(type(t) is int and 0 <= t < vocab for t in tokens)
                    and tokenizer.eos_token_id not in tokens[:-1],
                    "invalid generated tokens",
                )
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                same(r["response_tokens"], len(tokens), "response token length differs")
                same(r["response_text"], text, "response decode differs")
                stop = (
                    "eos"
                    if tokens[-1] == tokenizer.eos_token_id
                    else "length"
                    if len(tokens) == 256
                    else "other"
                )
                require(stop != "other" and stop == r["stop_reason"], "generation termination differs")
                parsed = task.parse_response(text)
                verified = task.verify(e, parsed)
                same(r["parsed_trace"], asdict(parsed), "original parser differs")
                same(r["verification"], asdict(verified), "original verifier differs")
                same(r["parse_error"], parsed.error_code, "parse error differs")
                tags = re.findall(r"<answer>\s*([01])\s*</answer>", text)
                require(
                    r["diagnostic_answer_tag_present"] is (len(tags) == 1)
                    and r["diagnostic_answer_tag_correct"] is (len(tags) == 1 and int(tags[0]) == e.label),
                    "answer-tag observation differs",
                )
                records.append(r)
                why = prefix_diagnosis(e, r, parsed, verified)
                if why:
                    detail.append(
                        dict(
                            ordinal=i,
                            block=prompt["block"],
                            condition=prompt["condition"],
                            structure=e.metadata["structure"],
                            **why,
                        )
                    )
        readouts = reduce_independently(records)
        same(report["models"][arm]["readouts"], readouts, "independent factorial reduction differs")
        results[arm] = readouts
        diagnostics[arm] = detail
    return dict(
        schema="quest-sdsc-student-name-cue-probe-raw-audit-v1",
        raw_replay_passed=True,
        diagnostic_complete=True,
        prompts_replayed=512,
        responses_replayed=1024,
        models=results,
        interpretation_scope=(
            "Pair/path factors apply only to branches; chain cells are generic naming-permutation "
            "negative controls. Signed pairs and repeated maps are not independent examples. "
            "Chain cross-map prompts intentionally include duplicates; compare within each map. "
            "Floor or ceiling outcomes cannot establish absence of cue reliance. "
            "All fixed models/maps remain separate; no efficacy or acceptance gate is applied."
        ),
        failed_prefix_diagnosis=diagnostics,
        failed_prefix_scope=(
            "posthoc structural diagnosis; synthetic closing tags and constant answer0 "
            "only for failed capped prefixes; original verifier scores unchanged"
        ),
        token_audit=token_audit,
        selected_checkpoint=None,
        preparation_complete=False,
        execution_acceptance_claim=False,
        gpu_numerics_independently_recomputed=False,
        **dict.fromkeys(FLAGS, False),
    )


def validate_dataset_isolation(report, files):
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    dataset = p.dataset_manifest(p.make_examples())
    same(document(files["name-cue-dataset-manifest.json"]), dataset, "retained dataset differs")
    same(sha(canonical(dataset)), report["dataset_sha256"], "report dataset differs")
    raw = files["name-cue-data-isolation.json"]
    value = document(raw)
    same(report["data_audit"]["isolation"], value, "retained isolation differs")
    require(
        report["data_audit"]["passed"] is True
        and value["passed"] is True
        and value["scientific_acceptance"] is False
        and value["formal_generated_responses_or_scores_read"] is False,
        "isolation scope differs",
    )
    same(value["dataset_manifest_sha256"], report["dataset_sha256"], "isolation dataset binding differs")
    data = p.proposed_student_name_cue_probe_protocol()["data"]
    for key, expected in [
        ("counts", data["isolation_population_counts"]),
        ("excluded_view_counts", data["isolation_excluded_view_counts"]),
        ("dimensions", data["isolation_dimensions"]),
    ]:
        same(value[key], expected, "isolation inventory differs")
    for key, inventory in [
        ("population_stream_sha256", value["counts"]),
        ("excluded_view_stream_sha256", value["excluded_view_counts"]),
    ]:
        require(
            set(value[key]) == set(inventory)
            and all(isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h) for h in value[key].values()),
            "isolation stream hashes differ",
        )
    same(
        document(files["name-cue-token-audit.json"]),
        report["data_audit"]["token_audit"],
        "retained token audit differs",
    )
    return dict(
        producer_isolation_consistency_verified=True,
        historical_isolation_independently_recomputed=False,
        producer_evidence_sha256=sha(raw),
        dataset_manifest_sha256=report["dataset_sha256"],
    )


def validate_publication(plan, publication_raw, receipt_sha256, files, binding):
    require(sha(publication_raw) == receipt_sha256, "externally bound publication hash differs")
    publication = document(publication_raw)
    require(
        plan.get("schema") == "quest-sdsc-student-name-cue-probe-plan-v1"
        and plan.get("task") == TASK
        and plan.get("mode") == "probe",
        "plan identity differs",
    )
    for k, v in dict(
        schema="quest-sdsc-student-name-cue-probe-publication-v1",
        task=TASK,
        mode="probe",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=True,
        stage_complete=True,
        diagnostic_complete=True,
        preparation_complete=False,
        selected_checkpoint=None,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=[],
        **dict.fromkeys(FLAGS, False),
    ).items():
        same(publication.get(k), v, "publication scope/binding differs: " + k)
    require(
        isinstance(publication.get("job_id"), str) and re.fullmatch("[1-9][0-9]*", publication["job_id"]),
        "publication job missing",
    )
    require(all(plan.get(k) is False for k in FLAGS), "plan overclaims acceptance")
    same(plan["protocol"]["science_file_sha256"], binding.science_file_sha256, "reviewed source differs")
    for key in ("implementation_commit", "acceptance_commit", "protocol_sha256", "artifact_sha256"):
        same(plan["protocol"][key], getattr(binding, key), "reviewed lineage differs")
    same(
        plan["control_sha256"]["tools/sdsc_student_name_cue_probe_audit.py"],
        sha(Path(__file__).read_bytes()),
        "auditor bytes unbound",
    )
    rows = publication["files"]
    require(
        isinstance(rows, list) and 0 < len(rows) <= 32 and len({r["path"] for r in rows}) == len(rows),
        "published inventory differs",
    )
    require(sum(r["size"] for r in rows) <= MAX_TOTAL_BYTES, "publication exceeds small bound")
    for row in rows:
        name = row["path"]
        require(
            isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
            "unsafe published path",
        )
        require(
            set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 < row["size"] <= MAX_BYTES
            and isinstance(row["sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid file metadata",
        )
        require(
            name in files and len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
            "published bytes differ",
        )
    require(set(files) == {r["path"] for r in rows}, "unbound publication bytes")
    report = document(files["name-cue-report.json"])
    for k, v in dict(
        job_id=publication["job_id"],
        run_id=plan["run_id"],
        plan_sha256=sha(canonical(plan)),
        source_code_sha256=plan["code_sha256"],
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
    ).items():
        same(report.get(k), v, "report provenance differs")
    names = [
        "name-cue-dataset-manifest.json",
        "name-cue-data-isolation.json",
        "name-cue-token-audit.json",
        "name-cue-prompts.jsonl",
    ] + [f"name-cue-records-{a}-rank-{rank}.jsonl" for a in ARMS for rank in (0, 1)]
    require(
        sorted(r["path"] for r in report["raw_artifacts"]) == sorted(names), "complete raw inventory required"
    )
    for row in report["raw_artifacts"]:
        same(
            row, next(r for r in rows if r["path"] == row["path"]), "report/publication raw identity differs"
        )
    return report, publication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    metadata = ROOT / (".opd-git" if (ROOT / ".opd-git").is_dir() else ".git")
    binding = p.resolve_student_name_cue_probe_protocol(ROOT, git_dir=metadata)
    plan = document(read(args.plan))
    verify_producer_head(plan, binding, metadata)
    raw = read(args.fetch_dir / "receipt.json")
    require(sha(raw) == args.receipt_sha256, "externally bound publication hash differs")
    publication = document(raw)
    files = {}
    for r in publication["files"]:
        name = r["path"]
        require(
            isinstance(name, str)
            and Path(name).name == name
            and name not in ("", ".", "..")
            and name not in files,
            "unsafe/duplicate fetched file",
        )
        files[name] = read(args.fetch_dir / name, MAX_BYTES)
        require(sum(map(len, files.values())) <= MAX_TOTAL_BYTES, "fetch exceeds bound")
    report, publication = validate_publication(plan, raw, args.receipt_sha256, files, binding)
    isolation = validate_dataset_isolation(report, files)
    prompts = lines(files["name-cue-prompts.jsonl"], 512)
    records = {
        a: [lines(files[f"name-cue-records-{a}-rank-{rank}.jsonl"], 256) for rank in (0, 1)] for a in ARMS
    }
    result = replay(report, prompts, records, load_tokenizer())
    result.update(
        job_id=publication["job_id"],
        isolation_evidence=isolation,
        publication_sha256=args.receipt_sha256,
        plan_sha256=sha(canonical(plan)),
        result_sha256=sha(files["name-cue-report.json"]),
        auditor_sha256=sha(Path(__file__).read_bytes()),
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        implementation_commit=binding.implementation_commit,
        acceptance_commit=binding.acceptance_commit,
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
