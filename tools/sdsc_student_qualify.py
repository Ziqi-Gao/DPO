#!/usr/bin/env python3
"""Once-only qualification of the frozen prepared student; no downstream acceptance.

Use tools/sdsc check and matching sync dry-run/sync first. Missing submission
acknowledgements require reconciliation. Raw responses remain on persistent
storage and only the bounded named report set can be fetched.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import shlex
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "qwen3-v2-student-qualification-v1"
CAP = 1024**2
MAX_FILE, MAX_FETCH, MAX_PLAN = 16 * CAP, 48 * CAP, 4 * CAP
PROTOCOL_PATH = "prereg/amendments/qwen3_student_qualification_v1.json"
PROTOCOL_MODULE = "src/posttrain_circuits/experiments/protocols/student_qualification.py"
FROZEN = {
    "tools/sdsc_student_prepare.py": "753608ec4d6fbbc5f9a09ff0d567092f69258298c5f32dce3620a6ed3a9f41af",
    "tools/sdsc_student_prepare_job.py": "bffab556c5c1b299c4f9b78b2142929f9f98d1fc346f4c1b4d97ba8fa587a08d",
    "tools/sdsc_student_prepare_worker.py": (
        "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1"
    ),
    "tools/sdsc_student_prepare_audit.py": "ef7160697aab41aea5d0aa067b96688e489c135ca9605b403deb4834efb9d80b",
    "tools/sdsc_student_quality_job.py": "44839b58cdf7a1e3dac2f3524f4ca078e3fb2278d44757f6014ba08bee43c0e3",
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
}


def helper(name, root=ROOT):
    key = "tools/" + name + ".py"
    path = root / key
    if key in FROZEN and hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN[key]:
        raise ValueError("immutable shared helper changed: " + key)
    spec = importlib.util.spec_from_file_location("_qualify_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_prep = helper("sdsc_student_prepare")
FROZEN.update(_prep.FROZEN)
require, canonical, sha, now = _prep.require, _prep.canonical, _prep.sha, _prep.now
safe, relative, write_once = _prep.safe, _prep.relative, _prep.write_once
read, document, run = _prep.read, _prep.document, _prep.run
job_queue, job_accounting = _prep.job_queue, _prep.job_accounting
live_binding = _prep._io.live_binding
validate_accounting = _prep._io.validate_accounting  # Exact W1/24 CPU/192 GiB.
validate_job_memory_events = _prep.validate_job_memory_events
verify_runtime = _prep.verify_runtime
_initial = _prep._initial
FLAGS = _prep.FLAGS
TOOLS = tuple(
    dict.fromkeys(
        (
            "tools/sdsc_student_qualify.py",
            "tools/sdsc_student_qualify_job.py",
            "tools/sdsc_student_qualify_worker.py",
            "tools/sdsc_student_qualify_audit.py",
            PROTOCOL_PATH,
            PROTOCOL_MODULE,
            *FROZEN,
            *_prep.TOOLS,
        )
    )
)
NAMES = (
    "qualification-report.json",
    "qualification-prompts.jsonl",
    "qualification-records.jsonl",
    "node-result.json",
    "memory.json",
)
FIT_JOB = "54562507"
FIT_PLAN_SHA = "3e858360407c78b3e2563959eb84dfc7961ff084366355a45f81ed3d85881b6d"
FIT_PUBLICATION_SHA = "98aff58e27bc88653453047d7610d5c1b2b648482b1ad72616f3628c8613b1bc"
FIT_REPORT_SHA = "1653f312b0732512fe3710b9424bdf9390646d589fa9dc2b921276f6e75962b3"
FIT_AUDIT_SHA = "411fd1f2bb4c0de4fcf30f37c3bc701b25c2edad38da651182bc0012a014cdaf"
PREPARED_SHA = "23854ce0d6db4cb01cae898beccf1ac7be7613e5ee01ff5b1626daafb8349d31"
PREPARED_ROW = dict(path="checkpoints/step-00000004.pt", size=8127108889, sha256=PREPARED_SHA)


def resources():
    return dict(
        account="nwu181",
        partition="nairr-gpu-shared",
        qos="nairr-gpu-shared-normal",
        gpu_type="h100",
        gpus=1,
        cpus=24,
        mem_gib=192,
        time="02:00:00",
    )


def science_identity(protocol):
    return dict(
        task=TASK,
        protocol_sha256=protocol["protocol_sha256"],
        prepared_checkpoint_sha256=PREPARED_SHA,
        fit_job_id=FIT_JOB,
        fit_publication_sha256=FIT_PUBLICATION_SHA,
        fit_audit_sha256=FIT_AUDIT_SHA,
        population="original-validation128-and-anti-shortcut128-plus640",
        training_updates=0,
        checkpoint_selection="frozen-earliest-development-band-step4",
    )


def audit_payload(raw):
    require(len(raw) <= CAP and sha(raw) == FIT_AUDIT_SHA, "fixed independent fit audit differs")
    return dict(sha256=FIT_AUDIT_SHA, size=len(raw), base64=base64.b64encode(raw).decode())


def decode_audit(record):
    require(
        set(record) == {"sha256", "size", "base64"} and record["sha256"] == FIT_AUDIT_SHA,
        "unreviewed fit audit",
    )
    raw = base64.b64decode(record["base64"], validate=True)
    require(0 < len(raw) == record["size"] <= CAP and sha(raw) == FIT_AUDIT_SHA, "fit audit bytes differ")
    report = json.loads(raw)
    require(
        report["raw_replay_passed"] is True
        and report["job_id"] == FIT_JOB
        and report["plan_sha256"] == FIT_PLAN_SHA
        and report["publication_sha256"] == FIT_PUBLICATION_SHA
        and report["prompts_replayed"] == 512
        and report["responses_replayed"] == 2048
        and report["selected_checkpoint"]["step"] == 4
        and report["selected_checkpoint"]["checkpoint_sha256"] == PREPARED_SHA
        and all(report.get(k) is False for k in FLAGS),
        "fit raw replay/selection differs",
    )
    return raw, report


def validate_plan(plan):
    require(
        len(canonical(plan)) <= MAX_PLAN
        and plan["schema"] == "quest-sdsc-student-qualification-plan-v1"
        and plan["task"] == TASK
        and plan["resources"] == resources(),
        "qualification scope/resources differ",
    )
    require(plan["science_identity"] == science_identity(plan["protocol"]), "science identity differs")
    require(
        set(plan["control_sha256"]) == set(TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in plan["control_sha256"].values())
        and all(plan["control_sha256"][p] == h for p, h in FROZEN.items()),
        "control pins differ",
    )
    science = plan["protocol"]["science_file_sha256"]
    required = {
        "tools/sdsc_student_qualify.py",
        "tools/sdsc_student_qualify_job.py",
        "tools/sdsc_student_qualify_worker.py",
        "tools/sdsc_student_qualify_audit.py",
        PROTOCOL_MODULE,
    }
    require(
        isinstance(science, dict)
        and required <= science.keys()
        and all(re.fullmatch("[a-f0-9]{64}", h) for h in science.values()),
        "accepted science inventory incomplete",
    )
    require(
        all(plan["control_sha256"][p] == h for p, h in science.items() if p in plan["control_sha256"]),
        "deployed control differs from accepted scientific byte",
    )
    require(plan["intent_id"] == execution_intent(plan), "execution intent differs")
    require(
        re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["run_id"])
        and re.fullmatch("[a-f0-9]{64}", plan["code_sha256"])
        and re.fullmatch("[a-f0-9]{64}", plan["manifest_sha256"]),
        "release identity differs",
    )
    for k, v in dict(
        release=CONTROL / "releases" / plan["run_id"],
        submission_dir=CONTROL / "student-qualify-submissions" / plan["intent_id"],
        claim=CONTROL / "student-qualify-claims" / (plan["intent_id"] + ".json"),
        scientific_claim=scientific_claim_path(plan),
        result_dir=PROJECT / "student-qualify" / plan["intent_id"],
    ).items():
        require(plan[k] == str(v), "unreviewed plan path: " + k)
    require(plan["job_name"] == "opd-squ-" + plan["intent_id"], "job name differs")
    fit = plan["fit"]
    require(
        set(fit) == {"plan", "plan_sha256", "job_id", "publication_sha256", "report_sha256", "audit"}
        and fit["job_id"] == FIT_JOB
        and fit["plan_sha256"] == FIT_PLAN_SHA == sha(canonical(fit["plan"]))
        and fit["publication_sha256"] == FIT_PUBLICATION_SHA
        and fit["report_sha256"] == FIT_REPORT_SHA,
        "frozen fit provenance differs",
    )
    _prep.validate_plan(fit["plan"])
    require(
        fit["plan"]["mode"] == "fit" and plan["prepared_checkpoint"] == PREPARED_ROW,
        "frozen selected checkpoint differs",
    )
    decode_audit(fit["audit"])
    require(
        plan["python"] == fit["plan"]["python"] and plan["hf_home"] == fit["plan"]["hf_home"],
        "fixed runtime/cache differs",
    )
    provenance = plan["provenance"]
    require(
        set(provenance) == {"directory", "manifest_sha256", "head"}
        and re.fullmatch("[a-f0-9]{64}", provenance["manifest_sha256"])
        and provenance["directory"] == str(CONTROL / "provenance" / provenance["manifest_sha256"])
        and provenance["head"] == plan["protocol"]["head"]
        and re.fullmatch("[a-f0-9]{40}", provenance["head"]),
        "genuine provenance differs",
    )
    require(
        plan["protocol"]["artifact_sha256"] == plan["control_sha256"][PROTOCOL_PATH]
        and plan["protocol"]["implementation_commit"] != plan["protocol"]["acceptance_commit"]
        and all(
            re.fullmatch("[a-f0-9]{40}", plan["protocol"][k])
            for k in ("implementation_commit", "acceptance_commit")
        ),
        "distinct accepted qualification review required",
    )
    require(all(plan.get(k) is False for k in FLAGS), "qualification cannot claim downstream acceptance")
    return plan


def verify_parent(plan, *, accounting=False):
    fit = plan["fit"]["plan"]
    directory = safe(fit["submission_dir"])
    require(
        read(directory / "plan.json") == canonical(fit)
        and document(safe(fit["scientific_claim"])) == _prep.scientific_claim_record(fit)
        and document(safe(fit["claim"]))["plan_sha256"] == FIT_PLAN_SHA,
        "fit permanent claims differ",
    )
    receipt = document(directory / "receipt.json")
    _prep.validate_submission_receipt(receipt, fit)
    require(receipt["job_id"] == FIT_JOB, "fit receipt job differs")
    saved = None if accounting else document(safe(plan["submission_dir"]) / "admission.json")["fit"]
    result = _prep.inspect_job(fit, receipt, accounting_snapshot=saved)
    require(
        result["success"] is True
        and result["preparation_complete"] is True
        and result["publication_sha256"] == FIT_PUBLICATION_SHA
        and result["result_sha256"] == FIT_REPORT_SHA,
        "fit terminal publication not accepted",
    )
    rows = {r["path"]: r for r in result["publication"]["large_files"]}
    require(rows.get(PREPARED_ROW["path"]) == PREPARED_ROW, "selected dense checkpoint publication differs")
    report = document(safe(fit["result_dir"]) / "prepare-report.json", FIT_REPORT_SHA)
    _, audit = decode_audit(plan["fit"]["audit"])
    require(
        audit["development"] == report["development"]
        and audit["selected_checkpoint"] == report["selected_checkpoint"],
        "audited fit selection differs",
    )
    return dict(result, plan_sha256=FIT_PLAN_SHA)


def validate_worker_report(report, plan, job, records, *, require_passed=True):
    expected = dict(
        schema="quest-sdsc-student-qualification-report-v1",
        job_id=job,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        prepared_checkpoint_sha256=PREPARED_SHA,
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        preparation_protocol_sha256=plan["fit"]["plan"]["protocol"]["protocol_sha256"],
        execution_complete=True,
    )
    require(
        all(report.get(k) == v for k, v in expected.items()) and all(report.get(k) is False for k in FLAGS),
        "worker identity/scope differs",
    )
    require(
        report.get("task") == TASK
        and report.get("raw_record_count")
        == report.get("expected_record_count")
        == report.get("prompt_count")
        == 896
        and report.get("parameters_unchanged") is True,
        "complete unmodified inference evidence required",
    )
    loading = report.get("loading", {})
    require(
        loading.get("prepared_checkpoint_sha256") == PREPARED_SHA
        and loading.get("exact_saved_master_reload")
        == dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True)
        and loading.get("forward_parameter_dtype") == "torch.bfloat16"
        and loading.get("all_parameters_frozen") is True
        and loading.get("gradient_checkpointing") is False
        and loading.get("use_cache") is False
        and loading.get("master_model_state_sha256")
        == "b0e00866275772f91b8df132cce7b5628fb6879125dffcce775a5d6939cc80fe"
        and re.fullmatch("[a-f0-9]{64}", loading.get("forward_model_state_sha256", "")),
        "exact prepared-master/native-BF16 loading evidence differs",
    )
    require(
        report.get("generation")
        == dict(
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
        "original generation policy differs",
    )
    fit = report.get("fit_evidence", {})
    require(
        fit.get("fit_job_id") == FIT_JOB
        and fit.get("publication_sha256") == FIT_PUBLICATION_SHA
        and fit.get("report_sha256") == FIT_REPORT_SHA
        and fit.get("independent_audit_sha256") == FIT_AUDIT_SHA
        and fit.get("selected_checkpoint") == decode_audit(plan["fit"]["audit"])[1]["selected_checkpoint"],
        "worker audited parent binding differs",
    )
    binding = report.get("scientific_binding", {})
    require(
        all(
            binding.get(k) == plan["protocol"][k]
            for k in ("head", "implementation_commit", "acceptance_commit", "science_file_sha256")
        )
        and re.fullmatch("[a-f0-9]{40}", binding.get("prereg_commit", "")),
        "worker accepted scientific identity differs",
    )
    gpu = report.get("gpu", {})
    require(
        "H100" in gpu.get("name", "")
        and gpu.get("logical_index") == 0
        and gpu.get("cpu_threads") == 24
        and gpu.get("torch") == "2.8.0+cu128"
        and isinstance(gpu.get("cuda_visible_devices"), str)
        and gpu["cuda_visible_devices"]
        and "," not in gpu["cuda_visible_devices"],
        "actual inference device differs",
    )
    validation = report["validation"]
    require(
        validation["num_examples"] == 128
        and all(
            type(validation[k]) is int and 0 <= validation[k] <= 128
            for k in ("answer_correct", "proof_correct", "format_valid")
        )
        and validation["proof_correct"] <= validation["answer_correct"] <= validation["format_valid"],
        "validation counts differ",
    )
    for metric, key in [
        ("answer_accuracy", "answer_correct"),
        ("exact_proof_accuracy", "proof_correct"),
        ("format_validity", "format_valid"),
    ]:
        require(
            type(validation[metric]) in (int, float) and validation[metric] == validation[key] / 128,
            "validation metric differs",
        )
    valid = validation["answer_correct"] >= 13
    require(validation["passed"] is valid, "validation gate differs")
    anti = report["anti_shortcut"]
    require(
        anti["model_checkpoint_hash"] == PREPARED_SHA
        and anti["iid_example_count"] == 128
        and anti["transformed_case_count"] == 640,
        "anti-shortcut identity differs",
    )
    # The domain reducer/independent raw auditor proves exact metrics; transport
    # additionally rejects internally inconsistent gate flags and changed floors.
    transformations = {
        "entity_symbol_renaming",
        "fact_order_permutation",
        "rule_order_permutation",
        "surface_template_paraphrase",
        "distractor_count_ood",
    }
    values = anti["transformation_accuracy"]
    require(
        isinstance(values, dict)
        and set(values) == transformations
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values.values()),
        "anti-shortcut accuracy inventory differs",
    )
    require(
        anti["max_shortcut_gap"] == 0.05
        and anti["minimum_iid_accuracy"] == 0.10
        and anti["minimum_transformed_accuracy"] == 0.08
        and anti["minimum_per_transformation_accuracy"] == 0.05,
        "anti-shortcut thresholds differ",
    )
    transformed = sum(values.values()) / len(values)
    iid = anti["iid_accuracy"]
    require(
        type(iid) in (int, float)
        and math.isfinite(iid)
        and 0 <= iid <= 1
        and anti["transformed_accuracy"] == transformed
        and anti["shortcut_gap"] == iid - transformed,
        "anti-shortcut aggregate differs",
    )
    capable = iid >= 0.10 and transformed >= 0.08 and all(v >= 0.05 for v in values.values())
    passed = capable and iid - transformed <= 0.05
    require(anti["capability_passed"] is capable and anti["passed"] is passed, "anti-shortcut gate differs")
    require(
        report.get("base_gate_passed") is valid and report.get("anti_shortcut_passed") is passed,
        "reported component gate differs",
    )
    require(
        anti.get("dataset_hash")
        == report.get("iid_examples_sha256")
        == "ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3"
        and anti.get("suite_hash")
        == report.get("suite_sha256")
        == "c5ed272d5fe8f6dd21a7743dfca44d3bdfa06db44499842b7cc44df807b8ced0"
        and anti.get("sha256") == sha(canonical({k: v for k, v in anti.items() if k != "sha256"})),
        "anti-shortcut population/report digest differs",
    )
    qualification = valid and passed
    require(
        report["qualification_passed"] is qualification and report["passed"] is qualification,
        "qualification outcome differs",
    )
    if require_passed:
        require(qualification, "frozen checkpoint failed qualification; no alternative checkpoint or retry")
    raw = report["raw_artifacts"]
    require(
        isinstance(raw, list) and len(raw) == 2 and {r["path"] for r in raw} == set(NAMES[1:3]),
        "complete raw response inventory missing",
    )
    require(all(records.get(r["path"]) == r for r in raw), "raw artifacts differ from publication")
    return qualification


def publication_records(publication, plan, job):
    expected = dict(
        task=TASK,
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        persistent_read_back_verified=True,
    )
    require(
        all(publication.get(k) == v for k, v in expected.items())
        and all(publication.get(k) is False for k in FLAGS),
        "publication identity/scope differs",
    )
    rows = publication["files"]
    records = {}
    require(isinstance(rows, list) and len(rows) <= len(NAMES), "invalid publication inventory")
    for row in rows:
        name = relative(row["path"])
        require(
            name in NAMES
            and name not in records
            and set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 <= row["size"] <= MAX_FILE
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid raw result inventory",
        )
        records[name] = row
    require(sum(r["size"] for r in rows) <= MAX_FETCH - CAP, "raw output exceeds bound")
    return records


def inspect_job(plan, receipt):
    job = receipt["job_id"]
    directory = safe(plan["submission_dir"])
    observed = (
        document(directory / "live-binding.json") if (directory / "live-binding.json").exists() else None
    )
    result = validate_accounting(plan, job, job_queue(job), job_accounting(job), observed)
    result.update(
        success=False,
        stage_complete=False,
        qualification_passed=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        checked_at=now(),
        **dict.fromkeys(FLAGS, False),
    )
    root = safe(plan["result_dir"])
    records = {}
    if (root / "receipt.json").exists():
        raw = read(root / "receipt.json")
        published = json.loads(raw)
        records = publication_records(published, plan, job)
        require({n for n in NAMES if (root / n).exists()} == set(records), "published inventory differs")
        for name, row in records.items():
            data = read(root / name)
            require(len(data) == row["size"] and sha(data) == row["sha256"], "persistent raw hash differs")
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=published,
            publication_sha256=sha(raw),
        )
    if result["accounting_complete"]:
        require(
            set(records) == set(NAMES)
            and published["passed"] is True
            and published["stage_complete"] is True,
            "completed allocation lacks complete successful publication",
        )
        validate_worker_report(document(root / "qualification-report.json"), plan, job, records)
        validate_memory(document(root / "memory.json"), job)
        node = document(root / "node-result.json")
        require(
            node["stage_complete"] is True
            and node["exit_code"] == 0
            and node["job_id"] == job
            and node["plan_sha256"] == sha(canonical(plan))
            and all(node.get(k) is False for k in FLAGS),
            "node identity/success differs",
        )
        result.update(
            success=True,
            stage_complete=True,
            qualification_passed=True,
            result_sha256=sha(read(root / "qualification-report.json")),
        )
    return result


def admission(plan):
    fit = verify_parent(plan, accounting=True)
    verify_science(plan)
    require(
        not safe(plan["scientific_claim"]).exists(),
        "scientific stage already claimed; reconcile, never resubmit",
    )
    require(
        not safe(plan["claim"]).exists() and not safe(plan["submission_dir"]).exists(),
        "intent already claimed",
    )
    require(
        safe(plan["python"]).is_file()
        and os.access(plan["python"], os.X_OK)
        and sha(read(plan["python"], 64 * CAP))
        == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "fixed runtime executable differs",
    )
    require(
        safe(plan["hf_home"]).is_dir() and PROJECT.is_dir() and os.access(PROJECT, os.W_OK),
        "storage unavailable",
    )
    return dict(fit=fit, runtime=verify_runtime(plan), gpu_concurrency=check_gpu_ceiling())


def prepare(args, cli):
    record = cli.run_record(args.run_id)
    require(record["state"] == "deployed", "first sync --dry-run then sync")
    manifest = record["manifest"]
    records = manifest_records(manifest)
    require(
        record["deployment"]["code_sha256"] == manifest["code_sha256"]
        and all(sha(read(ROOT / name)) == records[name]["sha256"] for name in TOOLS),
        "local/deployed controls differ",
    )
    fit = _prep.validate_plan(document(safe(args.fit_plan.absolute())))
    require(sha(canonical(fit)) == FIT_PLAN_SHA, "exact completed fit plan required")
    fetch = safe(args.fit_fetch_dir.absolute())
    report = document(fetch / "prepare-report.json", FIT_REPORT_SHA)
    publication = document(fetch / "receipt.json", FIT_PUBLICATION_SHA)
    rows = _prep.publication_records(publication, fit, FIT_JOB)
    _prep.validate_worker_report(report, fit, FIT_JOB, rows)
    for name, row in rows.items():
        data = read(fetch / name)
        require(len(data) == row["size"] and sha(data) == row["sha256"], "local fit evidence differs")
    status = document(safe(args.fit_status_file.absolute()))
    status = status.get("status", status)
    require(
        status["job_id"] == FIT_JOB
        and status["success"] is True
        and status["preparation_complete"] is True
        and status["publication_sha256"] == FIT_PUBLICATION_SHA
        and status["result_sha256"] == FIT_REPORT_SHA,
        "verified terminal fit status required",
    )
    audit = audit_payload(read(safe(args.fit_audit.absolute())))
    _, audited = decode_audit(audit)
    require(
        audited["development"] == report["development"]
        and audited["selected_checkpoint"] == report["selected_checkpoint"],
        "audited checkpoint selection differs",
    )
    protocol = protocol_binding(ROOT, args.science_git_head)
    plan = dict(
        schema="quest-sdsc-student-qualification-plan-v1",
        task=TASK,
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        resources=resources(),
        fit=dict(
            plan=fit,
            plan_sha256=FIT_PLAN_SHA,
            job_id=FIT_JOB,
            publication_sha256=FIT_PUBLICATION_SHA,
            report_sha256=FIT_REPORT_SHA,
            audit=audit,
        ),
        prepared_checkpoint=PREPARED_ROW,
        protocol=protocol,
        control_sha256={n: records[n]["sha256"] for n in TOOLS},
        provenance=dict(
            directory=args.provenance_dir,
            manifest_sha256=args.provenance_manifest_sha256,
            head=args.science_git_head,
        ),
        science_identity=science_identity(protocol),
        python=fit["python"],
        hf_home=fit["hf_home"],
        **dict.fromkeys(FLAGS, False),
    )
    intent = execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(CONTROL / "student-qualify-submissions" / intent),
        claim=str(CONTROL / "student-qualify-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-qualify" / intent),
        job_name="opd-squ-" + intent,
    )
    plan["scientific_claim"] = str(scientific_claim_path(plan))
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-qualify" / intent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"), plan_sha256=sha(canonical(plan)), resources=plan["resources"]
    )


def validate_memory(report, job):
    """Recompute raw limiting-ancestor peaks, including checkpoint publication."""
    phases = ("initial", "after_staging", "after_inference", "final", "after_publication")
    require(
        isinstance(report, dict) and set(phases) <= set(report),
        "successful diagnostic omits memory/publication measurement",
    )
    for evidence in report.values():
        validate_job_memory_events(evidence, job)
    previous_peak, previous_time, boundaries = 0, 0, None
    limit = 192 * 1024**3
    for phase in phases:
        evidence = report[phase]
        ancestors = evidence.get("ancestors")
        require(isinstance(ancestors, list) and ancestors, "raw memory hierarchy missing")
        finite = [x for x in ancestors if x.get("limit_bytes") is not None]
        require(
            finite and all(type(x.get("limit_bytes")) is int and x["limit_bytes"] > 0 for x in finite),
            "invalid raw memory limit",
        )
        require(min(x["limit_bytes"] for x in finite) == limit, "actual memory allocation differs")
        limiting = [x for x in finite if x["limit_bytes"] == limit]
        current_boundaries = sorted((x["path"], x["limit_bytes"]) for x in finite)
        require(boundaries is None or current_boundaries == boundaries, "memory cgroup boundaries changed")
        boundaries = current_boundaries
        for row in limiting:
            require(
                "job_" + job in Path(row["path"]).parts
                and all(type(row.get(k)) is int for k in ("current_bytes", "peak_bytes"))
                and 0 < row["current_bytes"] <= row["peak_bytes"] <= limit,
                "memory peak/current evidence differs from this allocation",
            )
        selected = max(limiting, key=lambda x: x["peak_bytes"])
        peak = selected["peak_bytes"]
        observed = evidence.get("observed_at_unix")
        required = max(32 * 1024**3, math.ceil(limit * 0.20))
        require(
            evidence.get("passed") is True
            and evidence.get("expected_limit_bytes") == limit == evidence.get("limit_bytes")
            and evidence.get("path") == selected["path"]
            and evidence.get("peak_bytes") == peak
            and evidence.get("current_bytes") == selected["current_bytes"]
            and evidence.get("headroom_bytes") == limit - peak >= required
            and evidence.get("minimum_headroom_bytes") == required
            and type(observed) in (int, float)
            and math.isfinite(observed)
            and observed >= previous_time
            and peak >= previous_peak,
            "memory aggregate peak, chronology or required headroom differs",
        )
        previous_peak, previous_time = peak, observed


def protocol_binding(root, expected_head=None):
    # A fresh interpreter prevents imports from another restored checkout being
    # reused as though they came from this genuine accepted scientific tree.
    code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "sys.path.insert(0,str(root/'src'));"
        "from posttrain_circuits.experiments.protocols.student_qualification "
        "import resolve_student_qualification_protocol;"
        "v=resolve_student_qualification_protocol(root,require_accepted=True,"
        "expected_head=sys.argv[2] or None,"
        "git_dir=root/('.opd-git' if (root/'.opd-git').is_dir() else '.git'));"
        "print(json.dumps({k:getattr(v,k) for k in ('protocol_sha256','artifact_sha256',"
        "'implementation_commit','acceptance_commit','head','science_file_sha256')}))"
    )
    result = run([sys.executable, "-I", "-B", "-c", code, str(root), expected_head or ""], timeout=180)
    require(
        result["returncode"] == 0,
        "accepted genuine scientific protocol did not resolve: " + result["stderr"][-1500:],
    )
    return json.loads(result["stdout"])


def scientific_claim_path(plan):
    return (
        CONTROL / "student-qualify-scientific-claims" / (sha(canonical(plan["science_identity"])) + ".json")
    )


def scientific_claim_record(plan):
    return dict(
        science_identity=plan["science_identity"],
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        submission_dir=plan["submission_dir"],
        no_retry=True,
    )


def execution_intent(plan):
    return sha(
        canonical(
            dict(
                science=plan["science_identity"],
                controls=plan["control_sha256"],
                provenance=plan["provenance"],
                resources=plan["resources"],
            )
        )
    )[:32]


def manifest_records(manifest):
    rows = manifest["files"]
    require(
        isinstance(rows, list) and 0 < len(rows) < 10000 and manifest["code_sha256"] == sha(canonical(rows)),
        "snapshot inventory differs",
    )
    records = {}
    for row in rows:
        name = relative(row["path"])
        require(
            name not in records
            and set(row) == {"path", "size", "sha256", "mode"}
            and type(row["size"]) is int
            and 0 <= row["size"] <= 4 * CAP
            and row["mode"] in (0o644, 0o755)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid source row",
        )
        records[name] = row
    require(
        set(TOOLS) <= records.keys() and sum(r["size"] for r in rows) == manifest["total_bytes"] <= 48 * CAP,
        "source snapshot lacks controls or exceeds bound",
    )
    return records


def verify_source(plan, root=None):
    release = safe(plan["release"])
    manifest = document(release / "manifest.json", plan["manifest_sha256"])
    require(
        manifest["run_id"] == plan["run_id"] and manifest["code_sha256"] == plan["code_sha256"],
        "release differs",
    )
    records = manifest_records(manifest)
    for name, digest in plan["protocol"]["science_file_sha256"].items():
        require(
            name in records and records[name]["sha256"] == digest,
            "deployed source differs from accepted implementation: " + name,
        )
    root = safe(root or release / "source")
    for name, row in records.items():
        raw = read(root / name, 4 * CAP)
        require(len(raw) == row["size"] and sha(raw) == row["sha256"], "source bytes differ: " + name)
    require(
        all(records[name]["sha256"] == digest for name, digest in plan["control_sha256"].items()),
        "controls differ",
    )
    return manifest


def verify_science(plan, destination=None):
    provenance = plan["provenance"]
    if destination is None:
        with tempfile.TemporaryDirectory(prefix="student-qualify-verify-") as temp:
            return verify_science(plan, Path(temp) / "science")
    restored = helper("sdsc_provenance").verify(
        safe(provenance["directory"]), provenance["manifest_sha256"], plan["code_sha256"], destination
    )
    require(restored["git_head"] == provenance["head"], "restored genuine HEAD differs")
    binding = protocol_binding(destination, provenance["head"])
    require(binding == plan["protocol"], "restored accepted successor differs")
    return restored


def check_gpu_ceiling():
    user = pwd.getpwuid(os.getuid()).pw_name
    query = run(
        ["squeue", "--noheader", "--array", "--local", "--states=all", "--user=" + user, "--format=%i|%T|%b"]
    )
    require(query["returncode"] == 0, "GPU inventory query unavailable")
    jobs, seen = [], set()
    for line in query["stdout"].splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        require(
            len(fields) == 3
            and re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", fields[0])
            and fields[0] not in seen,
            "ambiguous GPU inventory",
        )
        seen.add(fields[0])
        jobs.append(_initial.live_gpu_request(fields[0], user, fields[2].strip(), queue_state=fields[1]))
    total = sum(row["allocatable_gpus"] for row in jobs)
    require(total + 1 <= 4, "new qualification would exceed four allocatable GPUs")
    return dict(existing_allocatable_gpus=total, new_gpus=1, limit=4, jobs=jobs, query=query)


def sbatch(plan):
    directory = safe(plan["submission_dir"])
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=24",
        "--account=nwu181",
        "--partition=nairr-gpu-shared",
        "--qos=nairr-gpu-shared-normal",
        "--gpus=h100:1",
        "--mem=196608M",
        "--time=" + plan["resources"]["time"],
        "--signal=B:TERM@180",
        "--job-name=" + plan["job_name"],
        "--comment=" + plan["intent_id"],
        "--chdir=" + str(directory),
        "--output=" + str(directory / "slurm-%j.out"),
        "--error=" + str(directory / "slurm-%j.err"),
        "--export=NONE",
        str(directory / "job.sh"),
        sha(canonical(plan)),
    ]


def job_script(plan):
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_qualify_job.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p)]+sys.argv[3:];runpy.run_path(str(p),run_name="__main__")'
    )
    return (
        "#!/bin/bash\nset -euo pipefail\numask 077\nexec "
        + shlex.join(
            [
                plan["python"],
                "-I",
                "-B",
                "-u",
                "-c",
                launch,
                worker,
                plan["control_sha256"]["tools/sdsc_student_qualify_job.py"],
                str(Path(plan["submission_dir"]) / "plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def make_receipt(plan, job, reconciled=False):
    require(re.fullmatch("[1-9][0-9]*", job), "invalid job ID")
    return dict(
        task=TASK,
        job_id=job,
        intent_id=plan["intent_id"],
        run_id=plan["run_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        result_dir=plan["result_dir"],
        resources=plan["resources"],
        reconciled=reconciled,
        received_at=now(),
    )


def validate_submission_receipt(receipt, plan):
    expected = make_receipt(plan, receipt.get("job_id", ""))
    require(
        all(
            receipt.get(key) == value
            for key, value in expected.items()
            if key not in {"received_at", "reconciled"}
        ),
        "submission receipt differs",
    )


def remote_action(request):
    plan = validate_plan(request["plan"])
    verify_source(plan)
    directory, claim = safe(plan["submission_dir"]), safe(plan["claim"])
    action = request["action"]
    if action == "dry-run":
        checks = admission(plan)
        return dict(
            dry_run=True,
            plan_sha256=sha(canonical(plan)),
            argv=sbatch(plan),
            resources=plan["resources"],
            source_verified=True,
            parent_verified=True,
            accepted_protocol_verified=True,
            node_mounts_still_required=True,
            blockers=[],
            **checks,
        )
    if action == "submit":
        require(request.get("authorize") is True, "explicit authorization required")
        proof = request.get("dry_run")
        require(
            isinstance(proof, dict)
            and proof.get("dry_run") is True
            and proof.get("blockers") == []
            and proof.get("plan_sha256") == sha(canonical(plan))
            and proof.get("argv") == sbatch(plan),
            "matching dry-run required",
        )
        checks = admission(plan)
        directory.parent.mkdir(parents=True, exist_ok=True)
        claim.parent.mkdir(parents=True, exist_ok=True)
        science_claim = safe(plan["scientific_claim"])
        science_claim.parent.mkdir(parents=True, exist_ok=True)
        write_once(science_claim, canonical(scientific_claim_record(plan)))
        write_once(
            claim, canonical(dict(intent_id=plan["intent_id"], plan_sha256=sha(canonical(plan)), at=now()))
        )
        directory.mkdir(mode=0o700)
        write_once(directory / "plan.json", canonical(plan))
        write_once(directory / "job.sh", job_script(plan))
        write_once(directory / "admission.json", canonical(checks))
        write_once(directory / "submission-started.json", canonical(dict(argv=sbatch(plan), at=now())))
        try:
            result = run(sbatch(plan))
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", result["stdout"])
            require(result["returncode"] == 0 and match is not None, "missing trustworthy acknowledgement")
            receipt = make_receipt(plan, match.group(1))
            write_once(directory / "receipt.json", canonical(receipt))
            write_once(directory / "live-binding.json", canonical(live_binding(plan, receipt["job_id"])))
            return receipt
        except BaseException as error:
            write_once(
                directory / "unknown.json", canonical(dict(error=str(error)[:1000], no_retry=True, at=now()))
            )
            raise
    require(
        read(directory / "plan.json") == canonical(plan)
        and document(claim)["plan_sha256"] == sha(canonical(plan))
        and document(safe(plan["scientific_claim"])) == scientific_claim_record(plan),
        "permanent claims differ",
    )
    if action == "reconcile" and not (directory / "receipt.json").exists():
        user = pwd.getpwuid(os.getuid()).pw_name
        q = run(["squeue", "--noheader", "--user=" + user, "--name=" + plan["job_name"], "--format=%i|%j|%k"])
        a = run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--user=" + user,
                "--name=" + plan["job_name"],
                "--starttime=" + plan["created_at"][:10],
                "--format=JobIDRaw,JobName%100,Comment%100,User%64",
            ]
        )
        require(q["returncode"] == a["returncode"] == 0, "reconciliation unavailable; never retry")
        ids = set()
        for line in (q["stdout"] + "\n" + a["stdout"]).splitlines():
            fields = line.split("|")
            if (
                len(fields) >= 3
                and fields[1:3] == [plan["job_name"], plan["intent_id"]]
                and re.fullmatch("[1-9][0-9]*", fields[0])
                and (len(fields) == 3 or fields[3] == user)
            ):
                ids.add(fields[0])
        require(len(ids) == 1, "zero/ambiguous submission matches; never retry")
        write_once(directory / "receipt.json", canonical(make_receipt(plan, ids.pop(), True)))
    receipt = document(directory / "receipt.json")
    validate_submission_receipt(receipt, plan)
    if action == "reconcile":
        if not (directory / "live-binding.json").exists():
            write_once(directory / "live-binding.json", canonical(live_binding(plan, receipt["job_id"])))
        return receipt
    status = inspect_job(plan, receipt)
    if action == "status":
        return status
    require(action == "fetch", "unknown remote operation")
    files, total = {}, 0
    if status["publication_verified"]:
        records = publication_records(status["publication"], plan, receipt["job_id"])
        for name in (*records, "receipt.json"):
            data = read(safe(plan["result_dir"]) / name)
            expected = status["publication_sha256"] if name == "receipt.json" else records[name]["sha256"]
            require(sha(data) == expected, "artifact changed during fetch")
            files[name] = base64.b64encode(data).decode()
            total += len(data)
    for name, path in [
        ("worker.log", safe(plan["result_dir"]) / "worker.log"),
        ("slurm.out", directory / ("slurm-" + receipt["job_id"] + ".out")),
        ("slurm.err", directory / ("slurm-" + receipt["job_id"] + ".err")),
    ]:
        if path.exists():
            with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode), "nonregular log")
                stream.seek(max(0, info.st_size - 65536))
                data = stream.read(65536)
            files[name] = base64.b64encode(data).decode()
            total += len(data)
    require(total <= MAX_FETCH, "fetch exceeds bound; weights stay remote")
    return dict(status=status, receipt=receipt, files=files, bytes=total)


def validate_download(result, plan):
    validate_submission_receipt(result["receipt"], plan)
    require(
        isinstance(result["files"], dict)
        and set(result["files"]) <= set(NAMES) | {"receipt.json", "worker.log", "slurm.out", "slurm.err"},
        "unexpected fetched path",
    )
    files = {
        relative(name): base64.b64decode(value, validate=True) for name, value in result["files"].items()
    }
    require(
        sum(map(len, files.values())) == result["bytes"] <= MAX_FETCH
        and all(len(data) <= MAX_FILE for data in files.values()),
        "fetch size differs",
    )
    if result["status"]["publication_verified"]:
        publication = json.loads(files["receipt.json"])
        require(
            sha(files["receipt.json"]) == result["status"]["publication_sha256"]
            and publication == result["status"]["publication"],
            "publication differs",
        )
        rows = publication_records(publication, plan, result["receipt"]["job_id"])
        require(set(rows) <= files.keys(), "fetch missing published raw evidence")
        for name, row in rows.items():
            require(
                len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
                "fetched raw hash differs",
            )
    return files


def ssh_operation(cli, plan, action, authorize=False, dry_run=None):
    path = str(Path(plan["release"]) / "source/tools/sdsc_student_qualify.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p),"remote"];runpy.run_path(str(p),run_name="__main__")'
    )
    result = cli.ssh_call(
        [plan["python"], "-I", "-B", "-c", launch, path, plan["control_sha256"][TOOLS[0]]],
        data=canonical(dict(action=action, plan=plan, authorize=authorize, dry_run=dry_run)),
        timeout=240,
    )
    require(
        result.returncode == 0,
        "remote action failed/acknowledgement unknown; reconcile: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    require(len(result.stdout) <= 2 * MAX_FETCH, "remote response exceeds bound")
    return json.loads(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--run-id")
    for name in ("fit-fetch-dir", "fit-status-file", "fit-plan", "fit-audit", "plan", "dry-run-file"):
        parser.add_argument("--" + name, type=Path)
    for name in ("provenance-dir", "provenance-manifest-sha256", "science-git-head"):
        parser.add_argument("--" + name)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(
            Path(__file__).resolve().is_relative_to(CONTROL / "releases"),
            "Slurm operations cannot run on Quest",
        )
        raw = sys.stdin.buffer.read(MAX_PLAN + CAP + 1)
        require(len(raw) <= MAX_PLAN + CAP, "remote request exceeds bound")
        result = remote_action(json.loads(raw))
    else:
        cli = helper("sdsc_cli")
        if args.action == "prepare":
            require(
                all(
                    (
                        args.run_id,
                        args.fit_fetch_dir,
                        args.fit_status_file,
                        args.fit_plan,
                        args.fit_audit,
                        args.provenance_dir,
                        args.provenance_manifest_sha256,
                        args.science_git_head,
                    )
                ),
                "prepare requires release/verified-fit/audit/provenance",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-qualify" and path.name == "plan.json",
                "local plan path differs",
            )
            plan = validate_plan(document(path))
            require(
                path.parent.name == plan["intent_id"]
                and all(sha(read(ROOT / name)) == digest for name, digest in plan["control_sha256"].items()),
                "local controls changed",
            )
            action = "dry-run" if args.action == "submit" and args.dry_run else args.action
            dry_run = None
            if action == "submit":
                require(
                    args.authorize and args.dry_run_file is not None,
                    "submit requires authorize and matching dry-run-file",
                )
                dry_run = document(safe(args.dry_run_file.absolute()))
                cli.require_master()
                write_once(path.parent / "submission-started.json", canonical(dict(at=now(), no_retry=True)))
            result = ssh_operation(cli, plan, action, args.authorize, dry_run)
            if action == "fetch":
                files = validate_download(result, plan)
                directory = safe(ROOT / ".sdsc/fetched" / result["receipt"]["job_id"])
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination = Path(tempfile.mkdtemp(prefix="fetch-", dir=directory))
                for name, raw in files.items():
                    write_once(destination / name, raw)
                    require(read(destination / name) == raw, "local readback differs")
                result.pop("files")
                result["destination"] = str(destination)
                write_once(destination / "fetch-manifest.json", canonical(result))
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
            write_once(path.parent / (action + "-" + stamp + ".json"), canonical(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
