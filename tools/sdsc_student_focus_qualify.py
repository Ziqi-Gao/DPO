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
TASK = "qwen3-v2-student-focus-qualification-v1"
CAP = 1024**2
MAX_FILE, MAX_FETCH, MAX_PLAN = 16 * CAP, 48 * CAP, 4 * CAP
PROTOCOL_PATH = "prereg/amendments/qwen3_student_focus_qualification_v1.json"
PROTOCOL_MODULE = "src/posttrain_circuits/experiments/protocols/student_focus_qualification.py"
FROZEN = {
    "tools/sdsc_student_qualify_worker.py": (
        "31686d05b801efe487820604ca4dba274d53415fefe857a00a36cd7d73fd0f4a"
    ),
    "tools/sdsc_student_qualify_audit.py": "f9303357d29a21b05388cdf725a24133d1cb44015cb4c7e09f991fcb7d291596",
    "tools/sdsc_student_prepare.py": "753608ec4d6fbbc5f9a09ff0d567092f69258298c5f32dce3620a6ed3a9f41af",
    "tools/sdsc_student_prepare_job.py": "bffab556c5c1b299c4f9b78b2142929f9f98d1fc346f4c1b4d97ba8fa587a08d",
    "tools/sdsc_student_prepare_worker.py": (
        "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1"
    ),
    "tools/sdsc_student_prepare_audit.py": "ef7160697aab41aea5d0aa067b96688e489c135ca9605b403deb4834efb9d80b",
    "tools/sdsc_student_quality_job.py": "44839b58cdf7a1e3dac2f3524f4ca078e3fb2278d44757f6014ba08bee43c0e3",
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
    "tools/sdsc_student_branch_qualify_worker.py": (
        "35cd93ef1a36b7f53f4c887efe283bba5c15248c9cf4a836c1bcc74ab696515f"
    ),
    "tools/sdsc_student_focus.py": "36020e7fbaa923dbb3a559894d0e0fb11390e8d88efded30395a968e9620f0d7",
    "tools/sdsc_student_focus_job.py": "6a60a5298b89861d3182a5286af9d1b6ce83106089dbbc5b300bb0ac1d4eeffc",
    "tools/sdsc_student_focus_startup.py": "0a0cf9c3ea89fbc1748c8f9bfef16e15402d5280cb0a68a3807387bed45f1b06",
    "tools/sdsc_student_focus_worker.py": "cf14b68237e3bc1b6185637d1c615f3bdb726279fedda3c15cd0f5c3efcace06",
    "tools/sdsc_student_focus_audit.py": "d69f988bd062b0b9f22a0cd87e0b6c7554302dc60bb4a9556e692e7ec2bfd5fd",
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


_fit = helper("sdsc_student_focus")
FROZEN.update(_fit.FROZEN)
require, canonical, sha, now = _fit.require, _fit.canonical, _fit.sha, _fit.now
safe, relative, write_once = _fit.safe, _fit.relative, _fit.write_once
run = _fit.run
job_queue, job_accounting = _fit.job_queue, _fit.job_accounting
live_binding = _fit._io.live_binding
validate_accounting = _fit._io.validate_accounting  # Exact W1/24 CPU/192 GiB.
validate_job_memory_events = _fit.validate_job_memory_events
verify_runtime = _fit.verify_runtime
_initial = _fit._initial
FLAGS = _fit.FLAGS
TOOLS = tuple(
    dict.fromkeys(
        (
            "tools/sdsc_student_focus_qualify.py",
            "tools/sdsc_student_focus_qualify_job.py",
            "tools/sdsc_student_focus_qualify_worker.py",
            "tools/sdsc_student_focus_qualify_audit.py",
            PROTOCOL_PATH,
            PROTOCOL_MODULE,
            *FROZEN,
            *_fit.TOOLS,
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


def read(path, limit=MAX_FILE):
    return _fit._io.read(path, limit=limit)


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "evidence SHA differs")
    return json.loads(raw)


COMPLETION_NAMES = ("verified_status", "publication")


def qualification_domain():
    """Load this deployment's core; admission verifies its accepted source first."""
    path = ROOT / PROTOCOL_MODULE
    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("_focus_qualification_domain", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def completion_records(status, publication_raw):
    raw = dict(verified_status=canonical(status), publication=publication_raw)
    require(sum(map(len, raw.values())) <= MAX_PLAN, "completion evidence exceeds bound")
    return {
        name: dict(sha256=sha(data), size=len(data), base64=base64.b64encode(data).decode())
        for name, data in raw.items()
    }


def decode_completion_records(records, candidate, fit):
    """Bounded pure replay; admission still requires the full frozen live inspector."""
    require(
        isinstance(records, dict) and set(records) == set(COMPLETION_NAMES), "completion inventory differs"
    )
    raw, total = {}, len(canonical(fit))
    for name in COMPLETION_NAMES:
        row = records[name]
        require(
            isinstance(row, dict)
            and set(row) == {"sha256", "size", "base64"}
            and type(row["size"]) is int
            and 0 < row["size"] <= MAX_PLAN
            and isinstance(row["sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"])
            and isinstance(row["base64"], str)
            and len(row["base64"]) <= 4 * ((MAX_PLAN + 2) // 3),
            "completion record differs",
        )
        data = base64.b64decode(row["base64"], validate=True)
        total += len(data)
        require(
            total <= MAX_PLAN and len(data) == row["size"] and sha(data) == row["sha256"],
            "completion record bytes differ",
        )
        value = json.loads(data)
        require(
            isinstance(value, dict) and data == canonical(value),
            "completion JSON must preserve canonical bytes",
        )
        raw[name] = data
    qualification_domain().validate_parent_completion_evidence(
        candidate, fit_plan=fit, status=json.loads(raw["verified_status"]), publication_raw=raw["publication"]
    )
    return raw


def completion_summary(records, fit):
    raw = canonical(fit)
    return dict(
        fit_plan=dict(size=len(raw), sha256=sha(raw)),
        **{name: {key: row[key] for key in ("size", "sha256")} for name, row in records.items()},
    )


def qualification_payload(protocol):
    """Read the exact reviewed protocol artifact bound by this execution."""
    raw = read(ROOT / PROTOCOL_PATH)
    require(sha(raw) == protocol["artifact_sha256"], "qualification protocol artifact differs")
    payload = json.loads(raw)
    require(
        sha(canonical({key: value for key, value in payload.items() if key != "review"}))
        == protocol["protocol_sha256"],
        "qualification protocol core differs",
    )
    return payload


def prepared_initial(protocol):
    """Read the single candidate from the hash-bound accepted protocol artifact."""
    candidate = qualification_payload(protocol)["prepared_initial"]
    require(isinstance(candidate, dict), "completed single candidate not frozen")
    return candidate


def validate_parent_binding(protocol, fit):
    parent = qualification_payload(protocol)["preserved_base"]
    candidate = prepared_initial(protocol)
    expected = dict(
        protocol_sha256=parent["parent_protocol_core_sha256"],
        artifact_sha256=parent["parent_student_protocol_sha256"],
        implementation_commit=parent["parent_implementation_commit"],
        acceptance_commit=parent["parent_acceptance_commit"],
        head=candidate["fit_source_head"],
    )
    require(
        all(fit["protocol"].get(key) == value for key, value in expected.items())
        and fit["provenance"]["head"] == candidate["fit_source_head"]
        and parent["parent_student_protocol"] == _fit.PROTOCOL_PATH
        and fit["control_sha256"]["tools/sdsc_student_focus_audit.py"] == parent["parent_auditor_sha256"],
        "fit accepted parent lineage differs",
    )


def checkpoint_row(candidate):
    return dict(
        path=candidate["checkpoint_path"],
        size=candidate["checkpoint_size"],
        sha256=candidate["checkpoint_sha256"],
    )


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
    candidate = prepared_initial(protocol)
    return dict(
        task=TASK,
        protocol_sha256=protocol["protocol_sha256"],
        prepared_checkpoint_sha256=candidate["checkpoint_sha256"],
        prepared_master_model_state_sha256=candidate["master_model_state_sha256"],
        fit_job_id=candidate["fit_job_id"],
        fit_publication_sha256=candidate["fit_publication_sha256"],
        fit_audit_sha256=candidate["independent_raw_audit_sha256"],
        population="original-validation128-and-anti-shortcut128-plus640",
        training_updates=0,
        checkpoint_selection="frozen-earliest-eligible-complete-focus-development",
    )


def audit_payload(raw, candidate):
    expected = candidate["independent_raw_audit_sha256"]
    require(0 < len(raw) <= CAP and sha(raw) == expected, "fixed independent fit audit differs")
    return dict(sha256=expected, size=len(raw), base64=base64.b64encode(raw).decode())


def decode_audit(record, candidate, fit):
    expected_sha = candidate["independent_raw_audit_sha256"]
    require(
        isinstance(record, dict)
        and set(record) == {"sha256", "size", "base64"}
        and record["sha256"] == expected_sha
        and type(record["size"]) is int
        and 0 < record["size"] <= CAP
        and isinstance(record["base64"], str)
        and len(record["base64"]) <= 4 * ((CAP + 2) // 3),
        "unreviewed fit audit",
    )
    raw = base64.b64decode(record["base64"], validate=True)
    require(0 < len(raw) == record["size"] <= CAP and sha(raw) == expected_sha, "fit audit bytes differ")
    report = json.loads(raw)
    require(raw == canonical(report) + b"\n", "fit audit must preserve canonical JSON plus LF")
    expected = dict(
        schema="quest-sdsc-student-focus-raw-audit-v1",
        preparation_complete=True,
        raw_replay_passed=True,
        mode="fit",
        job_id=candidate["fit_job_id"],
        plan_sha256=candidate["fit_plan_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        auditor_sha256=fit["control_sha256"]["tools/sdsc_student_focus_audit.py"],
        protocol_sha256=fit["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=fit["protocol"]["artifact_sha256"],
        implementation_commit=fit["protocol"]["implementation_commit"],
        acceptance_commit=fit["protocol"]["acceptance_commit"],
        prompts_replayed=1536,
        responses_replayed=18432,
        execution_acceptance_claim=False,
        gpu_numerics_independently_recomputed=False,
    )
    require(
        all(canonical(report.get(key)) == canonical(value) for key, value in expected.items())
        and all(report.get(key) is False for key in FLAGS),
        "fit raw replay identity/counts differ",
    )
    domain = qualification_domain()
    expected_training = dict(
        training_order_reconstructed=True,
        training_rows_reconstructed=2048,
        consumed_training_rows=2048,
        optimizer_windows_replayed=32,
        full_population_input_tokens=1665064,
        training_input_tokens=1665064,
        checkpoint_cumulative_input_tokens=domain.PARENT_CHECKPOINT_INPUT_TOKENS,
        gpu_training_independently_recomputed=False,
    )
    require(
        canonical(report.get("training_evidence")) == canonical(expected_training),
        "complete independently reconstructed fit training evidence differs",
    )
    isolation = report.get("isolation_evidence")
    require(
        report.get("dataset_sha256") == domain.preparation.DATASET_MANIFEST_SHA256
        and isinstance(isolation, dict)
        and set(isolation)
        == {
            "producer_evidence_sha256",
            "dataset_manifest_sha256",
            "producer_isolation_consistency_verified",
            "historical_isolation_independently_recomputed",
        }
        and isinstance(isolation.get("producer_evidence_sha256"), str)
        and re.fullmatch("[a-f0-9]{64}", isolation["producer_evidence_sha256"])
        and isolation["dataset_manifest_sha256"] == report["dataset_sha256"]
        and isolation["producer_isolation_consistency_verified"] is True
        and isolation["historical_isolation_independently_recomputed"] is False,
        "fit audit isolation/population evidence differs",
    )
    selected = report.get("selected_checkpoint")
    require(
        isinstance(selected, dict)
        and selected.get("step") == candidate["checkpoint_step"]
        and selected.get("checkpoint_sha256") == candidate["checkpoint_sha256"],
        "fit raw replay has no frozen selected checkpoint",
    )
    require(
        _fit.select_development(fit, report["development"]) == selected,
        "fit audit is not the complete earliest eligible selection",
    )
    return raw, report


def validate_fit_evidence(report, publication, fit, audit_record, candidate):
    """Admit exactly the audited selected dense artifact, regardless of its index."""
    require(
        fit["task"] == _fit.TASK and fit["mode"] == "fit",
        "parent must be the accepted focus fit",
    )
    records = _fit.publication_records(publication, fit, candidate["fit_job_id"])
    _fit.validate_worker_report(report, fit, candidate["fit_job_id"], records)
    require(
        report["optimizer_steps"] == 32 and report["training_input_tokens"] == 1665064,
        "complete focus fit updates/tokens required",
    )
    _, audited = decode_audit(audit_record, candidate, fit)
    require(
        report["development"] == audited["development"]
        and report["selected_checkpoint"] == audited["selected_checkpoint"],
        "audited fit selection differs",
    )
    domain = qualification_domain()
    require(
        canonical(report.get("data_audit", {}).get("token_envelope"))
        == canonical(
            dict(
                fit_rows=2048,
                optimizer_windows=32,
                window_input_tokens=list(domain.PARENT_WINDOW_INPUT_TOKENS),
                total_input_tokens=1665064,
                token_budget=2000000,
            )
        )
        and report.get("dataset_sha256") == audited["dataset_sha256"]
        and records.get("prepare-data-isolation.json", {}).get("sha256")
        == audited["isolation_evidence"]["producer_evidence_sha256"],
        "fit audit training/isolation differs from published report",
    )
    selected = report["selected_checkpoint"]
    matches = [row for row in report["checkpoints"] if row["step"] == selected["step"]]
    require(len(matches) == 1, "selected checkpoint must occur exactly once")
    checkpoint = matches[0]
    expected = checkpoint_row(candidate)
    require(
        {key: checkpoint[key] for key in ("path", "size", "sha256")} == expected
        and checkpoint["sha256"] == selected["checkpoint_sha256"]
        and expected["path"] == f"checkpoints/step-{candidate['checkpoint_step']:08d}.pt"
        and type(expected["size"]) is int
        and 0 < expected["size"] <= 8 * 1024**3,
        "frozen selected checkpoint differs",
    )
    reloads = checkpoint["reload_by_rank"]
    require(
        [row.get("rank") for row in reloads] == [0, 1]
        and all(row.get("model_state_sha256") == candidate["master_model_state_sha256"] for row in reloads)
        and re.fullmatch("[a-f0-9]{64}", candidate["master_model_state_sha256"]),
        "selected checkpoint rank master hashes differ",
    )
    rows = {row["path"]: row for row in publication["large_files"]}
    require(rows.get(expected["path"]) == expected, "selected dense checkpoint publication differs")
    return checkpoint


def validate_plan(plan):
    candidate = prepared_initial(plan["protocol"])
    require(
        len(canonical(plan)) <= MAX_PLAN
        and plan["schema"] == "quest-sdsc-student-focus-qualification-plan-v1"
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(resources()),
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
        "tools/sdsc_student_focus_qualify.py",
        "tools/sdsc_student_focus_qualify_job.py",
        "tools/sdsc_student_focus_qualify_worker.py",
        "tools/sdsc_student_focus_qualify_audit.py",
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
        submission_dir=CONTROL / "student-focus-qualify-submissions" / plan["intent_id"],
        claim=CONTROL / "student-focus-qualify-claims" / (plan["intent_id"] + ".json"),
        scientific_claim=scientific_claim_path(plan),
        result_dir=PROJECT / "student-focus-qualify" / plan["intent_id"],
    ).items():
        require(plan[k] == str(v), "unreviewed plan path: " + k)
    require(plan["job_name"] == "opd-sfq-" + plan["intent_id"], "job name differs")
    fit = plan["fit"]
    require(
        set(fit)
        == {"plan", "plan_sha256", "job_id", "publication_sha256", "report_sha256", "audit", "completion"}
        and fit["job_id"] == candidate["fit_job_id"]
        and fit["plan_sha256"] == candidate["fit_plan_sha256"] == sha(canonical(fit["plan"]))
        and fit["plan"]["intent_id"] == candidate["fit_intent"]
        and fit["publication_sha256"] == candidate["fit_publication_sha256"]
        and fit["report_sha256"] == candidate["fit_report_sha256"],
        "frozen fit provenance differs",
    )
    _fit.validate_plan(fit["plan"])
    decode_completion_records(fit["completion"], candidate, fit["plan"])
    validate_parent_binding(plan["protocol"], fit["plan"])
    require(
        fit["plan"]["mode"] == "fit" and plan["prepared_checkpoint"] == checkpoint_row(candidate),
        "frozen selected checkpoint differs",
    )
    decode_audit(fit["audit"], candidate, fit["plan"])
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
    candidate = prepared_initial(plan["protocol"])
    fit = plan["fit"]["plan"]
    frozen = decode_completion_records(plan["fit"]["completion"], candidate, fit)
    directory = safe(fit["submission_dir"])
    require(document(directory / "plan.json") == fit, "stored fit plan differs")
    claim = document(safe(fit["claim"]))
    require(
        claim.get("intent_id") == fit["intent_id"]
        and claim.get("plan_sha256") == candidate["fit_plan_sha256"],
        "fit execution claim differs",
    )
    require(
        document(safe(fit["scientific_claim"])) == _fit.scientific_claim_record(fit),
        "fit scientific claim differs",
    )
    receipt = document(directory / "receipt.json")
    _fit.validate_submission_receipt(receipt, fit)
    require(receipt["job_id"] == candidate["fit_job_id"], "fit receipt job differs")
    # Admission freshly queries Slurm. The GPU node reuses this exact admission
    # accounting snapshot and replays every persistent raw execution validator.
    saved = None if accounting else document(safe(plan["submission_dir"]) / "admission.json")["fit"]
    result = _fit.inspect_job(fit, receipt, accounting_snapshot=saved)
    result = dict(result, plan_sha256=candidate["fit_plan_sha256"])
    root = safe(fit["result_dir"])
    current = completion_records(result, read(root / "receipt.json"))
    decoded = decode_completion_records(current, candidate, fit)
    require(decoded["publication"] == frozen["publication"], "fit immutable publication changed")
    report = document(root / "prepare-report.json", candidate["fit_report_sha256"])
    validate_fit_evidence(report, result["publication"], fit, plan["fit"]["audit"], candidate)
    return result


def validate_worker_report(report, plan, job, records, *, require_passed=True):
    candidate = prepared_initial(plan["protocol"])
    expected = dict(
        schema="quest-sdsc-student-focus-qualification-report-v1",
        job_id=job,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        prepared_checkpoint_sha256=candidate["checkpoint_sha256"],
        prepared_initial=candidate,
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        preparation_protocol_sha256=plan["fit"]["plan"]["protocol"]["protocol_sha256"],
        preparation_protocol_artifact_sha256=plan["fit"]["plan"]["protocol"]["artifact_sha256"],
        execution_complete=True,
    )
    require(
        all(canonical(report.get(k)) == canonical(v) for k, v in expected.items())
        and all(report.get(k) is False for k in FLAGS),
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
        loading.get("prepared_checkpoint_sha256") == candidate["checkpoint_sha256"]
        and loading.get("exact_saved_master_reload")
        == dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True)
        and loading.get("forward_parameter_dtype") == "torch.bfloat16"
        and loading.get("all_parameters_frozen") is True
        and loading.get("gradient_checkpointing") is False
        and loading.get("use_cache") is False
        and loading.get("master_model_state_sha256") == candidate["master_model_state_sha256"]
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
        fit.get("fit_job_id") == candidate["fit_job_id"]
        and fit.get("completion") == completion_summary(plan["fit"]["completion"], plan["fit"]["plan"])
        and fit.get("publication_sha256") == candidate["fit_publication_sha256"]
        and fit.get("report_sha256") == candidate["fit_report_sha256"]
        and fit.get("independent_audit_sha256") == candidate["independent_raw_audit_sha256"]
        and fit.get("checkpoint_path") == candidate["checkpoint_path"]
        and fit.get("checkpoint_size") == candidate["checkpoint_size"]
        and fit.get("master_model_state_sha256") == candidate["master_model_state_sha256"]
        and fit.get("selected_checkpoint")
        == decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])[1]["selected_checkpoint"],
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
        anti["model_checkpoint_hash"] == candidate["checkpoint_sha256"]
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
        schema="quest-sdsc-student-focus-qualification-publication-v1",
        task=TASK,
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        persistent_read_back_verified=True,
    )
    require(
        all(canonical(publication.get(k)) == canonical(v) for k, v in expected.items())
        and all(publication.get(k) is False for k in FLAGS),
        "publication identity/scope differs",
    )
    require(
        type(publication.get("passed")) is bool
        and type(publication.get("stage_complete")) is bool
        and publication["passed"] is publication["stage_complete"]
        and publication.get("qualification_passed") is publication["stage_complete"],
        "publication completion flags differ",
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


def validate_parent_fetch(fetch, publication, fit, candidate):
    """Parent preparation has a larger raw envelope than this W1 qualification."""
    rows = _fit.publication_records(publication, fit, candidate["fit_job_id"])
    total = 0
    for name, row in rows.items():
        data = _fit.read(fetch / name)
        total += len(data)
        require(total <= _fit.MAX_FETCH, "parent raw evidence exceeds fetch bound")
        require(len(data) == row["size"] and sha(data) == row["sha256"], "local fit evidence differs")
    return rows


def validate_completed_fetch(fetch, fit, status, candidate):
    """Revalidate the flat published raw startup/numerical/memory evidence locally."""
    publication_raw = read(fetch / "receipt.json")
    bundle = completion_records(status, publication_raw)
    decode_completion_records(bundle, candidate, fit)
    publication = json.loads(publication_raw)
    rows = validate_parent_fetch(fetch, publication, fit, candidate)
    required = _fit.required_raw_names("fit") | {
        "prepare-report.json",
        "node-result.json",
        "memory.json",
        *_fit.EXECUTION_NAMES,
    }
    require(required <= set(rows), "fit execution inventory incomplete")
    report = document(fetch / "prepare-report.json", candidate["fit_report_sha256"])
    _fit.validate_worker_report(report, fit, candidate["fit_job_id"], rows)
    _fit.validate_update_records(_fit.read(fetch / "prepare-updates.jsonl"), report)
    _fit.validate_large_outputs(report, publication["large_files"])
    _fit.validate_memory(document(fetch / "memory.json"), candidate["fit_job_id"])
    _fit.validate_startup_evidence(fit, fetch, candidate["fit_job_id"])
    node = document(fetch / "node-result.json")
    require(
        node.get("job_id") == candidate["fit_job_id"]
        and node.get("plan_sha256") == candidate["fit_plan_sha256"]
        and node.get("mode") == "fit"
        and node.get("stage_complete") is True
        and type(node.get("exit_code")) is int
        and node["exit_code"] == 0
        and node.get("large_artifacts") == publication["large_files"]
        and all(node.get(key) is False for key in FLAGS),
        "fit execution node differs",
    )
    return bundle


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
    protocol = protocol_binding(ROOT, args.science_git_head)
    candidate = prepared_initial(protocol)
    fit = _fit.validate_plan(document(safe(args.fit_plan.absolute())))
    require(sha(canonical(fit)) == candidate["fit_plan_sha256"], "exact completed fit plan required")
    fetch = safe(args.fit_fetch_dir.absolute())
    report = document(fetch / "prepare-report.json", candidate["fit_report_sha256"])
    publication = document(fetch / "receipt.json", candidate["fit_publication_sha256"])
    validate_parent_fetch(fetch, publication, fit, candidate)
    status = document(safe(args.fit_status_file.absolute()))
    status = status.get("status", status)
    require(
        status["job_id"] == candidate["fit_job_id"]
        and status["success"] is True
        and status["stage_complete"] is True
        and status["preparation_complete"] is True
        and status["publication_sha256"] == candidate["fit_publication_sha256"]
        and status["result_sha256"] == candidate["fit_report_sha256"],
        "verified terminal fit status required",
    )
    completed = validate_completed_fetch(fetch, fit, status, candidate)
    audit = audit_payload(read(safe(args.fit_audit.absolute()), CAP), candidate)
    validate_fit_evidence(report, publication, fit, audit, candidate)
    plan = dict(
        schema="quest-sdsc-student-focus-qualification-plan-v1",
        task=TASK,
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        resources=resources(),
        fit=dict(
            plan=fit,
            plan_sha256=candidate["fit_plan_sha256"],
            job_id=candidate["fit_job_id"],
            publication_sha256=candidate["fit_publication_sha256"],
            report_sha256=candidate["fit_report_sha256"],
            audit=audit,
            completion=completed,
        ),
        prepared_checkpoint=checkpoint_row(candidate),
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
        submission_dir=str(CONTROL / "student-focus-qualify-submissions" / intent),
        claim=str(CONTROL / "student-focus-qualify-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-focus-qualify" / intent),
        job_name="opd-sfq-" + intent,
    )
    plan["scientific_claim"] = str(scientific_claim_path(plan))
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-focus-qualify" / intent
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
        "from posttrain_circuits.experiments.protocols.student_focus_qualification "
        "import resolve_student_focus_qualification_protocol;"
        "v=resolve_student_focus_qualification_protocol(root,require_accepted=True,"
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
        CONTROL
        / "student-focus-qualify-scientific-claims"
        / (sha(canonical(plan["science_identity"])) + ".json")
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
        with tempfile.TemporaryDirectory(prefix="student-focus-qualify-verify-") as temp:
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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_focus_qualify_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_focus_qualify_job.py"],
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
    path = str(Path(plan["release"]) / "source/tools/sdsc_student_focus_qualify.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p),"remote"];runpy.run_path(str(p),run_name="__main__")'
    )
    result = cli.ssh_call(
        [
            plan["python"],
            "-I",
            "-B",
            "-c",
            launch,
            path,
            plan["control_sha256"]["tools/sdsc_student_focus_qualify.py"],
        ],
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
    for name in (
        "fit-fetch-dir",
        "fit-status-file",
        "fit-plan",
        "fit-audit",
        "plan",
        "dry-run-file",
    ):
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
                path.parent.parent == ROOT / ".sdsc/student-focus-qualify" and path.name == "plan.json",
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
