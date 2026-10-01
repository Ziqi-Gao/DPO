#!/usr/bin/env python3
"""Bounded prepared-student preflight/fit control. No retry and no G0 acceptance.

First use tools/sdsc check and sync --dry-run/sync. Prepare binds a genuine
accepted successor Git export. Submit requires its matching remote dry-run;
missing acknowledgement is reconciled against permanent claims, never retried.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
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
TASK = "qwen3-v2-student-preparation-v1"
CAP = 1024**2
MAX_FILE = 16 * CAP
MAX_FETCH = 48 * CAP
MAX_PLAN = 4 * CAP
RUNTIME_PACKAGES = {
    "accelerate": "1.10.1",
    "datasets": "4.0.0",
    "huggingface-hub": "0.36.2",
    "matplotlib": "3.10.5",
    "nvidia-nccl-cu12": "2.27.3",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pandas": "2.3.2",
    "pyarrow": "21.0.0",
    "pydantic": "2.11.7",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "scipy": "1.16.1",
    "statsmodels": "0.14.6",
    "tabulate": "0.9.0",
    "tokenizers": "0.22.0",
    "torch": "2.8.0+cu128",
    "transformer-lens": "2.16.1",
    "transformers": "4.56.2",
}
PROTOCOL_PATH = "prereg/amendments/qwen3_student_preparation_v1.json"
PROTOCOL_MODULE = "src/posttrain_circuits/experiments/protocols/student_preparation.py"
FROZEN = {
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
}
TOOLS = (
    "tools/sdsc_student_prepare.py",
    "tools/sdsc_student_prepare_job.py",
    "tools/sdsc_student_prepare_worker.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    *FROZEN,
    "tools/sdsc_provenance.py",
    "tools/sdsc_student_lr_probe.py",
    "tools/sdsc_student_quality_probe.py",
    PROTOCOL_PATH,
    PROTOCOL_MODULE,
)
NAMES = (
    "prepare-report.json",
    "prepare-updates.jsonl",
    "prepare-dev-records-rank-0.jsonl",
    "prepare-dev-records-rank-1.jsonl",
    "prepare-dev-prompts.jsonl",
    "prepare-dataset-manifest.json",
    "prepare-data-isolation.json",
    "prepare-token-audit.json",
    "prepare-preflight-records-rank-0.jsonl",
    "prepare-preflight-records-rank-1.jsonl",
    "node-result.json",
    "memory.json",
)
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)


def helper(name, root=ROOT):
    relative = "tools/" + name + ".py"
    path = root / relative
    if relative in FROZEN and hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN[relative]:
        raise ValueError("immutable shared helper changed: " + relative)
    spec = importlib.util.spec_from_file_location("_prepare_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_io = helper("sdsc_student_quality")
_lr = helper("sdsc_student_lr")
_initial = helper("sdsc_student_initial")
require, canonical, sha, now = _io.require, _io.canonical, _io.sha, _io.now
safe, relative, write_once = _io.safe, _io.relative, _io.write_once


# Only this new transport widens complete raw-result reads. Historical helpers
# and their original callers retain the original immutable eight-MiB bound.
def read(path, limit=MAX_FILE):
    return _io.read(path, limit=limit)


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "evidence SHA differs")
    return json.loads(raw)


run = _lr.run
job_queue, job_accounting = _lr.job_queue, _lr.job_accounting
validate_accounting, live_binding = _lr.validate_accounting, _lr.live_binding
validate_memory, validate_job_memory_events = _lr.validate_memory, _lr.validate_job_memory_events
PARENT_JOB, PARENT_INTENT = _lr.PARENT_JOB, _lr.PARENT_INTENT
PARENT_REPORT_SHA, INITIAL_SHA, CONFIG_SHA = _lr.PARENT_REPORT_SHA, _lr.INITIAL_SHA, _lr.CONFIG_SHA


def resources(mode):
    require(mode in {"preflight", "fit"}, "unknown preparation mode")
    return dict(
        account="nwu181",
        partition="nairr-gpu-shared",
        qos="nairr-gpu-shared-normal",
        gpu_type="h100",
        gpus=2,
        cpus=24,
        mem_gib=384,
        time="01:00:00" if mode == "preflight" else "04:00:00",
    )


def protocol_binding(root, expected_head=None):
    # A fresh interpreter prevents imports from another restored checkout being
    # reused as though they came from this genuine accepted scientific tree.
    code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "sys.path.insert(0,str(root/'src'));"
        "from posttrain_circuits.experiments.protocols.student_preparation "
        "import resolve_student_preparation_protocol;"
        "v=resolve_student_preparation_protocol(root,require_accepted=True,expected_head=sys.argv[2] or None,"
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


def science_identity(mode, protocol):
    return dict(
        task=TASK,
        mode=mode,
        protocol_sha256=protocol["protocol_sha256"],
        initial_sha256=INITIAL_SHA,
        seed=42,
        global_batch_size=64,
        optimizer_steps=4 if mode == "preflight" else 32,
        objective="independent-canonical-proof-common-initial",
        learning_rate=5e-5,
    )


def scientific_claim_path(plan):
    return (
        CONTROL / "student-prepare-scientific-claims" / (sha(canonical(plan["science_identity"])) + ".json")
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


def validate_plan(plan):
    require(
        len(canonical(plan)) <= MAX_PLAN
        and plan["schema"] == "quest-sdsc-student-prepare-plan-v1"
        and plan["task"] == TASK
        and plan["resources"] == resources(plan["mode"]),
        "plan scope/resources differ",
    )
    require(
        plan["science_identity"] == science_identity(plan["mode"], plan["protocol"]),
        "science identity differs",
    )
    require(
        set(plan["control_sha256"]) == set(TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in plan["control_sha256"].values())
        and all(plan["control_sha256"][p] == h for p, h in FROZEN.items()),
        "control pins differ",
    )
    science_files = plan["protocol"].get("science_file_sha256")
    executed = {
        "tools/sdsc_student_prepare.py",
        "tools/sdsc_student_prepare_job.py",
        "tools/sdsc_student_prepare_worker.py",
        "tools/sdsc_student_lr_probe.py",
        "tools/sdsc_student_quality_probe.py",
        "tools/sdsc_provenance.py",
        PROTOCOL_MODULE,
    }
    require(
        isinstance(science_files, dict)
        and executed <= set(science_files)
        and all(re.fullmatch("[a-f0-9]{64}", value) for value in science_files.values()),
        "accepted science inventory lacks executed controls",
    )
    require(
        all(
            plan["control_sha256"][name] == digest
            for name, digest in science_files.items()
            if name in plan["control_sha256"]
        ),
        "deployed control differs from accepted scientific byte",
    )
    require(plan["intent_id"] == execution_intent(plan), "execution intent differs")
    require(
        re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["run_id"])
        and re.fullmatch("[a-f0-9]{64}", plan["code_sha256"]),
        "invalid release identity",
    )
    for key, expected in dict(
        release=CONTROL / "releases" / plan["run_id"],
        submission_dir=CONTROL / "student-prepare-submissions" / plan["intent_id"],
        claim=CONTROL / "student-prepare-claims" / (plan["intent_id"] + ".json"),
        result_dir=PROJECT / "student-prepare" / plan["intent_id"],
        scientific_claim=scientific_claim_path(plan),
    ).items():
        require(plan[key] == str(expected), "unreviewed plan path: " + key)
    require(plan["job_name"] == "opd-spr-" + plan["intent_id"], "job name differs")
    receipt = plan["parent"]["receipt"]
    require(
        receipt["job_id"] == PARENT_JOB
        and receipt["intent_id"] == PARENT_INTENT
        and receipt["submission_dir"] == str(CONTROL / "submissions" / PARENT_INTENT)
        and receipt["science_git_head"] == _initial.SCIENCE_HEAD
        and receipt["bundle_sha256"] == _initial.PARENT_BUNDLE_SHA
        and plan["parent"]["report_sha256"] == PARENT_REPORT_SHA
        and plan["parent"]["publication_sha256"] == _initial.PARENT_PUBLICATION_SHA,
        "unreviewed parent provenance",
    )
    require(
        plan["checkpoints"]
        == [
            dict(label="initial", path="artifacts/initial_checkpoint.pt", size=3441276375, sha256=INITIAL_SHA)
        ]
        and plan["resolved_config"]
        == dict(path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=CONFIG_SHA),
        "original initial/config differs",
    )
    contract = helper("sdsc_student_contract")
    expected_dataset = [
        dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES
    ]
    require(plan["dataset_inputs"] == expected_dataset, "complete original dataset family required")
    require(
        plan["python"]
        == receipt["python"]
        == str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
        and plan["hf_home"] == receipt["hf_home"] == str(PROJECT / "cache/huggingface"),
        "runtime/cache differs",
    )
    provenance = plan["provenance"]
    require(
        set(provenance) == {"directory", "manifest_sha256", "head"}
        and re.fullmatch("[a-f0-9]{64}", provenance["manifest_sha256"])
        and provenance["directory"] == str(CONTROL / "provenance" / provenance["manifest_sha256"])
        and re.fullmatch("[a-f0-9]{40}", provenance["head"])
        and provenance["head"] == plan["protocol"]["head"],
        "successor genuine provenance differs",
    )
    require(
        plan["protocol"]["artifact_sha256"] == plan["control_sha256"][PROTOCOL_PATH]
        and plan["protocol"]["implementation_commit"] != plan["protocol"]["acceptance_commit"]
        and all(
            re.fullmatch("[a-f0-9]{40}", plan["protocol"][key])
            for key in ("implementation_commit", "acceptance_commit")
        ),
        "distinct accepted science binding required",
    )
    if plan["mode"] == "fit":
        preflight = plan["preflight"]
        require(
            isinstance(preflight, dict)
            and set(preflight) == {"plan", "plan_sha256"}
            and preflight["plan"]["mode"] == "preflight"
            and preflight["plan_sha256"] == sha(canonical(preflight["plan"])),
            "matching preflight missing",
        )
        validate_plan(preflight["plan"])
        for key in ("control_sha256", "protocol", "provenance", "code_sha256"):
            require(plan[key] == preflight["plan"][key], "fit/preflight implementation differs: " + key)
    else:
        require(plan["preflight"] is None, "preflight cannot resume another stage")
    require(all(plan.get(key) is False for key in FLAGS), "preparation cannot claim scientific acceptance")
    return plan


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
        with tempfile.TemporaryDirectory(prefix="student-prepare-verify-") as temp:
            return verify_science(plan, Path(temp) / "science")
    restored = helper("sdsc_provenance").verify(
        safe(provenance["directory"]), provenance["manifest_sha256"], plan["code_sha256"], destination
    )
    require(restored["git_head"] == provenance["head"], "restored genuine HEAD differs")
    binding = protocol_binding(destination, provenance["head"])
    require(binding == plan["protocol"], "restored accepted successor differs")
    return restored


def verify_parent(plan, *, accounting=False):
    return _initial.verify_parent(plan, accounting=accounting)


def verify_preflight(plan, *, accounting=True):
    if plan["mode"] == "preflight":
        return None
    previous = plan["preflight"]["plan"]
    directory = safe(previous["submission_dir"])
    require(
        read(directory / "plan.json") == canonical(previous)
        and document(safe(previous["scientific_claim"])) == scientific_claim_record(previous)
        and document(safe(previous["claim"]))["plan_sha256"] == sha(canonical(previous)),
        "preflight permanent claims differ",
    )
    receipt = document(directory / "receipt.json")
    validate_submission_receipt(receipt, previous)
    saved = None if accounting else document(safe(plan["submission_dir"]) / "admission.json")["preflight"]
    result = inspect_job(previous, receipt, accounting_snapshot=saved)
    require(
        result["success"] is True and result["stage_complete"] is True,
        "real matching preflight has not passed",
    )
    return dict(result, plan_sha256=sha(canonical(previous)))


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
    require(total + 2 <= 4, "new preparation would exceed four allocatable GPUs")
    return dict(existing_allocatable_gpus=total, new_gpus=2, limit=4, jobs=jobs, query=query)


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
        "--gpus=h100:2",
        "--mem=393216M",
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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_prepare_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_prepare_job.py"],
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
        mode=plan["mode"],
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


def validate_worker_report(report, plan, job, records):
    require(
        report.get("schema") == "quest-sdsc-student-prepare-report-v1"
        and report.get("mode") == plan["mode"]
        and report.get("passed") is True
        and report.get("execution_complete") is True
        and report.get("preparation_complete") is (plan["mode"] == "fit")
        and all(report.get(k) is False for k in FLAGS),
        "incomplete/overclaiming preparation report",
    )
    expected = dict(
        job_id=job,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        initial_checkpoint_sha256=INITIAL_SHA,
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        optimizer_steps=4 if plan["mode"] == "preflight" else 32,
        global_batch_size=64,
        world_size=2,
        learning_rate=5e-5,
    )
    require(
        all(report.get(k) == v for k, v in expected.items()), "worker scientific/execution binding differs"
    )
    require(
        report.get("data_audit", {}).get("passed") is True
        and report.get("checkpoint_restore", {}).get("passed") is True,
        "data or real full-state restore failed",
    )
    isolation = report["data_audit"].get("isolation", {})
    envelope = report["data_audit"].get("token_envelope", {})
    windows = envelope.get("window_input_tokens")
    require(
        isolation.get("passed") is True
        and isolation.get("counts") == {"original_family": 144000, "teacher_fit": 8192, "teacher_dev": 512}
        and isolation.get("dimensions") == ["semantic", "example_id", "pair_group_id", "pair_seed"]
        and isolation.get("preparation_manifest_sha256")
        == "274e5f303d1a44b69790db687fc9fdde43bfae5c27cc2e5c5f656cdd0bf28cc6",
        "complete disjoint preparation evidence missing",
    )
    require(
        isinstance(windows, list)
        and len(windows) == 32
        and all(type(value) is int and 0 < value <= 64 * 1536 for value in windows)
        and envelope.get("fit_rows") == 2048
        and envelope.get("optimizer_windows") == 32
        and envelope.get("token_budget") == 2000000
        and envelope.get("total_input_tokens") == sum(windows) <= 2000000
        and type(report.get("training_input_tokens")) is int
        and report["training_input_tokens"] == sum(windows[: report["optimizer_steps"]]),
        "global nonpadding input token accounting differs",
    )
    restored = report["checkpoint_restore"].get("ranks")
    require(
        isinstance(restored, list)
        and [row.get("rank") for row in restored] == [0, 1]
        and all(
            row.get("passed") is True
            and row.get("step") == 4
            and row.get("actual_accelerate_save_load") is True
            and all(
                re.fullmatch("[a-f0-9]{64}", row.get(key, ""))
                for key in ("model_sha256", "optimizer_sha256", "runtime_sha256")
            )
            for row in restored
        ),
        "actual two-rank full-state restore evidence missing",
    )
    checkpoints = report.get("checkpoints")
    steps = [4] if plan["mode"] == "preflight" else [4, 8, 16, 32]
    require(
        isinstance(checkpoints, list) and [row.get("step") for row in checkpoints] == steps,
        "fixed checkpoint schedule differs",
    )
    for row in checkpoints:
        require(
            row.get("path") == f"checkpoints/step-{row['step']:08d}.pt"
            and row.get("scope") == "common_student_preparation_model_only"
            and type(row.get("size")) is int
            and 0 < row["size"] <= 16 * 1024**3
            and re.fullmatch("[a-f0-9]{64}", row.get("sha256", "")),
            "dense checkpoint identity differs",
        )
        reloads = row.get("reload_by_rank")
        require(
            isinstance(reloads, list) and [r.get("rank") for r in reloads] == [0, 1],
            "both rank reloads required",
        )
        for item in reloads:
            loaded = item.get("exact_saved_master_reload", {})
            parity = item.get("root_export_logits")
            require(
                loaded.get("all_tensors_exact") is True
                and loaded.get("loaded_before_bf16_copy") is True
                and type(loaded.get("key_count")) is int
                and loaded["key_count"] > 0
                and isinstance(parity, list)
                and len(parity) == 2
                and all(
                    v.get("bitwise_equal") is True
                    and v.get("max_abs_error") == 0
                    and v.get("rms_error") == 0
                    and v.get("argmax_mismatches") == 0
                    for v in parity
                ),
                "actual saved-model reload/logit parity failed",
            )
    require(
        report.get("teacher_data_required") is False and report.get("source_kind") == "symbolic_canonical",
        "preparation source must be independent symbolic proofs",
    )
    development = report.get("development")
    selected = None
    if plan["mode"] == "preflight":
        require(
            development == [] and report.get("selected_checkpoint") is None,
            "preflight may not measure/select development",
        )
    else:
        require(
            isinstance(development, list) and len(development) == 4, "all four development reports required"
        )
        fields = {
            "step",
            "checkpoint_sha256",
            "examples_sha256",
            "num_examples",
            "answer_correct",
            "proof_correct",
            "format_valid",
        }
        for step, row, checkpoint in zip(steps, development, checkpoints, strict=True):
            require(
                set(row) == fields
                and type(row["step"]) is int
                and row["step"] == step
                and type(row["num_examples"]) is int
                and row["num_examples"] == 512
                and row["checkpoint_sha256"] == checkpoint["sha256"]
                and row["examples_sha256"]
                == "7722e04c365cd122b7ea9454f34062a7aced5e3a67a633b7048cee3aa9333690"
                and all(type(row[key]) is int for key in ("proof_correct", "answer_correct", "format_valid"))
                and 0 <= row["proof_correct"] <= row["answer_correct"] <= row["format_valid"] <= 512,
                "development population/strict counts differ",
            )
            if selected is None and 52 <= row["answer_correct"] <= 307:
                selected = row
        require(
            selected is not None and canonical(report.get("selected_checkpoint")) == canonical(selected),
            "fit selection is not the earliest eligible checkpoint",
        )
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list) and len(raw) >= 1 and len({r["path"] for r in raw}) == len(raw),
        "raw inventory missing",
    )
    for row in raw:
        require(row["path"] in records and row == records[row["path"]], "raw artifact inventory differs")


def publication_records(publication, plan, job):
    expected = dict(
        task=TASK,
        mode=plan["mode"],
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
    )
    require(
        all(publication.get(k) == v for k, v in expected.items())
        and all(publication.get(k) is False for k in FLAGS),
        "publication identity/scope differs",
    )
    rows = publication["files"]
    require(isinstance(rows, list) and len(rows) <= len(NAMES), "invalid publication inventory")
    records = {}
    for row in rows:
        name = relative(row["path"])
        require(
            name in NAMES
            and name not in records
            and set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 <= row["size"] <= MAX_FILE
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid small result",
        )
        records[name] = row
    require(sum(r["size"] for r in rows) <= MAX_FETCH - CAP, "small output exceeds bound")
    large = publication["large_files"]
    require(
        isinstance(large, list) and len(large) <= 32 and len({r["path"] for r in large}) == len(large),
        "large inventory differs",
    )
    for row in large:
        name = relative(row["path"])
        require(
            name.startswith(("checkpoints/", "resume/"))
            and set(row) == {"path", "size", "sha256"}
            and type(row["size"]) is int
            and 0 < row["size"] <= 32 * 1024**3
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid checkpoint inventory",
        )
    require(sum(r["size"] for r in large) <= 96 * 1024**3, "checkpoint publication exceeds bound")
    return records


def inspect_job(plan, receipt, *, accounting_snapshot=None):
    job = receipt["job_id"]
    directory = safe(plan["submission_dir"])
    observed = (
        document(directory / "live-binding.json") if (directory / "live-binding.json").exists() else None
    )
    if accounting_snapshot is None:
        queue, account = job_queue(job), job_accounting(job)
    else:
        require(
            accounting_snapshot["job_id"] == job
            and accounting_snapshot["plan_sha256"] == sha(canonical(plan)),
            "saved prerequisite accounting identity differs",
        )
        queue, account = accounting_snapshot["queue"], accounting_snapshot["accounting"]
    result = validate_accounting(plan, job, queue, account, observed)
    result.update(
        success=False,
        stage_complete=False,
        preparation_complete=False,
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
        require(
            {name for name in NAMES if (root / name).exists()} == set(records), "published inventory differs"
        )
        for name, row in records.items():
            data = read(root / name)
            require(len(data) == row["size"] and sha(data) == row["sha256"], "persistent result hash differs")
        for row in published["large_files"]:
            info = safe(root / row["path"]).lstat()
            require(
                stat.S_ISREG(info.st_mode) and info.st_size == row["size"],
                "persistent checkpoint size/type differs",
            )
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=published,
            publication_sha256=sha(raw),
        )
    if result["accounting_complete"]:
        required = {
            "prepare-report.json",
            "prepare-updates.jsonl",
            "prepare-dataset-manifest.json",
            "prepare-data-isolation.json",
            "prepare-token-audit.json",
            "node-result.json",
            "memory.json",
        }
        if plan["mode"] == "fit":
            required |= {
                "prepare-dev-prompts.jsonl",
                "prepare-dev-records-rank-0.jsonl",
                "prepare-dev-records-rank-1.jsonl",
            }
        else:
            required |= {"prepare-preflight-records-rank-0.jsonl", "prepare-preflight-records-rank-1.jsonl"}
        require(
            result["publication_verified"]
            and required <= set(records)
            and published["passed"] is True
            and published["stage_complete"] is True
            and len(published["large_files"]) > 0,
            "completed allocation lacks successful complete publication",
        )
        report = document(root / "prepare-report.json")
        validate_worker_report(report, plan, job, records)
        validate_memory(document(root / "memory.json"), job)
        node = document(root / "node-result.json")
        require(
            node.get("stage_complete") is True
            and node.get("exit_code") == 0
            and node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and node.get("large_artifacts") == published["large_files"]
            and all(node.get(k) is False for k in FLAGS),
            "node evidence differs",
        )
        result.update(
            success=True,
            stage_complete=True,
            preparation_complete=plan["mode"] == "fit",
            result_sha256=sha(read(root / "prepare-report.json")),
        )
    return result


def verify_runtime(plan):
    code = (
        "import json,platform,sys;from importlib.metadata import version;"
        "print(json.dumps(dict(python=platform.python_version(),executable=sys.executable,"
        "packages={name:version(name) for name in json.loads(sys.argv[1])})))"
    )
    result = run(
        [plan["python"], "-I", "-B", "-c", code, canonical(list(RUNTIME_PACKAGES)).decode()], timeout=45
    )
    require(result["returncode"] == 0, "fixed runtime metadata check failed")
    observed = json.loads(result["stdout"])
    require(
        observed == dict(python="3.12.13", executable=plan["python"], packages=RUNTIME_PACKAGES),
        "fixed runtime package identity differs",
    )
    return observed


def admission(plan):
    verify_parent(plan, accounting=True)
    verify_science(plan)
    preflight = verify_preflight(plan)
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
    return dict(preflight=preflight, runtime=verify_runtime(plan), gpu_concurrency=check_gpu_ceiling())


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
    path = str(Path(plan["release"]) / "source/tools/sdsc_student_prepare.py")
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
    fetch = safe(args.fetch_dir.absolute())
    report = document(fetch / "adapted-calibration.json", PARENT_REPORT_SHA)
    publication = document(fetch / "receipt.json", _initial.PARENT_PUBLICATION_SHA)
    receipt = document(ROOT / ".sdsc/submissions" / (PARENT_INTENT + ".json"))["receipt"]
    status = document(safe(args.status_file.absolute()))
    require(
        status.get("job_id") == PARENT_JOB
        and status.get("success") is True
        and status["result"]["result_sha256"] == PARENT_REPORT_SHA,
        "parent status not verified",
    )
    checkpoints, config = _lr.checkpoint_rows(publication, report)
    contract = helper("sdsc_student_contract")
    protocol = protocol_binding(ROOT, args.science_git_head)
    plan = dict(
        schema="quest-sdsc-student-prepare-plan-v1",
        task=TASK,
        mode=args.mode,
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        resources=resources(args.mode),
        parent=dict(
            receipt=receipt,
            report_sha256=PARENT_REPORT_SHA,
            publication_sha256=_initial.PARENT_PUBLICATION_SHA,
        ),
        checkpoints=checkpoints,
        resolved_config=config,
        protocol=protocol,
        dataset_inputs=[
            dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES
        ],
        control_sha256={name: records[name]["sha256"] for name in TOOLS},
        provenance=dict(
            directory=args.provenance_dir,
            manifest_sha256=args.provenance_manifest_sha256,
            head=args.science_git_head,
        ),
        science_identity=science_identity(args.mode, protocol),
        python=receipt["python"],
        hf_home=receipt["hf_home"],
        preflight=None,
        **dict.fromkeys(FLAGS, False),
    )
    if args.mode == "fit":
        require(args.preflight_plan is not None, "fit requires matching real preflight plan")
        previous = validate_plan(document(safe(args.preflight_plan.absolute())))
        plan["preflight"] = dict(plan=previous, plan_sha256=sha(canonical(previous)))
    intent = execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(CONTROL / "student-prepare-submissions" / intent),
        claim=str(CONTROL / "student-prepare-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-prepare" / intent),
        job_name="opd-spr-" + intent,
    )
    plan["scientific_claim"] = str(scientific_claim_path(plan))
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-prepare" / intent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        resources=plan["resources"],
        mode=args.mode,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--mode", choices=("preflight", "fit"))
    parser.add_argument("--run-id")
    for name in ("fetch-dir", "status-file", "plan", "preflight-plan", "dry-run-file"):
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
                        args.mode,
                        args.run_id,
                        args.fetch_dir,
                        args.status_file,
                        args.provenance_dir,
                        args.provenance_manifest_sha256,
                        args.science_git_head,
                    )
                ),
                "prepare requires mode/release/parent/provenance",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-prepare" and path.name == "plan.json",
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
