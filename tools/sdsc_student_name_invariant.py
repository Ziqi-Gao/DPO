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
TASK = "qwen3-v2-student-name-invariant-preparation-v1"
CAP = 1024**2
MAX_FILE = 16 * CAP
MAX_FETCH = 224 * CAP
MAX_RESPONSE = 320 * CAP
MAX_PROMPTS = 32 * CAP
MAX_AUDIT = 64 * 1024
RUNTIME_ARTIFACT_ROOT = PROJECT / "runtime-snapshots"
RUNTIME_SNAPSHOT = {
    "archive": {
        "path": (
            "/expanse/lustre/projects/nwu181/zgao12/OPD/runtime-snapshots/"
            "snapshot-2a3797d216384301877dfaf6a398431a/runtime.bin"
        ),
        "sha256": "7151cc2839d518474b3ab968064a14773a43728c1671aea462b53234da1db83d",
        "size": 7825971697,
    },
    "files_count": 49474,
    "manifest": {
        "path": (
            "/expanse/lustre/projects/nwu181/zgao12/OPD/runtime-snapshots/"
            "snapshot-2a3797d216384301877dfaf6a398431a/manifest.json"
        ),
        "sha256": "2cc4fa21afe2dc180379456d8dd2965e6a694ffb50d52cf0a11802bac0e910cd",
        "size": 10436951,
    },
    "python_relative_path": "bin/python3.12",
    "python_sha256": "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
    "runtime_root_sha256": "67c84e211ff24570ecd1f6a663133a9fd9923b2ddc56e3f14b2ac7c8d8d80819",
    "source_prefix": "/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-g0-py31213-cu128-v1",
    "total_bytes": 7825971675,
}
NATIVE_NAMES = (
    "runtime-stage.json",
    "runtime-manifest.json",
    "runtime-context.json",
    "runtime-early.json",
    *(f"runtime-rank-{rank}-{phase}.json" for rank in (0, 1) for phase in ("before", "after")),
)
MAX_NATIVE_TOTAL = 64 * CAP
PUBLICATION_RESERVE_SECONDS = {"preflight": 300, "fit": 600}
STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
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
PROTOCOL_PATH = "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json"
PROTOCOL_MODULE = "src/posttrain_circuits/experiments/protocols/student_name_invariant_preparation.py"
FROZEN = {
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_torch_import_probe_job.py": (
        "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866"
    ),
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
}
TOOLS = (
    "tools/sdsc_student_name_invariant_startup.py",
    "tools/sdsc_student_name_invariant.py",
    "tools/sdsc_student_name_invariant_job.py",
    "tools/sdsc_student_name_invariant_worker.py",
    "tools/sdsc_student_name_invariant_audit.py",
    "tools/sdsc_student_prepare_worker.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    *FROZEN,
    "tools/sdsc_provenance.py",
    "tools/sdsc_provenance_v2.py",
    "tools/sdsc_provenance_upload_v2.py",
    "tools/sdsc_runtime_snapshot.py",
    "tools/sdsc_torch_generated_origins.py",
    "tools/sdsc_student_name_invariant_native.py",
    "tools/sdsc_student_lr_probe.py",
    "tools/sdsc_student_quality_probe.py",
    PROTOCOL_PATH,
    PROTOCOL_MODULE,
)
EXECUTION_NAMES = (
    "early-node-startup.json",
    "rank-0-startup.json",
    "rank-1-startup.json",
    "rank-0-exit.json",
    "rank-1-exit.json",
    "startup.log",
    "trace-meta.json",
)
NAMES = (
    "prepare-report.json",
    "prepare-updates.jsonl",
    *(f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl" for step in STEPS for rank in (0, 1)),
    "prepare-dev-prompts.jsonl",
    "prepare-dataset-manifest.json",
    "prepare-data-isolation.json",
    "prepare-token-audit.json",
    "prepare-preflight-records-rank-0.jsonl",
    "prepare-preflight-records-rank-1.jsonl",
    "node-result.json",
    "memory.json",
    *EXECUTION_NAMES,
    *NATIVE_NAMES,
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
def file_limit(name):
    return (
        CAP if name in EXECUTION_NAMES else (MAX_PROMPTS if name == "prepare-dev-prompts.jsonl" else MAX_FILE)
    )


def required_raw_names(mode):
    require(mode in {"preflight", "fit"}, "unknown raw evidence mode")
    names = {
        "prepare-updates.jsonl",
        "prepare-dataset-manifest.json",
        "prepare-data-isolation.json",
        "prepare-token-audit.json",
    }
    if mode == "fit":
        names.add("prepare-dev-prompts.jsonl")
        names.update(
            f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl" for step in STEPS for rank in (0, 1)
        )
    else:
        names.update(f"prepare-preflight-records-rank-{rank}.jsonl" for rank in (0, 1))
    return names


def read(path, limit=None):
    return _io.read(path, limit=file_limit(Path(path).name) if limit is None else limit)


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
        time="01:00:00" if mode == "preflight" else "08:00:00",
    )


def protocol_binding(root, expected_head=None):
    # A fresh interpreter prevents imports from another restored checkout being
    # reused as though they came from this genuine accepted scientific tree.
    code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "sys.path.insert(0,str(root/'src'));"
        "from posttrain_circuits.experiments.protocols.student_name_invariant_preparation "
        "import resolve_student_name_invariant_preparation_protocol;"
        "v=resolve_student_name_invariant_preparation_protocol(root,require_accepted=True,"
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


def science_identity(mode, protocol):
    return dict(
        task=TASK,
        mode=mode,
        protocol_sha256=protocol["protocol_sha256"],
        initial_sha256=INITIAL_SHA,
        seed=42,
        global_batch_size=64,
        optimizer_steps=4 if mode == "preflight" else 32,
        objective="independent-role-invariant-canonical-proof-common-initial",
        learning_rate=2.5e-5,
    )


def scientific_claim_path(plan):
    return (
        CONTROL
        / "student-name-invariant-v1-scientific-claims"
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
                runtime_snapshot=plan["runtime_snapshot"],
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
        and plan["schema"] == "quest-sdsc-student-name-invariant-plan-v1"
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(resources(plan["mode"]))
        and "arm" not in plan,
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
    validate_runtime_snapshot(plan)
    science_files = plan["protocol"].get("science_file_sha256")
    executed = {
        "tools/sdsc_student_name_invariant.py",
        "tools/sdsc_student_name_invariant_job.py",
        "tools/sdsc_student_name_invariant_startup.py",
        "tools/sdsc_student_name_invariant_worker.py",
        "tools/sdsc_student_name_invariant_audit.py",
        "tools/sdsc_student_prepare_worker.py",
        "tools/sdsc_student_lr_probe.py",
        "tools/sdsc_student_quality_probe.py",
        "tools/sdsc_provenance_v2.py",
        "tools/sdsc_provenance_upload_v2.py",
        "tools/sdsc_runtime_snapshot.py",
        "tools/sdsc_torch_generated_origins.py",
        "tools/sdsc_student_name_invariant_native.py",
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
        submission_dir=CONTROL / "student-name-invariant-v1-submissions" / plan["intent_id"],
        claim=CONTROL / "student-name-invariant-v1-claims" / (plan["intent_id"] + ".json"),
        result_dir=PROJECT / "student-name-invariant-v1" / plan["intent_id"],
        scientific_claim=scientific_claim_path(plan),
    ).items():
        require(plan[key] == str(expected), "unreviewed plan path: " + key)
    require(plan["job_name"] == "opd-sni-" + plan["intent_id"], "job name differs")
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
        and provenance["directory"] == str(CONTROL / "provenance-v2" / provenance["manifest_sha256"])
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
            and set(preflight) == {"plan", "plan_sha256", "audit"}
            and preflight["plan"]["mode"] == "preflight"
            and preflight["plan_sha256"] == sha(canonical(preflight["plan"])),
            "matching preflight missing",
        )
        validate_plan(preflight["plan"])
        validate_preflight_audit(preflight["audit"], preflight["plan"])
        for key in ("control_sha256", "protocol", "provenance", "code_sha256", "runtime_snapshot"):
            require(plan[key] == preflight["plan"][key], "fit/preflight implementation differs: " + key)
    else:
        require(plan["preflight"] is None, "preflight cannot resume another stage")
    require(all(plan.get(key) is False for key in FLAGS), "preparation cannot claim scientific acceptance")
    return plan


def validate_runtime_snapshot(plan):
    value = plan.get("runtime_snapshot")
    require(value == RUNTIME_SNAPSHOT, "fixed exact-byte native runtime snapshot differs")
    helper("sdsc_runtime_snapshot").validate_descriptor(value)
    require(value["manifest"]["size"] <= MAX_FILE, "native runtime manifest exceeds publication bound")
    require(value["source_prefix"] + "/bin/python3.12" == plan["python"], "source Python identity differs")
    return value


def verify_runtime_snapshot(plan):
    value = validate_runtime_snapshot(plan)
    observed = helper("sdsc_runtime_snapshot").describe(
        Path(value["manifest"]["path"]).parent,
        value["manifest"]["sha256"],
        artifact_root=RUNTIME_ARTIFACT_ROOT,
    )
    require(observed == value, "native snapshot descriptor/manifest changed")
    archive = safe(value["archive"]["path"])
    require(
        archive.is_file() and archive.stat().st_size == value["archive"]["size"],
        "native archive size/type differs",
    )
    return dict(
        manifest_sha256=value["manifest"]["sha256"],
        archive_sha256=value["archive"]["sha256"],
        archive_bytes=value["archive"]["size"],
        node_full_archive_and_inventory_readback_required=True,
    )


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
        with tempfile.TemporaryDirectory(prefix="student-name-invariant-v1-verify-") as temp:
            return verify_science(plan, Path(temp) / "science")
    restored = helper("sdsc_provenance_v2").verify(
        safe(provenance["directory"]), provenance["manifest_sha256"], plan["code_sha256"], destination
    )
    require(restored["git_head"] == provenance["head"], "restored genuine HEAD differs")
    binding = protocol_binding(destination, provenance["head"])
    require(binding == plan["protocol"], "restored accepted successor differs")
    return restored


def verify_parent(plan, *, accounting=False):
    return _initial.verify_parent(plan, accounting=accounting)


def validate_preflight_audit(record, previous, result=None):
    """Bind an independent Quest replay to the exact real preflight publication."""
    require(
        isinstance(record, dict)
        and set(record) == {"document", "sha256"}
        and isinstance(record["document"], dict),
        "independent preflight audit missing",
    )
    audit = record["document"]
    raw = canonical(audit) + b"\n"
    require(len(raw) <= MAX_AUDIT and sha(raw) == record["sha256"], "preflight audit bytes differ")
    expected = dict(
        schema="quest-sdsc-student-name-invariant-raw-audit-v1",
        raw_replay_passed=True,
        mode="preflight",
        prompts_replayed=8,
        responses_replayed=8,
        preparation_complete=False,
        plan_sha256=sha(canonical(previous)),
        auditor_sha256=previous["control_sha256"]["tools/sdsc_student_name_invariant_audit.py"],
        **{
            key: previous["protocol"][key]
            for key in ("protocol_sha256", "implementation_commit", "acceptance_commit")
        },
        protocol_artifact_sha256=previous["protocol"]["artifact_sha256"],
    )
    require(
        canonical({k: audit.get(k) for k in expected}) == canonical(expected),
        "preflight raw audit binding differs",
    )
    require(
        re.fullmatch("[1-9][0-9]*", audit.get("job_id", ""))
        and re.fullmatch("[a-f0-9]{64}", audit.get("publication_sha256", ""))
        and re.fullmatch("[a-f0-9]{64}", audit.get("result_sha256", ""))
        and audit.get("execution_acceptance_claim") is False
        and audit.get("gpu_numerics_independently_recomputed") is False
        and all(audit.get(key) is False for key in FLAGS),
        "preflight audit scope or publication identity differs",
    )
    if result is not None:
        require(
            audit["job_id"] == result["job_id"]
            and audit["publication_sha256"] == result["publication_sha256"]
            and audit["result_sha256"] == result["result_sha256"],
            "preflight audit does not match verified persistent publication",
        )
    return audit


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
    validate_preflight_audit(plan["preflight"]["audit"], previous, result)
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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_name_invariant_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_name_invariant_job.py"],
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


def scientific_specification(plan):
    """Use the reviewed protocol for numerical requirements, never a v1 fallback."""
    raw = read(ROOT / PROTOCOL_PATH)
    require(sha(raw) == plan["protocol"]["artifact_sha256"], "report protocol artifact differs")
    payload = json.loads(raw)
    core = {key: value for key, value in payload.items() if key != "review"}
    require(sha(canonical(core)) == plan["protocol"]["protocol_sha256"], "report protocol core differs")
    return payload


def worker_budget_seconds(plan):
    """The pinned protocol reserves durable-copy time inside the Slurm walltime."""
    execution = scientific_specification(plan)["execution"]
    require(
        execution.get("publication_reserve_seconds") == PUBLICATION_RESERVE_SECONDS
        and execution.get("preflight_walltime_minutes") == 60
        and execution.get("fit_walltime_minutes") == 480
        and execution.get("minimum_node_local_free_gib") == 192
        and execution.get("maximum_persistent_large_gib") == 128
        and execution.get("maximum_small_fetch_mib") == MAX_FETCH // CAP
        and execution.get("maximum_prompt_file_mib") == MAX_PROMPTS // CAP
        and execution.get("maximum_other_small_file_mib") == MAX_FILE // CAP
        and execution.get("early_startup_timeout_seconds") == 300
        and execution.get("startup_thread_count") == 12
        and execution.get("flat_publication_binds_execution_and_science") is True
        and execution.get("each_checkpoint_full_development_context_native_BF16_finite_forward_each_rank")
        is True
        and execution.get("preflight_full_development_context_native_BF16_finite_forward_each_rank") is True
        and execution.get("development_context_probe_tokens") == 2456
        and execution.get("native_runtime_snapshot") == RUNTIME_SNAPSHOT
        and execution.get("native_runtime_origin_before_after_each_rank") is True
        and execution.get("strict_torch_generated_origin") is True
        and execution.get("native_runtime_stage_in_worker_budget") is True,
        "reviewed execution/storage envelope differs",
    )
    return execution[plan["mode"] + "_walltime_minutes"] * 60 - PUBLICATION_RESERVE_SECONDS[plan["mode"]]


def select_development(plan, development):
    """Run the actual accepted selector in a fresh scientific interpreter."""
    module = ROOT / PROTOCOL_MODULE
    require(
        sha(read(module)) == plan["protocol"]["science_file_sha256"][PROTOCOL_MODULE],
        "development selector source differs",
    )
    code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "sys.path.insert(0,str(root/'src'));"
        "from posttrain_circuits.experiments.protocols.student_name_invariant_preparation "
        "import select_checkpoint;"
        "print(json.dumps(select_checkpoint(json.loads(sys.argv[2])),allow_nan=False))"
    )
    result = run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT), canonical(development).decode()], timeout=180
    )
    require(result["returncode"] == 0, "reviewed development selection failed: " + result["stderr"][-1500:])
    return json.loads(result["stdout"])


def validate_worker_report(report, plan, job, records):
    specification = scientific_specification(plan)
    training, data = specification["training"], specification["data"]
    require(
        report.get("schema") == "quest-sdsc-student-name-invariant-report-v1"
        and report.get("mode") == plan["mode"]
        and "arm" not in report
        and "diagnostic_complete" not in report
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
        optimizer_steps=specification["execution"]["preflight_steps"]
        if plan["mode"] == "preflight"
        else training["optimizer_steps"],
        global_batch_size=training["global_batch_size"],
        world_size=training["world_size"],
        learning_rate=training["optimizer"]["learning_rate"],
        fit_base_examples=data["fit_base_examples"],
        fit_training_views=data["fit_examples"],
        consumed_fit_training_views=256 if plan["mode"] == "preflight" else 2048,
        development_base_examples=data["development_base_examples"],
        development_views=data["development_examples"],
        dataset_sha256=data["dataset_manifest_sha256"],
    )
    require(
        canonical({k: report.get(k) for k in expected}) == canonical(expected),
        "worker scientific/execution binding differs",
    )
    require(
        canonical(report.get("initial_loading"))
        == canonical(
            dict(
                all_parameters_trainable=True,
                all_tensors_exact=True,
                checkpoint_sha256=INITIAL_SHA,
                key_count=311,
            )
        ),
        "native FP32 full-parameter initialization differs",
    )
    require(
        canonical(report.get("fsdp"))
        == canonical(
            dict(
                effective_fsdp_sharding_strategy="FULL_SHARD",
                requested_fsdp_sharding_strategy="FULL_SHARD",
                fsdp_wrapper_count=29,
            )
        ),
        "fixed W2 FULL_SHARD differs",
    )
    require(
        report.get("development_generation")
        == dict(
            max_new_tokens=specification["development"]["max_new_tokens"],
            max_model_input_length=specification["development"]["max_model_input_tokens"],
            max_training_model_input_length=data["max_model_input_tokens"],
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed=specification["development"]["seed"],
            rank_partition=specification["development"]["rank_partition"],
            truncation=False,
        ),
        "development generation must preserve its reviewed native-BF16 token envelope",
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
        and isolation.get("counts") == data["isolation_population_counts"]
        and isolation.get("excluded_view_counts") == data["isolation_excluded_view_counts"]
        and isinstance(isolation.get("excluded_view_stream_sha256"), dict)
        and set(isolation["excluded_view_stream_sha256"]) == set(data["isolation_excluded_view_counts"])
        and all(
            isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value)
            for value in isolation["excluded_view_stream_sha256"].values()
        )
        and isolation.get("dimensions") == data["isolation_dimensions"]
        and isolation.get("dataset_manifest_sha256") == data["dataset_manifest_sha256"],
        "complete disjoint preparation evidence missing",
    )
    require(
        isinstance(windows, list)
        and len(windows) == training["optimizer_steps"]
        and all(
            type(value) is int and 0 < value <= training["global_batch_size"] * data["max_model_input_tokens"]
            for value in windows
        )
        and envelope.get("fit_rows") == data["fit_examples"]
        and envelope.get("optimizer_windows") == training["optimizer_steps"]
        and envelope.get("token_budget") == training["input_token_budget"]
        and envelope.get("total_input_tokens") == sum(windows) <= training["input_token_budget"]
        and type(report.get("training_input_tokens")) is int
        and report["training_input_tokens"] == sum(windows[: report["optimizer_steps"]]),
        "global nonpadding input token accounting differs",
    )
    restored = report["checkpoint_restore"].get("ranks")
    require(
        isinstance(restored, list)
        and all(type(row.get("rank")) is int for row in restored)
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
    steps = (
        [specification["execution"]["preflight_steps"]]
        if plan["mode"] == "preflight"
        else training["checkpoint_steps"]
    )
    require(
        isinstance(checkpoints, list)
        and all(type(row.get("step")) is int for row in checkpoints)
        and [row.get("step") for row in checkpoints] == steps,
        "fixed checkpoint schedule differs",
    )
    for row in checkpoints:
        require(
            row.get("path") == f"checkpoints/step-{row['step']:08d}.pt"
            and row.get("scope") == "student_name_invariant_preparation_model_only"
            and type(row.get("size")) is int
            and 0 < row["size"] <= 16 * 1024**3
            and re.fullmatch("[a-f0-9]{64}", row.get("sha256", "")),
            "dense checkpoint identity differs",
        )
        reloads = row.get("reload_by_rank")
        require(
            isinstance(reloads, list)
            and all(type(r.get("rank")) is int for r in reloads)
            and [r.get("rank") for r in reloads] == [0, 1]
            and len({r.get("model_state_sha256") for r in reloads}) == 1
            and all(re.fullmatch("[a-f0-9]{64}", r.get("model_state_sha256", "")) for r in reloads),
            "both rank reloads required",
        )
        for item in reloads:
            loaded = item.get("exact_saved_master_reload", {})
            parity = item.get("root_export_logits")
            require(
                loaded.get("all_tensors_exact") is True
                and loaded.get("loaded_before_bf16_copy") is True
                and type(loaded.get("key_count")) is int
                and loaded["key_count"] == 311
                and isinstance(parity, list)
                and len(parity) == 2
                and all(
                    v.get("bitwise_equal") is True
                    and v.get("max_abs_error") == 0
                    and v.get("rms_error") == 0
                    and type(v.get("argmax_mismatches")) is int
                    and v["argmax_mismatches"] == 0
                    and type(v.get("max_abs_error")) in (int, float)
                    and type(v.get("rms_error")) in (int, float)
                    for v in parity
                ),
                "actual saved-model reload/logit parity failed",
            )
            probe = item.get("inference_envelope_probe")
            expected_probe = dict(
                passed=True,
                input_length=specification["development"]["max_model_input_tokens"],
                token_id=17,
                all_logits_finite=True,
                use_cache=False,
                native_bfloat16=True,
                scope="synthetic_name_invariant_context_no_development_content",
            )
            require(probe == expected_probe, "both ranks must pass the reviewed inference-envelope preflight")
    require(
        report.get("teacher_data_required") is False
        and report.get("source_kind") == "symbolic_canonical_name_invariant_preparation",
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
            isinstance(development, list) and len(development) == len(steps),
            "all scheduled development reports required",
        )
        for step, row, checkpoint in zip(steps, development, checkpoints, strict=True):
            require(
                row.get("step") == step
                and row.get("checkpoint_sha256") == checkpoint["sha256"]
                and row.get("examples_sha256") == data["examples_sha256"]["student_dev"],
                "development population/checkpoint differs",
            )
        selected = select_development(plan, development)
        require(
            selected is not None and canonical(report.get("selected_checkpoint")) == canonical(selected),
            "fit selection is not the earliest fully eligible focus-preparation checkpoint",
        )
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list)
        and len({r["path"] for r in raw}) == len(raw)
        and {r["path"] for r in raw} == required_raw_names(plan["mode"]),
        "complete mode-specific raw inventory missing",
    )
    for row in raw:
        require(row["path"] in records and row == records[row["path"]], "raw artifact inventory differs")


def validate_update_records(raw, report):
    """Replay emitted optimizer cadence and token arithmetic, not GPU tensor math."""
    require(isinstance(raw, bytes) and len(raw) <= MAX_FILE, "update evidence exceeds bound")
    count = 4 if report.get("mode") == "preflight" else 32
    require(report.get("mode") in {"preflight", "fit"}, "unknown update mode")
    lines = raw.splitlines()
    require(
        len(lines) == count and all(line.strip() for line in lines),
        "complete mode-specific raw optimizer rows required",
    )
    rows = [json.loads(line) for line in lines]
    windows = report["data_audit"]["token_envelope"]["window_input_tokens"]
    require(len(windows) == 32, "complete population token audit required")
    cumulative = 0
    supervised = [0, 0]
    unit = "global_nonpadding_model_input_tokens_processed"

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    for step, row in enumerate(rows, 1):
        require(
            isinstance(row, dict)
            and set(row) == {"step", "ranks"}
            and type(row["step"]) is int
            and row["step"] == step
            and isinstance(row["ranks"], list)
            and len(row["ranks"]) == 2,
            "ordered two-rank optimizer rows required",
        )
        require(
            type(windows[step - 1]) is int and 0 < windows[step - 1] <= 64 * 1536,
            "invalid audited token window",
        )
        cumulative += windows[step - 1]
        local_total = 0
        for rank, record in enumerate(row["ranks"]):
            expected_slots = list(range((step - 1) * 64 + rank, step * 64, 2))
            expected_calls = [step - 1] * 7 + [step]
            require(
                isinstance(record, dict)
                and set(record)
                == {
                    "rank",
                    "step",
                    "global_slots",
                    "global_input_tokens",
                    "optimizer_calls",
                    "metric",
                    "token_budget",
                }
                and type(record["rank"]) is int
                and record["rank"] == rank
                and type(record["step"]) is int
                and record["step"] == step
                and canonical(record["global_slots"]) == canonical(expected_slots)
                and canonical(record["optimizer_calls"]) == canonical(expected_calls)
                and type(record["global_input_tokens"]) is int
                and record["global_input_tokens"] == windows[step - 1],
                "raw fixed-W2 global64 slot/cadence/token evidence differs",
            )
            expected_budget = dict(
                accepted_optimizer_updates=step,
                budget=2000000,
                consumed=cumulative,
                stop_reason=None,
                unit=unit,
            )
            require(
                canonical(record["token_budget"]) == canonical(expected_budget) and cumulative <= 2000000,
                "raw cumulative token budget differs",
            )
            metric = record["metric"]
            require(
                isinstance(metric, dict)
                and all(
                    value is None or (isinstance(value, str) and key == "token_budget_unit") or number(value)
                    for key, value in metric.items()
                ),
                "nonfinite or untyped update metric",
            )
            require(
                number(metric.get("verified_replay_loss"))
                and metric["verified_replay_loss"] >= 0
                and number(metric.get("parameter_update_norm"))
                and metric["parameter_update_norm"] > 0,
                "finite supervised loss and positive full-parameter update required",
            )
            expected_metric = dict(
                step=step,
                optimizer_updates=step,
                model_facing_input_tokens_processed=cumulative,
                model_facing_input_tokens_this_update=windows[step - 1],
                token_budget=2000000,
                token_budget_consumed=cumulative,
                token_budget_remaining=2000000 - cumulative,
                prompts_consumed=step * 32,
                trajectories_generated=step * 32,
                effective_positive_sequences=32,
                generated_trajectories=32,
                successful_trajectories=32,
                retry_count=0,
                reward_rate=1,
            )
            require(
                all(
                    number(metric.get(key)) and metric[key] == value for key, value in expected_metric.items()
                )
                and metric.get("token_budget_unit") == unit,
                "optimizer metric/token budget cross-binding differs",
            )
            local = metric.get("local_model_facing_input_tokens_this_update")
            response = metric.get("effective_supervised_tokens_this_update")
            require(
                type(local) is int
                and 0 < local <= 32 * 1536
                and number(response)
                and 0 < response <= local
                and metric.get("effective_supervised_tokens") == response,
                "rank-local input/supervision token evidence differs",
            )
            supervised[rank] += response
            require(
                metric.get("supervised_response_tokens")
                == metric.get("response_tokens_generated")
                == supervised[rank],
                "supervised-token cumulative evidence differs",
            )
            local_total += local
        require(local_total == windows[step - 1], "rank-local tokens do not sum to global window")
    require(
        type(report.get("training_input_tokens")) is int and cumulative == report["training_input_tokens"],
        "raw full-population tokens differ from report",
    )
    return dict(
        optimizer_steps=count,
        world_size=2,
        global_training_views=count * 64,
        training_input_tokens=cumulative,
        finite_loss_and_positive_update=True,
        gpu_tensors_independently_recomputed=False,
    )


def validate_large_outputs(report, rows):
    expected_resume = {
        "resume/step-00000004/" + name
        for name in (
            "custom_checkpoint_0.pkl",
            "optimizer.bin",
            "pytorch_model_fsdp.bin",
            "random_states_0.pkl",
            "random_states_1.pkl",
            "runtime-rank-0.json",
            "runtime-rank-1.json",
        )
    }
    actual = {row["path"]: row for row in rows}
    require(
        len(actual) == len(rows)
        and set(actual) == expected_resume | {cp["path"] for cp in report["checkpoints"]},
        "all mode-specific dense checkpoints and complete step-4 resume required",
    )
    require(
        all(
            actual[cp["path"]] == {key: cp[key] for key in ("path", "size", "sha256")}
            for cp in report["checkpoints"]
        ),
        "dense checkpoint report differs from durable publication",
    )


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
        canonical({k: publication.get(k) for k in expected}) == canonical(expected)
        and all(publication.get(k) is False for k in FLAGS),
        "publication identity/scope differs",
    )
    require(
        publication.get("schema") == "quest-sdsc-student-name-invariant-publication-v1"
        and "arm" not in publication
        and "diagnostic_complete" not in publication
        and "selected_checkpoint" not in publication
        and type(publication.get("stage_complete")) is bool
        and publication.get("passed") is publication["stage_complete"]
        and publication.get("preparation_complete")
        is (publication["stage_complete"] and plan["mode"] == "fit"),
        "publication completion semantics differ",
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
            and 0 <= row["size"] <= file_limit(name)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid small result",
        )
        records[name] = row
    require(sum(r["size"] for r in rows) <= MAX_FETCH - CAP, "small output exceeds bound")
    require(
        sum(row["size"] for name, row in records.items() if name in EXECUTION_NAMES) <= 2 * CAP,
        "startup evidence exceeds 2 MiB",
    )
    require(
        sum(row["size"] for name, row in records.items() if name in NATIVE_NAMES) <= MAX_NATIVE_TOTAL,
        "native evidence exceeds64MiB",
    )
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
    require(sum(r["size"] for r in large) <= 128 * 1024**3, "checkpoint publication exceeds bound")
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
        required = required_raw_names(plan["mode"]) | {
            "prepare-report.json",
            "node-result.json",
            "memory.json",
            *EXECUTION_NAMES,
            *NATIVE_NAMES,
        }
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
        validate_update_records(read(root / "prepare-updates.jsonl"), report)
        validate_large_outputs(report, published["large_files"])
        validate_memory(document(root / "memory.json"), job)
        validate_startup_evidence(plan, root, job)
        node = document(root / "node-result.json")
        require(
            node.get("stage_complete") is True
            and type(node.get("exit_code")) is int
            and node["exit_code"] == 0
            and node.get("mode") == plan["mode"]
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
    return dict(
        preflight=preflight,
        runtime=verify_runtime(plan),
        runtime_snapshot=verify_runtime_snapshot(plan),
        gpu_concurrency=check_gpu_ceiling(),
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
        and all(len(data) <= file_limit(name) for name, data in files.items()),
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
    path = str(Path(plan["release"]) / "source/tools/sdsc_student_name_invariant.py")
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
            plan["control_sha256"]["tools/sdsc_student_name_invariant.py"],
        ],
        data=canonical(dict(action=action, plan=plan, authorize=authorize, dry_run=dry_run)),
        timeout=240,
    )
    require(
        result.returncode == 0,
        "remote action failed/acknowledgement unknown; reconcile: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    require(len(result.stdout) <= MAX_RESPONSE, "remote response exceeds bound")
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
        schema="quest-sdsc-student-name-invariant-plan-v1",
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
        runtime_snapshot=RUNTIME_SNAPSHOT,
        hf_home=receipt["hf_home"],
        preflight=None,
        **dict.fromkeys(FLAGS, False),
    )
    if args.mode == "fit":
        require(args.preflight_plan is not None, "fit requires matching real preflight plan")
        previous = validate_plan(document(safe(args.preflight_plan.absolute())))
        require(args.preflight_audit is not None, "fit requires an independent preflight raw audit")
        raw_audit = read(safe(args.preflight_audit.absolute()), MAX_AUDIT)
        audited = json.loads(raw_audit)
        require(raw_audit == canonical(audited) + b"\n", "audit must preserve canonical JSON plus LF")
        plan["preflight"] = dict(
            plan=previous,
            plan_sha256=sha(canonical(previous)),
            audit=dict(document=audited, sha256=sha(raw_audit)),
        )
    intent = execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(CONTROL / "student-name-invariant-v1-submissions" / intent),
        claim=str(CONTROL / "student-name-invariant-v1-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-name-invariant-v1" / intent),
        job_name="opd-sni-" + intent,
    )
    plan["scientific_claim"] = str(scientific_claim_path(plan))
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-name-invariant-v1" / intent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        resources=plan["resources"],
        mode=args.mode,
    )


def validate_startup_evidence(plan, root, job):
    name = "tools/sdsc_student_name_invariant_native.py"
    require(
        sha(Path(__file__).with_name(Path(name).name).read_bytes())
        == plan["control_sha256"][name]
        == plan["protocol"]["science_file_sha256"][name],
        "native validator is not the accepted pinned source",
    )
    native = helper("sdsc_student_name_invariant_native")
    files = {name: read(root / name) for name in (*NATIVE_NAMES, *EXECUTION_NAMES, "node-result.json")}
    return native.validate_evidence(plan, files, job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--mode", choices=("preflight", "fit"))
    parser.add_argument("--run-id")
    for name in ("fetch-dir", "status-file", "plan", "preflight-plan", "preflight-audit", "dry-run-file"):
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
                path.parent.parent == ROOT / ".sdsc/student-name-invariant-v1" and path.name == "plan.json",
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
