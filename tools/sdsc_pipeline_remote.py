#!/usr/bin/env python3
"""Short-lived Expanse control operation. Every Slurm command stays on Expanse.

An exclusive stage claim is durable before sbatch. Ambiguous submissions are
reconciled by their unique scheduler name; they are never submitted again.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdsc_pipeline_storage as s

STAGES = (
    "g0",
    "preflight4",
    "prepare",
    "teacher-shard",
    "pilot-inputs",
    "train-cell",
    "initial-circuits",
    "final-circuits",
    "local-fork",
    "resume",
    "dynamics",
    "finalize",
)
GPU = {
    "g0": 2,
    "preflight4": 4,
    "prepare": 1,
    "teacher-shard": 1,
    "pilot-inputs": 1,
    "train-cell": 4,
    "initial-circuits": 1,
    "final-circuits": 1,
    "local-fork": 1,
    "resume": 4,
    "dynamics": 1,
    "finalize": 1,
}
ARRAY = {"teacher-shard": "0-15%4", "train-cell": "0-7%1"}
COUNTS = {"teacher-shard": 16, "train-cell": 8}
TIME = {
    "g0": "12:00:00",
    "preflight4": "00:30:00",
    "prepare": "00:30:00",
    "teacher-shard": "04:00:00",
    "pilot-inputs": "24:00:00",
    "train-cell": "12:00:00",
    "initial-circuits": "06:00:00",
    "final-circuits": "08:00:00",
    "local-fork": "12:00:00",
    "resume": "02:00:00",
    "dynamics": "02:00:00",
    "finalize": "02:00:00",
}
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def command(argv, timeout=60):
    r = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "TZ": "UTC"},
    )
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def checked_query(argv):
    # Transient scheduler/accounting failures get three bounded attempts. No
    # authentication loop exists here; this process already runs behind SSH.
    for attempt in range(3):
        result = command(argv)
        if result["returncode"] == 0:
            return result
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("scheduler query failed after bounded retries: " + argv[0])


def plan_for(request):
    flow = request.get("flow_id", "")
    s.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", flow), "invalid flow ID")
    root = s.CONTROL / "pipelines" / flow
    plan = s.document(root / "plan.json", request["plan_sha256"])
    s.require(plan["flow_id"] == flow and plan.get("authorized") is True, "pipeline is not authorized")
    s.require(
        plan["stages"] == list(STAGES) and plan["resources"] == resource_plan(),
        "unreviewed stage/resource plan",
    )
    s.require(plan["science_head"] == "0215c356355b29b5e2b407978a207db2156719e1", "unexpected science HEAD")
    manifest = s.document(s.CONTROL / "releases" / plan["run_id"] / "manifest.json")
    s.require(
        manifest["code_sha256"] == plan["code_sha256"] == s.sha(s.canonical(manifest["files"])),
        "release changed",
    )
    return plan, root


def resource_plan():
    return {
        stage: {
            "account": "nwu181",
            "partition": "nairr-gpu" if GPU[stage] == 4 else "nairr-gpu-shared",
            "qos": "nairr-gpu-normal" if GPU[stage] == 4 else "nairr-gpu-shared-normal",
            "nodes": 1,
            "gpus": GPU[stage],
            "gpu_type": "h100",
            "cpus": 24,
            "mem_gib": 192,
            "time": TIME[stage],
            "array": ARRAY.get(stage),
        }
        for stage in STAGES
    }


def queue_guard():
    result = checked_query(["squeue", "--user=zgao12", "--noheader", "--format=%i|%j|%T|%b|%Z"])
    rows = [x for x in result["stdout"].splitlines() if x.strip()]
    conflicting = [
        row for row in rows if "opd" in row.lower() and ("gpu" in row.lower() or "|opd-" in row.lower())
    ]
    s.require(not conflicting, "another pending/running OPD GPU job can allocate; stop before submission")
    return result


def sbatch(intent, *, walltime=None, test=False):
    stage = intent["stage"]
    r = resource_plan()[stage]
    out = Path(intent["directory"])
    argv = [
        "sbatch",
        "--parsable",
        "--account=" + r["account"],
        "--partition=" + r["partition"],
        "--qos=" + r["qos"],
        "--nodes=1",
        "--ntasks=1",
        "--gpus=h100:" + str(r["gpus"]),
        "--cpus-per-task=24",
        "--mem=192G",
        "--time=" + (walltime or r["time"]),
        "--export=NONE",
        "--no-requeue",
        "--signal=B:TERM@600",
        "--job-name=" + intent["job_name"],
        "--comment=" + intent["job_name"],
        "--chdir=" + str(s.CONTROL / "releases" / intent["run_id"]),
        "--output=" + str(out / "slurm-%A_%a.out"),
        "--error=" + str(out / "slurm-%A_%a.err"),
    ]
    if stage in ARRAY:
        argv.append("--array=" + ARRAY[stage])
    if test:
        argv.append("--test-only")
    argv.extend(
        [
            str(s.CONTROL / "releases" / intent["run_id"] / "source/tools/sdsc_pipeline_job.sh"),
            intent["python"],
            str(out / "request.json"),
            intent["request_sha256"],
        ]
    )
    return argv


def inspect_accounting(job, stage, queue, accounting):
    if queue["returncode"] or accounting["returncode"]:
        return "UNKNOWN", False
    if queue["stdout"].strip():
        return "ACTIVE", False
    rows = [line.split("|") for line in accounting["stdout"].splitlines() if line.strip()]
    count = COUNTS.get(stage)
    expected = {job + "_" + str(i) for i in range(count)} if count else {job}
    allocations = [r for r in rows if len(r) >= 3 and r[0] in expected]
    actual = {
        r[0] for r in rows if len(r) >= 3 and (r[0] == job or r[0].startswith(job + "_")) and "." not in r[0]
    }
    if actual - expected:
        return "FAILED", False
    if {r[0] for r in allocations} != expected or len(allocations) != len(expected):
        return "UNKNOWN", False
    related = [r for r in rows if r[0] in expected or any(r[0].startswith(x + ".") for x in expected)]
    if not all(r[1].split()[0] in TERMINAL for r in related):
        return "UNKNOWN", False
    success = all(r[1:3] == ["COMPLETED", "0:0"] for r in related)
    return ("COMPLETED" if success else "FAILED"), success


def inspect_stage(plan, root, stage):
    directory = root / stage
    receipt = s.document(directory / "submission.json")
    job = receipt["job_id"]
    queue = checked_query(["squeue", "--array", "--noheader", "--user=zgao12", "--format=%i|%T|%j"])
    queue["stdout"] = "\n".join(
        line
        for line in queue["stdout"].splitlines()
        if line.split("|", 1)[0] == job or line.split("|", 1)[0].startswith(job + "_")
    )
    accounting = checked_query(
        [
            "sacct",
            "-n",
            "-P",
            "--array",
            "--jobs",
            job,
            "--format=JobID%64,State,ExitCode,Elapsed,MaxRSS,JobIDRaw",
        ]
    )
    state, success = inspect_accounting(job, stage, queue, accounting)
    publications = []
    reports = []
    if success:
        for index in range(COUNTS.get(stage, 1)):
            suffix = str(index) if stage in COUNTS else "single"
            base = s.PROJECT / "pipeline-results" / plan["flow_id"] / stage / suffix
            publication = s.document(base / "receipt.json")
            result = s.document(base / "result.json", publication["result_sha256"])
            expected = {k: receipt[k] for k in ("flow_id", "stage", "run_id", "code_sha256")}
            expected["job_id"] = job + "_" + str(index) if stage in COUNTS else job
            s.require(
                all(publication.get(k) == v and result.get(k) == v for k, v in expected.items()),
                "publication identity differs",
            )
            s.require(
                publication.get("passed") is True
                and publication.get("persisted") is True
                and publication.get("persistent_read_back_verified") is True
                and result.get("passed") is True,
                "scientific stage/publication failed",
            )
            for name, entry in publication["delta"].items():
                s.relative(name)
                p = s.safe(entry["storage"])
                s.require(
                    base in p.parents and p.is_file() and p.stat().st_size == entry["size"],
                    "published artifact missing or resized",
                )
            publications.append(
                {
                    "path": str(base / "receipt.json"),
                    "sha256": s.sha(s.read(base / "receipt.json")),
                    "value": publication,
                }
            )
            reports.append(
                {"path": str(base / "result.json"), "sha256": publication["result_sha256"], "value": result}
            )
    return {
        "stage": stage,
        "job_id": job,
        "task": "qwen3-v2-" + stage,
        "run_id": plan["run_id"],
        "code_sha256": plan["code_sha256"],
        "state": state,
        "success": success,
        "queue": queue,
        "accounting": accounting,
        "result": {
            "verified": success,
            "verification_scope": (
                "report_hash_and_durable_publication_receipt; full_input_sha_checked_on_node_before_use"
            ),
            "result_sha256": reports[0]["sha256"] if len(reports) == 1 else None,
        },
        "publications": publications,
        "reports": reports,
        "submission": receipt,
    }


def verify_input(plan, root, request):
    stage = request["stage"]
    position = STAGES.index(stage)
    s.require(
        request["base_inventory"] and request["g0_root"].startswith("/scratch/"),
        "missing calibration input identity",
    )
    if position:
        previous = inspect_stage(plan, root, STAGES[position - 1])
        s.require(previous["success"], "previous scientific stage is not successfully terminal")
    # The caller carries all bytes hashes; full content is verified by the GPU
    # worker on node-local copy before any training, and reports are checked now.
    for entry in request["base_inventory"].values():
        p = s.safe(entry["storage"])
        s.require(
            s.PROJECT in p.parents and p.is_file() and p.stat().st_size == entry["size"],
            "input inventory unavailable",
        )
    s.require(
        request["run_id"] == plan["run_id"] and request["code_sha256"] == plan["code_sha256"],
        "request release differs",
    )
    for key in (
        "environment",
        "adapter_review",
        "provenance_dir",
        "provenance_sha256",
        "science_head",
        "protocol_sha256",
        "hf_home",
    ):
        s.require(
            s.canonical(request[key]) == s.canonical(plan[key]), "request changed planned binding: " + key
        )
    expected_python = plan["g0_python"] if stage == "g0" else plan["environment"]["runtime"]["python_path"]
    s.require(request["python"] == expected_python, "request changed fixed runtime")
    if stage == "g0":
        for name in ("calibration", "preflight2"):
            submitted = request[name]["submission"]
            actual = legacy_evidence({"job_id": submitted["job_id"], "intent_id": submitted["intent_id"]})
            s.require(
                actual["report_sha256"] == request[name]["report_sha256"]
                and actual["publication_receipt_sha256"] == request[name]["publication_receipt_sha256"],
                "legacy scientific prerequisite changed",
            )
    return queue_guard()


def submit(plan, root, request):
    stage = request["stage"]
    s.require(stage in STAGES, "unsupported stage")
    s.require(request["job_request"].get("stage") == stage, "inner/outer stage identity differs")
    directory = root / stage
    s.require(not directory.exists(), "stage already claimed; use reconcile/status, never resubmit")
    queue = verify_input(plan, root, request["job_request"])
    directory.mkdir(mode=0o700)
    job_request = request["job_request"]
    job_request.update(plan_sha256=request["plan_sha256"], flow_id=plan["flow_id"], stage=stage)
    request_sha = s.atomic(directory / "request.json", job_request)
    intent = {
        "flow_id": plan["flow_id"],
        "stage": stage,
        "run_id": plan["run_id"],
        "code_sha256": plan["code_sha256"],
        "directory": str(directory),
        "request_sha256": request_sha,
        "python": job_request["python"],
        "resources": resource_plan()[stage],
        "job_name": "opd-pipe-" + uuid.uuid4().hex,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    s.atomic(directory / "intent.json", intent)
    # Compare defensible bounds on the *same* script and allocation. The shorter
    # value remains the chosen no-history margin; no test-only ID is a real job.
    chosen = TIME[stage]
    h, m, sec = map(int, chosen.split(":"))
    seconds = h * 3600 + m * 60 + sec
    longer = min(seconds + max(900, seconds // 2), 48 * 3600)
    candidate = f"{longer // 3600:02}:{longer % 3600 // 60:02}:{longer % 60:02}"
    estimates = [
        {"time": value, **command(sbatch(intent, walltime=value, test=True), timeout=120)}
        for value in (chosen, candidate)
    ]
    s.atomic(
        directory / "scheduler-estimates.json",
        {"chosen": chosen, "queue_guard": queue, "estimates": estimates},
    )
    s.require(
        all(r["returncode"] == 0 for r in estimates), "scheduler test-only rejected resources; no submission"
    )
    # Recheck after scheduling probes, before the one and only sbatch attempt.
    queue_guard()
    s.atomic(directory / "submission-attempt.json", {"argv": sbatch(intent), "started": time.time()})
    result = command(sbatch(intent), timeout=120)
    s.atomic(directory / "sbatch-response.json", result)
    s.require(
        result["returncode"] == 0 and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?", result["stdout"]),
        "submission receipt ambiguous; reconcile this claimed stage, never retry sbatch",
    )
    receipt = {**intent, "job_id": result["stdout"].strip().split(";")[0]}
    s.atomic(directory / "submission.json", receipt)
    for name, argv in [
        ("queue-start", ["squeue", "--start", "--jobs", receipt["job_id"]]),
        ("job-detail", ["scontrol", "show", "job", receipt["job_id"]]),
    ]:
        s.atomic(directory / (name + ".json"), checked_query(argv))
    return receipt


def reconcile(plan, root, stage):
    directory = root / stage
    if (directory / "submission.json").exists():
        return s.document(directory / "submission.json")
    intent = s.document(directory / "intent.json")
    q = checked_query(["squeue", "--user=zgao12", "--noheader", "--format=%A|%j"])
    a = checked_query(
        [
            "sacct",
            "-n",
            "-P",
            "--user=zgao12",
            "--starttime",
            intent["created_at"][:19],
            "--name",
            intent["job_name"],
            "--format=JobID%64,JobName%80,State,ExitCode",
        ]
    )
    matches = set()
    for text in (q["stdout"], a["stdout"]):
        for line in text.splitlines():
            row = line.split("|")
            if len(row) >= 2 and row[1].strip() == intent["job_name"]:
                candidate = row[0].split(".")[0].split("_")[0]
                if re.fullmatch(r"[1-9][0-9]*", candidate):
                    matches.add(candidate)
    s.require(len(matches) == 1, "reconciliation inconclusive; keep claim, never resubmit")
    receipt = {**intent, "job_id": matches.pop(), "recovered": True}
    s.atomic(directory / "submission.json", receipt)
    return receipt


def fetch(plan, root, stage):
    s.require(stage in STAGES, "unsupported stage")
    receipt = s.document(root / stage / "submission.json")
    files = []
    total = 0
    for index in range(COUNTS.get(stage, 1)):
        suffix = str(index) if stage in COUNTS else "single"
        base = s.PROJECT / "pipeline-results" / plan["flow_id"] / stage / suffix
        if not base.is_dir():
            continue
        for name in ("result.json", "receipt.json", "publication-failure.json", "worker.log"):
            path = base / name
            if not path.is_file():
                continue
            if name == "worker.log":
                with s.safe(path).open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    raw = stream.read(65536)
            elif path.stat().st_size <= 2 * 1024**2:
                raw = s.read(path, 2 * 1024**2)
            else:
                continue
            total += len(raw)
            if total > 8 * 1024**2:
                return {"files": files, "truncated": True, "job_id": receipt["job_id"]}
            files.append(
                {
                    "path": suffix + "/" + name,
                    "sha256": s.sha(raw),
                    "size": len(raw),
                    "base64": base64.b64encode(raw).decode(),
                }
            )
    return {"files": files, "truncated": False, "job_id": receipt["job_id"]}


def dispatch(request):
    s.require(
        Path.home() == Path("/home/zgao12") and not os.environ.get("SLURM_JOB_ID"),
        "control operations require Expanse login identity",
    )
    plan, root = plan_for(request)
    action = request["action"]
    stage = request.get("stage")
    if action == "legacy":
        return legacy_evidence(request["binding"])
    if action == "submit":
        return submit(plan, root, request)
    if action == "status":
        return inspect_stage(plan, root, stage)
    if action == "reconcile":
        return reconcile(plan, root, stage)
    if action == "fetch":
        return fetch(plan, root, stage)
    if action == "exists":
        return {"claimed": (root / stage).exists(), "submitted": (root / stage / "submission.json").exists()}
    raise ValueError("unsupported control action")


def legacy_evidence(binding):
    legacy = s.helper("sdsc_remote")
    status = legacy.status({"root": str(s.CONTROL), **binding})
    s.require(status.get("success") is True, "upstream legacy job not successfully terminal")
    receipt, _ = legacy.bound_receipt({"root": str(s.CONTROL), **binding})
    name = legacy.TASK_RESULTS[receipt["task"]]
    root = Path(receipt["result_dir"])
    report_path = root / name
    publication_path = root / "receipt.json"
    publication = s.document(publication_path)
    report = s.document(report_path)
    source_manifest = s.CONTROL / "releases" / receipt["run_id"] / "manifest.json"
    return {
        "status": status,
        "submission": receipt,
        "report": report,
        "report_path": str(report_path),
        "report_sha256": s.sha(s.read(report_path)),
        "publication_receipt": publication,
        "publication_receipt_path": str(publication_path),
        "publication_receipt_sha256": s.sha(s.read(publication_path)),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": s.sha(s.read(source_manifest)),
    }


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(16 * 1024**2 + 1)
        s.require(len(raw) <= 16 * 1024**2, "request exceeds bound")
        value = dispatch(json.loads(raw))
        print(json.dumps({"ok": True, "value": value}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        sys.exit(1)
