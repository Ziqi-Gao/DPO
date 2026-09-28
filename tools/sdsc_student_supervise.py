"""One finite Quest observer: accepted v2 preflight -> one student calibration.

All scheduler operations use the existing SSH-only CLI. This process neither
deploys code nor retries, cancels, resumes, accepts G0, or starts a pilot.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "quest-sdsc-adapted-student-supervision-v1"
REQUEST_SCHEMA = "quest-sdsc-student-supervision-request-v1"
PREFLIGHT = "54506703"
PREFLIGHT_RUN = "20260928T183316Z-b542b2e7e7fa-1cd4c8d9"
PREFLIGHT_TASK = "qwen3-v2-adapted-preflight"
TASK = "qwen3-v2-adapted-calibration"
HEAD = "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3"
REMOTE = "/home/zgao12/quest-runs/OPD"
PROJECT = "/expanse/lustre/projects/nwu181/zgao12/OPD"
PYTHON = PROJECT + "/envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12"
PYTHON_SHA = "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19"
HF_HOME = PROJECT + "/cache/huggingface"
RESULT_ROOT = PROJECT + "/control-results"
PROTOCOL_PATH = "prereg/amendments/qwen3_adapted_student_calibration_v2.json"
FIXED_BINDINGS = {
    "science_git_head": HEAD,
    "teacher_job_id": "54496291",
    "student_protocol_sha256": "9277960e4ff3599c325ac0115888280ad32647891fd3841d045822bf7db2a320",
    "student_protocol_artifact_sha256": "2c15821442d3ae51da5d86917a5fbf160aac9a1bb1b6ef5e8944dab8f8c69274",
    "adapted_teacher_sha256": "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7",
    "teacher_acceptance_sha256": "5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337",
    "teacher_acceptance_inventory_sha256": "8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed",
}
RESOURCES = {
    "account": "nwu181",
    "partition": "nairr-gpu-shared",
    "qos": "nairr-gpu-shared-normal",
    "gpu_type": "h100",
    "gpus": 2,
    "cpus": 24,
    "mem_gib": 192,
    "time": "02:00:00",
}
PACKAGES = {
    "torch": "2.8.0+cu128",
    "numpy": "1.26.4",
    "transformers": "4.56.2",
    "accelerate": "1.10.1",
    "tokenizers": "0.22.0",
}
CONTROL_FILES = (
    "tools/sdsc_student_supervise.py",
    "tools/sdsc_student_supervise",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_provenance.py",
    "tools/sdsc_student_contract.py",
    "tools/sdsc_adapted_training_preflight.py",
    "tools/sdsc_adapted_calibration.py",
    "tools/sdsc_student_job.py",
    "tools/sdsc_student_job.sh",
)
POLL = 300
MAX_SECONDS = 14 * 24 * 3600
SHA = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
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
FINISHED = {"finished", "stopped", "calibration_complete", "completed", "failed"}
STATUS_KEYS = (
    "job_id",
    "run_id",
    "task",
    "hf_home",
    "provenance_dir",
    "provenance_manifest_sha256",
    "science_git_head",
    "bundle_sha256",
    "teacher_job_id",
    "preflight_job_id",
    "prerequisites_path",
    "prerequisites_sha256",
    *tuple(k for k in FIXED_BINDINGS if k not in {"science_git_head", "teacher_job_id"}),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(ord(c) < 32 for c in str(path)), "unsafe local path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink local path")
    return path


def read(path, maximum=4 * 1024**2):
    path = safe(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "control file size/type differs")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())

        def stamp(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        require(
            stamp(before) == stamp(after) == stamp(path.lstat()) and len(raw) == before.st_size,
            "control changed while read",
        )
    return raw


def document(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(read(path), object_pairs_hook=unique)
    require(isinstance(value, dict), "expected JSON object")
    canonical(value)
    return value


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, value, *, replace=False):
    path = safe(path)
    raw = canonical(value) + b"\n"
    temporary = path.with_name("." + path.name + ".tmp") if replace else path
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    sync_directory(path.parent)
    return sha(raw)


def pin(path, root=ROOT, maximum=4 * 1024**2):
    path = safe(path)
    require(root in path.parents, "evidence is outside Quest checkout")
    raw = read(path, maximum)
    return {"path": str(path.relative_to(root)), "size": len(raw), "sha256": sha(raw)}


def controls(root=ROOT):
    return {name: sha(read(root / name)) for name in CONTROL_FILES}


def deployed(run_id, root=ROOT):
    require(isinstance(run_id, str) and NAME.fullmatch(run_id), "invalid run ID")
    path = root / ".sdsc/runs" / (run_id + ".json")
    record = document(path)
    manifest, deployment = record.get("manifest", {}), record.get("deployment", {})
    require(record.get("state") == "deployed" and manifest.get("run_id") == run_id, "run is not deployed")
    require(manifest.get("git_head") == HEAD, "release must preserve actual accepted scientific HEAD")
    require(
        manifest.get("code_sha256") == sha(canonical(manifest.get("files"))), "release manifest hash differs"
    )
    files = {row["path"]: row for row in manifest["files"]}
    require(len(files) == len(manifest["files"]), "duplicate deployment file")
    require(
        files.get(PROTOCOL_PATH, {}).get("sha256") == FIXED_BINDINGS["student_protocol_artifact_sha256"],
        "release protocol differs",
    )
    require(
        deployment.get("ok") is True
        and deployment.get("run_id") == run_id
        and deployment.get("code_sha256") == manifest["code_sha256"]
        and deployment.get("release") == REMOTE + "/releases/" + run_id,
        "deployment receipt differs",
    )
    return manifest, pin(path, root)


def binding(job, root=ROOT):
    matches = []
    for path in (root / ".sdsc/submissions").glob("*.json"):
        value = document(path)
        if str(value.get("receipt", {}).get("job_id", "")) == job:
            matches.append((path, value))
    require(len(matches) == 1, "unknown or ambiguous job; reconcile without retry")
    path, value = matches[0]
    receipt, request = value.get("receipt", {}), value.get("request", {})
    require(value.get("state") == "submitted" and receipt.get("ok") is True, "unresolved submission")
    require(
        re.fullmatch(r"[0-9a-f]{32}", path.stem) and receipt.get("intent_id") == path.stem,
        "intent binding differs",
    )
    for key in (
        "intent_id",
        "task",
        "run_id",
        "code_sha256",
        "resources",
        "python",
        "hf_home",
        "provenance_dir",
        "provenance_manifest_sha256",
        "teacher_job_id",
        "preflight_job_id",
    ):
        require(receipt.get(key) == request.get(key), "request/receipt differ: " + key)
    manifest, run_pin = deployed(receipt["run_id"], root)
    require(receipt.get("code_sha256") == manifest["code_sha256"], "job deployment differs")
    return {"receipt": receipt, "submission": pin(path, root), "deployment": run_pin}


def validate_receipt(receipt, *, calibration, plan=None):
    require(
        receipt.get("ok") is True and re.fullmatch(r"[1-9][0-9]*", str(receipt.get("job_id", ""))),
        "no trustworthy job receipt",
    )
    require(
        all(receipt.get(k) == v for k, v in FIXED_BINDINGS.items()), "receipt scientific identity differs"
    )
    require(
        receipt.get("python") == PYTHON
        and receipt.get("hf_home") == HF_HOME
        and receipt.get("container") is None,
        "receipt runtime differs",
    )
    require(receipt.get("task") == (TASK if calibration else PREFLIGHT_TASK), "receipt task differs")
    expected_resources = RESOURCES if calibration else {**RESOURCES, "time": "01:00:00"}
    require(canonical(receipt.get("resources")) == canonical(expected_resources), "receipt resources differ")
    require(
        receipt.get("preflight_job_id") == (PREFLIGHT if calibration else None), "receipt preflight differs"
    )
    intent = receipt.get("intent_id", "")
    require(re.fullmatch(r"[a-f0-9]{32}", intent), "receipt intent invalid")
    require(
        receipt.get("prerequisites_path") == REMOTE + "/submissions/" + intent + "/prerequisites.json",
        "prerequisite locator differs",
    )
    for key in ("code_sha256", "bundle_sha256", "provenance_manifest_sha256", "prerequisites_sha256"):
        require(
            isinstance(receipt.get(key), str) and SHA.fullmatch(receipt[key]), "receipt hash missing: " + key
        )
    require(
        receipt.get("provenance_dir") == REMOTE + "/provenance/" + receipt["provenance_manifest_sha256"],
        "provenance locator differs",
    )
    require(
        receipt.get("result_dir") == RESULT_ROOT + "/" + receipt["run_id"] + "/" + intent,
        "persistent result locator differs",
    )
    if calibration:
        expected = plan["calibration"]
        require(receipt["job_id"] not in {PREFLIGHT, "54496291"}, "calibration job is not fresh")
        for key in ("run_id", "code_sha256", "provenance_dir", "provenance_manifest_sha256", "bundle_sha256"):
            require(receipt.get(key) == expected[key], "calibration receipt differs: " + key)
    else:
        require(
            receipt["job_id"] == PREFLIGHT and receipt.get("run_id") == PREFLIGHT_RUN,
            "only preflight 54506703 is authorized",
        )


def no_competitor(root, own_directory=None, *, before_submit=True):
    for path in (root / ".sdsc/supervision").glob("*/state.json"):
        if path.parent != own_directory:
            require(
                document(path).get("phase") in FINISHED, "another supervision flow is active or unresolved"
            )
    for path in (root / ".sdsc/submissions").glob("*.json"):
        value = document(path)
        require(
            value.get("state") not in {"sending", "unknown"}, "unknown submission requires reconciliation"
        )
        request = value.get("request", {})
        if before_submit:
            require(
                not (request.get("task") == TASK and request.get("preflight_job_id") == PREFLIGHT),
                "this preflight already has a calibration intent; never submit twice",
            )


def local_artifact(path, root):
    artifact = safe(path)
    relative = artifact.relative_to(root)
    replay = (
        len(relative.parts) == 5
        and relative.parts[:2] == (".sdsc", "replays")
        and re.fullmatch(r"replay-[0-9a-f]{32}", relative.parts[2])
        and relative.parts[3:] == ("stage", "provenance")
    )
    require(
        root / ".sdsc/provenance" in artifact.parents or replay,
        "provenance is outside exact local artifact/replay roots",
    )
    return artifact


def build_plan(request, root=ROOT, now=None):
    require(
        set(request)
        == {"schema", "flow_id", "run_id", "provenance_created", "provenance_uploaded", "runtime_evidence"}
        and request["schema"] == REQUEST_SCHEMA,
        "unexpected supervision request fields",
    )
    require(isinstance(request["flow_id"], str) and NAME.fullmatch(request["flow_id"]), "invalid flow ID")
    no_competitor(root)
    preflight = binding(PREFLIGHT, root)
    validate_receipt(preflight["receipt"], calibration=False)
    manifest, release_pin = deployed(request["run_id"], root)
    require(request["run_id"] != PREFLIGHT_RUN, "calibration requires fresh release")
    for path in (root / ".sdsc/submissions").glob("*.json"):
        require(
            document(path).get("request", {}).get("run_id") != request["run_id"],
            "calibration release already claimed",
        )
    created, uploaded, runtime = (
        document(request[key]) for key in ("provenance_created", "provenance_uploaded", "runtime_evidence")
    )
    require(uploaded.get("verified") is True, "provenance upload was not verified")
    for key, expected in {"git_head": HEAD, "wrapper_code_sha256": manifest["code_sha256"]}.items():
        require(created.get(key) == uploaded.get(key) == expected, "provenance identity differs: " + key)
    for key in ("manifest_sha256", "bundle_sha256"):
        require(
            isinstance(created.get(key), str)
            and SHA.fullmatch(created[key])
            and created[key] == uploaded.get(key),
            "provenance receipt hash differs",
        )
    remote_artifact = REMOTE + "/provenance/" + created["manifest_sha256"]
    require(
        uploaded.get("remote_artifact") == remote_artifact and uploaded.get("run_id") == request["run_id"],
        "provenance upload target differs",
    )
    artifact = local_artifact(created["artifact"], root)
    spec = importlib.util.spec_from_file_location(
        "student_supervision_provenance", root / "tools/sdsc_provenance.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    proof, wrapper, _ = helper.load_artifact(artifact, created["manifest_sha256"], manifest["code_sha256"])
    verified = helper.verify(artifact, created["manifest_sha256"], manifest["code_sha256"])
    require(
        verified.get("verified") is True and verified.get("git_head") == HEAD,
        "local genuine Git verification failed",
    )
    require(
        proof["git_head"] == HEAD
        and wrapper == manifest
        and proof["bundle"]["sha256"] == created["bundle_sha256"],
        "local provenance differs from deployed snapshot",
    )
    if "stdout" in runtime:
        require(runtime.get("returncode") == 0, "runtime check failed")
        runtime = json.loads(runtime["stdout"])
    require(
        runtime.get("verified") is True
        and runtime.get("python") == PYTHON
        and runtime.get("sha256") == PYTHON_SHA
        and str(runtime.get("version", "")).startswith("3.12.13")
        and runtime.get("packages") == PACKAGES
        and runtime.get("gpu_imported") is False,
        "fixed runtime evidence differs",
    )
    check_path = root / ".sdsc/check.json"
    check = document(check_path)
    require(
        check.get("connected") is True
        and check.get("target") == "zgao12@login.expanse.sdsc.edu"
        and check.get("quest_root") == str(root),
        "shared SSH discovery evidence differs",
    )
    require(check.get("control_python") == "/usr/bin/python3.11", "discovered remote control Python differs")
    pinned = [release_pin, preflight["submission"], preflight["deployment"], pin(check_path, root)]
    pinned.extend(
        pin(request[key], root) for key in ("provenance_created", "provenance_uploaded", "runtime_evidence")
    )
    # Full bundle/Git verification happens once above. Runtime uses its immutable
    # remote digest; polling only re-reads the small receipts and manifests.
    pinned.extend(pin(artifact / name, root) for name in ("manifest.json", "wrapper-manifest.json"))
    when = time.time() if now is None else now
    return {
        "schema": SCHEMA,
        "request": request,
        "flow_id": request["flow_id"],
        "quest_root": str(root),
        "quest_host": socket.gethostname().split(".")[0],
        "created_at_unix": when,
        "expires_at_unix": when + MAX_SECONDS,
        "poll_seconds": POLL,
        "control_sha256": controls(root),
        "evidence": pinned,
        "preflight": preflight,
        "runtime": runtime,
        "scientific_bindings": FIXED_BINDINGS,
        "calibration": {
            "run_id": request["run_id"],
            "code_sha256": manifest["code_sha256"],
            "provenance_dir": remote_artifact,
            "provenance_manifest_sha256": created["manifest_sha256"],
            "bundle_sha256": created["bundle_sha256"],
        },
        "resources": RESOURCES,
        "allowed_operations": ["status", "fetch", "submit_once"],
        "scope": "one_canonical_sft_calibration_only_no_G0_pilot_factorial_or_cancellation",
    }


def validate_plan(plan, root=ROOT):
    require(
        set(plan)
        == {
            "schema",
            "request",
            "flow_id",
            "quest_root",
            "quest_host",
            "created_at_unix",
            "expires_at_unix",
            "poll_seconds",
            "control_sha256",
            "evidence",
            "preflight",
            "runtime",
            "scientific_bindings",
            "calibration",
            "resources",
            "allowed_operations",
            "scope",
        },
        "unexpected plan fields",
    )
    require(
        plan.get("schema") == SCHEMA
        and plan.get("quest_root") == str(root)
        and plan.get("quest_host") == socket.gethostname().split(".")[0],
        "wrong supervisor schema/Quest location",
    )
    require(plan.get("control_sha256") == controls(root), "pinned control bytes changed")
    require(
        plan.get("scientific_bindings") == FIXED_BINDINGS
        and canonical(plan.get("resources")) == canonical(RESOURCES),
        "plan scientific/resource scope changed",
    )
    require(
        plan.get("scope") == "one_canonical_sft_calibration_only_no_G0_pilot_factorial_or_cancellation"
        and plan.get("allowed_operations") == ["status", "fetch", "submit_once"],
        "plan operations changed",
    )
    require(
        plan.get("poll_seconds") == POLL
        and all(
            type(plan.get(k)) in (int, float) and math.isfinite(plan[k])
            for k in ("created_at_unix", "expires_at_unix")
        )
        and plan["expires_at_unix"] - plan["created_at_unix"] == MAX_SECONDS,
        "finite observation bounds changed",
    )
    request = plan["request"]
    require(
        set(request)
        == {"schema", "flow_id", "run_id", "provenance_created", "provenance_uploaded", "runtime_evidence"}
        and request["schema"] == REQUEST_SCHEMA
        and request["flow_id"] == plan["flow_id"]
        and NAME.fullmatch(plan["flow_id"]),
        "plan request differs",
    )
    evidence = plan["evidence"]
    require(
        isinstance(evidence, list) and len(evidence) == 9 and len({row["path"] for row in evidence}) == 9,
        "plan must retain all nine distinct evidence files",
    )
    for row in evidence:
        require(pin(root / row["path"], root) == row, "pinned evidence changed: " + row["path"])
    require(binding(PREFLIGHT, root) == plan["preflight"], "preflight receipt/deployment changed")
    validate_receipt(plan["preflight"]["receipt"], calibration=False)
    manifest, _ = deployed(plan["calibration"]["run_id"], root)
    require(manifest["code_sha256"] == plan["calibration"]["code_sha256"], "calibration deployment changed")
    require(plan["calibration"]["run_id"] != PREFLIGHT_RUN, "calibration release must be fresh")
    created = document(request["provenance_created"])
    uploaded = document(request["provenance_uploaded"])
    calibration = plan["calibration"]
    require(
        set(calibration)
        == {"run_id", "code_sha256", "provenance_dir", "provenance_manifest_sha256", "bundle_sha256"}
        and calibration["run_id"] == request["run_id"],
        "calibration plan fields differ",
    )
    require(
        created.get("git_head") == uploaded.get("git_head") == HEAD
        and created.get("wrapper_code_sha256")
        == uploaded.get("wrapper_code_sha256")
        == calibration["code_sha256"]
        and uploaded.get("verified") is True
        and uploaded.get("run_id") == calibration["run_id"],
        "pinned provenance origin differs",
    )
    for key in ("manifest_sha256", "bundle_sha256"):
        expected = calibration["provenance_manifest_sha256" if key == "manifest_sha256" else key]
        require(
            isinstance(expected, str)
            and SHA.fullmatch(expected)
            and created.get(key) == uploaded.get(key) == expected,
            "pinned provenance digest differs",
        )
    require(
        calibration["provenance_dir"]
        == uploaded.get("remote_artifact")
        == REMOTE + "/provenance/" + calibration["provenance_manifest_sha256"],
        "remote provenance path differs",
    )
    runtime = document(request["runtime_evidence"])
    if "stdout" in runtime:
        require(runtime.get("returncode") == 0, "runtime check failed")
        runtime = json.loads(runtime["stdout"])
    require(
        runtime == plan["runtime"]
        and runtime.get("verified") is True
        and runtime.get("python") == PYTHON
        and runtime.get("sha256") == PYTHON_SHA
        and str(runtime.get("version", "")).startswith("3.12.13")
        and runtime.get("packages") == PACKAGES
        and runtime.get("gpu_imported") is False,
        "pinned runtime changed",
    )
    artifact = local_artifact(created["artifact"], root)
    expected_paths = {
        plan["preflight"]["submission"]["path"],
        plan["preflight"]["deployment"]["path"],
        ".sdsc/runs/" + calibration["run_id"] + ".json",
        ".sdsc/check.json",
    }
    expected_paths.update(
        str(safe(request[key]).relative_to(root))
        for key in ("provenance_created", "provenance_uploaded", "runtime_evidence")
    )
    expected_paths.update(
        str((artifact / name).relative_to(root)) for name in ("manifest.json", "wrapper-manifest.json")
    )
    require({row["path"] for row in evidence} == expected_paths, "required evidence omitted or substituted")
    return plan


def submit_arguments(plan):
    calibration = plan["calibration"]
    arguments = ["submit", calibration["run_id"], "--task", TASK]
    for key, value in RESOURCES.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    for key, value in {
        "python": PYTHON,
        "hf_home": HF_HOME,
        "result_root": RESULT_ROOT,
        "provenance_dir": calibration["provenance_dir"],
        "provenance_manifest_sha256": calibration["provenance_manifest_sha256"],
        "teacher_job_id": "54496291",
        "preflight_job_id": PREFLIGHT,
    }.items():
        arguments.extend(["--" + key.replace("_", "-"), value])
    return [*arguments, "--storage-confirmed", "--authorize"]


def command(arguments, plan, root=ROOT):
    require(
        arguments == submit_arguments(plan)
        or (
            len(arguments) == 2
            and arguments[0] in {"status", "fetch"}
            and re.fullmatch(r"[1-9][0-9]*", arguments[1])
        ),
        "operation is outside exact supervisor allowlist",
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(root / "tools/sdsc_cli.py"), *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=420,
    )
    require(completed.returncode == 0, "CLI stopped; no retry: " + completed.stderr[-2000:])
    require(len(completed.stdout) <= 4 * 1024**2, "CLI response exceeds bound")
    output = completed.stdout
    if output.startswith("submission_intent="):
        first, output = output.split("\n", 1)
        require(
            arguments[0] == "submit" and re.fullmatch(r"submission_intent=[a-f0-9]{32}", first),
            "unexpected submission preamble",
        )
    result = json.loads(output)
    require(isinstance(result, dict), "CLI returned non-object response")
    return result


def status_outcome(status, receipt):
    require(
        status.get("ok") is True and all(status.get(k) == receipt.get(k) for k in STATUS_KEYS),
        "status identity differs from immutable receipt",
    )
    require(
        status.get("queue", {}).get("returncode") == 0
        and status.get("accounting", {}).get("returncode") == 0,
        "scheduler query failed",
    )
    rows = [line.strip().split("|") for line in status["accounting"]["stdout"].splitlines() if line.strip()]
    job = receipt["job_id"]
    jobs = [row for row in rows if row[0] == job]
    steps = [row for row in rows if row[0].startswith(job + ".")]
    require(
        len(jobs) <= 1 and len(rows) == len(jobs + steps) and all(len(row) >= 3 for row in rows),
        "ambiguous/malformed accounting",
    )
    state = status.get("state", "UNKNOWN")
    accounting_complete = (
        len(jobs) == 1
        and {job + ".batch", job + ".extern"} <= {row[0] for row in steps}
        and len({row[0] for row in rows}) == len(rows)
        and all(row[1:3] == ["COMPLETED", "0:0"] for row in rows)
    )
    verified = status.get("result", {}).get("verified") is True and bool(
        SHA.fullmatch(str(status["result"].get("result_sha256", "")))
    )
    if status.get("success") is True:
        require(
            state == "COMPLETED" and verified,
            "success lacks verified scientific result",
        )
        require(
            not status["queue"]["stdout"].strip() and accounting_complete,
            "completion lacks successful main job, batch, extern and all steps",
        )
        return "success"
    queue_rows = [line.split("|") for line in status["queue"]["stdout"].splitlines() if line.strip()]
    if (
        status.get("success") is False
        and state == "COMPLETED"
        and verified
        and accounting_complete
        and len(queue_rows) == 1
        and len(queue_rows[0]) >= 3
        and queue_rows[0][:2] == [job, "COMPLETING"]
    ):
        # squeue is sampled before sacct. Permit only the observed publication
        # race, while preserving the empty-queue gate and a ten-minute bound.
        return "transition"
    normalized = str(state).split("+", 1)[0].split(" ", 1)[0]
    if normalized in TERMINAL:
        return "failure"
    require(state in ACTIVE, "unknown scheduler state; stop without retry")
    require(
        not any(
            row[1].split("+", 1)[0] in TERMINAL - {"COMPLETED"} or row[2] not in {"0:0", ""}
            for row in jobs + steps
        ),
        "accounting contains a failed step",
    )
    return "active"


def validate_fetch(result, receipt, status, root=ROOT):
    job = receipt["job_id"]
    require(
        result.get("job_id") == job
        and result.get("intent_id") == receipt["intent_id"]
        and type(result.get("bytes")) is int
        and 0 <= result["bytes"] <= 8 * 1024**2,
        "bounded fetch identity/size differs",
    )
    destination = safe(result["destination"])
    require(
        destination.parent == root / ".sdsc/fetched" / job and destination.name.startswith("fetch-"),
        "fetch escaped small-results directory",
    )
    names = result.get("files")
    report_name = (
        "adapted-preflight.json" if receipt["task"] == PREFLIGHT_TASK else "adapted-calibration.json"
    )
    allowed = {
        report_name,
        "receipt.json",
        "worker.log",
        "control-result.json",
        "train.log",
        "export.log",
        f"slurm-{job}.out",
        f"slurm-{job}.err",
    }
    require(
        isinstance(names, list) and len(names) == len(set(names)) and set(names) <= allowed,
        "unexpected fetch file",
    )
    actual_bytes = sum(len(read(destination / name, 1024**2)) for name in names)
    require(actual_bytes == result["bytes"], "fetched byte count differs")
    if status.get("success") is True:
        require(
            report_name in names
            and sha(read(destination / report_name, 1024**2)) == status["result"]["result_sha256"],
            "fetched report differs from semantically verified status",
        )
    return result


def run_flow(
    plan, write, *, root=ROOT, call=None, guard=None, sleep=time.sleep, clock=time.time, before_submit=None
):
    call = call or (lambda arguments: command(arguments, plan, root))
    guard = guard or (lambda: validate_plan(plan, root))
    before_submit = before_submit or (
        lambda: no_competitor(root, root / ".sdsc/supervision" / plan["flow_id"])
    )
    state = {
        "schema": SCHEMA,
        "plan_sha256": sha(canonical(plan) + b"\n"),
        "pid": os.getpid(),
        "quest_host": plan["quest_host"],
        "phase": "checking_preflight",
        "preflight_job_id": PREFLIGHT,
        "calibration_run_id": plan["calibration"]["run_id"],
        "submission_attempted": False,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
    }

    def record(**changes):
        state.update(changes, updated_at_unix=clock())
        write(dict(state))

    def boundary():
        guard()
        require(clock() < plan["expires_at_unix"], "supervision deadline reached; jobs remain untouched")

    def invoke(arguments):
        boundary()
        return call(arguments)

    def wait_success(receipt, phase):
        transitions = 0
        while True:
            status = invoke(["status", receipt["job_id"]])
            outcome = status_outcome(status, receipt)
            if outcome == "transition":
                transitions += 1
            record(phase=phase, observed_status=status, completion_transition_observations=transitions)
            require(transitions <= 2, "completion queue did not clear within two five-minute observations")
            require(not (transitions and outcome == "active"), "terminal accounting regressed to active")
            if outcome not in {"active", "transition"}:
                fetched = validate_fetch(invoke(["fetch", receipt["job_id"]]), receipt, status, root)
                record(last_fetch=fetched)
                require(
                    outcome == "success", "job failed scientific/accounting acceptance: " + receipt["job_id"]
                )
                return status
            boundary()
            sleep(min(POLL, max(0, plan["expires_at_unix"] - clock())))

    record()
    try:
        wait_success(plan["preflight"]["receipt"], "waiting_preflight")
        boundary()
        before_submit()
        record(phase="submitting_calibration", submission_attempted=True)
        receipt = invoke(submit_arguments(plan))
        validate_receipt(receipt, calibration=True, plan=plan)
        saved = binding(receipt["job_id"], root)
        require(saved["receipt"] == receipt, "returned submission differs from durable local receipt")
        record(phase="waiting_calibration", calibration_job_id=receipt["job_id"], calibration_binding=saved)
        prior_guard = guard

        def calibrated_guard():
            prior_guard()
            require(binding(receipt["job_id"], root) == saved, "calibration receipt/deployment changed")

        guard = calibrated_guard
        wait_success(receipt, "waiting_calibration")
        record(
            phase="calibration_complete", next_gate="complete_G0_requires_separate_review_and_actual_evidence"
        )
        return 0
    except BaseException as error:
        record(
            phase="stopped",
            error=f"{type(error).__name__}: {error}",
            no_retry=True,
            jobs_cancelled=False,
            submission_outcome_unknown=state["submission_attempted"] and not state.get("calibration_job_id"),
        )
        raise


def plan_file(path, expected, root=ROOT):
    path = safe(path)
    require(
        path.name == "plan.json" and path.parent.parent == root / ".sdsc/supervision",
        "plan must live in one local supervision directory",
    )
    require(SHA.fullmatch(expected or "") and sha(read(path)) == expected, "external plan SHA differs")
    plan = document(path)
    require(plan.get("flow_id") == path.parent.name, "plan flow directory differs")
    validate_plan(plan, root)
    return path, plan


def run(path, expected, *, root=ROOT, call=None, clock=time.time, sleep=time.sleep):
    path, plan = plan_file(path, expected, root)
    lock_path = safe(root / ".sdsc/student-supervision.lock")
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    previous_handlers = {}
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (path.parent / "state.json").exists(), "flow already started; do not re-arm")
        require(clock() < plan["expires_at_unix"], "plan already expired")
        no_competitor(root, path.parent)
        claims = safe(root / ".sdsc/student-supervision-claims")
        claims.mkdir(exist_ok=True, mode=0o700)
        claim_path = claims / ("preflight-" + PREFLIGHT + ".json")
        claim_sha = publish(
            claim_path,
            {
                "plan_sha256": expected,
                "plan": str(path),
                "preflight_job_id": PREFLIGHT,
                "run_id": plan["calibration"]["run_id"],
                "claimed_at_unix": clock(),
            },
        )

        def interrupted(number, _frame):
            raise InterruptedError(f"supervisor interrupted by signal {number}; no retry")

        for number in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[number] = signal.signal(number, interrupted)

        def guard():
            require(sha(read(path)) == expected, "immutable plan changed")
            require(sha(read(claim_path)) == claim_sha, "persistent submission claim changed")
            validate_plan(plan, root)
            no_competitor(root, path.parent, before_submit=False)

        return run_flow(
            plan,
            lambda value: publish(path.parent / "state.json", value, replace=True),
            root=root,
            call=call,
            clock=clock,
            sleep=sleep,
            guard=guard,
        )
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("validate", "run", "status"):
        command_parser = sub.add_parser(name)
        command_parser.add_argument("--plan", type=Path, required=True)
        command_parser.add_argument("--plan-sha256", required=True)
        if name == "run":
            command_parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = build_plan(document(args.request))
        path = safe(args.output)
        require(path == ROOT / ".sdsc/supervision" / plan["flow_id"] / "plan.json", "wrong plan destination")
        path.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
        digest = publish(path, plan)
        validate_plan(plan)
        print(
            json.dumps(
                {
                    "plan": str(path),
                    "plan_sha256": digest,
                    "started": False,
                    "submit_arguments": submit_arguments(plan),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        require(args.authorize, "existing user authorization must be acknowledged with --authorize")
        return run(args.plan, args.plan_sha256)
    else:
        path, plan = plan_file(args.plan, args.plan_sha256)
        result = {
            "valid": True,
            "plan_sha256": args.plan_sha256,
            "flow_id": plan["flow_id"],
            "expires_at_unix": plan["expires_at_unix"],
        }
        if args.command == "status":
            result["state"] = (
                document(path.parent / "state.json") if (path.parent / "state.json").exists() else None
            )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Student supervisor stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None
