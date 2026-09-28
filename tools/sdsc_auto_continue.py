"""Finite Quest continuation after one independently verified teacher diagnostic.

The existing read-only watcher owns all remote observation. This gate consumes
its local evidence, then may start exactly one authorized Quest Codex process.
It never invokes Slurm, installs a service, or retries a launched continuation.
"""

# Chinese user-facing notices intentionally use Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "quest-sdsc-auto-continuation-v1"
SUPPORTED_JOB = "54489646"
SUPPORTED_RUN = "20260928T005810Z-15cffdded272-cf5d74b8"
SUPPORTED_CODE_SHA256 = "15cffdded27232359c518ed4672d3062535f3606126054e60bc97f22b1058260"
POPULATION_SHA256 = "377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b"
V7_INSTRUCTIONS_SHA256 = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
V5_INSTRUCTIONS_SHA256 = "55edb00c4d57197224c49dddfcb36ad1152640c8f8a19e2b223c1f3e968c504f"
PROMPT_IDS_SHA256 = "730b18e04a39ce7ec2e6e3cf94f43e182c70dad634ea05a6ebd0b8124ce42cdf"
PROMPT_FILE = "docs/sdsc_auto_continuation_task.md"
POLL_SECONDS = 300
STALE_SECONDS = 20 * 60
MAX_SECONDS = 14 * 24 * 3600
CHILD_SECONDS = 8 * 3600
OWN_CONTROLS = ("tools/sdsc_auto_continue.py", "tools/sdsc_auto_continue")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch = load_module(ROOT / "tools/sdsc_probe_watch.py", "sdsc_auto_watch")
require = watch.require
safe_path = watch.safe_path
read_bytes = watch.read_bytes
read_json = watch.read_json
canonical = watch.canonical
digest = watch.digest
atomic = watch.atomic
CONTROL_FILES = (*OWN_CONTROLS, *watch.CONTROL_FILES)
PLAN_KEYS = {
    "schema", "quest_root", "quest_host", "job_id", "binding", "watch_plan",
    "watch_plan_sha256", "control_sha256", "prompt_file", "prompt_sha256",
    "codex", "authorized", "poll_seconds", "stale_seconds", "child_seconds",
    "created_at_unix", "expires_at_unix",
}


def controls(root):
    return {name: digest(read_bytes(root / name)) for name in CONTROL_FILES}


def executable_identity(path):
    # Resolve once during preparation; subsequent validation rejects a changed
    # executable or symlink at that pinned real path. Never inspect credentials.
    path = safe_path(path)
    require(path.is_absolute() and os.access(path, os.X_OK), "Codex executable is not executable")
    checksum = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 1024**3, "Invalid Codex executable")
        while chunk := stream.read(4 * 1024 * 1024):
            checksum.update(chunk)
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Codex executable changed during hashing",
    )
    return {"realpath": str(path), "sha256": checksum.hexdigest(), "size": before.st_size}


def watcher_plan(path, root):
    path = safe_path(path)
    require(
        path.name == "plan.json" and path.parent.parent == root / ".sdsc/supervision",
        "Watcher plan must belong to this checkout's supervision directory",
    )
    raw = read_bytes(path)
    plan = json.loads(raw)
    watch.validate_plan(plan, root)
    require(plan["binding"]["receipt"]["task"] == watch.TASK, "Continuation requires a prompt diagnostic")
    return path, raw, plan


def validate_plan(plan, root):
    require(isinstance(plan, dict) and set(plan) == PLAN_KEYS, "Invalid continuation plan fields")
    require(plan["schema"] == SCHEMA and plan["authorized"] is True, "Continuation is not authorized")
    require(plan["job_id"] == SUPPORTED_JOB, "This reviewed continuation supports only job " + SUPPORTED_JOB)
    require(plan["quest_root"] == str(root), "Continuation belongs to another checkout")
    require(plan["quest_host"] == socket.gethostname().split(".")[0], "SSH master belongs to another host")
    require(plan["control_sha256"] == controls(root), "Pinned continuation controls changed")
    require(
        plan["poll_seconds"] == POLL_SECONDS and plan["stale_seconds"] == STALE_SECONDS
        and plan["child_seconds"] == CHILD_SECONDS,
        "Continuation cadence or execution limits changed",
    )
    require(
        all(type(plan[key]) in (int, float) and math.isfinite(plan[key])
            for key in ("created_at_unix", "expires_at_unix"))
        and plan["expires_at_unix"] - plan["created_at_unix"] == MAX_SECONDS,
        "Continuation must expire after fourteen days",
    )
    require(plan["prompt_file"] == PROMPT_FILE, "Continuation prompt path changed")
    prompt = read_bytes(root / PROMPT_FILE, 128 * 1024)
    require(prompt.strip() and digest(prompt) == plan["prompt_sha256"], "Pinned continuation prompt changed")
    require(
        isinstance(plan["codex"], dict) and set(plan["codex"]) == {"realpath", "sha256", "size"}
        and plan["codex"] == executable_identity(Path(plan["codex"]["realpath"])),
        "Pinned Codex executable changed",
    )
    path, raw, observed = watcher_plan(root / plan["watch_plan"], root)
    require(str(path.relative_to(root)) == plan["watch_plan"], "Noncanonical watcher plan path")
    require(digest(raw) == plan["watch_plan_sha256"], "Pinned watcher plan changed")
    require(
        plan["job_id"] == observed["job_id"] and plan["binding"] == observed["binding"],
        "Pinned job receipt or deployed source changed",
    )
    receipt = observed["binding"]["receipt"]
    require(receipt["run_id"] == SUPPORTED_RUN and receipt["code_sha256"] == SUPPORTED_CODE_SHA256,
            "Diagnostic is not the reviewed v7 source release")
    return observed


def claim_path(root, job_id):
    require(isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id), "Invalid job ID")
    return safe_path(root / ".sdsc/supervision" / (".auto-continue-" + job_id + ".claim.json"))


def refuse_previous_start(root, job_id):
    require(not claim_path(root, job_id).exists(), "Continuation already claimed; reconcile, never retry")
    for path in (root / ".sdsc/supervision").glob("*/state.json"):
        state = read_json(path)
        require(
            not (state.get("schema") == SCHEMA and state.get("job_id") == job_id),
            "Continuation already started for this job; never re-arm a stopped or interrupted flow",
        )


def prepare(job_id, watch_plan, flow_name, codex_bin, *, authorize=False, root=ROOT, now=None):
    require(authorize is True, "Explicit --authorize is required to prepare automatic continuation")
    require(job_id == SUPPORTED_JOB, "This reviewed continuation supports only job " + SUPPORTED_JOB)
    require(
        isinstance(flow_name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", flow_name),
        "Invalid continuation flow name",
    )
    require(flow_name not in {".", ".."}, "Invalid continuation flow name")
    path, raw, observed = watcher_plan(watch_plan, root)
    require(job_id == observed["job_id"], "Requested job differs from watcher")
    refuse_previous_start(root, job_id)
    created = time.time() if now is None else now
    prompt = read_bytes(root / PROMPT_FILE, 128 * 1024)
    plan = {
        "schema": SCHEMA, "quest_root": str(root),
        "quest_host": socket.gethostname().split(".")[0], "job_id": job_id,
        "binding": observed["binding"], "watch_plan": str(path.relative_to(root)),
        "watch_plan_sha256": digest(raw), "control_sha256": controls(root),
        "prompt_file": PROMPT_FILE, "prompt_sha256": digest(prompt),
        "codex": executable_identity(Path(codex_bin).resolve(strict=True)),
        "authorized": True, "poll_seconds": POLL_SECONDS, "stale_seconds": STALE_SECONDS,
        "child_seconds": CHILD_SECONDS, "created_at_unix": created,
        "expires_at_unix": created + MAX_SECONDS,
    }
    validate_plan(plan, root)
    directory = safe_path(root / ".sdsc/supervision" / flow_name)
    directory.mkdir(mode=0o700, exist_ok=False)
    atomic(directory / "plan.json", plan)
    atomic(directory / "prepared.json", {"plan_sha256": digest(read_bytes(directory / "plan.json"))})
    return directory / "plan.json", plan


def inspect_watcher(plan, root, now):
    source = root / plan["watch_plan"]
    observed = read_json(source)
    state = read_json(source.parent / "state.json")
    require(
        state.get("schema") == watch.SCHEMA and state.get("job_id") == plan["job_id"]
        and state.get("quest_host") == plan["quest_host"]
        and state.get("plan_sha256") == digest(canonical(observed)),
        "Watcher state identity differs",
    )
    updated = state.get("updated_at_unix")
    require(
        type(updated) in (int, float) and math.isfinite(updated) and 0 <= now - updated <= STALE_SECONDS,
        "Watcher state is stale or has a future timestamp; reconcile without restarting",
    )
    phase = state.get("phase")
    require(phase in {"starting", "waiting", "collecting_terminal_results", "finished"},
            "Watcher stopped or has an unknown state; no continuation")
    if phase == "waiting":
        status = state.get("observed_status", {})
        normalized = str(status.get("state", "UNKNOWN")).split("+", 1)[0].split(" ", 1)[0]
        require(normalized in watch.ACTIVE | watch.TERMINAL, "Watcher observed an unknown Slurm state")
    if phase != "finished":
        return {"phase": "waiting", "watcher_phase": phase}
    status_raw = read_bytes(source.parent / "terminal-status.json")
    status = json.loads(status_raw)
    fetched = state.get("fetch")
    require(isinstance(fetched, dict), "Finished watcher has no bounded fetch evidence")
    # Read the actual report/publication and replay exact Slurm/accounting/hash
    # checks. Never treat cached conclusion.json or report.passed as this gate.
    conclusion = watch.conclusion(observed, status, fetched, root)
    if conclusion["diagnostic_completed"]:
        raw = read_bytes(Path(fetched["destination"]) / watch.REPORT)
        require(digest(raw) == conclusion["report_sha256"],
                "Diagnostic report changed during gate inspection")
        report = json.loads(raw)
        ordered = report.get("ordered_prompt_ids")
        require(
            report.get("candidate_instructions_sha256") == V7_INSTRUCTIONS_SHA256
            and report.get("baseline_instructions_sha256") == V5_INSTRUCTIONS_SHA256
            and report.get("candidate_protocol") == "qwen3-v2-g0-candidate-e-seed-42-prompt-v7"
            and report.get("baseline_protocol") == "qwen3-v2-g0-candidate-e-seed-42-prompt-v5"
            and report.get("candidate_protocol_review") == "proposed"
            and report.get("population_sha256") == POPULATION_SHA256
            and report.get("sampling_request_seed") == 31415,
            "Diagnostic is not the reviewed v7 versus v5 comparison",
        )
        require(
            isinstance(ordered, list) and len(ordered) == 32
            and all(isinstance(value, str) for value in ordered) and len(set(ordered)) == 32
            and digest(canonical(ordered)) == PROMPT_IDS_SHA256,
            "Diagnostic prompt population differs from the frozen first 32 training prompts",
        )
        require(
            all(report.get(key) is False for key in (
                "accepted_science", "full_teacher_ready", "training_started",
                "readiness_artifact_produced", "g0_passed", "execution_class_certified", "resumable",
            )),
            "Diagnostic claims formal readiness or training",
        )
    return {
        "phase": "finished", "conclusion": conclusion,
        "terminal_status_sha256": digest(status_raw), "fetch_sha256": digest(canonical(fetched)),
    }


def qualifies(assessment):
    result = assessment.get("conclusion", {})
    arm = result.get("candidate", {})
    return (
        result.get("diagnostic_completed") is True
        and all(type(arm.get(key)) is int
                for key in ("attempt_count", "total_prompts", "prompts_with_success"))
        and arm["attempt_count"] == 256 and arm["total_prompts"] == 32
        and arm["prompts_with_success"] == 32
    )


def codex_argv(plan, directory):
    # --approve-for-me implies workspace-write + auto-review in the reviewed
    # local CLI. It conflicts with explicit --sandbox; never add bypass flags.
    return [
        plan["codex"]["realpath"], "exec", "--approve-for-me", "--skip-git-repo-check",
        "-C", plan["quest_root"], "--json", "-o", str(directory / "continuation-last-message.txt"), "-",
    ]


def private_file(path):
    descriptor = os.open(safe_path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    return os.fdopen(descriptor, "wb")


def notice(directory, text, filename="notice.md"):
    require(filename in {"notice.md", "launcher-notice.md"}, "Invalid notice filename")
    path = safe_path(directory / filename)
    temporary = safe_path(directory / ("." + filename + ".tmp"))
    with private_file(temporary) as stream:
        stream.write((text + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_notice(directory, text):
    """Preserve a child's scientific notice and supply a fallback when absent."""
    notice(directory, text, "launcher-notice.md")
    child_notice = directory / "notice.md"
    exists = child_notice.exists() or child_notice.is_symlink()
    if not exists:
        notice(directory, text)
    return exists


def stop_owned_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def run_flow(plan, directory, root, guard, *, master=None, spawn=None, clock=None, sleep=None):
    clock, sleep = clock or time.time, sleep or time.sleep
    spawn = spawn or subprocess.Popen
    state = {
        "schema": SCHEMA, "job_id": plan["job_id"], "quest_host": plan["quest_host"],
        "plan_sha256": digest(canonical(plan)), "pid": os.getpid(), "phase": "starting",
        "child_started": False, "training_started": None, "scientific_success": None,
        "no_retry": True, "no_slurm_operations_by_gate": True,
    }
    child = None

    def record(**changes):
        state.update(changes, updated_at_unix=clock())
        atomic(directory / "state.json", state)

    def boundary():
        guard()
        require(clock() < plan["expires_at_unix"], "Fourteen-day continuation deadline reached")

    record()
    try:
        while True:
            boundary()
            assessment = inspect_watcher(plan, root, clock())
            if assessment["phase"] == "finished":
                break
            record(phase="waiting", watcher_phase=assessment["watcher_phase"])
            sleep(min(POLL_SECONDS, plan["expires_at_unix"] - clock()))
        atomic(directory / "diagnostic-gate.json", assessment)
        if not qualifies(assessment):
            result = assessment["conclusion"]
            coverage = result.get("candidate", {}).get("prompts_with_success", "未知")
            message = (
                f"自动续跑已停止：作业 {plan['job_id']} 的诊断有效覆盖为 {coverage}/32，"
                "要求完整完成 256 个候选并覆盖 32/32 提示，且 Slurm 和持久化校验通过。"
                "未启动后续 Quest Codex，未提交或取消任何后续作业。请查看 diagnostic-gate.json。"
            )
            record(phase="stopped", reason="diagnostic_gate_failed", diagnostic=assessment)
            notice(directory, message)
            return 2
        boundary()
        if master is None:
            cli = load_module(root / "tools/sdsc_cli.py", "sdsc_auto_cli")
            cli.require_master()
        else:
            master()
        boundary()
        require(inspect_watcher(plan, root, clock()) == assessment, "Terminal evidence changed before launch")
        context = {
            "flow_directory": str(directory), "job_id": plan["job_id"],
            "run_id": plan["binding"]["receipt"]["run_id"],
            "code_sha256": plan["binding"]["receipt"]["code_sha256"],
            "watcher_plan": str(root / plan["watch_plan"]),
            "diagnostic_gate": str(directory / "diagnostic-gate.json"),
            "diagnostic_report_sha256": assessment["conclusion"]["report_sha256"],
            "final_report_path": str(directory / "continuation-last-message.txt"),
        }
        prompt = b"Verified continuation context (JSON):\n" + canonical(context) + b"\n\nTask instructions:\n"
        task_bytes = read_bytes(root / PROMPT_FILE, 128 * 1024)
        require(digest(task_bytes) == plan["prompt_sha256"], "Actual continuation stdin prompt changed")
        prompt += task_bytes
        claim = {**context, "intent_id": plan["binding"]["receipt"]["intent_id"],
                 "plan_sha256": state["plan_sha256"], "claimed_at_unix": clock(), "pid": os.getpid()}
        # Permanent across flow directories, crashes and unknown launch receipts.
        # Never remove this claim, even if spawning or reporting subsequently fails.
        with private_file(claim_path(root, plan["job_id"])) as stream:
            stream.write(canonical(claim) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        record(phase="launch_claimed", claim=str(claim_path(root, plan["job_id"])))
        argv = codex_argv(plan, directory)
        with private_file(directory / "continuation-stdout.jsonl") as stdout, \
                private_file(directory / "continuation-stderr.log") as stderr:
            with private_file(directory / "continuation-last-message.txt"):
                pass
            child = spawn(argv, cwd=root, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                          close_fds=True, start_new_session=True, umask=0o077)
            record(phase="continuation_running", child_started=True, child_pid=child.pid, argv=argv,
                   final_report_path=context["final_report_path"])
            child.communicate(input=prompt, timeout=min(CHILD_SECONDS, plan["expires_at_unix"] - clock()))
        record(phase="continuation_finished", child_returncode=child.returncode,
               final_report_path=context["final_report_path"])
        process_message = (
            f"Quest 自动续跑进程已结束，退出码 {child.returncode}。"
            "该退出码不代表训练或科学验证成功；请阅读 continuation-last-message.txt 并核查真实作业和结果。"
        )
        if process_notice(directory, process_message):
            require(read_bytes(directory / "notice.md", 128 * 1024).decode("utf-8").strip(),
                    "Continuation notice is empty")
            record(child_notice_preserved=True)
        else:
            record(child_notice_preserved=False)
        return 0 if child.returncode == 0 else 1
    except subprocess.TimeoutExpired:
        stop_owned_child(child)
        record(phase="stopped", reason="continuation_timeout",
               child_returncode=getattr(child, "returncode", None))
        message = ("自动续跑达到最长八小时或总期限，仅停止本流程启动的 Quest Codex 子进程。"
                   "未取消任何 SDSC 作业；须核对提交回执和作业状态，禁止盲目重启。")
        if state["child_started"]:
            process_notice(directory, message)
        else:
            notice(directory, message)
        return 124
    except (Exception, KeyboardInterrupt) as error:
        stop_owned_child(child)
        record(phase="stopped", reason="guard_failed", error=type(error).__name__ + ": " + str(error))
        message = ("自动续跑已停止：" + str(error)
                   + "。未重试或取消任何 SDSC 作业；请核查本流程状态和回执后处理。")
        if state["child_started"]:
            process_notice(directory, message)
        else:
            notice(directory, message)
        return 1


def run_plan(path, root=ROOT, **runtime):
    path = safe_path(path)
    require(path.name == "plan.json" and path.parent.parent == root / ".sdsc/supervision",
            "Invalid plan path")
    raw = read_bytes(path)
    plan = json.loads(raw)
    require(not (path.parent / "state.json").exists(), "Continuation already started; never re-arm")
    lock_path = claim_path(root, plan.get("job_id")).with_suffix(".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Continuation lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refuse_previous_start(root, plan["job_id"])
        seal = read_bytes(path.parent / "prepared.json")

        def guard():
            require(read_bytes(path) == raw, "Immutable continuation plan changed")
            require(read_bytes(path.parent / "prepared.json") == seal, "Preparation seal changed")
            require(json.loads(seal) == {"plan_sha256": digest(raw)}, "Prepared plan hash differs")
            validate_plan(plan, root)

        return run_flow(plan, path.parent, root, guard, **runtime)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare", help="authorize and pin one finite continuation")
    prepare_command.add_argument("--job-id", required=True)
    prepare_command.add_argument("--watch-plan", type=Path, required=True)
    prepare_command.add_argument("--flow", required=True)
    prepare_command.add_argument("--codex-bin", type=Path, required=True)
    prepare_command.add_argument("--authorize", action="store_true", required=True)
    run_command = commands.add_parser("run", help="consume local evidence and start at most one Codex")
    run_command.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path, plan = prepare(args.job_id, args.watch_plan, args.flow, args.codex_bin,
                             authorize=args.authorize)
        print(json.dumps({"plan": str(path), "plan_sha256": digest(canonical(plan)),
                          "job_id": plan["job_id"]}))
        return 0
    return run_plan(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
