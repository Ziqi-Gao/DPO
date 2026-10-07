#!/usr/bin/env python3
"""Foreground Slurm worker: verify, stage locally, execute, and durably publish.

All scientific commands are fixed code-owned stages. Public source and model
snapshots are copied to node-local disk; stable paths exist only in a private
container mount namespace. Publication records only new/changed files, retaining
immutable earlier versions and their hashes in the input inventory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdsc_pipeline_remote as control
import sdsc_pipeline_storage as s

CHILD = None


def interrupted(signum, _frame):
    raise RuntimeError("allocation termination signal " + str(signum))


def stop_child():
    global CHILD
    if CHILD is None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(CHILD.pid, signal.SIGTERM)
    try:
        CHILD.wait(timeout=90)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(CHILD.pid, signal.SIGKILL)
        CHILD.wait(timeout=15)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(CHILD.pid, signal.SIGKILL)


def execute(argv, cwd, env, log):
    global CHILD
    print(s.canonical({"phase": "execute", "argv": argv}).decode(), flush=True)
    with log.open("ab", buffering=0) as stream:
        CHILD = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code = CHILD.wait()
    stop_child()
    s.require(code == 0, "stage worker failed; see persisted worker.log (exit " + str(code) + ")")


def container_command(request, work, namespace, g0root, argv):
    spec = request["environment"]["container"]
    image = s.safe(spec["image"])
    info = image.stat()
    s.require((info.st_size, info.st_mtime_ns) == (spec["size"], spec["mtime_ns"]), "container image changed")
    prefix = s.safe(request["python"]).parent.parent
    command = [spec["runtime"], "exec", "--nv", "--cleanenv", "--no-home"]
    binds = [
        (work, work, "rw"),
        (work, namespace, "rw"),
        (prefix, prefix, "ro"),
        (work / "g0", g0root, "rw"),
        (work / "science", namespace / "science", "ro"),
        (work / "source/tools", namespace / "tools", "ro"),
        (work / "huggingface", namespace / "huggingface", "ro"),
        (work / "mib", namespace / "mib", "ro"),
    ]
    # The small pilot venv inherits the already verified SDSC Conda packages;
    # expose that exact base read-only, never copy or modify the active runtime.
    base_prefix = request["environment"]["runtime"].get("base_prefix")
    if base_prefix and Path(base_prefix) != prefix:
        base_prefix = s.safe(base_prefix)
        binds.append((base_prefix, base_prefix, "ro"))
    # G0 producer may add new evidence; its original calibration and teacher
    # inputs remain immutable in every namespace. Later pilot stages see all G0
    # evidence read-only and publish only their own pilot output.
    if request["stage"] == "g0":
        binds.extend(
            (work / "g0" / x, g0root / x, "ro")
            for x in ("canonical_sft", "dataset", "teacher_demos", "initial_checkpoint.pt")
        )
    else:
        binds.append((work / "g0", g0root, "ro"))
    for host, guest, mode in binds:
        command.extend(["--bind", f"{host}:{guest}:{mode}"])
    command.extend(["--pwd", str(namespace / "science"), spec["image"], *argv])
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "SINGULARITY_TMPDIR": str(work / "tmp")}
    for key, value in os.environ.items():
        if key.startswith("SLURM_") or key == "CUDA_VISIBLE_DEVICES":
            env["SINGULARITYENV_" + key] = value
    for key, value in {
        "TMPDIR": str(namespace / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(namespace / "huggingface"),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(24 // control.GPU[request["stage"]]),
        "MKL_NUM_THREADS": str(24 // control.GPU[request["stage"]]),
        "OPENBLAS_NUM_THREADS": str(24 // control.GPU[request["stage"]]),
        "NUMEXPR_NUM_THREADS": str(24 // control.GPU[request["stage"]]),
        "MIB_REPOSITORY": str(namespace / "mib"),
        "XDG_CACHE_HOME": str(namespace / "cache"),
        "TRITON_CACHE_DIR": str(namespace / "triton"),
        "TORCH_EXTENSIONS_DIR": str(namespace / "torch-extensions"),
    }.items():
        env["SINGULARITYENV_" + key] = value
    return command, env


def stage_proofs(request, work, namespace):
    proofs = work / "proofs"
    proofs.mkdir(exist_ok=True)
    statuses = {}
    for name, status in request.get("statuses", {}).items():
        folder = proofs / name
        folder.mkdir()
        logical = namespace / "proofs" / name
        terminal = status["accounting"]["stdout"].encode()
        (folder / "terminal.txt").write_bytes(terminal)
        submission_hash = s.atomic(folder / "submission.json", status["submission"])
        entry = {
            "job_id": status["job_id"],
            "status": status,
            "terminal_path": str(logical / "terminal.txt"),
            "terminal_sha256": s.sha(terminal),
            "submission_receipt_path": str(logical / "submission.json"),
            "submission_receipt_sha256": submission_hash,
            "publications": [],
            "reports": [],
        }
        for kind in ("publications", "reports"):
            for i, record in enumerate(status[kind]):
                source = s.safe(record["path"])
                target = folder / (kind + "-" + str(i) + ".json")
                s.copy(source, target, {"size": source.stat().st_size, "sha256": record["sha256"]})
                entry[kind].append({"path": str(logical / target.name), "sha256": record["sha256"]})
        statuses[name] = entry
    return statuses


def worker(request, work, namespace, job):
    stage = request["stage"]
    g0root = Path(request["g0_root"])
    python = request["python"]
    s.stage_release(request["run_id"], request["code_sha256"], work / "source")
    proof = Path(request["provenance_dir"])
    provenance = s.helper("sdsc_provenance").verify(
        proof, request["provenance_sha256"], request["code_sha256"], destination=work / "science"
    )
    s.require(provenance["git_head"] == request["science_head"], "restored science HEAD differs")
    s.copy(proof / "manifest.json", work / "provenance-manifest.json")
    cache = s.safe(request["hf_home"])
    s.stage_models(cache, work / "huggingface")
    vendor = request["environment"]["mib"]
    manifest = s.document(vendor["manifest"], vendor["manifest_sha256"])
    s.require(manifest["head"] == "b759df34433c9e31043ba9e02908ce0bf20e894f", "MIB HEAD differs")
    for entry in manifest["files"]:
        relative = s.relative(entry["path"])
        s.copy(Path(vendor["path"]) / relative, work / "mib" / relative, entry)
    selection = s.helper("sdsc_pipeline_inputs")
    selected = selection.select_inventory(
        request["base_inventory"], stage, os.environ.get("SLURM_ARRAY_TASK_ID")
    )
    s.atomic(
        work / "staging-selection.json",
        selection.selection_summary(request["base_inventory"], stage, os.environ.get("SLURM_ARRAY_TASK_ID")),
    )
    s.stage_inventory(
        selected,
        {"g0": work / "g0", "pilot": work / "qwen3-v2/pilot", "proofs": work / "proofs"},
    )
    stages = stage_proofs(request, work, namespace)
    for name in ("tmp", "cache", "qwen3-v2/pilot", "g0", "proofs"):
        (work / name).mkdir(parents=True, exist_ok=True)
    log = work / "worker.log"
    if stage == "g0":
        calibration = request["calibration"]
        for name, key in (
            ("calibration-report.json", "report"),
            ("calibration-publication.json", "publication_receipt"),
        ):
            source = s.safe(calibration[key + "_path"])
            s.copy(
                source,
                work / "proofs" / name,
                {"size": source.stat().st_size, "sha256": calibration[key + "_sha256"]},
            )
        resume = {
            "schema": "quest-sdsc-resume-request-v1",
            "target": {"run_id": request["run_id"], "code_sha256": request["code_sha256"], "job_id": job},
            "work_dir": str(work),
            "workspace": str(work / "g0"),
            "science_root": str(work / "science"),
            "hf_home": str(work / "huggingface"),
            "python": python,
            "python_sha256": request["g0_python_sha256"],
            "reference_report": str(work / "proofs/calibration-report.json"),
            "reference_report_sha256": calibration["report_sha256"],
            "publication_receipt": str(work / "proofs/calibration-publication.json"),
            "publication_receipt_sha256": calibration["publication_receipt_sha256"],
            "expected_science_head": request["science_head"],
            "expected_protocol_sha256": request["protocol_sha256"],
            "container": request["environment"]["container"],
            "namespace_root": str(namespace),
            "readonly_binds": {
                role: {"host": str(work / host), "guest": str(namespace / guest)}
                for role, host, guest in (
                    ("science", "science", "science"),
                    ("tools", "source/tools", "tools"),
                    ("hf_home", "huggingface", "huggingface"),
                    ("mib", "mib", "mib"),
                )
            },
        }
        resume_hash = s.atomic(work / "resume-request.json", resume)
        execute(
            [
                python,
                "-I",
                "-B",
                "-u",
                str(work / "source/tools/sdsc_resume.py"),
                "--request",
                str(work / "resume-request.json"),
                "--request-sha256",
                resume_hash,
                "--execute",
            ],
            work,
            dict(os.environ),
            log,
        )
        preflight = request["preflight2"]
        for source, target, expected in (
            (proof / "manifest.json", "provenance-manifest.json", request["provenance_sha256"]),
            (Path(preflight["report_path"]), "preflight.json", preflight["report_sha256"]),
            (
                Path(preflight["source_manifest_path"]),
                "preflight-source-manifest.json",
                preflight["source_manifest_sha256"],
            ),
        ):
            s.copy(source, work / "g0" / target, {"size": source.stat().st_size, "sha256": expected})
        invoke = {
            "target": resume["target"],
            "g0_root": str(g0root),
            "science_root": str(namespace / "science"),
            "reference_report": str(namespace / "proofs/calibration-report.json"),
            "reference_report_sha256": calibration["report_sha256"],
            "science_head": request["science_head"],
            "protocol_sha256": request["protocol_sha256"],
            "python": python,
            "hf_home": str(namespace / "huggingface"),
            "mib_repository": str(namespace / "mib"),
            "adapter_review": request["adapter_review"],
            "provenance_sha256": request["provenance_sha256"],
            "preflight": preflight,
        }
        digest = s.atomic(work / "g0-request.json", invoke)
        argv = [
            python,
            "-I",
            "-B",
            "-u",
            str(namespace / "tools/sdsc_g0.py"),
            "--request",
            str(namespace / "g0-request.json"),
            "--request-sha256",
            digest,
        ]
        command, environment = container_command(request, work, namespace, g0root, argv)
        execute(command, work, environment, log)
        return work / "g0/g0.json"
    if stage == "preflight4":
        argv = [
            python,
            "-B",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(namespace / "tools/sdsc_pilot_preflight.py"),
            "--science-root",
            str(namespace / "science"),
            "--work-dir",
            str(namespace),
            "--output-dir",
            str(namespace / "qwen3-v2/pilot/preflight4"),
            "--hf-home",
            str(namespace / "huggingface"),
            "--run-id",
            request["run_id"],
            "--code-sha256",
            request["code_sha256"],
        ]
        command, environment = container_command(request, work, namespace, g0root, argv)
        execute(command, work, environment, log)
        return work / "qwen3-v2/pilot/preflight4/preflight.json"
    g0 = s.document(work / "g0/g0.json")
    s.require(g0.get("passed") is True, "pilot cannot precede passed G0")
    execution = s.document(work / "g0/sdsc-execution.json")
    preflight4 = request["statuses"]["preflight4"]
    preflight_report = work / "qwen3-v2/pilot/preflight4/preflight.json"
    # The status wrapper separately verifies durable publication. Bind its
    # semantic result SHA to the actual preflight report, not the wrapper file.
    status = dict(preflight4)
    status["result"] = {"verified": preflight4["success"], "result_sha256": s.sha(s.read(preflight_report))}
    admission = {
        "schema": "quest-sdsc-pilot-inputs-v1",
        "flow_id": request["flow_id"],
        "stage": stage,
        "target": {"run_id": request["run_id"], "code_sha256": request["code_sha256"]},
        "pilot_root": str(namespace / "qwen3-v2/pilot"),
        "g0_root": str(g0root),
        "g0": {
            "report_sha256": s.sha(s.read(work / "g0/g0.json")),
            "replay": {
                "runtime_config": str(g0root / "sdsc-runtime-config.json"),
                "execution_evidence": str(g0root / "sdsc-execution.json"),
                "execution_evidence_sha256": s.sha(s.read(work / "g0/sdsc-execution.json")),
                "expected_protocol_sha256": request["protocol_sha256"],
                "expected_initial_sha256": s.identity(work / "g0/initial_checkpoint.pt")["sha256"],
                "expected_adapter_sha256": execution["adapter_review"]["files"]["sdsc_finalize_g0.py"],
            },
        },
        "preflight4": {
            "report_path": str(namespace / "qwen3-v2/pilot/preflight4/preflight.json"),
            "report_sha256": s.sha(s.read(preflight_report)),
            "status": status,
            "publication_receipt_path": stages["preflight4"]["publications"][0]["path"],
            "publication_receipt_sha256": stages["preflight4"]["publications"][0]["sha256"],
            "wrapper_report_path": stages["preflight4"]["reports"][0]["path"],
            "wrapper_report_sha256": stages["preflight4"]["reports"][0]["sha256"],
        },
        "stages": {
            (
                {
                    "train-cell": "training",
                    "teacher-shard": "teacher_shards",
                    "initial-circuits": "initial_circuits",
                    "final-circuits": "final_circuits",
                    "local-fork": "local_fork",
                    "pilot-inputs": "pilot_inputs",
                }.get(k, k)
            ): v
            for k, v in stages.items()
        },
    }
    digest = s.atomic(work / "pilot-admission.json", admission)
    argv = [
        python,
        "-I",
        "-B",
        "-u",
        str(namespace / "tools/sdsc_pilot.py"),
        "--stage",
        stage,
        "--science-root",
        str(namespace / "science"),
        "--work-dir",
        str(namespace),
        "--pilot-root",
        str(namespace / "qwen3-v2/pilot"),
        "--g0-root",
        str(g0root),
        "--hf-home",
        str(namespace / "huggingface"),
        "--python",
        python,
        "--mib-repository",
        str(namespace / "mib"),
        "--inputs",
        str(namespace / "pilot-admission.json"),
        "--inputs-sha256",
        digest,
        "--run-id",
        request["run_id"],
        "--code-sha256",
        request["code_sha256"],
        "--provenance-manifest",
        str(namespace / "provenance-manifest.json"),
        "--provenance-manifest-sha256",
        request["provenance_sha256"],
    ]
    command, environment = container_command(request, work, namespace, g0root, argv)
    execute(command, work, environment, log)
    suffix = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
    return work / "qwen3-v2/pilot" / ("sdsc-stage-" + stage + "-" + suffix + ".json")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    request = s.document(args.request, args.request_sha256)
    stage = request["stage"]
    s.require(stage in control.STAGES, "unsupported stage")
    job = os.environ.get("SLURM_JOB_ID", "")
    s.require(re.fullmatch(r"[1-9][0-9]*", job), "real Slurm allocation required")
    s.require(
        os.environ.get("SLURM_CPUS_PER_TASK") == "24" and os.environ.get("SLURM_MEM_PER_NODE") == "196608",
        "fixed CPU/memory allocation required",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    s.require(
        len(visible) == control.GPU[stage] and len(set(visible)) == len(visible) and all(visible),
        "allocation GPU visibility differs",
    )
    index = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
    if stage in control.COUNTS:
        s.require(
            index.isdigit()
            and int(index) in range(control.COUNTS[stage])
            and os.environ.get("SLURM_ARRAY_TASK_COUNT") == str(control.COUNTS[stage]),
            "wrong array identity",
        )
        logical_job = os.environ["SLURM_ARRAY_JOB_ID"] + "_" + index
    else:
        s.require(index == "single", "unexpected array")
        logical_job = job
    namespace = Path("/opd-pipeline") / request["flow_id"]
    destination = s.PROJECT / "pipeline-results" / request["flow_id"] / stage / index
    destination.mkdir(parents=True, exist_ok=False)
    s.persistent(destination, writable=True)
    scratch = s.safe(
        os.environ.get("TMPDIR") or "/scratch/" + pwd.getpwuid(os.getuid()).pw_name + "/job_" + job
    )
    s.require(
        scratch.is_dir() and scratch.stat().st_uid == os.getuid() and s.mount(scratch)["fstype"] in s.LOCAL,
        "owned node-local scratch required",
    )
    work = Path(tempfile.mkdtemp(prefix="opd-pipeline-", dir=scratch))
    for name in ("tmp", "proofs", "qwen3-v2/pilot"):
        (work / name).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    report = {
        "flow_id": request["flow_id"],
        "stage": stage,
        "task": "qwen3-v2-" + stage,
        "job_id": logical_job,
        "allocation_job_id": job,
        "run_id": request["run_id"],
        "code_sha256": request["code_sha256"],
        "passed": False,
        "exit_code": 1,
        "work_dir": str(work),
        "namespace": str(namespace),
        "persistent_result_dir": str(destination),
        "resources": control.resource_plan()[stage],
        "request_sha256": args.request_sha256,
    }
    try:
        inner_path = worker(request, work, namespace, job)
        inner = s.document(inner_path)
        s.require(inner.get("passed") is True, "scientific report is not passed")
        logical = (
            Path(request["g0_root"]) / inner_path.relative_to(work / "g0")
            if work / "g0" in inner_path.parents
            else namespace / inner_path.relative_to(work)
        )
        report.update(
            passed=True,
            exit_code=0,
            scientific_report={
                "path": str(logical),
                "work_relative_path": str(inner_path.relative_to(work)),
                "sha256": s.sha(s.read(inner_path)),
                "value": inner,
            },
        )
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        print(json.dumps(report), file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_child()
    # Writer processes are gone. Publish diagnostics first; large artifacts must
    # finish and pass read-back before the receipt can claim durable success.
    try:
        roots = {"g0": work / "g0", "pilot": work / "qwen3-v2/pilot"}
        if (work / "worker.log").is_file():
            s.copy(work / "worker.log", destination / "worker.log")
        if (work / "staging-selection.json").is_file():
            s.copy(work / "staging-selection.json", destination / "staging-selection.json")
        for folder in ("proofs", "sdsc-resume-control"):
            diagnostic = work / folder
            if diagnostic.is_dir():
                for name, entry in s.records(diagnostic).items():
                    if entry["size"] <= 4 * 1024**2:
                        s.copy(diagnostic / name, destination / "diagnostics" / folder / name, entry)
        s.publish(roots, request["base_inventory"], destination, report)
    except BaseException as error:
        report["passed"] = False
        report["exit_code"] = 1
        report["publication_error"] = str(error)
        s.atomic(destination / "publication-failure.json", report)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
