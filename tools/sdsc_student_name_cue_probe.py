#!/usr/bin/env python3
"""Paired bounded student diagnostic control; no retry or model acceptance.

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
TASK = "qwen3-v2-student-name-cue-probe-v1"
CAP = 1024**2
MAX_FILE = 16 * CAP
MAX_FETCH = 128 * CAP
MAX_RESPONSE = 192 * CAP
MAX_PROMPTS = 32 * CAP
MAX_AUDIT = 64 * 1024
PUBLICATION_RESERVE_SECONDS = 600
ARMS = ("control", "treatment")
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
PROTOCOL_PATH = "prereg/amendments/qwen3_student_name_cue_probe_v1.json"
PROTOCOL_MODULE = "src/posttrain_circuits/experiments/protocols/student_name_cue_probe.py"
FROZEN = {
    "tools/sdsc_student_focus_lr_probe_startup.py": (
        "567ad446f028206c29b1dcffd209f23f1e7eedc5b615721582739ca7ca11faea"
    ),
    "tools/sdsc_student_focus_lr_probe.py": (
        "166abb9769bccae419119a2c31081ff5fc7fb8469215c03542068f7e82c6c8f7"
    ),
    "tools/sdsc_student_focus_lr_probe_job.py": (
        "1a4b131a22fb976d8f3dedebae9440ca3ac282a24986033705f9606fb40d8d1c"
    ),
    "tools/sdsc_student_focus_lr_probe_worker.py": (
        "c6c068f5ed409190090377b34a95acf9031c0b8d5df2a5794b4c2e38428fc029"
    ),
    "tools/sdsc_student_focus_lr_probe_audit.py": (
        "871692dbdf5f9b971686d2b4a2ec301a4355b7742dcffc68070e4721d1f304bb"
    ),
    "tools/sdsc_student_prepare_worker.py": (
        "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1"
    ),
    "tools/sdsc_cli.py": "205858ec0990e2f178e7b7de465cf3dcb711ecad2d30b1057ecce695447c9eb4",
    "tools/sdsc_remote.py": "a754b4a3d36d5482731446a86581e0a99dbc62433fa5e65bdbcd17b57eb80bbc",
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
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
    "tools/sdsc_student_lr_probe.py": "89588f0f56f9082921eae593b702bd202b8535eee7e5dd1e72218d49b2f42910",
    "tools/sdsc_student_quality_probe.py": "e0a0a9c32fd78191ed114de8423b096efe9db12cffefa0e372d93bbab52107e9",
    "prereg/amendments/qwen3_student_focus_lr_probe_v1.json": (
        "28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f"
    ),
    "src/posttrain_circuits/experiments/protocols/student_focus_lr_probe.py": (
        "45ae711b5044ccd84ec0b891f07f164484da98e8628c02f0b0862ca1589d172a"
    ),
    "tools/sdsc_provenance_v2.py": "b1c031acd95d422cdc81f56f3f7a9f6d5beef6878f04abfec4c9b47342497d0e",
    "tools/sdsc_provenance_upload_v2.py": "62913af491530d7ab2e30485718896b957c27425ad528fba0d3bf661cdccb8ed",
}
TOOLS = (
    *FROZEN,
    "tools/sdsc_student_name_cue_probe.py",
    "tools/sdsc_student_name_cue_probe_job.py",
    "tools/sdsc_student_name_cue_probe_startup.py",
    "tools/sdsc_student_name_cue_probe_worker.py",
    "tools/sdsc_student_name_cue_probe_audit.py",
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
RAW_NAMES = (
    "name-cue-prompts.jsonl",
    "name-cue-dataset-manifest.json",
    "name-cue-data-isolation.json",
    "name-cue-token-audit.json",
    *(f"name-cue-records-{arm}-rank-{rank}.jsonl" for arm in ARMS for rank in (0, 1)),
)
NAMES = ("name-cue-report.json", *RAW_NAMES, "node-result.json", "memory.json", *EXECUTION_NAMES)

FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)


PARENTS = {
    "control": {
        "job_id": "54673886",
        "intent_id": "11c0db9b296008aeda5df5d008c2572f",
        "plan_sha256": "61ca76a29a0a96113852dc9bb370f280c9ad326135c6e5797776cd4997fe92d2",
        "publication_sha256": "431d2cff78c620caf652982bf9363071f4d81b1d5ccf5d97b8349a60f1d9ade9",
        "report_sha256": "94c609bfbb3e3afce1d7524ec93fbbf768789ecaafb1c7e9297d1923bf3f0acb",
        "checkpoint": {
            "path": "checkpoints/step-00000032.pt",
            "sha256": "bd3321d28e0f3f9780d9156a9b6e5df7fa90415f7827c24c348313944baaf7f6",
            "size": 8127108953,
        },
        "learning_rate": 5e-05,
    },
    "treatment": {
        "job_id": "54673887",
        "intent_id": "63b098982006ba6b1faaddfc790c9b8c",
        "plan_sha256": "11e2849a6dbf50a273916c3c8081e467f5e6285130b2dc86b2577becb0b1ccd3",
        "publication_sha256": "88345bb575423c0eb1276aeef25317bd086d2da0ec2c8ed828f6ab909cd8de8a",
        "report_sha256": "edf876a600cc6d7867ddd268538ded6c7984cf74371c267d8d8656e9eeebc808",
        "checkpoint": {
            "path": "checkpoints/step-00000032.pt",
            "sha256": "7d7f14ebe39ba2ccae1ee737aeefa9cbfd9143bcd6f2e8cbd69d36143a82d9f5",
            "size": 8127108953,
        },
        "learning_rate": 2.5e-05,
    },
}


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
_parent = helper("sdsc_student_focus_lr_probe")
require, canonical, sha, now = _io.require, _io.canonical, _io.sha, _io.now
safe, relative, write_once = _io.safe, _io.relative, _io.write_once
run = _lr.run
job_queue, job_accounting = _lr.job_queue, _lr.job_accounting
validate_accounting, live_binding = _lr.validate_accounting, _lr.live_binding
validate_job_memory_events = _lr.validate_job_memory_events


def validate_memory(report, job):
    require(
        isinstance(report, dict) and "after_inference" in report and "after_training" not in report,
        "inference memory phases differ",
    )
    return _lr.validate_memory(
        {("after_training" if name == "after_inference" else name): value for name, value in report.items()},
        job,
    )


CONFIG_SHA = _lr.CONFIG_SHA


def file_limit(name):
    return CAP if name in EXECUTION_NAMES else (MAX_PROMPTS if name == "name-cue-prompts.jsonl" else MAX_FILE)


def required_raw_names(mode):
    require(mode == "probe", "only inference diagnostic mode admitted")
    return set(RAW_NAMES)


def read(path, limit=None):
    return _io.read(path, limit=file_limit(Path(path).name) if limit is None else limit)


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "evidence SHA differs")
    return json.loads(raw)


def resources(mode="probe"):
    require(mode == "probe", "only diagnostic mode admitted")
    return dict(
        account="nwu181",
        partition="nairr-gpu-shared",
        qos="nairr-gpu-shared-normal",
        gpu_type="h100",
        gpus=2,
        cpus=24,
        mem_gib=384,
        time="01:30:00",
    )


def protocol_binding(root, expected_head=None):
    # A fresh interpreter prevents imports from another restored checkout being
    # reused as though they came from this genuine accepted scientific tree.
    code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "sys.path.insert(0,str(root/'src'));"
        "from posttrain_circuits.experiments.protocols.student_name_cue_probe "
        "import resolve_student_name_cue_probe_protocol;"
        "v=resolve_student_name_cue_probe_protocol(root,require_accepted=True,"
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


def science_identity(protocol):
    return dict(
        task=TASK,
        mode="probe",
        protocol_sha256=protocol["protocol_sha256"],
        parents=PARENTS,
        inference_only=True,
        optimizer_steps=0,
        base_examples=64,
        responses=1024,
        objective="name-cue-diagnostic-no-model-selection",
    )


def scientific_claim_path(plan):
    return (
        CONTROL
        / "student-name-cue-probe-scientific-claims"
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


def validate_plan(plan):
    require(
        len(canonical(plan)) <= MAX_PLAN
        and plan["schema"] == "quest-sdsc-student-name-cue-probe-plan-v1"
        and plan["task"] == TASK
        and plan["mode"] == "probe"
        and canonical(plan["resources"]) == canonical(resources()),
        "plan scope/resources differ",
    )
    require(
        set(plan)
        == {
            "schema",
            "task",
            "mode",
            "run_id",
            "code_sha256",
            "manifest_sha256",
            "created_at",
            "resources",
            "parents",
            "resolved_config",
            "protocol",
            "dataset_inputs",
            "control_sha256",
            "provenance",
            "science_identity",
            "python",
            "hf_home",
            "intent_id",
            "release",
            "submission_dir",
            "claim",
            "result_dir",
            "job_name",
            "scientific_claim",
            *FLAGS,
        },
        "plan fields differ",
    )
    require(plan["science_identity"] == science_identity(plan["protocol"]), "science identity differs")
    require(
        set(plan["control_sha256"]) == set(TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in plan["control_sha256"].values())
        and all(plan["control_sha256"][n] == h for n, h in FROZEN.items()),
        "control pins differ",
    )
    science = plan["protocol"].get("science_file_sha256")
    executed = {
        "tools/sdsc_student_name_cue_probe.py",
        "tools/sdsc_student_name_cue_probe_job.py",
        "tools/sdsc_student_name_cue_probe_startup.py",
        "tools/sdsc_student_name_cue_probe_worker.py",
        "tools/sdsc_student_name_cue_probe_audit.py",
        "tools/sdsc_provenance_v2.py",
        "tools/sdsc_provenance_upload_v2.py",
        PROTOCOL_MODULE,
    }
    require(
        isinstance(science, dict)
        and len(science) == 168
        and executed <= science.keys()
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in science.values()),
        "explicit 168-file accepted science required",
    )
    require(
        all(plan["control_sha256"][n] == h for n, h in science.items() if n in plan["control_sha256"]),
        "deployed control differs from accepted science",
    )
    require(
        plan["intent_id"] == execution_intent(plan)
        and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["run_id"])
        and re.fullmatch("[a-f0-9]{64}", plan["code_sha256"]),
        "execution identity differs",
    )
    for key, value in dict(
        release=CONTROL / "releases" / plan["run_id"],
        submission_dir=CONTROL / "student-name-cue-probe-submissions" / plan["intent_id"],
        claim=CONTROL / "student-name-cue-probe-claims" / (plan["intent_id"] + ".json"),
        result_dir=PROJECT / "student-name-cue-probe" / plan["intent_id"],
        scientific_claim=scientific_claim_path(plan),
    ).items():
        require(plan[key] == str(value), "unreviewed plan path: " + key)
    require(plan["job_name"] == "opd-snc-" + plan["intent_id"], "job name differs")
    require(set(plan["parents"]) == set(ARMS), "both exact diagnostic parents required")
    for arm, expected in PARENTS.items():
        row = plan["parents"][arm]
        require(
            set(row)
            == {"plan_path", "plan_sha256", "receipt", "publication_sha256", "report_sha256", "checkpoint"},
            "parent fields differ",
        )
        require(
            row["plan_path"]
            == str(CONTROL / "student-focus-lr-probe-submissions" / expected["intent_id"] / "plan.json")
            and all(
                row[k] == expected[k]
                for k in ("plan_sha256", "publication_sha256", "report_sha256", "checkpoint")
            ),
            "historical parent pins differ",
        )
        receipt = row["receipt"]
        require(
            receipt["job_id"] == expected["job_id"]
            and receipt["intent_id"] == expected["intent_id"]
            and receipt["plan_sha256"] == expected["plan_sha256"]
            and receipt["arm"] == arm
            and receipt["task"] == _parent.TASK
            and receipt["result_dir"] == str(PROJECT / "student-focus-lr-probe" / expected["intent_id"]),
            "historical parent receipt differs",
        )
    require(
        plan["resolved_config"]
        == dict(path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=CONFIG_SHA),
        "frozen native config differs",
    )
    contract = helper("sdsc_student_contract")
    require(
        plan["dataset_inputs"]
        == [dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES],
        "complete original dataset family required",
    )
    require(
        plan["python"] == str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
        and plan["hf_home"] == str(PROJECT / "cache/huggingface"),
        "runtime/cache differs",
    )
    provenance = plan["provenance"]
    require(
        set(provenance) == {"directory", "manifest_sha256", "head"}
        and re.fullmatch("[a-f0-9]{64}", provenance["manifest_sha256"])
        and provenance["directory"] == str(CONTROL / "provenance-v2" / provenance["manifest_sha256"])
        and re.fullmatch("[a-f0-9]{40}", provenance["head"])
        and provenance["head"] == plan["protocol"]["head"],
        "explicit v2 provenance differs",
    )
    require(
        plan["protocol"]["artifact_sha256"] == plan["control_sha256"][PROTOCOL_PATH]
        and plan["protocol"]["implementation_commit"] != plan["protocol"]["acceptance_commit"]
        and all(
            re.fullmatch("[a-f0-9]{40}", plan["protocol"][k])
            for k in ("implementation_commit", "acceptance_commit")
        ),
        "distinct accepted science required",
    )
    require(all(plan[k] is False for k in FLAGS), "diagnostic cannot claim acceptance")
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
        with tempfile.TemporaryDirectory(prefix="student-name-cue-verify-") as temp:
            return verify_science(plan, Path(temp) / "science")
    restored = helper("sdsc_provenance_v2").verify(
        safe(provenance["directory"]), provenance["manifest_sha256"], plan["code_sha256"], destination
    )
    require(restored["git_head"] == provenance["head"], "restored genuine HEAD differs")
    binding = protocol_binding(destination, provenance["head"])
    require(binding == plan["protocol"], "restored accepted successor differs")
    return restored


def verify_parent(plan, *, accounting=False):
    """Keep historical v1 verification isolated from this task's v2 consumer."""
    checked = []
    for arm in ARMS:
        row = plan["parents"][arm]
        parent_plan = _parent.validate_plan(document(safe(row["plan_path"]), row["plan_sha256"]))
        _parent.validate_submission_receipt(row["receipt"], parent_plan)
        _parent.verify_source(parent_plan)
        _parent.verify_science(parent_plan)
        root = safe(row["receipt"]["result_dir"])
        publication = document(root / "receipt.json", row["publication_sha256"])
        report = document(root / "prepare-report.json", row["report_sha256"])
        records = _parent.publication_records(publication, parent_plan, row["receipt"]["job_id"])
        for name, record in records.items():
            raw = _parent.read(root / name)
            require(
                len(raw) == record["size"] and sha(raw) == record["sha256"], "historical raw artifact differs"
            )
        _parent.validate_worker_report(report, parent_plan, row["receipt"]["job_id"], records)
        _parent.validate_update_records(_parent.read(root / "prepare-updates.jsonl"), report)
        _parent.validate_large_outputs(report, publication["large_files"])
        _parent.validate_memory(document(root / "memory.json"), row["receipt"]["job_id"])
        _parent.validate_startup_evidence(parent_plan, root, row["receipt"]["job_id"])
        require(
            row["checkpoint"] in publication["large_files"]
            and report["selected_checkpoint"] is None
            and report["preparation_complete"] is False,
            "diagnostic parent scope differs",
        )
        if accounting:
            state = _parent.inspect_job(parent_plan, row["receipt"])
            require(
                state["success"] is True and state["accounting_complete"] is True,
                "parent accounting incomplete",
            )
        checked.append(
            dict(
                arm=arm,
                job_id=row["receipt"]["job_id"],
                publication_sha256=row["publication_sha256"],
                v1_provenance_verified=True,
            )
        )
    return checked


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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_name_cue_probe_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_name_cue_probe_job.py"],
                str(Path(plan["submission_dir"]) / "plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def make_receipt(plan, job, reconciled=False):
    require(re.fullmatch("[1-9][0-9]*", job), "invalid job ID")
    return dict(
        task=TASK,
        mode="probe",
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


def scientific_specification(plan):
    """Use the reviewed protocol for numerical requirements, never a v1 fallback."""
    raw = read(ROOT / PROTOCOL_PATH)
    require(sha(raw) == plan["protocol"]["artifact_sha256"], "report protocol artifact differs")
    payload = json.loads(raw)
    core = {key: value for key, value in payload.items() if key != "review"}
    require(sha(canonical(core)) == plan["protocol"]["protocol_sha256"], "report protocol core differs")
    return payload


def worker_budget_seconds(plan):
    execution = scientific_specification(plan)["execution"]
    require(
        execution["publication_reserve_seconds"] == 600
        and execution["walltime_minutes"] == 90
        and execution["gpu_count"] == 2
        and execution["cpu_count"] == 24
        and execution["host_memory_gib"] == 384,
        "inference execution envelope differs",
    )
    return 4800


def checkpoint_inputs(plan, directory):
    return {
        arm: dict(
            path=str(Path(directory) / (arm + ".pt")),
            size=row["checkpoint"]["size"],
            sha256=row["checkpoint"]["sha256"],
            job_id=row["receipt"]["job_id"],
            step=32,
            learning_rate=PARENTS[arm]["learning_rate"],
        )
        for arm, row in plan["parents"].items()
    }


def validate_worker_report(report, plan, job, records):
    expected = dict(
        schema="quest-sdsc-student-name-cue-probe-report-v1",
        task=TASK,
        mode="probe",
        job_id=job,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        passed=True,
        diagnostic_complete=True,
        execution_complete=True,
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
    )
    require(
        canonical({k: report.get(k) for k in expected}) == canonical(expected),
        "incomplete or overclaiming inference report",
    )
    helper("sdsc_student_name_cue_probe_worker").validate_report(
        report,
        protocol=scientific_specification(plan),
        checkpoints=checkpoint_inputs(plan, Path("checkpoints")),
    )
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list) and len(raw) == len(RAW_NAMES) and {r["path"] for r in raw} == set(RAW_NAMES),
        "complete inference raw inventory required",
    )
    require(all(row == records.get(row["path"]) for row in raw), "raw artifact binding differs")


def publication_records(publication, plan, job):
    expected = dict(
        schema="quest-sdsc-student-name-cue-probe-publication-v1",
        task=TASK,
        mode="probe",
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        preparation_complete=False,
        selected_checkpoint=None,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=[],
        **dict.fromkeys(FLAGS, False),
    )
    require(
        canonical({k: publication.get(k) for k in expected}) == canonical(expected)
        and type(publication.get("diagnostic_complete")) is bool
        and publication.get("passed") is publication["diagnostic_complete"]
        and publication.get("stage_complete") is publication["diagnostic_complete"],
        "publication scope/identity differs",
    )
    rows = publication.get("files")
    require(isinstance(rows, list) and len(rows) <= len(NAMES), "publication inventory differs")
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
            "invalid published raw file",
        )
        records[name] = row
    require(
        sum(r["size"] for r in rows) <= MAX_FETCH - CAP
        and sum(records[n]["size"] for n in EXECUTION_NAMES if n in records) <= 2 * CAP,
        "bounded publication exceeded",
    )
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
            "accounting snapshot differs",
        )
        queue, account = accounting_snapshot["queue"], accounting_snapshot["accounting"]
    result = validate_accounting(plan, job, queue, account, observed)
    result.update(
        success=False,
        diagnostic_complete=False,
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
        publication = json.loads(raw)
        records = publication_records(publication, plan, job)
        require({n for n in NAMES if (root / n).exists()} == set(records), "published inventory differs")
        for name, row in records.items():
            data = read(root / name)
            require(
                len(data) == row["size"] and sha(data) == row["sha256"], "persistent raw artifact differs"
            )
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=publication,
            publication_sha256=sha(raw),
        )
    if result["accounting_complete"]:
        require(
            result["publication_verified"]
            and set(NAMES) == set(records)
            and publication["diagnostic_complete"] is True,
            "completed allocation lacks complete publication",
        )
        validate_worker_report(document(root / "name-cue-report.json"), plan, job, records)
        validate_memory(document(root / "memory.json"), job)
        validate_startup_evidence(plan, root, job)
        node = document(root / "node-result.json")
        require(
            node.get("stage_complete") is True
            and type(node.get("exit_code")) is int
            and node["exit_code"] == 0
            and node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and all(node.get(k) is False for k in FLAGS),
            "node result differs",
        )
        result.update(
            success=True,
            stage_complete=True,
            diagnostic_complete=True,
            result_sha256=sha(read(root / "name-cue-report.json")),
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
    return dict(runtime=verify_runtime(plan), gpu_concurrency=check_gpu_ceiling())


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
    path = str(Path(plan["release"]) / "source/tools/sdsc_student_name_cue_probe.py")
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
            plan["control_sha256"]["tools/sdsc_student_name_cue_probe.py"],
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
    require(record["state"] == "deployed", "first sync dry-run then sync")
    manifest = record["manifest"]
    records = manifest_records(manifest)
    require(
        record["deployment"]["code_sha256"] == manifest["code_sha256"]
        and all(sha(read(ROOT / n)) == records[n]["sha256"] for n in TOOLS),
        "local/deployed controls differ",
    )
    parents = {}
    for arm in ARMS:
        pins = PARENTS[arm]
        fetch = safe(getattr(args, arm + "_fetch_dir").absolute())
        status = document(safe(getattr(args, arm + "_status_file").absolute()))
        original = document(
            ROOT / ".sdsc/student-focus-lr-probe" / pins["intent_id"] / "plan.json", pins["plan_sha256"]
        )
        fetched = document(fetch / "fetch-manifest.json")
        receipt = fetched["receipt"]
        _parent.validate_plan(original)
        _parent.validate_submission_receipt(receipt, original)
        publication = document(fetch / "receipt.json", pins["publication_sha256"])
        report = document(fetch / "prepare-report.json", pins["report_sha256"])
        require(
            status.get("job_id") == pins["job_id"]
            and status.get("success") is True
            and status.get("accounting_complete") is True
            and status.get("publication_sha256") == pins["publication_sha256"]
            and status.get("result_sha256") == pins["report_sha256"],
            "parent terminal verification required",
        )
        rows = _parent.publication_records(publication, original, pins["job_id"])
        _parent.validate_worker_report(report, original, pins["job_id"], rows)
        _parent.validate_large_outputs(report, publication["large_files"])
        for name, row in rows.items():
            raw = _parent.read(fetch / name)
            require(len(raw) == row["size"] and sha(raw) == row["sha256"], "local parent raw hash differs")
        parents[arm] = dict(
            plan_path=str(CONTROL / "student-focus-lr-probe-submissions" / pins["intent_id"] / "plan.json"),
            plan_sha256=pins["plan_sha256"],
            receipt=receipt,
            publication_sha256=pins["publication_sha256"],
            report_sha256=pins["report_sha256"],
            checkpoint=pins["checkpoint"],
        )
    protocol = protocol_binding(ROOT, args.science_git_head)
    contract = helper("sdsc_student_contract")
    plan = dict(
        schema="quest-sdsc-student-name-cue-probe-plan-v1",
        task=TASK,
        mode="probe",
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        resources=resources(),
        parents=parents,
        resolved_config=dict(
            path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=CONFIG_SHA
        ),
        protocol=protocol,
        dataset_inputs=[
            dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES
        ],
        control_sha256={n: records[n]["sha256"] for n in TOOLS},
        provenance=dict(
            directory=args.provenance_dir,
            manifest_sha256=args.provenance_manifest_sha256,
            head=args.science_git_head,
        ),
        science_identity=science_identity(protocol),
        python=str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12"),
        hf_home=str(PROJECT / "cache/huggingface"),
        **dict.fromkeys(FLAGS, False),
    )
    intent = execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(CONTROL / "student-name-cue-probe-submissions" / intent),
        claim=str(CONTROL / "student-name-cue-probe-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-name-cue-probe" / intent),
        job_name="opd-snc-" + intent,
    )
    plan["scientific_claim"] = str(scientific_claim_path(plan))
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-name-cue-probe" / intent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        resources=plan["resources"],
        mode="probe",
    )


def validate_startup_evidence(plan, root, job):
    startup = helper("sdsc_student_name_cue_probe_startup")
    node = document(root / "node-result.json")
    work = Path(node["work_dir"])
    visible = node["cuda_visible_devices"]
    worker_sha = plan["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"]
    args = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(work / "artifacts")]
    for name, scope, rank in (
        ("early-node-startup.json", "early_node", None),
        ("rank-0-startup.json", "rank_entry", 0),
        ("rank-1-startup.json", "rank_entry", 1),
    ):
        raw_report = document(root / name)
        startup.validate_report(
            raw_report,
            worker_sha256=worker_sha,
            job_id=job,
            expected_cuda_visible_devices=visible,
            scope=scope,
            rank=rank,
            original_argv=args if rank is not None else None,
        )
        require(raw_report["cuda_ready"] is True, "raw startup CUDA observations failed")
    for rank in (0, 1):
        value = document(root / f"rank-{rank}-exit.json")
        require(
            value.get("schema") == startup.EXIT_SCHEMA
            and value.get("job_id") == job
            and type(value.get("rank")) is int
            and value["rank"] == rank
            and value.get("worker_invoked") is True
            and value.get("original_worker_sha256") == worker_sha
            and value.get("original_argv") == args
            and type(value.get("exit_code")) is int
            and value["exit_code"] == 0
            and "error" not in value
            and all(value.get(k) is False for k in FLAGS),
            "diagnostic rank exit failed",
        )
    trace = startup.validate_trace_meta(
        document(root / "trace-meta.json"),
        worker_sha256=worker_sha,
        job_id=job,
        expected_cuda_visible_devices=visible,
        expected_python=plan["python"],
        expected_argv=helper("sdsc_student_name_cue_probe_job").probe_argv(
            plan, work / "artifacts", dict(job_id=job, cuda_visible_devices=visible)
        ),
        log_bytes=read(root / "startup.log", CAP),
    )
    require(trace["probe_passed"] is True, "diagnostic early probe failed")
    require(
        sum((root / name).stat().st_size for name in EXECUTION_NAMES) <= 2 * CAP,
        "startup evidence exceeds2MiB",
    )
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--run-id")
    for name in (
        "control-fetch-dir",
        "treatment-fetch-dir",
        "control-status-file",
        "treatment-status-file",
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
                        args.control_fetch_dir,
                        args.treatment_fetch_dir,
                        args.control_status_file,
                        args.treatment_status_file,
                        args.provenance_dir,
                        args.provenance_manifest_sha256,
                        args.science_git_head,
                    )
                ),
                "prepare requires release/both parent fetches and statuses/v2 provenance",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-name-cue-probe" and path.name == "plan.json",
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
