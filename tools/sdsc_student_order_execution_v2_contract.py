#!/usr/bin/env python3
"""Reviewed execution successor for bounded cold-start observation; V4 science is immutable."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CONTRACT_PATH = "prereg/amendments/qwen3_student_order_execution_recovery_v2.json"
CONTRACT_ID = "qwen3-student-order-execution-recovery-v2"
CONTROL_PATHS = tuple(
    "tools/sdsc_student_order_execution_v2" + suffix + ".py" for suffix in ("_contract", "", "_job", "_probe")
)
PARENT = dict(
    protocol_path="prereg/amendments/qwen3_student_order_preparation_v4.json",
    protocol_sha256="b3517d592adfa25247072c6f7e39c703ce01cff272c47b9a49383af7890906d8",
    artifact_sha256="0642a565f0ec3c1d3d12c5aedf1157adc848a2ca6c766af9c7e439c06c86f55d",
    implementation_commit="8036f8af6cbb67edef6df1c8d8ff0cee65a1d84b",
    acceptance_commit="d4db85327f0165f2aec1ea53caefd1693663f9b2",
)
FROZEN_DEPENDENCIES = {
    "tools/sdsc_student_order_execution_contract.py": (
        "7163d33dee68c531dda03ea6e2a32136078e2557dfefd4c450a763929367b476"
    ),
    "tools/sdsc_student_order_execution.py": (
        "1692fae5b84070961de1d33d2774b75ac46e40ee5b0666adb798b8352ad2f1c7"
    ),
    "tools/sdsc_student_order_execution_job.py": (
        "ebf004742cca8f350ff090abfc8ce3fbb3d874a87623755014f9e8766783e51b"
    ),
    "tools/sdsc_student_order_execution_worker.py": (
        "366b606532b6a476cda53dfd847eccc083adf6da226626bc3d196e20f48448e5"
    ),
    "tools/sdsc_cuda_diagnostic.py": "b7b082ddad9595d8365522834ec02b691b92287c72f0a506b5ef2ed0871f181c",
    "tools/sdsc_cuda_diagnostic_job.py": "9572e9bcaa572a6d7995d483da0d24e8ce81f7ff496139d7f36119d669149e4b",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_torch_import_probe.py": "91db7d0d0fbf6a2257c11b70a969d9193db8e512f24b3e765cf3ffb9504c8917",
    "tools/sdsc_torch_import_probe_job.py": (
        "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866"
    ),
    "tools/sdsc_torch_import_probe_worker.py": (
        "b7f4cc07720172794e98106b5c03cf59e8657e88bd7b520e098bb65e561b0261"
    ),
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
}
PREVIOUS_EXECUTION = {
    "contract_path": "prereg/amendments/qwen3_student_order_execution_recovery_v1.json",
    "core_sha256": "4f745b038209cd5cdcd884e0acf3cf76b136e4699a41d6c6af6af62ef4aec06e",
    "artifact_sha256": "d433ac93a19eb2a52fbec91cd6a6f58e0800faaafb863804e9729f336569f782",
    "implementation_commit": "8086b38a1b7a8ad72f8acded8ef05157972c4613",
    "acceptance_commit": "23bc1aae3c68c14353d3a62bb88e6f516c8c9b65",
}
DIAGNOSTIC_IMPLEMENTATION_COMMIT = "c1b0b5904ebb509c5c3e442bd87ea9c56e5fdc31"
FAILED = {
    "job_id": "54643463",
    "plan_sha256": "34b45d2b888bf7bc6b7a96c72ce5a9b664a080356cfae2bc8f10dbd4eb9a730d",
    "publication_sha256": "d7ddc12da44da08189e057347b6be24dff23db1410a384513279f1e632c2002a",
    "report_sha256": "63099dc1c2d40c7cb4cffae46d53ad1511de72a7f316663f8312fa4837b79468",
    "execution_plan_sha256": "73478b7e5a45324a4f957d3ba60fe2f64053c75cb8ef63cb3db8ff04daf94c07",
    "execution_publication_sha256": "d4b4ee9effd7bc7ca842c09d384d882b31e40c83e80cc94bd3b22f1f9b289bfd",
    "execution_node_sha256": "0e5e39afdda0ea4238872ede638eabaa17415aaf4f4e452c6331d71b77a983ce",
}
DIAGNOSTIC = {
    "job_id": "54643629",
    "plan_sha256": "570e39613cff38031375ebbdea149adbc197dd17ad51be8fc74abeb2bd32f21d",
    "publication_sha256": "3ad6a79f3fbdc49b5270a333ca9aa59b3d14bf53797198bad30f5726badc18a2",
    "report_sha256": "9b1fc5ed109db6725b8d094333809c10de535ab88adbc01422780899ab62d1a9",
    "node_sha256": "27e167f0ccdac46b0c32cfb1b100d5ade379688210ec5a16a24afe3e5debabd4",
    "memory_sha256": "fde24b590b50b42a6580c16ecd718ae15c744d59641140c05228ec27dac6e541",
}
REVIEW_PROPOSED = dict(
    status="proposed",
    reviewed_implementation_commit=None,
    reviewer=None,
    reviewed_at_utc=None,
    rationale=None,
)
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
MAX_BYTES = 4 * 1024**2
COMMIT = re.compile(r"[a-f0-9]{40}\Z")


class ExecutionContractError(ValueError):
    """Execution-only recovery evidence or review was not admitted."""


def require(condition, message):
    if not condition:
        raise ExecutionContractError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def proposed_execution_contract():
    return dict(
        schema="opd-student-order-execution-recovery-v2",
        contract_id=CONTRACT_ID,
        parent_science=copy.deepcopy(PARENT),
        previous_execution=copy.deepcopy(PREVIOUS_EXECUTION),
        diagnostic_implementation_commit=DIAGNOSTIC_IMPLEMENTATION_COMMIT,
        failed_preflight=copy.deepcopy(FAILED),
        diagnostic=copy.deepcopy(DIAGNOSTIC),
        controls=list(CONTROL_PATHS),
        frozen_dependencies=copy.deepcopy(FROZEN_DEPENDENCIES),
        scope=dict(
            scientific_change=False,
            model_data_seed_threshold_change=False,
            teacher_or_formal_protocol_change=False,
            automatic_retry=False,
            diagnostic_identifies_original_node_cause=False,
            diagnostic_proves_thread_policy_necessary=False,
            same_node_aba_separates_cache_from_order=False,
            diagnostic_substitutes_for_recovery_preflight=False,
            **dict.fromkeys(FLAGS, False),
        ),
        execution=dict(
            early_probe_timeout_seconds=300,
            early_probe_timeout_covers_import_cuda_and_tiny_checks=True,
            early_probe_thread_environment=dict(
                OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12"
            ),
            early_probe_stack_dump_interval_seconds=30,
            early_probe_importtime=True,
            early_probe_timestamped_progress=True,
            retained_startup_log_max_bytes=1048576,
            retained_startup_log_truncation_and_raw_progress_required=True,
            frozen_v1_rank_training_wrapper_unchanged=True,
            early_probe_before_large_input_staging=True,
            early_probe_separate_child_reaped_before_training=True,
            rank_startup_predicates_before_unchanged_worker=True,
            preserve_original_cuda_visible_devices=True,
            preserve_original_worker_13_input_keys=True,
            preserve_original_scientific_report_and_raw_inventory=True,
            original_scientific_auditor_required=True,
            separate_execution_receipt_required=True,
            inner_science_and_outer_execution_plan_hashes_distinct=True,
            original_resources_walltime_publication_reserves_unchanged=True,
        ),
        claims=dict(
            maximum_preflight_submissions=1,
            maximum_fit_submissions=1,
            preflight_key="original_science_identity_with_stage+accepted_execution_core",
            matching_recovery_preflight_and_independent_8_response_audit_required=True,
            original_v4_fit_scientific_claim_also_required=True,
            failed_or_unknown_claims_never_deleted_or_rearmed=True,
            fresh_failed_and_diagnostic_terminal_reconciliation_required=True,
        ),
        review=copy.deepcopy(REVIEW_PROPOSED),
    )


def validate_execution_contract(payload, *, require_accepted=False):
    expected = proposed_execution_contract()
    require(isinstance(payload, dict) and set(payload) == set(expected), "execution contract fields differ")
    review = payload["review"]
    require(
        isinstance(review, dict) and set(review) == set(REVIEW_PROPOSED), "execution review fields differ"
    )
    if review.get("status") == "proposed":
        require(review == REVIEW_PROPOSED and not require_accepted, "execution contract is not accepted")
    else:
        require(review.get("status") == "accepted", "unknown execution review status")
        require(
            isinstance(review["reviewed_implementation_commit"], str)
            and COMMIT.fullmatch(review["reviewed_implementation_commit"]),
            "full execution implementation commit required",
        )
        require(
            all(isinstance(review[k], str) and review[k].strip() for k in ("reviewer", "rationale")),
            "execution reviewer and rationale required",
        )
        stamp = review["reviewed_at_utc"]
        require(
            isinstance(stamp, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", stamp),
            "UTC review timestamp required",
        )
        try:
            datetime.fromisoformat(stamp[:-1] + "+00:00")
        except ValueError as error:
            raise ExecutionContractError("invalid execution review date") from error
    expected["review"] = review
    require(canonical(payload) == canonical(expected), "execution contract differs from frozen recovery")
    return copy.deepcopy(payload)


def load_execution_contract(raw, *, require_accepted=False):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES, "invalid execution artifact bound")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate execution JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ExecutionContractError("invalid execution JSON") from error
    return validate_execution_contract(value, require_accepted=require_accepted)


def _read(root, name):
    path = root / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), "execution source symlink rejected")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_BYTES, "invalid execution source type/size"
        )
        raw = stream.read(MAX_BYTES + 1)
        require(len(raw) == info.st_size, "execution source changed during read")
        return raw


def _git_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    return env


def _parent_binding(root, metadata, head):
    code = (
        "import json,sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));"
        "from posttrain_circuits.experiments.protocols.student_order_preparation "
        "import resolve_student_order_preparation_protocol;"
        "v=resolve_student_order_preparation_protocol(r,git_dir=sys.argv[2],expected_head=sys.argv[3]);"
        "print(json.dumps({k:getattr(v,k) for k in ('protocol_sha256','artifact_sha256',"
        "'implementation_commit','acceptance_commit','head','science_file_sha256')}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(root), str(metadata), head],
        capture_output=True,
        env=_git_environment(),
        timeout=180,
        check=False,
    )
    require(
        result.returncode == 0,
        "original accepted V4 science did not resolve: " + result.stderr.decode("utf8", "replace")[-1500:],
    )
    binding = json.loads(result.stdout)
    require(
        all(binding.get(k) == v for k, v in PARENT.items() if k != "protocol_path")
        and binding.get("head") == head
        and len(binding.get("science_file_sha256", {})) == 130,
        "original V4 science binding differs",
    )
    return binding


@dataclass(frozen=True)
class ResolvedExecutionContract:
    payload: dict
    core_sha256: str
    artifact_sha256: str
    implementation_commit: str | None
    acceptance_commit: str | None
    head: str
    control_file_sha256: dict
    science_binding: dict

    def as_dict(self):
        return {
            k: copy.deepcopy(getattr(self, k))
            for k in (
                "core_sha256",
                "artifact_sha256",
                "implementation_commit",
                "acceptance_commit",
                "head",
                "control_file_sha256",
                "science_binding",
            )
        }


def resolve_execution_contract(repo_root, *, require_accepted=True, git_dir=None, expected_head=None):
    root = Path(repo_root).absolute()
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    require(root.is_dir() and root.resolve() == root, "real execution source root required")
    require(
        metadata in (root / ".git", root / ".opd-git") and metadata.is_dir() and not metadata.is_symlink(),
        "real execution Git metadata required",
    )

    def git(*args):
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(metadata),
                "--work-tree",
                str(root),
                "-C",
                str(root),
                *args,
            ],
            capture_output=True,
            env=_git_environment(),
            timeout=30,
            check=False,
        )
        require(result.returncode == 0, "execution Git query failed: " + args[0])
        return result.stdout

    def text(*args):
        return git(*args).decode().strip()

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    require(
        COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "execution HEAD differs"
    )
    require(
        text("rev-parse", "--is-shallow-repository") == "false", "complete execution review history required"
    )
    raw = _read(root, CONTRACT_PATH)
    payload = load_execution_contract(raw, require_accepted=require_accepted)
    science = _parent_binding(root, metadata, head)
    hashes = {p: sha(_read(root, p)) for p in CONTROL_PATHS}
    for path, digest in FROZEN_DEPENDENCIES.items():
        require(
            sha(_read(root, path)) == digest == sha(git("show", f"{head}:{path}")),
            "frozen execution dependency changed: " + path,
        )
    previous_path = PREVIOUS_EXECUTION["contract_path"]
    require(
        sha(_read(root, previous_path))
        == PREVIOUS_EXECUTION["artifact_sha256"]
        == sha(git("show", f"{head}:{previous_path}")),
        "accepted v1 execution artifact changed",
    )
    implementation = payload["review"]["reviewed_implementation_commit"]
    acceptance = None
    if payload["review"]["status"] == "accepted":
        require(
            implementation != head and text("merge-base", implementation, head) == implementation,
            "execution implementation must be a distinct ancestor",
        )
        proposed = load_execution_contract(git("show", f"{implementation}:{CONTRACT_PATH}"))
        require(
            proposed["review"] == REVIEW_PROPOSED, "execution implementation must contain proposed contract"
        )
        touches = text(
            "rev-list", "--reverse", "--ancestry-path", f"{implementation}..{head}", "--", CONTRACT_PATH
        ).splitlines()
        require(
            len(touches) == 1 and COMMIT.fullmatch(touches[0]), "exactly one later execution review required"
        )
        acceptance = touches[0]
        require(
            len(text("rev-list", "--parents", "-n", "1", acceptance).split()) == 2,
            "execution acceptance must have one parent",
        )
        changed = {
            p.decode()
            for p in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", acceptance).split(b"\0")
            if p
        }
        require(
            CONTRACT_PATH in changed and changed <= {CONTRACT_PATH, "docs/refactor/current_handoff.md"},
            "execution acceptance contains non-review changes",
        )
        accepted_raw = git("show", f"{acceptance}:{CONTRACT_PATH}")
        accepted = load_execution_contract(accepted_raw, require_accepted=True)
        require(
            accepted["review"]["reviewed_implementation_commit"] == implementation,
            "execution acceptance reviews another implementation",
        )
        accepted["review"] = copy.deepcopy(REVIEW_PROPOSED)
        require(canonical(accepted) == canonical(proposed), "execution acceptance changed contract core")
        require(
            raw == accepted_raw == git("show", f"{head}:{CONTRACT_PATH}"),
            "execution contract differs from accepted bytes",
        )
        require(
            not git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--",
                CONTRACT_PATH,
                *CONTROL_PATHS,
                *FROZEN_DEPENDENCIES,
                previous_path,
            ),
            "execution controls have staged changes",
        )
        for path, digest in hashes.items():
            entry = text("ls-tree", implementation, "--", path).split(maxsplit=1)
            require(entry and entry[0] in {"100644", "100755"}, "execution control is not a regular Git blob")
            require(
                sha(git("show", f"{implementation}:{path}")) == digest == sha(git("show", f"{head}:{path}")),
                "execution control differs from reviewed implementation: " + path,
            )
        require(
            text("merge-base", PARENT["acceptance_commit"], implementation) == PARENT["acceptance_commit"],
            "original V4 accepted ancestry missing",
        )
        for ancestor in (PREVIOUS_EXECUTION["acceptance_commit"], DIAGNOSTIC_IMPLEMENTATION_COMMIT):
            require(
                text("merge-base", ancestor, implementation) == ancestor,
                "accepted execution/diagnostic ancestry missing",
            )
        for path, digest in {
            **FROZEN_DEPENDENCIES,
            previous_path: PREVIOUS_EXECUTION["artifact_sha256"],
        }.items():
            entry = text("ls-tree", implementation, "--", path).split(maxsplit=1)
            require(entry and entry[0] in {"100644", "100755"}, "frozen dependency is not a regular Git blob")
            require(
                sha(git("show", f"{implementation}:{path}")) == digest,
                "implementation changed a frozen execution dependency: " + path,
            )
    require(
        text("rev-parse", "HEAD") == head and _read(root, CONTRACT_PATH) == raw,
        "execution HEAD/artifact changed during validation",
    )
    require(
        all(
            sha(_read(root, p)) == h
            for p, h in {
                **hashes,
                **FROZEN_DEPENDENCIES,
                previous_path: PREVIOUS_EXECUTION["artifact_sha256"],
            }.items()
        ),
        "execution controls changed during validation",
    )
    core = copy.deepcopy(payload)
    core.pop("review")
    return ResolvedExecutionContract(
        payload, sha(canonical(core)), sha(raw), implementation, acceptance, head, hashes, science
    )


def _terminal_status(status, frozen, *, failed):
    require(isinstance(status, dict) and status.get("job_id") == frozen["job_id"], "prerequisite job differs")
    queue, account = status.get("queue"), status.get("accounting")
    require(
        isinstance(queue, dict)
        and type(queue.get("returncode")) is int
        and queue["returncode"] == 0
        and queue.get("stdout") == "",
        "prerequisite queue not empty or unavailable",
    )
    require(
        isinstance(account, dict)
        and type(account.get("returncode")) is int
        and account["returncode"] == 0
        and isinstance(account.get("stdout"), str),
        "prerequisite accounting unavailable",
    )
    rows = [line.split("|") for line in account["stdout"].splitlines() if line.strip()]
    job = frozen["job_id"]
    require(
        len(rows) == 3
        and {row[0] for row in rows} == {job, job + ".batch", job + ".extern"}
        and all(
            row[1:3]
            == (["COMPLETED", "0:0"] if not failed or row[0].endswith(".extern") else ["FAILED", "1:0"])
            for row in rows
        ),
        "prerequisite terminal step outcome differs",
    )
    require(
        status.get("state") == ("FAILED" if failed else "COMPLETED"), "prerequisite terminal state differs"
    )
    require(status.get("success") is (not failed), "prerequisite success differs")
    # The frozen scientific accounting producer uses this name for zero-exit
    # accounting acceptance, not mere terminality; a FAILED job has False.
    require(status.get("accounting_complete") is (not failed), "prerequisite accounting completion differs")
    require(all(status.get(key) is False for key in FLAGS), "prerequisite scientific scope differs")


def _publication(status, frozen, *, execution=False):
    key = "execution_publication" if execution else "publication"
    digest_key = key + "_sha256"
    require(status.get(key + "_verified") is True, "prerequisite publication not verified")
    if not execution:
        require(status.get("artifact_hashes_verified") is True, "prerequisite artifact hashes not verified")
    value = status.get(key)
    require(
        isinstance(value, dict)
        and status.get(digest_key) == frozen[digest_key]
        and sha(canonical(value)) == frozen[digest_key],
        "prerequisite publication bytes differ",
    )
    require(
        value.get("job_id") == frozen["job_id"]
        and value.get("plan_sha256") == frozen["execution_plan_sha256" if execution else "plan_sha256"]
        and value.get("persistent_read_back_verified") is True
        and all(value.get(key) is False for key in FLAGS),
        "prerequisite publication lineage or scope differs",
    )
    rows = value.get("files")
    require(
        isinstance(rows, list)
        and all(
            isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and type(row.get("size")) is int
            and row["size"] > 0
            and isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
            for row in rows
        )
        and len({row["path"] for row in rows}) == len(rows),
        "prerequisite artifact inventory ambiguous",
    )
    return value, {row["path"]: row for row in rows}


def validate_prerequisites(failed_status, diagnostic_status):
    """Cross-bind freshly verified frozen producers; this does not perform I/O.

    The caller must use the immutable v1 combined execution inspector and import
    diagnostic inspector, including their raw-file, runtime and accounting
    validation, immediately before admission. Status dictionaries alone cannot
    establish freshness or replace either producer. Diagnostic completion grants
    no CUDA, model, scientific or recovery-preflight acceptance.
    """
    _terminal_status(failed_status, FAILED, failed=True)
    _terminal_status(diagnostic_status, DIAGNOSTIC, failed=False)
    require(
        diagnostic_status.get("terminal") is True, "diagnostic terminal observation missing or contradictory"
    )
    failed, records = _publication(failed_status, FAILED)
    execution, execution_records = _publication(failed_status, FAILED, execution=True)
    require(
        failed.get("task") == "qwen3-v2-student-order-preparation-v4"
        and failed.get("mode") == "preflight"
        and failed.get("large_files") == []
        and all(failed.get(key) is False for key in ("passed", "stage_complete", "preparation_complete"))
        and records.get("node-result.json", {}).get("sha256") == FAILED["report_sha256"],
        "failure is not preserved zero-update early-startup admission",
    )
    require(
        execution.get("task") == "qwen3-v2-student-order-execution-v1"
        and execution.get("science_plan_sha256") == FAILED["plan_sha256"]
        and execution.get("execution_core_sha256") == PREVIOUS_EXECUTION["core_sha256"]
        and execution.get("startup_passed") is False
        and execution_records.get("execution-node.json", {}).get("sha256") == FAILED["execution_node_sha256"]
        and all(
            failed_status.get(key) is False
            for key in (
                "startup_passed",
                "stage_complete",
                "preparation_complete",
                "scientific_stage_complete",
            )
        ),
        "failed combined execution evidence differs",
    )
    diagnostic, records = _publication(diagnostic_status, DIAGNOSTIC)
    require(
        diagnostic.get("task") == "sdsc-torch-import-probe-v1"
        and all(
            diagnostic_status.get(key) is True and diagnostic.get(key) is True
            for key in ("diagnostic_complete", "imports_ready")
        )
        and all(
            diagnostic.get(key) is False for key in ("cuda_observed", "model_loaded", "training_started")
        ),
        "diagnostic is not complete import-only infrastructure evidence",
    )
    expected = {"import-observations.json", "node-result.json", "memory.json"}
    expected.update(
        label + suffix for label in ("A1", "B", "A2") for suffix in ("-report.json", ".log", "-proc.jsonl")
    )
    require(
        set(records) == expected
        and all(
            records[name]["sha256"] == DIAGNOSTIC[key]
            for name, key in (
                ("import-observations.json", "report_sha256"),
                ("node-result.json", "node_sha256"),
                ("memory.json", "memory_sha256"),
            )
        ),
        "diagnostic retained observation graph differs",
    )
