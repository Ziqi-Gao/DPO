#!/usr/bin/env python3
"""Narrow execution-only recovery review; the accepted V4 science stays separate."""

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

CONTRACT_PATH = "prereg/amendments/qwen3_student_order_execution_recovery_v1.json"
CONTRACT_ID = "qwen3-student-order-execution-recovery-v1"
CONTROL_PATHS = tuple(
    "tools/sdsc_student_order_execution" + suffix + ".py" for suffix in ("_contract", "", "_job", "_worker")
)
PARENT = dict(
    protocol_path="prereg/amendments/qwen3_student_order_preparation_v4.json",
    protocol_sha256="b3517d592adfa25247072c6f7e39c703ce01cff272c47b9a49383af7890906d8",
    artifact_sha256="0642a565f0ec3c1d3d12c5aedf1157adc848a2ca6c766af9c7e439c06c86f55d",
    implementation_commit="8036f8af6cbb67edef6df1c8d8ff0cee65a1d84b",
    acceptance_commit="d4db85327f0165f2aec1ea53caefd1693663f9b2",
)
FROZEN_DEPENDENCIES = {
    "tools/sdsc_cuda_diagnostic.py": "b7b082ddad9595d8365522834ec02b691b92287c72f0a506b5ef2ed0871f181c",
    "tools/sdsc_cuda_diagnostic_job.py": "9572e9bcaa572a6d7995d483da0d24e8ce81f7ff496139d7f36119d669149e4b",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
}
FAILED = dict(
    job_id="54623814",
    plan_sha256="579e9e029568fca662a830c3e50496dac33b40ee8d71d0c91f7027ea1ff16630",
    publication_sha256="cf75c0bc2c533b148113103353a9c3e7982972c1235c10160bf800ce7272c396",
    report_sha256="de0f3152885fb97e15f46c4fb840bfa03dd805da4f90b617edf257ae01476ae6",
)
DIAGNOSTIC = dict(
    job_id="54626913",
    plan_sha256="d8c40f45e61d50ff25182b7870d4224f07ed342eaec9c53eb430074512bfefda",
    publication_sha256="f90a64d4085b77326cd346f374491cd6bb51820f21a459544563574f497eae84",
    report_sha256="1bc3c15fe285e9fd1ed28158309124c504774463f26fc28be6e7ec8e81867a60",
    node_sha256="41f1cf6526af04c14c84b3a9ec904789ee536265b8d7a5c6954c33cfa5a5ff7a",
    memory_sha256="0230292a345dda3c3db28d8974af6a9de8183391643dd9cba895f767ba9891c9",
)
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
        schema="opd-student-order-execution-recovery-v1",
        contract_id=CONTRACT_ID,
        parent_science=copy.deepcopy(PARENT),
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
            diagnostic_substitutes_for_recovery_preflight=False,
            **dict.fromkeys(FLAGS, False),
        ),
        execution=dict(
            early_probe_timeout_seconds=120,
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
            "frozen diagnostic dependency changed: " + path,
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
    require(
        text("rev-parse", "HEAD") == head and _read(root, CONTRACT_PATH) == raw,
        "execution HEAD/artifact changed during validation",
    )
    require(
        all(sha(_read(root, p)) == h for p, h in hashes.items()),
        "execution controls changed during validation",
    )
    core = copy.deepcopy(payload)
    core.pop("review")
    return ResolvedExecutionContract(
        payload, sha(canonical(core)), sha(raw), implementation, acceptance, head, hashes, science
    )


def validate_prerequisites(failed_status, diagnostic_status):
    """Check already freshly queried/byte-verified statuses; does no remote I/O."""
    for status, frozen in ((failed_status, FAILED), (diagnostic_status, DIAGNOSTIC)):
        require(
            isinstance(status, dict) and status.get("job_id") == frozen["job_id"], "prerequisite job differs"
        )
        require(
            status.get("queue", {}).get("returncode") == 0 and status["queue"].get("stdout") == "",
            "prerequisite queue not empty",
        )
        require(
            status.get("publication_verified") is True
            and status.get("artifact_hashes_verified") is True
            and status.get("publication_sha256") == frozen["publication_sha256"],
            "prerequisite publication not verified",
        )
        publication = status["publication"]
        require(
            publication.get("job_id") == frozen["job_id"]
            and publication.get("plan_sha256") == frozen["plan_sha256"]
            and all(status.get(k) is False and publication.get(k) is False for k in FLAGS),
            "prerequisite lineage or scientific scope differs",
        )
        rows = publication.get("files", [])
        require(
            isinstance(rows, list) and len({r["path"] for r in rows}) == len(rows),
            "prerequisite artifact inventory ambiguous",
        )
        records = {r["path"]: r for r in rows}
        name = "prepare-report.json" if frozen is FAILED else "cuda-diagnostic.json"
        require(records.get(name, {}).get("sha256") == frozen["report_sha256"], "prerequisite report differs")
        account = status.get("accounting", {})
        require(account.get("returncode") == 0, "prerequisite accounting unavailable")
        lines = [line.split("|") for line in account.get("stdout", "").splitlines() if line.strip()]
        job = frozen["job_id"]
        require(
            len(lines) == 3 and {r[0] for r in lines} == {job, job + ".batch", job + ".extern"},
            "prerequisite terminal steps incomplete",
        )
        expected = "FAILED" if frozen is FAILED else "COMPLETED"
        require(
            all(
                r[1:3]
                == (
                    ["COMPLETED", "0:0"]
                    if r[0].endswith(".extern") or expected == "COMPLETED"
                    else ["FAILED", "1:0"]
                )
                for r in lines
            ),
            "prerequisite terminal step outcome differs",
        )
        require(status.get("state") == expected, "prerequisite terminal state differs")
    require(
        failed_status.get("success") is False
        and failed_status["publication"].get("large_files") == []
        and failed_status["publication"].get("passed") is False,
        "failure is not preserved zero-update CUDA admission",
    )
    require(
        all(
            diagnostic_status.get(k) is True
            for k in ("terminal", "accounting_complete", "success", "diagnostic_complete", "cuda_ready")
        ),
        "diagnostic is not terminal and accepted as infrastructure evidence",
    )
    records = {r["path"]: r for r in diagnostic_status["publication"]["files"]}
    require(
        set(records) == {"cuda-diagnostic.json", "node-result.json", "memory.json"}
        and records["node-result.json"]["sha256"] == DIAGNOSTIC["node_sha256"]
        and records["memory.json"]["sha256"] == DIAGNOSTIC["memory_sha256"],
        "diagnostic raw evidence differs",
    )
