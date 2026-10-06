#!/usr/bin/env python3
"""One fenced unknown-submission replacement; accepted preparation science remains immutable."""

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

CONTRACT_PATH = "prereg/amendments/qwen3_student_name_invariant_execution_recovery_v1.json"
CONTRACT_ID = "qwen3-student-name-invariant-execution-recovery-v1"
CONTROL_PATHS = (
    "tools/sdsc_student_name_invariant_recovery_contract.py",
    "tools/sdsc_student_name_invariant_recovery.py",
    "tools/sdsc_student_name_invariant_recovery_job.py",
    "tools/sdsc_student_execution_fence.py",
    "tools/sdsc_observed_command.py",
)
PARENT = {
    "acceptance_commit": "f7fc25c921f12e60b694508c01cb804114ffd24a",
    "artifact_sha256": "0346b251cc31370f862252a419751aea40ddbe0e138fa4c391bdc1c39170286b",
    "implementation_commit": "1d5b41ea188955a339533f3329948bcb9a066128",
    "protocol_path": "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json",
    "protocol_sha256": "58f66c4a114f249dc8b69e7172112dcffb94e3f0e1cc7d1f399db0b0c66a3a09",
}
OLD = {
    "intent_id": "19665dfbd26d7b0024a46d9b7b811ce4",
    "node_sha256": "4d9b45d6a5ea6cba717982d32aebc434121c16735a768d8f196e5c050b1ad31f",
    "plan_sha256": "b88f012fb8c30a9527d6300680dbabe29a944fdff64b29339a28f0c34764af5f",
}
FROZEN_DEPENDENCIES = {
    "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json": (
        "0346b251cc31370f862252a419751aea40ddbe0e138fa4c391bdc1c39170286b"
    ),
    "src/posttrain_circuits/experiments/protocols/student_name_invariant_preparation.py": (
        "f448e9af9205fbf7788e1dd5bd5b7bdf82ec16f34d80c1812bd33f8a569e066d"
    ),
    "tools/sdsc_cli.py": "205858ec0990e2f178e7b7de465cf3dcb711ecad2d30b1057ecce695447c9eb4",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
    "tools/sdsc_provenance_upload_v2.py": "62913af491530d7ab2e30485718896b957c27425ad528fba0d3bf661cdccb8ed",
    "tools/sdsc_provenance_v2.py": "b1c031acd95d422cdc81f56f3f7a9f6d5beef6878f04abfec4c9b47342497d0e",
    "tools/sdsc_remote.py": "a754b4a3d36d5482731446a86581e0a99dbc62433fa5e65bdbcd17b57eb80bbc",
    "tools/sdsc_runtime_snapshot.py": "fbf05d78819b13c1eeaab4937adb5a09f0af1b10ce73d89fca796e8aef5c1e99",
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_lr_probe.py": "89588f0f56f9082921eae593b702bd202b8535eee7e5dd1e72218d49b2f42910",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
    "tools/sdsc_student_name_invariant.py": (
        "54d320d31905faae45ffed7e816b5b87ba20bac3ddec5b3ca11c1d33ff7d8918"
    ),
    "tools/sdsc_student_name_invariant_audit.py": (
        "cf278c5477d17006098ad6c443e3431e078b252feffc6013d75099ec694e19eb"
    ),
    "tools/sdsc_student_name_invariant_job.py": (
        "4d9b45d6a5ea6cba717982d32aebc434121c16735a768d8f196e5c050b1ad31f"
    ),
    "tools/sdsc_student_name_invariant_native.py": (
        "94b6f1434bd617108c0cf22afa3f8ec54dff0bdb610387457a7b604e285bd609"
    ),
    "tools/sdsc_student_name_invariant_startup.py": (
        "d5993f6b0abf3981f1328b08e81b5078acf8fd7a43f35a7e942e8b9b62cf158f"
    ),
    "tools/sdsc_student_name_invariant_worker.py": (
        "a4eed88edeef942dc8c9fabc7502f0a7c6a3e1ec0d5164d15799b571b3d80529"
    ),
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_prepare_worker.py": (
        "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1"
    ),
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_student_quality_probe.py": "e0a0a9c32fd78191ed114de8423b096efe9db12cffefa0e372d93bbab52107e9",
    "tools/sdsc_torch_generated_origins.py": (
        "022be69b30181f3af9187a646b16c6b04125c26f4c6297f06ce523046860d27a"
    ),
    "tools/sdsc_torch_import_probe_job.py": (
        "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866"
    ),
}
RUNTIME_PREFIX = "/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-g0-py31213-cu128-v1"
PYTHON_SHA256 = "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19"
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
    """Unreviewed execution recovery or provenance."""


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
        schema="opd-student-name-invariant-execution-recovery-v1",
        contract_id=CONTRACT_ID,
        parent_science=copy.deepcopy(PARENT),
        old_unknown_submission=copy.deepcopy(OLD),
        controls=list(CONTROL_PATHS),
        frozen_dependencies=copy.deepcopy(FROZEN_DEPENDENCIES),
        scope=dict(
            scientific_change=False,
            stage="preflight_only",
            model_data_seed_threshold_change=False,
            checkpoint_selection=False,
            automatic_retry=False,
            fit_or_formal_qualification_authorized_by_this_contract=False,
            **dict.fromkeys(FLAGS, False),
        ),
        execution=dict(
            gpu_count=2,
            cpu_count=24,
            host_memory_gib=384,
            walltime_seconds=3600,
            worker_deadline_seconds=3300,
            original_main_publication_reserve_seconds=300,
            wrapper_setup_seconds_before_entry_limit=60,
            wrapper_setup_is_outside_original_main_timer=True,
            wrapper_setup_consumes_original_allocation_reserve=True,
            wrapper_limit_checked_before_original_main_not_hard_io_interrupt=True,
            original_term_signal_seconds_before_allocation_end=180,
            per_rank_threads=12,
            optimizer_steps=4,
            global_batch_size=64,
            maximum_concurrently_allocatable_gpus=4,
            unresolved_old_gpu_reservation=2,
            maximum_submission_calls=1,
            submit_observation_timeout_seconds=180,
            bounded_stdout_stderr_evidence_before_receipt_required=True,
            incomplete_or_ambiguous_acknowledgement_stops_without_retry=True,
            preserve_original_node_worker_startup_native_auditor_bytes=True,
            original_node_invoked_without_monkeypatch=True,
            original_raw_native_audit_required=True,
            node_mount_memory_runtime_and_budget_guards_unchanged=True,
            no_new_runtime_snapshot_or_package_installation=True,
        ),
        fence=dict(
            gate="exclusive_original_result_directory_mkdir_before_model_work",
            exact_old_plan_and_claims_bound=True,
            no_symlink_owned_persistent_path=True,
            only_this_operation_winning_atomic_mkdir_is_success=True,
            existing_directory_or_unknown_operation_never_adopted=True,
            bound_marker_directory_identity_durable_readback_required=True,
            permanent_retention_required=True,
            no_claim_reset_source_change_or_fabricated_receipt=True,
            does_not_resolve_original_scheduler_outcome=True,
            does_not_cancel_or_prevent_late_allocation=True,
            original_requested_gpu_hour_envelope=2,
            fresh_admission_and_new_node_verify_same_fence=True,
        ),
        claims=dict(
            maximum_recovery_submissions=1,
            key="fixed_old_intent_plus_original_science_identity",
            key_independent_of_runtime_code_release_and_new_intent=True,
            original_unknown_scientific_and_execution_claims_preserved=True,
            fresh_submission_and_result_identity_required=True,
            recovery_must_not_call_original_submit_admission=True,
            no_delete_rearm_auto_retry_or_replacement_chain=True,
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
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_OPTIONAL_LOCKS="0",
        LC_ALL="C",
    )
    return env


def _parent_binding(root, metadata, head):
    code = (
        "import json,sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));"
        "from posttrain_circuits.experiments.protocols.student_name_invariant_preparation "
        "import resolve_student_name_invariant_preparation_protocol;"
        "v=resolve_student_name_invariant_preparation_protocol(r,git_dir=sys.argv[2],expected_head=sys.argv[3]);"
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
        "original accepted name-invariant science did not resolve: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    binding = json.loads(result.stdout)
    require(
        all(binding.get(k) == v for k, v in PARENT.items() if k != "protocol_path")
        and binding.get("head") == head
        and len(binding.get("science_file_sha256", {})) == 203,
        "original name-invariant science binding differs",
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
            "original name-invariant accepted ancestry missing",
        )
        for path, digest in FROZEN_DEPENDENCIES.items():
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
            }.items()
        ),
        "execution controls changed during validation",
    )
    core = copy.deepcopy(payload)
    core.pop("review")
    return ResolvedExecutionContract(
        payload, sha(canonical(core)), sha(raw), implementation, acceptance, head, hashes, science
    )
