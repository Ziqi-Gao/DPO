#!/usr/bin/env python3
"""Reviewed consumer for one full fit after the successful fenced preparation preflight."""

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

CONTRACT_PATH = "prereg/amendments/qwen3_student_name_invariant_fit_execution_v1.json"
CONTRACT_ID = "qwen3-student-name-invariant-fit-execution-v1"
CONTROL_PATHS = (
    "tools/sdsc_student_name_invariant_fit_contract.py",
    "tools/sdsc_student_name_invariant_fit.py",
    "tools/sdsc_student_name_invariant_fit_job.py",
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
PREFLIGHT = {
    "job_id": "54692520",
    "intent_id": "00b0029c5bac2523196425ebb37a226b",
    "recovery_plan_sha256": "56d06d504637814e8b876ca6f2fe6c5c46cd70696aa7cd5590900783d572eeee",
    "plan_sha256": "9bf5a62880fd96501b2f2d326caf51ee177d228f7ebd754cd5a30f1852a67520",
    "publication_sha256": "aad46e81aee94b33f3e9af0863a8cd2a4ccdb8c5f6a875369fe4ace9fb7d66f5",
    "result_sha256": "9830f6bbcd873451e0a91257431fc39c7d7b22aef98826b41b4c593fb8777991",
    "audit_sha256": "a1360693838e1a1ebee8a9a38747df7eb553438903ec2e91497c19c722e2188d",
    "audit_size": 2544,
    "entry_sha256": "04fc52d27924bc035ce68d274fb3576ae2bb9109d80bb0eeb1a0aaba27a0a5e3",
    "exit_sha256": "538524c225686431d473483700eb2140bba469984dfe92b31b0c040ce2ebc802",
    "fence_sha256": "8dd9a2e4cd1b5dc716eaac32b6b293cb86f4b3c82f6f95177eb5a1ed2f392bfe",
    "science_head": "dbd97a27ad0b836144b9bab2671c1c8b7bc50cd4",
    "science_run_id": "20261006T051432Z-61ba795cb4c2-7b5da995",
    "science_code_sha256": "61ba795cb4c2849d831ea8013df6a295210a9dfdcfc7e21e6740a4e20ecb3cde",
    "science_provenance_manifest_sha256": "36d9b96474abf0289502e5d9337876ec35d417463cadd473906d74e50feea70f",
}
PARENT_EXECUTION = {
    "implementation_commit": "0fdc01f07e0e5a32ecac449e3dbc4b7a42c487bf",
    "acceptance_commit": "dbd97a27ad0b836144b9bab2671c1c8b7bc50cd4",
    "artifact_sha256": "e3b016317a015347498b8c0baf419dcc7f9244dbcc6e5da3213dc820010e568c",
    "core_sha256": "7dfbf7c36180d905ce918a52bbc2a67b2f637632f28985a291ebbb415ccf528b",
}
FROZEN_DEPENDENCIES = {
    "prereg/amendments/qwen3_student_name_invariant_execution_recovery_v1.json": "e3b016317a015347498b8c0baf419dcc7f9244dbcc6e5da3213dc820010e568c",
    "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json": "0346b251cc31370f862252a419751aea40ddbe0e138fa4c391bdc1c39170286b",
    "src/posttrain_circuits/experiments/protocols/student_name_invariant_preparation.py": "f448e9af9205fbf7788e1dd5bd5b7bdf82ec16f34d80c1812bd33f8a569e066d",
    "tools/sdsc_cli.py": "205858ec0990e2f178e7b7de465cf3dcb711ecad2d30b1057ecce695447c9eb4",
    "tools/sdsc_cuda_diagnostic_worker.py": "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b",
    "tools/sdsc_observed_command.py": "80034eaf8229f93a560ead6ee517412aee7d8fdd7b43aa61c1dd78c9efa7696a",
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
    "tools/sdsc_provenance_upload_v2.py": "62913af491530d7ab2e30485718896b957c27425ad528fba0d3bf661cdccb8ed",
    "tools/sdsc_provenance_v2.py": "b1c031acd95d422cdc81f56f3f7a9f6d5beef6878f04abfec4c9b47342497d0e",
    "tools/sdsc_remote.py": "a754b4a3d36d5482731446a86581e0a99dbc62433fa5e65bdbcd17b57eb80bbc",
    "tools/sdsc_runtime_snapshot.py": "fbf05d78819b13c1eeaab4937adb5a09f0af1b10ce73d89fca796e8aef5c1e99",
    "tools/sdsc_student_branch_qualify_job.py": "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3",
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_execution_fence.py": "b722756beead7dded50afdd92d2399285596bfa8e6b2e051e57a19e7fde4234e",
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_lr_probe.py": "89588f0f56f9082921eae593b702bd202b8535eee7e5dd1e72218d49b2f42910",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
    "tools/sdsc_student_name_invariant.py": "54d320d31905faae45ffed7e816b5b87ba20bac3ddec5b3ca11c1d33ff7d8918",
    "tools/sdsc_student_name_invariant_audit.py": "cf278c5477d17006098ad6c443e3431e078b252feffc6013d75099ec694e19eb",
    "tools/sdsc_student_name_invariant_job.py": "4d9b45d6a5ea6cba717982d32aebc434121c16735a768d8f196e5c050b1ad31f",
    "tools/sdsc_student_name_invariant_native.py": "94b6f1434bd617108c0cf22afa3f8ec54dff0bdb610387457a7b604e285bd609",
    "tools/sdsc_student_name_invariant_recovery.py": "27f22c0b90528fed9736fd23eded289c2887ad82a47adf511d16c509ff580806",
    "tools/sdsc_student_name_invariant_recovery_contract.py": "f2a189dd5e178fdd4e1ff9a8f7c6c324ebe4f75c6132bf438bdb87114cca5a98",
    "tools/sdsc_student_name_invariant_recovery_job.py": "a10c42e56f1d76118b18f96f5d045d8ac79c074961f0023aaddf6c81f367c36b",
    "tools/sdsc_student_name_invariant_startup.py": "d5993f6b0abf3981f1328b08e81b5078acf8fd7a43f35a7e942e8b9b62cf158f",
    "tools/sdsc_student_name_invariant_worker.py": "a4eed88edeef942dc8c9fabc7502f0a7c6a3e1ec0d5164d15799b571b3d80529",
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_prepare_worker.py": "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1",
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_student_quality_probe.py": "e0a0a9c32fd78191ed114de8423b096efe9db12cffefa0e372d93bbab52107e9",
    "tools/sdsc_torch_generated_origins.py": "022be69b30181f3af9187a646b16c6b04125c26f4c6297f06ce523046860d27a",
    "tools/sdsc_torch_import_probe_job.py": "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866",
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
        schema="opd-student-name-invariant-fit-execution-v1",
        contract_id=CONTRACT_ID,
        parent_science=copy.deepcopy(PARENT),
        parent_execution=copy.deepcopy(PARENT_EXECUTION),
        successful_preflight=copy.deepcopy(PREFLIGHT),
        controls=list(CONTROL_PATHS),
        frozen_dependencies=dict(sorted(FROZEN_DEPENDENCIES.items())),
        scope=dict(
            stage="original_full_fit_only",
            scientific_change=False,
            model_data_seed_threshold_change=False,
            checkpoint_selection_change=False,
            formal_qualification_authorized_by_this_contract=False,
            automatic_retry=False,
            **dict.fromkeys(FLAGS, False),
        ),
        dual_provenance=dict(
            inner_retains_exact_successful_preflight_science_source=True,
            outer_binds_distinct_reviewed_execution_source=True,
            both_full_genuine_v2_histories_verified=True,
            no_rewrite_of_inner_scientific_inputs_or_source=True,
        ),
        execution=dict(
            gpu_count=2,
            cpu_count=24,
            per_rank_threads=12,
            host_memory_gib=384,
            global_batch_size=64,
            optimizer_steps=32,
            learning_rate=2.5e-5,
            token_budget=2000000,
            walltime_seconds=28800,
            worker_deadline_seconds=28200,
            original_main_publication_reserve_seconds=600,
            wrapper_setup_seconds_before_entry_limit=60,
            wrapper_setup_is_outside_original_main_timer=True,
            wrapper_setup_consumes_original_allocation_reserve=True,
            wrapper_limit_checked_before_scientific_body_not_hard_io_interrupt=True,
            original_term_signal_seconds_before_allocation_end=180,
            maximum_submission_calls=1,
            submit_observation_timeout_seconds=180,
            bounded_stdout_stderr_evidence_before_receipt_required=True,
            incomplete_or_ambiguous_acknowledgement_stops_without_retry=True,
            unresolved_old_gpu_reservation=2,
            maximum_concurrently_allocatable_gpus=4,
            no_new_runtime_snapshot_or_package_installation=True,
            node_mount_memory_runtime_and_budget_guards_unchanged=True,
            original_worker_startup_native_auditor_bytes_preserved=True,
            explicit_versioned_node_orchestration_no_monkeypatch=True,
            fresh_original_initial_weights_not_preflight_checkpoint=True,
        ),
        preflight_validation=dict(
            exact_successful_recovery_identity_and_publication_required=True,
            original_full_raw_and_native_audit_required=True,
            terminal_successful_accounting_and_normal_wrapper_exit_required=True,
            persistent_artifacts_claims_and_retained_fence_reverified=True,
            original_unknown_preflight_scientific_claim_preserved=True,
            gpu_node_uses_bound_successful_admission_accounting_snapshot=True,
            no_repeat_of_successful_preflight=True,
            no_preflight_checkpoint_promotion=True,
        ),
        claims=dict(
            fresh_normal_fit_scientific_and_execution_claims=True,
            one_fit_stage_bound_to_original_science_identity=True,
            fresh_submission_and_result_identity_required=True,
            no_delete_rearm_auto_retry_or_replacement_chain=True,
            original_unknown_and_successful_recovery_claims_preserved=True,
            this_is_not_another_recovery_submission=True,
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
        "import json,sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r/'tools'));"
        "import sdsc_student_name_invariant_recovery_contract as p;"
        "v=p.resolve_execution_contract(r,git_dir=sys.argv[2],expected_head=sys.argv[3]).as_dict();"
        "print(json.dumps(v))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(root), str(metadata), head],
        capture_output=True,
        env=_git_environment(),
        timeout=240,
        check=False,
    )
    require(
        result.returncode == 0,
        "accepted preflight execution/science did not resolve: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    binding = json.loads(result.stdout)
    require(
        all(binding.get(k) == v for k, v in PARENT_EXECUTION.items()) and binding.get("head") == head,
        "accepted preflight execution lineage differs",
    )
    science = binding["science_binding"]
    require(
        all(science.get(k) == v for k, v in PARENT.items() if k != "protocol_path")
        and science.get("head") == head
        and len(science.get("science_file_sha256", {})) == 203,
        "original accepted science binding differs",
    )
    return science


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
            "original preparation accepted ancestry missing",
        )
        require(
            text("merge-base", PARENT_EXECUTION["acceptance_commit"], implementation)
            == PARENT_EXECUTION["acceptance_commit"],
            "successful preflight execution accepted ancestry missing",
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
