#!/usr/bin/env python3
"""One reviewed infrastructure recovery; accepted name-cue science remains immutable."""

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

CONTRACT_PATH = "prereg/amendments/qwen3_student_name_cue_execution_recovery_v1.json"
CONTRACT_ID = "qwen3-student-name-cue-execution-recovery-v1"
CONTROL_PATHS = (
    "tools/sdsc_student_name_cue_recovery_contract.py",
    "tools/sdsc_student_name_cue_recovery.py",
    "tools/sdsc_student_name_cue_recovery_job.py",
    "tools/sdsc_student_name_cue_recovery_audit.py",
    "tools/sdsc_runtime_snapshot.py",
)
PARENT = {
    "acceptance_commit": "f1674f27b0b488ef1690fb5b67de639d64d3107f",
    "artifact_sha256": "fc6f77ba5b0944b4b9f06a99c4f35bb390443d6deec069c59a19ea87b4eca8ec",
    "implementation_commit": "39489f76da0e26c357f329ef7fb60f1ec11a0f2d",
    "protocol_path": "prereg/amendments/qwen3_student_name_cue_probe_v1.json",
    "protocol_sha256": "14bf71622418cdf15733ed8dbf777bc1c62b01ed7e676638238a60f3e3560bb1",
}
FAILED = {
    "intent_id": "43b6665d68c9b525320fc8e60ed16c2b",
    "job_id": "54681802",
    "memory_sha256": "9d4ad03c110207d4b1ed1bd1ea5216dbf4bb5871fc61e048bcd0420e2ae56cd2",
    "node_sha256": "fba001ea479f4484500721bcacebe5d679a589f6e1c656e10a5d93b2fa881274",
    "plan_sha256": "614f6f48bd498983ce820f9c86856b2494e349b8d9dc625b6f5fb75d431fe858",
    "publication_sha256": "a6750c0cd9afcbc9c42a0768f5f3ce32f37e2e0736ec1aa0ef038bbedea2074e",
    "startup_log_sha256": "aace40213a60ee25d5c1cb26a1382e38bac4861b3434267f051a5e0320261f58",
    "trace_sha256": "6c08b183a79352e95ba94de3d23cd73d06256027bbfc2c9ac9ec9bb0491e2f03",
}
FROZEN_DEPENDENCIES = {
    "prereg/amendments/qwen3_student_focus_lr_probe_v1.json": (
        "28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f"
    ),
    "prereg/amendments/qwen3_student_name_cue_probe_v1.json": (
        "fc6f77ba5b0944b4b9f06a99c4f35bb390443d6deec069c59a19ea87b4eca8ec"
    ),
    "src/posttrain_circuits/experiments/protocols/student_focus_lr_probe.py": (
        "45ae711b5044ccd84ec0b891f07f164484da98e8628c02f0b0862ca1589d172a"
    ),
    "src/posttrain_circuits/experiments/protocols/student_name_cue_probe.py": (
        "035e53b42d032a32998fcb6d985857d0651294dc78d2827c9d817f378f419606"
    ),
    "tools/sdsc_cli.py": "205858ec0990e2f178e7b7de465cf3dcb711ecad2d30b1057ecce695447c9eb4",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_provenance.py": "ebd6c21aa487e01d2d01b6ed0cae8df088e91781bf63d1670fbdff4dd2d95fc0",
    "tools/sdsc_provenance_upload_v2.py": "62913af491530d7ab2e30485718896b957c27425ad528fba0d3bf661cdccb8ed",
    "tools/sdsc_provenance_v2.py": "b1c031acd95d422cdc81f56f3f7a9f6d5beef6878f04abfec4c9b47342497d0e",
    "tools/sdsc_remote.py": "a754b4a3d36d5482731446a86581e0a99dbc62433fa5e65bdbcd17b57eb80bbc",
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_focus_lr_probe.py": (
        "166abb9769bccae419119a2c31081ff5fc7fb8469215c03542068f7e82c6c8f7"
    ),
    "tools/sdsc_student_focus_lr_probe_audit.py": (
        "871692dbdf5f9b971686d2b4a2ec301a4355b7742dcffc68070e4721d1f304bb"
    ),
    "tools/sdsc_student_focus_lr_probe_job.py": (
        "1a4b131a22fb976d8f3dedebae9440ca3ac282a24986033705f9606fb40d8d1c"
    ),
    "tools/sdsc_student_focus_lr_probe_startup.py": (
        "567ad446f028206c29b1dcffd209f23f1e7eedc5b615721582739ca7ca11faea"
    ),
    "tools/sdsc_student_focus_lr_probe_worker.py": (
        "c6c068f5ed409190090377b34a95acf9031c0b8d5df2a5794b4c2e38428fc029"
    ),
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_lr_probe.py": "89588f0f56f9082921eae593b702bd202b8535eee7e5dd1e72218d49b2f42910",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
    "tools/sdsc_student_name_cue_probe.py": (
        "a851a3ffa68fffef5b702c90786ab3a8903a58175d856ba7d096b81a7b6dfa62"
    ),
    "tools/sdsc_student_name_cue_probe_audit.py": (
        "ad94a1ce759ac443ca3ac9f49902fda3a3fe0c025c68605e33bc50d4431f7eb2"
    ),
    "tools/sdsc_student_name_cue_probe_job.py": (
        "7b7267a81f427bfc973fe9ea32fe7c5d093b09797c9bf980b6ce727de2834b3b"
    ),
    "tools/sdsc_student_name_cue_probe_startup.py": (
        "82ddc69ec96ee038599812ca852907ebe13bfa6623b5cce45d9a6468ad691e81"
    ),
    "tools/sdsc_student_name_cue_probe_worker.py": (
        "13efac8c8070e36890fd115198f2dc5e3d3b4f49fffe2d0ebdf3575c54029051"
    ),
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_prepare_worker.py": (
        "993456eba4d6388374f48e6dc5e5339d28e40f1665310666afbded2dea2931f1"
    ),
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_student_quality_probe.py": "e0a0a9c32fd78191ed114de8423b096efe9db12cffefa0e372d93bbab52107e9",
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
        schema="opd-student-name-cue-execution-recovery-v1",
        contract_id=CONTRACT_ID,
        parent_science=copy.deepcopy(PARENT),
        failed_job=copy.deepcopy(FAILED),
        controls=list(CONTROL_PATHS),
        frozen_dependencies=copy.deepcopy(FROZEN_DEPENDENCIES),
        scope=dict(
            scientific_change=False,
            training=False,
            model_data_seed_threshold_change=False,
            checkpoint_selection=False,
            automatic_retry=False,
            **dict.fromkeys(FLAGS, False),
        ),
        execution=dict(
            gpu_count=2,
            cpu_count=24,
            host_memory_gib=384,
            walltime_seconds=5400,
            worker_deadline_seconds=4800,
            runtime_staging_counts_inside_worker_deadline=True,
            publication_reserve_seconds=600,
            early_probe_whole_child_seconds=300,
            per_rank_threads=12,
            maximum_concurrently_allocatable_gpus=4,
            maximum_execution_fetch_mib=64,
            maximum_combined_fetch_mib=192,
            maximum_runtime_manifest_mib=16,
            inference_only=True,
            optimizer_steps=0,
            total_responses=1024,
            preserve_original_worker_and_auditor_bytes=True,
            original_raw_auditor_required=True,
            original_inner_plan_python_is_source_identity_only=True,
            relocated_runtime_path_explicit_in_separate_execution_evidence=True,
            separate_execution_publication_required=True,
            node_mount_and_memory_guards_required=True,
        ),
        runtime=dict(
            source_prefix=RUNTIME_PREFIX,
            python_relative_path="bin/python3.12",
            python_sha256=PYTHON_SHA256,
            method="exact_complete_prefix_snapshot_to_node_local_native_runtime",
            archive_format="manifest_described_sorted_regular_file_stream",
            immutable_manifest_and_archive_external_hashes_required=True,
            all_regular_file_bytes_modes_and_internal_relative_symlinks_verified=True,
            external_or_absolute_symlinks_rejected=True,
            no_install_rewrite_conda_unpack_container_or_glibc_change=True,
            actual_relocated_python_prefix_modules_and_native_dependencies_verified=True,
            original_package_versions_and_CUDA_NCCL_semantics_unchanged=True,
        ),
        claims=dict(
            maximum_recovery_submissions=1,
            key="original_science_identity_plus_fixed_failed_job54681802",
            key_independent_of_runtime_code_release_and_intent=True,
            original_failed_scientific_claim_preserved_and_verified=True,
            exact_failed_publication_and_no_inference_evidence_required=True,
            fresh_exact_terminal_accounting_and_empty_queue_required=True,
            failed_accounting_complete_success_flag_not_required=True,
            no_delete_rearm_auto_retry_or_changed_failure_parent=True,
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
        "from posttrain_circuits.experiments.protocols.student_name_cue_probe "
        "import resolve_student_name_cue_probe_protocol;"
        "v=resolve_student_name_cue_probe_protocol(r,git_dir=sys.argv[2],expected_head=sys.argv[3]);"
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
        "original accepted name-cue science did not resolve: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    binding = json.loads(result.stdout)
    require(
        all(binding.get(k) == v for k, v in PARENT.items() if k != "protocol_path")
        and binding.get("head") == head
        and len(binding.get("science_file_sha256", {})) == 168,
        "original name-cue science binding differs",
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
            "original name-cue accepted ancestry missing",
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
