"""Qualify one independently audited order-prepared student under original gates.

The source is generic while a preparation is running. An unbound draft is valid
only as proposed documentation; an accepted protocol must freeze one actual
candidate. Operational parent completion and raw replay are separately required
by the controller and worker before that candidate can be used.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.git_provenance import _git_environment
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.teacher_adaptation_protocol import BASE_PREREG_SHA256, REVIEW_PROPOSED
from posttrain_circuits.experiments.protocols import student_order_preparation as preparation
from posttrain_circuits.experiments.protocols import student_qualification as original

PROTOCOL_PATH = "prereg/amendments/qwen3_student_order_qualification_v1.json"
PROTOCOL_ID = "qwen3-student-order-qualification-v1"
# Frozen scientific ancestry, never the source HEAD of a later recovered fit.
PARENT_HEAD = "d4db85327f0165f2aec1ea53caefd1693663f9b2"
PARENT_IMPLEMENTATION_COMMIT = "8036f8af6cbb67edef6df1c8d8ff0cee65a1d84b"
PARENT_PROTOCOL_SHA256 = "0642a565f0ec3c1d3d12c5aedf1157adc848a2ca6c766af9c7e439c06c86f55d"
PARENT_PROTOCOL_CORE_SHA256 = "b3517d592adfa25247072c6f7e39c703ce01cff272c47b9a49383af7890906d8"
PARENT_AUDITOR_SHA256 = "0b1ce443f468d16feb07490d12e6970cd232e3e73bb5f5791adf3ab82cedbb38"
RECOVERY_PATH = "prereg/amendments/qwen3_student_order_execution_recovery_v2.json"
RECOVERY_IMPLEMENTATION_COMMIT = "0848946f1942f0b844eecec4824cad104e2ff850"
RECOVERY_ACCEPTANCE_COMMIT = "bc3f0efa903a33e95a0878270f9c2650a1877020"
RECOVERY_CORE_SHA256 = "2515416bf1bebcf1ad36f78dc4033cf4adfa76ab29d018a11c8568469938cecb"
RECOVERY_ARTIFACT_SHA256 = "d565481421c914409e6ffaaef4e4a404e808665dcf77c1ee1a8d5963de615b3e"
RECOVERY_CONTROL_SHA256 = {
    "tools/sdsc_student_order_execution_v2_contract.py": (
        "1dee269e149c2693b9fcfa37f1f572ea8bb2fde9c744596b69904d78a9b23bac"
    ),
    "tools/sdsc_student_order_execution_v2.py": (
        "fcd74eb8c048830ecbb01fef0a4a3a57d354359b57d8958837abdd9bfca269da"
    ),
    "tools/sdsc_student_order_execution_v2_job.py": (
        "112245116eb02f0f413b09550252b1dff03b8093696feddb43cd1cc11ba2e51d"
    ),
    "tools/sdsc_student_order_execution_v2_probe.py": (
        "ade43340c9c06cc0948b92def7e91e47b6f1605a9fb04418a4d1feb1cbe84da3"
    ),
}
FROZEN_EXECUTION_SHA256 = {
    **RECOVERY_CONTROL_SHA256,
    RECOVERY_PATH: RECOVERY_ARTIFACT_SHA256,
    "tools/sdsc_cuda_diagnostic.py": "b7b082ddad9595d8365522834ec02b691b92287c72f0a506b5ef2ed0871f181c",
    "tools/sdsc_cuda_diagnostic_job.py": "9572e9bcaa572a6d7995d483da0d24e8ce81f7ff496139d7f36119d669149e4b",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
    "tools/sdsc_student_order_execution.py": (
        "1692fae5b84070961de1d33d2774b75ac46e40ee5b0666adb798b8352ad2f1c7"
    ),
    "tools/sdsc_student_order_execution_contract.py": (
        "7163d33dee68c531dda03ea6e2a32136078e2557dfefd4c450a763929367b476"
    ),
    "tools/sdsc_student_order_execution_job.py": (
        "ebf004742cca8f350ff090abfc8ce3fbb3d874a87623755014f9e8766783e51b"
    ),
    "tools/sdsc_student_order_execution_worker.py": (
        "366b606532b6a476cda53dfd847eccc083adf6da226626bc3d196e20f48448e5"
    ),
    "tools/sdsc_torch_import_probe.py": "91db7d0d0fbf6a2257c11b70a969d9193db8e512f24b3e765cf3ffb9504c8917",
    "tools/sdsc_torch_import_probe_job.py": (
        "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866"
    ),
    "tools/sdsc_torch_import_probe_worker.py": (
        "b7f4cc07720172794e98106b5c03cf59e8657e88bd7b520e098bb65e561b0261"
    ),
    "prereg/amendments/qwen3_student_order_execution_recovery_v1.json": (
        "d433ac93a19eb2a52fbec91cd6a6f58e0800faaafb863804e9729f336569f782"
    ),
}
# Named pure-function reuse belongs only to qualification, not the frozen fit.
FROZEN_QUALIFICATION_HELPER_SHA256 = {
    "tools/sdsc_student_branch_qualify_worker.py": (
        "35cd93ef1a36b7f53f4c887efe283bba5c15248c9cf4a836c1bcc74ab696515f"
    ),
}
FORMAL_PROMPTS_SHA256 = "ac58b320c219c8611943d84fa194c4658b24d641e665c6a9f3c9f0b0a783c51b"
FORMAL_PROMPTS_SIZE = 13_466_230
MAX_CHECKPOINT_BYTES = 8 * 1024**3
FLAGS = original.FLAGS
FROZEN_PROTOCOL_SHA256 = {
    **preparation.FROZEN_PROTOCOL_SHA256,
    preparation.PROTOCOL_PATH: PARENT_PROTOCOL_SHA256,
}
FROZEN_SCIENCE_PATHS = tuple(preparation.SCIENCE_PATHS)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    *FROZEN_QUALIFICATION_HELPER_SHA256,
    "src/posttrain_circuits/experiments/protocols/student_order_qualification.py",
    "tools/sdsc_student_order_qualify.py",
    "tools/sdsc_student_order_qualify_job.py",
    "tools/sdsc_student_order_qualify_worker.py",
    "tools/sdsc_student_order_qualify_audit.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 4 * 1024 * 1024
PREPARED_INITIAL_POLICY = {
    "selection": "frozen_earliest_development_candidate_before_this_candidate_qualification",
    "checkpoint_reselection_or_retraining": False,
    "load": "exact_FP32_saved_masters_before_one_native_BF16_inference_copy",
    "identity": "dense_checkpoint_bytes_never_native_HF_revision",
}
PREPARED_INITIAL_FIELDS = frozenset(
    {
        "fit_job_id",
        "fit_intent",
        "fit_plan_sha256",
        "fit_source_head",
        "fit_execution_plan_sha256",
        "fit_execution_publication_sha256",
        "fit_publication_sha256",
        "fit_report_sha256",
        "independent_raw_audit_sha256",
        "checkpoint_step",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size",
        "master_model_state_sha256",
        *PREPARED_INITIAL_POLICY,
    }
)


class StudentOrderQualificationError(ValueError):
    """The candidate, original gates or immutable review lineage is invalid."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentOrderQualificationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def validate_prepared_initial(value: Any, *, require_bound: bool = True) -> dict[str, Any] | None:
    """Validate identity shape, not substitute for actual parent evidence replay.

    Only an explicitly unbound *proposed* draft may contain None. All production
    entrypoints use the default bound contract. The reviewed JSON freezes these
    bytes; the controller/worker must compare them with the successful parent,
    its independent 18,432-response audit and the unique selected checkpoint.
    """
    if value is None:
        _require(not require_bound, "prepared initial is not bound to an audited fit")
        return None
    _require(
        isinstance(value, dict) and set(value) == PREPARED_INITIAL_FIELDS, "prepared initial fields differ"
    )
    _require(
        isinstance(value["fit_job_id"], str) and re.fullmatch(r"[1-9][0-9]{0,11}", value["fit_job_id"]),
        "invalid fit job identity",
    )
    _require(
        isinstance(value["fit_intent"], str)
        and re.fullmatch(r"[0-9a-f]{32}", value["fit_intent"])
        and value["fit_intent"] != "0" * 32,
        "invalid fit intent",
    )
    _require(
        isinstance(value["fit_source_head"], str)
        and _COMMIT.fullmatch(value["fit_source_head"])
        and value["fit_source_head"] not in {"0" * 40, PARENT_HEAD},
        "prepared initial must bind the actual recovered fit source HEAD",
    )
    for key in (
        "fit_execution_plan_sha256",
        "fit_execution_publication_sha256",
        "fit_plan_sha256",
        "fit_publication_sha256",
        "fit_report_sha256",
        "independent_raw_audit_sha256",
        "checkpoint_sha256",
        "master_model_state_sha256",
    ):
        _require(
            isinstance(value[key], str) and _SHA256.fullmatch(value[key]) and value[key] != "0" * 64,
            f"invalid prepared initial digest: {key}",
        )
    _require(
        value["fit_execution_plan_sha256"] != value["fit_plan_sha256"]
        and value["fit_execution_publication_sha256"] != value["fit_publication_sha256"],
        "execution and scientific artifacts must remain distinct",
    )
    step = value["checkpoint_step"]
    _require(
        type(step) is int and step in preparation.CHECKPOINT_STEPS,
        "prepared initial step is not a V4 development checkpoint",
    )
    _require(
        value["checkpoint_path"] == f"checkpoints/step-{step:08d}.pt",
        "prepared initial checkpoint path differs from selected step",
    )
    _require(
        type(value["checkpoint_size"]) is int and 0 < value["checkpoint_size"] <= MAX_CHECKPOINT_BYTES,
        "prepared initial checkpoint size exceeds staging bound",
    )
    _require(
        _canonical({key: value[key] for key in PREPARED_INITIAL_POLICY})
        == _canonical(PREPARED_INITIAL_POLICY),
        "prepared initial policy differs",
    )
    return copy.deepcopy(value)


def validate_parent_recovery_evidence(
    prepared_initial: Any,
    *,
    execution_plan: Any,
    status: Any,
    execution_publication_raw: bytes,
    science_publication_raw: bytes,
) -> dict[str, Any]:
    """Cross-bind an audited candidate to a freshly verified recovery completion.

    This pure check is additional to the accepted execution controller's complete
    plan/startup/rank-exit/accounting validation and the unchanged independent
    18,432-response scientific audit. It does not replace either verifier or
    independently recompute GPU arithmetic. Callers must run both before use.
    """
    candidate = validate_prepared_initial(prepared_initial)
    _require(isinstance(execution_plan, dict), "recovered fit plan required")
    _require(
        execution_plan.get("schema") == "quest-sdsc-student-order-execution-plan-v2"
        and execution_plan.get("task") == "qwen3-v2-student-order-execution-v2"
        and hashlib.sha256(_canonical(execution_plan)).hexdigest() == candidate["fit_execution_plan_sha256"],
        "recovered fit execution plan differs",
    )
    inner = execution_plan.get("science_plan")
    _require(
        isinstance(inner, dict)
        and inner.get("task") == "qwen3-v2-student-order-preparation-v4"
        and inner.get("mode") == "fit"
        and inner.get("intent_id") == candidate["fit_intent"]
        and execution_plan.get("intent_id") == candidate["fit_intent"]
        and hashlib.sha256(_canonical(inner)).hexdigest() == candidate["fit_plan_sha256"],
        "parent must be the bound complete V4 fit, never a preflight or diagnostic",
    )
    expected_science = dict(
        protocol_sha256=PARENT_PROTOCOL_CORE_SHA256,
        artifact_sha256=PARENT_PROTOCOL_SHA256,
        implementation_commit=PARENT_IMPLEMENTATION_COMMIT,
        acceptance_commit=PARENT_HEAD,
        head=candidate["fit_source_head"],
    )
    binding = inner.get("protocol")
    _require(
        isinstance(binding, dict)
        and all(binding.get(k) == v for k, v in expected_science.items())
        and isinstance(inner.get("provenance"), dict)
        and inner["provenance"].get("head") == candidate["fit_source_head"],
        "fit scientific ancestry or actual source HEAD differs",
    )
    _require(
        execution_plan.get("execution")
        == dict(
            core_sha256=RECOVERY_CORE_SHA256,
            artifact_sha256=RECOVERY_ARTIFACT_SHA256,
            implementation_commit=RECOVERY_IMPLEMENTATION_COMMIT,
            acceptance_commit=RECOVERY_ACCEPTANCE_COMMIT,
            head=candidate["fit_source_head"],
            control_file_sha256=RECOVERY_CONTROL_SHA256,
            science_binding=binding,
        ),
        "fit execution recovery binding differs",
    )
    _require(isinstance(status, dict), "verified combined recovery status required")
    _require(
        all(
            status.get(k) is True
            for k in (
                "accounting_complete",
                "success",
                "stage_complete",
                "preparation_complete",
                "scientific_stage_complete",
                "publication_verified",
                "artifact_hashes_verified",
                "execution_publication_verified",
                "startup_passed",
            )
        )
        and status.get("state") == "COMPLETED"
        and status.get("job_id") == candidate["fit_job_id"]
        and status.get("result_sha256") == candidate["fit_report_sha256"],
        "successful complete recovery status required",
    )
    queue = status.get("queue")
    accounting = status.get("accounting")
    _require(
        isinstance(queue, dict)
        and type(queue.get("returncode")) is int
        and queue["returncode"] == 0
        and isinstance(queue.get("stdout"), str)
        and not queue["stdout"].strip()
        and isinstance(accounting, dict)
        and type(accounting.get("returncode")) is int
        and accounting["returncode"] == 0,
        "completed fit requires successful accounting and an empty queue query",
    )
    for prefix, raw, digest_key in (
        ("execution_", execution_publication_raw, "fit_execution_publication_sha256"),
        ("", science_publication_raw, "fit_publication_sha256"),
    ):
        _require(
            isinstance(raw, bytes)
            and 0 < len(raw) <= _MAX_BYTES
            and hashlib.sha256(raw).hexdigest() == candidate[digest_key]
            and status.get(prefix + "publication_sha256") == candidate[digest_key],
            "bound publication bytes differ",
        )
        try:
            publication = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise StudentOrderQualificationError("invalid publication JSON") from error
        _require(
            isinstance(publication, dict)
            and publication == status.get(prefix + "publication")
            and publication.get("job_id") == candidate["fit_job_id"]
            and publication.get("persistent_read_back_verified") is True,
            "publication identity differs",
        )
        if prefix:
            _require(
                publication.get("schema") == "quest-sdsc-student-order-execution-publication-v2"
                and publication.get("task") == execution_plan["task"]
                and publication.get("plan_sha256") == candidate["fit_execution_plan_sha256"]
                and publication.get("science_plan_sha256") == candidate["fit_plan_sha256"]
                and publication.get("execution_core_sha256") == RECOVERY_CORE_SHA256
                and publication.get("startup_passed") is True,
                "execution publication does not bind this recovered fit",
            )
        else:
            _require(
                publication.get("task") == inner["task"]
                and publication.get("mode") == "fit"
                and publication.get("plan_sha256") == candidate["fit_plan_sha256"]
                and publication.get("passed") is True
                and publication.get("stage_complete") is True,
                "science publication does not bind this completed fit",
            )
        _require(all(publication.get(k) is False for k in FLAGS), "parent cannot self-accept a student")
    _require(
        all(obj.get(k) is False for obj in (execution_plan, inner, status) for k in FLAGS),
        "preparation cannot grant formal acceptance",
    )
    return candidate


def proposed_student_order_qualification_protocol(
    prepared_initial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a proposed protocol; None is an intentionally unusable draft."""
    value = original.proposed_student_qualification_protocol()
    value.update(kind="opd_student_order_qualification_protocol", protocol_id=PROTOCOL_ID)
    value["scope"].update(
        purpose="fixed_order_prepared_initial_base_and_anti_shortcut_qualification",
        prior_complete_formal896_exposure_job="54606205",
        subsequent_preparation_adaptation_after_prior_formal_exposure_disclosed=True,
        formal_population_is_unexposed_holdout=False,
        historical_failed_preparations_and_qualifications_preserved=True,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_student_protocol=preparation.PROTOCOL_PATH,
        parent_student_protocol_sha256=PARENT_PROTOCOL_SHA256,
        parent_protocol_core_sha256=PARENT_PROTOCOL_CORE_SHA256,
        parent_implementation_commit=PARENT_IMPLEMENTATION_COMMIT,
        parent_acceptance_commit=PARENT_HEAD,
        parent_auditor_sha256=PARENT_AUDITOR_SHA256,
        frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
        frozen_qualification_helper_sha256=dict(FROZEN_QUALIFICATION_HELPER_SHA256),
        parent_execution_recovery=dict(
            contract_path=RECOVERY_PATH,
            core_sha256=RECOVERY_CORE_SHA256,
            artifact_sha256=RECOVERY_ARTIFACT_SHA256,
            implementation_commit=RECOVERY_IMPLEMENTATION_COMMIT,
            acceptance_commit=RECOVERY_ACCEPTANCE_COMMIT,
            frozen_execution_sha256=dict(FROZEN_EXECUTION_SHA256),
            actual_fit_source_head="prepared_initial.fit_source_head",
            source_head_policy="accepted_recovery_ancestor_of_fit_ancestor_of_candidate_implementation",
        ),
    )
    value["prepared_initial"] = validate_prepared_initial(prepared_initial, require_bound=False)
    value["parent_fit_admission"] = dict(
        task="qwen3-v2-student-order-preparation-v4",
        mode="fit",
        optimizer_steps=32,
        training_input_tokens=1_614_932,
        development_steps=list(preparation.CHECKPOINT_STEPS),
        development_prompts=1536,
        raw_responses=18432,
        successful_accounting_and_persistent_artifact_verification_required=True,
        passed_execution_complete_preparation_complete_required=True,
        combined_recovery_status_required=True,
        startup_and_both_rank_exit_evidence_required=True,
        execution_and_scientific_publications_separately_bound=True,
        accepted_recovery_resolver_and_fresh_controller_status_required=True,
        original_science_only_or_diagnostic_status_admissible=False,
        independent_full_raw_replay_and_exact_parent_lineage_required=True,
        audit_encoding="canonical_JSON_plus_LF",
        audit_claim_is_execution_acceptance=False,
        selector="unchanged_parent_earliest_eligible_of_all_twelve_development_checkpoints",
        null_selection_admissible=False,
        checkpoint_lookup="unique_selected_step_and_SHA_matching_publication_path_size_SHA",
        reload_ranks=[0, 1],
        reload_fp32_parameter_count=311,
        reload_bfloat16_max_absolute_error=0.0,
        equal_rank_master_SHA_bound_before_BF16_copy=True,
        parent_dense_format="student_order_dense_v4",
        parent_dense_scope="common_student_order_model_only",
        original_checkpoint_path_retained=True,
        staged_checkpoint_path="checkpoints/prepared-selected.pt",
        parent_report_flags_must_be_false=list(FLAGS),
    )
    value["evidence"].update(
        original_formal_prompts_sha256=FORMAL_PROMPTS_SHA256,
        original_formal_prompts_size=FORMAL_PROMPTS_SIZE,
    )
    value["execution"].update(
        publication_reserve_seconds=300,
        max_checkpoint_bytes=MAX_CHECKPOINT_BYTES,
        minimum_node_local_free_gib=32,
        max_small_file_bytes=16 * 1024**2,
        max_total_small_bytes=48 * 1024**2,
        parent_max_shared_prompts_bytes=32 * 1024**2,
        parent_max_total_small_bytes=224 * 1024**2,
    )
    value["science_files"] = list(SCIENCE_PATHS)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    value["review_contract"]["actual_single_candidate_bound_before_implementation_commit"] = True
    value["review_contract"]["accepted_unbound_draft_forbidden"] = True
    return value


def validate_student_order_qualification_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    _require(isinstance(payload, dict), "protocol fields differ")
    candidate = validate_prepared_initial(payload.get("prepared_initial"), require_bound=False)
    expected = proposed_student_order_qualification_protocol(candidate)
    _require(isinstance(payload, dict) and set(payload) == set(expected), "protocol fields differ")
    review = payload["review"]
    _require(isinstance(review, dict) and set(review) == set(REVIEW_PROPOSED), "review fields differ")
    if review["status"] == "proposed":
        _require(review == REVIEW_PROPOSED and not require_accepted, "qualification protocol is not accepted")
    else:
        _require(review["status"] == "accepted", "unknown review status")
        commit = review["reviewed_implementation_commit"]
        _require(
            isinstance(commit, str) and _COMMIT.fullmatch(commit), "review needs full implementation commit"
        )
        for key in ("reviewer", "rationale"):
            _require(isinstance(review[key], str) and review[key].strip(), f"review requires {key}")
        stamp = review["reviewed_at_utc"]
        _require(isinstance(stamp, str) and _TIMESTAMP.fullmatch(stamp), "review requires UTC timestamp")
        try:
            datetime.fromisoformat(stamp[:-1] + "+00:00")
        except ValueError as error:
            raise StudentOrderQualificationError("invalid review date") from error
    validate_prepared_initial(candidate, require_bound=require_accepted or review["status"] == "accepted")
    expected["review"] = copy.deepcopy(review)
    try:
        _require(
            _canonical(payload) == _canonical(expected),
            "qualification protocol differs from frozen order v1",
        )
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentOrderQualificationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_order_qualification_protocol(
    raw: bytes, *, require_accepted: bool = False
) -> dict[str, Any]:
    _require(isinstance(raw, bytes) and 0 < len(raw) <= _MAX_BYTES, "invalid protocol byte bound")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate protocol JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise StudentOrderQualificationError("invalid protocol JSON") from error
    return validate_student_order_qualification_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_order_qualification_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


@dataclass(frozen=True)
class ResolvedStudentOrderQualificationProtocol:
    payload: dict[str, Any]
    protocol_sha256: str
    artifact_sha256: str
    implementation_commit: str | None
    acceptance_commit: str | None
    head: str
    science_file_sha256: dict[str, str]
    tracked_worktree_clean: bool

    @property
    def science_implementation_sha256(self) -> str:
        return sha256_value(self.science_file_sha256)

    @property
    def review_status(self) -> str:
        return self.payload["review"]["status"]


def resolve_student_order_qualification_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentOrderQualificationProtocol:
    root = Path(repo_root).absolute()
    _require(root.is_dir() and root.resolve() == root, "source root must be a real directory")
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    _require(
        metadata in (root / ".git", root / ".opd-git") and metadata.is_dir() and not metadata.is_symlink(),
        "qualification needs real .git or .opd-git metadata",
    )

    def git(*args: str) -> bytes:
        result = subprocess.run(
            (
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
            ),
            capture_output=True,
            env=_git_environment(),
            timeout=30,
            check=False,
        )
        _require(result.returncode == 0, f"real Git provenance query failed: {args[0]}")
        return result.stdout

    def text(*args: str) -> str:
        return git(*args).decode("utf-8", errors="strict").strip()

    def read(path: str) -> bytes:
        return read_regular_bytes_nofollow(
            root / path, context=f"student qualification source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_order_qualification_protocol(raw, require_accepted=require_accepted)
    _require(
        hashlib.sha256(read("prereg/qwen3_v2.yaml")).hexdigest() == BASE_PREREG_SHA256,
        "original frozen preregistration changed",
    )
    hashes = {path: hashlib.sha256(read(path)).hexdigest() for path in SCIENCE_PATHS}
    _require(
        all(hashes.get(path) == digest for path, digest in FROZEN_QUALIFICATION_HELPER_SHA256.items()),
        "frozen qualification helper changed",
    )
    implementation, acceptance = payload["review"]["reviewed_implementation_commit"], None
    if payload["review"]["status"] == "accepted":
        _require(
            implementation != head and text("merge-base", implementation, head) == implementation,
            "review implementation is not a distinct ancestor",
        )
        proposed = load_student_order_qualification_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
        _require(proposed["review"] == REVIEW_PROPOSED, "implementation protocol must be proposed")
        touching = text(
            "rev-list", "--reverse", "--ancestry-path", f"{implementation}..{head}", "--", PROTOCOL_PATH
        ).splitlines()
        _require(
            len(touching) == 1 and _COMMIT.fullmatch(touching[0]), "exactly one later review change required"
        )
        acceptance = touching[0]
        _require(
            len(text("rev-list", "--parents", "-n", "1", acceptance).split()) == 2,
            "review-only acceptance must have one parent",
        )
        changed = {
            path.decode()
            for path in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", acceptance).split(b"\0")
            if path
        }
        _require(
            PROTOCOL_PATH in changed and changed <= {PROTOCOL_PATH, "docs/refactor/current_handoff.md"},
            "acceptance contains non-review changes",
        )
        reviewed_raw = git("show", f"{acceptance}:{PROTOCOL_PATH}")
        reviewed = load_student_order_qualification_protocol(reviewed_raw, require_accepted=True)
        _require(
            reviewed["review"]["reviewed_implementation_commit"] == implementation,
            "acceptance reviews another implementation",
        )
        reviewed["review"] = copy.deepcopy(REVIEW_PROPOSED)
        _require(_canonical(reviewed) == _canonical(proposed), "acceptance changes scientific protocol")
        _require(
            raw == reviewed_raw == git("show", f"{head}:{PROTOCOL_PATH}"),
            "current protocol differs from accepted bytes",
        )
        for revision in (implementation, head):
            _require(
                hashlib.sha256(git("show", f"{revision}:prereg/qwen3_v2.yaml")).hexdigest()
                == BASE_PREREG_SHA256,
                "committed original preregistration changed",
            )
        _require(
            not git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--",
                PROTOCOL_PATH,
                "prereg/qwen3_v2.yaml",
                *SCIENCE_PATHS,
            ),
            "named scientific files have staged changes",
        )
        for path, current_sha in hashes.items():
            entry = text("ls-tree", implementation, "--", path).split(maxsplit=1)
            _require(
                entry and entry[0] in {"100644", "100755"}, f"science path is not a regular blob: {path}"
            )
            blob = git("show", f"{implementation}:{path}")
            _require(
                hashlib.sha256(blob).hexdigest() == current_sha and git("show", f"{head}:{path}") == blob,
                f"named science differs from accepted implementation: {path}",
            )
        _require(
            text("merge-base", PARENT_HEAD, implementation) == PARENT_HEAD, "original student lineage missing"
        )
        fit_head = payload["prepared_initial"]["fit_source_head"]
        _require(
            text("merge-base", RECOVERY_ACCEPTANCE_COMMIT, fit_head) == RECOVERY_ACCEPTANCE_COMMIT
            and text("merge-base", fit_head, implementation) == fit_head,
            "recovered fit source must descend from accepted execution and precede candidate review",
        )
        for path, expected_sha in FROZEN_EXECUTION_SHA256.items():
            _require(
                hashlib.sha256(read(path)).hexdigest() == expected_sha,
                f"frozen recovery execution changed: {path}",
            )
            for revision in (RECOVERY_ACCEPTANCE_COMMIT, fit_head, implementation, head):
                _require(
                    hashlib.sha256(git("show", f"{revision}:{path}")).hexdigest() == expected_sha,
                    f"committed recovery execution changed: {path}",
                )
            _require(
                not git("diff", "--cached", "--name-only", "--", path),
                "frozen recovery execution has staged changes",
            )
        for path in FROZEN_SCIENCE_PATHS:
            _require(
                git("show", f"{PARENT_HEAD}:{path}") == git("show", f"{fit_head}:{path}") == read(path),
                f"frozen parent student science changed: {path}",
            )
        for path, expected_sha in FROZEN_PROTOCOL_SHA256.items():
            _require(
                hashlib.sha256(read(path)).hexdigest() == expected_sha,
                f"frozen parent protocol changed: {path}",
            )
            for revision in (PARENT_HEAD, fit_head, implementation, head):
                _require(
                    hashlib.sha256(git("show", f"{revision}:{path}")).hexdigest() == expected_sha,
                    f"committed frozen parent protocol changed: {path}",
                )
            _require(
                not git("diff", "--cached", "--name-only", "--", path),
                "frozen parent protocol has staged changes",
            )
    clean = not git("diff", "--name-only", "HEAD", "--")
    _require(
        text("rev-parse", "--verify", "HEAD^{commit}") == head and read(PROTOCOL_PATH) == raw,
        "Git HEAD or protocol changed during validation",
    )
    for path, expected_sha in hashes.items():
        _require(
            hashlib.sha256(read(path)).hexdigest() == expected_sha,
            f"source changed during validation: {path}",
        )
    if payload["review"]["status"] == "accepted":
        for path, expected_sha in FROZEN_PROTOCOL_SHA256.items():
            _require(
                hashlib.sha256(read(path)).hexdigest() == expected_sha,
                f"frozen parent protocol changed during validation: {path}",
            )
        for path, expected_sha in FROZEN_EXECUTION_SHA256.items():
            _require(
                hashlib.sha256(read(path)).hexdigest() == expected_sha,
                f"frozen recovery execution changed during validation: {path}",
            )
    return ResolvedStudentOrderQualificationProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
    )
