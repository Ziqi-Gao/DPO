"""Bounded transport for a separately accepted adapted-teacher qualification.

This module is stdlib-only. Scientific acceptance is resolved from restored,
committed source by the selected runtime; transport never accepts proposed YAML
or turns an exploratory fit report into formal teacher readiness.
"""

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

TASK = "qwen3-v2-teacher-qualify"
RESULT = "teacher-qualify.json"
CLAIM_NAMESPACE = "opd-qwen3-teacher-adaptation-confirmation-20260928"
CLAIMS_ROOT = Path("/home/zgao12/quest-runs/OPD/teacher-qualification-claims")
VALIDATION_SHA256 = "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3"
SUPPLEMENTAL_ORDER_SHA256 = "532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073"
MAX_METADATA_BYTES = 4 * 1024**2
MAX_CHECKPOINT_BYTES = 24 * 1024**3
MAX_FIT_EVIDENCE_BYTES = 128 * 1024**2
MAX_SCIENTIFIC_BYTES = 256 * 1024**2
MAX_UNIQUE_SCIENTIFIC_BYTES = 128 * 1024**2
MAX_LOG_BYTES = 64 * 1024**2
MAX_OUTPUT_BYTES = 8 * 1024**3
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
ROLES = ("train_probe", "supplemental", "formal_readiness", "formal_store")
CHECKS = {
    "genuine_reviewed_science",
    "verified_fit_and_selection",
    "verified_dense_checkpoint",
    "all_four_ranks",
    "train_probe",
    "supplemental",
    "formal_readiness",
    "formal_store",
    "cgroup_headroom",
}
SMALL_RESULTS = {
    "qualification-manifest.json",
    "progress.json",
    "stage-summary.json",
    "acceptance-evidence.json",
}
# Shared fit science that is not part of its distributed execution fingerprint.
# Qualification-only code intentionally has no corresponding historical fit blob.
FIT_SCIENCE_ORIGIN_PATHS = (
    "configs/config.yaml",
    "configs/task/proofgraph_main.yaml",
    "configs/teacher/qwen3_v2_teacher_8b.yaml",
    "configs/g0/qwen3_v2_eap_separation.yaml",
    "configs/experiment/canonical_sft.yaml",
    "src/posttrain_circuits/core/config.py",
    "src/posttrain_circuits/artifacts/hashing.py",
    "src/posttrain_circuits/artifacts/io.py",
    "src/posttrain_circuits/artifacts/compatibility.py",
    "src/posttrain_circuits/datasets/proofgraph/contracts.py",
    "src/posttrain_circuits/datasets/proofgraph/generation.py",
    "src/posttrain_circuits/datasets/proofgraph/splits.py",
    "src/posttrain_circuits/datasets/proofgraph/serialization.py",
    "src/posttrain_circuits/datasets/proofgraph/rendering.py",
    "src/posttrain_circuits/datasets/proofgraph/parsing.py",
    "src/posttrain_circuits/datasets/proofgraph/verification.py",
    "src/posttrain_circuits/datasets/teacher_demos/contracts.py",
    "src/posttrain_circuits/learning/teacher/seeding.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha256_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe(path):
    path = Path(path)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and not any(ord(c) < 32 for c in str(path))
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "unsafe qualification path",
    )
    return path


def relative(value):
    require(
        isinstance(value, str) and value and not any(ord(c) < 32 for c in value),
        "invalid relative qualification path",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts and str(path) == value,
        "unsafe relative qualification path",
    )
    return value


def small_bytes(path, limit=MAX_METADATA_BYTES):
    path = safe(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, "oversized/nonregular metadata")
        data = stream.read(limit + 1)
        require(len(data) == info.st_size and len(data) <= limit, "metadata changed while reading")
    return data


def record_map(rows):
    require(isinstance(rows, list), "missing file inventory")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "size", "sha256"}, "invalid file record")
        name = relative(row["path"])
        require(
            name not in result
            and type(row["size"]) is int
            and row["size"] >= 0
            and SHA256.fullmatch(str(row["sha256"])),
            "duplicate/invalid file inventory",
        )
        result[name] = row
    return result


def verify_fit_origin(upstream, protocol):
    """Bind historical fit science to the newly reviewed qualification source."""
    spec = importlib.util.spec_from_file_location(
        "_qualify_fit_origin", Path(__file__).with_name("sdsc_teacher_fit_contract.py")
    )
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    manifest = upstream["release_manifest"]
    execution = fit.execution_plan(manifest)
    origins = {row["path"]: row["sha256"] for row in manifest["files"]}
    reviewed = protocol.get("science_file_sha256")
    require(
        isinstance(reviewed, dict)
        and all(
            SHA256.fullmatch(str(reviewed.get(name, ""))) and origins.get(name) == reviewed[name]
            for name in (*fit.KERNEL_PATHS, *FIT_SCIENCE_ORIGIN_PATHS)
        ),
        "fit origin differs from reviewed qualification science",
    )
    report = upstream["report"]
    require(
        canonical(report.get("execution_plan")) == canonical(execution)
        and report.get("execution_plan_sha256") == sha256_value(execution),
        "fit report execution differs from its reviewed origin",
    )
    return execution


def selected_checkpoint(upstream):
    """Resolve only the fit-selected checkpoint; no caller-controlled weight path.

    The caller first verifies accounting and the complete fit publication. Login
    metadata checks do not read large weights; the GPU-node staging step hashes
    every selected byte and verifies a second read of the local copy.
    """
    report = upstream["report"]
    receipt = upstream["receipt"]
    require(
        receipt.get("task") == "qwen3-v2-teacher-fit"
        and report.get("task") == receipt["task"]
        and report.get("mode") == "full-fit"
        and report.get("passed") is True
        and report.get("optimizer_steps") == 512,
        "qualification needs a completed fixed full fit",
    )
    selected = report.get("selected_checkpoint_sha256")
    require(SHA256.fullmatch(str(selected)), "full fit did not select a passing checkpoint")
    root = safe(Path(upstream["result_dir"]) / "artifacts")
    raw = small_bytes(root / "checkpoint-manifest.json")
    require(
        hashlib.sha256(raw).hexdigest() == report.get("checkpoint_manifest_sha256"),
        "fit checkpoint manifest changed",
    )
    manifest = json.loads(raw)
    require(
        manifest.get("sha256")
        == sha256_value({k: v for k, v in manifest.items() if k != "sha256"})
        == report.get("checkpoint_set_sha256"),
        "fit checkpoint set identity changed",
    )
    candidates = manifest.get("checkpoints", [])
    passing = [row for row in candidates if row.get("dev_metrics_passed") is True]
    require(
        passing and passing[0].get("adapted_teacher_sha256") == selected,
        "checkpoint selection is not the first scheduled development PASS",
    )
    row = passing[0]
    directory = relative(row["directory"])
    require(
        directory == f"checkpoints/step-{row['step']:06d}" and row["step"] in (128, 256, 384, 512),
        "invalid selected checkpoint directory",
    )
    checkpoint = safe(root / directory)
    dense_raw = small_bytes(checkpoint / "dense-manifest.json")
    require(
        hashlib.sha256(dense_raw).hexdigest() == row.get("dense_manifest_sha256"),
        "selected dense manifest physical hash differs",
    )
    dense = json.loads(dense_raw)
    require(
        dense.get("sha256") == selected == sha256_value({k: v for k, v in dense.items() if k != "sha256"})
        and dense.get("artifact_kind") == "adapted_dense_teacher"
        and dense.get("formal_teacher_accepted") is False,
        "selected adapted weight identity differs",
    )
    inventory = record_map(dense.get("files"))
    published = record_map(upstream["publication_receipt"]["files"])
    require(
        inventory and sum(row["size"] for row in inventory.values()) <= MAX_CHECKPOINT_BYTES,
        "selected checkpoint exceeds staging envelope",
    )
    records = []
    for name, item in sorted(inventory.items()):
        require(PurePosixPath(name).parts[0] in {"merged", "adapter"}, "unexpected selected weight file")
        source = safe(checkpoint / name)
        info = source.stat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size == item["size"], "selected weight metadata changed"
        )
        published_name = "artifacts/" + directory + "/" + name
        require(
            published.get(published_name) == dict(item, path=published_name),
            "selected checkpoint was not durably published",
        )
        records.append(item)
    dense_record = dict(
        path="dense-manifest.json", size=len(dense_raw), sha256=hashlib.sha256(dense_raw).hexdigest()
    )
    published_name = "artifacts/" + directory + "/dense-manifest.json"
    require(
        published.get(published_name) == dict(dense_record, path=published_name),
        "selected dense manifest publication differs",
    )
    records.append(dense_record)
    return dict(
        checkpoint_root=str(checkpoint),
        adapted_teacher_sha256=selected,
        selected_step=row["step"],
        dense_manifest_sha256=row["dense_manifest_sha256"],
        fit_job_id=receipt["job_id"],
        files=records,
        checkpoint_content_rehashed=False,
    )


def reserve_claims(root, binding):
    """Permanent exclusive reservations; failed/unknown attempts require audit.

    The experiment key deliberately excludes checkpoint and protocol hashes so
    neither model reselection nor a renamed protocol can re-expose this cohort.
    Partial creation is also retained. There is no automatic rollback/retry.
    """
    checkpoint = binding.get("adapted_teacher_sha256")
    require(SHA256.fullmatch(str(checkpoint)), "claim lacks checkpoint identity")
    require(
        binding.get("supplemental_namespace") == CLAIM_NAMESPACE,
        "qualification supplemental namespace differs",
    )
    root = safe(root)
    rows = []
    for category, key in (("experiments", CLAIM_NAMESPACE), ("checkpoints", checkpoint)):
        directory = safe(root / "teacher-qualification-claims" / category)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = safe(directory / (key + ".json"))
        value = dict(
            binding,
            schema="quest-sdsc-teacher-qualification-claim-v1",
            claim_kind="reserved_before_inference",
            claim_category=category,
        )
        raw = canonical(value) + b"\n"
        try:
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except FileExistsError:
            raise ValueError(
                "Qualification already reserved; reconcile its original intent, never retry or reselect"
            ) from None
        require(small_bytes(path) == raw, "qualification claim read-back differs")
        rows.append(dict(path=str(path), size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), value=value))
    return rows


def fit_evidence(upstream):
    records = record_map(upstream["publication_receipt"]["files"])
    required = {
        "teacher-fit.json",
        "artifacts/checkpoint-manifest.json",
        "artifacts/checkpoint-selection.json",
        "artifacts/data-isolation.json",
        "artifacts/train-metrics.jsonl",
    }
    for step in (128, 256, 384, 512):
        prefix = f"artifacts/checkpoints/step-{step:06d}/"
        required.update(
            prefix + name for name in ("manifest.json", "dense-manifest.json", "dev-capability.json")
        )
        required.update(
            prefix + f"eval/rank-{rank}/{name}.jsonl"
            for rank in range(4)
            for name in ("generations", "prefix-scores")
        )
        required.update(
            prefix + "eval/" + name
            for name in ("semantic-prefix-manifest.json", "tokenized-prefix-manifest.json")
        )
    require(required <= set(records), "full-fit raw development evidence is incomplete")
    selected = [records[name] for name in sorted(required)]
    require(
        sum(row["size"] for row in selected) <= MAX_FIT_EVIDENCE_BYTES,
        "full-fit evidence exceeds the 128 MiB staging limit",
    )
    return dict(root=upstream["result_dir"], files=selected, total_bytes=sum(row["size"] for row in selected))


def stage_fit_evidence(evidence, destination):
    root, destination = safe(evidence["root"]), safe(destination)
    records = record_map(evidence["files"])
    total = sum(row["size"] for row in records.values())
    require(
        total == evidence["total_bytes"] and total <= MAX_FIT_EVIDENCE_BYTES,
        "raw fit evidence exceeds its bound",
    )
    require(not destination.exists(), "fit evidence staging directory already exists")
    destination.mkdir(mode=0o700)
    for name, row in records.items():
        require(Path(name).suffix in {".json", ".jsonl"}, "fit evidence cannot include tensors")
        raw = small_bytes(root / name, MAX_FIT_EVIDENCE_BYTES)
        require(
            len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "published raw fit evidence changed",
        )
        target = safe(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        require(small_bytes(target, MAX_FIT_EVIDENCE_BYTES) == raw, "raw fit evidence read-back differs")
        target.chmod(0o444)
    for directory, _, _ in os.walk(destination, topdown=False):
        Path(directory).chmod(0o555)
    return dict(files=list(records.values()), total_bytes=total, read_back_verified=True)


def verify_claims(claims, expected_root=None):
    require(
        claims.get("schema") == "quest-sdsc-teacher-qualification-reservations-v1",
        "qualification reservation schema differs",
    )
    binding = claims.get("binding", {})
    require(
        binding.get("supplemental_namespace") == CLAIM_NAMESPACE
        and binding.get("validation_file_sha256") == VALIDATION_SHA256
        and binding.get("supplemental_order_sha256") == SUPPLEMENTAL_ORDER_SHA256,
        "supplemental one-shot namespace or cohort differs",
    )
    require(
        re.fullmatch(r"[a-f0-9]{32}", str(binding.get("intent_id", "")))
        and all(
            SHA256.fullmatch(str(binding.get(key, "")))
            for key in ("code_sha256", "adapted_teacher_sha256", "protocol_sha256", "prerequisites_sha256")
        ),
        "reservation identity is invalid",
    )
    reservations = claims.get("reservations")
    require(isinstance(reservations, list) and len(reservations) == 2, "missing permanent reservations")
    expected_root = safe(CLAIMS_ROOT if expected_root is None else expected_root)
    stage_root = safe(claims["stage_root"])
    require(stage_root == expected_root / "stages" / binding["intent_id"], "stage claim root differs")
    categories = set()
    for row in reservations:
        raw = small_bytes(row["path"])
        value = json.loads(raw)
        category = value.get("claim_category")
        expected_name = CLAIM_NAMESPACE if category == "experiments" else binding["adapted_teacher_sha256"]
        require(
            category in {"experiments", "checkpoints"}
            and category not in categories
            and Path(row["path"]) == expected_root / category / (expected_name + ".json")
            and value.get("claim_kind") == "reserved_before_inference"
            and hashlib.sha256(raw).hexdigest() == row.get("sha256")
            and len(raw) == row.get("size")
            and value == row.get("value")
            and all(value.get(key) == item for key, item in binding.items()),
            "permanent qualification reservation changed",
        )
        categories.add(category)
    expected = {role: binding["intent_id"] + ":" + role for role in ROLES}
    require(claims.get("stage_claim_ids") == expected, "stage claim identity differs")
    return binding


def exclusive_json(path, value):
    path = safe(path)
    raw = canonical(value) + b"\n"
    require(len(raw) <= MAX_METADATA_BYTES, "claim exceeds metadata bound")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(small_bytes(path) == raw, "persistent claim read-back failed")
    return value


def claim_stage(claims, role, predecessor_sha256, *, job_id):
    binding = verify_claims(claims)
    require(role in ROLES and SHA256.fullmatch(str(predecessor_sha256)), "invalid stage/previous evidence")
    require(re.fullmatch(r"[1-9][0-9]*", str(job_id)), "stage needs real Slurm identity")
    stage_root = safe(claims["stage_root"])
    index = ROLES.index(role)
    if index:
        previous = json.loads(small_bytes(stage_root / (ROLES[index - 1] + ".outcome.json")))
        require(
            previous.get("evidence_sha256") == predecessor_sha256
            and previous.get("metrics_passed") is True
            and previous.get("job_id") == job_id
            and previous.get("claim_id") == claims["stage_claim_ids"][ROLES[index - 1]],
            "previous stage is not sealed and passing",
        )
    value = dict(
        binding,
        schema="quest-sdsc-teacher-qualification-stage-claim-v1",
        job_id=job_id,
        role=role,
        claim_id=claims["stage_claim_ids"][role],
        predecessor_sha256=predecessor_sha256,
        claimed_before_inference=True,
    )
    return exclusive_json(stage_root / (role + ".claim.json"), value)


def seal_stage(claims, role, evidence_path, summary, *, job_id):
    verify_claims(claims)
    require(role in ROLES, "unknown qualification stage")
    stage_root = safe(claims["stage_root"])
    claim_raw = small_bytes(stage_root / (role + ".claim.json"))
    claim = json.loads(claim_raw)
    # Evidence can be larger than fetched small reports, but remains bounded.
    raw = small_bytes(evidence_path, MAX_FIT_EVIDENCE_BYTES)
    evidence = json.loads(raw)
    require(
        claim.get("job_id") == job_id
        and claim.get("claim_id") == claims["stage_claim_ids"][role]
        and evidence.get("claim_id") == claim["claim_id"]
        and evidence.get("role") == role
        and evidence.get("predecessor_sha256") == claim["predecessor_sha256"]
        and evidence.get("sha256") == sha256_value({k: v for k, v in evidence.items() if k != "sha256"})
        and type(summary.get("metrics_passed")) is bool,
        "stage evidence does not bind its prior persistent claim",
    )
    value = dict(
        schema="quest-sdsc-teacher-qualification-stage-outcome-v1",
        job_id=job_id,
        role=role,
        claim_id=claim["claim_id"],
        claim_sha256=hashlib.sha256(claim_raw).hexdigest(),
        predecessor_sha256=claim["predecessor_sha256"],
        evidence_sha256=evidence["sha256"],
        evidence_file_sha256=hashlib.sha256(raw).hexdigest(),
        evidence_file_size=len(raw),
        summary_sha256=sha256_value(summary),
        metrics_passed=summary["metrics_passed"],
    )
    return exclusive_json(stage_root / (role + ".outcome.json"), value)


def resolved_protocol(root, expected_head):
    """Executed only in the selected runtime against a restored genuine checkout."""
    root = safe(root)
    sys.path.insert(0, str(root / "src"))
    from dataclasses import asdict

    from posttrain_circuits.artifacts.teacher_adaptation_protocol import resolve_teacher_adaptation_protocol

    resolved = resolve_teacher_adaptation_protocol(root, require_accepted=True, expected_head=expected_head)
    result = asdict(resolved)
    result.update(
        science_implementation_sha256=resolved.science_implementation_sha256,
        review_status=resolved.review_status,
    )
    require(
        result["review_status"] == "accepted" and result["head"] == expected_head,
        "qualification protocol is not genuinely accepted",
    )
    require("torch" not in sys.modules, "metadata-only protocol resolution imported Torch")
    return result


def resolve_from_bundle(artifact, manifest_sha256, code_sha256, expected_head):
    import platform

    require(platform.python_version() == "3.12.13", "qualification metadata needs the pinned Python runtime")
    require(
        hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "qualification Python executable bytes differ",
    )
    spec = importlib.util.spec_from_file_location(
        "_qualify_provenance", Path(__file__).with_name("sdsc_provenance.py")
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    with tempfile.TemporaryDirectory(prefix="opd-qualify-protocol-") as directory:
        checkout = Path(directory) / "science"
        try:
            restored = verifier.verify(Path(artifact), manifest_sha256, code_sha256, checkout)
            require(restored["git_head"] == expected_head, "restored protocol Git HEAD differs")
            return resolved_protocol(checkout, expected_head)
        finally:
            # Only this new private temporary checkout is made removable.
            if checkout.exists():
                for root, _, _ in os.walk(checkout):
                    Path(root).chmod(0o700)


def validate_report(report, proof):
    require(
        isinstance(report, dict)
        and report.get("task") == TASK
        and report.get("artifact_kind") == "adapted_teacher_qualification_execution",
        "qualification report task/artifact differs",
    )
    require(
        all(
            report.get(key) is True for key in ("passed", "scientific_evidence_verified", "protocol_accepted")
        )
        and all(
            report.get(key) is False
            for key in (
                "formal_teacher_accepted",
                "accepted_science",
                "full_teacher_ready",
                "g0_passed",
                "execution_class_certified",
                "student_training_started",
                "teacher_training_started",
                "training_started",
                "readiness_artifact_produced",
                "original_128_token_readiness_pass_claim",
            )
        )
        and type(report.get("exit_code")) is int
        and report["exit_code"] == 0,
        "qualification is incomplete or self-claims independent acceptance",
    )
    require(
        isinstance(report.get("checks"), dict)
        and set(report["checks"]) == CHECKS
        and all(value is True for value in report["checks"].values()),
        "qualification checks differ",
    )
    protocol, selected = proof["protocol"], proof["selected_checkpoint"]
    expected = dict(
        proof["target"],
        teacher_job_id=proof["teacher_job_id"],
        adapted_teacher_sha256=selected["adapted_teacher_sha256"],
        selected_step=selected["selected_step"],
        science_git_head=protocol["head"],
        protocol_sha256=protocol["protocol_sha256"],
        protocol_artifact_sha256=protocol["artifact_sha256"],
        protocol_implementation_commit=protocol["implementation_commit"],
        protocol_acceptance_commit=protocol["acceptance_commit"],
        science_implementation_sha256=protocol["science_implementation_sha256"],
        fit_publication_receipt_sha256=proof["upstream"]["publication_receipt_sha256"],
    )
    require(
        all(report.get(key) == value for key, value in expected.items()), "qualification provenance differs"
    )
    require(
        all(
            SHA256.fullmatch(str(report.get(key, "")))
            for key in ("acceptance_evidence_sha256", "acceptance_evidence_file_sha256", "selection_sha256")
        ),
        "qualification evidence hashes are missing",
    )
    summaries = report.get("stage_summaries")
    require(
        isinstance(summaries, dict)
        and set(summaries) == set(ROLES)
        and all(isinstance(row, dict) and row.get("metrics_passed") is True for row in summaries.values()),
        "qualification stage completion is missing",
    )
    require(
        type(report.get("world_size")) is int and report["world_size"] == 4,
        "qualification did not execute four ranks",
    )
    spec = importlib.util.spec_from_file_location(
        "_qualify_fit_memory_contract", Path(__file__).with_name("sdsc_teacher_fit_contract.py")
    )
    fit_contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_contract)
    fit_contract.validate_rank_memory(report.get("rank_summaries"), report.get("job_id"))
    require(
        all(
            row.get("world_size") == 4
            and row.get("logical_device") == row["rank"]
            and row.get("torch_num_threads") == 6
            and "H100" in str(row.get("gpu_name", ""))
            for row in report["rank_summaries"]
        ),
        "qualification GPU/thread evidence differs",
    )


def validate_acceptance_file(output, report, claims):
    raw = small_bytes(Path(output) / "acceptance-evidence.json", MAX_FIT_EVIDENCE_BYTES)
    evidence = json.loads(raw)
    require(
        hashlib.sha256(raw).hexdigest() == report["acceptance_evidence_file_sha256"]
        and evidence.get("sha256")
        == report["acceptance_evidence_sha256"]
        == sha256_value({key: value for key, value in evidence.items() if key != "sha256"})
        and evidence.get("artifact_kind") == "adapted_teacher_acceptance_evidence"
        and evidence.get("scientific_evidence_verified") is True
        and evidence.get("formal_teacher_accepted") is False
        and evidence.get("protocol_sha256") == report["protocol_sha256"],
        "acceptance evidence content differs or claims external acceptance",
    )
    verify_claims(claims)
    previous = report["selection_sha256"]
    for role in ROLES:
        claim_raw = small_bytes(Path(claims["stage_root"]) / (role + ".claim.json"))
        claim = json.loads(claim_raw)
        outcome = json.loads(small_bytes(Path(claims["stage_root"]) / (role + ".outcome.json")))
        directory = Path(output) / "stages" / role
        raw_evidence = small_bytes(directory / "evidence.json", MAX_FIT_EVIDENCE_BYTES)
        stage = json.loads(raw_evidence)
        summary = json.loads(small_bytes(directory / "summary.json", MAX_FIT_EVIDENCE_BYTES))
        require(
            claim.get("schema") == "quest-sdsc-teacher-qualification-stage-claim-v1"
            and outcome.get("schema") == "quest-sdsc-teacher-qualification-stage-outcome-v1"
            and claim.get("claimed_before_inference") is True
            and all(claim.get(key) == value for key, value in claims["binding"].items())
            and claim.get("predecessor_sha256")
            == previous
            == outcome.get("predecessor_sha256")
            == stage.get("predecessor_sha256")
            and claim.get("job_id") == report["job_id"] == outcome.get("job_id")
            and claim.get("role") == role == outcome.get("role") == stage.get("role")
            and claim.get("claim_id")
            == outcome.get("claim_id")
            == claims["stage_claim_ids"][role]
            == stage.get("claim_id")
            and outcome.get("claim_sha256") == hashlib.sha256(claim_raw).hexdigest()
            and outcome.get("evidence_sha256")
            == stage.get("sha256")
            == sha256_value({key: value for key, value in stage.items() if key != "sha256"})
            and outcome.get("evidence_file_sha256") == hashlib.sha256(raw_evidence).hexdigest()
            and outcome.get("evidence_file_size") == len(raw_evidence)
            and outcome.get("summary_sha256") == sha256_value(summary)
            and outcome.get("metrics_passed") is True
            and summary.get("metrics_passed") is True
            and json.loads(small_bytes(directory / "claim.json")) == claim
            and json.loads(small_bytes(directory / "outcome.json")) == outcome,
            "persistent stage claims/outcomes differ from published evidence",
        )
        previous = stage["sha256"]
    return evidence


def validate_inventory(output, report):
    root = safe(output)
    raw = small_bytes(root / "qualification-manifest.json")
    manifest = json.loads(raw)
    require(
        hashlib.sha256(raw).hexdigest() == report.get("qualification_manifest_sha256")
        and manifest.get("sha256")
        == report.get("qualification_inventory_sha256")
        == sha256_value({key: value for key, value in manifest.items() if key != "sha256"})
        and manifest.get("artifact_kind") == "adapted_teacher_qualification_inventory"
        and manifest.get("formal_teacher_accepted") is False
        and all(manifest.get(key) == report.get(key) for key in ("run_id", "job_id", "code_sha256"))
        and manifest.get("teacher_identity", {}).get("teacher_checkpoint_sha256")
        == report.get("adapted_teacher_sha256"),
        "qualification inventory identity differs",
    )
    records = record_map(manifest.get("files"))
    unique = {}
    for row in records.values():
        require(
            row["sha256"] not in unique or unique[row["sha256"]] == row["size"],
            "duplicate content identity has inconsistent size",
        )
        unique[row["sha256"]] = row["size"]
    require(
        sum(row["size"] for row in records.values()) <= MAX_SCIENTIFIC_BYTES
        and sum(unique.values()) <= MAX_UNIQUE_SCIENTIFIC_BYTES,
        "qualification scientific evidence exceeds physical/unique bounds",
    )
    prefixes = {"inputs", "fit-evidence", "tokenizer", "development", "stages", "teacher-store"}
    names = {"selection.json", "acceptance-evidence.json"}
    actual = set()
    for path in root.rglob("*"):
        safe(path)
        name = path.relative_to(root).as_posix()
        if path.is_file() and (name.split("/")[0] in prefixes or name in names):
            actual.add(name)
    require(actual == set(records), "qualification scientific inventory is incomplete")
    for name, row in records.items():
        path = safe(root / name)
        require(
            path.suffix not in {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}
            and stat.S_ISREG(path.stat().st_mode)
            and path.stat().st_size == row["size"],
            "qualification inventory contains tensors or changed file metadata",
        )
    return records


def stage_checkpoint(selection, destination):
    """Copy and hash immutable selected bytes twice; never stage optimizer state."""
    source = safe(selection["checkpoint_root"])
    destination = safe(destination)
    require(not destination.exists(), "selected checkpoint staging directory already exists")
    destination.mkdir(mode=0o700)
    records = record_map(selection["files"])
    require(
        sum(row["size"] for row in records.values()) <= MAX_CHECKPOINT_BYTES,
        "selected checkpoint exceeds staging envelope",
    )
    for name, row in records.items():
        require(
            name == "dense-manifest.json" or PurePosixPath(name).parts[0] in {"merged", "adapter"},
            "checkpoint staging includes an unsupported file",
        )
        path = safe(source / name)
        target = safe(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        with (
            os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming,
            target.open("xb") as outgoing,
        ):
            before = os.fstat(incoming.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size == row["size"], "checkpoint metadata differs"
            )
            size = 0
            for block in iter(lambda: incoming.read(8 * 1024**2), b""):
                size += len(block)
                require(size <= row["size"], "checkpoint grew during staging")
                hasher.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
        require(
            size == row["size"]
            and hasher.hexdigest() == row["sha256"]
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "selected checkpoint changed during staging",
        )
        hasher = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                hasher.update(block)
        require(hasher.hexdigest() == row["sha256"], "local selected checkpoint read-back differs")
        target.chmod(0o444)
    for directory, _, _ in os.walk(destination, topdown=False):
        Path(directory).chmod(0o555)
    return dict(
        adapted_teacher_sha256=selection["adapted_teacher_sha256"],
        files=list(records.values()),
        checkpoint_content_rehashed=True,
        node_local_read_back_verified=True,
    )


if __name__ == "__main__":
    require(
        len(sys.argv) == 6 and sys.argv[1] == "--resolve-protocol",
        "only protocol metadata resolution is supported",
    )
    print(canonical(resolve_from_bundle(*sys.argv[2:])).decode())
