#!/usr/bin/env python3
"""Independently replay a published learned-teacher qualification on CPU.

Run with the verified runtime's ``python -I -B`` and a real restored accepted
source checkout. This command reads no weights, executes no inference or Slurm
command, and never produces an independent acceptance attestation. Accounting
and the reviewer decision remain explicit external checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

MAX_SCIENCE_BYTES = 256 * 1024**2
MAX_UNIQUE_SCIENCE_BYTES = 128 * 1024**2
MAX_LOG_BYTES = 64 * 1024**2
MAX_CONTROL_BYTES = 4 * 1024**2
MAX_AUDIT_BYTES = 1024**2
TASK = "qwen3-v2-teacher-qualify"
ROLES = ("train_probe", "supplemental", "formal_readiness", "formal_store")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TENSOR_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}
SCIENCE_PREFIXES = {"inputs", "fit-evidence", "tokenizer", "development", "stages", "teacher-store"}
SCIENCE_CONTROLS = {
    "teacher-qualify.json",
    "artifacts/teacher-qualify.json",
    "artifacts/qualification-manifest.json",
    "artifacts/selection.json",
    "artifacts/acceptance-evidence.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse(raw):
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key in published evidence")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite JSON number in published evidence")

    return json.loads(raw, object_pairs_hook=mapping, parse_constant=invalid)


def relative(value):
    require(isinstance(value, str) and value and not any(ord(c) < 32 for c in value), "invalid path")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and str(path) == value and ".." not in path.parts,
        "unsafe relative evidence path",
    )
    return value


def safe_read(path, limit):
    """Stable, single-link regular bytes; all path ancestors must be real."""
    path = Path(path)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and not any(part.is_symlink() for part in (path, *path.parents)),
        "evidence path contains a symlink or traversal",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "evidence is not one regular file")
        require(before.st_size <= limit, "published evidence exceeds the bounded CPU audit")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == before.st_size <= limit
        and all(getattr(before, key) == getattr(after, key) for key in keys),
        "published evidence changed during its actual read",
    )
    return raw


def scientific_file(name):
    parts = PurePosixPath(name).parts
    return name in SCIENCE_CONTROLS or (
        len(parts) > 2 and parts[0] == "artifacts" and parts[1] in SCIENCE_PREFIXES
    )


def publication(result_root, expected_sha256, job_id):
    """Verify every actually published byte, with no checkpoint-file exception."""
    root = Path(result_root).absolute()
    require(
        SHA256.fullmatch(str(expected_sha256)), "an externally pinned publication receipt SHA is required"
    )
    require(re.fullmatch(r"[1-9][0-9]*", str(job_id)), "actual job identity required")
    receipt_raw = safe_read(root / "receipt.json", MAX_CONTROL_BYTES)
    require(
        hashlib.sha256(receipt_raw).hexdigest() == expected_sha256, "publication receipt identity differs"
    )
    receipt = parse(receipt_raw)
    require(
        receipt.get("task") == TASK
        and receipt.get("job_id") == job_id
        and all(receipt.get(key) is True for key in ("passed", "persisted", "persistent_read_back_verified")),
        "qualification publication is incomplete or belongs to another job",
    )
    records, raw_files, science_total, log_total = {}, {}, 0, 0
    unique_science = {}
    require(isinstance(receipt.get("files"), list), "publication lacks its file inventory")
    for row in receipt["files"]:
        require(
            isinstance(row, dict) and set(row) == {"path", "size", "sha256"}, "invalid publication record"
        )
        name = relative(row["path"])
        require(
            name not in records
            and name != "receipt.json"
            and type(row["size"]) is int
            and row["size"] >= 0
            and SHA256.fullmatch(str(row["sha256"])),
            "duplicate or invalid publication record",
        )
        require(
            PurePosixPath(name).suffix not in TENSOR_SUFFIXES, "qualification audit must not read weights"
        )
        if not scientific_file(name):
            log_total += row["size"]
            require(log_total <= MAX_LOG_BYTES, "operational log exceeds audit bound")
        else:
            science_total += row["size"]
            require(science_total <= MAX_SCIENCE_BYTES, "scientific evidence exceeds 256 MiB physical bound")
            unique_science[row["sha256"], row["size"]] = row["size"]
            require(
                sum(unique_science.values()) <= MAX_UNIQUE_SCIENCE_BYTES,
                "scientific evidence exceeds 128 MiB unique-content bound",
            )
        raw = safe_read(root / name, MAX_SCIENCE_BYTES if scientific_file(name) else MAX_LOG_BYTES)
        require(
            len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
            f"actual publication file differs: {name}",
        )
        records[name], raw_files[name] = row, raw
    require("teacher-qualify.json" in records, "publication omits the qualification report")
    actual = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        require(
            not any((Path(directory) / name).is_symlink() for name in directories),
            "linked evidence directory",
        )
        for name in filenames:
            item = Path(directory) / name
            require(not item.is_symlink(), "linked publication file")
            actual.add(item.relative_to(root).as_posix())
    require(actual == set(records) | {"receipt.json"}, "publication has unbound extra or missing files")
    return {
        "root": root,
        "receipt": receipt,
        "receipt_raw": receipt_raw,
        "receipt_sha256": expected_sha256,
        "files": records,
        "raw": raw_files,
        "scientific_bytes": science_total,
        "unique_scientific_bytes": sum(unique_science.values()),
        "log_bytes": log_total,
    }


def document(published, path):
    require(path in published["raw"], f"missing actual published artifact: {path}")
    return parse(published["raw"][path])


def jsonlines(raw):
    require(not raw or raw.endswith(b"\n"), "partial final raw JSONL record")
    rows = []
    for line in raw.splitlines():
        require(bool(line), "empty raw JSONL record")
        rows.append(parse(line))
    return rows


def verify_stage_raw(published, role, evidence, *, raw_prefix=None):
    """Reassemble actual rank logs without altering any generated measurement."""
    prefix = raw_prefix or "artifacts/stages/" + role + "/raw/"
    selected = {name: raw for name, raw in published["raw"].items() if name.startswith(prefix)}
    if role == "development" and raw_prefix is not None:
        # The original fit eval directory also contains these two immutable
        # manifests, which the independent spec reconstruction checks separately.
        for name in ("semantic-prefix-manifest.json", "tokenized-prefix-manifest.json"):
            selected.pop(prefix + name, None)
    names = ("attempts",) if role in {"train_probe", "formal_store"} else ("generations", "prefix-scores")
    rank_counts = []
    combined = {name: [] for name in names}
    for count in (4,):
        expected = {prefix + f"rank-{rank}/{name}.jsonl" for rank in range(count) for name in names}
        if set(selected) == expected:
            rank_counts.append(count)
    require(len(rank_counts) == 1, "stage raw rank inventory is incomplete or unexpected")
    world = rank_counts[0]
    prompt_ranks = (
        {
            prompt: index % world
            for index, prompt in enumerate(dict.fromkeys(row["prompt_id"] for row in evidence["attempts"]))
        }
        if "attempts" in combined
        else {}
    )
    for rank in range(world):
        for name in names:
            rows = jsonlines(selected[prefix + f"rank-{rank}/{name}.jsonl"])
            if name == "generations":
                start = 128 if role == "supplemental" else 0
                require(
                    all(
                        type(row.get("global_index")) is int and (row["global_index"] - start) % world == rank
                        for row in rows
                    ),
                    "raw readiness generation belongs to another rank",
                )
            elif name == "prefix-scores":
                require(
                    all(
                        type(row.get("global_index")) is int and row["global_index"] % world == rank
                        for row in rows
                    ),
                    "raw prefix measurement belongs to another rank",
                )
            elif name == "attempts":
                require(
                    all(prompt_ranks.get(row.get("prompt_id")) == rank for row in rows),
                    "raw candidate belongs to another rank",
                )
            combined[name].extend(rows)
    if "attempts" in combined:
        order = {row["attempt_id"]: index for index, row in enumerate(evidence["attempts"])}
        require(len(order) == len(evidence["attempts"]), "duplicate accepted-envelope candidate")
        require(all(row.get("attempt_id") in order for row in combined["attempts"]), "unknown raw candidate")
        observed = sorted(combined["attempts"], key=lambda row: order[row["attempt_id"]])
        require(
            canonical(observed) == canonical(evidence["attempts"]),
            "sealed candidate evidence differs from raw output",
        )
    else:
        # The producer seals all_gather_object's rank-major order. Retain that
        # exact order; sorting would change the published evidence identity.
        generations = combined["generations"]
        scores = combined["prefix-scores"]
        require(
            canonical(generations) == canonical(evidence["generations"]),
            "sealed generations differ from raw output",
        )
        require(
            canonical(scores) == canonical(evidence["prefix_scores"]),
            "sealed prefix scores differ from raw output",
        )
    return {
        "world_size": world,
        "raw_file_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in selected.items()},
    }


def verify_claim_chain(published, claims, stages, summaries, selection_sha256, contract, *, claims_root):
    """Read the permanent claims themselves, not only producer-copied assertions."""
    binding = contract.verify_claims(claims, expected_root=Path(claims_root))
    receipt = published["receipt"]
    for key in ("run_id", "code_sha256"):
        require(binding.get(key) == receipt.get(key), "claim belongs to another source/run")
    snapshots = {}
    for row in claims["reservations"]:
        raw = safe_read(Path(row["path"]), MAX_CONTROL_BYTES)
        require(
            hashlib.sha256(raw).hexdigest() == row["sha256"] and parse(raw) == row["value"],
            "permanent reservation differs from staged evidence",
        )
        snapshots[row["path"]] = hashlib.sha256(raw).hexdigest()
    previous = selection_sha256
    for role in ROLES:
        stage_root = Path(claims["stage_root"])
        claim_path, outcome_path = (
            stage_root / (role + suffix) for suffix in (".claim.json", ".outcome.json")
        )
        claim_raw, outcome_raw = (safe_read(path, MAX_CONTROL_BYTES) for path in (claim_path, outcome_path))
        claim, outcome = parse(claim_raw), parse(outcome_raw)
        evidence = stages[role]
        evidence_raw = published["raw"]["artifacts/stages/" + role + "/evidence.json"]
        require(
            claim == document(published, "artifacts/stages/" + role + "/claim.json")
            and outcome == document(published, "artifacts/stages/" + role + "/outcome.json"),
            "published stage copies differ from permanent claim/outcome",
        )
        require(
            claim.get("schema") == "quest-sdsc-teacher-qualification-stage-claim-v1"
            and claim.get("claimed_before_inference") is True
            and all(claim.get(key) == value for key, value in binding.items())
            and claim.get("role") == role
            and claim.get("job_id") == receipt["job_id"]
            and claim.get("claim_id") == claims["stage_claim_ids"][role] == evidence.get("claim_id")
            and claim.get("predecessor_sha256") == previous == evidence.get("predecessor_sha256"),
            "permanent stage claim breaks the fixed checkpoint or predecessor chain",
        )
        expected = {
            "schema": "quest-sdsc-teacher-qualification-stage-outcome-v1",
            "job_id": receipt["job_id"],
            "role": role,
            "claim_id": claim["claim_id"],
            "claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
            "predecessor_sha256": previous,
            "evidence_sha256": evidence["sha256"],
            "evidence_file_sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "evidence_file_size": len(evidence_raw),
            "summary_sha256": sha(summaries[role]),
            "metrics_passed": True,
        }
        require(
            canonical(outcome) == canonical(expected),
            "persistent stage outcome differs from actual evidence/replay",
        )
        snapshots[str(claim_path)] = hashlib.sha256(claim_raw).hexdigest()
        snapshots[str(outcome_path)] = hashlib.sha256(outcome_raw).hexdigest()
        previous = evidence["sha256"]
    return snapshots


def _sibling(root, name):
    path = Path(root) / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_qualification_audit_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_once(path, value):
    path = Path(path).absolute()
    require(not any(item.is_symlink() for item in (path, *path.parents)), "unsafe audit destination")
    raw = canonical(value) + b"\n"
    require(len(raw) <= MAX_AUDIT_BYTES, "audit result is not small metadata")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(safe_read(path, MAX_CONTROL_BYTES) == raw, "audit report failed read-back")


def verify_store(published, evidence, summary, identity, protocol_sha256):
    """Derive the accepted view again from independently verified raw attempts."""
    prefix = "artifacts/teacher-store/"
    ledger = jsonlines(published["raw"][prefix + "attempts.jsonl"])
    require(
        canonical(ledger) == canonical(evidence["attempts"]),
        "teacher store ledger differs from measured candidates",
    )
    view_payload = {
        "format_version": 1,
        "artifact_kind": "adapted_teacher_accepted_view",
        "teacher_identity": identity,
        "protocol_sha256": protocol_sha256,
        "evidence_sha256": evidence["sha256"],
        "attempt_ids": summary["accepted_attempt_ids"],
        "formal_teacher_accepted": False,
    }
    view = {**view_payload, "sha256": sha(view_payload)}
    require(
        canonical(document(published, prefix + "accepted-view.json")) == canonical(view),
        "teacher store accepted view is not derived from verifier replay",
    )
    records = []
    for name in ("attempts.jsonl", "accepted-view.json"):
        raw = published["raw"][prefix + name]
        records.append({"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    payload = {
        "format_version": 1,
        "artifact_kind": "adapted_teacher_candidate_store",
        "teacher_identity": identity,
        "protocol_sha256": protocol_sha256,
        "evidence_sha256": evidence["sha256"],
        "ordered_prompt_ids": list(dict.fromkeys(row["prompt_id"] for row in ledger)),
        "attempt_count": len(ledger),
        "covered_prompts": summary["covered_prompts"],
        "accepted_view_sha256": view["sha256"],
        "formal_teacher_accepted": False,
        "files": records,
    }
    expected = {**payload, "sha256": sha(payload)}
    require(
        canonical(document(published, prefix + "manifest.json")) == canonical(expected),
        "teacher store manifest differs from independently derived contents",
    )
    return expected


def audit_result(
    *, science_root, expected_head, result_root, publication_sha256, job_id, claims_root, git_dir=None
):
    """Independent CPU replay. Caller separately verifies real Slurm accounting."""
    science_root = Path(science_root).absolute()
    require(science_root.resolve() == science_root, "audit science root is not a real path")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(science_root / "src"))
    from posttrain_circuits.artifacts.teacher_adaptation_protocol import resolve_teacher_adaptation_protocol

    resolved = resolve_teacher_adaptation_protocol(science_root, expected_head=expected_head, git_dir=git_dir)
    executable_sha = hashlib.sha256(safe_read(Path(__file__).absolute(), MAX_CONTROL_BYTES)).hexdigest()
    require(
        executable_sha == resolved.science_file_sha256["tools/sdsc_teacher_qualification_audit.py"],
        "executing auditor differs from reviewed scientific source",
    )
    require(
        resolved.review_status == "accepted" and resolved.tracked_worktree_clean,
        "formal CPU audit requires an actual clean accepted science checkout",
    )
    published = publication(result_root, publication_sha256, job_id)
    report = document(published, "teacher-qualify.json")
    require(
        report.get("task") == TASK
        and report.get("job_id") == job_id
        and report.get("passed") is True
        and report.get("scientific_evidence_verified") is True
        and report.get("exit_code") == 0
        and type(report.get("exit_code")) is int
        and report.get("world_size") == 4
        and type(report.get("world_size")) is int
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
        ),
        "producer report is incomplete, mismatched or self-accepting",
    )
    for key in ("run_id", "code_sha256"):
        require(report.get(key) == published["receipt"].get(key), "report/receipt source identity differs")
    protocol_binding = {
        key: getattr(resolved, key)
        for key in (
            "protocol_sha256",
            "artifact_sha256",
            "implementation_commit",
            "acceptance_commit",
            "head",
            "science_implementation_sha256",
        )
    }
    require(
        report.get("protocol_resolution") == protocol_binding, "report has another actual protocol lineage"
    )
    for field, key in (
        ("protocol_sha256", "protocol_sha256"),
        ("protocol_artifact_sha256", "artifact_sha256"),
        ("protocol_implementation_commit", "implementation_commit"),
        ("protocol_acceptance_commit", "acceptance_commit"),
        ("science_implementation_sha256", "science_implementation_sha256"),
        ("science_git_head", "head"),
    ):
        require(report.get(field) == protocol_binding[key], "top-level report protocol binding differs")
    contract, worker = (
        _sibling(science_root, name) for name in ("sdsc_teacher_qualify_contract", "sdsc_teacher_qualify")
    )
    prerequisites = document(published, "artifacts/inputs/prerequisites.json")
    claims = document(published, "artifacts/inputs/claims.json")
    contract.validate_report(report, prerequisites)
    for name in ("prerequisites", "claims"):
        raw = published["raw"][f"artifacts/inputs/{name}.json"]
        require(hashlib.sha256(raw).hexdigest() == report[name + "_sha256"], "original input bytes changed")
    expected_protocol = prerequisites["protocol"]
    require(
        all(expected_protocol.get(key) == value for key, value in protocol_binding.items())
        and expected_protocol.get("science_file_sha256") == resolved.science_file_sha256
        and expected_protocol.get("payload") == resolved.payload,
        "prerequisite differs from the genuinely accepted protocol",
    )
    binding = contract.verify_claims(claims, expected_root=Path(claims_root))
    require(
        binding["protocol_sha256"] == resolved.protocol_sha256
        and binding["prerequisites_sha256"] == report["prerequisites_sha256"]
        and binding["adapted_teacher_sha256"] == report["adapted_teacher_sha256"]
        and prerequisites.get("target")
        == {
            "run_id": report["run_id"],
            "code_sha256": report["code_sha256"],
            "intent_id": binding["intent_id"],
        },
        "original reservation/prerequisite/checkpoint identity differs",
    )
    fit = worker.load_bound_fit_evidence(
        published["root"] / "artifacts/fit-evidence",
        prerequisites,
        contract,
        publication_receipt_path=published["root"] / "artifacts/inputs/fit-publication-receipt.json",
    )
    from transformers import AutoTokenizer

    from posttrain_circuits.artifacts.teacher_acceptance import build_teacher_acceptance
    from posttrain_circuits.core.config import compose_config

    guards = _sibling(science_root, "sdsc_teacher_prepare")
    config = compose_config(list(guards.OVERRIDES), config_root=science_root / "configs")
    selected = prerequisites["selected_checkpoint"]
    expected_tokenizer = {
        "artifacts/tokenizer/" + Path(row["path"]).name: row
        for row in selected["files"]
        if len(PurePosixPath(row["path"]).parts) == 2
        and PurePosixPath(row["path"]).parts[0] == "merged"
        and Path(row["path"]).name in worker.TOKENIZER_FILES
    }
    require(
        set(expected_tokenizer)
        == {name for name in published["files"] if name.startswith("artifacts/tokenizer/")},
        "published tokenizer inventory differs from the selected dense checkpoint",
    )
    for name, record in expected_tokenizer.items():
        require(
            all(published["files"][name][key] == record[key] for key in ("size", "sha256")),
            "tokenizer bytes do not belong to the selected checkpoint",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        published["root"] / "artifacts/tokenizer", local_files_only=True, trust_remote_code=False
    )
    inputs = worker.replay_development(fit, config, tokenizer, resolved.protocol_sha256)
    require(
        inputs["selection"] == document(published, "artifacts/selection.json"),
        "selected checkpoint differs from full development replay",
    )
    require(
        inputs["selection"]["sha256"] == report["selection_sha256"]
        and inputs["selection"]["selected_step"] == report["selected_step"]
        and inputs["selection"]["teacher_identity"]["teacher_checkpoint_sha256"]
        == report["adapted_teacher_sha256"],
        "report chose another checkpoint",
    )
    development_raw_proofs = []
    for checkpoint, measured in zip(inputs["checkpoints"], inputs["development"], strict=True):
        prefix = f"artifacts/development/step-{checkpoint['step']:06d}/"
        development_raw_proofs.append(
            {
                "step": checkpoint["step"],
                **verify_stage_raw(
                    published,
                    "development",
                    checkpoint["evidence"],
                    raw_prefix=f"artifacts/fit-evidence/artifacts/checkpoints/step-{checkpoint['step']:06d}/eval/",
                ),
            }
        )
        for name, expected in (
            ("spec", inputs["specs"]["development"]),
            ("evidence", checkpoint["evidence"]),
            ("summary", measured["summary"]),
        ):
            require(
                canonical(document(published, prefix + name + ".json")) == canonical(expected),
                "published development conversion differs from original raw fit measurements",
            )
    stages, bindings, raw_proofs = {}, dict(inputs["expected_bindings"]), {}
    for role in ROLES:
        prefix = "artifacts/stages/" + role + "/"
        require(
            document(published, prefix + "spec.json") == inputs["specs"][role],
            "stage evaluation specification differs",
        )
        stages[role] = document(published, prefix + "evidence.json")
        outcome = parse(safe_read(Path(claims["stage_root"]) / (role + ".outcome.json"), MAX_CONTROL_BYTES))
        bindings[role] = {
            "sha256": outcome["evidence_sha256"],
            "claim_id": outcome["claim_id"],
            "predecessor_sha256": outcome["predecessor_sha256"],
        }
        raw_proofs[role] = verify_stage_raw(published, role, stages[role])
    # This is the independent complete domain replay, not a producer PASS check.
    recomputed = build_teacher_acceptance(
        protocol_sha256=resolved.protocol_sha256,
        fit_publication_sha256=inputs["fit_publication_sha256"],
        fit_plan_sha256=inputs["fit_plan_sha256"],
        checkpoints=inputs["checkpoints"],
        stages=stages,
        expected_bindings=bindings,
        examples_by_role=inputs["examples_by_role"],
        specs=inputs["specs"],
        tokenizer=tokenizer,
        teacher_config=config["teacher"],
    )
    published_acceptance = document(published, "artifacts/acceptance-evidence.json")
    require(
        canonical(published_acceptance) == canonical(recomputed),
        "producer acceptance evidence differs from independent complete replay",
    )
    require(
        report["acceptance_evidence_sha256"] == recomputed["sha256"]
        and report["acceptance_evidence_file_sha256"]
        == published["files"]["artifacts/acceptance-evidence.json"]["sha256"],
        "report acceptance binding differs",
    )
    summaries = recomputed["stage_summaries"]
    for role in ROLES:
        require(
            canonical(document(published, f"artifacts/stages/{role}/summary.json"))
            == canonical(summaries[role]),
            "published stage summary differs from actual independent reduction",
        )
        compact = {
            key: value
            for key, value in summaries[role].items()
            if key not in {"full_generation_rows", "accepted_attempt_ids", "prefix_scores"}
        }
        require(
            report["stage_summaries"][role] == compact, "report stage summary differs from independent replay"
        )
    claim_proof = verify_claim_chain(
        published, claims, stages, summaries, inputs["selection"]["sha256"], contract, claims_root=claims_root
    )
    store = verify_store(
        published,
        stages["formal_store"],
        summaries["formal_store"],
        recomputed["teacher_identity"],
        resolved.protocol_sha256,
    )
    require(store["sha256"] == report["teacher_store_sha256"], "report store identity differs")
    inventory = document(published, "artifacts/qualification-manifest.json")
    require(
        inventory.get("artifact_kind") == "adapted_teacher_qualification_inventory"
        and inventory.get("formal_teacher_accepted") is False
        and inventory.get("teacher_identity") == recomputed["teacher_identity"]
        and all(inventory.get(key) == report[key] for key in ("job_id", "run_id", "code_sha256"))
        and inventory.get("sha256")
        == sha({key: value for key, value in inventory.items() if key != "sha256"})
        == report["qualification_inventory_sha256"]
        and published["files"]["artifacts/qualification-manifest.json"]["sha256"]
        == report["qualification_manifest_sha256"],
        "qualification inventory identity differs",
    )
    inventory_records = contract.record_map(inventory["files"])
    expected_names = {
        name.removeprefix("artifacts/")
        for name in published["files"]
        if name.startswith("artifacts/")
        and (
            name.split("/")[1]
            in {"inputs", "fit-evidence", "tokenizer", "development", "stages", "teacher-store"}
            or name in {"artifacts/selection.json", "artifacts/acceptance-evidence.json"}
        )
    }
    require(set(inventory_records) == expected_names, "scientific inventory omits or adds evidence")
    for name, record in inventory_records.items():
        require(
            published["files"]["artifacts/" + name] == {**record, "path": "artifacts/" + name},
            "inventory and actual publication differ",
        )
    # Recheck the actual bytes after parsing/replay and the permanent history.
    for name, record in published["files"].items():
        limit = MAX_SCIENCE_BYTES if scientific_file(name) else MAX_LOG_BYTES
        require(
            hashlib.sha256(safe_read(published["root"] / name, limit)).hexdigest() == record["sha256"],
            "publication changed during audit",
        )
    require(
        safe_read(published["root"] / "receipt.json", MAX_CONTROL_BYTES) == published["receipt_raw"],
        "receipt changed during audit",
    )
    for name, expected in claim_proof.items():
        require(
            hashlib.sha256(safe_read(Path(name), MAX_CONTROL_BYTES)).hexdigest() == expected,
            "permanent claim changed during audit",
        )
    final_source = resolve_teacher_adaptation_protocol(
        science_root, expected_head=expected_head, git_dir=git_dir
    )
    require(
        final_source.science_implementation_sha256 == resolved.science_implementation_sha256,
        "reviewed scientific source changed during replay",
    )
    payload = {
        "format_version": 1,
        "artifact_kind": "independent_adapted_teacher_cpu_audit",
        "job_id": job_id,
        "run_id": report["run_id"],
        "code_sha256": report["code_sha256"],
        "protocol_resolution": protocol_binding,
        "publication_receipt_sha256": publication_sha256,
        "scientific_physical_bytes": published["scientific_bytes"],
        "scientific_unique_content_bytes": published["unique_scientific_bytes"],
        "log_bytes": published["log_bytes"],
        "actual_file_records": list(published["files"].values()),
        "permanent_claim_file_sha256": claim_proof,
        "raw_stage_proofs": raw_proofs,
        "raw_development_proofs": development_raw_proofs,
        "acceptance_evidence_sha256": recomputed["sha256"],
        "recomputed_acceptance_evidence": recomputed,
        "teacher_identity": recomputed["teacher_identity"],
        "selection": recomputed["selection"],
        "teacher_store_sha256": store["sha256"],
        "stage_summaries": {role: report["stage_summaries"][role] for role in ROLES},
        "raw_evidence_independently_replayed": True,
        "passed": True,
        "formal_teacher_accepted": False,
        "weights_read_or_model_inference_performed": False,
        "accounting_checked_by_this_program": False,
        "external_accounting_and_publication_origin_review_required": True,
        "independent_final_acceptance_attestation_required": True,
        "student_training_started": False,
    }
    result = {**payload, "sha256": sha(payload)}
    require(len(canonical(result)) + 1 <= MAX_AUDIT_BYTES, "audit report exceeds one MiB")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "result-root", "claims-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-head", "publication-sha256", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--git-dir", type=Path)
    args = parser.parse_args(argv)
    output, root = args.output.absolute(), args.result_root.absolute()
    require(
        output != root and root not in output.parents,
        "audit output must not modify the published evidence tree",
    )
    report = audit_result(
        science_root=args.science_root,
        expected_head=args.expected_head,
        result_root=root,
        publication_sha256=args.publication_sha256,
        job_id=args.job_id,
        claims_root=args.claims_root,
        git_dir=args.git_dir,
    )
    _write_once(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": report["sha256"],
                "passed": report["passed"],
                "formal_teacher_accepted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
