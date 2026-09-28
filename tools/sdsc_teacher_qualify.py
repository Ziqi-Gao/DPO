#!/usr/bin/env python3
"""Measure one selected dense teacher; independent acceptance remains external."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from datetime import timedelta
from pathlib import Path

ROLES = ("train_probe", "supplemental", "formal_readiness", "formal_store")
CHECKS = (
    "genuine_reviewed_science",
    "verified_fit_and_selection",
    "verified_dense_checkpoint",
    "all_four_ranks",
    *ROLES,
    "cgroup_headroom",
)
TOKENIZER_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
}


def sibling(name):
    spec = importlib.util.spec_from_file_location(
        "_qualification_" + name, Path(__file__).with_name(name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    sibling("sdsc_teacher_probe").progress_json(path, value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, contract, limit=None):
    return json.loads(contract.small_bytes(path, limit or contract.MAX_METADATA_BYTES))


def verify_tree(root, records, contract, *, total_limit):
    """Hash actual regular staged bytes, rejecting extras, aliases and changes."""
    root = contract.safe(root)
    by_name = contract.record_map(records)
    require(sum(row["size"] for row in records) <= total_limit, "staged evidence exceeds its byte limit")
    actual = set()
    for path in root.rglob("*"):
        contract.safe(path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), "nonregular staged entry")
    require(actual == set(by_name), "staged file population differs")
    for name, row in by_name.items():
        path = contract.safe(root / name)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == row["size"],
                "staged file is not one bounded regular file",
            )
            hasher, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                count += len(block)
                require(count <= row["size"], "staged file grew while hashing")
                hasher.update(block)
            after = os.fstat(stream.fileno())
        require(
            count == row["size"]
            and hasher.hexdigest() == row["sha256"]
            and all(
                getattr(before, key) == getattr(after, key)
                for key in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ),
            "staged content changed or differs from publication",
        )
    return by_name


def load_bound_fit_evidence(root, prerequisites, contract, *, publication_receipt_path=None):
    """Rehash the complete bounded raw fit subset against its verified receipt."""
    upstream = prerequisites["upstream"]
    expected = contract.fit_evidence(upstream)
    require(expected == prerequisites["fit_evidence"], "staged fit subset differs from original publication")
    verify_tree(root, expected["files"], contract, total_limit=contract.MAX_FIT_EVIDENCE_BYTES)
    publication = contract.small_bytes(publication_receipt_path or upstream["publication_receipt_path"])
    require(
        digest(publication) == upstream["publication_receipt_sha256"]
        and json.loads(publication) == upstream["publication_receipt"],
        "fit publication receipt changed",
    )
    documents = {}
    for row in expected["files"]:
        raw = contract.small_bytes(Path(root) / row["path"], contract.MAX_FIT_EVIDENCE_BYTES)
        require(
            len(raw) == row["size"] and digest(raw) == row["sha256"],
            "fit evidence changed before its actual parse",
        )
        documents[row["path"]] = (
            [json.loads(line) for line in raw.splitlines()]
            if row["path"].endswith(".jsonl")
            else json.loads(raw)
        )
    report = documents["teacher-fit.json"]
    require(report == upstream["report"], "staged fit report differs from prerequisite")
    fit_contract = sibling("sdsc_teacher_fit_contract")
    execution = contract.verify_fit_origin(upstream, prerequisites["protocol"])
    fit_contract.validate_report(report, execution, "qwen3-v2-teacher-fit")
    receipt = upstream["receipt"]
    require(
        all(report[key] == receipt[key] for key in ("job_id", "run_id", "code_sha256"))
        and report["job_id"] == prerequisites["teacher_job_id"],
        "fit source/job identity differs",
    )
    require(
        upstream["status"].get("success") is True and upstream["status"].get("state") == "COMPLETED",
        "full-fit accounting is not verified complete",
    )
    manifest = documents["artifacts/checkpoint-manifest.json"]
    require(
        manifest["sha256"]
        == report["checkpoint_set_sha256"]
        == contract.sha256_value({key: value for key, value in manifest.items() if key != "sha256"}),
        "fit checkpoint set identity differs",
    )
    published = contract.record_map(upstream["publication_receipt"]["files"])
    require(
        published["artifacts/checkpoint-manifest.json"]["sha256"] == report["checkpoint_manifest_sha256"],
        "fit checkpoint manifest physical identity differs",
    )
    for row in contract.record_map(manifest["files"]).values():
        require(
            published.get("artifacts/" + row["path"]) == dict(row, path="artifacts/" + row["path"]),
            "checkpoint inventory differs from durable publication",
        )
    require(
        [row["step"] for row in manifest["checkpoints"]] == [128, 256, 384, 512],
        "fit did not publish all four scheduled checkpoints",
    )
    for row in manifest["checkpoints"]:
        prefix = "artifacts/" + row["directory"] + "/"
        require(row["directory"] == f"checkpoints/step-{row['step']:06d}", "checkpoint directory differs")
        for name, key in (
            ("dense-manifest.json", "dense_manifest_sha256"),
            ("manifest.json", "trainer_state_manifest_sha256"),
            ("dev-capability.json", "dev_metrics_sha256"),
        ):
            require(published[prefix + name]["sha256"] == row[key], "fit checkpoint record hash differs")
        dense, state = documents[prefix + "dense-manifest.json"], documents[prefix + "manifest.json"]
        require(
            dense["sha256"] == row["adapted_teacher_sha256"]
            and state["sha256"] == contract.sha256_value({k: v for k, v in state.items() if k != "sha256"}),
            "dense/state checkpoint identity differs",
        )
        provenance = dense["training_provenance"]
        require(
            all(provenance[key] == report[key] for key in ("job_id", "run_id", "code_sha256"))
            and provenance["dataset_manifest_sha256"] == report["adaptation_dataset_sha256"]
            and provenance["optimizer_steps"] == row["step"]
            and provenance["consumed_tokens"] == row["consumed_tokens"]
            and provenance["train_metrics_sha256"] == row["training_metrics_sha256"]
            and state["metadata"]["completed_updates"] == row["step"]
            and state["metadata"]["consumed_tokens"] == row["consumed_tokens"],
            "dense checkpoint training provenance differs",
        )
    return {
        "documents": documents,
        "report": report,
        "checkpoint_manifest": manifest,
        "records": expected["files"],
        "fit_publication_sha256": upstream["publication_receipt_sha256"],
        "publication_receipt_bytes": publication,
    }


def replay_development(fit, config, tokenizer, protocol_sha256):
    """Reframe already published measurements as audit evidence, never new claims.

    The fit:<job>:development:<step> identifiers are post-publication audit IDs,
    not assertions that the old fit performed new-protocol inference or claims.
    Every generation/score remains unchanged and old summaries must reproduce.
    """
    from posttrain_circuits.artifacts import teacher_acceptance as acceptance
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.splits import build_split

    report, documents = fit["report"], fit["documents"]
    require(
        report["task_configuration"] == config["task"]
        and report["teacher_configuration"] == config["teacher"],
        "fit task/inference configuration changed",
    )
    task = ProofGraphTask()
    train = build_split(task, "train", 256, 42, config["task"])
    validation = build_split(task, "validation", 256, 42, config["task"])
    examples = {
        "development": build_split(task, "validation", 512, 70_000_042, config["task"]),
        "train_probe": train[:32],
        "supplemental": validation[128:256],
        "formal_readiness": validation[:128],
        "formal_store": train,
    }
    specs = {
        role: acceptance.build_evaluation_spec(
            role, rows, tokenizer, config["teacher"], protocol_sha256=protocol_sha256
        )
        for role, rows in examples.items()
    }
    checkpoints, bindings, previous = [], {}, fit["fit_publication_sha256"]
    for row in fit["checkpoint_manifest"]["checkpoints"]:
        step, prefix = row["step"], "artifacts/" + row["directory"] + "/"
        dense = documents[prefix + "dense-manifest.json"]
        model_identity = acceptance.adapted_teacher_identity(dense, expected_manifest_sha256=dense["sha256"])
        evidence = acceptance.sealed(
            {
                "format_version": 1,
                "artifact_kind": "adapted_teacher_readiness_evidence",
                "role": "development",
                "teacher_identity": model_identity.to_mapping(),
                "spec_sha256": specs["development"]["sha256"],
                "claim_id": f"fit:{report['job_id']}:development:{step}",
                "predecessor_sha256": previous,
                "generations": [
                    value
                    for rank in range(4)
                    for value in documents[prefix + f"eval/rank-{rank}/generations.jsonl"]
                ],
                "prefix_scores": [
                    value
                    for rank in range(4)
                    for value in documents[prefix + f"eval/rank-{rank}/prefix-scores.jsonl"]
                ],
            }
        )
        bindings[f"development:{step}"] = {
            key: evidence[key] for key in ("sha256", "claim_id", "predecessor_sha256")
        }
        checkpoints.append({"step": step, "manifest": dense, "evidence": evidence})
        previous = evidence["sha256"]
        old = documents[prefix + "dev-capability.json"]
        require(
            old["adapted_teacher_sha256"] == dense["sha256"]
            and old["semantic_prefix_manifest_sha256"]
            == specs["development"]["semantic_prefix_manifest_sha256"]
            and old["tokenized_prefix_manifest_sha256"]
            == specs["development"]["tokenized_prefix_manifest_sha256"]
            and old["alignment_skips"] == specs["development"]["alignment_skips"]
            and documents[prefix + "eval/semantic-prefix-manifest.json"]["sha256"]
            == old["semantic_prefix_manifest_sha256"]
            and documents[prefix + "eval/tokenized-prefix-manifest.json"]["sha256"]
            == old["tokenized_prefix_manifest_sha256"],
            "published development probe identity differs",
        )
    checked = acceptance.validate_development_selection(
        protocol_sha256=protocol_sha256,
        fit_publication_sha256=fit["fit_publication_sha256"],
        fit_plan_sha256=report["actual_plan_sha256"],
        checkpoints=checkpoints,
        expected_bindings=bindings,
        examples=examples["development"],
        spec=specs["development"],
        tokenizer=tokenizer,
        teacher_config=config["teacher"],
    )
    for measured, original in zip(
        checked["development"], fit["checkpoint_manifest"]["checkpoints"], strict=True
    ):
        old = documents["artifacts/" + original["directory"] + "/dev-capability.json"]
        require(
            all(
                measured["summary"][key] == old[key]
                for key in ("metrics", "checks", "thresholds", "metrics_passed", "full_generation_rows")
            )
            and measured["summary"]["metrics_passed"] is original["dev_metrics_passed"],
            "raw development replay differs from original full scientific summary",
        )
    require(
        checked["selection"]["teacher_identity"]["teacher_checkpoint_sha256"]
        == report["selected_checkpoint_sha256"],
        "recomputed first passing checkpoint differs from the fit selection",
    )
    return {
        **checked,
        "checkpoints": checkpoints,
        "expected_bindings": bindings,
        "examples_by_role": examples,
        "specs": specs,
        "fit_publication_sha256": fit["fit_publication_sha256"],
        "fit_plan_sha256": report["actual_plan_sha256"],
    }


def metadata_preflight(args, contract):
    guards, fit = sibling("sdsc_teacher_prepare"), sibling("sdsc_teacher_fit")
    require(contract.SHA256.fullmatch(args.code_sha256), "invalid deployment identity")
    provenance = guards.validate_checkout(args)
    allocation, runtime, environment = fit.runtime_and_allocation(
        args, sibling("sdsc_teacher_fit_contract"), guards
    )
    for path in (args.checkpoint_root, args.fit_evidence_root, args.prerequisites, args.claims):
        contract.safe(path)
        require(args.work_dir in path.parents, "qualification inputs must be staged in node-local work")
    raw = contract.small_bytes(args.prerequisites)
    prerequisites, claims = json.loads(raw), read_json(args.claims, contract)
    binding = contract.verify_claims(claims)
    require(
        prerequisites.get("schema") == "quest-sdsc-teacher-qualification-prerequisites-v1"
        and prerequisites.get("task") == contract.TASK
        and prerequisites["target"]
        == {"run_id": args.run_id, "code_sha256": args.code_sha256, "intent_id": binding["intent_id"]}
        and all(
            binding[key] == value
            for key, value in {
                "run_id": args.run_id,
                "code_sha256": args.code_sha256,
                "prerequisites_sha256": digest(raw),
                "adapted_teacher_sha256": args.checkpoint_sha256,
                "teacher_job_id": prerequisites["teacher_job_id"],
            }.items()
        ),
        "qualification prerequisite/claim bindings differ",
    )
    require(
        not any(Path(claims["stage_root"]).glob("*.json")), "qualification stage already claimed; no replay"
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.science_root / "src"))
    from posttrain_circuits.artifacts.teacher_adaptation_protocol import resolve_teacher_adaptation_protocol

    resolved = resolve_teacher_adaptation_protocol(
        args.science_root, expected_head=provenance["science_git_head"]
    )
    require(
        resolved.review_status == "accepted" and resolved.tracked_worktree_clean,
        "qualification requires genuine clean accepted scientific source",
    )
    expected = prerequisites["protocol"]
    for key in (
        "protocol_sha256",
        "artifact_sha256",
        "implementation_commit",
        "acceptance_commit",
        "head",
        "science_file_sha256",
        "science_implementation_sha256",
        "review_status",
        "payload",
    ):
        require(getattr(resolved, key) == expected[key], "accepted protocol differs: " + key)
    require(binding["protocol_sha256"] == resolved.protocol_sha256, "claim protocol differs")
    selected = prerequisites["selected_checkpoint"]
    require(
        selected["adapted_teacher_sha256"] == args.checkpoint_sha256, "selected checkpoint identity differs"
    )
    verify_tree(args.checkpoint_root, selected["files"], contract, total_limit=contract.MAX_CHECKPOINT_BYTES)
    fit_evidence = load_bound_fit_evidence(args.fit_evidence_root, prerequisites, contract)
    from posttrain_circuits.core.config import compose_config

    config = compose_config(list(guards.OVERRIDES), config_root=args.science_root / "configs")
    return dict(
        provenance=provenance,
        allocation=allocation,
        runtime=runtime,
        environment=environment,
        prerequisites=prerequisites,
        claims=claims,
        binding=binding,
        resolved=resolved,
        fit=fit_evidence,
        config=config,
        prerequisites_sha256=digest(raw),
        claims_sha256=digest(contract.small_bytes(args.claims)),
    )


def observation(output, report, rank):
    probe = sibling("sdsc_teacher_probe")
    directory = output if rank == 0 else output / f"rank-{rank}-progress"
    directory.mkdir(parents=True, exist_ok=True)

    class QualificationObservation(probe.ProbeObservation):
        def snapshot(self):
            value = super().snapshot()
            value.update(
                artifact_kind="teacher_qualification_progress",
                teacher_training_started=False,
                student_training_started=False,
                training_started=False,
                formal_teacher_accepted=False,
                readiness_artifact_produced=False,
            )
            return value

    return QualificationObservation(directory, report)


def copy_small_inputs(args, metadata, contract):
    output = args.output_dir
    (output / "inputs").mkdir()
    for name, path in (("prerequisites.json", args.prerequisites), ("claims.json", args.claims)):
        (output / "inputs" / name).write_bytes(contract.small_bytes(path))
    (output / "inputs" / "fit-publication-receipt.json").write_bytes(
        metadata["fit"]["publication_receipt_bytes"]
    )
    for row in metadata["fit"]["records"]:
        target = output / "fit-evidence" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = contract.small_bytes(args.fit_evidence_root / row["path"], contract.MAX_FIT_EVIDENCE_BYTES)
        require(
            len(raw) == row["size"] and digest(raw) == row["sha256"],
            "fit evidence changed before preservation",
        )
        target.write_bytes(raw)
    (output / "tokenizer").mkdir()
    for row in metadata["prerequisites"]["selected_checkpoint"]["files"]:
        relative = Path(row["path"])
        if relative.parts[0] == "merged" and len(relative.parts) == 2 and relative.name in TOKENIZER_FILES:
            raw = contract.small_bytes(args.checkpoint_root / relative, 32 * 1024**2)
            require(len(raw) == row["size"] and digest(raw) == row["sha256"], "tokenizer bytes changed")
            (output / "tokenizer" / relative.name).write_bytes(raw)


def persist_development(output, inputs):
    for checkpoint, measured in zip(inputs["checkpoints"], inputs["development"], strict=True):
        directory = output / "development" / f"step-{checkpoint['step']:06d}"
        directory.mkdir(parents=True)
        for name, value in (
            ("spec", inputs["specs"]["development"]),
            ("evidence", checkpoint["evidence"]),
            ("summary", measured["summary"]),
        ):
            write_json(directory / (name + ".json"), value)
    write_json(output / "selection.json", inputs["selection"])


def compact_summary(summary):
    return {
        key: value
        for key, value in summary.items()
        if key not in {"full_generation_rows", "accepted_attempt_ids", "prefix_scores"}
    }


def measure_stage(
    role,
    *,
    inputs,
    loaded,
    identity,
    config,
    output,
    claims,
    job_id,
    predecessor,
    contract,
    observe,
    rank,
    control,
):
    import torch.distributed as dist

    from posttrain_circuits.artifacts import teacher_acceptance as acceptance
    from posttrain_circuits.learning.teacher import adapted_candidate_generation as producer
    from posttrain_circuits.learning.teacher.adaptation_evaluation import build_adaptation_probes
    from posttrain_circuits.learning.teacher.adaptation_fit import _shared_call

    directory, spec, examples = (
        output / "stages" / role,
        inputs["specs"][role],
        inputs["examples_by_role"][role],
    )

    def claim():
        value = contract.claim_stage(claims, role, predecessor, job_id=job_id)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "claim.json", value)
        write_json(directory / "spec.json", spec)
        return value

    _shared_call(lambda: claim() if rank == 0 else None, control, "persistent claim before " + role)
    if role in acceptance.CANDIDATE_ROLES:
        measured = _shared_call(
            lambda: producer.measure_candidate_shard(
                examples,
                loaded,
                config["teacher"],
                identity,
                directory / "raw" / f"rank-{rank}",
                rank=rank,
                world_size=4,
                observe=observe,
            ),
            control,
            role + " candidates",
        )
    else:
        probes = _shared_call(
            lambda: build_adaptation_probes(examples, loaded, config["teacher"])[1],
            control,
            role + " prefix construction",
        )
        measured = _shared_call(
            lambda: producer.measure_readiness_shard(
                examples,
                probes,
                loaded,
                config["teacher"],
                identity,
                directory / "raw" / f"rank-{rank}",
                source_index_start=128 if role == "supplemental" else 0,
                rank=rank,
                world_size=4,
                observe=observe,
            ),
            control,
            role + " readiness",
        )
    gathered = [None] * 4
    dist.all_gather_object(gathered, measured, group=control)

    def verify_and_seal():
        payload = {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_candidate_evidence"
            if role in acceptance.CANDIDATE_ROLES
            else "adapted_teacher_readiness_evidence",
            "role": role,
            "teacher_identity": identity.to_mapping(),
            "spec_sha256": spec["sha256"],
            "claim_id": claims["stage_claim_ids"][role],
            "predecessor_sha256": predecessor,
        }
        if role in acceptance.CANDIDATE_ROLES:
            ordered = {example.example_id: index for index, example in enumerate(examples)}
            payload["attempts"] = sorted(
                [row for shard in gathered for row in shard],
                key=lambda row: (ordered[row["prompt_id"]], row["candidate_index"]),
            )
        else:
            require(
                {shard["rank"] for shard in gathered} == {0, 1, 2, 3}
                and all(shard["world_size"] == 4 for shard in gathered),
                "measurement ranks differ",
            )
            payload.update(
                generations=[row for shard in gathered for row in shard["generations"]],
                prefix_scores=[row for shard in gathered for row in shard["prefix_scores"]],
            )
        evidence = acceptance.sealed(payload)
        binding = {key: evidence[key] for key in ("sha256", "claim_id", "predecessor_sha256")}
        validator = (
            acceptance.validate_candidate_evidence
            if role in acceptance.CANDIDATE_ROLES
            else acceptance.validate_readiness_evidence
        )
        summary = validator(
            spec,
            evidence,
            examples=examples,
            tokenizer=loaded.tokenizer,
            teacher_config=config["teacher"],
            identity=identity,
            expected_binding=binding,
            protocol_sha256=spec["protocol_sha256"],
        )
        write_json(directory / "evidence.json", evidence)
        write_json(directory / "summary.json", summary)
        outcome = contract.seal_stage(claims, role, directory / "evidence.json", summary, job_id=job_id)
        write_json(directory / "outcome.json", outcome)
        return {"evidence": evidence, "summary": summary, "binding": binding, "outcome": outcome}

    result = _shared_call(
        lambda: verify_and_seal() if rank == 0 else None, control, role + " evidence replay"
    )
    values = [result]
    dist.broadcast_object_list(values, src=0, group=control)
    return values[0]


def write_teacher_store(output, evidence, summary, identity, protocol_sha256):
    from posttrain_circuits.artifacts.teacher_acceptance import sealed

    directory = output / "teacher-store"
    directory.mkdir()
    with (directory / "attempts.jsonl").open("x") as stream:
        for row in evidence["attempts"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    view = sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_accepted_view",
            "teacher_identity": identity.to_mapping(),
            "protocol_sha256": protocol_sha256,
            "evidence_sha256": evidence["sha256"],
            "attempt_ids": summary["accepted_attempt_ids"],
            "formal_teacher_accepted": False,
        }
    )
    write_json(directory / "accepted-view.json", view)
    manifest = sealed(
        {
            "format_version": 1,
            "artifact_kind": "adapted_teacher_candidate_store",
            "teacher_identity": identity.to_mapping(),
            "protocol_sha256": protocol_sha256,
            "evidence_sha256": evidence["sha256"],
            "ordered_prompt_ids": list(dict.fromkeys(row["prompt_id"] for row in evidence["attempts"])),
            "attempt_count": len(evidence["attempts"]),
            "covered_prompts": summary["covered_prompts"],
            "accepted_view_sha256": view["sha256"],
            "formal_teacher_accepted": False,
            "files": [
                {
                    "path": name,
                    "size": (directory / name).stat().st_size,
                    "sha256": digest((directory / name).read_bytes()),
                }
                for name in ("attempts.jsonl", "accepted-view.json")
            ],
        }
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def finish_outputs(output, report, inputs, stages, identity, tokenizer, config, contract):
    from posttrain_circuits.artifacts.teacher_acceptance import build_teacher_acceptance

    evidence = build_teacher_acceptance(
        protocol_sha256=report["protocol_sha256"],
        fit_publication_sha256=inputs["fit_publication_sha256"],
        fit_plan_sha256=inputs["fit_plan_sha256"],
        checkpoints=inputs["checkpoints"],
        stages={role: stages[role]["evidence"] for role in ROLES},
        expected_bindings={
            **inputs["expected_bindings"],
            **{role: stages[role]["binding"] for role in ROLES},
        },
        examples_by_role=inputs["examples_by_role"],
        specs=inputs["specs"],
        tokenizer=tokenizer,
        teacher_config=config["teacher"],
    )
    store = write_teacher_store(
        output,
        stages["formal_store"]["evidence"],
        stages["formal_store"]["summary"],
        identity,
        report["protocol_sha256"],
    )
    write_json(output / "acceptance-evidence.json", evidence)
    records = []
    for path in sorted(output.rglob("*")):
        contract.safe(path)
        name = path.relative_to(output).as_posix()
        if path.is_file() and (
            name.split("/")[0]
            in {"inputs", "fit-evidence", "tokenizer", "development", "stages", "teacher-store"}
            or name in {"selection.json", "acceptance-evidence.json"}
        ):
            records.append({"path": name, "size": path.stat().st_size, "sha256": digest(path.read_bytes())})
    unique = {(row["size"], row["sha256"]) for row in records}
    require(
        sum(row["size"] for row in records) <= 256 * 1024**2
        and sum(size for size, _ in unique) <= 128 * 1024**2,
        "qualification evidence exceeds physical or unique-content audit bound",
    )
    manifest = {
        "artifact_kind": "adapted_teacher_qualification_inventory",
        "files": records,
        "run_id": report["run_id"],
        "job_id": report["job_id"],
        "code_sha256": report["code_sha256"],
        "teacher_identity": identity.to_mapping(),
        "formal_teacher_accepted": False,
    }
    manifest["sha256"] = contract.sha256_value(manifest)
    write_json(output / "qualification-manifest.json", manifest)
    report.update(
        acceptance_evidence_sha256=evidence["sha256"],
        acceptance_evidence_file_sha256=digest((output / "acceptance-evidence.json").read_bytes()),
        qualification_manifest_sha256=digest((output / "qualification-manifest.json").read_bytes()),
        qualification_inventory_sha256=manifest["sha256"],
        teacher_store_sha256=store["sha256"],
        scientific_inventory_bytes=sum(row["size"] for row in records),
        unique_scientific_inventory_bytes=sum(size for size, _ in unique),
        passed=True,
        scientific_evidence_verified=True,
        exit_code=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "science-root",
        "work-dir",
        "output-dir",
        "hf-home",
        "provenance-manifest",
        "checkpoint-root",
        "prerequisites",
        "claims",
        "fit-evidence-root",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("run-id", "code-sha256", "job-id", "provenance-manifest-sha256", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    contract = sibling("sdsc_teacher_qualify_contract")
    metadata = metadata_preflight(args, contract)
    if args.validate_only:
        require("torch" not in sys.modules, "metadata validation unexpectedly imported Torch")
        print(
            json.dumps({"validated": True, "protocol_accepted": True, "formal_teacher_accepted": False}),
            flush=True,
        )
        return 0
    rank, local, world = (int(os.environ.get(key, "-1")) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4 and rank == local and 0 <= rank < 4, "qualification requires four local ranks")
    resolved = metadata["resolved"]
    report = {
        "task": contract.TASK,
        "artifact_kind": "adapted_teacher_qualification_execution",
        "run_id": args.run_id,
        "code_sha256": args.code_sha256,
        "job_id": args.job_id,
        "intent_id": metadata["binding"]["intent_id"],
        "teacher_job_id": metadata["prerequisites"]["teacher_job_id"],
        "adapted_teacher_sha256": args.checkpoint_sha256,
        "selected_step": metadata["prerequisites"]["selected_checkpoint"]["selected_step"],
        "prerequisites_sha256": metadata["prerequisites_sha256"],
        "claims_sha256": metadata["claims_sha256"],
        **metadata["provenance"],
        "protocol_sha256": resolved.protocol_sha256,
        "protocol_artifact_sha256": resolved.artifact_sha256,
        "protocol_implementation_commit": resolved.implementation_commit,
        "protocol_acceptance_commit": resolved.acceptance_commit,
        "science_implementation_sha256": resolved.science_implementation_sha256,
        "protocol_resolution": {
            key: getattr(resolved, key)
            for key in (
                "protocol_sha256",
                "artifact_sha256",
                "implementation_commit",
                "acceptance_commit",
                "head",
                "science_implementation_sha256",
            )
        },
        "protocol_accepted": True,
        "allocation": metadata["allocation"],
        "runtime": metadata["runtime"],
        "environment": metadata["environment"],
        "world_size": 4,
        "passed": False,
        "scientific_evidence_verified": False,
        "exit_code": 1,
        "formal_teacher_accepted": False,
        "accepted_science": False,
        "full_teacher_ready": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "student_training_started": False,
        "teacher_training_started": False,
        "training_started": False,
        "readiness_artifact_produced": False,
        "original_128_token_readiness_pass_claim": False,
        "stage_summaries": {},
        "checks": {key: False for key in CHECKS},
    }
    observe = observation(args.output_dir, report, rank)
    started = time.monotonic()
    try:
        observe.phase("qualification_scientific_imports", rank=rank)
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer

        from posttrain_circuits.artifacts.teacher_identity import TeacherIdentity
        from posttrain_circuits.learning.teacher.adaptation_fit import _shared_call
        from posttrain_circuits.models.adapted_teacher import load_adapted_teacher

        torch.set_num_threads(6)
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        control = dist.group.WORLD

        def select_assigned_device():
            require(
                torch.cuda.device_count() == 4 and "H100" in torch.cuda.get_device_name(local),
                "actual allocation is not four H100s",
            )
            torch.cuda.set_device(local)

        _shared_call(select_assigned_device, control, "assigned logical H100 validation")
        report["checks"].update(genuine_reviewed_science=True, all_four_ranks=True)
        tokenizer = _shared_call(
            lambda: AutoTokenizer.from_pretrained(
                args.checkpoint_root / "merged", local_files_only=True, trust_remote_code=False
            ),
            control,
            "verified tokenizer load",
        )
        observe.phase("replay_all_published_development", rank=rank)
        inputs = _shared_call(
            lambda: replay_development(
                metadata["fit"], metadata["config"], tokenizer, resolved.protocol_sha256
            ),
            control,
            "full development and selection replay",
        )
        identity = TeacherIdentity.from_mapping(inputs["selection"]["teacher_identity"])
        require(
            identity.teacher_checkpoint_sha256 == args.checkpoint_sha256
            and inputs["selection"]["selected_step"] == report["selected_step"],
            "selected identity changed",
        )
        report.update(
            selection_sha256=inputs["selection"]["sha256"],
            fit_publication_receipt_sha256=inputs["fit_publication_sha256"],
        )
        report["checks"]["verified_fit_and_selection"] = True
        _shared_call(
            lambda: (
                copy_small_inputs(args, metadata, contract),
                persist_development(args.output_dir, inputs),
            )
            if rank == 0
            else None,
            control,
            "preserve replayable qualification inputs",
        )
        observe.phase("selected_dense_teacher_load", rank=rank)
        loaded = _shared_call(
            lambda: load_adapted_teacher(
                args.checkpoint_root,
                args.checkpoint_sha256,
                metadata["config"]["teacher"],
                device=torch.device("cuda", local),
            ),
            control,
            "actual selected dense checkpoint load",
        )
        report["checks"]["verified_dense_checkpoint"] = True
        predecessor, stages = inputs["selection"]["sha256"], {}
        for role in ROLES:
            report["active_stage"] = role
            observe.phase("qualification_" + role, rank=rank)
            result = measure_stage(
                role,
                inputs=inputs,
                loaded=loaded,
                identity=identity,
                config=metadata["config"],
                output=args.output_dir,
                claims=metadata["claims"],
                job_id=args.job_id,
                predecessor=predecessor,
                contract=contract,
                observe=observe,
                rank=rank,
                control=control,
            )
            stages[role] = result
            report["stage_summaries"][role] = compact_summary(result["summary"])
            report["checks"][role] = result["summary"]["metrics_passed"]
            _shared_call(
                lambda role=role: require(report["checks"][role], role + " scientific gate failed"),
                control,
                "conditional scientific progression",
            )
            predecessor = result["evidence"]["sha256"]
        memory = _shared_call(
            lambda: sibling("sdsc_teacher_fit_memory").memory_envelope(
                evidence_path=args.output_dir / f"memory-final-{rank}.json"
            ),
            control,
            "qualification memory headroom",
        )
        ranks = [None] * 4
        dist.all_gather_object(
            ranks,
            {
                "rank": rank,
                "world_size": world,
                "logical_device": local,
                "gpu_name": torch.cuda.get_device_name(local),
                "torch_num_threads": torch.get_num_threads(),
                "cgroup_memory": memory,
            },
            group=control,
        )
        report["rank_summaries"], report["checks"]["cgroup_headroom"] = ranks, True
        _shared_call(
            lambda: finish_outputs(
                args.output_dir,
                report,
                inputs,
                stages,
                identity,
                loaded.tokenizer,
                metadata["config"],
                contract,
            )
            if rank == 0
            else None,
            control,
            "complete independent replay evidence",
        )
        observe.phase("qualification_evidence_complete", rank=rank)
        dist.destroy_process_group()
        return 0
    except BaseException as error:
        report.update(error=type(error).__name__ + ": " + str(error), failure_stage=observe.stage)
        observe.failed(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if rank == 0:
            write_json(args.output_dir / "teacher-qualify.json", report)
        observe.close()


if __name__ == "__main__":
    raise SystemExit(main())
