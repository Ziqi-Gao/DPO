"""Actual-file audit boundary tests; no GPU, jobs or model-quality claims."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location("audit_test_" + name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("sdsc_teacher_qualification_audit")
contract = module("sdsc_teacher_qualify_contract")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = audit.canonical(value) + b"\n"
    path.write_bytes(raw)
    return raw


def published(tmp_path, files=None):
    root = tmp_path / "published"
    root.mkdir()
    content = {"teacher-qualify.json": b'{"passed":true}\n', **(files or {})}
    records = []
    for name, raw in content.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        records.append({"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    receipt = {
        "task": audit.TASK,
        "job_id": "12345",
        "run_id": "fixture-run",
        "code_sha256": "a" * 64,
        "passed": True,
        "persisted": True,
        "persistent_read_back_verified": True,
        "files": records,
    }
    raw = put(root / "receipt.json", receipt)
    return root, receipt, hashlib.sha256(raw).hexdigest()


def test_publication_audit_reads_actual_bytes_and_does_not_modify_them(tmp_path):
    root, receipt, expected = published(
        tmp_path, {"artifacts/stages/raw.jsonl": b'{"raw":1}\n', "worker.log": b"log\n"}
    )
    before = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    checked = audit.publication(root, expected, "12345")
    assert checked["receipt"] == receipt
    assert checked["raw"]["artifacts/stages/raw.jsonl"] == b'{"raw":1}\n'
    assert checked["scientific_bytes"] == len(before["teacher-qualify.json"]) + len(
        before["artifacts/stages/raw.jsonl"]
    )
    assert checked["log_bytes"] == 4
    assert before == {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "fault",
    ["changed", "missing", "extra", "symlink", "duplicate", "badsize", "weight", "failed", "wronghash"],
)
def test_publication_cannot_trust_a_rehashed_pass_flag_or_unbound_files(tmp_path, fault):
    root, receipt, expected = published(tmp_path, {"artifacts/stages/raw.json": b'{"measure":0}\n'})
    item = root / "artifacts/stages/raw.json"
    if fault == "changed":
        item.write_bytes(b'{"measure":1}\n')
    elif fault == "missing":
        item.unlink()
    elif fault == "extra":
        (root / "not-in-receipt.json").write_text("{}")
    elif fault == "symlink":
        item.unlink()
        item.symlink_to(root / "teacher-qualify.json")
    elif fault == "duplicate":
        receipt["files"].append(receipt["files"][0])
    elif fault == "badsize":
        receipt["files"][0]["size"] = audit.MAX_SCIENCE_BYTES + 1
    elif fault == "weight":
        receipt["files"].append({"path": "artifacts/model.safetensors", "size": 0, "sha256": "a" * 64})
    elif fault == "failed":
        receipt["passed"] = False
    else:
        expected = "b" * 64
    if fault in {"duplicate", "badsize", "weight", "failed"}:
        expected = hashlib.sha256(put(root / "receipt.json", receipt)).hexdigest()
    with pytest.raises((ValueError, OSError)):
        audit.publication(root, expected, "12345")


def test_strict_raw_parsing_rejects_duplicate_nan_and_partial_lines():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            audit.parse(raw)
    for raw in (b'{"x":1}', b'{"x":1}\n\n'):
        with pytest.raises(ValueError):
            audit.jsonlines(raw)


def stage_raw(role="supplemental"):
    rows = [
        {"global_index": 128 + rank, "actual_sampling_seed": 170 + rank, "response": str(rank)}
        for rank in range(4)
    ]
    scores = [
        {"global_index": rank, "side": side, "score": -float(rank + 1)}
        for rank in range(4)
        for side in (0, 1)
    ]
    raw = {}
    for rank in range(4):
        prefix = f"artifacts/stages/{role}/raw/rank-{rank}/"
        raw[prefix + "generations.jsonl"] = audit.canonical(rows[rank]) + b"\n"
        raw[prefix + "prefix-scores.jsonl"] = b"".join(
            audit.canonical(row) + b"\n" for row in scores[2 * rank : 2 * rank + 2]
        )
    return {"raw": raw}, {"generations": rows, "prefix_scores": scores}


def test_stage_sealed_evidence_must_equal_all_actual_rank_rows():
    publication, evidence = stage_raw()
    assert audit.verify_stage_raw(publication, "supplemental", evidence)["world_size"] == 4
    altered = copy.deepcopy(evidence)
    altered["generations"][0]["response"] = "corrected proof"
    with pytest.raises(ValueError, match="sealed generations"):
        audit.verify_stage_raw(publication, "supplemental", altered)


@pytest.mark.parametrize("fault", ["missing_rank", "unknown_raw", "wrong_rank", "wrong_prefix_side"])
def test_raw_stage_cannot_drop_or_move_rank_records(fault):
    publication, evidence = stage_raw()
    if fault == "missing_rank":
        del publication["raw"]["artifacts/stages/supplemental/raw/rank-3/generations.jsonl"]
    elif fault == "unknown_raw":
        publication["raw"]["artifacts/stages/supplemental/raw/extra.json"] = b"{}"
    elif fault == "wrong_rank":
        left = "artifacts/stages/supplemental/raw/rank-0/generations.jsonl"
        right = "artifacts/stages/supplemental/raw/rank-1/generations.jsonl"
        publication["raw"][left], publication["raw"][right] = (
            publication["raw"][right],
            publication["raw"][left],
        )
    else:
        evidence["prefix_scores"][0]["side"] = 1
    with pytest.raises(ValueError):
        audit.verify_stage_raw(publication, "supplemental", evidence)


def claim_fixture(tmp_path):
    project = tmp_path / "project"
    root = project / "teacher-qualification-claims"
    binding = {
        "intent_id": "1" * 32,
        "run_id": "fixture-run",
        "code_sha256": "a" * 64,
        "adapted_teacher_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "prerequisites_sha256": "d" * 64,
        "supplemental_namespace": contract.CLAIM_NAMESPACE,
        "validation_file_sha256": contract.VALIDATION_SHA256,
        "supplemental_order_sha256": contract.SUPPLEMENTAL_ORDER_SHA256,
    }
    claims = {
        "schema": "quest-sdsc-teacher-qualification-reservations-v1",
        "binding": binding,
        "reservations": contract.reserve_claims(project, binding),
        "stage_root": str(root / "stages" / binding["intent_id"]),
        "stage_claim_ids": {role: binding["intent_id"] + ":" + role for role in audit.ROLES},
    }
    previous, stages, summaries, raw = "e" * 64, {}, {}, {}
    for role in audit.ROLES:
        stage_root = Path(claims["stage_root"])
        claim = {
            **binding,
            "schema": "quest-sdsc-teacher-qualification-stage-claim-v1",
            "job_id": "12345",
            "role": role,
            "claim_id": claims["stage_claim_ids"][role],
            "predecessor_sha256": previous,
            "claimed_before_inference": True,
        }
        claim_raw = put(stage_root / (role + ".claim.json"), claim)
        payload = {"role": role, "claim_id": claim["claim_id"], "predecessor_sha256": previous}
        evidence = {**payload, "sha256": audit.sha(payload)}
        evidence_raw = audit.canonical(evidence) + b"\n"
        summary = {"metrics_passed": True, "fixed_measured_rows": 128}
        outcome = {
            "schema": "quest-sdsc-teacher-qualification-stage-outcome-v1",
            "job_id": "12345",
            "role": role,
            "claim_id": claim["claim_id"],
            "claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
            "predecessor_sha256": previous,
            "evidence_sha256": evidence["sha256"],
            "evidence_file_sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "evidence_file_size": len(evidence_raw),
            "summary_sha256": audit.sha(summary),
            "metrics_passed": True,
        }
        put(stage_root / (role + ".outcome.json"), outcome)
        for name, value in (("claim", claim), ("outcome", outcome), ("evidence", evidence)):
            raw[f"artifacts/stages/{role}/{name}.json"] = audit.canonical(value) + b"\n"
        stages[role], summaries[role], previous = evidence, summary, evidence["sha256"]
    published = {"receipt": {"job_id": "12345", "run_id": "fixture-run", "code_sha256": "a" * 64}, "raw": raw}
    return published, claims, stages, summaries, root


def test_claim_audit_reads_permanent_namespace_and_every_stage_outcome(tmp_path):
    published, claims, stages, summaries, root = claim_fixture(tmp_path)
    checked = audit.verify_claim_chain(
        published, claims, stages, summaries, "e" * 64, contract, claims_root=root
    )
    assert len(checked) == 10
    assert str(root / "experiments" / (contract.CLAIM_NAMESPACE + ".json")) in checked


@pytest.mark.parametrize(
    "fault", ["copied_claim_root", "wrong_previous", "self_passed", "different_job", "changed_permanent"]
)
def test_claim_audit_rejects_published_self_assertions_and_wrong_history(tmp_path, fault):
    published, claims, stages, summaries, root = claim_fixture(tmp_path)
    path = Path(claims["stage_root"]) / "supplemental.outcome.json"
    if fault == "copied_claim_root":
        root = tmp_path / "different-claims"
    elif fault == "wrong_previous":
        stages["supplemental"]["predecessor_sha256"] = "f" * 64
    elif fault == "self_passed":
        summaries["supplemental"]["metrics_passed"] = False
    elif fault == "different_job":
        published["receipt"]["job_id"] = "99999"
    else:
        outcome = json.loads(path.read_text())
        outcome["evidence_sha256"] = "f" * 64
        put(path, outcome)
    with pytest.raises(ValueError):
        audit.verify_claim_chain(published, claims, stages, summaries, "e" * 64, contract, claims_root=root)


def test_audit_metadata_is_create_once_and_read_back_verified(tmp_path):
    path = tmp_path / "audit.json"
    value = {"formal_teacher_accepted": False, "raw_evidence_verified": True}
    audit._write_once(path, value)
    assert json.loads(path.read_text()) == value
    with pytest.raises(FileExistsError):
        audit._write_once(path, value)


def test_physical_and_unique_scientific_bounds_are_independent(tmp_path, monkeypatch):
    root, _receipt, digest = published(
        tmp_path, {"artifacts/stages/a.json": b"x" * 100, "artifacts/stages/b.json": b"x" * 100}
    )
    monkeypatch.setattr(audit, "MAX_UNIQUE_SCIENCE_BYTES", 130)
    monkeypatch.setattr(audit, "MAX_SCIENCE_BYTES", 250)
    result = audit.publication(root, digest, "12345")
    assert result["scientific_bytes"] == 216
    assert result["unique_scientific_bytes"] == 116
    monkeypatch.setattr(audit, "MAX_SCIENCE_BYTES", 215)
    with pytest.raises(ValueError, match="physical bound"):
        audit.publication(root, digest, "12345")
    monkeypatch.setattr(audit, "MAX_SCIENCE_BYTES", 250)
    monkeypatch.setattr(audit, "MAX_UNIQUE_SCIENCE_BYTES", 115)
    with pytest.raises(ValueError, match="unique-content bound"):
        audit.publication(root, digest, "12345")


def test_candidate_rank_ownership_cannot_be_hidden_by_sorted_envelope():
    rows = [dict(prompt_id=f"prompt-{rank}", attempt_id=f"candidate-{rank}") for rank in range(4)]
    evidence = {"attempts": rows}
    raw = {
        f"artifacts/stages/train_probe/raw/rank-{rank}/attempts.jsonl": audit.canonical(row) + b"\n"
        for rank, row in enumerate(rows)
    }
    assert audit.verify_stage_raw({"raw": raw}, "train_probe", evidence)["world_size"] == 4
    left, right = list(raw)[:2]
    raw[left], raw[right] = raw[right], raw[left]
    with pytest.raises(ValueError, match="candidate belongs to another rank"):
        audit.verify_stage_raw({"raw": raw}, "train_probe", evidence)


def test_one_rank_raw_cannot_satisfy_four_rank_qualification():
    publication, evidence = stage_raw()
    publication["raw"] = {key: value for key, value in publication["raw"].items() if "/rank-0/" in key}
    with pytest.raises(ValueError, match="rank inventory"):
        audit.verify_stage_raw(publication, "supplemental", evidence)


def test_complete_cpu_audit_replays_all_cohorts_and_rejects_rehashed_raw_tampering(tmp_path, monkeypatch):
    """Real domain/cohorts/verifier/claims/files; synthetic tokenizer and scores.

    Genuine Git provenance, upstream staged-fit byte loading and tokenizer
    loading have dedicated real-boundary tests. Only those three external
    inputs are fixtures here. No scientific reducer or completion gate is
    mocked, and no fixture output is represented as a GPU measurement.
    """
    from transformers import AutoTokenizer

    from posttrain_circuits.artifacts import teacher_acceptance as gate
    from posttrain_circuits.artifacts import teacher_adaptation_protocol as protocol_module

    def load_test(name):
        spec = importlib.util.spec_from_file_location(
            "audit_fixture_" + name, Path(__file__).with_name(name + ".py")
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    fixtures = load_test("test_teacher_qualify_worker")
    boundary = load_test("test_teacher_qualify_boundary")
    reference, worker = fixtures.reference, fixtures.worker
    ctx = reference.context.__wrapped__(reference.data.__wrapped__(), monkeypatch)
    fit, config = fixtures.raw_fit.__wrapped__(ctx)
    inputs = worker.replay_development(fit, config, ctx.tokenizer, reference.PROTOCOL)
    identity = reference.identity(256)
    root, project = tmp_path / "published", tmp_path / "project"
    output, claims_root = root / "artifacts", project / "teacher-qualification-claims"
    output.mkdir(parents=True)
    for name, value in fit["documents"].items():
        path = output / "fit-evidence" / name
        if name.endswith(".jsonl"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(audit.canonical(row) + b"\n" for row in value))
        else:
            put(path, value)
    report, prerequisite = boundary.report_fixture()
    protocol_binding = {
        **prerequisite["protocol"],
        "protocol_sha256": reference.PROTOCOL,
    }
    resolved = SimpleNamespace(
        **protocol_binding,
        payload={"fixture": "explicit synthetic accepted source"},
        science_file_sha256={
            "tools/sdsc_teacher_qualification_audit.py": hashlib.sha256(
                Path(audit.__file__).read_bytes()
            ).hexdigest()
        },
        review_status="accepted",
        tracked_worktree_clean=True,
    )
    prerequisite.update(
        protocol={
            **protocol_binding,
            "payload": resolved.payload,
            "science_file_sha256": resolved.science_file_sha256,
        },
        teacher_job_id=fit["report"]["job_id"],
        upstream={"publication_receipt_sha256": fit["fit_publication_sha256"]},
    )
    tokenizer_raw = put(output / "tokenizer/tokenizer.json", {"fixture": "not a real tokenizer"})
    prerequisite["selected_checkpoint"] = {
        "adapted_teacher_sha256": identity.teacher_checkpoint_sha256,
        "selected_step": 256,
        "files": [
            {
                "path": "merged/tokenizer.json",
                "size": len(tokenizer_raw),
                "sha256": hashlib.sha256(tokenizer_raw).hexdigest(),
            }
        ],
    }
    prerequisite_raw = put(output / "inputs/prerequisites.json", prerequisite)
    binding = {
        **prerequisite["target"],
        "adapted_teacher_sha256": identity.teacher_checkpoint_sha256,
        "protocol_sha256": reference.PROTOCOL,
        "prerequisites_sha256": hashlib.sha256(prerequisite_raw).hexdigest(),
        "supplemental_namespace": contract.CLAIM_NAMESPACE,
        "validation_file_sha256": contract.VALIDATION_SHA256,
        "supplemental_order_sha256": contract.SUPPLEMENTAL_ORDER_SHA256,
    }
    claims = {
        "schema": "quest-sdsc-teacher-qualification-reservations-v1",
        "binding": binding,
        "reservations": contract.reserve_claims(project, binding),
        "stage_root": str(claims_root / "stages" / binding["intent_id"]),
        "stage_claim_ids": {role: binding["intent_id"] + ":" + role for role in audit.ROLES},
    }
    claims_raw = put(output / "inputs/claims.json", claims)
    put(
        output / "inputs/fit-publication-receipt.json", {"fixture": "upstream byte-loader covered separately"}
    )
    monkeypatch.setattr(contract, "CLAIMS_ROOT", claims_root)
    worker.persist_development(output, inputs)
    stages, previous = {}, inputs["selection"]["sha256"]
    for role in audit.ROLES:
        directory = output / "stages" / role
        claim = contract.claim_stage(claims, role, previous, job_id=report["job_id"])
        put(directory / "claim.json", claim)
        put(directory / "spec.json", inputs["specs"][role])
        evidence = reference.envelope(ctx, role, identity, previous)
        evidence["claim_id"] = claims["stage_claim_ids"][role]
        names = ("attempts",) if role in gate.CANDIDATE_ROLES else ("generations", "prefix_scores")
        for name in names:
            all_rows, by_rank = evidence[name], []
            prompt_order = {example.example_id: index for index, example in enumerate(ctx.examples[role])}
            for rank in range(4):
                rows = [
                    row
                    for row in all_rows
                    if (prompt_order[row["prompt_id"]] if name == "attempts" else row["global_index"]) % 4
                    == rank
                ]
                path = directory / "raw" / f"rank-{rank}" / (name.replace("_", "-") + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"".join(audit.canonical(row) + b"\n" for row in rows))
                by_rank.extend(rows)
            if name != "attempts":
                evidence[name] = by_rank
        reference.reseal(evidence)
        function = (
            gate.validate_candidate_evidence
            if role in gate.CANDIDATE_ROLES
            else gate.validate_readiness_evidence
        )
        summary = function(
            inputs["specs"][role],
            evidence,
            examples=ctx.examples[role],
            tokenizer=ctx.tokenizer,
            teacher_config=ctx.teacher,
            identity=identity,
            expected_binding=reference.pin(evidence),
            protocol_sha256=reference.PROTOCOL,
        )
        put(directory / "evidence.json", evidence)
        put(directory / "summary.json", summary)
        outcome = contract.seal_stage(
            claims, role, directory / "evidence.json", summary, job_id=report["job_id"]
        )
        put(directory / "outcome.json", outcome)
        stages[role] = {"evidence": evidence, "summary": summary, "binding": reference.pin(evidence)}
        previous = evidence["sha256"]
    report.update(
        teacher_job_id=fit["report"]["job_id"],
        adapted_teacher_sha256=identity.teacher_checkpoint_sha256,
        selected_step=256,
        protocol_resolution=protocol_binding,
        protocol_sha256=reference.PROTOCOL,
        fit_publication_receipt_sha256=fit["fit_publication_sha256"],
        prerequisites_sha256=hashlib.sha256(prerequisite_raw).hexdigest(),
        claims_sha256=hashlib.sha256(claims_raw).hexdigest(),
        selection_sha256=inputs["selection"]["sha256"],
        stage_summaries={role: worker.compact_summary(value["summary"]) for role, value in stages.items()},
    )
    worker.finish_outputs(output, report, inputs, stages, identity, ctx.tokenizer, config, contract)
    put(root / "teacher-qualify.json", report)

    def seal_publication():
        records = [
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        ]
        value = {key: report[key] for key in ("task", "job_id", "run_id", "code_sha256")}
        raw = put(
            root / "receipt.json",
            {
                **value,
                "passed": True,
                "persisted": True,
                "persistent_read_back_verified": True,
                "files": records,
            },
        )
        return hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(
        protocol_module, "resolve_teacher_adaptation_protocol", lambda *_args, **_kwargs: resolved
    )
    monkeypatch.setattr(worker, "load_bound_fit_evidence", lambda *_args, **_kwargs: fit)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: ctx.tokenizer)
    original_sibling = audit._sibling
    monkeypatch.setattr(
        audit,
        "_sibling",
        lambda root, name: {"sdsc_teacher_qualify_contract": contract, "sdsc_teacher_qualify": worker}.get(
            name
        )
        or original_sibling(root, name),
    )
    original_replay, replay_calls, replay_results = gate.build_teacher_acceptance, [], []

    def complete_replay(**kwargs):
        replay_calls.append(([value["step"] for value in kwargs["checkpoints"]], list(kwargs["stages"])))
        value = original_replay(**kwargs)
        replay_results.append(value)
        return value

    monkeypatch.setattr(gate, "build_teacher_acceptance", complete_replay)
    args = dict(
        science_root=ROOT,
        expected_head=resolved.head,
        result_root=root,
        job_id=report["job_id"],
        claims_root=claims_root,
    )
    result = audit.audit_result(**args, publication_sha256=seal_publication())
    assert result["passed"] is True and result["formal_teacher_accepted"] is False
    assert result["accounting_checked_by_this_program"] is False
    assert result["selection"]["selected_step"] == 256
    assert result["stage_summaries"]["formal_store"]["attempts"] == 2048
    assert result["stage_summaries"]["formal_store"]["covered_prompts"] == 256
    assert replay_calls == [([128, 256, 384, 512], list(audit.ROLES))]
    assert result["recomputed_acceptance_evidence"] is replay_results[0]
    assert result["recomputed_acceptance_evidence"]["sha256"] == result["acceptance_evidence_sha256"]
    assert result["recomputed_acceptance_evidence"]["formal_teacher_accepted"] is False
    assert len(audit.canonical(result)) + 1 <= audit.MAX_AUDIT_BYTES
    print("full_fixture_audit_bytes=" + str(len(audit.canonical(result)) + 1))
    raw_path = output / "stages/formal_readiness/raw/rank-0/generations.jsonl"
    rows = audit.jsonlines(raw_path.read_bytes())
    rows[0]["response_text"] = "post-hoc repaired output"
    raw_path.write_bytes(b"".join(audit.canonical(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="sealed generations differ"):
        audit.audit_result(**args, publication_sha256=seal_publication())
