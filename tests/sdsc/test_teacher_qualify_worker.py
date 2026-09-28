"""CPU qualification orchestration/replay tests; no GPU or accepted-teacher claim.

The complete graph populations/verifier are real. Synthetic tokenization and
prefix measurements reuse the explicit acceptance fixtures, never model output
claimed to come from a real teacher. GPU/control calls are replaced only in the
orchestration tests; separate tests exercise actual byte and cohort validation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.artifacts import teacher_acceptance as gate
from posttrain_circuits.artifacts import teacher_adaptation_protocol
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.learning.teacher import adaptation_fit, adapted_candidate_generation
from posttrain_circuits.models import adapted_teacher

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


worker = module("_test_qualify_worker", ROOT / "tools/sdsc_teacher_qualify.py")
contract = worker.sibling("sdsc_teacher_qualify_contract")
reference = module("_qualification_scientific_fixtures", ROOT / "tests/unit/test_teacher_acceptance.py")


def raw_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"


def file_record(path, root):
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size": len(raw), "sha256": worker.digest(raw)}


@pytest.fixture(scope="module")
def scientific_data():
    return reference.data.__wrapped__()


@pytest.fixture
def context(scientific_data, monkeypatch):
    return reference.context.__wrapped__(scientific_data, monkeypatch)


@pytest.fixture
def raw_fit(context):
    config = compose_config(list(worker.sibling("sdsc_teacher_prepare").OVERRIDES))
    documents, checkpoints, previous = {}, [], reference.PUBLICATION
    for step in gate.CHECKPOINT_STEPS:
        dense = reference.dense_manifest(step)
        measured = reference.envelope(
            context, "development", reference.identity(step), previous, passed=step != 128
        )
        summary = gate.validate_readiness_evidence(
            context.specs["development"],
            measured,
            examples=context.examples["development"],
            tokenizer=context.tokenizer,
            teacher_config=context.teacher,
            identity=reference.identity(step),
            expected_binding=reference.pin(measured),
            protocol_sha256=reference.PROTOCOL,
        )
        prefix = f"artifacts/checkpoints/step-{step:06d}/"
        documents[prefix + "dense-manifest.json"] = dense
        documents[prefix + "dev-capability.json"] = {
            **summary,
            "adapted_teacher_sha256": dense["sha256"],
            "semantic_prefix_manifest_sha256": "d" * 64,
            "tokenized_prefix_manifest_sha256": "e" * 64,
            "alignment_skips": [],
        }
        documents[prefix + "eval/semantic-prefix-manifest.json"] = {"sha256": "d" * 64}
        documents[prefix + "eval/tokenized-prefix-manifest.json"] = {"sha256": "e" * 64}
        for rank in range(4):
            documents[prefix + f"eval/rank-{rank}/generations.jsonl"] = measured["generations"][rank::4]
            documents[prefix + f"eval/rank-{rank}/prefix-scores.jsonl"] = [
                row for row in measured["prefix_scores"] if row["global_index"] % 4 == rank
            ]
        checkpoints.append(
            {
                "step": step,
                "directory": f"checkpoints/step-{step:06d}",
                "dev_metrics_passed": summary["metrics_passed"],
            }
        )
        previous = measured["sha256"]
    return {
        "documents": documents,
        "checkpoint_manifest": {"checkpoints": checkpoints},
        "fit_publication_sha256": reference.PUBLICATION,
        "report": {
            "job_id": "54494742",
            "task_configuration": config["task"],
            "teacher_configuration": config["teacher"],
            "actual_plan_sha256": reference.PLAN,
            "selected_checkpoint_sha256": reference.identity(256).teacher_checkpoint_sha256,
        },
    }, config


def test_staged_tree_hashes_actual_bytes_and_rejects_races_aliases_and_extras(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "value.json"
    path.write_bytes(b"first")
    records = [file_record(path, root)]
    assert worker.verify_tree(root, records, contract, total_limit=10) == {"value.json": records[0]}
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="content changed"):
        worker.verify_tree(root, records, contract, total_limit=10)
    path.write_bytes(b"first")
    (root / "extra").write_bytes(b"")
    with pytest.raises(ValueError, match="population"):
        worker.verify_tree(root, records, contract, total_limit=10)
    (root / "extra").unlink()
    os.link(path, tmp_path / "alias")
    with pytest.raises(ValueError, match="regular"):
        worker.verify_tree(root, records, contract, total_limit=10)
    (tmp_path / "alias").unlink()
    path.unlink()
    path.symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="unsafe"):
        worker.verify_tree(root, records, contract, total_limit=10)


def test_staged_tree_fifo_and_oversize_fail_without_open_block(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="nonregular"):
        worker.verify_tree(tmp_path, [], contract, total_limit=1)
    (tmp_path / "fifo").unlink()
    (tmp_path / "x").write_bytes(b"xx")
    with pytest.raises(ValueError, match="byte limit"):
        worker.verify_tree(tmp_path, [file_record(tmp_path / "x", tmp_path)], contract, total_limit=1)


def test_complete_four_checkpoint_raw_replay_retains_every_row_and_first_pass(raw_fit, context):
    fit, config = raw_fit
    result = worker.replay_development(fit, config, context.tokenizer, reference.PROTOCOL)
    assert result["selection"]["selected_step"] == 256
    assert [row["summary"]["metrics_passed"] for row in result["development"]] == [False, True, True, True]
    for checkpoint in result["checkpoints"]:
        prefix = f"artifacts/checkpoints/step-{checkpoint['step']:06d}/eval/"
        assert checkpoint["evidence"]["generations"] == [
            row for rank in range(4) for row in fit["documents"][prefix + f"rank-{rank}/generations.jsonl"]
        ]
        assert checkpoint["evidence"]["claim_id"] == f"fit:54494742:development:{checkpoint['step']}"
    assert result["specs"]["supplemental"]["source_indices"] == list(range(128, 256))


@pytest.mark.parametrize("fault", ["summary", "configuration", "selection", "generation"])
def test_development_cannot_trust_old_pass_or_change_source(raw_fit, context, fault):
    fit, config = raw_fit
    prefix = "artifacts/checkpoints/step-000256/"
    if fault == "summary":
        fit["documents"][prefix + "dev-capability.json"]["metrics"]["answer_accuracy"] = 0.0
    elif fault == "configuration":
        fit["report"]["teacher_configuration"] = {**config["teacher"], "hidden_override": True}
    elif fault == "selection":
        fit["report"]["selected_checkpoint_sha256"] = reference.identity(384).teacher_checkpoint_sha256
    else:
        fit["documents"][prefix + "eval/rank-2/generations.jsonl"].pop()
    with pytest.raises(ValueError):
        worker.replay_development(fit, config, context.tokenizer, reference.PROTOCOL)


def test_ledger_and_accepted_view_use_actual_verification_and_explicit_dense_identity(context, tmp_path):
    evidence = reference.envelope(context, "train_probe", reference.identity(), "f" * 64)
    # One entire prompt fails; preserve all attempts instead of dropping failures.
    failed = reference.envelope(context, "train_probe", reference.identity(), "f" * 64, passed=False)
    evidence["attempts"][:8] = failed["attempts"][:8]
    reference.reseal(evidence)
    summary = reference.replay(context, evidence)
    assert summary["covered_prompts"] == 31 and not summary["metrics_passed"]
    manifest = worker.write_teacher_store(
        tmp_path, evidence, summary, reference.identity(), reference.PROTOCOL
    )
    rows = [
        json.loads(line) for line in (tmp_path / "teacher-store/attempts.jsonl").read_bytes().splitlines()
    ]
    view = json.loads((tmp_path / "teacher-store/accepted-view.json").read_bytes())
    assert rows == evidence["attempts"] and len(rows) == 256
    assert view["attempt_ids"] == summary["accepted_attempt_ids"] and len(view["attempt_ids"]) == 248
    assert manifest["formal_teacher_accepted"] is view["formal_teacher_accepted"] is False
    assert all("teacher_id" not in row and "teacher_revision" not in row for row in rows)
    assert all(row["teacher_identity_sha256"] == reference.identity().sha256 for row in rows)


@pytest.mark.parametrize("passed", [False, True])
def test_claim_precedes_actual_producer_and_all_rank_rows_are_replayed(
    context, tmp_path, monkeypatch, passed
):
    calls = []
    role = "train_probe"
    identity = reference.identity()
    measured = reference.envelope(context, role, identity, "f" * 64, passed=passed)
    index = {example.example_id: i for i, example in enumerate(context.examples[role])}
    shards = [
        [row for row in measured["attempts"] if index[row["prompt_id"]] % 4 == rank] for rank in range(4)
    ]

    def claim(_claims, actual_role, predecessor, *, job_id):
        calls.append("claim")
        assert actual_role == role and predecessor == "f" * 64 and job_id == "123"
        return {"claimed_before_inference": True}

    def produce(*args, **kwargs):
        assert calls == ["claim"]
        calls.append("produce")
        assert kwargs["rank"] == 0 and kwargs["world_size"] == 4
        return shards[0]

    def seal(_claims, actual_role, evidence_path, summary, *, job_id):
        assert calls == ["claim", "produce"]
        calls.append("seal")
        assert summary["metrics_passed"] is passed
        assert json.loads(evidence_path.read_bytes())["attempts"] == measured["attempts"]
        return {"metrics_passed": passed}

    monkeypatch.setattr(adaptation_fit, "_shared_call", lambda fn, _group, _phase: fn())
    monkeypatch.setattr(adapted_candidate_generation, "measure_candidate_shard", produce)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, **_kw: output.__setitem__(slice(None), shards),
    )
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *_a, **_kw: None)
    result = worker.measure_stage(
        role,
        inputs={"specs": context.specs, "examples_by_role": context.examples},
        loaded=SimpleNamespace(tokenizer=context.tokenizer),
        identity=identity,
        config={"teacher": context.teacher},
        output=tmp_path,
        claims={"stage_claim_ids": {role: measured["claim_id"]}},
        job_id="123",
        predecessor="f" * 64,
        contract=SimpleNamespace(claim_stage=claim, seal_stage=seal),
        observe=None,
        rank=0,
        control=None,
    )
    assert calls == ["claim", "produce", "seal"]
    assert result["summary"]["metrics_passed"] is passed
    assert (tmp_path / "stages/train_probe/summary.json").is_file()


def test_failed_claim_never_enters_producer(context, tmp_path, monkeypatch):
    monkeypatch.setattr(adaptation_fit, "_shared_call", lambda fn, _group, _phase: fn())
    monkeypatch.setattr(
        adapted_candidate_generation,
        "measure_candidate_shard",
        lambda *_a, **_kw: pytest.fail("inference after claimed exposure"),
    )

    def claimed(*_args, **_kwargs):
        raise ValueError("already claimed")

    with pytest.raises(ValueError, match="already claimed"):
        worker.measure_stage(
            "train_probe",
            inputs={"specs": context.specs, "examples_by_role": context.examples},
            loaded=None,
            identity=None,
            config={"teacher": context.teacher},
            output=tmp_path,
            claims={},
            job_id="123",
            predecessor="f" * 64,
            contract=SimpleNamespace(claim_stage=claimed),
            observe=None,
            rank=0,
            control=None,
        )
    assert not (tmp_path / "stages").exists()


def test_progress_never_claims_teacher_or_student_training(tmp_path):
    observe = worker.observation(
        tmp_path, {"task": contract.TASK, "job_id": "123", "run_id": "test", "code_sha256": "a" * 64}, 0
    )
    try:
        observe.phase("qualification_supplemental")
        value = json.loads((tmp_path / "progress.json").read_bytes())
        assert value["artifact_kind"] == "teacher_qualification_progress"
        assert not any(
            value[name]
            for name in (
                "training_started",
                "teacher_training_started",
                "student_training_started",
                "formal_teacher_accepted",
            )
        )
    finally:
        observe.close()


def test_cli_help_and_import_do_not_import_numerical_libraries():
    script = (
        "import runpy,sys; runpy.run_path('tools/sdsc_teacher_qualify.py',run_name='qualification_import');"
        "assert not any(x in sys.modules for x in ('torch','transformers','numpy')); print('stdlib-only')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert result.stdout.strip() == "stdlib-only"
    help_result = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_teacher_qualify.py"), "--help"],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert "--fit-evidence-root" in help_result.stdout and "--checkpoint-sha256" in help_result.stdout


def test_validate_only_exits_before_rank_resolution_or_numerical_imports():
    # Metadata/Git/host edges are exercised separately; this isolates the CLI's
    # validate-only dispatch, in a fresh interpreter without test-suite Torch.
    script = """
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('qualification','tools/sdsc_teacher_qualify.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
sys.path.insert(0,str(Path('src').absolute()))
import posttrain_circuits.core.config
import posttrain_circuits.artifacts.teacher_adaptation_protocol
worker.metadata_preflight=lambda args,contract: {}
names=('science-root','work-dir','output-dir','hf-home','provenance-manifest','checkpoint-root','prerequisites','claims','fit-evidence-root','run-id','code-sha256','job-id','provenance-manifest-sha256','checkpoint-sha256')
assert worker.main([part for name in names for part in ('--'+name,'/tmp/test-only')]+['--validate-only'])==0
assert not any(name in sys.modules for name in ('torch','transformers','numpy'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "validated": True,
        "protocol_accepted": True,
        "formal_teacher_accepted": False,
    }


@pytest.fixture
def metadata_inputs(tmp_path, monkeypatch):
    """Real args/claims/hashes; host and Git acceptance are explicit test stubs."""
    work = tmp_path / "work"
    work.mkdir()
    args = SimpleNamespace(
        science_root=ROOT,
        work_dir=work,
        output_dir=work / "artifacts",
        hf_home=work / "hf",
        provenance_manifest=work / "provenance.json",
        checkpoint_root=work / "selected-checkpoint",
        prerequisites=work / "prerequisites.json",
        claims=work / "claims.json",
        fit_evidence_root=work / "fit-evidence",
        run_id="qualification-fixture",
        code_sha256="b" * 64,
        job_id="12346",
        provenance_manifest_sha256="a" * 64,
        checkpoint_sha256=reference.identity(256).teacher_checkpoint_sha256,
    )
    for path in (args.checkpoint_root, args.fit_evidence_root, args.hf_home, args.output_dir):
        path.mkdir()
    (args.checkpoint_root / "dense-manifest.json").write_bytes(raw_json(reference.dense_manifest(256)))
    protocol = dict(
        protocol_sha256=reference.PROTOCOL,
        artifact_sha256="d" * 64,
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        head="3" * 40,
        science_file_sha256={"reviewed-file": "f" * 64},
        science_implementation_sha256="e" * 64,
        review_status="accepted",
        payload={"test": True},
    )
    selected = {
        "adapted_teacher_sha256": args.checkpoint_sha256,
        "selected_step": 256,
        "files": [file_record(args.checkpoint_root / "dense-manifest.json", args.checkpoint_root)],
    }
    proof = dict(
        schema="quest-sdsc-teacher-qualification-prerequisites-v1",
        task=contract.TASK,
        target=dict(run_id=args.run_id, code_sha256=args.code_sha256, intent_id="a" * 32),
        teacher_job_id="12345",
        protocol=protocol,
        selected_checkpoint=selected,
    )
    args.prerequisites.write_bytes(raw_json(proof))
    binding = dict(
        **proof["target"],
        teacher_job_id="12345",
        adapted_teacher_sha256=args.checkpoint_sha256,
        protocol_sha256=reference.PROTOCOL,
        prerequisites_sha256=worker.digest(args.prerequisites.read_bytes()),
        supplemental_namespace=contract.CLAIM_NAMESPACE,
        validation_file_sha256=contract.VALIDATION_SHA256,
        supplemental_order_sha256=contract.SUPPLEMENTAL_ORDER_SHA256,
    )
    claims_root = tmp_path / "teacher-qualification-claims"
    monkeypatch.setattr(contract, "CLAIMS_ROOT", claims_root)
    claims = dict(
        schema="quest-sdsc-teacher-qualification-reservations-v1",
        binding=binding,
        reservations=contract.reserve_claims(tmp_path, binding),
        stage_root=str(claims_root / "stages" / binding["intent_id"]),
        stage_claim_ids={role: binding["intent_id"] + ":" + role for role in worker.ROLES},
    )
    args.claims.write_bytes(raw_json(claims))
    provenance = dict(
        science_git_head=protocol["head"],
        bundle_sha256="a" * 64,
        provenance_manifest_sha256=args.provenance_manifest_sha256,
    )
    guards, fit = worker.sibling("sdsc_teacher_prepare"), worker.sibling("sdsc_teacher_fit")
    fit_contract = worker.sibling("sdsc_teacher_fit_contract")
    monkeypatch.setattr(
        guards,
        "validate_checkout",
        lambda actual: provenance if actual is args else pytest.fail("wrong arguments"),
    )
    monkeypatch.setattr(fit.importlib.metadata, "version", lambda name: fit_contract.DEPENDENCIES[name])
    monkeypatch.setattr(
        fit.platform, "python_version", lambda: fit_contract.FIXED_EXECUTION["runtime"]["python"]
    )
    monkeypatch.setattr(
        fit, "file_hash", lambda _path: fit_contract.FIXED_EXECUTION["runtime"]["python_sha256"]
    )
    host_calls = []

    def host_environment(actual, actual_guards):
        assert actual is args and actual_guards is guards
        assert args.work_dir in args.output_dir.parents and args.work_dir in args.hf_home.parents
        host_calls.append("host")
        return {"node_local_mount": "test-only", "cgroup_memory": {"passed": True}}

    monkeypatch.setattr(fit, "sibling", lambda name: SimpleNamespace(local_environment=host_environment))
    original_sibling = worker.sibling
    modules = {
        "sdsc_teacher_prepare": guards,
        "sdsc_teacher_fit": fit,
        "sdsc_teacher_fit_contract": fit_contract,
        "sdsc_teacher_qualify_contract": contract,
    }
    monkeypatch.setattr(
        worker, "sibling", lambda name: modules[name] if name in modules else original_sibling(name)
    )
    monkeypatch.setattr(
        teacher_adaptation_protocol,
        "resolve_teacher_adaptation_protocol",
        lambda root, *, expected_head: SimpleNamespace(**protocol, tracked_worktree_clean=True)
        if root == ROOT and expected_head == protocol["head"]
        else pytest.fail("wrong resolver identity"),
    )
    monkeypatch.setattr(
        worker,
        "load_bound_fit_evidence",
        lambda root, prerequisite, actual_contract: {"fixture": True}
        if root == args.fit_evidence_root and prerequisite == proof and actual_contract is contract
        else pytest.fail("wrong fit evidence arguments"),
    )
    for name, value in dict(
        SLURM_JOB_ID=args.job_id,
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="196608",
        SLURM_JOB_CPUS_PER_NODE="72",
        CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d",
    ).items():
        monkeypatch.setenv(name, value)
    return args, proof, claims, host_calls


def test_metadata_uses_real_allocation_guard_actual_claims_and_config(metadata_inputs):
    args, proof, claims, calls = metadata_inputs
    value = worker.metadata_preflight(args, contract)
    assert calls == ["host"]
    assert value["allocation"]["allocated_cpus_on_node"] == 72
    assert value["allocation"]["cuda_visible_devices"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert value["prerequisites"] == proof and value["claims"] == claims
    assert value["config"]["teacher"]["prompt_protocol"]["enable_thinking"] is False


@pytest.mark.parametrize("fault", ["job", "allocation", "claim", "source", "bytes", "outside"])
def test_metadata_failure_prevents_fit_read_or_any_inference(metadata_inputs, monkeypatch, fault):
    args, proof, claims, _calls = metadata_inputs
    if fault == "job":
        monkeypatch.setenv("SLURM_JOB_ID", "999")
    elif fault == "allocation":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    elif fault == "claim":
        directory = Path(claims["stage_root"])
        directory.mkdir(parents=True)
        (directory / "train_probe.json").write_bytes(b"{}")
    elif fault == "source":
        monkeypatch.setattr(
            teacher_adaptation_protocol,
            "resolve_teacher_adaptation_protocol",
            lambda *_a, **_kw: SimpleNamespace(
                **{**proof["protocol"], "review_status": "proposed"}, tracked_worktree_clean=True
            ),
        )
    elif fault == "bytes":
        (args.checkpoint_root / "dense-manifest.json").write_bytes(b"{}")
    else:
        args.checkpoint_root = args.work_dir.parent / "outside"
    monkeypatch.setattr(
        worker, "load_bound_fit_evidence", lambda *_a, **_kw: pytest.fail("fit read after failed authority")
    )
    with pytest.raises(ValueError):
        worker.metadata_preflight(args, contract)


@pytest.mark.parametrize(
    "failed_stage", [None, "train_probe", "supplemental", "formal_readiness", "formal_store"]
)
def test_main_wires_four_gpu_arguments_and_stops_immediately_at_failed_gate(
    metadata_inputs, monkeypatch, failed_stage
):
    args, _proof, _claims, _calls = metadata_inputs
    metadata = worker.metadata_preflight(args, contract)
    monkeypatch.setattr(
        worker,
        "metadata_preflight",
        lambda actual, actual_contract: metadata
        if vars(actual) == {**vars(args), "validate_only": False} and actual_contract is contract
        else pytest.fail("CLI argument mismatch"),
    )
    calls = []
    identity = reference.identity(256)
    inputs = {
        "selection": {"sha256": "9" * 64, "selected_step": 256, "teacher_identity": identity.to_mapping()},
        "fit_publication_sha256": reference.PUBLICATION,
    }

    def replay(*_args):
        calls.append("replay")
        return inputs

    def load(root, sha, config, *, device):
        assert calls == ["replay", "preserve"]
        assert root == args.checkpoint_root and sha == args.checkpoint_sha256
        assert config == metadata["config"]["teacher"] and str(device) == "cuda:0"
        calls.append("dense-load")
        return SimpleNamespace(tokenizer="tokenizer")

    def measure(role, **kwargs):
        assert kwargs["rank"] == 0 and kwargs["job_id"] == args.job_id and kwargs["identity"] == identity
        assert calls[-1] == (
            "dense-load" if role == "train_probe" else worker.ROLES[worker.ROLES.index(role) - 1]
        )
        calls.append(role)
        return {"summary": {"metrics_passed": role != failed_stage}, "evidence": {"sha256": "8" * 64}}

    def finish(output, report, *_args):
        assert calls[-4:] == list(worker.ROLES) and output == args.output_dir
        assert all(report["checks"].values())
        calls.append("finish")
        report.update(passed=True, scientific_evidence_verified=True, exit_code=0)

    monkeypatch.setattr(worker, "replay_development", replay)
    monkeypatch.setattr(worker, "copy_small_inputs", lambda *_a: calls.append("preserve"))
    monkeypatch.setattr(worker, "persist_development", lambda *_a: None)
    monkeypatch.setattr(worker, "measure_stage", measure)
    monkeypatch.setattr(worker, "finish_outputs", finish)
    monkeypatch.setattr(adapted_teacher, "load_adapted_teacher", load)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_kw: "tokenizer")
    monkeypatch.setattr(adaptation_fit, "_shared_call", lambda fn, _group, _phase: fn())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _rank: "NVIDIA H100 test fixture")
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda rank: None if rank == 0 else pytest.fail("physical device")
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, **_kw: None if backend == "gloo" else pytest.fail("non-Gloo"),
    )
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, **_kw: output.__setitem__(
            slice(None), [{**value, "rank": rank, "logical_device": rank} for rank in range(4)]
        ),
    )
    original_sibling = worker.sibling
    monkeypatch.setattr(
        worker,
        "sibling",
        lambda name: SimpleNamespace(memory_envelope=lambda **_kw: {"passed": True})
        if name == "sdsc_teacher_fit_memory"
        else original_sibling(name),
    )
    for name, value in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="4").items():
        monkeypatch.setenv(name, value)
    argv = [part for key, value in vars(args).items() for part in ("--" + key.replace("_", "-"), str(value))]
    if failed_stage is None:
        assert worker.main(argv) == 0
    else:
        with pytest.raises(ValueError, match="scientific gate failed"):
            worker.main(argv)
        assert calls[-1] == failed_stage and "finish" not in calls
    report = json.loads((args.output_dir / "teacher-qualify.json").read_bytes())
    assert report["passed"] is (failed_stage is None)
    assert report["protocol_resolution"]["head"] == metadata["resolved"].head
    assert all(
        report[name] is False
        for name in (
            "formal_teacher_accepted",
            "training_started",
            "teacher_training_started",
            "student_training_started",
            "accepted_science",
            "full_teacher_ready",
        )
    )


@pytest.fixture
def bound_fit_files(tmp_path, monkeypatch):
    """Publication-binding fixture, not a substitute for fit scientific validation."""
    root = tmp_path / "fit"
    root.mkdir()
    documents, checkpoint_rows = {}, []
    actual_fit_contract = worker.sibling("sdsc_teacher_fit_contract")
    origin_paths = (*actual_fit_contract.KERNEL_PATHS, *contract.FIT_SCIENCE_ORIGIN_PATHS)
    release = {"files": [{"path": path, "sha256": "e" * 64} for path in origin_paths]}
    execution = actual_fit_contract.execution_plan(release)
    report = dict(
        job_id="12345",
        run_id="fit-test",
        code_sha256="b" * 64,
        adaptation_dataset_sha256=gate.FULL_FIT_DATASET_SHA256,
        execution_plan=execution,
        execution_plan_sha256=contract.sha256_value(execution),
    )
    for step in (128, 256, 384, 512):
        prefix = f"artifacts/checkpoints/step-{step:06d}/"
        provenance = {
            **{key: report[key] for key in ("job_id", "run_id", "code_sha256")},
            "dataset_manifest_sha256": report["adaptation_dataset_sha256"],
            "optimizer_steps": step,
            "consumed_tokens": step * 100,
            "train_metrics_sha256": "a" * 64,
        }
        dense = gate.sealed({"training_provenance": provenance})
        state = gate.sealed({"metadata": {"completed_updates": step, "consumed_tokens": step * 100}})
        documents[prefix + "dense-manifest.json"] = dense
        documents[prefix + "manifest.json"] = state
        documents[prefix + "dev-capability.json"] = {"test_only": True}
        for rank in range(4):
            for name in ("generations", "prefix-scores"):
                documents[prefix + f"eval/rank-{rank}/{name}.jsonl"] = [{"test_only": rank}]
        for name in ("semantic", "tokenized"):
            documents[prefix + f"eval/{name}-prefix-manifest.json"] = {"test_only": True}
        checkpoint_rows.append(
            dict(
                step=step,
                directory=f"checkpoints/step-{step:06d}",
                adapted_teacher_sha256=dense["sha256"],
                consumed_tokens=step * 100,
                training_metrics_sha256="a" * 64,
            )
        )
    documents.update(
        {
            "artifacts/checkpoint-selection.json": {},
            "artifacts/data-isolation.json": {},
            "artifacts/train-metrics.jsonl": [{"test_only": True}],
        }
    )
    for name, value in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"".join(raw_json(row) for row in value) if name.endswith(".jsonl") else raw_json(value)
        )
    for row in checkpoint_rows:
        prefix = "artifacts/" + row["directory"] + "/"
        for name, key in (
            ("dense-manifest.json", "dense_manifest_sha256"),
            ("manifest.json", "trainer_state_manifest_sha256"),
            ("dev-capability.json", "dev_metrics_sha256"),
        ):
            row[key] = worker.digest((root / (prefix + name)).read_bytes())
    manifest = gate.sealed(
        {
            "checkpoints": checkpoint_rows,
            "files": [
                file_record(path, root / "artifacts") for path in sorted(root.rglob("*")) if path.is_file()
            ],
        }
    )
    (root / "artifacts/checkpoint-manifest.json").write_bytes(raw_json(manifest))
    report.update(
        checkpoint_set_sha256=manifest["sha256"], checkpoint_manifest_sha256=worker.digest(raw_json(manifest))
    )
    (root / "teacher-fit.json").write_bytes(raw_json(report))
    publication = {"files": [file_record(path, root) for path in sorted(root.rglob("*")) if path.is_file()]}
    receipt_path = tmp_path / "fit-receipt.json"
    receipt_path.write_bytes(raw_json(publication))
    upstream = dict(
        report=report,
        receipt={key: report[key] for key in ("job_id", "run_id", "code_sha256")},
        status={"success": True, "state": "COMPLETED"},
        release_manifest=release,
        result_dir=str(root),
        publication_receipt=publication,
        publication_receipt_path=str(receipt_path),
        publication_receipt_sha256=worker.digest(raw_json(publication)),
    )
    prerequisites = {
        "upstream": upstream,
        "teacher_job_id": report["job_id"],
        "fit_evidence": contract.fit_evidence(upstream),
        "protocol": {"science_file_sha256": {path: "e" * 64 for path in origin_paths}},
    }
    fit_contract = SimpleNamespace(
        validate_report=lambda actual, _plan, task: worker.require(
            actual == report and task == "qwen3-v2-teacher-fit", "wrong fit validator call"
        ),
    )
    original_sibling = worker.sibling
    monkeypatch.setattr(
        worker,
        "sibling",
        lambda name: fit_contract if name == "sdsc_teacher_fit_contract" else original_sibling(name),
    )
    return root, prerequisites, receipt_path


def test_raw_fit_load_binds_actual_publication_and_offline_receipt(bound_fit_files, tmp_path):
    root, prerequisite, receipt = bound_fit_files
    local_receipt = tmp_path / "offline-receipt.json"
    local_receipt.write_bytes(receipt.read_bytes())
    receipt.unlink()
    value = worker.load_bound_fit_evidence(
        root, prerequisite, contract, publication_receipt_path=local_receipt
    )
    assert len(value["checkpoint_manifest"]["checkpoints"]) == 4
    assert value["publication_receipt_bytes"] == local_receipt.read_bytes()
    local_receipt.write_bytes(b"{}")
    with pytest.raises(ValueError, match="receipt changed"):
        worker.load_bound_fit_evidence(root, prerequisite, contract, publication_receipt_path=local_receipt)


def test_fit_evidence_change_between_initial_hash_and_actual_parse_is_rejected(bound_fit_files, monkeypatch):
    root, prerequisite, _receipt = bound_fit_files
    original = worker.verify_tree

    def changed_after_hash(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "artifacts/data-isolation.json").write_bytes(b'{"changed":true}')
        return result

    monkeypatch.setattr(worker, "verify_tree", changed_after_hash)
    with pytest.raises(ValueError, match="actual parse"):
        worker.load_bound_fit_evidence(root, prerequisite, contract)


@pytest.mark.parametrize("changed", [True, False])
def test_fit_origin_requires_every_current_reviewed_kernel(bound_fit_files, changed):
    root, prerequisite, _receipt = bound_fit_files
    if changed:
        prerequisite["protocol"]["science_file_sha256"][
            "src/posttrain_circuits/datasets/proofgraph/generation.py"
        ] = "f" * 64
    else:
        prerequisite["protocol"]["science_file_sha256"].clear()
    with pytest.raises(ValueError, match="reviewed qualification science"):
        worker.load_bound_fit_evidence(root, prerequisite, contract)


def test_complete_output_replays_all_gates_and_retains_formal_false(context, tmp_path):
    full = reference.complete_inputs(context)
    stages = {}
    for role in worker.ROLES:
        evidence = full["stages"][role]
        validator = (
            gate.validate_candidate_evidence
            if role in gate.CANDIDATE_ROLES
            else gate.validate_readiness_evidence
        )
        summary = validator(
            context.specs[role],
            evidence,
            examples=context.examples[role],
            tokenizer=context.tokenizer,
            teacher_config=context.teacher,
            identity=reference.identity(256),
            expected_binding=reference.pin(evidence),
            protocol_sha256=reference.PROTOCOL,
        )
        stages[role] = {"evidence": evidence, "summary": summary, "binding": reference.pin(evidence)}
    report = {
        "protocol_sha256": reference.PROTOCOL,
        "job_id": "123",
        "run_id": "test",
        "code_sha256": "a" * 64,
    }
    inputs = {
        key: full[key]
        for key in (
            "checkpoints",
            "expected_bindings",
            "examples_by_role",
            "specs",
            "fit_publication_sha256",
            "fit_plan_sha256",
        )
    }
    worker.finish_outputs(
        tmp_path,
        report,
        inputs,
        stages,
        reference.identity(256),
        context.tokenizer,
        {"teacher": context.teacher},
        contract,
    )
    acceptance = json.loads((tmp_path / "acceptance-evidence.json").read_bytes())
    manifest = json.loads((tmp_path / "qualification-manifest.json").read_bytes())
    assert report["passed"] and report["scientific_evidence_verified"]
    assert acceptance["scientific_evidence_verified"] and not acceptance["formal_teacher_accepted"]
    assert report["acceptance_evidence_sha256"] == acceptance["sha256"]
    assert report["scientific_inventory_bytes"] == sum(row["size"] for row in manifest["files"])
    assert len((tmp_path / "teacher-store/attempts.jsonl").read_bytes().splitlines()) == 2048
    for row in manifest["files"]:
        assert file_record(tmp_path / row["path"], tmp_path) == row
