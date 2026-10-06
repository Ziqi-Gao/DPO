"""Focus-worker regressions using real CPU Qwen/AdamW/Accelerate paths."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_focus_worker", ROOT / "tools/sdsc_student_name_invariant_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(count=128):
    from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord

    result = []
    for index in range(count):
        response = [5 + index % 7] * (1 + index % 3) + [3]
        row = TrajectoryRecord(
            trajectory_id="",
            prompt_id=f"example-{index}",
            split="cpu_fixture",
            prompt_text="fixture",
            input_ids=[2, 4] + [8] * (index % 2),
            response_ids=response,
            response_text="fixture",
            response_token_mask=[True] * len(response),
            behavior_policy_id="symbolic-canonical-proof",
            behavior_policy_revision="v1",
            policy_version=0,
            sampling_request_seed=0,
            actual_sampling_seed=0,
            sampling_cursor_id=str(index),
            sampling_protocol_id="deterministic-canonical-no-sampling-v1",
            sampling_temperature=0.0,
            top_p=1.0,
            behavior_logprobs=[float("nan")] * len(response),
            verifier_reward=1.0,
            verification_trace={"reward": 1.0},
        )
        row.trajectory_id = row.expected_trajectory_id
        row.validate()
        result.append(row)
    return result


@pytest.mark.parametrize(
    "schema,mode,extra",
    [
        ("quest-sdsc-student-branch-inputs-v3", "preflight", {}),
        ("quest-sdsc-student-order-inputs-v4", "fit", {}),
        ("quest-sdsc-student-batch-probe-inputs-v1", "probe", {"arm": "control"}),
        ("quest-sdsc-student-name-invariant-inputs-v1", "probe", {}),
        ("quest-sdsc-student-name-invariant-inputs-v1", "fit", {"arm": "treatment"}),
        ("quest-sdsc-student-name-invariant-inputs-v1", "fit", {"learning_rate": 1e-4}),
    ],
)
def test_previous_input_schema_is_rejected_before_any_science_load(tmp_path, schema, mode, extra):
    m = worker()
    # Matching resource shapes cannot relabel historical preparation as focus V5.
    inputs = dict.fromkeys(
        (
            "schema",
            "mode",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "original_science_files",
            "dataset_root",
            "hf_home",
            "protocol",
            "original_config",
            "initial_checkpoint",
        )
    )
    inputs.update(schema=schema, mode=mode, **extra)
    with pytest.raises(ValueError, match="preparation input identity"):
        m.staged_science(inputs, tmp_path / "output")


def test_source_is_symbolic_one_pass_and_rejects_changed_resume():
    from posttrain_circuits.learning.contracts import PromptBatch

    m = worker()
    source = m.CanonicalSource(records(3))
    source.get_batch(None, PromptBatch(["example-1"], ["fixture"]), 0)
    assert source.state_dict()["kind"] == "symbolic_canonical"
    assert source.records["example-1"].teacher_id is None
    snapshot = source.state_dict()
    source.load_state_dict(snapshot)
    with pytest.raises(ValueError, match="repeated"):
        source.get_batch(None, PromptBatch(["example-1"], ["fixture"]), 0)
    for bad in (
        {**snapshot, "identity_sha256": "0" * 64},
        {**snapshot, "cursor": {"example-0": True}},
        {**snapshot, "cursor": {"missing": 1}},
        {**snapshot, "kind": "teacher_demo"},
    ):
        with pytest.raises(ValueError):
            source.load_state_dict(bad)


def test_real_canonical_encoding_has_no_teacher_and_masks_only_response():
    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    tokenizer = build_tiny_tokenizer()
    views = p.fit_views(p.make_examples(p.DIFFICULTY)["student_fit"][:2])
    encoded = [p.encode_view(view, tokenizer, {}) for view in views]
    converted = m.canonical_records(views, encoded, tokenizer, {})
    assert len(converted) == 16 and len({row.prompt_id for row in converted}) == 16
    assert {view.view for view in views} == set(p.FIT_VIEW_NAMES)
    row = converted[0]
    assert row.teacher_id is None and row.teacher_topk_ids == []
    assert row.verifier_reward == 1.0 and row.response_ids[-1] == tokenizer.eos_token_id
    assert row.input_ids + row.response_ids == encoded[0]["input_ids"]
    batch = VerifiedReplaySupervisor(0, retry_limit=0).prepare_targets(
        TrajectoryBatch(converted, 0), None, None
    )
    for index, record in enumerate(converted):
        prefix, end = len(record.input_ids), len(record.input_ids) + len(record.response_ids)
        assert not batch.response_mask[index, :prefix].any()
        assert batch.response_mask[index, prefix:end].all()
        assert not batch.response_mask[index, end:].any()


def test_real_tokenizer_binds_all_2048_actual_views_and_negative_rotation():
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    spec = importlib.util.spec_from_file_location(
        "_focus_canonical_audit", ROOT / "tools/sdsc_student_name_invariant_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    tokenizer = audit.load_tokenizer()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    bases = p.make_examples(p.DIFFICULTY)["student_fit"]
    assert {example.label for example in bases} == {0, 1}
    views = p.fit_views(bases)
    encoded = [p.encode_view(view, tokenizer, config, training=True) for view in views]
    m = worker()
    converted = m.canonical_records(views, encoded, tokenizer, config)
    assert len(converted) == len({row.prompt_id for row in converted}) == 2048
    assert sum(len(row["input_ids"]) for row in encoded) == 1_662_052
    for offset, base_example in enumerate(bases):
        names = p.FIT_VIEW_NAMES if base_example.label else (*p.FIT_VIEW_NAMES[1:], p.FIT_VIEW_NAMES[0])
        assert tuple(view.view for view in views[offset * 8 : (offset + 1) * 8]) == tuple(names)
    for view, row, record in zip(views, encoded, converted, strict=True):
        prompt = format_model_prompt(view.prompt, tokenizer, config)
        assert record.raw_prompt_text == view.prompt == prompt.raw_prompt
        assert record.prompt_text == prompt.model_facing_prompt
        assert record.raw_prompt_sha256 == prompt.raw_prompt_sha256
        assert record.model_facing_prompt_sha256 == prompt.model_facing_prompt_sha256
        assert tokenizer.encode(record.prompt_text, add_special_tokens=False) == record.input_ids
        assert record.input_ids + record.response_ids == row["input_ids"]
        assert record.verifier_reward == 1.0 and record.response_ids[-1] == tokenizer.eos_token_id
        if view.view == "surface_template_paraphrase":
            assert view.prompt != ProofGraphTask().render(view.example)
    # Exercise the observed V3 coverage gap at the actual worker conversion seam:
    # pure-order prompts keep the original atoms and target IDs, while their
    # rendered line order must survive encoding and collation unchanged.
    grouped = {}
    for view, record in zip(views, converted, strict=True):
        grouped.setdefault(view.source_example_id, {})[view.view] = (view, record)
    for group in grouped.values():
        identity, identity_record = group["identity"]
        for name in ("fact_order_permutation", "rule_order_permutation"):
            changed, changed_record = group[name]
            assert changed.example.facts == identity.example.facts
            assert changed.example.rules == identity.example.rules
            assert changed_record.response_ids == identity_record.response_ids
            assert changed_record.raw_prompt_text == changed.prompt
            assert (changed_record.input_ids != identity_record.input_ids) is (
                changed.prompt != identity.prompt
            )
    modified_views = list(views)
    modified_views[0] = p.PreparedView(
        views[0].source_example_id, views[0].view, views[0].example, views[0].prompt + "\nextra"
    )
    with pytest.raises(ValueError, match="prompt text/token prefix differ"):
        m.canonical_records(modified_views, encoded, tokenizer, config)
    corrupted = copy.deepcopy(encoded)
    corrupted[0]["input_ids"][0] ^= 1
    with pytest.raises(ValueError, match="prompt text/token prefix differ"):
        m.canonical_records(views, corrupted, tokenizer, config)
    corrupted = copy.deepcopy(encoded)
    corrupted[0]["input_ids"][-1] ^= 1
    with pytest.raises(ValueError, match="response/token IDs differ"):
        m.canonical_records(views, corrupted, tokenizer, config)


def test_saved_dense_is_real_reload_and_never_formal_initial(tmp_path):
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    model = build_tiny_qwen3(7)
    state = model.state_dict()
    path = tmp_path / "step.pt"
    m.save_dense(
        state,
        path,
        step=4,
        inputs={"mode": "preflight", "protocol": {"sha256": "a" * 64}},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
    )
    saved = torch.load(path, weights_only=False)
    reloaded = build_tiny_qwen3(8)
    assert m.util.strict_load_export(reloaded, saved["model"])["all_tensors_exact"]
    assert m.tree_hash(reloaded.state_dict()) == m.tree_hash(state)
    assert not saved["student_accepted"] and not saved["formal_initial_accepted"]
    assert saved["format"] == "student_name_invariant_preparation_dense_v1"
    assert saved["scope"] == "student_name_invariant_preparation_model_only"
    with pytest.raises(ValueError, match="already exists"):
        m.save_dense(
            state,
            path,
            step=4,
            inputs={"mode": "preflight", "protocol": {"sha256": "a" * 64}},
            dataset_sha="b" * 64,
            protocol_sha="c" * 64,
        )


def test_step_observer_detects_extra_or_stale_optimizer_cadence():
    m = worker()
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    observer = m.StepObserver(optimizer)
    parameter.sum().backward()
    optimizer.step()
    assert observer.calls == 1
    optimizer.state[parameter]["step"].zero_()
    parameter.sum().backward()
    with pytest.raises(ValueError, match="cadence"):
        optimizer.step()
    observer.close()


def test_failure_preserves_partial_raw_without_claiming_success(tmp_path, monkeypatch):
    m = worker()
    path, output = tmp_path / "inputs.json", tmp_path / "output"
    path.write_text("{}")
    monkeypatch.setenv("RANK", "0")
    assert m.main(["--inputs-json", str(path), "--output-dir", str(output)]) == 1
    report = json.loads((output / "prepare-report.json").read_text())
    assert not report["passed"] and not report["student_accepted"] and not report["g0_passed"]
    assert "input identity" in report["error"]


def test_genuine_native_bf16_long_context_preflight_has_no_development_data():
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    model = build_tiny_qwen3(12).to(dtype=torch.bfloat16).eval()
    model.config.max_position_embeddings = 2456
    before = m.tree_hash(model.state_dict())
    actual = m.inference_envelope_probe(model, enabled=True)
    assert actual["passed"] and actual["input_length"] == 2456 and actual["all_logits_finite"]
    assert m.tree_hash(model.state_dict()) == before
    assert m.inference_envelope_probe(model, enabled=False) is None
    model.config.max_position_embeddings = 2244
    with pytest.raises(ValueError, match="context shorter"):
        m.inference_envelope_probe(model, enabled=True)
    model.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="native BF16"):
        m.inference_envelope_probe(model, enabled=True)


def test_development_aggregation_rejects_missing_transform_and_structure_rows():
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS

    m = worker()
    good = dict(num_examples=128, answer_correct=50, proof_correct=45, format_valid=120)
    structure = dict(
        chain=dict(num_examples=50, answer_correct=20, proof_correct=18, format_valid=45),
        branch=dict(num_examples=40, answer_correct=15, proof_correct=13, format_valid=40),
        converging_dag=dict(num_examples=38, answer_correct=15, proof_correct=14, format_valid=35),
    )
    locals_ = [
        dict(
            rank=rank,
            counts={key: value * 6 for key, value in good.items()},
            per_view_counts={name: copy.deepcopy(good) for name in ("identity", *TRANSFORMATIONS)},
            structure_counts=copy.deepcopy(structure),
        )
        for rank in range(2)
    ]
    result = m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)
    assert result["num_examples"] == 256 and result["proof_correct"] == 90
    assert result["structure_counts"]["branch"]["num_examples"] == 80
    locals_[0]["per_view_counts"][TRANSFORMATIONS[0]]["num_examples"] -= 1
    with pytest.raises(ValueError, match="population incomplete"):
        m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)
    locals_[0]["per_view_counts"][TRANSFORMATIONS[0]]["num_examples"] += 1
    locals_[0]["structure_counts"]["branch"]["num_examples"] -= 1
    locals_[0]["structure_counts"]["branch"]["format_valid"] -= 1
    with pytest.raises(ValueError, match="structure totals"):
        m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)


def test_complete_development_base_shards_balance_every_view_and_keep_seed_identity():
    from collections import Counter

    m = worker()
    shards = [m.generation_ordinals(1536, rank, evaluate=True) for rank in (0, 1)]
    assert len(shards[0]) == len(shards[1]) == 768
    assert not set(shards[0]) & set(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(range(1536))
    for rank, shard in enumerate(shards):
        assert shard == sorted(shard)
        assert Counter(index % 6 for index in shard) == {view: 128 for view in range(6)}
        assert all((ordinal // 6) % 2 == rank for ordinal in shard)
    assert m.generation_ordinals(8, 0, evaluate=False) == [0, 2, 4, 6]
    assert m.generation_ordinals(8, 1, evaluate=False) == [1, 3, 5, 7]
    with pytest.raises(ValueError, match="population"):
        m.generation_ordinals(1535, 0, evaluate=True)


def test_fit_early_checkpoints_do_not_broaden_training_only_preflight():
    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p

    m = worker()
    assert m.checkpoint_steps("preflight") == (4,)
    assert m.checkpoint_steps("fit") == p.CHECKPOINT_STEPS == (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    settings = m.specification()
    assert settings["training"]["optimizer_steps"] == 32
    assert settings["training"]["input_token_budget"] == 2_000_000
    assert settings["data"]["fit_examples"] == 2048
    assert settings["data"]["fit_base_examples"] == 256
    assert len(settings["data"]["fit_views"]) == 8
    with pytest.raises(ValueError, match="mode"):
        m.checkpoint_steps("old_fit")


@pytest.mark.parametrize("mode,eligible", [("preflight", False), ("fit", False), ("fit", True)])
def test_execute_preserves_mode_schedule_restore_and_full_failure_evidence(
    tmp_path, monkeypatch, mode, eligible
):
    """Real orchestration/reporting; CUDA, loading and optimization are explicit spies."""
    from types import SimpleNamespace

    import accelerate

    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p

    m = worker()
    bases = p._build_bases()
    config = {"model": {}, "task": p.DIFFICULTY}
    settings = m.specification()
    spec = importlib.util.spec_from_file_location(
        "_focus_transport_seam", ROOT / "tests/sdsc/test_student_name_invariant_transport.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    controller = fixtures.c.__wrapped__(monkeypatch)
    plan = fixtures.plan.__wrapped__(controller, tmp_path, monkeypatch)
    plan["mode"] = mode
    fixtures.bind(controller, plan)
    hardware_fixture = fixtures.report(controller, plan)
    dataset = p.dataset_manifest(bases)
    binding = SimpleNamespace(
        payload=settings,
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        artifact_sha256=plan["protocol"]["artifact_sha256"],
        implementation_commit="c" * 40,
    )
    monkeypatch.setattr(
        m,
        "load_data",
        lambda *args: (
            config,
            tmp_path / "native.pt",
            bases,
            dataset,
            hardware_fixture["data_audit"]["isolation"],
            binding,
        ),
    )
    monkeypatch.setattr(torch, "__version__", "2.8.0+cu128")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda rank: "H100")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(
        m.importlib.metadata,
        "version",
        lambda name: {
            "transformers": "4.56.2",
            "accelerate": "1.10.1",
            "tokenizers": "0.22.0",
        }[name],
    )
    monkeypatch.setattr(
        accelerate, "Accelerator", lambda **kwargs: SimpleNamespace(num_processes=2, mixed_precision="bf16")
    )
    for key, value in dict(
        RANK="0",
        WORLD_SIZE="2",
        LOCAL_RANK="0",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        monkeypatch.setenv(key, value)
    loaded = SimpleNamespace(tokenizer=SimpleNamespace())
    monkeypatch.setattr(
        m.util, "load_training_initial", lambda *args: (loaded, hardware_fixture["initial_loading"])
    )
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())

    def gather(value):
        other = copy.deepcopy(value)
        if "rank" in other:
            other["rank"] = 1
        return [value, other]

    monkeypatch.setattr(m.util, "gather", gather)
    monkeypatch.setattr(p, "encode_view", lambda *args, **kwargs: {"input_ids": [1, 2], "prefix_length": 1})
    monkeypatch.setattr(
        p,
        "validate_encoded_fit",
        lambda rows: dict(
            fit_rows=len(rows),
            optimizer_windows=32,
            window_input_tokens=[128] * 32,
            total_input_tokens=4096,
            token_budget=2_000_000,
        ),
    )
    monkeypatch.setattr(
        m,
        "canonical_records",
        lambda views, *args: [SimpleNamespace(prompt_id=x.example.example_id) for x in views],
    )
    trainer = SimpleNamespace(_terminate=False, global_step=0, token_budget=SimpleNamespace(consumed=0))
    seen = dict(updates=[], checkpoints=[], restores=[], max_steps=[])

    def build(*args, max_steps, **kwargs):
        seen["max_steps"].append(max_steps)
        return trainer, object(), hardware_fixture["fsdp"]

    def train(_trainer, ids, rows, rank, step, *, observer):
        assert len(ids) == len(rows) == 2048 and rank == 0
        seen["updates"].append(step)
        trainer.token_budget.consumed += 128
        trainer.global_step = step
        return {"step": step, "fixture_only": True}

    def restore(_trainer, output, rank):
        seen["restores"].append(seen["updates"][-1])
        return hardware_fixture["checkpoint_restore"]["ranks"][0]

    def checkpoint(_trainer, model_config, tokenizer, views, encoded, **kwargs):
        step, evaluate = kwargs["step"], kwargs["evaluate"]
        seen["checkpoints"].append((step, evaluate, len(views)))
        digest = f"{step:064x}"
        summary = None
        if evaluate:
            count = dict(num_examples=256, answer_correct=0, proof_correct=0, format_valid=0)
            structure = {
                name: dict(
                    count, num_examples=sum(x.metadata["structure"] == name for x in bases["student_dev"])
                )
                for name in p.STRUCTURES
            }
            if eligible:
                for counts in structure.values():
                    counts.update(
                        answer_correct=counts["num_examples"] // 3,
                        proof_correct=counts["num_examples"] // 3,
                        format_valid=counts["num_examples"],
                    )
                correct = sum(counts["proof_correct"] for counts in structure.values())
                count.update(answer_correct=correct, proof_correct=correct, format_valid=256)
            summary = dict(
                step=step,
                checkpoint_sha256=digest,
                examples_sha256=p.EXAMPLES_SHA256["student_dev"],
                **count,
                per_view_counts={name: copy.deepcopy(count) for name in p.DEV_VIEW_NAMES},
                structure_counts=structure,
            )
        exported = copy.deepcopy(hardware_fixture["checkpoints"][0])
        exported.update(step=step, sha256=digest, path=f"checkpoints/step-{step:08d}.pt")
        return exported, summary

    monkeypatch.setattr(m, "build_trainer", build)
    monkeypatch.setattr(m, "train_window", train)
    monkeypatch.setattr(m, "full_state_restore", restore)
    monkeypatch.setattr(m, "checkpoint_and_development", checkpoint)
    monkeypatch.setattr(m, "StepObserver", lambda optimizer: SimpleNamespace(close=lambda: None))
    inputs = dict(
        mode=mode,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=controller.sha(controller.canonical(plan)),
    )
    report = m.execute(inputs, tmp_path, tmp_path)
    expected_steps = 4 if mode == "preflight" else 32
    schedule = [4] if mode == "preflight" else list(m.STEPS)
    assert seen["updates"] == list(range(1, expected_steps + 1))
    assert seen["max_steps"] == [expected_steps] and seen["restores"] == [4]
    assert seen["checkpoints"] == [(step, mode == "fit", 1536 if mode == "fit" else 8) for step in schedule]
    assert report["optimizer_steps"] == expected_steps and report["execution_complete"] is True
    assert report["source_kind"] == "symbolic_canonical_name_invariant_preparation"
    assert report["consumed_fit_training_views"] == expected_steps * 64
    assert report["training_input_tokens"] == expected_steps * 128
    assert report["preparation_complete"] is (mode == "fit")
    assert "arm" not in report and "diagnostic_complete" not in report
    assert all(report[name] is False for name in m.FLAGS)
    assert report["passed"] is (mode == "preflight" or eligible)
    assert (report["selected_checkpoint"] is not None) is eligible
    if eligible:
        assert report["selected_checkpoint"]["step"] == 1
    assert len(report["development"]) == (12 if mode == "fit" else 0)
    assert report["fit_base_examples"] == 256 and report["fit_training_views"] == 2048
    saved = json.loads((tmp_path / "prepare-report.json").read_text())
    assert (
        saved == report
        and len((tmp_path / "prepare-updates.jsonl").read_text().splitlines()) == expected_steps
    )
    assert (tmp_path / "prepare-dev-prompts.jsonl").exists() is (mode == "fit")
    # Actual execute-produced identity/populations/selection reach the real controller.
    # The explicitly spied GPU methods above supply hardware evidence; these file
    # placeholders only exercise report inventory, not raw replay or GPU acceptance.
    for name in controller.required_raw_names(mode):
        path = tmp_path / name
        if not path.exists():
            path.write_bytes(b"{}\n")
    report["raw_artifacts"] = m.raw_artifacts(tmp_path)
    inventory = {row["path"]: row for row in report["raw_artifacts"]}
    if report["passed"]:
        controller.validate_worker_report(report, plan, "123", inventory)
        corrupted = copy.deepcopy(report)
        corrupted["source_kind"] = "symbolic_canonical_anchor_probe"
        with pytest.raises(ValueError, match="source must"):
            controller.validate_worker_report(corrupted, plan, "123", inventory)
    else:
        with pytest.raises(ValueError, match="incomplete/overclaiming"):
            controller.validate_worker_report(report, plan, "123", inventory)


@pytest.mark.parametrize("evaluate", [False, True])
def test_actual_dense_reload_and_raw_generation_paths(tmp_path, monkeypatch, capsys, evaluate):
    """Real tiny-Qwen forward/export; explicit mock boundaries are CUDA/FSDP/generation."""
    from contextlib import nullcontext
    from types import MethodType, SimpleNamespace

    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p
    from posttrain_circuits.models import loading
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    torch.set_num_threads(1)
    tokenizer = build_tiny_tokenizer()
    bases = p._build_bases()
    views = p.development_views(bases["student_dev"]) if evaluate else p.fit_views(bases["student_fit"])[:8]
    encoded = [p.encode_view(view, tokenizer, {}, training=not evaluate) for view in views]
    original = exact_shape_tiny_qwen(71)
    masters = copy.deepcopy(original.state_dict())
    original.config.max_position_embeddings = 2456
    original.to(dtype=torch.bfloat16).eval()
    trainer = SimpleNamespace(model=original, _full_model_state_for_checkpoint=lambda: copy.deepcopy(masters))

    def loader(config, for_training=False):
        del config, for_training
        model = exact_shape_tiny_qwen(72)
        model.config.max_position_embeddings = 2456

        def generate(self, *, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 256 and not kwargs["do_sample"] and not kwargs["use_cache"]
            return torch.cat((input_ids, torch.full((1, 1), tokenizer.eos_token_id, dtype=torch.long)), dim=1)

        model.generate = MethodType(generate, model)
        return SimpleNamespace(model=model, tokenizer=tokenizer)

    monkeypatch.setattr(loading, "load_model_and_tokenizer", loader)
    monkeypatch.setattr(m.util, "measurement", lambda *args: nullcontext())
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())

    def gather(actual):
        other = copy.deepcopy(actual)
        other["rank"] = 1
        other["counts"] = m.empty_counts()
        other["per_view_counts"] = {name: m.empty_counts() for name in p.DEV_VIEW_NAMES}
        other["structure_counts"] = {name: m.empty_counts() for name in p.STRUCTURES}
        for ordinal in m.generation_ordinals(len(views), 1, evaluate=evaluate):
            other["counts"]["num_examples"] += 1
            if evaluate:
                view = views[ordinal]
                other["per_view_counts"][view.view]["num_examples"] += 1
                if view.view == "identity":
                    other["structure_counts"][view.example.metadata["structure"]]["num_examples"] += 1
        return [actual, other]

    monkeypatch.setattr(m.util, "gather", gather)
    checkpoint, summary = m.checkpoint_and_development(
        trainer,
        {},
        tokenizer,
        views,
        encoded,
        output=tmp_path,
        rank=0,
        step=4,
        inputs={"mode": "fit" if evaluate else "preflight", "protocol": {"sha256": "a" * 64}},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
        dev_sha=p.EXAMPLES_SHA256["student_dev"],
        evaluate=evaluate,
    )
    assert checkpoint["reload_by_rank"][0]["exact_saved_master_reload"]["all_tensors_exact"]
    assert checkpoint["reload_by_rank"][0]["exact_saved_master_reload"]["key_count"] == 311
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    if evaluate:
        assert progress and {row["stage"] for row in progress} == {"name_invariant_development"}
    else:
        assert progress == []  # Four preflight records never reach the 32-row progress cadence.
    assert (
        checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["scope"]
        == "synthetic_name_invariant_context_no_development_content"
    )
    assert all(row["bitwise_equal"] for row in checkpoint["reload_by_rank"][0]["root_export_logits"])
    path = tmp_path / (
        "prepare-dev-records-step-00000004-rank-0.jsonl"
        if evaluate
        else "prepare-preflight-records-rank-0.jsonl"
    )
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(raw) == (768 if evaluate else 4)
    assert [row["ordinal"] for row in raw] == m.generation_ordinals(len(views), 0, evaluate=evaluate)
    assert all(row["view"] == views[row["ordinal"]].view for row in raw)
    assert all(row["source_example_id"] == views[row["ordinal"]].source_example_id for row in raw)
    assert all(row["checkpoint_sha256"] == checkpoint["sha256"] for row in raw)
    if evaluate:
        assert summary["num_examples"] == 256 and summary["answer_correct"] == 0
        assert all(row["num_examples"] == 256 for row in summary["per_view_counts"].values())
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["input_length"] == 2456
    else:
        assert summary is None
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["input_length"] == 2456


@pytest.mark.parametrize("evaluate", [False, True])
def test_pinned_tokenizer_actual_worker_records_replay_independently(tmp_path, monkeypatch, evaluate):
    """Real full-prefix raw seam; tiny model and collectives are explicit CPU mocks."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p
    from posttrain_circuits.models import loading
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    spec = importlib.util.spec_from_file_location(
        "_order_audit_seam", ROOT / "tools/sdsc_student_name_invariant_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    tokenizer = audit.load_tokenizer()
    m = worker()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    bases = p._build_bases()
    views = p.development_views(bases["student_dev"]) if evaluate else p.fit_views(bases["student_fit"])[:8]
    encoded = [p.encode_view(view, tokenizer, config, training=not evaluate) for view in views]
    correct = [not evaluate or (ordinal // 6) % 3 == 0 for ordinal in range(len(views))]
    targets = {
        tuple(row["input_ids"][: row["prefix_length"]]): row["input_ids"][row["prefix_length"] :]
        if correct[ordinal]
        else [tokenizer.eos_token_id]
        for ordinal, row in enumerate(encoded)
    }

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(311)])
            self.config = SimpleNamespace(max_position_embeddings=2456, vocab_size=len(tokenizer))

        def forward(self, *, input_ids, **kwargs):
            del kwargs
            return SimpleNamespace(logits=self.weights[0].expand(1, input_ids.shape[1], 2))

        def generate(self, *, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 256
            target = targets[tuple(input_ids[0].tolist())]
            return torch.cat((input_ids, torch.tensor([target], dtype=torch.long)), dim=1)

    original = Toy()
    masters = copy.deepcopy(original.state_dict())
    original.to(dtype=torch.bfloat16).eval()
    trainer = SimpleNamespace(model=original, _full_model_state_for_checkpoint=lambda: copy.deepcopy(masters))
    monkeypatch.setattr(
        loading,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(model=Toy(), tokenizer=tokenizer),
    )
    monkeypatch.setattr(m.util, "measurement", lambda *args: nullcontext())
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())

    def gather(actual):
        other = copy.deepcopy(actual)
        other["rank"] = 1 - actual["rank"]
        other["counts"] = m.empty_counts()
        other["per_view_counts"] = {name: m.empty_counts() for name in p.DEV_VIEW_NAMES}
        other["structure_counts"] = {name: m.empty_counts() for name in p.STRUCTURES}
        for ordinal in m.generation_ordinals(len(views), other["rank"], evaluate=evaluate):
            verification = dict(
                answer_correct=correct[ordinal], reward=float(correct[ordinal]), parse_valid=correct[ordinal]
            )
            m.add_verification(other["counts"], verification)
            if evaluate:
                view = views[ordinal]
                m.add_verification(other["per_view_counts"][view.view], verification)
                if view.view == "identity":
                    m.add_verification(
                        other["structure_counts"][view.example.metadata["structure"]], verification
                    )
        return [actual, other] if actual["rank"] == 0 else [other, actual]

    monkeypatch.setattr(m.util, "gather", gather)
    checkpoints, summaries, step_records = [], [], {}
    for step in m.STEPS if evaluate else (4,):
        results = []
        for rank in (0, 1):
            checkpoint, summary = m.checkpoint_and_development(
                trainer,
                config,
                tokenizer,
                views,
                encoded,
                output=tmp_path,
                rank=rank,
                step=step,
                inputs={"mode": "fit" if evaluate else "preflight", "protocol": {"sha256": "a" * 64}},
                dataset_sha="b" * 64,
                protocol_sha="c" * 64,
                dev_sha=p.EXAMPLES_SHA256["student_dev"],
                evaluate=evaluate,
            )
            results.append(summary)
        assert results[0] == results[1]
        checkpoints.append(checkpoint)
        if summary is not None:
            summaries.append(summary)
        step_records[step] = []
        for rank in (0, 1):
            name = (
                f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl"
                if evaluate
                else f"prepare-preflight-records-rank-{rank}.jsonl"
            )
            step_records[step].append(
                [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
            )
    prompts = (
        [
            dict(
                ordinal=ordinal,
                example=asdict(view.example),
                prompt_ids=row["input_ids"][: row["prefix_length"]],
                source_example_id=view.source_example_id,
                view=view.view,
                prompt_text=format_model_prompt(view.prompt, tokenizer, config).model_facing_prompt,
            )
            for ordinal, (view, row) in enumerate(zip(views, encoded, strict=True))
        ]
        if evaluate
        else []
    )
    selected = p.select_checkpoint(summaries) if evaluate else None
    report = dict(
        schema=m.SCHEMA,
        mode="fit" if evaluate else "preflight",
        execution_complete=True,
        preparation_complete=evaluate,
        optimizer_steps=32 if evaluate else 4,
        fit_base_examples=256,
        fit_training_views=2048,
        consumed_fit_training_views=2048 if evaluate else 256,
        development_base_examples=256,
        development_views=1536,
        source_kind="symbolic_canonical_name_invariant_preparation",
        learning_rate=2.5e-5,
        checkpoints=checkpoints,
        development=summaries,
        selected_checkpoint=selected,
        passed=selected is not None if evaluate else True,
        development_generation=dict(
            max_new_tokens=256,
            max_model_input_length=2456,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
            truncation=False,
        ),
        **m.FLAGS,
    )
    result = audit.replay(report, prompts, step_records, tokenizer)
    assert result["raw_replay_passed"] and result["responses_replayed"] == (18432 if evaluate else 8)
    if evaluate:
        assert result["selected_checkpoint"]["step"] == 1
        assert all(row["num_examples"] == 256 for row in summaries[0]["per_view_counts"].values())
    else:
        assert all(row["verification"]["reward"] == 1.0 for group in step_records[4] for row in group)


def run_distributed_child(output):
    """Real Gloo: actual focus row order, 32 global64 updates, restore at step4."""
    from collections import Counter

    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as p
    from posttrain_circuits.learning.collation import collate_trajectories
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
    from posttrain_circuits.learning.training.optimizer import build_adamw
    from posttrain_circuits.learning.training.schedules import PromptScheduler
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    seed_everything(42)
    bases = p.make_examples(p.DIFFICULTY)["student_fit"]
    control_views = p.fit_views(bases)
    toy_rows = records(2048)
    for row, view in zip(toy_rows, control_views, strict=True):
        row.prompt_id = view.example.example_id
        row.sampling_cursor_id = row.prompt_id
        row.trajectory_id = row.expected_trajectory_id
        row.validate()
    by_id = {row.prompt_id: row for row in toy_rows}
    actual_views = p.fit_views(bases)
    rows = [by_id[view.example.example_id] for view in actual_views]
    encoded = [dict(input_ids=row.input_ids + row.response_ids) for row in rows]
    ids = [row.prompt_id for row in rows]
    rank = int(os.environ["RANK"])
    model = build_tiny_qwen3(91)
    initial = copy.deepcopy(model.state_dict())
    optimizer = build_adamw(model.parameters(), learning_rate=m.LR, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    prompt_scheduler = PromptScheduler.for_distributed_rank(
        ids, ["fixture"] * len(rows), 4, rank=rank, world_size=2
    )
    trainer = FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=m.CanonicalSource(rows),
        supervisor=VerifiedReplaySupervisor(0, retry_limit=0),
        config=TrainerConfig(
            max_steps=32,
            token_budget=2_000_000,
            gradient_accumulation_steps=8,
            backend="accelerate",
            require_evaluation_metrics=False,
        ),
        run_dir=output / "training",
    )
    observer = m.StepObserver(optimizer)
    reference = build_tiny_qwen3(99)
    reference.load_state_dict(initial)
    reference_optimizer = build_adamw(reference.parameters(), learning_rate=m.LR, weight_decay=0)
    max_error = 0.0
    seen = []
    restored = None
    for step in range(1, 33):
        result = m.train_window(trainer, ids, encoded, rank, step, observer=observer)
        offset = (step - 1) * 64
        window = actual_views[offset : offset + 64]
        nonbranch = "chain" if (step - 1) % 2 == 0 else "converging_dag"
        assert Counter(view.example.metadata["structure"] for view in window) == {"branch": 48, nonbranch: 16}
        owned = window[rank::2]
        assert Counter(view.example.metadata["structure"] for view in owned) == {"branch": 24, nonbranch: 8}
        assert Counter(view.example.label for view in owned) == {0: 16, 1: 16}
        assert Counter(view.view for view in owned) == dict.fromkeys(p.FIT_VIEW_NAMES, 4)
        assert result["global_slots"] == list(range(offset + rank, offset + 64, 2))
        seen.extend(result["global_slots"])
        assert trainer.token_budget.consumed == sum(len(row["input_ids"]) for row in encoded[: step * 64])
        reference_optimizer.zero_grad()
        batch = collate_trajectories(TrajectoryBatch(rows[offset : offset + 64], 0), pad_token_id=0)
        loss = VerifiedReplaySupervisor(0, retry_limit=0).compute_loss(reference, batch).loss
        loss.backward()
        reference_optimizer.step()
        actual = trainer._accelerator.unwrap_model(trainer.model).state_dict()
        error = max(
            float((tensor - reference.state_dict()[name]).abs().max()) for name, tensor in actual.items()
        )
        max_error = max(max_error, error)
        assert max_error < 3e-6, (step, max_error)
        if step == 4:
            restored = m.full_state_restore(trainer, output, rank)
            assert restored["passed"]
            state_after = m.runtime_state(trainer)
        if step == 5:
            expected_model = m.tree_hash(list(trainer.model.parameters()))
            expected_optimizer = m.tree_hash(trainer.optimizer.state_dict())
            expected_runtime = m.runtime_state(trainer)
            trainer._accelerator.load_state(str(output / "resume" / "step-00000004"))
            m.restore_runtime(trainer, state_after)
            observer.calls = 4
            m.train_window(trainer, ids, encoded, rank, 5, observer=observer)
            assert m.tree_hash(list(trainer.model.parameters())) == expected_model
            assert m.tree_hash(trainer.optimizer.state_dict()) == expected_optimizer
            assert m.runtime_state(trainer) == expected_runtime
    assert seen == list(range(rank, 2048, 2))
    assert trainer.global_step == observer.calls == 32
    assert len(trainer.state_source.cursor) == 1024
    assert all(value == 1 for value in trainer.state_source.cursor.values())
    window_tokens = [
        sum(len(row["input_ids"]) for row in encoded[start : start + 64]) for start in range(0, 2048, 64)
    ]
    m.validate_training_completion(
        dict(
            window_input_tokens=window_tokens,
            fit_rows=2048,
            optimizer_windows=32,
            token_budget=2_000_000,
            total_input_tokens=sum(window_tokens),
        ),
        mode="fit",
        completed_updates=trainer.global_step,
        consumed_input_tokens=trainer.token_budget.consumed,
        consumed_training_views=len(rows),
    )
    m.atomic(
        output / f"rank-{rank}.json",
        dict(
            passed=True,
            completed_updates=32,
            global_training_rows=2048,
            max_global64_update_error=max_error,
            next_update_equal=True,
            restore=restored,
        ),
    )
    torch.distributed.destroy_process_group()


def test_actual_two_rank_global64_loss_and_full_state_next_update(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES="",
        ACCELERATE_USE_CPU="true",
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-B", __file__, "--gloo-child", str(tmp_path)],
            env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for rank in range(2)
    ]
    try:
        results = [child.communicate(timeout=120)[0] for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    assert [child.returncode for child in children] == [0, 0], "\n".join(results)
    for rank in range(2):
        value = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert value["next_update_equal"] and value["completed_updates"] == 32
        assert value["global_training_rows"] == 2048


def exact_shape_tiny_qwen(seed):
    """311-key Qwen3 architecture with tiny widths; no hardware claim."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from posttrain_circuits.utils.tiny_model import tiny_vocabulary

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tiny_vocabulary()),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=2456,
            use_cache=False,
        )
    )
    assert len(model.state_dict()) == 311
    return model


if __name__ == "__main__" and "--gloo-child" in sys.argv:
    run_distributed_child(Path(sys.argv[-1]))


def test_real_node_input_shape_enters_fresh_worker_with_science_cwd(tmp_path):
    from types import SimpleNamespace

    m = worker()
    spec = importlib.util.spec_from_file_location(
        "_prepare_node", ROOT / "tools/sdsc_student_name_invariant_job.py"
    )
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    science, output = tmp_path / "science", tmp_path / "output"
    (science / "src").mkdir(parents=True)
    (tmp_path / "huggingface").mkdir()
    output.mkdir()
    protocol_path = "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json"
    (science / protocol_path).parent.mkdir(parents=True)
    (science / protocol_path).write_text('{"fixture":true}')
    frozen = {}
    for index in range(49):
        name = f"src/frozen_{index}.py"
        (science / name).write_text(f"VALUE = {index}\n")
        frozen[name] = m.file_sha(science / name)
    control = SimpleNamespace(
        read=lambda path: path.read_bytes(),
        PROTOCOL_PATH=protocol_path,
        INITIAL_SHA=m.util.INITIAL_SHA,
        sha=m.digest,
        canonical=m.canonical,
    )
    plan = dict(
        mode="preflight",
        run_id="fixture",
        code_sha256="a" * 64,
        resolved_config={"size": 7923, "sha256": m.util.CONFIG_SHA},
        protocol={"protocol_sha256": "b" * 64},
    )
    inputs = node.build_worker_inputs(plan, tmp_path, science, {"job_id": "12345"}, frozen, control)
    path = tmp_path / "inputs.json"
    path.write_bytes(m.canonical(inputs))
    child = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
inputs = m.document(Path(sys.argv[2]))
work = m.staged_science(inputs, Path(sys.argv[3]))
assert work == Path(sys.argv[3]).parent
assert set(inputs['protocol']) == {'path', 'size', 'sha256', 'protocol_sha256'}
print('node_to_worker_admitted')
"""
    env = dict(os.environ, HF_HOME=str(tmp_path / "huggingface"), SLURM_JOB_ID="12345")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_name_invariant_worker.py"),
            str(path),
            str(output),
        ],
        cwd=science,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "node_to_worker_admitted"
    wrong = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_name_invariant_worker.py"),
            str(path),
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert wrong.returncode != 0 and "cwd must equal" in wrong.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "old12",
        "short_prefix",
        "wrong_total",
        "missing_window",
        "bool_count",
        "float_audit_total",
        "over_budget",
    ],
)
@pytest.mark.parametrize("mode", ["preflight", "fit"])
def test_completion_binds_mode_prefix_and_actual_tokens(mutation, mode):
    m = worker()
    audit = dict(
        window_input_tokens=[100] * 32,
        fit_rows=2048,
        optimizer_windows=32,
        total_input_tokens=3200,
        token_budget=2_000_000,
    )
    steps = 4 if mode == "preflight" else 32
    actual = dict(
        mode=mode,
        completed_updates=steps,
        consumed_input_tokens=steps * 100,
        consumed_training_views=steps * 64,
    )
    if mutation == "old12":
        actual.update(completed_updates=12, consumed_training_views=768)
    elif mutation == "short_prefix":
        actual["consumed_input_tokens"] = 1200
    elif mutation == "wrong_total":
        audit["total_input_tokens"] = 3199
    elif mutation == "missing_window":
        audit["window_input_tokens"].pop()
    elif mutation == "bool_count":
        actual["completed_updates"] = True
    elif mutation == "float_audit_total":
        audit["total_input_tokens"] = 3200.0
    elif mutation == "over_budget":
        audit.update(window_input_tokens=[100_000] * 32, total_input_tokens=3_200_000)
        actual["consumed_input_tokens"] = 3_200_000
    if mutation is None:
        m.validate_training_completion(audit, **actual)
    else:
        with pytest.raises(ValueError):
            m.validate_training_completion(audit, **actual)


@pytest.mark.parametrize(
    "mutation", ["format", "scope", "mode", "step", "parent", "accepted", "extra", "arm"]
)
def test_dense_metadata_rejects_relabelled_probe_and_changed_provenance(tmp_path, mutation):
    m = worker()
    path = tmp_path / "checkpoint.pt"
    m.save_dense(
        {"fixture": torch.ones(1)},
        path,
        step=4,
        inputs={"mode": "fit", "protocol": {"sha256": "a" * 64}},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
    )
    saved = torch.load(path, weights_only=False)
    kwargs = dict(
        step=4, mode="fit", dataset_sha="b" * 64, protocol_sha="c" * 64, protocol_artifact_sha="a" * 64
    )
    m.validate_dense_payload(saved, **kwargs)
    key, value = {
        "format": ("format", "student_batch_probe_dense_v1"),
        "scope": ("scope", "student_batch_probe_diagnostic_model_only"),
        "mode": ("mode", "preflight"),
        "step": ("global_step", True),
        "parent": ("parent_checkpoint_sha256", "0" * 64),
        "accepted": ("student_accepted", True),
        "extra": ("selected_checkpoint", 4),
        "arm": ("arm", "control"),
    }[mutation]
    saved[key] = value
    with pytest.raises(ValueError):
        m.validate_dense_payload(saved, **kwargs)
