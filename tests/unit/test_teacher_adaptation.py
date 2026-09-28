"""Pure CPU contracts for separate canonical-proof teacher adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_example, render_target
from posttrain_circuits.datasets.proofgraph.splits import canonical_semantic_key
from posttrain_circuits.learning.teacher import adaptation

DIFFICULTY = {"depth": 3, "distractors": 4, "structures": ["chain", "branch", "converging_dag"]}


@pytest.fixture(scope="module")
def examples():
    return adaptation.make_examples(DIFFICULTY)


def test_generated_fit_and_dev_are_deterministic_balanced_disjoint_pairs(examples):
    repeated = adaptation.make_examples(DIFFICULTY)
    assert adaptation.dataset_manifest(examples) == adaptation.dataset_manifest(repeated)
    expected = {"teacher_fit": (256, 70_000_042), "teacher_dev": (32, 80_000_042)}
    task = ProofGraphTask()
    inventories = []
    for role, rows in examples.items():
        count, first_seed = expected[role]
        assert len(rows) == count
        assert Counter(row.label for row in rows) == {0: count // 2, 1: count // 2}
        assert set(Counter(row.pair_group_id for row in rows).values()) == {2}
        seeds = {row.metadata["pair_seed"] for row in rows}
        assert min(seeds) == first_seed and len(seeds) == count // 2
        assert max(seeds) < first_seed + count * 100
        for row in rows:
            assert task.verify(row, task.parse_response(render_target(row))).reward == 1.0
        manifest = adaptation.dataset_manifest({role: rows})[role]
        assert manifest["ordered_ids"] == [row.example_id for row in rows]
        assert manifest["examples_sha256"] == sha256_value([asdict(row) for row in rows])
        inventories.append(adaptation.isolation_inventory({role: rows}))
    for field in ("semantic", "example_id", "pair_group_id", "pair_seed"):
        assert inventories[0][field].isdisjoint(inventories[1][field])


@pytest.mark.parametrize("collision", ["semantic", "example_id", "pair_group_id", "pair_seed"])
def test_each_formal_overlap_identity_independently_rejects(examples, collision):
    inventory = adaptation.isolation_inventory(examples)
    original = examples["teacher_fit"][0]
    formal = ProofGraphTask().generate_pair(42, DIFFICULTY)[0]
    adaptation.reject_formal_overlap(inventory, formal)
    if collision == "semantic":
        candidate = copy.deepcopy(original)
        candidate.example_id = formal.example_id
        candidate.pair_group_id = formal.pair_group_id
        candidate.metadata["pair_seed"] = formal.metadata["pair_seed"]
    else:
        candidate = copy.deepcopy(formal)
        if collision == "pair_seed":
            candidate.metadata["pair_seed"] = original.metadata["pair_seed"]
        else:
            setattr(candidate, collision, getattr(original, collision))
    observed = {
        "semantic": canonical_semantic_key(candidate),
        "example_id": candidate.example_id,
        "pair_group_id": candidate.pair_group_id,
        "pair_seed": candidate.metadata["pair_seed"],
    }
    assert {name for name, value in observed.items() if value in inventory[name]} == {collision}
    with pytest.raises(ValueError, match="overlaps the original formal dataset"):
        adaptation.reject_formal_overlap(inventory, candidate)


def test_isolation_inventory_rejects_a_pair_crossing_fit_and_dev(examples):
    pair = examples["teacher_fit"][:2]
    assert pair[0].pair_group_id == pair[1].pair_group_id
    with pytest.raises(ValueError, match="pair group crosses"):
        adaptation.isolation_inventory({"teacher_fit": pair[:1], "teacher_dev": pair[1:]})


class BoundaryTokenizer:
    """Detect forbidden joined-string retokenization and implicit truncation."""

    chat_template = "test-only non-thinking template"

    def __init__(self, example, *, prefix_tokens=7, target_tokens=5, eos=999):
        self.prefix_tokens, self.target_tokens = prefix_tokens, target_tokens
        self.eos_token_id = eos
        self.raw = render_example(example)
        self.prompt = (
            "<|im_start|>user\n" + self.raw + ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )
        self.target = render_target(example)
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert messages == [{"role": "user", "content": self.raw}]
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return self.prompt

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls.append(text)
        if text == self.prompt:
            return [101] * self.prefix_tokens
        if text == self.target:
            return [201] * self.target_tokens
        raise AssertionError("Unexpected concatenated or rewritten model input")


def model_config(tokenizer):
    return {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }


def test_encoding_preserves_prompt_and_response_boundary_and_supervises_eos(examples):
    example = examples["teacher_fit"][0]
    before = asdict(example)
    tokenizer = BoundaryTokenizer(example)
    row = adaptation.encode_example(example, tokenizer, model_config(tokenizer))
    assert tokenizer.calls == [tokenizer.prompt, tokenizer.target]
    assert row["input_ids"] == [101] * 7 + [201] * 5 + [999]
    assert row["labels"] == [-100] * 7 + [201] * 5 + [999]
    assert (row["prefix_length"], row["response_length"]) == (7, 6)
    assert row["prompt_sha256"] == sha256_value(tokenizer.prompt)
    assert row["target_sha256"] == sha256_value(tokenizer.target)
    assert asdict(example) == before


@pytest.mark.parametrize(
    ("prefix", "target", "eos", "error"),
    [
        (1247, 10, 999, "token envelope"),
        (10, 256, 999, "token envelope"),
        (0, 3, 999, "nonempty"),
        (3, 0, 999, "nonempty"),
        (3, 3, None, "EOS"),
    ],
)
def test_encoding_rejects_excess_length_or_missing_targets_without_truncation(
    examples,
    prefix,
    target,
    eos,
    error,
):
    example = examples["teacher_fit"][0]
    tokenizer = BoundaryTokenizer(example, prefix_tokens=prefix, target_tokens=target, eos=eos)
    with pytest.raises(ValueError, match=error):
        adaptation.encode_example(example, tokenizer, model_config(tokenizer))
    assert tokenizer.calls == [tokenizer.prompt, tokenizer.target]


def test_exact_prompt_response_limits_fit_without_dropping_any_token(examples):
    example = examples["teacher_fit"][0]
    tokenizer = BoundaryTokenizer(example, prefix_tokens=1246, target_tokens=255)
    row = adaptation.encode_example(example, tokenizer, model_config(tokenizer))
    assert len(row["input_ids"]) == 1502
    assert row["prefix_length"] == 1246 and row["response_length"] == 256
    assert row["labels"][-1] == tokenizer.eos_token_id


def test_collation_and_loss_are_sequence_mean_and_ignore_prompt_and_padding():
    rows = [
        {"input_ids": [0, 0, 1, 2], "labels": [-100, -100, 1, 2]},
        {"input_ids": [0, 0, 2, 2, 1, 1, 2], "labels": [-100, -100, 2, 2, 1, 1, 2]},
    ]
    batch = adaptation.collate(rows, pad_id=0, device=torch.device("cpu"))
    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"][0].tolist() == [-100, -100, 1, 2, -100, -100, -100]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    logits = torch.zeros((2, 7, 3), dtype=torch.float64)
    for row_index, target_probability in enumerate((0.5, 0.25)):
        for token_index, label in enumerate(batch["labels"][row_index].tolist()):
            if label != -100:
                probabilities = torch.full((3,), (1 - target_probability) / 2, dtype=torch.float64)
                probabilities[label] = target_probability
                logits[row_index, token_index - 1] = probabilities.log()
    logits.requires_grad_(True)
    loss = adaptation.response_sequence_loss(logits, batch["labels"])
    expected = (-math.log(0.5) - math.log(0.25)) / 2
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    assert float(loss.detach()) != pytest.approx((-2 * math.log(0.5) - 5 * math.log(0.25)) / 7)
    loss.backward()
    active = torch.zeros((2, 7), dtype=torch.bool)
    active[:, :-1] = batch["labels"][:, 1:] != -100
    assert logits.grad[~active].abs().sum() == 0
    assert bool((logits.grad[active].abs().sum(-1) > 0).all())
    altered = logits.detach().clone()
    altered[~active] = torch.tensor([-70.0, 80.0, 90.0], dtype=altered.dtype)
    assert float(adaptation.response_sequence_loss(altered, batch["labels"])) == pytest.approx(expected)


def test_loss_rejects_even_one_sequence_without_response_targets():
    labels = torch.tensor([[-100, 1, 2], [-100, -100, -100]])
    with pytest.raises(ValueError, match="no response targets"):
        adaptation.response_sequence_loss(torch.zeros((2, 3, 3)), labels)
    with pytest.raises(ValueError, match="empty"):
        adaptation.collate([], pad_id=0, device="cpu")


def checkpoint_fixture(tmp_path):
    merged, adapter = tmp_path / "merged", tmp_path / "adapter"
    merged.mkdir()
    adapter.mkdir()
    save_file({"model.weight": torch.ones((2, 2))}, merged / "model.safetensors")
    save_file({"lora_A.weight": torch.zeros((1, 2))}, adapter / "adapter_model.safetensors")
    (merged / "config.json").write_text(
        json.dumps({"_name_or_path": "Qwen/Qwen3-8B", "_commit_hash": adaptation.PREFLIGHT["base_revision"]})
    )
    return merged, adapter


def test_checkpoint_identity_binds_dense_weights_even_with_unchanged_fake_base_metadata(tmp_path):
    merged, _ = checkpoint_fixture(tmp_path)
    kwargs = {"base_revision": adaptation.PREFLIGHT["base_revision"], "plan": adaptation.PREFLIGHT}
    first = adaptation.checkpoint_manifest(tmp_path, **kwargs)
    assert first == adaptation.checkpoint_manifest(tmp_path, **kwargs)
    assert first["artifact_kind"] == "adapted_dense_teacher"
    assert first["formal_teacher_accepted"] is False
    assert first["sha256"] == sha256_value({key: value for key, value in first.items() if key != "sha256"})
    assert first["adaptation_plan_sha256"] == sha256_value(adaptation.PREFLIGHT)
    before_config = (merged / "config.json").read_bytes()
    save_file({"model.weight": torch.full((2, 2), 2.0)}, merged / "model.safetensors")
    second = adaptation.checkpoint_manifest(tmp_path, **kwargs)
    assert (merged / "config.json").read_bytes() == before_config
    assert first["base_revision"] == second["base_revision"]
    assert first["sha256"] != second["sha256"]
    first_weights = next(row for row in first["files"] if row["path"] == "merged/model.safetensors")
    second_weights = next(row for row in second["files"] if row["path"] == "merged/model.safetensors")
    assert first_weights["sha256"] != second_weights["sha256"]


def test_base_config_without_dense_weights_cannot_be_a_teacher_checkpoint(tmp_path):
    merged, _ = checkpoint_fixture(tmp_path)
    (merged / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="no dense safe tensors"):
        adaptation.checkpoint_manifest(tmp_path, base_revision="claimed-base", plan=adaptation.PREFLIGHT)


@pytest.mark.parametrize("link_kind", ["file", "nested_directory", "merged_directory"])
def test_checkpoint_rejects_symlinks(tmp_path, link_kind):
    root = tmp_path / "checkpoint"
    root.mkdir()
    merged, adapter = checkpoint_fixture(root)
    if link_kind == "file":
        (merged / "foreign.safetensors").symlink_to(adapter / "adapter_model.safetensors")
    elif link_kind == "nested_directory":
        (merged / "foreign").symlink_to(adapter, target_is_directory=True)
    else:
        merged.rename(root / "real_merged")
        merged.symlink_to(root / "real_merged", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|real merged/adapter"):
        adaptation.checkpoint_manifest(root, base_revision="claimed-base", plan=adaptation.PREFLIGHT)


class PhaseRecorder:
    def __init__(self):
        self.events = []

    def phase(self, name, **context):
        self.events.append((name, context))


def tiny_adapter_fixture():
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

    model, tokenizer = build_tiny_qwen3(17), build_tiny_tokenizer()
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    rows = [
        {
            "example_id": f"fit-{index}",
            "input_ids": [2, 4 + index, 7, 8 + index, 3],
            "labels": [-100, -100, 7, 8 + index, 3],
        }
        for index in range(16)
    ]
    dev = [{"example_id": "dev-0", "input_ids": [2, 15, 9, 3], "labels": [-100, -100, 9, 3]}]
    plan = {
        **adaptation.PREFLIGHT,
        "train_examples": 16,
        "dev_examples": 1,
        "optimizer_steps": 8,
        "global_batch_size": 2,
        "lora_rank": 2,
        "lora_alpha": 4,
    }
    return model, tokenizer, rows, dev, plan


def test_real_tiny_qwen_lora_train_merge_save_and_offline_reload(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    assert peft.__version__ == "0.17.1"
    from posttrain_circuits.artifacts.hashing import sha256_file
    from posttrain_circuits.models.loading import tokenizer_fingerprint

    model, tokenizer, rows, dev, plan = tiny_adapter_fixture()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    phases = PhaseRecorder()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CPU path touched CUDA"))
    provenance = {
        "job_id": "test-job",
        "run_id": "test-run",
        "code_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
    }
    reloaded, saved_tokenizer, metrics, manifest = adaptation.run_adapter_preflight(
        model,
        tokenizer,
        rows,
        dev,
        tmp_path,
        observe=phases,
        plan=plan,
        training_provenance=provenance,
    )
    assert next(reloaded.parameters()).device.type == "cpu"
    assert next(reloaded.parameters()).dtype == torch.float32
    assert not any("lora_" in name for name, _ in reloaded.named_parameters())
    changed = []
    for name, value in reloaded.named_parameters():
        if not torch.equal(value, before[name]):
            changed.append(name)
    assert changed and all(name.split(".")[-2] in adaptation.LORA_MODULES for name in changed)
    assert metrics["adapter_update_l2"] > 0 and metrics["optimizer_steps"] == 8
    assert metrics["adapter_merge_max_logit_error"] < 1e-5
    assert metrics["dense_reload_max_logit_error"] < 1e-5
    assert metrics["dense_adapted_modules_sha256_before"] != metrics["dense_adapted_modules_sha256_after"]
    assert len(metrics["adapter_modules"]) == 7
    assert metrics["optimizer_defaults"]["betas"] == (0.9, 0.999)
    assert metrics["optimizer_defaults"]["eps"] == 1e-8
    order = torch.randperm(16, generator=torch.Generator().manual_seed(plan["seed"])).tolist()
    assert metrics["ordered_consumed_example_ids"] == [rows[index]["example_id"] for index in order]
    assert metrics["consumed_tokens"] == sum(len(row["input_ids"]) for row in rows)
    records = [json.loads(line) for line in (tmp_path / "train-metrics.jsonl").read_text().splitlines()]
    assert len(records) == 8 and [row["samples"] for row in records] == list(range(2, 17, 2))
    assert all(math.isfinite(row["loss"]) and row["grad_norm"] > 0 for row in records)
    assert [identity for row in records for identity in row["example_ids"]] == metrics[
        "ordered_consumed_example_ids"
    ]
    assert all(not identity.startswith("dev-") for row in records for identity in row["example_ids"])
    assert manifest["training_provenance"] == {
        **provenance,
        "optimizer_steps": 8,
        "consumed_tokens": 80,
        "train_metrics_sha256": sha256_file(tmp_path / "train-metrics.jsonl"),
    }
    assert manifest["adaptation_plan_sha256"] == sha256_value(plan)
    assert metrics["adapted_teacher_sha256"] == manifest["sha256"]
    assert manifest["formal_teacher_accepted"] is False
    assert tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(saved_tokenizer)
    assert saved_tokenizer.chat_template == tokenizer.chat_template
    config = json.loads((tmp_path / "merged/config.json").read_text())
    assert not config.get("_commit_hash")
    names = [name for name, _ in phases.events]
    assert names.index("teacher_training_start") < names.index("teacher_optimizer_window")
    assert names.count("teacher_optimizer_window_committed") == 8
    assert names[-1] == "checkpoint_manifest"


@pytest.mark.parametrize(
    "fault", ["missing_window", "overlap", "prompt_label", "missing_eos", "too_long", "module_change"]
)
def test_real_helper_rejects_bad_inputs_before_installing_adapter_or_training(tmp_path, fault):
    pytest.importorskip("peft")
    model, tokenizer, rows, dev, plan = tiny_adapter_fixture()
    if fault == "missing_window":
        rows.pop()
    elif fault == "overlap":
        dev[0]["example_id"] = rows[0]["example_id"]
    elif fault == "prompt_label":
        rows[0]["labels"][0] = 2
    elif fault == "missing_eos":
        rows[0]["input_ids"][-1] = rows[0]["labels"][-1] = 4
    elif fault == "too_long":
        rows[0]["input_ids"] = [2] * 1537
        rows[0]["labels"] = [-100] * 1536 + [3]
    elif fault == "module_change":
        plan["lora_modules"] = ["q_proj"]
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    phases = PhaseRecorder()
    with pytest.raises(ValueError):
        adaptation.run_adapter_preflight(model, tokenizer, rows, dev, tmp_path, observe=phases, plan=plan)
    assert phases.events == []
    assert list(tmp_path.iterdir()) == []
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_checkpoint_manifest_binds_training_data_and_actual_training_summary(tmp_path):
    checkpoint_fixture(tmp_path)
    initial = {
        "dataset_manifest_sha256": "a" * 64,
        "train_metrics_sha256": "b" * 64,
        "optimizer_steps": 8,
        "consumed_tokens": 1234,
    }
    first = adaptation.checkpoint_manifest(
        tmp_path,
        base_revision="base",
        plan=adaptation.PREFLIGHT,
        training_provenance=initial,
    )
    for key in initial:
        changed = {**initial, key: "c" * 64 if key.endswith("sha256") else initial[key] + 1}
        second = adaptation.checkpoint_manifest(
            tmp_path,
            base_revision="base",
            plan=adaptation.PREFLIGHT,
            training_provenance=changed,
        )
        assert first["sha256"] != second["sha256"]
        assert first["files"] == second["files"]


def test_adaptation_progress_reports_teacher_training_without_student_or_readiness_claim(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools/sdsc_teacher_adapt.py"
    spec = importlib.util.spec_from_file_location("_test_teacher_adaptation_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    report = {"task": worker.TASK, "job_id": "test", "run_id": "test-run", "code_sha256": "a" * 64}
    observe = worker.adaptation_observation(worker.sibling("sdsc_teacher_probe"), tmp_path, report)
    try:
        initial = json.loads((tmp_path / "progress.json").read_text())
        assert initial["artifact_kind"] == "teacher_adaptation_progress"
        assert initial["training_started"] is initial["teacher_training_started"] is False
        observe.phase("teacher_training_start")
        observe.phase("teacher_optimizer_window_committed", completed_optimizer_steps=2)
        observe.failed(RuntimeError("test export failure after teacher training"))
        final = json.loads((tmp_path / "progress.json").read_text())
        assert final["training_started"] is final["teacher_training_started"] is True
        assert final["completed_optimizer_steps"] == 2
        assert final["student_training_started"] is final["readiness_artifact_produced"] is False
        assert final["passed"] is final["accepted_science"] is final["full_teacher_ready"] is False
        assert final["not_a_completion_report"] is True
        assert report["training_started"] is report["teacher_training_started"] is True
        assert report["completed_optimizer_steps"] == 2
    finally:
        observe.close()
