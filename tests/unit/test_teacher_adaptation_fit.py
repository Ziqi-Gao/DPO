"""Real CPU/Gloo tests of teacher DDP scaling, failure, and optimizer-boundary resume."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.learning.teacher import adaptation_fit as fit
from posttrain_circuits.learning.teacher.adaptation import (
    LORA_MODULES,
    PREFLIGHT,
    collate,
    response_sequence_loss,
)


class Observation:
    def phase(self, *_args, **_kwargs):
        pass


def rows_for(count):
    result = []
    for index in range(count):
        prefix = [2, 4 + index % 23]
        target = [11 + index % 13] * (1 + index % 4) + [3]
        result.append(
            {
                "example_id": f"fit-{index}",
                "input_ids": prefix + target,
                "labels": [-100] * len(prefix) + target,
            }
        )
    return result


def tiny_plan(world=4, *, epochs=1):
    return fit.FitPlan(
        mode="cpu_test",
        world_size=world,
        train_examples=64,
        epochs=epochs,
        lora_rank=2,
        lora_alpha=4,
        warmup_steps=0,
        checkpoint_steps=tuple(range(1, epochs + 1)),
    )


def identity():
    return {
        "base_revision": PREFLIGHT["base_revision"],
        "tokenizer_sha256": "a" * 64,
        "fit_dataset_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
    }


def _distributed_case(rank, world, rendezvous, output, case, resume_path):
    from peft import get_peft_model_state_dict

    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

    torch.set_num_threads(1)
    output = Path(output)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=dt.timedelta(seconds=45),
    )
    try:
        plan = (
            fit.FitPlan.preflight()
            if case == "preflight"
            else tiny_plan(world, epochs=2 if case in {"full", "interrupt", "resume", "bad_resume"} else 1)
        )
        rows = rows_for(plan.train_examples)
        if case == "bad_row" and rank == 1:
            rows[0]["labels"][-1] = -100
        base, tokenizer = build_tiny_qwen3(17), build_tiny_tokenizer()
        if case == "bad_forward" and rank == 1:

            def fail_forward(*_args, **_kwargs):
                raise RuntimeError("injected rank-local forward failure")

            base.forward = fail_forward

        def hook(_path, manifest):
            # Evaluation may consume random numbers; the domain restores them.
            torch.rand(7)
            if case == "interrupt" and manifest["metadata"]["completed_updates"] == 1:
                raise RuntimeError("intentional interruption after committed checkpoint")

        try:
            model, summary = fit.run_adapter_fit(
                base,
                tokenizer,
                rows,
                output,
                identity=identity(),
                observe=Observation(),
                plan=plan,
                resume_from=Path(resume_path) if resume_path else None,
                checkpoint_hook=hook,
            )
            if rank == 0:
                torch.save(
                    {key: tensor.detach().cpu() for key, tensor in get_peft_model_state_dict(model).items()},
                    output / "result-adapter.pt",
                )
            (output / f"result-rank{rank}.json").write_text(
                json.dumps({"ok": True, "summary": summary}, allow_nan=False)
            )
        except RuntimeError as error:
            if case not in {"bad_row", "bad_forward", "interrupt", "bad_resume"}:
                raise
            (output / f"result-rank{rank}.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(error),
                        "adapter_installed": any("lora_" in name for name, _ in base.named_parameters()),
                    }
                )
            )
    finally:
        dist.destroy_process_group()


def run_case(tmp_path, name, *, world=4, case="one", resume=None):
    pytest.importorskip("peft")
    output = tmp_path / name
    output.mkdir()
    mp.spawn(
        _distributed_case,
        args=(
            world,
            str(tmp_path / f"rendezvous-{name}"),
            str(output),
            case,
            str(resume) if resume else None,
        ),
        nprocs=world,
        join=True,
    )
    return output, [json.loads((output / f"result-rank{rank}.json").read_text()) for rank in range(world)]


def reference_update():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

    model, tokenizer = build_tiny_qwen3(17), build_tiny_tokenizer()
    torch.manual_seed(271828)
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=list(LORA_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    # One physically batched global64 reference, with real padding and variable
    # response lengths, independently computes the required sequence mean.
    rows = rows_for(64)
    batch = collate([rows[index] for index in fit.epoch_order(tiny_plan(), 0)], tokenizer.pad_token_id, "cpu")
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).logits
    loss = response_sequence_loss(logits, batch["labels"])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return get_peft_model_state_dict(model), float(loss.detach()), float(norm)


@pytest.mark.parametrize("world", [2, 4])
def test_real_gloo_ddp_update_matches_global64_sequence_mean(tmp_path, world):
    output, results = run_case(tmp_path, "ddp", world=world)
    assert all(row["ok"] for row in results)
    summaries = [row["summary"] for row in results]
    assert len({row["adapter_sha256"] for row in summaries}) == 1
    actual = torch.load(output / "result-adapter.pt", weights_only=True)
    expected, loss, norm = reference_update()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=3e-7, rtol=2e-3)
    metrics = json.loads((output / "train-metrics.jsonl").read_text())
    assert metrics["loss"] == pytest.approx(loss, abs=2e-6)
    assert metrics["grad_norm"] == pytest.approx(norm, rel=2e-5)
    expected_ids = [rows_for(64)[index]["example_id"] for index in fit.epoch_order(tiny_plan(world), 0)]
    shards = [
        json.loads((output / f"fit-samples-rank{rank}.jsonl").read_text())["example_ids"]
        for rank in range(world)
    ]
    assert all(shard == expected_ids[rank::world] for rank, shard in enumerate(shards))
    assert set().union(*map(set, shards)) == set(expected_ids)
    assert sum(map(len, shards)) == 64
    assert metrics["global_example_ids_sha256"] == sha256_value(expected_ids)
    assert summaries[0]["consumed_tokens"] == sum(len(row["input_ids"]) for row in rows_for(64))
    assert summaries[0]["same_world_resume_verified"] is False
    assert summaries[0]["student_training_started"] is False


def test_one_bad_rank_stops_every_rank_before_adapter_or_backward(tmp_path):
    output, results = run_case(tmp_path, "bad", case="bad_row")
    assert all(not row["ok"] and "input validation" in row["error"] for row in results)
    assert all(row["adapter_installed"] is False for row in results)
    assert not (output / "checkpoints").exists()


def test_real_same_world_resume_matches_uninterrupted_next_update(tmp_path):
    complete, full = run_case(tmp_path, "full", world=2, case="full")
    stopped, interrupted = run_case(tmp_path, "stopped", world=2, case="interrupt")
    assert all(not row["ok"] and "intentional interruption" in row["error"] for row in interrupted)
    checkpoint = stopped / "checkpoints/step-000001"
    resumed, results = run_case(tmp_path, "resumed", world=2, case="resume", resume=checkpoint)
    assert all(row["ok"] for row in results)
    assert results[0]["summary"]["adapter_sha256"] == full[0]["summary"]["adapter_sha256"]
    assert results[0]["summary"]["consumed_tokens"] == full[0]["summary"]["consumed_tokens"]
    assert results[0]["summary"]["resumed_updates"] == 1
    for filename in ("trainer.pt", "rng-rank0.pt", "rng-rank1.pt"):
        first = torch.load(complete / "checkpoints/step-000002" / filename, weights_only=True)
        second = torch.load(resumed / "checkpoints/step-000002" / filename, weights_only=True)
        assert fit._state_digest(first) == fit._state_digest(second)
    for rank in range(2):
        assert (complete / f"fit-samples-rank{rank}.jsonl").read_bytes() == (
            resumed / f"fit-samples-rank{rank}.jsonl"
        ).read_bytes()
    assert len((resumed / "train-metrics.jsonl").read_text().splitlines()) == 2
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    expected = {key: manifest["metadata"][key] for key in ("identity", "world_size", "rows_sha256")}
    fit.validate_fit_checkpoint(checkpoint, expected)
    with pytest.raises(ValueError, match="identity differs: world_size"):
        fit.validate_fit_checkpoint(checkpoint, {**expected, "world_size": 4})
    with pytest.raises(ValueError, match="identity differs: rows_sha256"):
        fit.validate_fit_checkpoint(checkpoint, {**expected, "rows_sha256": "d" * 64})
    # Parent-owned export/evaluation paths are allowed, unrelated files are not.
    (checkpoint / "merged").mkdir()
    (checkpoint / "dense-manifest.json").write_text("{}")
    fit.validate_fit_checkpoint(checkpoint, expected)
    (checkpoint / "unexpected.txt").write_text("not allowed")
    with pytest.raises(ValueError, match="unknown entries"):
        fit.validate_fit_checkpoint(checkpoint, expected)
    (checkpoint / "unexpected.txt").unlink()
    original = (checkpoint / "trainer.pt").read_bytes()
    (checkpoint / "trainer.pt").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    rejected, failures = run_case(tmp_path, "tampered", world=2, case="bad_resume", resume=checkpoint)
    assert all(not row["ok"] and "checkpoint file differs" in row["error"] for row in failures)
    assert all(row["adapter_installed"] is False for row in failures)
    assert not (rejected / "train-metrics.jsonl").exists()


def test_four_rank_preflight_really_perturbs_and_reloads_optimizer_and_rng(tmp_path):
    output, results = run_case(tmp_path, "preflight", world=4, case="preflight")
    assert all(row["ok"] and row["summary"]["same_world_resume_verified"] is True for row in results)
    for row in results:
        summary = row["summary"]
        assert summary["completed_updates"] == 8 and summary["completed_samples"] == 512
        assert set(summary["resume_verification"]) == {"adapter", "optimizer", "rng"}
        assert len(summary["step_seconds"]) == 8 and min(summary["step_seconds"]) > 0
        assert summary["peak_gpu_reserved_bytes"] == 0
    checkpoint = output / "checkpoints/step-000008"
    assert (checkpoint / "adapter/adapter_model.safetensors").is_file()
    assert (checkpoint / "adapter/adapter_config.json").is_file()
    assert len((output / "train-metrics.jsonl").read_text().splitlines()) == 8


def test_frozen_plan_lr_trace_and_partition_are_explicit():
    plan = fit.FitPlan()
    plan.validate(device_type="cuda")
    assert plan.steps == 512 and fit.FitPlan.preflight().steps == 8
    trace = fit.learning_rate_trace(plan)
    assert len(trace) == 512
    assert trace[0] == pytest.approx(1e-4 / 16) and trace[15] == 1e-4 and trace[-1] == 0
    assert all(left <= right for left, right in pairwise(trace[:16]))
    assert all(left >= right for left, right in pairwise(trace[15:]))
    assert fit.learning_rate_trace(fit.FitPlan.preflight()) == (1e-4,) * 8
    assert fit.epoch_order(plan, 0) != fit.epoch_order(plan, 1)
    assert fit.epoch_order(plan, 0) == fit.epoch_order(plan, 0)
    for epoch in range(4):
        assert set(fit.epoch_order(plan, epoch)) == set(range(8192))
    for change in ({"world_size": 2}, {"lora_rank": 64}, {"learning_rate": 1e-3}, {"epochs": 5}):
        with pytest.raises(ValueError):
            replace(plan, **change).validate(device_type="cuda")
    with pytest.raises(ValueError, match="CPU model"):
        tiny_plan().validate(device_type="cuda")
    with pytest.raises(ValueError, match="global teacher window"):
        fit.rank_window([0] * 64, rank=0, world_size=4)
    assert asdict(plan)["checkpoint_steps"] == (128, 256, 384, 512)


@pytest.mark.parametrize(
    ("prefix", "response", "valid"),
    [(1256, 162, True), (1280, 255, True), (1281, 10, False), (10, 256, False)],
)
def test_fit_encoder_preserves_long_rows_with_separate_reviewed_envelope(
    monkeypatch, prefix, response, valid
):
    from types import SimpleNamespace

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.models import prompt_protocol

    example = ProofGraphTask().generate_pair(70_000_042, {"depth": 3})[0]
    target = render_target(example)
    calls = []

    class Tokenizer:
        eos_token_id = 999

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            calls.append(text)
            assert text in {"formatted prompt", target}
            return [11] * prefix if text == "formatted prompt" else [22] * response

    monkeypatch.setattr(
        prompt_protocol,
        "format_model_prompt",
        lambda *_: SimpleNamespace(
            model_facing_prompt="formatted prompt",
            model_facing_prompt_sha256="a" * 64,
        ),
    )
    if valid:
        row = fit.encode_fit_example(example, Tokenizer(), {})
        assert len(row["input_ids"]) == prefix + response + 1
        assert row["labels"] == [-100] * prefix + [22] * response + [999]
        assert row["prefix_length"] == prefix and row["response_length"] == response + 1
    else:
        with pytest.raises(ValueError, match="1280/256/1536"):
            fit.encode_fit_example(example, Tokenizer(), {})
    assert calls == ["formatted prompt", target]


def test_full_fit_data_have_exact_roles_without_double_dev_seed_offset():
    import yaml

    from posttrain_circuits.learning.teacher.adaptation import isolation_inventory

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/task/proofgraph_main.yaml").read_text()
    )
    result = fit.make_fit_examples(config, fit.FitPlan())
    assert len(result["teacher_fit"]) == 8192 and len(result["teacher_dev"]) == 512
    fit_identity = isolation_inventory({"fit": result["teacher_fit"]})
    dev_identity = isolation_inventory({"dev": result["teacher_dev"]})
    assert min(fit_identity["pair_seed"]) == 70_000_042
    assert min(dev_identity["pair_seed"]) == 80_000_042
    assert all(fit_identity[field].isdisjoint(dev_identity[field]) for field in fit_identity)
    assert sum(row.label for row in result["teacher_fit"]) == 4096
    assert sum(row.label for row in result["teacher_dev"]) == 256


def test_one_failed_rank_forward_aborts_before_any_optimizer_update(tmp_path):
    output, results = run_case(tmp_path, "forward-failure", case="bad_forward")
    assert all(not row["ok"] and "rank-local forward failure" in row["error"] for row in results)
    assert (output / "train-metrics.jsonl").read_text() == ""
    assert not (output / "checkpoints").exists()
