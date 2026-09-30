"""Independent real-model CPU reference for diagnostic canonical response NLL."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import pytest
import torch

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "student_quality_nll_test", ROOT / "tools/sdsc_student_quality_probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canonical_nll_matches_hf_labels_loss_on_real_tiny_qwen(dtype):
    """HF's independent labels path must include the first response and EOS.

    Different target lengths distinguish token-weighting from an accidental
    example mean. Both models and tokenizers are entirely local random fixtures.
    """
    task = ProofGraphTask()
    examples = [
        task.generate(91, {"depth": 1, "distractors": 0, "structure": "chain"}),
        task.generate(92, {"depth": 3, "distractors": 0, "structure": "chain"}),
    ]
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen3(89).to(dtype).eval()
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        amp = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if dtype == torch.bfloat16 else contextlib.nullcontext()
        )
        lengths, nll_sums, correct = [], [], 0
        with torch.no_grad(), amp:
            actual = worker().teacher_forced_metrics(model, tokenizer, examples, {})
            for example in examples:
                prefix = tokenizer.encode(
                    format_model_prompt(task.render(example), tokenizer, {}).model_facing_prompt,
                    add_special_tokens=False,
                )
                response = tokenizer.encode(render_target(example), add_special_tokens=False)
                response.append(tokenizer.eos_token_id)
                ids = torch.tensor([prefix + response])
                labels = torch.full_like(ids, -100)
                labels[:, len(prefix):] = ids[:, len(prefix):]
                output = model(input_ids=ids, labels=labels)
                nll_sums.append(float(output.loss) * len(response))
                lengths.append(len(response))
                predictions = output.logits[0, len(prefix) - 1:-1].float().argmax(-1)
                correct += int((predictions == torch.tensor(response)).sum())
        assert len(set(lengths)) == 2
        assert actual["examples"] == 2
        assert actual["response_tokens_including_eos"] == sum(lengths)
        assert actual["canonical_response_nll"] == pytest.approx(
            sum(nll_sums) / sum(lengths), rel=2e-7, abs=2e-7
        )
        assert actual["canonical_response_token_accuracy"] == correct / sum(lengths)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(
            torch.equal(initial_parameters[name], parameter)
            for name, parameter in model.named_parameters()
        )
        assert not model.training
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(previous_threads)


def test_canonical_nll_rejects_real_model_nonfinite_logits():
    task = ProofGraphTask()
    example = task.generate(94, {"depth": 1, "distractors": 0, "structure": "chain"})
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen3(93).eval()
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError, match="nonfinite diagnostic canonical NLL"):
            worker().teacher_forced_metrics(model, tokenizer, [example], {})
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(previous_threads)
