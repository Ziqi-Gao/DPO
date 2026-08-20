"""Model-generated ProofGraph evaluation callbacks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizerBase

from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.metrics import aggregate_verification
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample


def build_proofgraph_evaluator(
    examples: list[TaskExample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_completion_length: int,
    model_config: dict[str, object] | None = None,
) -> Callable[[torch.nn.Module], dict[str, float]]:
    if not examples:
        raise ValueError("evaluation requires at least one ProofGraph example")
    if max_completion_length < 1:
        raise ValueError("max_completion_length must be positive")
    task = ProofGraphTask()

    @torch.no_grad()
    def evaluate(model: torch.nn.Module) -> dict[str, float]:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        results = []
        try:
            for example in examples:
                prompt = format_model_prompt(
                    task.render(example), tokenizer, model_config
                ).model_facing_prompt
                encoded = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)
                maximum_positions = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0))
                available = (
                    maximum_positions - encoded.shape[1] if maximum_positions else max_completion_length
                )
                new_tokens = max(1, min(max_completion_length, available))
                generate = getattr(model, "generate", None)
                if not callable(generate):
                    raise TypeError("evaluation model must provide Hugging Face generate()")
                generated = generate(
                    input_ids=encoded,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
                response_ids = generated[0, encoded.shape[1] :].tolist()
                response_text = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                )
                results.append(task.verify(example, task.parse_response(response_text)))
        finally:
            model.train(was_training)
        metrics = aggregate_verification(results)
        return {
            "validation_accuracy": metrics["answer_accuracy"],
            "exact_proof_accuracy": metrics["exact_proof_accuracy"],
            "format_validity": metrics["format_validity"],
        }

    return evaluate
