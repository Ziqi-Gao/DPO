"""Readiness compares a whole target under each context with its own history."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.cli.evaluate_teacher_readiness import _prefix_scores


class ConditionalTableModel(torch.nn.Module):
    """A deterministic causal CPU model, with different next-token histories."""

    def __init__(self, probabilities):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.probabilities = probabilities
        self.inputs = []

    def forward(self, input_ids):
        self.inputs.append(tuple(input_ids[0].tolist()))
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float64)
        for batch, row in enumerate(input_ids.tolist()):
            for position in range(len(row)):
                probabilities = self.probabilities.get(tuple(row[: position + 1]))
                if probabilities is not None:
                    logits[batch, position] = float("-inf")
                    for token, probability in probabilities.items():
                        logits[batch, position, token] = math.log(probability)
        return SimpleNamespace(logits=logits)


def make_spec(clean_context, corrupt_context, clean_target, corrupt_target):
    # CircuitProbeSpec production inputs are shape aligned. This small scorer
    # fixture also varies context length to detect accidental position reuse.
    return SimpleNamespace(
        probe_id="conditional-target-fixture",
        stage="first_rule_selection",
        clean_input_ids=clean_context + clean_target[:-1],
        corrupt_input_ids=corrupt_context + corrupt_target[:-1],
        clean_target_ids=clean_target,
        corrupt_target_ids=corrupt_target,
        clean_metric_positions=tuple(
            range(len(clean_context) - 1, len(clean_context) - 1 + len(clean_target))
        ),
        corrupt_metric_positions=tuple(
            range(len(corrupt_context) - 1, len(corrupt_context) - 1 + len(corrupt_target))
        ),
        clean_context="clean context",
        corrupt_context="counterfactual context",
    )


@pytest.mark.parametrize("corrupt_context", [(1,), (1, 7)])
def test_multitoken_causal_shift_and_alternative_use_same_target_history(corrupt_context):
    clean_context = (0,)
    model = ConditionalTableModel({
        clean_context: {2: .6, 4: .4},
        corrupt_context: {2: .4, 4: .6},
        (*clean_context, 2): {3: .6, 5: .4},
        (*corrupt_context, 4): {3: .95, 5: .05},
        (*corrupt_context, 2): {3: .8, 5: .2},
        (*clean_context, 4): {3: .01, 5: .99},
    })
    scores = _prefix_scores(model, make_spec(clean_context, corrupt_context, (2, 3), (4, 5)), top_k=2)
    clean, corrupt = scores
    # The former implementation uses P(3 | corrupt context, 4), making this
    # log(.36/.38) < 0. The specified SAME target instead needs history token 2.
    assert clean.causal_shift_logprob == pytest.approx(math.log(.36 / .32), abs=1e-6)
    assert clean.causal_shift_valid is True
    assert clean.target_log_probability == pytest.approx(math.log(.36), abs=1e-6)
    assert clean.alternative_log_probability == pytest.approx(math.log(.396), abs=1e-6)
    assert clean.target_logprob_margin == pytest.approx(math.log(.36 / .396), abs=1e-6)
    assert corrupt.causal_shift_logprob == pytest.approx(math.log(.03 / .396), abs=1e-6)
    assert corrupt.alternative_log_probability == pytest.approx(math.log(.32), abs=1e-6)
    assert clean.top1_correct is True and corrupt.top1_correct is False
    assert all(score.target_in_topk and score.minimum_topk_mass == 1.0 for score in scores)
    assert set(model.inputs) == {
        (*clean_context, 2), (*corrupt_context, 4),
        (*clean_context, 4), (*corrupt_context, 2),
    }


def test_full_three_token_targets_condition_every_later_token_on_that_target():
    model = ConditionalTableModel({
        (0,): {2: .6, 4: .4}, (1,): {2: .4, 4: .6},
        (0, 2): {3: .6, 5: .4}, (1, 4): {3: .95, 5: .05},
        (1, 2): {3: .8, 5: .2}, (0, 4): {3: .01, 5: .99},
        (0, 2, 3): {6: .9, 7: .1}, (1, 2, 3): {6: .2, 7: .8},
        (1, 4, 5): {6: .7, 7: .3}, (0, 4, 5): {6: .9, 7: .1},
    })
    clean, corrupt = _prefix_scores(model, make_spec((0,), (1,), (2, 3, 6), (4, 5, 7)), top_k=2)
    assert clean.target_log_probability == pytest.approx(math.log(.6 * .6 * .9), abs=1e-6)
    assert clean.alternative_log_probability == pytest.approx(math.log(.4 * .99 * .1), abs=1e-6)
    assert clean.causal_shift_logprob == pytest.approx(math.log((.6 * .6 * .9) / (.4 * .8 * .2)), abs=1e-6)
    assert corrupt.causal_shift_logprob == pytest.approx(
        math.log((.6 * .05 * .3) / (.4 * .99 * .1)), abs=1e-6
    )
    assert set(model.inputs) == {(0, 2, 3), (1, 4, 5), (0, 4, 5), (1, 2, 3)}


@pytest.mark.parametrize("shared_target_prefix", [(), (2,)])
def test_single_token_and_shared_target_history_keep_original_likelihoods(shared_target_prefix):
    probabilities = {(0,): {2: 1.0}, (1,): {2: 1.0}} if shared_target_prefix else {}
    probabilities.update({
        (0, *shared_target_prefix): {3: .8, 5: .2},
        (1, *shared_target_prefix): {3: .3, 5: .7},
    })
    model = ConditionalTableModel(probabilities)
    clean, corrupt = _prefix_scores(
        model,
        make_spec((0,), (1,), (*shared_target_prefix, 3), (*shared_target_prefix, 5)),
        top_k=2,
    )
    assert clean.causal_shift_logprob == pytest.approx(math.log(.8 / .3), abs=1e-6)
    assert corrupt.causal_shift_logprob == pytest.approx(math.log(.7 / .2), abs=1e-6)
    assert clean.alternative_log_probability == pytest.approx(math.log(.2), abs=1e-6)
    assert corrupt.alternative_log_probability == pytest.approx(math.log(.3), abs=1e-6)
    assert clean.top1_correct and corrupt.top1_correct
    assert clean.target_in_topk and corrupt.target_in_topk
