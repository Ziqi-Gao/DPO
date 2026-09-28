"""Exercise CPU collation with a device-moving forward without allocating CUDA."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import TrajectoryBatch
from posttrain_circuits.learning.supervision import verified_replay
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor


def trajectories(rewards=(1.0, 1.0)):
    records = []
    for index, reward in enumerate(rewards):
        response = [3, 4][: 1 + index % 2]
        record = TrajectoryRecord(
            trajectory_id="",
            prompt_id=f"p{index}",
            split="train",
            prompt_text="fixture",
            input_ids=[1, 2],
            response_ids=response,
            response_text="fixture",
            response_token_mask=[True] * len(response),
            behavior_policy_id="fixture",
            behavior_policy_revision="fixture",
            policy_version=0,
            sampling_request_seed=42,
            actual_sampling_seed=index,
            sampling_cursor_id=f"cursor-{index}",
            sampling_protocol_id="device-regression-fixture",
            sampling_temperature=1.0,
            top_p=1.0,
            behavior_logprobs=[0.0] * len(response),
            verifier_reward=reward,
        )
        record.trajectory_id = record.expected_trajectory_id
        records.append(record)
    return TrajectoryBatch(records, 0)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_collation_to_canonical_supervisor_aligns_real_cross_entropy_devices(monkeypatch, dtype):
    supervisor = CanonicalSFTSupervisor(0)
    batch = supervisor.prepare_targets(trajectories(), None, None)
    original = {key: getattr(batch, key).clone() for key in ("input_ids", "response_mask", "rewards")}
    assert all(value.device.type == "cpu" for value in original.values())
    forward_calls = []
    observed = []
    cuda_before = torch.cuda.is_initialized()

    def forward(**kwargs):
        # FSDP moves a copy of model kwargs. It does not mutate the CPU batch.
        forward_calls.append({key: value.device.type for key, value in kwargs.items()})
        return SimpleNamespace(logits=torch.empty((*batch.input_ids.shape, 7), device="meta", dtype=dtype))

    def cross_entropy_boundary(logits, token_ids, response_mask, rewards, *, normalization):
        # Meta dispatch checks the same cross-entropy device invariant without
        # any CUDA driver access. This also executes the real causal shift.
        nll = functional.cross_entropy(
            logits[:, :-1].reshape(-1, 7), token_ids[:, 1:].reshape(-1), reduction="none"
        )
        assert nll.device == logits.device
        assert response_mask.device == rewards.device == token_ids.device == logits.device
        assert (token_ids.dtype, response_mask.dtype, rewards.dtype) == (
            torch.long,
            torch.bool,
            torch.float32,
        )
        assert normalization == "sequence"
        observed.append(str(logits.device))
        # Meta tensors cannot supply data-dependent metric scalar values. Only
        # replace that terminal scalar; numerical loss/gradient tests run below.
        return torch.tensor(1.25)

    monkeypatch.setattr(verified_replay, "verified_replay_loss", cross_entropy_boundary)
    output = supervisor.compute_loss(forward, batch)
    assert output.metrics["verified_replay_loss"] == 1.25
    assert forward_calls == [{"input_ids": "cpu", "attention_mask": "cpu"}]
    assert observed == ["meta"]
    for key, value in original.items():
        assert getattr(batch, key).device.type == "cpu"
        assert torch.equal(getattr(batch, key), value)
    assert torch.cuda.is_initialized() == cuda_before


@pytest.mark.unit
@pytest.mark.parametrize("normalization", ["sequence", "token"])
@pytest.mark.parametrize("canonical", [False, True])
def test_device_alignment_preserves_loss_gradients_masks_metadata_and_rng(normalization, canonical):
    supervisor_type = CanonicalSFTSupervisor if canonical else verified_replay.VerifiedReplaySupervisor
    supervisor = supervisor_type(0, normalization=normalization)
    batch = supervisor.prepare_targets(trajectories((1.0, 1.0) if canonical else (1.0, 1.0, 0.0)), None, None)
    shape = (*batch.input_ids.shape, 7)
    logits = (torch.arange(torch.tensor(shape).prod()).reshape(shape).float() / 17).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    before = torch.get_rng_state().clone()
    expected = verified_replay.verified_replay_loss(
        reference, batch.input_ids, batch.response_mask, batch.rewards, normalization=normalization
    )
    actual = supervisor.compute_loss(lambda **_: SimpleNamespace(logits=logits), batch)
    expected.backward()
    actual.loss.backward()
    assert torch.equal(actual.loss, expected)
    assert torch.equal(logits.grad, reference.grad)
    assert torch.equal(torch.get_rng_state(), before)
    assert actual.metrics == {"verified_replay_loss": float(expected.detach()), **batch.metadata}
    # Prompt/padding and rejected trajectories contribute no gradient.
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    if not canonical:
        assert torch.count_nonzero(logits.grad[-1]) == 0
