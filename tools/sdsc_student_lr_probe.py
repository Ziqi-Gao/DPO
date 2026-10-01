#!/usr/bin/env python3
"""Frozen, train-only four-update LR diagnosis; never model/G0 acceptance.

The production trainer owns FSDP preparation, collection, loss, accumulation,
AdamW stepping and scheduler cadence. This separate worker bounds the run and
adds measurements; it never calls the formal evaluator or publishes acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCHEMA = "quest-sdsc-student-lr-probe-v1"
INPUT_SCHEMA = "quest-sdsc-student-lr-inputs-v1"
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"
INITIAL_SIZE = 3441276375
CONFIG_SHA = "05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe"
FAMILY_SHA = "bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189"
TRAIN_SHA = "377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b"
INSTRUCTION_SHA = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
TOKENIZER_SHA = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
STORE_FILE_SHA = "8dc9928dae74340366b07fc385d0eeb046ddfd288859ef7ac5150fc0436c7a41"
STORE_SHA = "f2e9e8171e3289bfd8bb356c79df67e56612f2ad067c63a4583b017c79025a88"
ATTEMPTS_SHA = "42e57c6ff09a38b2570c25502428baef885e5d1ff78cbd18adf2f67852b47c60"
ARMS = (("lr-5e-4", 5e-4), ("lr-5e-5", 5e-5))
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "teacher_accepted",
    "teacher_accepted_under_candidate",
    "accepted_science",
    "formal_prompt_accepted",
)
FALSE = {key: False for key in FLAGS}
RAW_NAMES = ("lr-batches.jsonl", "lr-prompts.jsonl", "lr-records.jsonl")
CHUNK = 1 << 18


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def same(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def document(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    require(isinstance(value, dict), "JSON object required")
    canonical(value)
    return value


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def confined(path, work, *, directory=False):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute local path required")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink rejected")
    require(
        work in path.resolve().parents and (path.is_dir() if directory else path.is_file()),
        "input must be within node-local work",
    )
    return path


def verified(record, work, expected_sha=None, expected_size=None):
    require(
        isinstance(record, dict) and set(record) == {"path", "sha256", "size"}, "file record fields differ"
    )
    path = confined(record["path"], work)
    require(
        type(record["size"]) is int and record["size"] > 0 and path.stat().st_size == record["size"],
        "input size/type differs",
    )
    require(file_sha(path) == record["sha256"] == (expected_sha or record["sha256"]), "input SHA differs")
    require(expected_size is None or record["size"] == expected_size, "fixed input size differs")
    return path


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_lr_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_phase(operation, *, rank, world_size):
    """Align local failures before the next collective, without hiding its cause.

    Never wrap a collective operation in this helper: an exception inside FSDP
    must escape immediately so the launcher terminates the other rank. No
    exception/finally path enters a barrier.
    """
    import torch

    value, error = None, None
    try:
        value = operation()
    except Exception as exc:
        error = {"rank": rank, "type": type(exc).__name__, "message": str(exc)}
    errors = [error]
    if world_size > 1:
        errors = [None] * world_size
        torch.distributed.all_gather_object(errors, error)
    require(not any(errors), "rank-local phase failed: " + json.dumps(errors))
    return value


@contextlib.contextmanager
def measurement(model, trainer=None):
    """Forward-only measurements preserve every RNG, mode and training cursor."""
    from posttrain_circuits.core.seeding import RNGState

    rng = RNGState.capture()
    flags = [(module, module.training) for module in model.modules()]
    before = None
    if trainer is not None:
        before = canonical(
            {
                "source": trainer.state_source.state_dict(),
                "prompts": trainer.prompt_scheduler.state_dict(),
                "budget": trainer.token_budget.state_dict(),
                "step": trainer.global_step,
                "cumulative": trainer.cumulative_counts,
                "optimizer": optimizer_observation(getattr(trainer, "optimizer", None)),
                "scheduler": trainer.scheduler.state_dict() if hasattr(trainer, "scheduler") else None,
            }
        )
    try:
        model.eval()
        yield
    finally:
        for module, flag in flags:
            module.training = flag
        rng.restore()
        if trainer is not None:
            after = canonical(
                {
                    "source": trainer.state_source.state_dict(),
                    "prompts": trainer.prompt_scheduler.state_dict(),
                    "budget": trainer.token_budget.state_dict(),
                    "step": trainer.global_step,
                    "cumulative": trainer.cumulative_counts,
                    "optimizer": optimizer_observation(getattr(trainer, "optimizer", None)),
                    "scheduler": trainer.scheduler.state_dict() if hasattr(trainer, "scheduler") else None,
                }
            )
            require(after == before, "diagnostic measurement advanced training state")


def optimizer_observation(optimizer):
    """Detect mutation of Adam state without copying full-sized moment tensors."""
    if optimizer is None:
        return None
    import torch

    groups = [
        {key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups
    ]
    states = []
    for parameter, state in optimizer.state.items():
        values = {}
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                values[name] = dict(
                    identity=id(value),
                    mutation_version=value._version,
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    device=str(value.device),
                )
            else:
                values[name] = value
        states.append(dict(parameter_identity=id(parameter), state=values))
    return dict(
        groups=groups,
        parameter_groups=[[id(p) for p in group["params"]] for group in optimizer.param_groups],
        states=states,
    )


def response_labels(input_ids, attention_mask, response_mask):
    import torch

    require(
        input_ids.ndim == 2 and input_ids.shape == attention_mask.shape == response_mask.shape,
        "batch shapes differ",
    )
    require(
        input_ids.dtype == torch.long and attention_mask.dtype == response_mask.dtype == torch.bool,
        "batch dtype differs",
    )
    require(
        not bool((response_mask & ~attention_mask).any()) and not bool(response_mask[:, 0].any()),
        "response mask includes padding or unpredicted first position",
    )
    require(bool(response_mask.any(dim=1).all()), "empty response supervision")
    labels = input_ids.clone()
    labels[~response_mask] = -100
    return labels


def teacher_forced_rows(model, rows):
    """Actual HF labels CE per sequence vs original sequence-normalized loss.

    HF's single multi-sequence loss is a token mean and is intentionally not
    compared to the production mean-of-sequence-means. Each row is scored once.
    """
    import torch

    from posttrain_circuits.learning.supervision.losses import verified_replay_loss

    device = next(model.parameters()).device
    results = []
    with measurement(model), torch.no_grad():
        for row in rows:
            prefix, response = row["input_ids"], row["response_ids"]
            require(prefix and response and len(prefix) + len(response) <= 1536, "NLL sequence envelope")
            ids = torch.tensor([[*prefix, *response]], device=device, dtype=torch.long)
            attention = torch.ones_like(ids, dtype=torch.bool)
            mask = torch.zeros_like(attention)
            mask[:, len(prefix) :] = True
            labels = response_labels(ids, attention, mask)
            output = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False)
            logits = output.logits.float()
            production = verified_replay_loss(
                logits, ids, mask, torch.ones(1, device=device), normalization="sequence"
            )
            hf = output.loss.float()
            require(
                bool(torch.isfinite(production)) and bool(torch.isfinite(hf)), "nonfinite teacher-forced loss"
            )
            error = abs(float(production) - float(hf))
            tolerance = 2e-5 * max(1.0, abs(float(hf)))
            require(error <= tolerance, "actual HF labels CE differs from production sequence loss")
            target_logits = logits[:, len(prefix) - 1 : -1]
            correct = int((target_logits.argmax(-1) == ids[:, len(prefix) :]).sum())
            results.append(
                dict(
                    prompt_id=row["prompt_id"],
                    response_tokens=len(response),
                    hf_sequence_nll=float(hf),
                    production_sequence_nll=float(production),
                    labels_ce_abs_error=error,
                    correct_tokens=correct,
                )
            )
    return results


def aggregate_nll(rows):
    require(rows and len({row["prompt_id"] for row in rows}) == len(rows), "NLL cohort incomplete/duplicated")
    tokens = sum(row["response_tokens"] for row in rows)
    return dict(
        examples=len(rows),
        response_tokens_including_eos=tokens,
        sequence_mean_nll=sum(row["production_sequence_nll"] for row in rows) / len(rows),
        hf_sequence_mean_nll=sum(row["hf_sequence_nll"] for row in rows) / len(rows),
        token_mean_nll=sum(row["hf_sequence_nll"] * row["response_tokens"] for row in rows) / tokens,
        token_accuracy=sum(row["correct_tokens"] for row in rows) / tokens,
        max_labels_ce_abs_error=max(row["labels_ce_abs_error"] for row in rows),
        per_example=rows,
    )


def first_step_statistics(before, gradient, after, *, learning_rate):
    """FP64 mathematical AdamW prediction, with a predeclared FP32 roundoff bound.

    At fresh step one, WD=0, bias correction reduces AdamW to
    theta-lr*g/(abs(g)+eps). A 32*float32-eps*max(|old|,|pred|,lr)
    coordinate bound is an engineering consistency check, never a quality gate.
    """
    import torch

    require(
        before.dtype == gradient.dtype == after.dtype == torch.float32, "first-step master/gradient dtype"
    )
    require(before.shape == gradient.shape == after.shape, "first-step parameter shape")
    require(learning_rate in (5e-4, 5e-5), "unreviewed learning rate")
    out = dict(
        elements=0,
        gradient_squared=0.0,
        update_squared=0.0,
        gradient_dot_update=0.0,
        residual_squared=0.0,
        max_abs_residual=0.0,
        max_roundoff_ratio=0.0,
        roundoff_violations=0,
        gradient_dot_roundoff_bound=0.0,
    )
    flat = (value.detach().reshape(-1) for value in (before, gradient, after))
    old, grad, new = flat
    for start in range(0, old.numel(), CHUNK):
        a, g, b = (value[start : start + CHUNK].cpu().double() for value in (old, grad, new))
        require(all(bool(torch.isfinite(value).all()) for value in (a, g, b)), "nonfinite first-step tensors")
        predicted = a - learning_rate * g / (g.abs() + 1e-8)
        delta, residual = b - a, b - predicted
        bound = (
            32
            * torch.finfo(torch.float32).eps
            * torch.maximum(torch.maximum(a.abs(), predicted.abs()), torch.full_like(a, learning_rate))
        )
        out["elements"] += a.numel()
        out["gradient_squared"] += float(g.square().sum())
        out["update_squared"] += float(delta.square().sum())
        out["gradient_dot_update"] += float((g * delta).sum())
        out["residual_squared"] += float(residual.square().sum())
        out["gradient_dot_roundoff_bound"] += float((g.abs() * bound).sum())
        out["max_abs_residual"] = max(out["max_abs_residual"], float(residual.abs().max()))
        out["max_roundoff_ratio"] = max(out["max_roundoff_ratio"], float((residual.abs() / bound).max()))
        out["roundoff_violations"] += int((residual.abs() > bound).sum())
    return out


class AdamWObserver:
    """Observe the actual prepared raw optimizer, without changing its math."""

    def __init__(self, optimizer, learning_rate, before_parameters):
        import torch

        require(type(optimizer) is torch.optim.AdamW and not optimizer.state, "fresh raw AdamW required")
        require(len(optimizer.param_groups) == 1, "one AdamW parameter group required")
        self.optimizer, self.learning_rate = optimizer, learning_rate
        self.before_parameters = before_parameters
        self.calls, self.rows, self.saved = 0, [], []
        self.handles = [
            optimizer.register_step_pre_hook(self.pre),
            optimizer.register_step_post_hook(self.post),
        ]

    def pre(self, optimizer, args, kwargs):
        import torch

        del args, kwargs
        group = optimizer.param_groups[0]
        require(
            group["lr"] == self.learning_rate
            and group["betas"] == (0.9, 0.95)
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0.0
            and not group["amsgrad"]
            and not group["maximize"],
            "AdamW scientific options changed",
        )
        require(self.calls < 4, "unexpected extra optimizer step")
        self.saved = []
        originals = self.before_parameters() if self.calls == 0 else [None] * len(group["params"])
        require(len(originals) == len(group["params"]), "original trainer update baseline missing")
        for parameter, original in zip(group["params"], originals, strict=True):
            require(
                parameter.dtype == torch.float32
                and parameter.grad is not None
                and parameter.grad.dtype == torch.float32
                and bool(torch.isfinite(parameter.grad).all()),
                "missing/nonfinite/incorrectly typed actual reduced gradient",
            )
            state = optimizer.state.get(parameter, {})
            require(
                (not state and self.calls == 0) or float(state.get("step", -1)) == self.calls,
                "AdamW state is not at the expected fresh step",
            )
            if self.calls == 0:
                require(
                    original.device.type == "cpu"
                    and original.dtype == torch.float32
                    and original.shape == parameter.shape,
                    "original trainer baseline differs",
                )
                # Retain references only. AdamW leaves .grad intact through its
                # post hook; the production trainer zeros it after we return.
                self.saved.append((parameter, original, parameter.grad, parameter.grad._version))

    def post(self, optimizer, args, kwargs):
        del args, kwargs
        self.calls += 1
        steps = [
            float(optimizer.state[parameter]["step"]) for parameter in optimizer.param_groups[0]["params"]
        ]
        require(steps and all(step == self.calls for step in steps), "actual AdamW step count differs")
        row = dict(step=self.calls, parameter_tensors=len(steps), all_parameter_steps_equal=True)
        if self.calls == 1:
            combined = None
            for parameter, original, gradient, version in self.saved:
                require(
                    parameter.grad is gradient and gradient._version == version,
                    "AdamW mutated the captured gradient before observation",
                )
                stats = first_step_statistics(original, gradient, parameter, learning_rate=self.learning_rate)
                if combined is None:
                    combined = stats
                else:
                    for key, value in stats.items():
                        combined[key] = (
                            max(combined[key], value) if key.startswith("max_") else combined[key] + value
                        )
            require(
                combined is not None and combined["gradient_squared"] > 0 and combined["update_squared"] > 0,
                "first step has no gradient/update",
            )
            require(
                combined["roundoff_violations"] == 0
                and combined["gradient_dot_update"] <= combined["gradient_dot_roundoff_bound"],
                "first AdamW update differs from its mathematical prediction",
            )
            row["first_step_prediction"] = combined
        self.saved = []
        self.rows.append(row)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.saved = []


def strict_load_export(model, state):
    import torch

    require(isinstance(state, dict) and state, "exported state missing")
    model.to(device="cpu", dtype=torch.float32)
    current = model.state_dict()
    require(current.keys() == state.keys(), "exported state keys differ")
    for name, tensor in state.items():
        require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.shape == current[name].shape
            and tensor.dtype == current[name].dtype
            and (
                not tensor.is_floating_point()
                or (tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()))
            ),
            "exported master state precision/shape differs: " + name,
        )
    model.load_state_dict(state, strict=True)
    require(
        all(torch.equal(tensor, state[name]) for name, tensor in model.state_dict().items()),
        "FP32 exported state did not load exactly",
    )
    return dict(all_tensors_exact=True, key_count=len(state), loaded_before_bf16_copy=True)


def difference_statistics(reference, actual):
    import torch

    require(
        reference.shape == actual.shape and reference.ndim >= 2 and reference.numel() > 0,
        "parity logits shape differs",
    )
    count, squared, maximum = 0, 0.0, 0.0
    left, right = reference.detach().reshape(-1), actual.detach().reshape(-1)
    for start in range(0, left.numel(), CHUNK):
        a, b = (value[start : start + CHUNK].cpu().double() for value in (left, right))
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "nonfinite parity logits")
        delta = a - b
        count += delta.numel()
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return dict(
        shape=list(reference.shape),
        elements=count,
        max_abs_error=maximum,
        rms_error=math.sqrt(squared / count),
        argmax_mismatches=int((reference.argmax(-1).cpu() != actual.argmax(-1).cpu()).sum()),
        compared_positions=reference.numel() // reference.shape[-1],
        bitwise_equal=bool(torch.equal(reference.cpu(), actual.cpu())),
        formal_parity_gate_applied=False,
    )


def window_row(record, slot):
    return dict(
        global_slot=slot,
        prompt_id=record.prompt_id,
        attempt_id=record.sampling_cursor_id,
        input_ids=list(record.input_ids),
        response_ids=list(record.response_ids),
        response_token_mask=list(record.response_token_mask),
    )


def batch_capture(trajectories, supervision, *, rank, microstep, slots, pad_token_id):
    """Compare actual production tensors with independently assembled spans."""
    import torch

    require(len(trajectories.records) == len(slots) == 4, "fixed microbatch size differs")
    labels = response_labels(supervision.input_ids, supervision.attention_mask, supervision.response_mask)
    width = max(len(record.input_ids) + len(record.response_ids) for record in trajectories.records)
    require(list(supervision.input_ids.shape) == [4, width], "unexpected collated padding extent")
    rows = []
    for index, (record, slot) in enumerate(zip(trajectories.records, slots, strict=True)):
        prefix, response = record.input_ids, record.response_ids
        require(
            record.verifier_reward == 1.0 and record.response_token_mask == [True] * len(response),
            "unaccepted/masked teacher response",
        )
        length = len(prefix) + len(response)
        same(
            supervision.input_ids[index].tolist(),
            [*prefix, *response, *([pad_token_id] * (width - length))],
            "actual training input differs from selected teacher record",
        )
        same(
            supervision.attention_mask[index].tolist(),
            [True] * length + [False] * (width - length),
            "actual attention mask differs",
        )
        same(
            supervision.response_mask[index].tolist(),
            [False] * len(prefix) + [True] * len(response) + [False] * (width - length),
            "actual response mask differs",
        )
        same(
            labels[index].tolist(),
            [-100] * len(prefix) + response + [-100] * (width - length),
            "actual labels differ",
        )
        rows.append(window_row(record, slot))
    require(
        supervision.rewards.dtype == torch.float32 and bool((supervision.rewards == 1).all()),
        "actual reward gate differs",
    )
    return dict(
        rank=rank,
        microstep=microstep,
        global_slots=list(slots),
        records=rows,
        input_ids=supervision.input_ids.tolist(),
        attention_mask=supervision.attention_mask.tolist(),
        response_mask=supervision.response_mask.tolist(),
        labels=labels.tolist(),
    )


def gather(value, world_size=2):
    import torch

    values = [None] * world_size
    torch.distributed.all_gather_object(values, value)
    return values


def distributed_nll(trainer, rows, rank):
    with measurement(trainer.model, trainer):
        local = teacher_forced_rows(trainer.model, rows[rank::2])
        combined = [row for member in gather(local) for row in member]
    by_id = {row["prompt_id"]: row for row in combined}
    return aggregate_nll([by_id[row["prompt_id"]] for row in rows])


def validate_data_audit(inputs, work):
    spec = inputs["data_audit"]
    require(isinstance(spec, dict) and set(spec) == {"report", "capture"}, "data audit input schema")
    paths = {}
    for name in ("report", "capture"):
        path = verified(spec[name], work)
        require(path.stat().st_size <= 2 * 1024**2, "data audit exceeds bounded input size")
        paths[name] = path
    report, capture = (document(paths[name]) for name in ("report", "capture"))
    same(report["schema"], "quest-sdsc-student-lr-data-audit-v1", "independent data audit schema")
    require(
        report["passed"] is True
        and report["diagnostic_complete"] is True
        and all(report.get(key) is False for key in FLAGS),
        "independent data audit not admitted",
    )
    for key, value in dict(
        parent_job_id="54548846",
        initial_checkpoint_sha256=INITIAL_SHA,
        teacher_store_manifest_sha256=STORE_FILE_SHA,
        teacher_store_sha256=STORE_SHA,
        teacher_attempts_sha256=ATTEMPTS_SHA,
        tokenizer_fingerprint=TOKENIZER_SHA,
        baseline_instruction_sha256=INSTRUCTION_SHA,
        capture_sha256=spec["capture"]["sha256"],
    ).items():
        same(report[key], value, "independent data audit binding differs: " + key)
    same(capture["schema"], "quest-sdsc-student-lr-first64-capture-v1", "capture schema differs")
    require(set(capture) == {"schema", "rows"}, "capture fields differ")
    rows = capture["rows"]
    require(isinstance(rows, list) and len(rows) == 64, "first64 capture incomplete")
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row)
            == {"global_slot", "prompt_id", "attempt_id", "input_ids", "response_ids", "response_token_mask"},
            "capture row fields differ",
        )
        for name in ("input_ids", "response_ids"):
            require(
                isinstance(row[name], list)
                and row[name]
                and all(type(value) is int and value >= 0 for value in row[name]),
                "capture token type differs",
            )
        require(
            type(row["global_slot"]) is int
            and type(row["prompt_id"]) is str
            and type(row["attempt_id"]) is str,
            "capture identity type differs",
        )
        same(row["response_token_mask"], [True] * len(row["response_ids"]), "capture masks differ")
        require(
            row["response_ids"][-1] == 151645 and row["response_ids"].count(151645) == 1,
            "capture EOS differs",
        )
    same(report["window_sha256"], digest(canonical(rows)), "first64 semantic identity differs")
    same([row["global_slot"] for row in rows], list(range(64)), "first64 slot order differs")
    same(report["ordered_prompt_ids"], [row["prompt_id"] for row in rows], "audit prompt population differs")
    same(
        report["selected_attempt_ids"], [row["attempt_id"] for row in rows], "audit selected attempts differ"
    )
    same(
        [
            sum(len(row["input_ids"]) + len(row["response_ids"]) for row in rows),
            sum(len(row["response_ids"]) for row in rows),
        ],
        [59336, 5896],
        "original first64 token counts differ",
    )
    return report, rows


def staged_science(inputs, output):
    work = output.parent.resolve()
    require(
        inputs.get("schema") == INPUT_SCHEMA and inputs.get("parent_job_id") == "54548846", "input identity"
    )
    require(
        set(inputs)
        == {
            "schema",
            "parent_job_id",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "resolved_config",
            "initial_checkpoint",
            "teacher_bundle_root",
            "dataset_root",
            "data_audit",
            "hf_home",
            "dataset_manifest_sha256",
        },
        "unexpected LR input fields",
    )
    for name in ("source_code_sha256", "plan_sha256"):
        require(
            isinstance(inputs[name], str) and re.fullmatch("[a-f0-9]{64}", inputs[name]), "invalid input SHA"
        )
    require(inputs["job_id"] == os.environ.get("SLURM_JOB_ID"), "job allocation differs")
    require(inputs["dataset_manifest_sha256"] == FAMILY_SHA, "dataset manifest identity differs")
    cache = confined(inputs["hf_home"], work, directory=True)
    require(str(cache) == os.environ.get("HF_HOME"), "offline cache differs")
    science = confined(inputs["science_root"], work, directory=True)
    # The immutable teacher reader resolves its accepted Git history from cwd.
    # A source-only deployment snapshot must never become that authority.
    require(Path.cwd().resolve() == science.resolve(), "worker cwd must equal verified science root")
    require(
        not any(
            name == "posttrain_circuits" or name.startswith("posttrain_circuits.") for name in sys.modules
        ),
        "science imported before verified root",
    )
    sys.path.insert(0, str(science / "src"))
    return work


def load_inputs(inputs, work):
    import yaml

    from posttrain_circuits.datasets.proofgraph import rendering
    from posttrain_circuits.datasets.proofgraph.serialization import deserialize_example

    config_path = verified(inputs["resolved_config"], work, CONFIG_SHA, 7923)
    checkpoint = verified(inputs["initial_checkpoint"], work, INITIAL_SHA, INITIAL_SIZE)
    config = yaml.safe_load(config_path.read_text())
    same(config["trainer"]["learning_rate"], 0.0005, "original LR configuration differs")
    require(
        config["trainer"]["max_model_input_length"] == 1536
        and config["trainer"]["global_batch_size"] == 64
        and config["trainer"]["max_microbatch_size"] == 4
        and config["trainer"]["weight_decay"] == 0,
        "original trainer contract differs",
    )
    require(
        digest(rendering.RESPONSE_FORMAT_INSTRUCTIONS.encode()) == INSTRUCTION_SHA,
        "original instruction differs",
    )
    data_root = confined(inputs["dataset_root"], work, directory=True)
    manifest, train = (
        confined(str(data_root / name), work) for name in ("manifest.json", "train/examples.jsonl")
    )
    require(
        file_sha(manifest) == FAMILY_SHA and file_sha(train) == TRAIN_SHA, "original train data bytes differ"
    )
    with train.open() as stream:
        rows = [json.loads(line) for line in itertools.islice(stream, 256)]
    examples = [deserialize_example(row) for row in rows]
    require(
        len(examples) == len({example.example_id for example in examples}) == 256,
        "training population incomplete",
    )
    audit_report, captured = validate_data_audit(inputs, work)
    teacher_root = confined(inputs["teacher_bundle_root"], work, directory=True)
    from posttrain_circuits.artifacts.adapted_teacher_sft import read_accepted_teacher_sft

    demonstrations, manifest = read_accepted_teacher_sft(teacher_root, config=config)
    same(
        manifest["ordered_prompt_ids"],
        [example.example_id for example in examples],
        "teacher/train population differs",
    )
    same(
        audit_report["ordered_prompt_ids"],
        [example.example_id for example in examples[:64]],
        "first64 family order differs",
    )
    return config, checkpoint, examples, demonstrations, audit_report, captured


def load_training_initial(model_config, checkpoint):
    import torch

    from posttrain_circuits.artifacts.checkpoints import load_checkpoint_model_state
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(model_config, for_training=True)
    state = load_checkpoint_model_state(checkpoint)
    current = loaded.model.state_dict()
    require(state.keys() == current.keys(), "initial keys differ")
    for name, tensor in state.items():
        require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == current[name].dtype
            and tensor.shape == current[name].shape
            and (
                not tensor.is_floating_point()
                or (tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor).all()))
            ),
            "initial tensor precision/shape",
        )
    loaded.model.load_state_dict(state, strict=True)
    require(
        all(torch.equal(value, state[name]) for name, value in loaded.model.state_dict().items()),
        "initial exact load",
    )
    require(loaded.tokenizer_hash == TOKENIZER_SHA, "initial tokenizer differs")
    return loaded, dict(
        checkpoint_sha256=INITIAL_SHA,
        all_tensors_exact=True,
        key_count=len(state),
        all_parameters_trainable=all(p.requires_grad for p in loaded.model.parameters()),
    )


def prepare_prompts(examples, tokenizer, model_config):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    result, canonical_rows = [], []
    task = ProofGraphTask()
    for ordinal, example in enumerate(examples[:32]):
        raw = task.render(example)
        text = format_model_prompt(raw, tokenizer, model_config).model_facing_prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(len(ids) + 256 <= 1536, "full generation allowance does not fit")
        result.append(
            dict(
                ordinal=ordinal,
                example=asdict(example),
                raw_prompt=raw,
                prompt_text=text,
                prompt_ids=ids,
                prompt_sha256=digest(text.encode()),
                prompt_token_sha256=digest(canonical(ids)),
            )
        )
        target = [
            *tokenizer.encode(task.canonical_target(example), add_special_tokens=False),
            tokenizer.eos_token_id,
        ]
        require(len(ids) + len(target) <= 1536, "canonical scoring target does not fit")
        canonical_rows.append(dict(prompt_id=example.example_id, input_ids=ids, response_ids=target))
    return result, canonical_rows


def build_trainer(loaded, demonstrations, examples, original_config, checkpoint, lr, rank, directory):
    import torch

    from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
    from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
    from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
    from posttrain_circuits.learning.training.optimizer import build_adamw
    from posttrain_circuits.learning.training.schedules import PromptScheduler

    source = TeacherDemoStateSource(demonstrations)
    ids = [example.example_id for example in examples]
    texts = {record.prompt_id: record.raw_prompt_text for record in demonstrations}
    scheduler = PromptScheduler.for_allocation_neutral_rank(
        ids, [texts[key] for key in ids], rank=rank, world_size=2, global_batch_size=64, max_microbatch_size=4
    )
    optimizer = build_adamw(loaded.model.parameters(), learning_rate=lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    config = copy.deepcopy(original_config)
    config["production_safety"]["initial_checkpoint_path"] = str(checkpoint)
    config["trainer"]["learning_rate"] = lr
    trainer = FactorialTrainer(
        model=loaded.model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        prompt_scheduler=scheduler,
        state_source=source,
        supervisor=CanonicalSFTSupervisor(
            loaded.tokenizer.pad_token_id, normalization="sequence", minimum_positives=1, retry_limit=0
        ),
        config=TrainerConfig(
            max_steps=4,
            token_budget=2000000,
            steps_per_round=1,
            learning_rate=lr,
            checkpoint_every=20,
            evaluation_every=20,
            backend="accelerate",
            gradient_accumulation_steps=8,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
            max_completion_length=128,
            require_evaluation_metrics=False,
        ),
        run_dir=directory,
        resolved_config=config,
        manifest_hashes={"initial_checkpoint": INITIAL_SHA},
        git_commit="6c04f804b302184b8ff95d00fab404e0531ed8d6",
        implementation_dirty=False,
        probe_input_ids=None,
        evaluation_fn=None,
    )
    require(
        trainer._accelerator.mixed_precision == "bf16" and trainer._gradient_accumulation_steps == 8,
        "actual precision/accumulation differs",
    )
    require(
        all(p.dtype == torch.float32 and p.requires_grad for p in trainer.model.parameters()),
        "actual FSDP master dtype/training scope differs",
    )
    expected = dict(
        requested_fsdp_sharding_strategy="FULL_SHARD",
        effective_fsdp_sharding_strategy="FULL_SHARD",
        fsdp_wrapper_count=29,
    )
    same(trainer._fsdp_sharding_contract, expected, "actual W2 FSDP tree differs")
    require(
        trainer.global_step == 0 and not source.state_dict()["cursor"] and not optimizer.state,
        "arm did not start fresh",
    )
    return trainer, optimizer


def raw_artifacts(output, *, complete=False):
    rows = []
    for name in RAW_NAMES:
        path = output / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size:
            rows.append(dict(path=name, size=path.stat().st_size, sha256=file_sha(path)))
    require(not complete or len(rows) == len(RAW_NAMES), "completed raw evidence missing")
    return rows


def save_progress(output, report):
    report["raw_artifacts"] = raw_artifacts(output)
    atomic(output / "lr-probe.json", report)


def progress_marker(output, *, rank, phase, arm=None, step=None, **metrics):
    """At most 47 boundary markers for both arms; never one write per response."""
    if rank == 0:
        atomic(
            output / "progress.json",
            dict(
                phase=phase, arm=arm, step=step, passed=False, diagnostic_complete=False, **FALSE, **metrics
            ),
        )


def append_row(path, row):
    with path.open("ab") as stream:
        stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_record(output, state, *, arm, lr, inputs):
    import torch

    target = output / "checkpoints" / (arm + "-step4.pt")
    target.parent.mkdir(exist_ok=True)
    require(not target.exists(), "diagnostic checkpoint already exists")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(
            dict(
                format="student_lr_diagnostic_model_only_v1",
                model=state,
                arm=arm,
                learning_rate=lr,
                global_step=4,
                parent_checkpoint_sha256=INITIAL_SHA,
                job_id=inputs["job_id"],
                run_id=inputs["run_id"],
                scope="diagnostic_model_only",
                **FALSE,
            ),
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return dict(
        path=target.relative_to(output).as_posix(),
        size=target.stat().st_size,
        sha256=file_sha(target),
        arm=arm,
        step=4,
        scope="diagnostic_model_only",
    )


def generate_records(model, tokenizer, examples, prompts, *, arm, step, on_record):
    """Standalone greedy inference only; callback durably saves each complete row."""
    import torch

    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    require(arm in dict(ARMS) and step in (0, 4), "generation arm/step differs")
    require(len(examples) == len(prompts) == 32, "generation cohort must be the fixed first32")
    quality = helper("sdsc_student_quality_probe")
    records = []
    with measurement(model), torch.no_grad():
        for ordinal, (example, prompt) in enumerate(zip(examples, prompts, strict=True)):
            require(len(prompt["prompt_ids"]) + 256 <= 1536, "generation would truncate")
            ids = torch.tensor([prompt["prompt_ids"]], device=next(model.parameters()).device)
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                max_new_tokens=256,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            require(
                generated.ndim == 2
                and generated.shape[0] == 1
                and torch.equal(generated[0, : ids.shape[1]], ids[0]),
                "generation changed its prompt",
            )
            row = quality.response_record(
                example,
                generated[0, ids.shape[1] :].tolist(),
                tokenizer,
                prompt_ids=prompt["prompt_ids"],
                prompt_text=prompt["prompt_text"],
                cap=256,
            )
            row.update(
                arm=arm,
                step=step,
                ordinal=ordinal,
                prompt_text=prompt["prompt_text"],
                parsed_trace=asdict(ProofGraphTask().parse_response(row["response_text"])),
            )
            on_record(row)
            records.append(row)
    metrics = quality.summarize(records)
    metrics["answer_accuracy"] = metrics.pop("validation_accuracy")
    return metrics


def export_and_generate(
    trainer,
    model_config,
    tokenizer,
    examples,
    prompts,
    *,
    output,
    report,
    arm_report,
    arm,
    lr,
    step,
    rank,
    inputs,
):
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    progress_marker(output, rank=rank, phase="export_generation_start", arm=arm, step=step)
    references = []
    with measurement(trainer.model, trainer), torch.no_grad():
        if step == 4:
            for prompt in prompts[:4]:
                ids = torch.tensor([prompt["prompt_ids"]], device=next(trainer.model.parameters()).device)
                logits = trainer.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False
                ).logits.float()
                if rank == 0:
                    references.append(logits.detach().cpu())
                del logits
        # All ranks enter the original production FSDP full-state export.
        state = trainer._full_model_state_for_checkpoint()

        def standalone():
            if rank != 0:
                return None
            require(
                state
                and all(
                    not value.is_floating_point() or value.dtype == torch.float32 for value in state.values()
                ),
                "exported state is not FP32 master weights",
            )
            state_sha = torch_state_hash(state)
            if step == 0:
                arm_report["initial_state_sha256"] = state_sha
            else:
                record = checkpoint_record(output, state, arm=arm, lr=lr, inputs=inputs)
                arm_report["checkpoint"] = record
                report["checkpoints"].append(record)
                save_progress(output, report)
            loaded = load_model_and_tokenizer(model_config, for_training=False)
            loading = strict_load_export(loaded.model, state)
            model = loaded.model.to(
                device=next(trainer.model.parameters()).device, dtype=torch.bfloat16
            ).eval()
            parity = []
            if step == 4:
                for prompt, reference in zip(prompts[:4], references, strict=True):
                    ids = torch.tensor([prompt["prompt_ids"]], device=next(model.parameters()).device)
                    actual = (
                        model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids, dtype=torch.bool),
                            use_cache=False,
                        )
                        .logits.float()
                        .cpu()
                    )
                    parity.append(dict(ordinal=prompt["ordinal"], **difference_statistics(reference, actual)))
                    del actual
                arm_report["root_vs_export_logits"] = parity
            start = report["raw_record_count"]

            def publish_row(row):
                append_row(output / "lr-records.jsonl", row)
                report["raw_record_count"] += 1
                save_progress(output, report)

            metrics = generate_records(
                model, tokenizer, examples[:32], prompts, arm=arm, step=step, on_record=publish_row
            )
            arm_report["generation"].append(
                dict(
                    step=step,
                    record_start=start,
                    record_count=32,
                    metrics=metrics,
                    exported_state_sha256=state_sha,
                    strict_fp32_loading=loading,
                    path="exported_FP32_exact_load_then_BF16_standalone_no_explicit_autocast",
                )
            )
            save_progress(output, report)
            del model, loaded
            gc.collect()
            torch.cuda.empty_cache()
            return state_sha

        # Local rank-zero inference/file work is aligned before training resumes.
        result = local_phase(standalone, rank=rank, world_size=2)
    state, references = None, []
    gc.collect()
    progress_marker(output, rank=rank, phase="export_generation_complete", arm=arm, step=step)
    return result


def collect_window(trainer, expected_ids, *, rank, step, pad_token_id):
    batches, captures = [], []
    for microstep in range(8):
        prompt_batch = trainer.prompt_scheduler.next_batch()
        slots = trainer._batch_partition_plan.global_slots_for_microbatch(microstep)
        actual_ids = [expected_ids[(step - 1) * 64 + slot] for slot in slots]
        same(list(prompt_batch.prompt_ids), actual_ids, "training window repeated/reordered prompts")
        trajectories, supervision, attempts = trainer._collect_trajectories(prompt_batch)
        require(attempts == 1, "teacher source unexpectedly retried")
        trainer._register_collection(prompt_batch, trajectories, supervision, attempts)
        captures.append(
            batch_capture(
                trajectories,
                supervision,
                rank=rank,
                microstep=microstep,
                slots=[(step - 1) * 64 + slot for slot in slots],
                pad_token_id=pad_token_id,
            )
        )
        batches.append(supervision)
    return batches, captures


def validate_first_window(captures, expected):
    rows = sorted(
        (row for member in captures for batch in member for row in batch["records"]),
        key=lambda row: row["global_slot"],
    )
    same(rows, expected, "actual production first64 differs from independent audited capture")
    return digest(canonical(rows))


def run_arm(
    inputs,
    original_config,
    checkpoint,
    examples,
    demonstrations,
    expected_window,
    *,
    arm,
    lr,
    rank,
    output,
    report,
):
    import torch

    from posttrain_circuits.core.seeding import RNGState, seed_everything

    seed_everything(42)
    progress_marker(output, rank=rank, phase="arm_initializing", arm=arm, step=0)
    loaded, loading = local_phase(
        lambda: load_training_initial(original_config["model"], checkpoint), rank=rank, world_size=2
    )
    prompts, canonical_rows = local_phase(
        lambda: prepare_prompts(examples, loaded.tokenizer, original_config["model"]), rank=rank, world_size=2
    )
    directory = output.parent / ("training-" + arm)
    trainer, optimizer = build_trainer(
        loaded, demonstrations, examples, original_config, checkpoint, lr, rank, directory
    )
    observer = AdamWObserver(optimizer, lr, lambda: trainer._parameters_before_update)
    arm_report = dict(
        arm=arm,
        learning_rate=lr,
        updates=[],
        generation=[],
        initial_loading=loading,
        prepared_execution=trainer._checkpoint_batch_partition_state(),
        initial_rng_by_rank=gather(digest(canonical(RNGState.capture().as_dict()))),
    )
    if rank == 0:
        report["arms"].append(arm_report)
        if arm == ARMS[0][0]:
            for prompt in prompts:
                append_row(output / "lr-prompts.jsonl", prompt)
        save_progress(output, report)
    arm_report["teacher_target_initial"] = distributed_nll(trainer, expected_window, rank)
    arm_report["canonical_initial"] = distributed_nll(trainer, canonical_rows, rank)
    progress_marker(
        output,
        rank=rank,
        phase="initial_nll_complete",
        arm=arm,
        step=0,
        teacher_nll=arm_report["teacher_target_initial"]["sequence_mean_nll"],
        canonical_nll=arm_report["canonical_initial"]["sequence_mean_nll"],
    )
    export_and_generate(
        trainer,
        original_config["model"],
        loaded.tokenizer,
        examples,
        prompts,
        output=output,
        report=report,
        arm_report=arm_report,
        arm=arm,
        lr=lr,
        step=0,
        rank=rank,
        inputs=inputs,
    )
    expected_ids = [example.example_id for example in examples]
    for step in range(1, 5):
        update_started = time.perf_counter()
        progress_marker(output, rank=rank, phase="fixed_teacher_nll_before", arm=arm, step=step)
        before = distributed_nll(trainer, expected_window, rank)
        batches, captures = local_phase(
            lambda step=step: collect_window(
                trainer, expected_ids, rank=rank, step=step, pad_token_id=loaded.tokenizer.pad_token_id
            ),
            rank=rank,
            world_size=2,
        )
        gathered_captures = gather(captures)
        if step == 1:
            same(
                validate_first_window(gathered_captures, expected_window),
                digest(canonical(expected_window)),
                "first-window comparison changed",
            )
            if rank == 0 and arm == ARMS[0][0]:
                for member in gathered_captures:
                    for capture in member:
                        append_row(output / "lr-batches.jsonl", capture)
        admitted, total = trainer.token_budget.reserve_optimizer_update(
            sum(int(batch.attention_mask.sum()) for batch in batches)
        )
        require(admitted, "four-step diagnostic exhausted original training token budget")
        trainer._current_global_update_tokens = total
        progress_marker(output, rank=rank, phase="optimizer_window", arm=arm, step=step)
        syncs, counts = [], []
        for batch in batches:
            # This is the actual production forward/backward/step implementation.
            syncs.append(trainer._training_micro_step(batch, expected_sequence_count=4))
            counts.append(observer.calls)
        same(syncs, [False] * 7 + [True], "FSDP accumulation boundaries differ")
        same(counts, [step - 1] * 7 + [step], "optimizer stepped before/after reviewed boundary")
        metric = trainer._finish_optimizer_update(started=update_started)
        require(
            trainer.global_step == observer.calls == step and trainer.scheduler.last_epoch == step,
            "raw optimizer/scheduler/global cadence differs",
        )
        after = distributed_nll(trainer, expected_window, rank)
        local = dict(
            rank=rank,
            **observer.rows[-1],
            scheduler_last_epoch=trainer.scheduler.last_epoch,
            microstep_sync=syncs,
            actual_optimizer_calls=counts,
        )
        cursor = trainer.state_source.state_dict()
        expected_cursor = {expected_ids[index]: 1 for index in range(rank, step * 64, 2)}
        same(
            cursor["cursor"],
            expected_cursor,
            "teacher cursor did not consume successive unique training windows",
        )
        row = dict(
            step=step,
            teacher_target_before=before,
            teacher_target_after=after,
            production_metrics=metric,
            optimizer_by_rank=gather(local),
            cursors_by_rank=gather(
                dict(rank=rank, cursor=cursor["cursor"], state_sha256=digest(canonical(cursor)))
            ),
            token_budget=trainer.token_budget.state_dict(),
        )
        arm_report["updates"].append(row)
        if rank == 0:
            save_progress(output, report)
            print(
                json.dumps(
                    dict(
                        stage="lr_update",
                        arm=arm,
                        step=step,
                        teacher_before=before["sequence_mean_nll"],
                        teacher_after=after["sequence_mean_nll"],
                    )
                ),
                flush=True,
            )
        progress_marker(
            output,
            rank=rank,
            phase="update_nll_complete",
            arm=arm,
            step=step,
            teacher_before=before["sequence_mean_nll"],
            teacher_after=after["sequence_mean_nll"],
        )
        del batches, captures, gathered_captures
    arm_report["canonical_final"] = distributed_nll(trainer, canonical_rows, rank)
    export_and_generate(
        trainer,
        original_config["model"],
        loaded.tokenizer,
        examples,
        prompts,
        output=output,
        report=report,
        arm_report=arm_report,
        arm=arm,
        lr=lr,
        step=4,
        rank=rank,
        inputs=inputs,
    )
    observer.close()
    trainer._accelerator.free_memory()
    trainer, optimizer, observer, loaded = None, None, None, None
    gc.collect()
    torch.cuda.empty_cache()
    # A peer-local cleanup error must surface before the other arm starts.
    local_phase(lambda: None, rank=rank, world_size=2)


def execute(inputs, output, work):
    import torch
    from accelerate import Accelerator

    rank, world = int(os.environ.get("RANK", "-1")), int(os.environ.get("WORLD_SIZE", "0"))
    require(
        world == 2 and rank in (0, 1) and int(os.environ.get("LOCAL_RANK", "-1")) == rank,
        "diagnostic requires one-node W2 launcher",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 2, "two assigned CUDA devices required"
    )
    require(
        all("H100" in torch.cuda.get_device_name(index) for index in range(2)), "two H100 devices required"
    )
    for name, value in dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        require(os.environ.get(name) == value, "execution environment differs: " + name)
    require(torch.__version__ == "2.8.0+cu128", "fixed Torch runtime differs")
    for package, version in {
        "transformers": "4.56.2",
        "accelerate": "1.10.1",
        "tokenizers": "0.22.0",
    }.items():
        require(importlib.metadata.version(package) == version, "fixed runtime differs: " + package)
    torch.cuda.set_device(rank)
    torch.set_num_threads(12)
    bootstrap = Accelerator(gradient_accumulation_steps=8, step_scheduler_with_optimizer=False)
    require(
        bootstrap.num_processes == 2 and bootstrap.mixed_precision == "bf16",
        "actual launcher precision/world differs",
    )
    config, checkpoint, examples, demos, data_audit, captured = local_phase(
        lambda: load_inputs(inputs, work), rank=rank, world_size=world
    )
    report = dict(
        schema=SCHEMA,
        task="qwen3-v2-student-lr-diagnostic-v1",
        passed=False,
        diagnostic_complete=False,
        **FALSE,
        **{
            key: inputs[key]
            for key in ("parent_job_id", "job_id", "run_id", "source_code_sha256", "plan_sha256")
        },
        scope="train-only four-update LR diagnostic; no acceptance, validation or test split",
        initial_checkpoint_sha256=INITIAL_SHA,
        original_instruction_sha256=INSTRUCTION_SHA,
        data_audit_sha256=inputs["data_audit"]["report"]["sha256"],
        data_capture_sha256=inputs["data_audit"]["capture"]["sha256"],
        audited_window_sha256=data_audit["window_sha256"],
        generation=dict(
            max_new_tokens=256,
            max_model_input_length=1536,
            do_sample=False,
            use_cache=False,
            truncation=False,
        ),
        resources=dict(gpus=2, gpu_type="h100", cpus=24, memory_gib=384, time="01:00:00"),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=0.0,
            schedule="constant_1.0",
            clipping=False,
            warmup=False,
        ),
        numerical_checks=dict(
            labels_ce_relative_bound=2e-5,
            first_step_coordinate_bound="32*eps32*max(abs(old),abs(pred),lr)",
            formal_quality_gate=False,
        ),
        raw_record_count=0,
        raw_artifacts=[],
        arms=[],
        checkpoints=[],
    )
    started = time.monotonic()
    for arm, lr in ARMS:
        run_arm(
            inputs,
            config,
            checkpoint,
            examples,
            demos,
            captured,
            arm=arm,
            lr=lr,
            rank=rank,
            output=output,
            report=report,
        )
    if rank == 0:
        same(
            [arm["initial_state_sha256"] for arm in report["arms"]],
            [report["arms"][0]["initial_state_sha256"]] * 2,
            "arms did not restore identical initial master state",
        )
        same(
            report["arms"][0]["initial_rng_by_rank"],
            report["arms"][1]["initial_rng_by_rank"],
            "arm initial RNG differs",
        )
        require(
            report["raw_record_count"] == 128 and len(report["checkpoints"]) == 2, "diagnostic incomplete"
        )
        report.update(
            passed=True,
            diagnostic_complete=True,
            elapsed_seconds=time.monotonic() - started,
            raw_artifacts=raw_artifacts(output, complete=True),
        )
        atomic(output / "lr-probe.json", report)
    # If rank-zero final publication fails, it exits nonzero immediately; the
    # launcher's failure semantics terminate the peer. No error-path barrier.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", dest="inputs", required=True, type=Path)
    parser.add_argument("--output-dir", dest="output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(exist_ok=True)
    try:
        inputs = document(args.inputs)
        work = staged_science(inputs, args.output)
        execute(inputs, args.output, work)
        return 0
    except Exception as error:
        # Do not enter a collective after a rank-local or collective exception.
        # Accelerate/torchrun must terminate its peer; the node records failure.
        rank = os.environ.get("RANK", "0")
        failure = dict(
            passed=False,
            diagnostic_complete=False,
            **FALSE,
            error=f"{type(error).__name__}: {error}",
            failed_rank=rank,
        )
        atomic(args.output / ("rank-" + rank + "-failure.json"), failure)
        if rank == "0":
            path = args.output / "lr-probe.json"
            try:
                report = document(path) if path.is_file() else {"schema": SCHEMA}
            except (OSError, ValueError):
                report = {"schema": SCHEMA}
            report.update(failure, raw_artifacts=raw_artifacts(args.output))
            atomic(path, report)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
