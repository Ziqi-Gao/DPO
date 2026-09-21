"""CPU evidence for diagnostic contracts; no model load, SSH, or GPU allocation."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.cli import evaluate_teacher_readiness as readiness_cli
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.teacher_demos.contracts import TeacherCandidateOutput
from posttrain_circuits.learning.teacher import demo_generation
from posttrain_circuits.learning.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    evaluate_teacher_readiness,
    validate_teacher_readiness_artifact,
)
from posttrain_circuits.models import loading, prompt_protocol
from posttrain_circuits.utils.smoke import build_smoke_examples

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capability_worker_test", ROOT / "tools/sdsc_teacher_capability.py"
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
OVERRIDES = (
    "g0=qwen3_v2_eap_separation",
    "experiment=canonical_sft",
    "task.num_examples=256",
    "state_source.num_candidates=8",
    "seed=42",
)


def prefix_scores():
    return [
        TeacherPrefixScore(
            probe_id=stage + kind,
            stage=stage,
            prefix_kind=kind,
            target_ids=(10,),
            top1_correct=True,
            target_in_topk=True,
            minimum_topk_mass=0.95,
            causal_shift_valid=True,
            target_log_probability=-0.2,
            alternative_log_probability=-1.0,
            target_logprob_margin=0.8,
            causal_shift_logprob=0.5,
        )
        for stage in ("first_rule_selection", "intermediate_conclusion")
        for kind in ("canonical", "corrupted_or_initial_student")
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "syntax",
        "wrong_answer",
        "wrong_proof",
        "branch_order",
        "missing_stage",
        "top1",
        "coverage",
        "minimum_mass",
        "causal_validity",
        "negative_shift",
        "no_prefix_scores",
    ],
)
def test_metric_reduction_exactly_matches_authoritative_readiness(fault):
    task = ProofGraphTask()
    examples = build_smoke_examples(4, seed=17)
    generated = {example.example_id: task.canonical_target(example) for example in examples}
    scores = prefix_scores()
    first = examples[0]
    if fault == "syntax":
        generated[first.example_id] += " unrelated prose"
    elif fault == "wrong_answer":
        generated[first.example_id] = generated[first.example_id].replace(
            f"<answer>{first.label}", f"<answer>{1 - first.label}"
        )
    elif fault == "wrong_proof":
        generated[first.example_id] = generated[first.example_id].replace("(F01)", "(F99)", 1)
    elif fault == "branch_order":
        branch = task.generate_pair(42, {"structure": "branch", "depth": 2, "distractors": 4})[0]
        steps = list(branch.canonical_proof)
        assert len(steps) == 3
        reordered = [
            replace(steps[1], step_id="S01"),
            replace(steps[0], step_id="S02"),
            replace(steps[2], citations=("S02", "S01")),
        ]
        from posttrain_circuits.datasets.proofgraph.rendering import render_step

        text = "<proof>\n" + "\n".join(render_step(step) for step in reordered)
        text += f"\n</proof>\n<answer>{branch.label}</answer>"
        assert task.verify(branch, task.parse_response(text)).reward == 1.0
        examples = [branch]
        generated = {branch.example_id: text}
    elif fault == "missing_stage":
        scores = [row for row in scores if row.stage != "intermediate_conclusion"]
    elif fault == "top1":
        scores[0] = replace(scores[0], top1_correct=False)
    elif fault == "coverage":
        scores[0] = replace(scores[0], target_in_topk=False)
    elif fault == "minimum_mass":
        scores[0] = replace(scores[0], minimum_topk_mass=0.89)
    elif fault == "causal_validity":
        scores[0] = replace(scores[0], causal_shift_valid=False)
    elif fault == "negative_shift":
        scores[0] = replace(scores[0], causal_shift_logprob=-0.01)
    elif fault == "no_prefix_scores":
        scores = []
    thresholds = TeacherReadinessThresholds(**worker.THRESHOLDS)
    actual = worker.capability_metrics(examples, generated, scores, thresholds)
    # These bindings exist only inside this CPU comparison fixture. The worker
    # does not construct them or serialize a formal readiness artifact.
    reference = evaluate_teacher_readiness(
        examples,
        generated,
        scores,
        thresholds,
        bindings={
            "teacher_model_revision": "fixture",
            "tokenizer_revision": "fixture",
            "dataset_hash": "a" * 64,
            "prefix_probe_hash": "b" * 64,
            "code_commit": "fixture-only",
            "prereg_commit": "fixture-only",
        },
    )
    for key in ("metrics", "checks", "thresholds", "full_generation_rows"):
        assert actual[key] == reference[key]
    assert actual["metrics_passed"] is reference["passed"]
    if fault == "branch_order":
        assert actual["metrics"]["exact_proof_accuracy"] == 0.0


def test_metrics_reject_missing_generation_population():
    examples = build_smoke_examples(2)
    with pytest.raises(ValueError, match="exactly cover"):
        worker.capability_metrics(examples, {}, prefix_scores(), TeacherReadinessThresholds())


def population_file(path, count=10000):
    path.write_text("".join(json.dumps({"example_id": f"prompt-{i:05d}"}) + "\n" for i in range(count)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_population_fixed_first128_hashes_complete_10000_rows(tmp_path):
    path = tmp_path / "examples.jsonl"
    checksum = population_file(path)
    rows = worker.population(path, checksum)
    assert len(rows) == 128 and rows[-1]["example_id"] == "prompt-00127"
    raw = path.read_bytes().replace(b"prompt-09999", b"prompt-09998")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="differs"):
        worker.population(path, checksum)


@pytest.mark.parametrize("fault", ["short", "extra", "duplicate", "symlink", "fifo", "oversize_line"])
def test_population_rejects_unbounded_or_ambiguous_inputs(tmp_path, fault):
    path = tmp_path / "examples.jsonl"
    count = 9999 if fault == "short" else 10001 if fault == "extra" else 10000
    checksum = population_file(path, count)
    if fault == "duplicate":
        raw = path.read_bytes().replace(b"prompt-00127", b"prompt-00000")
        path.write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
    elif fault == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    elif fault == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif fault == "oversize_line":
        path.write_bytes(b"x" * (worker.MAX_RECORD_BYTES + 1))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        worker.population(path, checksum)


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    config = compose_config(list(OVERRIDES), config_root=ROOT / "configs")
    examples = build_smoke_examples(128, seed=13)
    calls, reports, score_calls = [], [], []
    state = SimpleNamespace(invalid=False, fail_at=None, fail_prefix=False, wrong_revision=False)
    output = tmp_path / "results"

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [1, 2, 3]

    loaded = SimpleNamespace(
        model=object(),
        tokenizer=Tokenizer(),
        model_id="Qwen/Qwen3-8B",
        tokenizer_id="Qwen/Qwen3-8B",
        resolved_model_commit=worker.TEACHER_REVISION,
        resolved_tokenizer_commit=worker.TEACHER_REVISION,
        requested_tokenizer_revision=worker.TEACHER_REVISION,
        tokenizer_hash="a" * 64,
    )

    def load_model(teacher, *, for_training):
        assert teacher == config["teacher"] and for_training is False
        if state.wrong_revision:
            loaded.resolved_model_commit = "0" * 40
        return loaded

    class Generator:
        def __init__(self, model, tokenizer, *, max_new_tokens, model_config):
            assert model is loaded.model and tokenizer is loaded.tokenizer
            assert max_new_tokens == 128 and model_config == config["teacher"]

        def __call__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == state.fail_at:
                raise RuntimeError("injected generation failure")
            text = "unparseable" if state.invalid else ProofGraphTask().canonical_target(kwargs["example"])
            return TeacherCandidateOutput(
                response_text=text,
                response_ids=[17, 18],
                token_logprobs=[-0.25, -0.5],
                logprob_status="available",
                finish_reason="eos",
            )

    def score(model, spec, *, top_k):
        assert model is loaded.model and top_k == 128
        score_calls.append(spec.stage)
        if state.fail_prefix:
            raise RuntimeError("injected prefix failure")
        return tuple(row for row in prefix_scores() if row.stage == spec.stage)

    def publish(path, report):
        path.write_text(json.dumps(report, allow_nan=False))
        if path.name == "teacher-capability-probe.json":
            reports.append(report)

    guards = {
        "SHA256": re.compile(r"[0-9a-f]{64}\Z"),
        "OVERRIDES": OVERRIDES,
        "real_path": lambda path: path,
        "allocation": lambda args: {"job_id": args.job_id, "cpus": 24},
        "runtime_identity": lambda: {"CPU_fixture": True},
        "local_environment": lambda args: {},
        "gpu_identity": lambda: {"CPU_fixture_not_real_GPU": True},
        "final_usage": lambda: {},
        "validate_prompts": lambda config, examples: {"maximum_prompt_tokens": 1000},
        "publish": publish,
    }

    class GuardLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__dict__.update(guards)

    original_spec = importlib.util.spec_from_file_location

    def guard_spec(name, path, *args, **kwargs):
        if name == "sdsc_teacher_guards":
            return importlib.util.spec_from_loader(name, GuardLoader())
        return original_spec(name, path, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", guard_spec)
    monkeypatch.setattr(worker, "population", lambda: [asdict(example) for example in examples])
    monkeypatch.setattr(loading, "load_model_and_tokenizer", load_model)
    monkeypatch.setattr(loading, "move_model_to_local_cuda", lambda model: model)
    monkeypatch.setattr(demo_generation, "HfTeacherCandidateGenerator", Generator)
    monkeypatch.setattr(readiness_cli, "_prefix_scores", score)
    monkeypatch.setattr(
        prompt_protocol, "format_model_prompt", lambda raw, *_: SimpleNamespace(model_facing_prompt=raw)
    )
    specs = [
        SimpleNamespace(stage=stage)
        for stage in ("first_rule_selection", "intermediate_conclusion", "final_answer")
    ]
    monkeypatch.setattr(
        worker, "prefix_probes", lambda *_: ({"sha256": "b" * 64}, specs, {"sha256": "c" * 64}, [])
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    argv = [
        "--science-root",
        str(ROOT),
        "--work-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--hf-home",
        str(tmp_path / "hf"),
        "--run-id",
        "fixture",
        "--job-id",
        "12345",
        "--code-sha256",
        "a" * 64,
    ]
    return SimpleNamespace(
        argv=argv,
        config=config,
        examples=examples,
        calls=calls,
        reports=reports,
        output=output,
        state=state,
        score_calls=score_calls,
    )


@pytest.mark.parametrize("invalid", [False, True])
def test_diagnostic_preserves_actual_generation_and_never_claims_readiness(invocation, invalid):
    invocation.state.invalid = invalid
    assert worker.main(invocation.argv) == 0
    (report,) = invocation.reports
    rows = [json.loads(line) for line in (invocation.output / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == len(invocation.calls) == 128
    assert report["ordered_prompt_ids"] == [example.example_id for example in invocation.examples]
    for index, (row, call) in enumerate(zip(rows, invocation.calls, strict=True)):
        assert row["actual_sampling_seed"] == call["actual_sampling_seed"] == 42 + index
        assert call["candidate_index"] == 0
        assert {key: call[key] for key in ("temperature", "top_p", "top_k", "min_p")} == {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
        }
        assert row["response_ids"] == [17, 18] and row["token_logprobs"] == [-0.25, -0.5]
        assert row["accepted"] is not invalid
    assert report["passed"] is True and report["exit_code"] == 0
    assert report["metrics_passed"] is not invalid
    assert report["thresholds"] == worker.THRESHOLDS
    assert report["artifact_kind"] == "teacher_capability_diagnostic"
    assert "not probabilities of a stochastic behavior policy" in report["generation_score_semantics"]
    assert report["prefix_score_count"] == 4
    assert invocation.score_calls == ["first_rule_selection", "intermediate_conclusion"]
    for name in (
        "accepted_science",
        "full_teacher_ready",
        "readiness",
        "readiness_artifact_produced",
        "training_started",
        "g0_passed",
        "execution_class_certified",
        "resumable",
    ):
        assert report[name] is False
    assert not any(key in report for key in ("bindings", "code_commit", "prereg_commit"))
    assert not (invocation.output / "teacher_readiness.json").exists()
    # Even giving this diagnostic a self-consistent content digest cannot turn
    # it into a formal readiness artifact: scientific schema/bindings are absent.
    hashed = {**report, "sha256": sha256_value(report)}
    with pytest.raises(ValueError):
        validate_teacher_readiness_artifact(hashed)


@pytest.mark.parametrize("fault", ["generation", "prefix", "revision"])
def test_failure_preserves_partial_evidence_and_failed_diagnostic(invocation, fault):
    if fault == "generation":
        invocation.state.fail_at = 3
    elif fault == "prefix":
        invocation.state.fail_prefix = True
    else:
        invocation.state.wrong_revision = True
    with pytest.raises((RuntimeError, ValueError), match="injected|revision differs"):
        worker.main(invocation.argv)
    (report,) = invocation.reports
    assert report["passed"] is False and report["exit_code"] == 1 and report["metrics_passed"] is False
    assert report["readiness_artifact_produced"] is False
    if fault != "revision":
        assert len((invocation.output / "attempts.jsonl").read_text().splitlines()) == (
            2 if fault == "generation" else 128
        )


def test_validate_only_creates_no_results_and_performs_no_model_work(invocation):
    assert worker.main([*invocation.argv, "--validate-only"]) == 0
    assert not invocation.output.exists()
    assert invocation.calls == invocation.reports == invocation.score_calls == []


def test_changed_thresholds_fail_before_generation(invocation, monkeypatch):
    from posttrain_circuits.core import config as config_api

    changed = {
        **invocation.config,
        "teacher_readiness": {**worker.THRESHOLDS, "minimum_teacher_answer_accuracy": 0.1},
    }
    monkeypatch.setattr(config_api, "compose_config", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="protocol differs"):
        worker.main(invocation.argv)
    assert invocation.calls == [] and not invocation.output.exists()
