"""CPU contract tests for the exploratory worker; no real model or GPU load."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.datasets.proofgraph.contracts import Literal, Rule, TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_AVAILABLE,
    TeacherCandidateOutput,
)
from posttrain_circuits.learning.teacher import demo_generation
from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed
from posttrain_circuits.models import loading, prompt_protocol
from posttrain_circuits.utils.smoke import build_smoke_examples

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "teacher_probe_worker_test", ROOT / "tools/sdsc_teacher_probe.py"
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


def _population_file(path: Path, count: int = 257) -> str:
    path.write_bytes(
        b"".join(
            json.dumps({"example_id": f"prompt-{index:04d}"}).encode() + b"\n"
            for index in range(count)
        )
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_population_keeps_first_256_but_hashes_the_complete_file(tmp_path):
    path = tmp_path / "examples.jsonl"
    checksum = _population_file(path)
    rows = worker.population(path, checksum)
    assert len(rows) == 256
    assert rows[0]["example_id"] == "prompt-0000"
    assert rows[-1]["example_id"] == "prompt-0255"
    with path.open("ab") as stream:
        stream.write(b'{"example_id":"later-tampering"}\n')
    with pytest.raises(ValueError, match="differs"):
        worker.population(path, checksum)


@pytest.mark.parametrize("fault", ["too_short", "duplicate", "symlink"])
def test_population_rejects_incomplete_or_ambiguous_identity(tmp_path, fault):
    path = tmp_path / "examples.jsonl"
    checksum = _population_file(path, 255 if fault == "too_short" else 256)
    if fault == "duplicate":
        raw = path.read_bytes().replace(b"prompt-0255", b"prompt-0000")
        path.write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
    elif fault == "symlink":
        link = tmp_path / "linked-examples.jsonl"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        worker.population(path, checksum)


def test_population_bounds_records_even_after_retained_population(tmp_path):
    path = tmp_path / "examples.jsonl"
    _population_file(path, 256)
    with path.open("ab") as stream:
        stream.write(b'{"example_id":"' + b"x" * (128 * 1024) + b'"}\n')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="bounded|record|size"):
        worker.population(path, checksum)


def test_population_rejects_a_fifo_without_blocking(tmp_path):
    path = tmp_path / "examples.fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="invalid train split file"):
        worker.population(path, "a" * 64)


def test_population_rejects_oversized_file_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    checksum = _population_file(path)
    original = path.stat()
    monkeypatch.setattr(
        worker.os,
        "fstat",
        lambda _: SimpleNamespace(st_mode=original.st_mode, st_size=1024**3 + 1),
    )
    with pytest.raises(ValueError, match="invalid train split file"):
        worker.population(path, checksum)


def test_comparison_freezes_v5_baseline_and_identifies_proposed_v7():
    assert hashlib.sha256(worker.BASELINE_INSTRUCTIONS.encode()).hexdigest() == (
        "55edb00c4d57197224c49dddfcb36ad1152640c8f8a19e2b223c1f3e968c504f"
    )
    assert worker.BASELINE_PROTOCOL == "qwen3-v2-g0-candidate-e-seed-42-prompt-v5"
    assert worker.CANDIDATE_PROTOCOL == "qwen3-v2-g0-candidate-e-seed-42-prompt-v7"


@pytest.mark.parametrize(
    ("conclusions", "expected"),
    [
        (["TRUE A", "NOT B"], (1, 1, 0, True)),
        (["A", "NOT B"], (1, 0, 1, False)),
        (["TRUE A", "B"], (2, 1, 1, False)),
        (["NOT A", "NOT B"], (0, 0, 0, None)),
        ([], (0, 0, 0, None)),
    ],
)
def test_TRUE_diagnostic_observes_parsed_positive_steps_without_semantic_changes(conclusions, expected):
    body = "\n".join(
        f"S{index:02d}: R01(F01) -> {literal}"
        for index, literal in enumerate(conclusions, 1)
    )
    # Inline tags are legal under the original parser, as is a bare positive
    # literal. The diagnostic must preserve both facts rather than reparse a
    # stricter dialect or rewrite the response.
    text = f"<proof>{body}</proof><answer>1</answer>"
    parsed = ProofGraphTask().parse_response(text)
    before = asdict(parsed)
    diagnostic = worker.output_contract_diagnostic(parsed)
    assert diagnostic["parse_valid"] is True
    assert tuple(diagnostic[key] for key in (
        "parsed_positive_conclusions", "parsed_positive_conclusions_with_TRUE",
        "parsed_positive_conclusions_without_TRUE", "all_positive_conclusions_use_TRUE",
    )) == expected
    assert asdict(parsed) == before


def test_TRUE_diagnostic_does_not_infer_parsed_steps_from_invalid_or_truncated_proofs():
    text = "<proof>\nS01: R01(F01) -> TRUE A\nS02: F02 -> B\n</proof><answer>1</answer>"
    for response in (text, text[:text.index("</proof>")]):
        parsed = ProofGraphTask().parse_response(response)
        diagnostic = worker.output_contract_diagnostic(parsed)
        assert diagnostic["parse_valid"] is False
        assert diagnostic["raw_TRUE_step_lines"] == 1
        assert diagnostic["parsed_positive_conclusions"] is None
        assert diagnostic["all_positive_conclusions_use_TRUE"] is None


def test_TRUE_diagnostic_keeps_embedded_closing_tag_rejected_by_original_verifier():
    example = TaskExample(
        example_id="embedded-closing-tag", facts={"F01": Literal("C")},
        rules={
            "R01": Rule("R01", (Literal("C"),), Literal("A")),
            "R02": Rule("R02", (Literal("A"),), Literal("B")),
        },
        query=Literal("B"), label=1, canonical_proof=[],
    )
    text = "<proof>S01: R01(F01) -> A</proof>\nS02: R02(S01) -> B</proof><answer>1</answer>"
    task = ProofGraphTask()
    parsed = task.parse_response(text)
    assert parsed.parse_valid and len(parsed.steps) == 2
    before = asdict(parsed)
    verification = task.verify(example, parsed)
    assert verification.reward == 0.0 and verification.error_code == "conclusion_mismatch"
    diagnostic = worker.output_contract_diagnostic(parsed)
    assert diagnostic["parsed_positive_conclusions"] == 2
    assert diagnostic["parsed_positive_conclusions_without_TRUE"] == 2
    assert diagnostic["all_positive_conclusions_use_TRUE"] is False
    assert asdict(parsed) == before
    assert task.verify(example, parsed) == verification


def test_TRUE_diagnostic_does_not_confuse_citation_arrow_with_conclusion_arrow():
    example = TaskExample(
        example_id="citation-arrow", facts={"F01": Literal("C")},
        rules={"R01": Rule("R01", (Literal("C"),), Literal("A"))},
        query=Literal("A"), label=1, canonical_proof=[],
    )
    text = "<proof>S01: R01(F01->oops) -> TRUE A</proof><answer>1</answer>"
    task = ProofGraphTask()
    parsed = task.parse_response(text)
    assert parsed.parse_valid
    before = asdict(parsed)
    verification = task.verify(example, parsed)
    assert verification.reward == 0.0 and verification.error_code == "unknown_citation"
    diagnostic = worker.output_contract_diagnostic(parsed)
    assert diagnostic["parsed_positive_conclusions"] == 1
    assert diagnostic["parsed_positive_conclusions_with_TRUE"] == 1
    assert diagnostic["parsed_positive_conclusions_without_TRUE"] == 0
    assert diagnostic["all_positive_conclusions_use_TRUE"] is True
    assert asdict(parsed) == before
    assert task.verify(example, parsed) == verification


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    # Compose the actual production configuration rather than inventing model
    # keys in a fake mapping. This catches expensive after-load API mistakes.
    config = compose_config(list(OVERRIDES), config_root=ROOT / "configs")
    assert config["teacher"]["model_revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
    assert "revision" not in config["teacher"]
    examples = build_smoke_examples(256, seed=9)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("fixture presence; population is injected below\n")
    output = tmp_path / "results"
    calls = []
    reports = []
    state = SimpleNamespace(
        fail_at=None, all_invalid=False, reject_revision=False,
        bare_positives=False, embedded_closing_tag=False,
    )

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [1, 2, 3]

    loaded = SimpleNamespace(
        model=object(),
        tokenizer=Tokenizer(),
        model_id=config["teacher"]["model_name_or_path"],
        requested_model_revision=config["teacher"]["model_revision"],
        resolved_model_commit=config["teacher"]["model_revision"],
    )

    def load_model(teacher, *, for_training):
        assert teacher == config["teacher"]
        assert for_training is False
        if state.reject_revision:
            loaded.resolved_model_commit = "0" * 40
        return loaded

    class Generator:
        def __init__(self, model, tokenizer, *, max_new_tokens, model_config):
            assert model is loaded.model and tokenizer is loaded.tokenizer
            assert max_new_tokens == 256 and model_config == config["teacher"]
            self.task = ProofGraphTask()

        def __call__(self, **kwargs):
            prompt = self.task.render(kwargs["example"])
            calls.append({**kwargs, "prompt": prompt})
            if len(calls) == state.fail_at:
                raise RuntimeError("injected inference failure")
            baseline = len(calls) <= worker.PROBE_PROMPTS
            if baseline:
                assert prompt.endswith(worker.BASELINE_INSTRUCTIONS)
            text = "<proof>\nS01: R01(F01,F02) -> TRUE X\n</proof>\n<answer>1</answer>"
            if not state.all_invalid and not baseline and kwargs["candidate_index"] == 0:
                text = ProofGraphTask().canonical_target(kwargs["example"])
            if state.bare_positives:
                text = text.replace("-> TRUE ", "-> ")
            if state.embedded_closing_tag:
                text = (
                    "<proof>S01: R01(F01) -> A</proof>\n"
                    "S02: R02(S01) -> B</proof><answer>1</answer>"
                )
            return TeacherCandidateOutput(
                response_text=text,
                response_ids=[17, 18],
                token_logprobs=[-0.25, -0.5],
                logprob_status=LOGPROB_AVAILABLE,
                finish_reason="eos",
            )

    def publish(path, report):
        path.write_text(json.dumps(report))
        reports.append(report)

    def validate_prompts(actual_config, actual_examples):
        assert actual_config == config
        assert [example.example_id for example in actual_examples] == [
            example.example_id for example in examples
        ]
        return {"minimum_prompt_tokens": 10, "maximum_prompt_tokens": 1000}

    guards = {
        "SHA256": re.compile(r"[0-9a-f]{64}\Z"),
        "OVERRIDES": OVERRIDES,
        "require": worker.require,
        "real_path": lambda value: value,
        "allocation": lambda args: {"job_id": args.job_id, "cpus": 24},
        "runtime_identity": lambda: {"cpu_fixture": True},
        "local_environment": lambda args: {"cpu_fixture": True},
        "gpu_identity": lambda: {"mock": "not a GPU observation"},
        "final_usage": lambda: {"mock": "no allocation"},
        "validate_prompts": validate_prompts,
        "publish": publish,
    }

    class GuardLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__dict__.update(guards)

    original_spec = importlib.util.spec_from_file_location

    def guard_spec(name, location, *args, **kwargs):
        if name == "sdsc_teacher_guards":
            return importlib.util.spec_from_loader(name, GuardLoader())
        return original_spec(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", guard_spec)
    monkeypatch.setattr(worker, "DATASET", dataset)
    monkeypatch.setattr(worker, "population", lambda: [asdict(example) for example in examples])
    monkeypatch.setattr(loading, "load_model_and_tokenizer", load_model)
    monkeypatch.setattr(loading, "move_model_to_local_cuda", lambda model: model)
    monkeypatch.setattr(demo_generation, "HfTeacherCandidateGenerator", Generator)
    monkeypatch.setattr(
        prompt_protocol,
        "format_model_prompt",
        lambda text, *_: SimpleNamespace(model_facing_prompt=text),
    )
    # The real entry point's process-level import changes must not leak into
    # other tests in the same interpreter.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    argv = [
        "--science-root", str(ROOT), "--work-dir", str(tmp_path),
        "--output-dir", str(output), "--hf-home", str(tmp_path / "hf"),
        "--run-id", "diagnostic-test", "--code-sha256", "a" * 64, "--job-id", "12345",
    ]
    return SimpleNamespace(
        argv=argv, config=config, examples=examples, output=output,
        calls=calls, reports=reports, state=state,
    )


def test_probe_uses_actual_model_keys_exact_seeds_raw_outputs_and_original_verifier(invocation):
    prior_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    assert worker.main(invocation.argv) == 0
    report, = invocation.reports
    rows = [json.loads(line) for line in (invocation.output / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == len(invocation.calls) == 288
    assert report["arms"]["baseline"]["attempt_count"] == 32
    assert report["arms"]["candidate"]["attempt_count"] == 256
    assert report["paired_candidate_zero"]["attempt_count"] == 32
    assert report["arms"]["baseline"]["accepted_count"] == 0
    assert report["arms"]["candidate"]["prompts_with_success"] == 32
    assert report["arms"]["candidate"]["accepted_count"] == 32
    assert report["paired_candidate_zero"]["accepted_count"] == 32
    assert report["baseline_protocol"] == worker.BASELINE_PROTOCOL
    assert report["candidate_protocol"] == worker.CANDIDATE_PROTOCOL
    assert report["candidate_protocol_review"] == "proposed"
    assert report["ordered_prompt_ids"] == [example.example_id for example in invocation.examples[:32]]
    assert report["execution_diagnostics"]["completed_attempt_counts"] == {"baseline": 32, "candidate": 256}
    assert "partial_results" not in report
    assert {signum: signal.getsignal(signum) for signum in prior_handlers} == prior_handlers
    progress = json.loads((invocation.output / "progress.json").read_text())
    assert progress["passed"] is False and progress["not_a_completion_report"] is True
    assert progress["completed_attempt_counts"] == {"baseline": 32, "candidate": 256}
    for row, call in zip(rows, invocation.calls, strict=True):
        identity = sha256_value(asdict(call["example"]))
        assert row["actual_sampling_seed"] == teacher_candidate_seed(31415, identity, row["candidate_index"])
        assert call["actual_sampling_seed"] == row["actual_sampling_seed"]
        assert {key: call[key] for key in ("temperature", "top_p", "top_k", "min_p")} == {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        }
        assert row["token_logprobs"] == [-0.25, -0.5]
        assert row["response_ids"] == [17, 18]
        task = ProofGraphTask()
        verification = task.verify(call["example"], task.parse_response(row["response_text"]))
        assert row["verification_trace"] == json.loads(json.dumps(asdict(verification)))
        assert row["accepted"] is (verification.reward == 1.0)
        assert row["output_contract_diagnostic"] == worker.output_contract_diagnostic(
            task.parse_response(row["response_text"])
        )
    baseline = {row["prompt_id"]: row for row in rows if row["variant"] == "baseline"}
    paired = {
        row["prompt_id"]: row
        for row in rows if row["variant"] == "candidate" and row["candidate_index"] == 0
    }
    assert {key: row["actual_sampling_seed"] for key, row in baseline.items()} == {
        key: row["actual_sampling_seed"] for key, row in paired.items()
    }


def test_bare_positive_proofs_still_count_as_accepted_but_TRUE_diagnostic_marks_mismatch(invocation):
    invocation.state.bare_positives = True
    assert worker.main(invocation.argv) == 0
    report, = invocation.reports
    assert report["arms"]["candidate"]["accepted_count"] == 32
    contract = report["arms"]["candidate"]["output_contract_diagnostic"]
    assert contract["parsed_positive_conclusions"] > 0
    assert contract["parsed_positive_conclusions_with_TRUE"] == 0
    assert contract["outputs_with_unmarked_positive_conclusions"] > 0
    assert contract["outputs_with_all_positive_conclusions_using_TRUE"] == 0


def test_embedded_closing_tag_does_not_interrupt_complete_raw_ledger_publication(invocation):
    invocation.state.embedded_closing_tag = True
    assert worker.main(invocation.argv) == 0
    report, = invocation.reports
    assert report["passed"] is True
    assert report["arms"]["candidate"]["accepted_count"] == 0
    rows = [json.loads(line) for line in (invocation.output / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 288
    for row, call in zip(rows, invocation.calls, strict=True):
        task = ProofGraphTask()
        parsed = task.parse_response(row["response_text"])
        verification = task.verify(call["example"], parsed)
        assert parsed.parse_valid and verification.reward == 0.0
        assert row["verification_trace"] == json.loads(json.dumps(asdict(verification)))
        assert row["output_contract_diagnostic"]["parsed_positive_conclusions"] == 2


def test_completed_diagnostic_with_no_valid_proofs_never_claims_readiness(invocation):
    invocation.state.all_invalid = True
    assert worker.main(invocation.argv) == 0
    report, = invocation.reports
    assert report["passed"] is True  # Execution completion only, expressly labelled.
    assert report["arms"]["candidate"]["accepted_count"] == 0
    assert report["exploratory"] is True
    for field in (
        "accepted_science", "full_teacher_ready", "g0_passed", "execution_class_certified",
        "training_started", "readiness_artifact_produced", "resumable",
    ):
        assert report[field] is False


@pytest.mark.parametrize("cause", ["inference", "revision"])
def test_worker_persists_failure_and_does_not_report_complete_probe(invocation, cause):
    if cause == "inference":
        invocation.state.fail_at = 3
    else:
        invocation.state.reject_revision = True
    with pytest.raises((RuntimeError, ValueError), match="injected inference|revision differs"):
        worker.main(invocation.argv)
    report, = invocation.reports
    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["full_teacher_ready"] is False
    if cause == "inference":
        assert len((invocation.output / "attempts.jsonl").read_text().splitlines()) == 2
    else:
        assert invocation.calls == []


def test_validate_only_does_not_generate_or_create_results(invocation):
    assert worker.main([*invocation.argv, "--validate-only"]) == 0
    assert invocation.calls == invocation.reports == []
    assert not invocation.output.exists()


INTERRUPTION_CHILD = r'''
import importlib.util
import json
import sys
import time
from pathlib import Path
root, work = Path(sys.argv[1]), Path(sys.argv[2])
mode = sys.argv[3]
sys.path.insert(0, str(root / "src"))
spec = importlib.util.spec_from_file_location("interrupted_probe_fixture",
    root / "tests/sdsc/test_teacher_probe_worker.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import pytest
patch = pytest.MonkeyPatch()
fixture = module.invocation.__wrapped__(work, patch)
def pause():
    (work / "ready").write_text("signal-ready")
    time.sleep(90)
if mode == "progress_publication":
    original_replace = module.worker.os.replace
    def interrupted_replace(source, destination):
        if Path(destination).name == "progress.json":
            value = json.loads(Path(source).read_text())
            if (value["stage"] == "candidate_generation"
                    and value["completed_attempt_counts"]["baseline"] == 2):
                pause()
        return original_replace(source, destination)
    patch.setattr(module.worker.os, "replace", interrupted_replace)
if mode == "model_load":
    original = module.loading.load_model_and_tokenizer
    def load(*args, **kwargs):
        pause()
        return original(*args, **kwargs)
    patch.setattr(module.loading, "load_model_and_tokenizer", load)
else:
    original = module.demo_generation.HfTeacherCandidateGenerator
    class PausedGenerator(original):
        def __call__(self, **kwargs):
            if len(fixture.calls) == 2:
                pause()
            return super().__call__(**kwargs)
    patch.setattr(module.demo_generation, "HfTeacherCandidateGenerator", PausedGenerator)
module.worker.main(fixture.argv)
'''


@pytest.mark.parametrize(
    ("mode", "signum", "completed", "failure_stage"),
    [
        ("model_load", signal.SIGTERM, 0, "model_load"),
        ("after_two_candidates", signal.SIGTERM, 2, "candidate_generation"),
        ("progress_publication", signal.SIGTERM, 2, "candidate_generation"),
        ("after_two_candidates", signal.SIGKILL, 2, "candidate_generation"),
    ],
)
def test_real_subprocess_interruption_retains_stage_and_committed_evidence(
    tmp_path, mode, signum, completed, failure_stage,
):
    # Run the actual worker.main in an independent OS process with the existing
    # fake-model fixture. A real signal must exercise the lifecycle; injecting a
    # Python exception alone missed the original SIGTERM/finally defect.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with (tmp_path / "stdout.log").open("wb") as stdout, (tmp_path / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", INTERRUPTION_CHILD, str(ROOT), str(tmp_path), mode],
            stdout=stdout, stderr=stderr, env=environment,
        )
        try:
            deadline = time.monotonic() + 90
            while not (tmp_path / "ready").exists():
                assert child.poll() is None, (tmp_path / "stderr.log").read_text()
                assert time.monotonic() < deadline, "CPU fixture did not reach interruption point"
                time.sleep(0.05)
            output = tmp_path / "results"
            before = json.loads((output / "progress.json").read_text())
            if mode == "progress_publication":
                # The durable previous snapshot remains readable while the
                # next fsynced temporary file has not yet been renamed.
                assert before["stage"] == "prompt_encoding"
                pending_files = list(output.glob(".progress.json*.tmp"))
                assert len(pending_files) == 1
                pending = json.loads(pending_files[0].read_text())
                assert pending["stage"] == failure_stage
            else:
                assert before["stage"] == failure_stage
            assert before["completed_attempt_counts"] == {"baseline": completed, "candidate": 0}
            assert before["passed"] is False and before["not_a_completion_report"] is True
            child.send_signal(signum)
            child.wait(timeout=15)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    ledger = output / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    assert len(rows) == completed
    assert all(row["variant"] == "baseline" for row in rows)
    progress = json.loads((output / "progress.json").read_text())
    if signum == signal.SIGKILL:
        # An uncatchable kill cannot publish a final report or traceback. Only
        # the already fsynced progress and committed ledger are guaranteed.
        assert child.returncode == -signal.SIGKILL
        assert not (output / "teacher-prompt-probe.json").exists()
        assert progress["stage"] == failure_stage
        assert (output / "worker-stacks.log").read_bytes() == b""
    else:
        assert child.returncode == 128 + signal.SIGTERM
        assert "pause" in (output / "worker-stacks.log").read_text()
        report = json.loads((output / "teacher-prompt-probe.json").read_text())
        assert report["passed"] is False
        assert report["exit_code"] == 128 + signal.SIGTERM
        assert report["interrupted_signal"] == "SIGTERM"
        assert report["failure_stage"] == failure_stage
        assert report["partial_results"]["not_a_quality_estimate"] is True
        assert report["partial_results"]["completed_attempt_counts"] == {
            "baseline": completed, "candidate": 0,
        }
        assert report["partial_results"]["ledger_committed_bytes"] == (
            ledger.stat().st_size if completed else 0
        )
        assert "arms" not in report and "paired_candidate_zero" not in report
        assert progress["stage"] == "interrupted" and progress["failure_stage"] == failure_stage
        assert not list(output.glob(".progress.json*.tmp"))
        for field in ("accepted_science", "full_teacher_ready", "g0_passed", "training_started"):
            assert report[field] is False
    assert progress["passed"] is False and progress["not_a_completion_report"] is True
    assert progress["stage_elapsed_seconds"] >= 0 and progress["elapsed_seconds"] >= 0


def test_config_failure_is_observed_before_any_model_work(invocation, monkeypatch):
    def fail(*_args):
        raise RuntimeError("injected configuration failure")

    monkeypatch.setattr(worker, "probe_config", fail)
    with pytest.raises(RuntimeError, match="configuration failure"):
        worker.main(invocation.argv)
    report, = invocation.reports
    assert report["passed"] is False and report["failure_stage"] == "config_composition"
    assert report["partial_results"]["completed_attempt_counts"] == {"baseline": 0, "candidate": 0}
    assert invocation.calls == []


def test_progress_publication_does_not_remove_a_preexisting_unowned_temporary(tmp_path):
    foreign = tmp_path / ".progress.json.tmp"
    foreign.write_text("not created by this publication")
    worker.progress_json(tmp_path / "progress.json", {"stage": "observed"})
    assert json.loads((tmp_path / "progress.json").read_text()) == {"stage": "observed"}
    assert foreign.read_text() == "not created by this publication"
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []


def test_progress_cleanup_error_does_not_mask_original_interruption(tmp_path, monkeypatch):
    def interrupt(*_args):
        raise worker.ProbeInterrupted(signal.SIGTERM)

    unlink = Path.unlink

    def denied_cleanup(path, *args, **kwargs):
        if path.name.startswith(".progress.json."):
            raise PermissionError("injected cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(worker.os, "replace", interrupt)
    monkeypatch.setattr(Path, "unlink", denied_cleanup)
    with pytest.raises(worker.ProbeInterrupted) as error:
        worker.progress_json(tmp_path / "progress.json", {"stage": "observed"})
    assert error.value.signum == signal.SIGTERM
    assert not (tmp_path / "progress.json").exists()


SAFE_OBSERVATION_CHILD = r'''
import faulthandler
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
def forbidden(*args, **kwargs):
    raise AssertionError("Asynchronous faulthandler traversal is forbidden")
for name in ("dump_traceback_later", "cancel_dump_traceback_later", "register", "unregister"):
    setattr(faulthandler, name, forbidden)
spec = importlib.util.spec_from_file_location("safe_probe_observation", root / "tools/sdsc_teacher_probe.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 64)).eval()
inputs = torch.randn(4, 64, device="cpu")
torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
identity = {"task": worker.TASK, "job_id": "12345", "run_id": "cpu-observation-fixture",
            "code_sha256": "a" * 64}
output.mkdir()
observation = worker.ProbeObservation(output, identity)
try:
    observation.phase("cpu_forward")
    started, iterations = time.monotonic(), 0
    with torch.inference_mode():
        while time.monotonic() - started < 1:
            result = model(inputs)
            iterations += 1
    assert bool(torch.isfinite(result).all())
    assert iterations > 100
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    observation.phase("cpu_forward_finished")
finally:
    observation.close()
print(json.dumps({"cpu_forward_passed": True, "iterations": iterations,
                  "torch_rng_unchanged": True, "python_rng_unchanged": True}), flush=True)
'''


def test_production_observer_supports_real_cpu_forward_without_asynchronous_native_traversal(tmp_path):
    # The retained diagnostic A/B reproduces a native SIGSEGV with an enabled
    # timer. This safe regression never enables that known crashing stress path:
    # forbid its APIs, then exercise the actual observer around real CPU Torch.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    output = tmp_path / "observation"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", SAFE_OBSERVATION_CHILD, str(ROOT), str(output)],
        env=environment, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    final = json.loads(result.stdout.splitlines()[-1])
    assert final["cpu_forward_passed"] is True and final["iterations"] > 100
    assert final["torch_rng_unchanged"] is True and final["python_rng_unchanged"] is True
    progress = json.loads((output / "progress.json").read_text())
    assert progress["stage"] == "cpu_forward_finished"
    assert progress["passed"] is False and progress["not_a_completion_report"] is True
    assert (output / "worker-stacks.log").read_bytes() == b""
