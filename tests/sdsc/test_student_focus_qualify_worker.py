"""Bounded qualification regressions using original scorers and actual CPU state loads."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from posttrain_circuits.datasets.proofgraph.anti_shortcut import build_anti_shortcut_suite
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_qualify", ROOT / "tools/sdsc_student_focus_qualify_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


class Pieces:
    """Lossless local text tokens keep original parsing observable without a Hub dependency."""

    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.forward, self.backward = {}, {}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        pieces = [text[start : start + 4] for start in range(0, len(text), 4)]
        for piece in pieces:
            if piece not in self.forward:
                token = len(self.forward) + 2
                self.forward[piece] = token
                self.backward[token] = piece
        return [self.forward[piece] for piece in pieces]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.tensor([self.encode(text, add_special_tokens=add_special_tokens)])
        )

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(self.backward[int(value)] for value in ids if value > 1)


class Responses(torch.nn.Module):
    def __init__(self, tokenizer, texts):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
        self.tokenizer, self.texts, self.calls = tokenizer, texts, []
        self.eval()

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled() and not self.training
        assert not torch.is_autocast_enabled("cpu")
        text = self.texts[len(self.calls)]
        self.calls.append(kwargs)
        response = [*self.tokenizer.encode(text, add_special_tokens=False), self.tokenizer.eos_token_id]
        return torch.cat((kwargs["input_ids"], torch.tensor([response])), dim=1)


@pytest.fixture
def population():
    task = ProofGraphTask()
    cfg = dict(depth=1, distractors=0, structure="chain")
    validation = build_split(task, "validation", 128, 42, cfg)
    iid = build_split(task, "iid_test", 128, 42, cfg)
    return validation, iid, build_anti_shortcut_suite(iid, seed=42, distractor_ood_count=32)


def execute_fixture(worker, population, responses):
    validation, iid, cases = population
    tokenizer = Pieces()
    prompts, examples = worker.prepare_prompts(validation, iid, cases, tokenizer, {}, max_positions=4096)
    model = Responses(tokenizer, responses)
    records = []
    base, anti, raw = worker.score_original(
        model,
        tokenizer,
        validation,
        iid,
        cases,
        prompts,
        examples,
        {},
        checkpoint_sha="e" * 64,
        code_commit="a" * 40,
        prereg_commit="b" * 40,
        on_record=records.append,
    )
    assert raw == records and len(records) == len(model.calls) == 896
    assert "generate" not in model.__dict__
    for ordinal, record in enumerate(records):
        assert record["ordinal"] == ordinal and record["response_ids"][-1] == 1
        assert record["prompt_ids"] == model.calls[ordinal]["input_ids"][0].tolist()
        assert record["verification"] == asdict(
            ProofGraphTask().verify(
                examples[ordinal], ProofGraphTask().parse_response(record["response_text"])
            )
        )
        assert set(model.calls[ordinal]) == {
            "input_ids",
            "max_new_tokens",
            "do_sample",
            "pad_token_id",
            "eos_token_id",
            "use_cache",
        }
        assert record["prepared_checkpoint_sha256"] == "e" * 64
    return base, anti, records


def test_original_answer_metric_differs_from_anti_full_proof_reward(worker, population):
    validation, iid, cases = population
    task = ProofGraphTask()
    texts = [f"<proof></proof><answer>{x.label}</answer>" for x in validation]
    texts += [task.canonical_target(x) for x in iid]
    texts += [task.canonical_target(x.example) for x in cases]
    base, anti, records = execute_fixture(worker, population, texts)
    assert base == dict(
        num_examples=128,
        answer_correct=128,
        proof_correct=0,
        format_valid=128,
        answer_accuracy=1.0,
        exact_proof_accuracy=0.0,
        format_validity=1.0,
        passed=True,
    )
    assert anti["passed"] is True and anti["iid_accuracy"] == anti["transformed_accuracy"] == 1.0
    assert records[256]["transformation"] == "entity_symbol_renaming"
    assert records[261]["source_example_id"] != records[256]["source_example_id"]
    assert anti["prereg_version"] == "qwen3_v2"


def test_base_failure_does_not_skip_any_of_the_768_anti_responses(worker, population):
    validation, iid, cases = population
    task = ProofGraphTask()
    texts = ["invalid" for _ in validation] + [task.canonical_target(x) for x in iid]
    texts += [task.canonical_target(x.example) for x in cases]
    base, anti, _ = execute_fixture(worker, population, texts)
    assert base["passed"] is False and base["answer_correct"] == 0 and anti["passed"] is True


@pytest.mark.parametrize(
    "iid_count,transformed_count,expected",
    [(30, 24, True), (30, 23, False), (13, 11, True), (13, 10, False), (12, 7, False)],
)
def test_original_anti_capability_and_gap_thresholds(
    worker, population, iid_count, transformed_count, expected
):
    validation, iid, cases = population
    task = ProofGraphTask()

    def text(example, good):
        return task.canonical_target(example) if good else f"<proof></proof><answer>{example.label}</answer>"

    texts = [task.canonical_target(x) for x in validation]
    texts += [text(x, index < iid_count) for index, x in enumerate(iid)]
    texts += [text(case.example, index // 5 < transformed_count) for index, case in enumerate(cases)]
    base, anti, _ = execute_fixture(worker, population, texts)
    assert base["passed"] is True and anti["passed"] is expected
    assert anti["iid_accuracy"] == iid_count / 128
    assert all(x == transformed_count / 128 for x in anti["transformation_accuracy"].values())


def test_long_original_anti_prompt_permitted_and_over2244_rejected_before_generation(worker, population):
    _, iid, _ = population

    class LengthTokens(Pieces):
        def __init__(self, length):
            super().__init__()
            self.length = length

        def encode(self, text, *, add_special_tokens):
            return [2] * self.length

    prompts, _ = worker.prepare_prompts([], iid[:1], [], LengthTokens(1988), {}, max_positions=4000)
    assert len(prompts[0]["prompt_ids"]) + 256 == 2244
    for n, positions in [(1989, 4000), (1988, 2243)]:
        with pytest.raises(ValueError, match="exceeds context"):
            worker.prepare_prompts([], iid[:1], [], LengthTokens(n), {}, max_positions=positions)
    with pytest.raises(ValueError, match="exceeds context"):
        worker.prepare_prompts(iid[:1], [], [], LengthTokens(1281), {}, max_positions=4000)


def test_recording_rejects_changed_generation_prompt_and_restores_method(worker, population):
    examples = population[0][:1]
    tokenizer = Pieces()
    model = Responses(tokenizer, ["invalid"])
    prompts, all_examples = worker.prepare_prompts(examples, [], [], tokenizer, {}, max_positions=4000)
    with (
        pytest.raises(ValueError, match="actual prompt differs"),
        worker.recording_generate(
            model, tokenizer, all_examples, prompts, lambda row: None, checkpoint_sha="e" * 64
        ),
        torch.no_grad(),
    ):
        model.generate(
            input_ids=torch.tensor([[999]]),
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
        )
    assert not model.calls and "generate" not in model.__dict__


def dense_fixture(worker, tmp_path):
    from posttrain_circuits.experiments.protocols import student_focus_preparation as p

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        head_dim=4,
        intermediate_size=16,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = {k: v.detach().clone().float() + 0.00037 for k, v in model.state_dict().items()}
    assert len(state) == 311
    identity = dict(
        preparation_protocol_sha256="a" * 64,
        preparation_protocol_artifact_sha256="b" * 64,
        prepared_initial=candidate_fixture(),
    )
    path = tmp_path / "dense.pt"
    producer = worker.sibling("sdsc_student_focus_worker")
    producer.save_dense(
        state,
        path,
        step=2,
        inputs={"mode": "fit", "protocol": {"sha256": "b" * 64}},
        dataset_sha=p.DATASET_MANIFEST_SHA256,
        protocol_sha="a" * 64,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    from posttrain_circuits.artifacts.checkpoints import torch_state_hash

    identity["prepared_initial"].update(
        checkpoint_sha256=worker.file_sha(path),
        checkpoint_size=path.stat().st_size,
        master_model_state_sha256=torch_state_hash(state),
    )
    model.bfloat16()
    loaded = SimpleNamespace(
        model=model,
        resolved_model_commit=worker.util.REVISION,
        resolved_tokenizer_commit=worker.util.REVISION,
        tokenizer_hash=worker.util.TOKENIZER_SHA,
        chat_template_sha256=worker.util.TEMPLATE_SHA,
        prompt_protocol="qwen3_non_thinking_v1",
    )
    return path, saved, identity, loaded


def test_actual_fp32_checkpoint_load_then_single_bf16_rounding(worker, tmp_path, monkeypatch):
    from posttrain_circuits.models import loading

    path, saved, identity, loaded = dense_fixture(worker, tmp_path)
    identity["prepared_initial"].update(
        checkpoint_sha256=worker.file_sha(path), checkpoint_size=path.stat().st_size
    )
    monkeypatch.setattr(loading, "load_model_and_tokenizer", lambda cfg, for_training: loaded)
    result, evidence = worker.load_prepared({}, path, identity, device=torch.device("cpu"))
    assert evidence["exact_saved_master_reload"] == dict(
        all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
    )
    assert evidence["all_parameters_frozen"] is True
    assert all(
        torch.equal(value, saved["model"][key].bfloat16()) for key, value in result.model.state_dict().items()
    )
    with torch.no_grad():
        logits = result.model(torch.tensor([[2, 3, 4]]), use_cache=False).logits
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "change",
    [
        "global_step",
        "mode",
        "protocol_sha256",
        "formal_initial_accepted",
        "model_precision",
        "model_nonfinite",
        "model_key",
        "model_extra_key",
        "model_shape",
        "master_hash",
        "format",
        "scope",
    ],
)
def test_corrupt_dense_metadata_or_master_state_rejected(worker, tmp_path, monkeypatch, change):
    from posttrain_circuits.models import loading

    path, saved, identity, loaded = dense_fixture(worker, tmp_path)
    if change == "global_step":
        saved[change] = 8
    elif change == "mode":
        saved[change] = "preflight"
    elif change == "protocol_sha256":
        saved[change] = "c" * 64
    elif change == "formal_initial_accepted":
        saved[change] = True
    elif change == "model_precision":
        saved["model"][next(iter(saved["model"]))] = next(iter(saved["model"].values())).bfloat16()
    elif change == "model_nonfinite":
        saved["model"][next(iter(saved["model"]))].flatten()[0] = float("nan")
    elif change == "model_key":
        saved["model"].pop(next(iter(saved["model"])))
    elif change == "model_extra_key":
        saved["model"]["unexpected.weight"] = torch.zeros(1)
    elif change == "model_shape":
        saved["model"][next(iter(saved["model"]))] = torch.zeros(1)
    elif change == "master_hash":
        identity["prepared_initial"]["master_model_state_sha256"] = "f" * 64
    else:
        saved[change] = "historical_v1_or_v2"
    torch.save(saved, path)
    identity["prepared_initial"].update(
        checkpoint_sha256=worker.file_sha(path), checkpoint_size=path.stat().st_size
    )
    monkeypatch.setattr(loading, "load_model_and_tokenizer", lambda cfg, for_training: loaded)
    with pytest.raises(ValueError):
        worker.load_prepared({}, path, identity, device=torch.device("cpu"))


def test_partial_failure_preserves_raw_and_never_claims_acceptance(worker, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(worker, "staged_science", lambda value, output: tmp_path)

    def fail(*args):
        worker.atomic(out / "qualification-report.json", dict(schema=worker.SCHEMA, raw_record_count=1))
        (out / "qualification-records.jsonl").write_text('{"ordinal":0}\n')
        raise ValueError("fixture interruption")

    monkeypatch.setattr(worker, "execute", fail)
    assert worker.main(["--inputs-json", str(inputs), "--output-dir", str(out)]) == 1
    report = json.loads((out / "qualification-report.json").read_text())
    assert (
        report["raw_record_count"] == 1
        and report["passed"] is report["execution_complete"] is report["qualification_passed"] is False
    )
    assert len(report["raw_artifacts"]) == 1 and all(report[k] is False for k in worker.FLAGS)


def test_cli_complete_scientific_failure_returns_two(worker, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(worker, "staged_science", lambda value, output: tmp_path)
    monkeypatch.setattr(worker, "execute", lambda *args: dict(passed=False, execution_complete=True))
    assert worker.main(["--inputs-json", str(inputs), "--output-dir", str(out)]) == 2


@pytest.mark.parametrize("outcome", ["pass", "validation_failure", "anti_gap_failure"])
def test_real_worker_execute_to_independent_auditor_all896_pinned_tokens(
    worker, tmp_path, monkeypatch, outcome
):
    """Actual producer report/raw formatting reaches independent replay in both outcomes.

    Only GPU/model loading and accepted-node input staging are replaced. The
    original scorers, producer wrapper, pinned tokenizer, parsers, report writer
    and independent auditor all execute without replacement.
    """
    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.experiments.protocols import student_focus_qualification as qualification
    from posttrain_circuits.experiments.protocols.student_preparation import CHAT_TEMPLATE_SHA256, DIFFICULTY

    spec = importlib.util.spec_from_file_location(
        "_actual_qualification_auditor", ROOT / "tools/sdsc_student_focus_qualify_audit.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    tokenizer = auditor.load_tokenizer()
    task = ProofGraphTask()
    validation = build_split(task, "validation", 128, 42, DIFFICULTY)
    iid = build_split(task, "iid_test", 128, 42, DIFFICULTY)
    cases = build_anti_shortcut_suite(iid, seed=42, distractor_ood_count=32)
    base_fails = outcome == "validation_failure"
    texts = ["invalid" if base_fails else task.canonical_target(x) for x in validation]
    texts += [task.canonical_target(x) for x in iid]
    texts += ["invalid" if outcome == "anti_gap_failure" else task.canonical_target(x.example) for x in cases]
    model = Responses(tokenizer, texts)
    model.config = SimpleNamespace(max_position_embeddings=32768)
    loaded = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        tokenizer_hash=worker.util.TOKENIZER_SHA,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        prompt_protocol="qwen3_non_thinking_v1",
        resolved_model_commit=worker.util.REVISION,
    )
    model_config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        }
    }
    candidate, parent, _, _, evidence = parent_evidence_fixture(worker, tmp_path / "fit-evidence")
    fit_evidence = worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)
    control = worker.sibling("sdsc_student_focus_qualify")
    control.ROOT = tmp_path / "explicit-future-protocol-fixture"
    protocol_path = control.ROOT / qualification.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    domain_path = control.ROOT / control.PROTOCOL_MODULE
    domain_path.parent.mkdir(parents=True)
    domain_path.write_bytes((ROOT / control.PROTOCOL_MODULE).read_bytes())
    payload = qualification.proposed_student_focus_qualification_protocol(candidate)
    protocol_path.write_bytes(worker.canonical(payload) + b"\n")
    protocol = dict(
        protocol_sha256=qualification.protocol_core_sha256(payload),
        artifact_sha256=worker.file_sha(protocol_path),
        head="a" * 40,
        implementation_commit="b" * 40,
        acceptance_commit="c" * 40,
        science_file_sha256={},
    )
    fit_plan = json.loads(Path(evidence["completion"]["fit_plan"]["path"]).read_bytes())
    plan = dict(
        protocol=protocol,
        fit=dict(
            plan=fit_plan,
            audit=control.audit_payload(Path(evidence["independent_audit"]["path"]).read_bytes(), candidate),
            completion=control.completion_records(
                json.loads(Path(evidence["completion"]["verified_status"]["path"]).read_bytes()),
                Path(evidence["completion"]["publication"]["path"]).read_bytes(),
            ),
        ),
        run_id="fixture",
        code_sha256="c" * 64,
    )
    identity = dict(
        prepared_initial=candidate,
        protocol_sha256=protocol["protocol_sha256"],
        protocol_artifact_sha256=protocol["artifact_sha256"],
        preparation_protocol_sha256=parent.protocol_sha256,
        preparation_protocol_artifact_sha256=parent.artifact_sha256,
        fit_evidence=fit_evidence,
        scientific_binding={
            key: protocol[key]
            for key in ("head", "implementation_commit", "acceptance_commit", "science_file_sha256")
        }
        | {"prereg_commit": "b" * 40},
        iid_examples_sha256="ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3",
        suite_sha256="c5ed272d5fe8f6dd21a7743dfca44d3bdfa06db44499842b7cc44df807b8ced0",
    )
    monkeypatch.setattr(
        worker,
        "load_inputs",
        lambda *args: ({"model": model_config}, tmp_path / "unused", validation, iid, cases, identity),
    )
    monkeypatch.setattr(
        worker,
        "runtime",
        lambda: dict(
            scope="Explicit CPU fixture; these runtime observations are synthetic, not GPU evidence",
            name="H100 fixture",
            logical_index=0,
            cpu_threads=24,
            torch="2.8.0+cu128",
            cuda_visible_devices="fixture",
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_prepared",
        lambda *args, **kwargs: (
            loaded,
            dict(
                forward_model_state_sha256=torch_state_hash(model.state_dict()),
                prepared_checkpoint_sha256=identity["prepared_initial"]["checkpoint_sha256"],
                master_model_state_sha256=identity["prepared_initial"]["master_model_state_sha256"],
                exact_saved_master_reload=dict(
                    all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
                ),
                forward_parameter_dtype="torch.bfloat16",
                all_parameters_frozen=True,
                gradient_checkpointing=False,
                use_cache=False,
            ),
        ),
    )
    output = tmp_path / "result"
    output.mkdir()
    inputs = dict(
        job_id="1",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=worker.digest(worker.canonical(plan)),
    )
    actual = worker.execute(inputs, output, tmp_path)
    published = json.loads((output / "qualification-report.json").read_text())
    assert (
        actual == published
        and published["raw_record_count"] == 896
        and published["execution_complete"] is True
        and published["task"] == control.TASK == "qwen3-v2-student-focus-qualification-v1"
        and published["schema"] == worker.SCHEMA
    )
    assert (output / "qualification-prompts.jsonl").stat().st_size == 13466230
    assert (
        worker.file_sha(output / "qualification-prompts.jsonl")
        == "ac58b320c219c8611943d84fa194c4658b24d641e665c6a9f3c9f0b0a783c51b"
    )
    prompts = [json.loads(s) for s in (output / "qualification-prompts.jsonl").read_text().splitlines()]
    records = [json.loads(s) for s in (output / "qualification-records.jsonl").read_text().splitlines()]
    result = auditor.replay(published, prompts, records, tokenizer, candidate=identity["prepared_initial"])
    assert (
        result["raw_replay_passed"] is True
        and result["prompts_replayed"] == result["responses_replayed"] == 896
    )
    assert result["qualification_passed"] is (outcome == "pass")
    assert published["anti_shortcut_passed"] is (outcome != "anti_gap_failure")
    assert published["base_gate_passed"] is (not base_fails)
    assert published["parameters_unchanged"] is True
    assert all(published[key] is False for key in worker.FLAGS)
    assert control.validate_worker_report(
        published,
        plan,
        "1",
        {row["path"]: row for row in published["raw_artifacts"]},
        require_passed=False,
    ) is (outcome == "pass")
    if outcome != "pass":
        with pytest.raises(ValueError, match="failed qualification"):
            control.validate_worker_report(
                published, plan, "1", {row["path"]: row for row in published["raw_artifacts"]}
            )


def test_changed_pure_helper_rejected_before_science_or_torch_import(worker, tmp_path):
    import shutil
    import subprocess
    import sys

    for name in (
        "sdsc_student_focus_qualify_worker.py",
        "sdsc_student_branch_qualify_worker.py",
        "sdsc_student_initial_probe.py",
    ):
        shutil.copyfile(Path(worker.__file__).with_name(name), tmp_path / name)
    target = tmp_path / "sdsc_student_branch_qualify_worker.py"
    target.write_bytes(target.read_bytes() + b"\n# changed immutable helper\n")
    script = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("changed_helper_boundary", sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except ValueError as error:
    assert str(error) == "frozen original qualification helper differs"
else:
    raise AssertionError("changed helper accepted")
assert "torch" not in sys.modules
assert not any(x == "posttrain_circuits" or x.startswith("posttrain_circuits.") for x in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path / "sdsc_student_focus_qualify_worker.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def candidate_fixture(**changes):
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    value = dict(
        fit_job_id="99900001",
        fit_intent="1" * 32,
        fit_source_head=q.PARENT_HEAD,
        fit_plan_sha256="2" * 64,
        fit_publication_sha256="3" * 64,
        fit_report_sha256="4" * 64,
        independent_raw_audit_sha256="5" * 64,
        checkpoint_step=2,
        checkpoint_path="checkpoints/step-00000002.pt",
        checkpoint_sha256="9" * 64,
        checkpoint_size=8127108953,
        master_model_state_sha256="6" * 64,
        **q.PREPARED_INITIAL_POLICY,
    )
    value.update(changes)
    return value


def completion_fixture(worker, candidate, report_size, isolation_record=None):
    """Synthetic previously-verified transport state; never a GPU acceptance claim."""
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    fit_plan = dict(
        schema="quest-sdsc-student-focus-plan-v1",
        task="qwen3-v2-student-focus-preparation-v5",
        mode="fit",
        intent_id=candidate["fit_intent"],
        run_id="fixture-parent",
        code_sha256="d" * 64,
        provenance={"head": candidate["fit_source_head"]},
        protocol=dict(
            protocol_sha256=q.PARENT_PROTOCOL_CORE_SHA256,
            artifact_sha256=q.PARENT_PROTOCOL_SHA256,
            implementation_commit=q.PARENT_IMPLEMENTATION_COMMIT,
            acceptance_commit=q.PARENT_HEAD,
            head=candidate["fit_source_head"],
            science_file_sha256={name: worker.file_sha(ROOT / name) for name in q.preparation.SCIENCE_PATHS},
        ),
        control_sha256=dict(q.FROZEN_EXECUTION_SHA256),
        **worker.FLAGS,
    )
    candidate["fit_plan_sha256"] = worker.digest(worker.canonical(fit_plan))
    publication = dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=fit_plan["task"],
        mode="fit",
        job_id=candidate["fit_job_id"],
        intent_id=candidate["fit_intent"],
        run_id=fit_plan["run_id"],
        code_sha256=fit_plan["code_sha256"],
        plan_sha256=candidate["fit_plan_sha256"],
        passed=True,
        stage_complete=True,
        preparation_complete=True,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=[dict(path="prepare-report.json", size=report_size, sha256=candidate["fit_report_sha256"])],
        large_files=[
            dict(
                path=candidate["checkpoint_path"],
                size=candidate["checkpoint_size"],
                sha256=candidate["checkpoint_sha256"],
            )
        ],
        **worker.FLAGS,
    )
    if isolation_record is not None:
        publication["files"].append(isolation_record)
    candidate["fit_publication_sha256"] = worker.digest(worker.canonical(publication))
    status = dict(
        accounting_complete=True,
        success=True,
        stage_complete=True,
        preparation_complete=True,
        publication_verified=True,
        artifact_hashes_verified=True,
        state="COMPLETED",
        job_id=candidate["fit_job_id"],
        queue=dict(returncode=0, stdout="", stderr=""),
        accounting=dict(returncode=0, stdout="synthetic completed-accounting fixture", stderr=""),
        result_sha256=candidate["fit_report_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
        publication=publication,
        **worker.FLAGS,
    )
    return {
        "fit_plan": worker.canonical(fit_plan),
        "verified_status": worker.canonical(status),
        "publication": worker.canonical(publication),
    }


def rewrite_parent(
    worker,
    work,
    candidate,
    report,
    audit,
    *,
    preserve_audit=(),
    audit_lf=True,
    isolation_record_override=None,
):
    """Rehash all outer records so each negative reaches the semantic boundary."""
    work.mkdir(parents=True, exist_ok=True)
    completion_fixture(worker, candidate, 1)
    report["plan_sha256"] = candidate["fit_plan_sha256"]
    report_path = work / "fit-report.json"
    report_path.write_bytes(worker.canonical(report))
    candidate["fit_report_sha256"] = worker.file_sha(report_path)
    isolation_path = work / "prepare-data-isolation.json"
    worker.atomic(isolation_path, report["data_audit"]["isolation"])
    isolation_record = dict(
        path=isolation_path.name,
        size=isolation_path.stat().st_size,
        sha256=worker.file_sha(isolation_path),
    )
    completion = completion_fixture(
        worker,
        candidate,
        report_path.stat().st_size,
        isolation_record if isolation_record_override is None else isolation_record_override,
    )
    for key, value in dict(
        result_sha256=candidate["fit_report_sha256"],
        plan_sha256=candidate["fit_plan_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
    ).items():
        if key not in preserve_audit:
            audit[key] = value
    audit_path = work / "fit-audit.json"
    audit_path.write_bytes(worker.canonical(audit) + (b"\n" if audit_lf else b""))
    candidate["independent_raw_audit_sha256"] = worker.file_sha(audit_path)

    def record(path):
        return dict(path=str(path), size=path.stat().st_size, sha256=worker.file_sha(path))

    completion_records = {}
    for name, raw in completion.items():
        path = work / ("fit-" + name.replace("_", "-") + ".json")
        path.write_bytes(raw)
        completion_records[name] = record(path)
    return dict(
        fit_job_id=candidate["fit_job_id"],
        publication_sha256=candidate["fit_publication_sha256"],
        report=record(report_path),
        independent_audit=record(audit_path),
        completion=completion_records,
    )


def parent_evidence_fixture(worker, work):
    """No real candidate: generated development counts plus explicit synthetic GPU evidence."""
    import copy

    from posttrain_circuits.experiments.protocols import student_focus_preparation as p
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    candidate = candidate_fixture()
    parent = SimpleNamespace(
        protocol_sha256=q.PARENT_PROTOCOL_CORE_SHA256,
        artifact_sha256=q.PARENT_PROTOCOL_SHA256,
        implementation_commit=q.PARENT_IMPLEMENTATION_COMMIT,
        acceptance_commit=q.PARENT_HEAD,
        science_file_sha256={"tools/sdsc_student_focus_audit.py": q.PARENT_AUDITOR_SHA256},
    )
    bases = p.make_examples(p.DIFFICULTY)["student_dev"]
    structure_sizes = {name: sum(x.metadata["structure"] == name for x in bases) for name in p.STRUCTURES}
    development, checkpoints = [], []
    for index, step in enumerate(p.CHECKPOINT_STEPS):
        count = 0 if step == 1 else 27
        counts = dict(num_examples=256, answer_correct=count, proof_correct=count, format_valid=256)
        structures = {
            name: dict(
                num_examples=structure_sizes[name],
                answer_correct=value if count else 0,
                proof_correct=value if count else 0,
                format_valid=structure_sizes[name],
            )
            for name, value in dict(branch=7, chain=9, converging_dag=11).items()
        }
        checkpoint_sha = f"{100 + index:064x}"
        development.append(
            dict(
                step=step,
                checkpoint_sha256=checkpoint_sha,
                examples_sha256=p.EXAMPLES_SHA256["student_dev"],
                **counts,
                per_view_counts={name: dict(counts) for name in p.DEV_VIEW_NAMES},
                structure_counts=structures,
            )
        )
        checkpoints.append(
            dict(
                step=step,
                path=f"checkpoints/step-{step:08d}.pt",
                size=candidate["checkpoint_size"],
                sha256=checkpoint_sha,
                scope="student_focus_preparation_model_only",
                reload_by_rank=[
                    dict(
                        rank=rank,
                        model_state_sha256=candidate["master_model_state_sha256"],
                        exact_saved_master_reload=dict(
                            all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
                        ),
                        root_export_logits=[
                            dict(bitwise_equal=True, max_abs_error=0, rms_error=0, argmax_mismatches=0),
                            dict(bitwise_equal=True, max_abs_error=0, rms_error=0, argmax_mismatches=0),
                        ],
                        inference_envelope_probe=dict(
                            passed=True,
                            input_length=2454,
                            token_id=17,
                            all_logits_finite=True,
                            use_cache=False,
                            native_bfloat16=True,
                            scope="synthetic_focus_context_no_development_content",
                        ),
                    )
                    for rank in (0, 1)
                ],
            )
        )
    selected = p.select_checkpoint(development)
    assert selected["step"] == 2
    candidate["checkpoint_sha256"] = selected["checkpoint_sha256"]
    windows = [
        47756,
        52480,
        53968,
        51224,
        46940,
        53348,
        51852,
        48916,
        51400,
        53816,
        51828,
        56828,
        48348,
        52468,
        50496,
        52072,
        53568,
        56800,
        49300,
        59284,
        52496,
        46996,
        47104,
        54372,
        48296,
        53312,
        50628,
        55640,
        52068,
        57604,
        51808,
        52048,
    ]
    assert sum(windows) == 1665064
    isolation = dict(
        passed=True,
        dataset_manifest_sha256=p.DATASET_MANIFEST_SHA256,
        scientific_acceptance=False,
        formal_generated_responses_or_scores_read=False,
        scope="synthetic previously verified producer consistency fixture",
    )
    report = dict(
        schema="quest-sdsc-student-focus-report-v1",
        mode="fit",
        job_id=candidate["fit_job_id"],
        passed=True,
        execution_complete=True,
        preparation_complete=True,
        optimizer_steps=32,
        consumed_fit_training_views=2048,
        fit_base_examples=256,
        fit_training_views=2048,
        development_base_examples=256,
        development_views=1536,
        training_input_tokens=1665064,
        world_size=2,
        global_batch_size=64,
        initial_checkpoint_sha256=worker.util.INITIAL_SHA,
        learning_rate=5e-5,
        protocol_sha256=parent.protocol_sha256,
        protocol_artifact_sha256=parent.artifact_sha256,
        dataset_sha256=p.DATASET_MANIFEST_SHA256,
        source_kind="symbolic_canonical_focus_preparation",
        teacher_data_required=False,
        data_audit=dict(
            passed=True,
            isolation=isolation,
            token_envelope=dict(
                fit_rows=2048,
                optimizer_windows=32,
                window_input_tokens=windows,
                total_input_tokens=1665064,
                token_budget=2000000,
            ),
        ),
        development=development,
        selected_checkpoint=selected,
        checkpoints=checkpoints,
        **worker.FLAGS,
    )
    audit = dict(
        schema="quest-sdsc-student-focus-raw-audit-v1",
        dataset_sha256=p.DATASET_MANIFEST_SHA256,
        raw_replay_passed=True,
        preparation_complete=True,
        prompts_replayed=1536,
        responses_replayed=18432,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
        mode="fit",
        job_id=candidate["fit_job_id"],
        auditor_sha256=q.PARENT_AUDITOR_SHA256,
        protocol_sha256=parent.protocol_sha256,
        protocol_artifact_sha256=parent.artifact_sha256,
        implementation_commit=parent.implementation_commit,
        acceptance_commit=parent.acceptance_commit,
        development=copy.deepcopy(development),
        selected_checkpoint=copy.deepcopy(selected),
        training_evidence=dict(
            training_order_reconstructed=True,
            training_rows_reconstructed=2048,
            consumed_training_rows=2048,
            optimizer_windows_replayed=32,
            full_population_input_tokens=1665064,
            training_input_tokens=1665064,
            checkpoint_cumulative_input_tokens={
                str(step): sum(windows[:step]) for step in p.CHECKPOINT_STEPS
            },
            gpu_training_independently_recomputed=False,
        ),
        isolation_evidence=dict(
            producer_evidence_sha256=worker.digest(worker.canonical(isolation) + b"\n"),
            dataset_manifest_sha256=p.DATASET_MANIFEST_SHA256,
            producer_isolation_consistency_verified=True,
            historical_isolation_independently_recomputed=False,
        ),
        **worker.FLAGS,
    )
    evidence = rewrite_parent(worker, work, candidate, report, audit)
    return candidate, parent, report, audit, evidence


def test_flat_parent_with_non4_selection_and_equal_accepted_source_head(worker, tmp_path):
    candidate, parent, report, audit, evidence = parent_evidence_fixture(worker, tmp_path)
    result = worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)
    assert result["selected_checkpoint"]["step"] == 2
    assert result["master_model_state_sha256"] == candidate["master_model_state_sha256"]
    assert set(result["completion"]) == {"fit_plan", "verified_status", "publication"}
    assert result["completion"] == {
        k: {n: v[n] for n in ("size", "sha256")} for k, v in evidence["completion"].items()
    }


@pytest.mark.parametrize("source", ["frozen_producer_atomic", "actual_preflight_metadata"])
def test_actual_isolation_bytes_bind_parent_publication_and_audit(worker, tmp_path, source):
    """Only isolation metadata is real; the future eligible fit remains a CPU fixture."""
    candidate, parent, report, audit, _ = parent_evidence_fixture(worker, tmp_path)
    path = tmp_path / "actual-producer-isolation.json"
    producer = worker.sibling("sdsc_student_focus_worker")
    if source == "actual_preflight_metadata":
        actual = ROOT / ".sdsc/fetched/54661792/fetch-m238fgjm/prepare-data-isolation.json"
        if not actual.is_file():
            pytest.skip("bounded historical producer metadata is not available")
        raw = actual.read_bytes()
        assert worker.digest(raw) == "ecdd79ddd3949b29befdf9dc0710a2ab6a009c7d7784700905a8a7456e799843"
        report["data_audit"]["isolation"] = json.loads(raw)
    producer.atomic(path, report["data_audit"]["isolation"])
    raw = path.read_bytes()
    assert raw == worker.canonical(report["data_audit"]["isolation"]) + b"\n"
    audit["isolation_evidence"]["producer_evidence_sha256"] = worker.file_sha(path)
    evidence = rewrite_parent(worker, tmp_path, candidate, report, audit)
    publication = worker.document(Path(evidence["completion"]["publication"]["path"]))
    row = next(r for r in publication["files"] if r["path"] == "prepare-data-isolation.json")
    assert row == dict(path="prepare-data-isolation.json", size=len(raw), sha256=worker.digest(raw))
    worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)


@pytest.mark.parametrize("mutation", ["canonical_without_lf", "wrong_size", "wrong_path"])
def test_rehashed_isolation_publication_rejects_changed_producer_bytes(worker, tmp_path, mutation):
    candidate, parent, report, audit, _ = parent_evidence_fixture(worker, tmp_path)
    raw = worker.canonical(report["data_audit"]["isolation"]) + b"\n"
    row = dict(path="prepare-data-isolation.json", size=len(raw), sha256=worker.digest(raw))
    if mutation == "canonical_without_lf":
        row.update(size=len(raw) - 1, sha256=worker.digest(raw[:-1]))
        audit["isolation_evidence"]["producer_evidence_sha256"] = row["sha256"]
    elif mutation == "wrong_size":
        row["size"] += 1
    else:
        row["path"] = "different-isolation.json"
    evidence = rewrite_parent(
        worker,
        tmp_path,
        candidate,
        report,
        audit,
        isolation_record_override=row,
    )
    with pytest.raises(ValueError, match="published retained-isolation bytes"):
        worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)


@pytest.mark.parametrize(
    "mutation",
    [
        "preflight",
        "failed",
        "null",
        "late_selection",
        "missing_step",
        "tokens",
        "old_schema",
        "old_source",
        "consumed256",
        "view_count",
        "raw_count",
        "audit_schema",
        "audit_result",
        "audit_newline",
        "training_prefix",
        "training_consumed",
        "training_gpu_claim",
        "isolation_hash",
        "isolation_claim",
        "rank_master",
        "rank_missing",
        "rank_parity",
        "rank_bool",
        "rank_context",
        "checkpoint_path",
        "checkpoint_scope",
    ],
)
def test_rehashed_parent_semantic_contradictions_are_rejected(worker, tmp_path, mutation):
    candidate, parent, report, audit, evidence = parent_evidence_fixture(worker, tmp_path)
    preserve = ()
    if mutation == "preflight":
        report["mode"] = "preflight"
    elif mutation == "failed":
        report["passed"] = False
    elif mutation == "null":
        report["selected_checkpoint"] = audit["selected_checkpoint"] = None
    elif mutation == "late_selection":
        report["selected_checkpoint"] = audit["selected_checkpoint"] = report["development"][2]
    elif mutation == "missing_step":
        report["checkpoints"].pop()
    elif mutation == "tokens":
        report["training_input_tokens"] -= 1
    elif mutation == "old_schema":
        report["schema"] = "quest-sdsc-student-order-report-v4"
    elif mutation == "old_source":
        report["source_kind"] = "symbolic_canonical_order_preparation"
    elif mutation == "consumed256":
        report["consumed_fit_training_views"] = 256
    elif mutation == "view_count":
        report["development_views"] = 768
    elif mutation == "raw_count":
        audit["responses_replayed"] -= 1
    elif mutation == "audit_schema":
        audit["schema"] = "quest-sdsc-student-order-raw-audit-v4"
    elif mutation == "audit_result":
        audit["result_sha256"] = "0" * 64
        preserve = ("result_sha256",)
    elif mutation == "training_prefix":
        audit["training_evidence"]["checkpoint_cumulative_input_tokens"]["2"] -= 1
    elif mutation == "training_consumed":
        audit["training_evidence"]["consumed_training_rows"] = 256
    elif mutation == "training_gpu_claim":
        audit["training_evidence"]["gpu_training_independently_recomputed"] = True
    elif mutation == "isolation_hash":
        audit["isolation_evidence"]["producer_evidence_sha256"] = "0" * 64
    elif mutation == "isolation_claim":
        audit["isolation_evidence"]["historical_isolation_independently_recomputed"] = True
    elif mutation == "rank_master":
        report["checkpoints"][1]["reload_by_rank"][1]["model_state_sha256"] = "f" * 64
    elif mutation == "rank_missing":
        report["checkpoints"][1]["reload_by_rank"].pop()
    elif mutation == "rank_parity":
        report["checkpoints"][1]["reload_by_rank"][1]["root_export_logits"][0]["bitwise_equal"] = False
    elif mutation == "rank_bool":
        report["checkpoints"][1]["reload_by_rank"][0]["rank"] = False
    elif mutation == "rank_context":
        report["checkpoints"][1]["reload_by_rank"][1]["inference_envelope_probe"]["input_length"] = 2244
    elif mutation == "checkpoint_path":
        report["checkpoints"][1]["path"] = "checkpoints/step-00000004.pt"
    elif mutation == "checkpoint_scope":
        report["checkpoints"][1]["scope"] = "common_student_order_model_only"
    else:
        assert mutation == "audit_newline"
    evidence = rewrite_parent(
        worker,
        tmp_path,
        candidate,
        report,
        audit,
        preserve_audit=preserve,
        audit_lf=mutation != "audit_newline",
    )
    with pytest.raises(ValueError):
        worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "bool_size",
        "oversize",
        "sha",
        "newline",
        "duplicate_json",
        "status_false",
        "source_head",
        "old_publication",
        "wrong_plan",
    ],
)
def test_completion_boundaries_reject_rehashed_bad_records(worker, tmp_path, mutation):
    candidate, parent, report, audit, evidence = parent_evidence_fixture(worker, tmp_path)
    records = evidence["completion"]
    if mutation == "missing":
        records.pop("publication")
    elif mutation == "extra":
        records["recovery"] = {}
    else:
        row = records["verified_status"]
        path = Path(row["path"])
        if mutation == "bool_size":
            row["size"] = True
        elif mutation == "oversize":
            row["size"] = worker.MAX_COMPLETION_BYTES + 1
        elif mutation == "sha":
            row["sha256"] = "0" * 64
        else:
            value = json.loads(path.read_bytes())
            if mutation == "newline":
                data = worker.canonical(value) + b"\n"
            elif mutation == "duplicate_json":
                data = b'{"success":true,"success":false}'
            else:
                if mutation == "status_false":
                    value["success"] = False
                elif mutation == "source_head":
                    candidate["fit_source_head"] = "f" * 40
                elif mutation == "wrong_plan":
                    candidate["fit_plan_sha256"] = "f" * 64
                elif mutation == "old_publication":
                    value["publication"]["schema"] = "quest-sdsc-student-order-execution-publication-v2"
                    pub = worker.canonical(value["publication"])
                    pr = records["publication"]
                    Path(pr["path"]).write_bytes(pub)
                    pr.update(size=len(pub), sha256=worker.digest(pub))
                    value["publication_sha256"] = candidate["fit_publication_sha256"] = worker.digest(pub)
                    evidence["publication_sha256"] = candidate["fit_publication_sha256"]
                data = worker.canonical(value)
            path.write_bytes(data)
            row.update(size=len(data), sha256=worker.digest(data))
    with pytest.raises(ValueError):
        worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)


def test_unbound_candidate_fails_before_model_loading(worker, tmp_path, monkeypatch):
    from posttrain_circuits.models import loading

    monkeypatch.setattr(
        loading,
        "load_model_and_tokenizer",
        lambda *a, **k: pytest.fail("unbound candidate reached model loading"),
    )
    with pytest.raises(ValueError, match="not bound"):
        worker.load_prepared(
            {}, tmp_path / "absent.pt", {"prepared_initial": None}, device=torch.device("cpu")
        )


def test_original_scoring_aliases_and_prior_exposure_disclosure_unchanged(worker):
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    assert worker.FROZEN_QUALIFICATION_HELPER_SHA256 == q.FROZEN_QUALIFICATION_HELPER_SHA256
    assert worker.score_original is worker._original.score_original
    assert worker.recording_generate is worker._original.recording_generate
    assert worker.GENERATION == worker._original.GENERATION
    assert (
        q.proposed_student_focus_qualification_protocol()["scope"]["formal_population_is_unexposed_holdout"]
        is False
    )


def test_node_dictionary_reaches_isolated_worker_with_flat_completion(worker, tmp_path):
    """Real14-input producer/consumer; future acceptance and large family are explicit CPU fixtures."""
    import copy
    import os
    import shutil
    import subprocess
    import sys

    from posttrain_circuits.artifacts.adapted_student_protocol import SCIENCE_PATHS
    from posttrain_circuits.experiments.protocols import student_focus_preparation as p
    from posttrain_circuits.experiments.protocols import student_focus_qualification as q

    control = worker.sibling("sdsc_student_focus_qualify")
    node = worker.sibling("sdsc_student_focus_qualify_job")
    science = tmp_path / "science"
    shutil.copytree(ROOT / "src", science / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in {*q.SCIENCE_PATHS, p.PROTOCOL_PATH, "prereg/qwen3_v2.yaml"}:
        target = science / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    originals = {name: worker.file_sha(science / name) for name in SCIENCE_PATHS}
    assert len(originals) == 49
    assert control.sha(control.canonical(originals)) == control._initial.NAMED_SCIENCE_MAP_SHA
    for name in ("config", "evidence", "checkpoints", "huggingface", "dataset/validation", "artifacts"):
        (tmp_path / name).mkdir(parents=True)
    shutil.copyfile(
        ROOT / ".sdsc/diagnostics/student-lr-v1/parent-resolved-config.yaml",
        tmp_path / "config/resolved_config.yaml",
    )
    candidate, parent, report, audit, _ = parent_evidence_fixture(worker, tmp_path / "evidence")
    checkpoint = tmp_path / "checkpoints/prepared-selected.pt"
    checkpoint.write_bytes(b"explicit model-load fixture boundary; actual record hash and path\n")
    candidate.update(checkpoint_sha256=worker.file_sha(checkpoint), checkpoint_size=checkpoint.stat().st_size)
    report["checkpoints"][1].update(sha256=candidate["checkpoint_sha256"], size=candidate["checkpoint_size"])
    report["development"][1]["checkpoint_sha256"] = candidate["checkpoint_sha256"]
    report["selected_checkpoint"] = p.select_checkpoint(report["development"])
    audit["development"] = copy.deepcopy(report["development"])
    audit["selected_checkpoint"] = copy.deepcopy(report["selected_checkpoint"])
    evidence = rewrite_parent(worker, tmp_path / "evidence", candidate, report, audit)
    fit_plan = json.loads(Path(evidence["completion"]["fit_plan"]["path"]).read_bytes())
    # Real node uses these exact staged filenames. The fixture rewrites only its own temporary files.
    shutil.copyfile(evidence["completion"]["fit_plan"]["path"], tmp_path / "evidence/fit-plan.json")
    payload = q.proposed_student_focus_qualification_protocol(candidate)
    protocol_path = science / q.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(worker.canonical(payload) + b"\n")
    binding = dict(
        payload=payload,
        protocol_sha256=q.protocol_core_sha256(payload),
        artifact_sha256=worker.file_sha(protocol_path),
        head="a" * 40,
        implementation_commit="b" * 40,
        acceptance_commit="c" * 40,
        science_file_sha256={name: worker.file_sha(science / name) for name in q.SCIENCE_PATHS},
    )
    parent.head = binding["head"]
    parent.science_file_sha256 = {name: worker.file_sha(science / name) for name in p.SCIENCE_PATHS}
    for name in ("dataset/manifest.json", "dataset/validation/examples.jsonl"):
        (tmp_path / name).write_bytes(b"explicit large-family fixture boundary\n")
    control.ROOT = science
    plan = dict(
        fit={"plan": fit_plan},
        run_id="20261004T220000Z-fixture",
        code_sha256="d" * 64,
        protocol=binding,
        prepared_checkpoint=control.checkpoint_row(candidate),
    )
    inputs = node.build_worker_inputs(plan, tmp_path, science, {"job_id": "123"}, originals, control)
    assert len(inputs) == 14 and inputs["schema"] == worker.INPUT_SCHEMA
    assert set(inputs["preparation_evidence"]["completion"]) == {"fit_plan", "verified_status", "publication"}
    (tmp_path / "inputs.json").write_bytes(worker.canonical(inputs))
    (tmp_path / "bindings.json").write_bytes(
        worker.canonical(dict(qualification=binding, preparation=vars(parent)))
    )
    script = r"""
import copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
work=Path(sys.argv[1]);source=work/"science/tools/sdsc_student_focus_qualify_worker.py"
spec=importlib.util.spec_from_file_location("isolated_focus_worker",source);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
inputs=json.loads((work/"inputs.json").read_bytes())
assert not any(x=="posttrain_circuits" or x.startswith("posttrain_circuits.") for x in sys.modules)
for mutation in ("extra","job","science"):
 bad=copy.deepcopy(inputs)
 if mutation=="extra":bad["unexpected"]=True
 elif mutation=="job":bad["job_id"]="124"
 else:bad["original_science_files"][next(iter(bad["original_science_files"]))]="0"*64
 try:worker.staged_science(bad,work/"artifacts")
 except ValueError:pass
 else:raise AssertionError("staging accepted "+mutation)
assert worker.staged_science(inputs,work/"artifacts")==work
from posttrain_circuits.artifacts import runs
from posttrain_circuits.datasets.proofgraph import family
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.experiments.protocols import student_focus_preparation as p
from posttrain_circuits.experiments.protocols import student_focus_qualification as q
bindings=json.loads((work/"bindings.json").read_bytes())
# Explicit fixture boundaries: no real future acceptance or large dataset is claimed.
q.resolve_student_focus_qualification_protocol=lambda *a,**k:SimpleNamespace(**bindings["qualification"])
p.resolve_student_focus_preparation_protocol=lambda *a,**k:SimpleNamespace(**bindings["preparation"])
runs.require_git_output=lambda args:"b"*40
splits={name:build_split(ProofGraphTask(),name,128,42,p.DIFFICULTY) for name in ("validation","iid_test")}
family.load_dataset_family=lambda path:SimpleNamespace(examples=splits.__getitem__,manifest={"sha256":"e"*64})
original_sha=worker.file_sha
fixture_hashes={work/"dataset/manifest.json":worker.util.FAMILY_SHA,work/"dataset/validation/examples.jsonl":worker.util.VALIDATION_SHA}
def checked_fixture_sha(path):
 path=Path(path)
 if path in fixture_hashes:
  assert path.read_bytes()==b"explicit large-family fixture boundary\n"
  return fixture_hashes[path]
 return original_sha(path)
worker.file_sha=checked_fixture_sha
config,checkpoint,validation,iid,cases,identity=worker.load_inputs(inputs,work)
assert checkpoint==work/"checkpoints/prepared-selected.pt"
assert len(validation)==len(iid)==128 and len(cases)==640
assert identity["fit_evidence"]["selected_checkpoint"]["step"]==2
assert identity["fit_evidence"]["selected_checkpoint"]["answer_correct"]==27
assert set(identity["fit_evidence"]["completion"])=={"fit_plan","verified_status","publication"}
for mutation in ("checkpoint_fields","report_digest","unbound"):
 bad=copy.deepcopy(inputs)
 if mutation=="checkpoint_fields":bad["prepared_checkpoint"]["unexpected"]=True
 elif mutation=="report_digest":bad["preparation_evidence"]["report"]["sha256"]="0"*64
 else:bindings["qualification"]["payload"]["prepared_initial"]=None
 try:worker.load_inputs(bad,work)
 except ValueError:pass
 else:raise AssertionError("nested binding accepted "+mutation)
print(json.dumps({"population":896,"selected_step":2,"input_count":len(inputs)}))
"""
    environment = dict(os.environ, SLURM_JOB_ID="123", HF_HOME=str(tmp_path / "huggingface"))
    for key in ("GIT_DIR", "GIT_WORK_TREE", "PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path)],
        cwd=science,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(population=896, selected_step=2, input_count=14)
