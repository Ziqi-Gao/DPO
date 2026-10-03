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


def candidate_fixture(**changes):
    value = dict(
        fit_job_id="99900001",
        fit_intent="1" * 32,
        fit_plan_sha256="2" * 64,
        fit_publication_sha256="3" * 64,
        fit_report_sha256="4" * 64,
        independent_raw_audit_sha256="5" * 64,
        checkpoint_step=2,
        checkpoint_path="checkpoints/step-00000002.pt",
        checkpoint_sha256="9" * 64,
        checkpoint_size=8127108889,
        master_model_state_sha256="6" * 64,
        selection="frozen_earliest_development_candidate_before_this_candidate_qualification",
        checkpoint_reselection_or_retraining=False,
        load="exact_FP32_saved_masters_before_one_native_BF16_inference_copy",
        identity="dense_checkpoint_bytes_never_native_HF_revision",
    )
    value.update(changes)
    return value


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_qualify", ROOT / "tools/sdsc_student_branch_qualify_worker.py"
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
    from posttrain_circuits.experiments.protocols import student_branch_preparation as p

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
    saved = dict(
        format="student_branch_dense_v3",
        model=state,
        global_step=2,
        parent_checkpoint_sha256=worker.util.INITIAL_SHA,
        dataset_sha256=p.DATASET_MANIFEST_SHA256,
        protocol_sha256="a" * 64,
        protocol_artifact_sha256="b" * 64,
        learning_rate=5e-5,
        scope="common_student_branch_model_only",
        **worker.FLAGS,
    )
    path = tmp_path / "dense.pt"
    torch.save(saved, path)
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
    from posttrain_circuits.experiments.protocols.student_preparation import CHAT_TEMPLATE_SHA256, DIFFICULTY

    spec = importlib.util.spec_from_file_location(
        "_actual_qualification_auditor", ROOT / "tools/sdsc_student_branch_qualify_audit.py"
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
    identity = dict(
        prepared_initial=candidate_fixture(),
        scientific_binding={"head": "a" * 40, "prereg_commit": "b" * 40},
        iid_examples_sha256="ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3",
        suite_sha256="c5ed272d5fe8f6dd21a7743dfca44d3bdfa06db44499842b7cc44df807b8ced0",
    )
    monkeypatch.setattr(
        worker,
        "load_inputs",
        lambda *args: ({"model": model_config}, tmp_path / "unused", validation, iid, cases, identity),
    )
    monkeypatch.setattr(worker, "runtime", lambda: {"scope": "CPU fixture; no GPU inference"})
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
    inputs = dict(job_id="1", run_id="fixture", source_code_sha256="c" * 64, plan_sha256="d" * 64)
    actual = worker.execute(inputs, output, tmp_path)
    published = json.loads((output / "qualification-report.json").read_text())
    assert (
        actual == published
        and published["raw_record_count"] == 896
        and published["execution_complete"] is True
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


def parent_evidence_fixture(worker, tmp_path):
    """Synthetic successful V3 result, with real population/selector and explicit fake GPU evidence."""
    from posttrain_circuits.experiments.protocols import student_branch_preparation as preparation

    candidate = candidate_fixture()
    parent = SimpleNamespace(
        protocol_sha256="a" * 64,
        artifact_sha256="b" * 64,
        implementation_commit="c" * 40,
        acceptance_commit="d" * 40,
        science_file_sha256={"tools/sdsc_student_branch_audit.py": "e" * 64},
    )
    bases = preparation.make_examples(preparation.DIFFICULTY)["student_dev"]
    structure_sizes = {
        name: sum(x.metadata["structure"] == name for x in bases) for name in preparation.STRUCTURES
    }
    development, checkpoints = [], []
    for index, step in enumerate(preparation.CHECKPOINT_STEPS):
        count = 0 if step == 1 else 26
        counts = dict(num_examples=256, answer_correct=count, proof_correct=count, format_valid=256)
        structures = {
            name: dict(
                num_examples=structure_sizes[name],
                answer_correct=value if count else 0,
                proof_correct=value if count else 0,
                format_valid=structure_sizes[name],
            )
            for name, value in dict(branch=11, chain=9, converging_dag=6).items()
        }
        checkpoint_sha = f"{100 + index:064x}"
        development.append(
            dict(
                step=step,
                checkpoint_sha256=checkpoint_sha,
                examples_sha256=preparation.EXAMPLES_SHA256["student_dev"],
                **counts,
                per_view_counts={name: dict(counts) for name in preparation.DEV_VIEW_NAMES},
                structure_counts=structures,
            )
        )
        checkpoints.append(
            dict(
                step=step,
                path=f"checkpoints/step-{step:08d}.pt",
                size=candidate["checkpoint_size"],
                sha256=checkpoint_sha,
                scope="common_student_branch_model_only",
                reload_by_rank=[
                    dict(
                        rank=rank,
                        model_state_sha256=candidate["master_model_state_sha256"],
                        exact_saved_master_reload=dict(
                            all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
                        ),
                        root_export_logits=[
                            dict(bitwise_equal=True, max_abs_error=0, rms_error=0, argmax_mismatches=0)
                        ]
                        * 2,
                    )
                    for rank in (0, 1)
                ],
            )
        )
    selected = preparation.select_checkpoint(development)
    assert selected["step"] == 2
    candidate["checkpoint_sha256"] = selected["checkpoint_sha256"]
    report = dict(
        schema="quest-sdsc-student-branch-report-v3",
        mode="fit",
        job_id=candidate["fit_job_id"],
        passed=True,
        execution_complete=True,
        preparation_complete=True,
        optimizer_steps=32,
        training_input_tokens=1652820,
        world_size=2,
        global_batch_size=64,
        initial_checkpoint_sha256=worker.util.INITIAL_SHA,
        learning_rate=5e-5,
        protocol_sha256=parent.protocol_sha256,
        protocol_artifact_sha256=parent.artifact_sha256,
        dataset_sha256=preparation.DATASET_MANIFEST_SHA256,
        plan_sha256=candidate["fit_plan_sha256"],
        development=development,
        selected_checkpoint=selected,
        checkpoints=checkpoints,
        **worker.FLAGS,
    )
    report_path = tmp_path / "fit-report.json"
    report_path.write_bytes(worker.canonical(report) + b"\n")
    candidate["fit_report_sha256"] = worker.file_sha(report_path)
    audit = dict(
        raw_replay_passed=True,
        prompts_replayed=1536,
        responses_replayed=18432,
        gpu_numerics_independently_recomputed=False,
        execution_acceptance_claim=False,
        mode="fit",
        job_id=candidate["fit_job_id"],
        publication_sha256=candidate["fit_publication_sha256"],
        plan_sha256=candidate["fit_plan_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        auditor_sha256=parent.science_file_sha256["tools/sdsc_student_branch_audit.py"],
        protocol_sha256=parent.protocol_sha256,
        protocol_artifact_sha256=parent.artifact_sha256,
        implementation_commit=parent.implementation_commit,
        acceptance_commit=parent.acceptance_commit,
        development=development,
        selected_checkpoint=selected,
        **worker.FLAGS,
    )
    audit_path = tmp_path / "fit-audit.json"
    audit_path.write_bytes(worker.canonical(audit) + b"\n")
    candidate["independent_raw_audit_sha256"] = worker.file_sha(audit_path)

    def record(path):
        return dict(path=str(path), size=path.stat().st_size, sha256=worker.file_sha(path))

    evidence = dict(
        fit_job_id=candidate["fit_job_id"],
        publication_sha256=candidate["fit_publication_sha256"],
        report=record(report_path),
        independent_audit=record(audit_path),
    )
    return candidate, parent, report, audit, evidence


def test_non4_selected_parent_has_complete_strict_raw_and_rank_master_binding(worker, tmp_path):
    candidate, parent, report, audit, evidence = parent_evidence_fixture(worker, tmp_path)
    result = worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)
    assert result["selected_checkpoint"]["step"] == 2
    assert result["master_model_state_sha256"] == candidate["master_model_state_sha256"]
    assert report["checkpoints"][0]["step"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "preflight",
        "failed",
        "incomplete",
        "null",
        "late_selection",
        "missing_step",
        "tokens",
        "raw_count",
        "gpuclaim",
        "executionclaim",
        "audit_result",
        "audit_protocol",
        "audit_implementation",
        "audit_acceptance",
        "audit_tool",
        "audit_summary",
        "checkpoint_path",
        "checkpoint_size",
        "rank_master",
        "rank_missing",
        "rank_parity",
        "audit_newline",
    ],
)
def test_parent_binding_rejects_rehashed_wrong_evidence(worker, tmp_path, corruption):
    candidate, parent, report, audit, evidence = parent_evidence_fixture(worker, tmp_path)
    if corruption == "preflight":
        report["mode"] = "preflight"
    elif corruption == "failed":
        report["passed"] = False
    elif corruption == "incomplete":
        report["execution_complete"] = False
    elif corruption == "null":
        report["selected_checkpoint"] = audit["selected_checkpoint"] = None
    elif corruption == "late_selection":
        report["selected_checkpoint"] = audit["selected_checkpoint"] = report["development"][2]
    elif corruption == "missing_step":
        report["checkpoints"].pop()
    elif corruption == "tokens":
        report["training_input_tokens"] -= 1
    elif corruption == "raw_count":
        audit["responses_replayed"] -= 1
    elif corruption == "gpuclaim":
        audit["gpu_numerics_independently_recomputed"] = True
    elif corruption == "executionclaim":
        audit["execution_acceptance_claim"] = True
    elif corruption.startswith("audit_") and corruption not in ("audit_summary", "audit_newline"):
        key = {
            "audit_result": "result_sha256",
            "audit_protocol": "protocol_sha256",
            "audit_implementation": "implementation_commit",
            "audit_acceptance": "acceptance_commit",
            "audit_tool": "auditor_sha256",
        }[corruption]
        audit[key] = "0" * len(audit[key])
    elif corruption == "audit_summary":
        import copy

        audit["development"] = copy.deepcopy(audit["development"])
        audit["development"][0]["answer_correct"] = 1
    elif corruption == "checkpoint_path":
        report["checkpoints"][1]["path"] = "checkpoints/step-00000004.pt"
    elif corruption == "checkpoint_size":
        report["checkpoints"][1]["size"] -= 1
    elif corruption == "rank_master":
        report["checkpoints"][1]["reload_by_rank"][1]["model_state_sha256"] = "f" * 64
    elif corruption == "rank_missing":
        report["checkpoints"][1]["reload_by_rank"].pop()
    elif corruption == "rank_parity":
        report["checkpoints"][1]["reload_by_rank"][1]["root_export_logits"][0]["bitwise_equal"] = False
    report_path = Path(evidence["report"]["path"])
    report_path.write_bytes(worker.canonical(report) + b"\n")
    candidate["fit_report_sha256"] = worker.file_sha(report_path)
    if corruption != "audit_result":
        audit["result_sha256"] = candidate["fit_report_sha256"]
    audit_path = Path(evidence["independent_audit"]["path"])
    audit_path.write_bytes(worker.canonical(audit) + (b"" if corruption == "audit_newline" else b"\n"))
    candidate["independent_raw_audit_sha256"] = worker.file_sha(audit_path)
    for key, path in (("report", report_path), ("independent_audit", audit_path)):
        evidence[key] = dict(path=str(path), size=path.stat().st_size, sha256=worker.file_sha(path))
    with pytest.raises(ValueError):
        worker.validate_preparation_evidence(evidence, tmp_path, candidate, parent)


def test_node_input_builder_reaches_isolated_worker_with_non4_candidate(worker, tmp_path):
    """Actual node dictionary→fresh worker; only future acceptance and large family are fixture boundaries."""
    import os
    import shutil
    import subprocess
    import sys

    from posttrain_circuits.artifacts.adapted_student_protocol import SCIENCE_PATHS
    from posttrain_circuits.experiments.protocols import student_branch_preparation as preparation
    from posttrain_circuits.experiments.protocols import student_branch_qualification as qualification

    control = worker.sibling("sdsc_student_branch_qualify")
    node = worker.sibling("sdsc_student_branch_qualify_job")
    science = tmp_path / "science"
    shutil.copytree(ROOT / "src", science / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in {*qualification.SCIENCE_PATHS, preparation.PROTOCOL_PATH, "prereg/qwen3_v2.yaml"}:
        target = science / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    original = {name: worker.file_sha(science / name) for name in SCIENCE_PATHS}
    assert len(original) == 49
    assert control.sha(control.canonical(original)) == control._initial.NAMED_SCIENCE_MAP_SHA
    for directory in ("config", "evidence", "checkpoints", "huggingface", "dataset/validation", "artifacts"):
        (tmp_path / directory).mkdir(parents=True)
    shutil.copyfile(
        ROOT / ".sdsc/diagnostics/student-lr-v1/parent-resolved-config.yaml",
        tmp_path / "config/resolved_config.yaml",
    )
    candidate, parent, report, audit, _ = parent_evidence_fixture(worker, tmp_path / "evidence")
    checkpoint = tmp_path / "checkpoints/prepared-selected.pt"
    checkpoint.write_bytes(b"explicit model-load fixture boundary; real record hash and path\n")
    candidate.update(checkpoint_sha256=worker.file_sha(checkpoint), checkpoint_size=checkpoint.stat().st_size)
    report["checkpoints"][1].update(sha256=candidate["checkpoint_sha256"], size=candidate["checkpoint_size"])
    report["development"][1]["checkpoint_sha256"] = candidate["checkpoint_sha256"]
    report["selected_checkpoint"] = preparation.select_checkpoint(report["development"])
    audit["selected_checkpoint"] = report["selected_checkpoint"]
    parent.protocol_sha256 = preparation.protocol_core_sha256(
        json.loads((science / preparation.PROTOCOL_PATH).read_bytes())
    )
    parent.artifact_sha256 = worker.file_sha(science / preparation.PROTOCOL_PATH)
    parent.implementation_commit = qualification.PARENT_IMPLEMENTATION_COMMIT
    parent.acceptance_commit = qualification.PARENT_HEAD
    parent.science_file_sha256 = {name: worker.file_sha(science / name) for name in preparation.SCIENCE_PATHS}
    report.update(protocol_sha256=parent.protocol_sha256, protocol_artifact_sha256=parent.artifact_sha256)
    report_path = tmp_path / "evidence/fit-report.json"
    report_path.write_bytes(worker.canonical(report) + b"\n")
    candidate["fit_report_sha256"] = worker.file_sha(report_path)
    audit.update(
        result_sha256=candidate["fit_report_sha256"],
        protocol_sha256=parent.protocol_sha256,
        protocol_artifact_sha256=parent.artifact_sha256,
        implementation_commit=parent.implementation_commit,
        acceptance_commit=parent.acceptance_commit,
        auditor_sha256=parent.science_file_sha256["tools/sdsc_student_branch_audit.py"],
    )
    audit_path = tmp_path / "evidence/fit-audit.json"
    audit_path.write_bytes(worker.canonical(audit) + b"\n")
    candidate["independent_raw_audit_sha256"] = worker.file_sha(audit_path)
    payload = qualification.proposed_student_branch_qualification_protocol(candidate)
    (science / qualification.PROTOCOL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (science / qualification.PROTOCOL_PATH).write_bytes(worker.canonical(payload) + b"\n")
    protocol = dict(
        payload=payload,
        protocol_sha256=qualification.protocol_core_sha256(payload),
        artifact_sha256=worker.file_sha(science / qualification.PROTOCOL_PATH),
        head="a" * 40,
        implementation_commit="b" * 40,
        acceptance_commit="c" * 40,
        science_file_sha256={name: worker.file_sha(science / name) for name in qualification.SCIENCE_PATHS},
    )
    parent.head = protocol["head"]
    for name in ("dataset/manifest.json", "dataset/validation/examples.jsonl"):
        (tmp_path / name).write_bytes(b"explicit large-family fixture boundary\n")
    # The real builder reads its protocol artifact; route this test instance to the isolated fixture root.
    control.ROOT = science
    plan = dict(
        fit={"plan": {"protocol": vars(parent)}},
        run_id="20261003T120000Z-fixture",
        code_sha256="d" * 64,
        protocol=protocol,
        prepared_checkpoint=control.checkpoint_row(candidate),
    )
    inputs = node.build_worker_inputs(plan, tmp_path, science, {"job_id": "123"}, original, control)
    (tmp_path / "inputs.json").write_bytes(control.canonical(inputs))
    (tmp_path / "bindings.json").write_bytes(
        control.canonical(dict(qualification=protocol, preparation=vars(parent)))
    )
    for args in (
        ["init", "--quiet"],
        ["add", "prereg/qwen3_v2.yaml"],
        ["commit", "--quiet", "-m", "fixture"],
    ):
        subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-C",
                str(science),
                *args,
            ],
            check=True,
        )
    script = r"""
import copy, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
work=Path(sys.argv[1]);source=work/"science/tools/sdsc_student_branch_qualify_worker.py"
spec=importlib.util.spec_from_file_location("fresh_worker",source);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
inputs=json.loads((work/"inputs.json").read_bytes())
assert not any(k=="posttrain_circuits" or k.startswith("posttrain_circuits.") for k in sys.modules)
for mutation in ("extra","job","science"):
 bad=copy.deepcopy(inputs)
 if mutation=="extra":bad["unexpected"]=True
 elif mutation=="job":bad["job_id"]="124"
 else:bad["original_science_files"][next(iter(bad["original_science_files"]))]="0"*64
 try:worker.staged_science(bad,work/"artifacts")
 except ValueError:pass
 else:raise AssertionError("staging accepted "+mutation)
assert worker.staged_science(inputs,work/"artifacts")==work
from posttrain_circuits.datasets.proofgraph import family
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.experiments.protocols import student_branch_preparation as prep
from posttrain_circuits.experiments.protocols import student_branch_qualification as qual
bindings=json.loads((work/"bindings.json").read_bytes())
# Only unissued future acceptance is replaced; candidate and every nested evidence validator run unchanged.
qual.resolve_student_branch_qualification_protocol=lambda *a,**k:SimpleNamespace(**bindings["qualification"])
prep.resolve_student_branch_preparation_protocol=lambda *a,**k:SimpleNamespace(**bindings["preparation"])
splits={name:build_split(ProofGraphTask(),name,128,42,prep.DIFFICULTY) for name in ("validation","iid_test")}
family.load_dataset_family=lambda path:SimpleNamespace(examples=splits.__getitem__,manifest={"sha256":"e"*64})
original_sha=worker.file_sha
remote_hashes={work/"dataset/manifest.json":worker.util.FAMILY_SHA,work/"dataset/validation/examples.jsonl":worker.util.VALIDATION_SHA}
def artifact_sha(path):
 path=Path(path)
 if path in remote_hashes:
  assert path.read_bytes()==b"explicit large-family fixture boundary\n"
  return remote_hashes[path]
 return original_sha(path)
worker.file_sha=artifact_sha
config,checkpoint,validation,iid,cases,identity=worker.load_inputs(inputs,work)
assert checkpoint==work/"checkpoints/prepared-selected.pt"
assert len(validation)==len(iid)==128 and len(cases)==640
assert identity["fit_evidence"]["selected_checkpoint"]["step"]==2
assert identity["fit_evidence"]["selected_checkpoint"]["answer_correct"]==26
assert identity["prepared_initial"]==bindings["qualification"]["payload"]["prepared_initial"]
for mutation in ("checkpoint_schema","parent_sha","unbound"):
 bad=copy.deepcopy(inputs)
 if mutation=="checkpoint_schema":bad["prepared_checkpoint"]["unexpected"]=True
 elif mutation=="parent_sha":bad["preparation_evidence"]["report"]["sha256"]="0"*64
 else:bindings["qualification"]["payload"]["prepared_initial"]=None
 try:worker.load_inputs(bad,work)
 except ValueError:pass
 else:raise AssertionError("nested validation accepted "+mutation)
print(json.dumps({"validated_records":896,"selected_step":2}))
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
    assert json.loads(result.stdout)["selected_step"] == 2
