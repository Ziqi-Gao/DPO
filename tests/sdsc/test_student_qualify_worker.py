"""Bounded qualification regressions using original scorers and actual CPU state loads."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
        "_test_qualify", ROOT / "tools/sdsc_student_qualify_worker.py"
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
    from posttrain_circuits.experiments.protocols import student_preparation as p

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
    identity = dict(preparation_protocol_sha256="a" * 64, preparation_protocol_artifact_sha256="b" * 64)
    saved = dict(
        format="student_preparation_dense_v1",
        model=state,
        global_step=4,
        parent_checkpoint_sha256=worker.util.INITIAL_SHA,
        dataset_sha256=p.DATASET_MANIFEST_SHA256,
        protocol_sha256="a" * 64,
        protocol_artifact_sha256="b" * 64,
        learning_rate=5e-5,
        scope="common_student_preparation_model_only",
        **worker.FLAGS,
    )
    path = tmp_path / "dense.pt"
    torch.save(saved, path)
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
    from posttrain_circuits.experiments.protocols import student_qualification as p
    from posttrain_circuits.models import loading

    path, saved, identity, loaded = dense_fixture(worker, tmp_path)
    monkeypatch.setattr(p, "CHECKPOINT_SHA256", worker.file_sha(path))
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
    ],
)
def test_corrupt_dense_metadata_or_master_state_rejected(worker, tmp_path, monkeypatch, change):
    from posttrain_circuits.experiments.protocols import student_qualification as p
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
    else:
        saved["model"].pop(next(iter(saved["model"])))
    torch.save(saved, path)
    monkeypatch.setattr(p, "CHECKPOINT_SHA256", worker.file_sha(path))
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


@pytest.mark.parametrize("base_fails", [False, True])
def test_real_worker_execute_to_independent_auditor_all896_pinned_tokens(
    worker, tmp_path, monkeypatch, base_fails
):
    """Actual producer report/raw formatting reaches independent replay in both outcomes.

    Only GPU/model loading and accepted-node input staging are replaced. The
    original scorers, producer wrapper, pinned tokenizer, parsers, report writer
    and independent auditor all execute without replacement.
    """
    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.experiments.protocols.student_preparation import CHAT_TEMPLATE_SHA256, DIFFICULTY

    spec = importlib.util.spec_from_file_location(
        "_actual_qualification_auditor", ROOT / "tools/sdsc_student_qualify_audit.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    tokenizer = auditor.load_tokenizer()
    task = ProofGraphTask()
    validation = build_split(task, "validation", 128, 42, DIFFICULTY)
    iid = build_split(task, "iid_test", 128, 42, DIFFICULTY)
    cases = build_anti_shortcut_suite(iid, seed=42, distractor_ood_count=32)
    texts = ["invalid" if base_fails else task.canonical_target(x) for x in validation]
    texts += [task.canonical_target(x) for x in iid] + [task.canonical_target(x.example) for x in cases]
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
            {"forward_model_state_sha256": torch_state_hash(model.state_dict())},
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
    prompts = [json.loads(s) for s in (output / "qualification-prompts.jsonl").read_text().splitlines()]
    records = [json.loads(s) for s in (output / "qualification-records.jsonl").read_text().splitlines()]
    result = auditor.replay(published, prompts, records, tokenizer)
    assert (
        result["raw_replay_passed"] is True
        and result["prompts_replayed"] == result["responses_replayed"] == 896
    )
    assert result["qualification_passed"] is (not base_fails)
    assert published["anti_shortcut_passed"] is True and published["base_gate_passed"] is (not base_fails)
    assert published["parameters_unchanged"] is True
    assert all(published[key] is False for key in worker.FLAGS)


def test_node_input_builder_reaches_isolated_worker_source_and_input_validation(worker, tmp_path):
    """The genuine producer dictionary reaches the real consumer in a fresh interpreter.

    This tests the staging/worker seam, not GPU loading or acceptance. Historical
    small artifacts and source bytes are genuine. Explicit fixture boundaries
    replace only future qualification acceptance, the remote 8 GB checkpoint,
    and loading all 144k family examples. Population generation and hashes remain
    real, as do every nested input-record and parent-evidence validation.
    """
    from posttrain_circuits.artifacts.adapted_student_protocol import SCIENCE_PATHS
    from posttrain_circuits.experiments.protocols import student_qualification as qualification

    control = worker.sibling("sdsc_student_qualify")
    node = worker.sibling("sdsc_student_qualify_job")
    science = tmp_path / "science"
    shutil.copytree(ROOT / "src", science / "src", ignore=shutil.ignore_patterns("__pycache__"))
    paths = {*qualification.SCIENCE_PATHS, qualification.PROTOCOL_PATH, control._prep.PROTOCOL_PATH}
    paths.add("prereg/qwen3_v2.yaml")
    for name in paths:
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
    assert worker.file_sha(tmp_path / "config/resolved_config.yaml") == worker.util.CONFIG_SHA
    shutil.copyfile(
        ROOT / ".sdsc/fetched/54562507/fetch-hmcmblh_/prepare-report.json",
        tmp_path / "evidence/fit-report.json",
    )
    shutil.copyfile(
        ROOT / ".sdsc/diagnostics/student-preparation-v1/fit-actual-independent-audit-20261002.json",
        tmp_path / "evidence/fit-audit.json",
    )
    for name in (
        "checkpoints/prepared-step4.pt",
        "dataset/manifest.json",
        "dataset/validation/examples.jsonl",
    ):
        (tmp_path / name).write_bytes(b"explicit remote-artifact fixture boundary\n")
    parent = json.loads(
        (ROOT / ".sdsc/student-prepare/524f6a6e8f1541502f315b820880ea4c/plan.json").read_text()
    )
    payload = qualification.proposed_student_qualification_protocol()
    protocol = dict(
        payload=payload,
        protocol_sha256=qualification.protocol_core_sha256(payload),
        artifact_sha256=worker.file_sha(science / qualification.PROTOCOL_PATH),
        head="a" * 40,
        implementation_commit="b" * 40,
        acceptance_commit="c" * 40,
        science_file_sha256={name: worker.file_sha(science / name) for name in qualification.SCIENCE_PATHS},
    )
    plan = dict(
        fit={"plan": parent},
        run_id="20261002T010000Z-fixture",
        code_sha256="d" * 64,
        protocol=protocol,
    )
    inputs = node.build_worker_inputs(plan, tmp_path, science, {"job_id": "123"}, original, control)
    (tmp_path / "inputs.json").write_bytes(control.canonical(inputs))
    (tmp_path / "bindings.json").write_bytes(
        control.canonical(dict(qualification=protocol, preparation=parent["protocol"]))
    )
    for argv in (
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
                *argv,
            ],
            check=True,
            capture_output=True,
        )
    script = r"""
import copy, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
work = Path(sys.argv[1])
source = work / "science/tools/sdsc_student_qualify_worker.py"
spec = importlib.util.spec_from_file_location("fresh_qualification_worker", source)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
inputs = json.loads((work / "inputs.json").read_text())
assert not any(k == "posttrain_circuits" or k.startswith("posttrain_circuits.") for k in sys.modules)
for mutation in ("extra", "job", "science"):
    bad = copy.deepcopy(inputs)
    if mutation == "extra":
        bad["unexpected"] = True
    elif mutation == "job":
        bad["job_id"] = "124"
    else:
        bad["original_science_files"][next(iter(bad["original_science_files"]))] = "0" * 64
    try:
        worker.staged_science(bad, work / "artifacts")
    except ValueError:
        pass
    else:
        raise AssertionError("staging accepted mutation " + mutation)
assert worker.staged_science(inputs, work / "artifacts") == work
from posttrain_circuits.datasets.proofgraph import family
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.experiments.protocols import student_preparation as prep
from posttrain_circuits.experiments.protocols import student_qualification as qual
bindings = json.loads((work / "bindings.json").read_text())
qual.resolve_student_qualification_protocol = lambda *a, **k: SimpleNamespace(**bindings["qualification"])
prep.resolve_student_preparation_protocol = lambda *a, **k: SimpleNamespace(**bindings["preparation"])
splits = {
    name: build_split(ProofGraphTask(), name, 128, 42, prep.DIFFICULTY)
    for name in ("validation", "iid_test")
}
family.load_dataset_family = lambda path: SimpleNamespace(
    examples=splits.__getitem__, manifest={"sha256": "e" * 64}
)
original_sha, original_verified = worker.file_sha, worker.util.verified
remote_hashes = {
    work / "dataset/manifest.json": worker.util.FAMILY_SHA,
    work / "dataset/validation/examples.jsonl": worker.util.VALIDATION_SHA,
}
def artifact_sha(path):
    path = Path(path)
    if path in remote_hashes:
        assert path.read_bytes() == b"explicit remote-artifact fixture boundary\n"
        return remote_hashes[path]
    return original_sha(path)
def verified(row, root, digest, size):
    if row["path"] == str(work / "checkpoints/prepared-step4.pt"):
        assert root == work and digest == qual.CHECKPOINT_SHA256 and size == qual.CHECKPOINT_SIZE
        assert row["sha256"] == digest and row["size"] == size
        return Path(row["path"])
    return original_verified(row, root, digest, size)
worker.file_sha, worker.util.verified = artifact_sha, verified
config, checkpoint, validation, iid, cases, identity = worker.load_inputs(inputs, work)
assert checkpoint == work / "checkpoints/prepared-step4.pt"
assert len(validation) == len(iid) == 128 and len(cases) == 640
assert identity["fit_evidence"]["selected_checkpoint"]["answer_correct"] == 175
assert identity["protocol_sha256"] == bindings["qualification"]["protocol_sha256"]
assert identity["preparation_protocol_artifact_sha256"] == qual.PARENT_PROTOCOL_SHA256
bad = copy.deepcopy(inputs)
bad["prepared_checkpoint"]["unexpected"] = True
try:
    worker.load_inputs(bad, work)
except ValueError as error:
    assert "file record fields differ" in str(error)
else:
    raise AssertionError("nested checkpoint schema was not checked")
print(json.dumps({"validated_records": 896, "fit_job_id": identity["fit_evidence"]["fit_job_id"]}))
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
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"validated_records": 896, "fit_job_id": "54562507"}
