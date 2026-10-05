"""Real producer/replay boundaries; CPU tiny models do not certify GPU execution."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    spec = importlib.util.spec_from_file_location("_cue_test_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("sdsc_student_name_cue_probe_worker")
audit = load("sdsc_student_name_cue_probe_audit")


@lru_cache(maxsize=1)
def fixture_records():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    views = p.development_views(p.make_examples()["student_dev"])
    return [
        dict(
            ordinal=i,
            block=v.block,
            condition=v.condition,
            structure=v.example.metadata["structure"],
            source_example_id=v.source_example_id,
            verification=dict(reward=1.0, answer_correct=True, parse_valid=True),
            stop_reason="eos",
        )
        for i, v in enumerate(views)
    ]


def fixture_data_audit():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    data = p.proposed_student_name_cue_probe_protocol()["data"]
    isolation = dict(
        passed=True,
        scientific_acceptance=False,
        formal_generated_responses_or_scores_read=False,
        dataset_manifest_sha256=p.DATASET_MANIFEST_SHA256,
        counts=copy.deepcopy(data["isolation_population_counts"]),
        excluded_view_counts=copy.deepcopy(data["isolation_excluded_view_counts"]),
        dimensions=copy.deepcopy(data["isolation_dimensions"]),
        population_stream_sha256={k: "1" * 64 for k in data["isolation_population_counts"]},
        excluded_view_stream_sha256={k: "2" * 64 for k in data["isolation_excluded_view_counts"]},
    )
    token = dict(
        passed=True,
        scientific_acceptance=False,
        matched_prefix_and_canonical_EOS_lengths=True,
        views=512,
        bases=64,
        maximum_prefix_tokens=8,
        maximum_canonical_response_tokens=12,
        maximum_prefix_plus_cap=264,
        input_token_lengths_sha256="3" * 64,
    )
    return dict(passed=True, isolation=isolation, token_audit=token)


def valid_report(records=None):
    """Pure synthetic execution report for controller rejection fixtures, no GPU claim."""
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    records = fixture_records() if records is None else records
    models = {}
    exact = dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True)
    parity = dict(
        shape=[1, 8, 151936],
        elements=8 * 151936,
        compared_positions=8,
        bitwise_equal=True,
        argmax_mismatches=0,
        max_abs_error=0.0,
        rms_error=0.0,
        formal_parity_gate_applied=False,
    )
    for arm in worker.ARMS:
        d = p.SOURCE_CHECKPOINTS[arm]
        ranks = [
            dict(
                rank=rank,
                reference_saved_master_reload=copy.deepcopy(exact),
                exact_saved_master_reload=copy.deepcopy(exact),
                model_state_sha256=d["model_state_sha256"],
                native_bf16_reload_logits=[copy.deepcopy(parity), copy.deepcopy(parity)],
                inference_envelope_probe=dict(
                    passed=True,
                    input_length=2454,
                    token_id=17,
                    all_logits_finite=True,
                    native_bfloat16=True,
                    use_cache=False,
                    scope="synthetic_diagnostic_context_no_development_content",
                ),
                counts=worker.counts([records[i] for i in worker.ordinals(rank)]),
                completed_ordinals=worker.ordinals(rank),
                generation_seconds=1.0,
            )
            for rank in (0, 1)
        ]
        models[arm] = dict(
            arm=arm,
            source_job_id=d["job_id"],
            source_step=32,
            learning_rate=d["learning_rate"],
            checkpoint_sha256=d["sha256"],
            checkpoint_size=d["size"],
            model_state_sha256=d["model_state_sha256"],
            ranks=ranks,
            readouts=worker.reduce_records(records),
        )
    names = [
        "name-cue-dataset-manifest.json",
        "name-cue-data-isolation.json",
        "name-cue-token-audit.json",
        "name-cue-prompts.jsonl",
    ] + [f"name-cue-records-{a}-rank-{rank}.jsonl" for a in worker.ARMS for rank in (0, 1)]
    return dict(
        schema=worker.SCHEMA,
        task=worker.TASK,
        mode="probe",
        job_id="1234",
        run_id="fixture",
        source_code_sha256="1" * 64,
        plan_sha256="2" * 64,
        protocol_sha256="3" * 64,
        protocol_artifact_sha256="4" * 64,
        dataset_sha256=p.DATASET_MANIFEST_SHA256,
        passed=True,
        diagnostic_complete=True,
        execution_complete=True,
        preparation_complete=False,
        inference_only=True,
        optimizer_steps=0,
        training_input_tokens=0,
        world_size=2,
        base_examples=64,
        views_per_model=512,
        total_responses=1024,
        selected_checkpoint=None,
        **worker.FLAGS,
        data_audit=fixture_data_audit(),
        models=models,
        raw_artifacts=[dict(path=n, size=1, sha256="5" * 64) for n in sorted(names)],
    )


def checkpoint_inputs():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    return {
        a: {k: d[k] for k in ("path", "size", "sha256", "job_id", "step", "learning_rate")}
        for a, d in p.SOURCE_CHECKPOINTS.items()
    }


def test_report_execution_only_positive_and_rank_balanced():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    report = valid_report()
    checked_protocol = json.loads((ROOT / p.PROTOCOL_PATH).read_text())
    assert checked_protocol == p.proposed_student_name_cue_probe_protocol()
    worker.validate_report(report, protocol=checked_protocol, checkpoints=checkpoint_inputs())
    rows = fixture_records()
    for rank in (0, 1):
        assert worker.ordinals(rank) == p.generation_ordinals(rank)
        for block in (0, 1):
            for condition in worker.CONDITIONS:
                selected = [
                    rows[i]
                    for i in worker.ordinals(rank)
                    if rows[i]["block"] == block and rows[i]["condition"] == condition
                ]
                assert sum(x["structure"] == "branch" for x in selected) == 24
                assert sum(x["structure"] == "chain" for x in selected) == 8


@pytest.mark.parametrize(
    "change",
    [
        "training",
        "selection",
        "acceptance",
        "missing_model",
        "wrong_lr",
        "wrong_sha",
        "master",
        "rank",
        "parity",
        "context",
        "extra_field",
        "artifact",
        "cell",
        "contrast",
        "interaction",
        "count_bool",
        "time_nan",
    ],
)
def test_report_corruptions_rejected(change):
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    r = valid_report()
    m = r["models"]["treatment"]
    z = m["ranks"][0]
    if change == "training":
        r["optimizer_steps"] = 1
    elif change == "selection":
        r["selected_checkpoint"] = {"step": 32}
    elif change == "acceptance":
        r["formal_initial_accepted"] = True
    elif change == "missing_model":
        del r["models"]["control"]
    elif change == "wrong_lr":
        m["learning_rate"] = 5e-5
    elif change == "wrong_sha":
        m["checkpoint_sha256"] = "0" * 64
    elif change == "master":
        m["model_state_sha256"] = "0" * 64
    elif change == "rank":
        z["completed_ordinals"][0] = 1
    elif change == "parity":
        z["native_bf16_reload_logits"][0]["bitwise_equal"] = False
    elif change == "context":
        z["inference_envelope_probe"]["input_length"] = 2198
    elif change == "extra_field":
        r["selected_model"] = "control"
    elif change == "artifact":
        r["raw_artifacts"].pop()
    elif change == "cell":
        m["readouts"]["cells"]["1"]["break_paths"]["branch"]["proof_correct"] -= 1
    elif change == "contrast":
        m["readouts"]["contrasts"]["0"]["branch"]["pair_cue_paths_present"]["proof_correct"][
            "difference_numerator"
        ] = 1
    elif change == "interaction":
        m["readouts"]["contrasts"]["0"]["branch"]["interaction"]["proof_correct"]["numerator"] = 1
    elif change == "count_bool":
        z["counts"]["proof_correct"] = True
    elif change == "time_nan":
        z["generation_seconds"] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        worker.validate_report(
            r, protocol=p.proposed_student_name_cue_probe_protocol(), checkpoints=checkpoint_inputs()
        )


@pytest.fixture(scope="module")
def population():
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    tokenizer = audit.load_tokenizer()
    views = p.development_views(p.make_examples()["student_dev"])
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    rows, prompts, tokens = worker.prepare_views(views, tokenizer, config)
    return tokenizer, views, rows, prompts, tokens


@pytest.fixture(scope="module")
def independent_population(population):
    return audit.population(population[0])


@pytest.fixture(scope="module")
def produced(population, tmp_path_factory):
    """Actual worker generation loop calls a real tiny CPU tensor model, then audits1024."""
    import torch

    tokenizer, views, rows, prompts, tokens = population

    class TinyOneStepModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.25, 1.0]), requires_grad=False)
            self.config = SimpleNamespace(max_position_embeddings=2454)

        def forward(self, ids):
            return self.weight * ids.float().mean().clamp(min=1)

        def generate(
            self,
            *,
            input_ids,
            attention_mask,
            max_new_tokens,
            do_sample,
            use_cache,
            pad_token_id,
            eos_token_id,
        ):
            assert (
                max_new_tokens == 256
                and do_sample is False
                and use_cache is False
                and attention_mask.dtype == torch.bool
            )
            assert self.forward(input_ids).argmax().item() == 1
            return torch.cat((input_ids, torch.tensor([[eos_token_id]], device=input_ids.device)), dim=1)

    output = tmp_path_factory.mktemp("actual-cue-producer")
    inputs = {
        k: v
        for k, v in valid_report().items()
        if k in ("job_id", "run_id", "plan_sha256", "source_code_sha256")
    }
    records = {}
    for arm in worker.ARMS:
        records[arm] = [
            worker.generate_records(
                TinyOneStepModel(),
                tokenizer,
                views,
                rows,
                prompts,
                arm=arm,
                rank=rank,
                output=output,
                inputs=inputs,
            )
            for rank in (0, 1)
        ]
    allrows = sorted([r for group in records["control"] for r in group], key=lambda r: r["ordinal"])
    report = valid_report(allrows)
    report["data_audit"]["token_audit"] = tokens
    return report, prompts, records, output


def test_actual_tiny_worker_all1024_rows_to_original_auditor(population, produced):
    report, prompts, records, output = produced
    result = audit.replay(report, prompts, records, population[0])
    assert result["raw_replay_passed"] and result["responses_replayed"] == 1024
    assert result["selected_checkpoint"] is None and not result["formal_initial_accepted"]
    assert len(list(output.glob("name-cue-records-*.jsonl"))) == 4
    assert all(len(path.read_text().splitlines()) == 256 for path in output.glob("name-cue-records-*.jsonl"))
    for arm in worker.ARMS:
        assert result["models"][arm] == report["models"][arm]["readouts"]
        failures = result["failed_prefix_diagnosis"][arm]
        assert len(failures) == 512
        assert all(row["condition"] in worker.CONDITIONS for row in failures)
        assert all(row["error_category"] for row in failures)


@pytest.mark.parametrize(
    "change",
    [
        "arm",
        "source_job",
        "checkpoint",
        "lr",
        "block",
        "condition",
        "rank",
        "ordinal",
        "prompt",
        "response_text",
        "early_eos",
        "bool_token",
        "short_stop",
        "verifier",
        "parsed",
        "tags",
        "extra",
        "missing",
        "swapped_rows",
        "canonical_length",
        "readout",
    ],
)
def test_original_raw_auditor_rejects_mismatches(
    population, independent_population, produced, monkeypatch, change
):
    report, prompts, records, _ = produced
    report, prompts, records = copy.deepcopy(report), copy.deepcopy(prompts), copy.deepcopy(records)
    expected, examples, token_audit, dataset = independent_population
    monkeypatch.setattr(audit, "population", lambda _: (expected, examples, token_audit, dataset))
    r = records["treatment"][0][0]
    if change == "arm":
        r["arm"] = "control"
    elif change == "source_job":
        r["source_job_id"] = "999"
    elif change == "checkpoint":
        r["checkpoint_sha256"] = "0" * 64
    elif change == "lr":
        r["learning_rate"] = 5e-5
    elif change == "block":
        r["block"] = 1 - r["block"]
    elif change == "condition":
        r["condition"] = "wrong"
    elif change == "rank":
        r["rank"] = 1
    elif change == "ordinal":
        r["ordinal"] += 1
    elif change == "prompt":
        r["prompt_ids"][0] += 1
    elif change == "response_text":
        r["response_text"] = "proof"
    elif change == "early_eos":
        r["response_ids"] = [population[0].eos_token_id, 17]
    elif change == "bool_token":
        r["response_ids"] = [True]
    elif change == "short_stop":
        r["response_ids"] = [17]
    elif change == "verifier":
        r["verification"]["reward"] = 1.0
    elif change == "parsed":
        r["parsed_trace"]["answer"] = 1
    elif change == "tags":
        r["diagnostic_answer_tag_present"] = True
    elif change == "extra":
        r["selected"] = True
    elif change == "missing":
        records["treatment"][0].pop()
    elif change == "swapped_rows":
        records["treatment"][0][0], records["treatment"][0][1] = (
            records["treatment"][0][1],
            records["treatment"][0][0],
        )
    elif change == "canonical_length":
        prompts[0]["canonical_response_tokens"] += 1
    elif change == "readout":
        report["models"]["control"]["readouts"]["cells"]["0"]["preserve"]["branch"]["proof_correct"] += 1
    with pytest.raises((ValueError, KeyError)):
        audit.replay(report, prompts, records, population[0])


def test_real_target_proofs_replay_and_pair_factorial_direction(
    population, independent_population, produced, monkeypatch
):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    tokenizer, views, _, prompts, _ = population
    report, _, records, _ = produced
    report, records = copy.deepcopy(report), copy.deepcopy(records)
    task = ProofGraphTask()
    quality = worker.sibling("sdsc_student_quality_probe")
    for arm in worker.ARMS:
        flat = []
        for group in records[arm]:
            for r in group:
                i = r["ordinal"]
                v = views[i]
                if v.condition == "preserve":
                    text = task.canonical_target(v.example)
                    tokens = [*tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id]
                    r.update(
                        quality.response_record(
                            v.example,
                            tokens,
                            tokenizer,
                            prompt_ids=prompts[i]["prompt_ids"],
                            prompt_text=prompts[i]["prompt_text"],
                            cap=256,
                        )
                    )
                    r["parsed_trace"] = asdict(task.parse_response(text))
                flat.append(r)
        report["models"][arm]["readouts"] = worker.reduce_records(flat)
    expected, examples, ta, ds = independent_population
    monkeypatch.setattr(audit, "population", lambda _: (expected, examples, ta, ds))
    result = audit.replay(report, prompts, records, tokenizer)
    c = result["models"]["control"]["contrasts"]["0"]["branch"]["pair_cue_paths_present"]["proof_correct"]
    assert c["present_only"] == 48 and c["absent_only"] == 0 and c["difference_numerator"] == 48
    assert (
        result["models"]["control"]["contrasts"]["0"]["chain"]["interaction"]["proof_correct"]["numerator"]
        == 16
    )


@pytest.mark.parametrize("change", ["arm", "scope", "format", "step", "lr", "flag", "extra"])
def test_dense_provenance_rejection(change):
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    d = p.SOURCE_CHECKPOINTS["control"]
    saved = dict(
        format=d["dense_format"],
        arm="control",
        mode="probe",
        model={},
        global_step=32,
        parent_checkpoint_sha256=d["parent_checkpoint_sha256"],
        dataset_sha256=d["dataset_sha256"],
        protocol_sha256=d["protocol_sha256"],
        protocol_artifact_sha256=d["protocol_artifact_sha256"],
        learning_rate=d["learning_rate"],
        scope=d["scope"],
        **worker.FLAGS,
    )
    worker.validate_dense_payload(saved, "control")
    if change == "arm":
        saved["arm"] = "treatment"
    elif change == "scope":
        saved["scope"] = "formal_initial"
    elif change == "format":
        saved["format"] = "student_dense_v1"
    elif change == "step":
        saved["global_step"] = 16
    elif change == "lr":
        saved["learning_rate"] = 2.5e-5
    elif change == "flag":
        saved["student_accepted"] = True
    elif change == "extra":
        saved["optimizer"] = {}
    with pytest.raises(ValueError):
        worker.validate_dense_payload(saved, "control")


def test_real_tiny_fp32_reload_bf16_duplicate_parity_and2454_context(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.models import loading

    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        use_cache=False,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).float().eval().requires_grad_(False)
    state = model.state_dict()
    arm = "control"
    d = copy.deepcopy(worker.source_descriptor(arm))
    saved = dict(
        format=d["dense_format"],
        arm=arm,
        mode="probe",
        model=state,
        global_step=32,
        parent_checkpoint_sha256=d["parent_checkpoint_sha256"],
        dataset_sha256=d["dataset_sha256"],
        protocol_sha256=d["protocol_sha256"],
        protocol_artifact_sha256=d["protocol_artifact_sha256"],
        learning_rate=d["learning_rate"],
        scope=d["scope"],
        **worker.FLAGS,
    )
    path = tmp_path / "tiny.pt"
    torch.save(saved, path)
    d.update(
        size=path.stat().st_size, sha256=worker.file_sha(path), model_state_sha256=torch_state_hash(state)
    )
    monkeypatch.setattr(worker, "source_descriptor", lambda _: d)
    monkeypatch.setattr(worker, "SOURCE_TENSOR_COUNT", len(state))

    def factory(config, for_training):
        assert for_training is False
        return SimpleNamespace(
            model=Qwen3ForCausalLM(cfg).eval().requires_grad_(False),
            resolved_model_commit=worker.specification().BASE_REVISION,
            resolved_tokenizer_commit=worker.specification().BASE_REVISION,
            tokenizer_hash=worker.util.TOKENIZER_SHA,
            chat_template_sha256=worker.specification().CHAT_TEMPLATE_SHA256,
            prompt_protocol="qwen3_non_thinking_v1",
        )

    monkeypatch.setattr(loading, "load_model_and_tokenizer", factory)
    loaded, evidence = worker.load_prepared({}, path, arm, torch.device("cpu"))
    rows = [dict(input_ids=[1, 2, 3], prefix_length=3), dict(input_ids=[4, 5], prefix_length=2)]
    exact, parity, context = worker.duplicate_reload_parity(loaded, path, arm, rows, torch.device("cpu"))
    assert exact == evidence["exact_saved_master_reload"] and all(x["bitwise_equal"] for x in parity)
    assert context["input_length"] == 2454 and context["all_logits_finite"]
    assert all(x.dtype == torch.bfloat16 and not x.requires_grad for x in loaded.model.parameters())


def test_capped_prefix_diagnosis_does_not_change_original_failure(population):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    example = population[1][0].example
    task = ProofGraphTask()
    text = task.canonical_target(example)
    lines = text.splitlines()
    raw = "\n".join(lines[:3])
    parsed = task.parse_response(raw)
    verified = task.verify(example, parsed)
    record = dict(response_text=raw, stop_reason="length", verification=asdict(verified))
    before = copy.deepcopy(record)
    detail = audit.prefix_diagnosis(example, record, parsed, verified)
    assert (
        record == before
        and verified.reward == 0
        and detail["synthetic_closure_used"]
        and detail["original_score_unchanged"]
    )


@pytest.mark.parametrize(
    "mutation", [None, "original_lf", "dataset", "embedded", "population", "flag", "token"]
)
def test_retained_dataset_isolation_original_bytes_bound(mutation):
    from posttrain_circuits.experiments.protocols import student_name_cue_probe as p

    report = valid_report()
    iso = report["data_audit"]["isolation"]
    token = report["data_audit"]["token_audit"]
    files = {
        "name-cue-dataset-manifest.json": audit.canonical(p.dataset_manifest(p.make_examples())) + b"\n",
        "name-cue-data-isolation.json": audit.canonical(iso) + b"\n",
        "name-cue-token-audit.json": audit.canonical(token) + b"\n",
    }
    if mutation == "dataset":
        iso["dataset_manifest_sha256"] = "0" * 64
    elif mutation == "embedded":
        iso["counts"]["original_family"] -= 1
    elif mutation == "population":
        iso["counts"]["original_family"] -= 1
        files["name-cue-data-isolation.json"] = audit.canonical(iso) + b"\n"
    elif mutation == "flag":
        iso["formal_generated_responses_or_scores_read"] = True
        files["name-cue-data-isolation.json"] = audit.canonical(iso) + b"\n"
    elif mutation == "token":
        token["views"] = 511
    if mutation not in (None, "original_lf"):
        with pytest.raises(ValueError):
            audit.validate_dataset_isolation(report, files)
    else:
        result = audit.validate_dataset_isolation(report, files)
        assert result["producer_evidence_sha256"] == audit.sha(files["name-cue-data-isolation.json"])
        assert result["producer_evidence_sha256"] != audit.sha(audit.canonical(iso))
        assert result["historical_isolation_independently_recomputed"] is False


def publication_fixture():
    report = valid_report()
    data = b"{}\n"
    files = {r["path"]: data for r in report["raw_artifacts"]}
    report["raw_artifacts"] = [
        dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in sorted(files.items())
    ]
    binding = SimpleNamespace(
        science_file_sha256={"fixture": "1" * 64},
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        protocol_sha256="3" * 64,
        artifact_sha256="4" * 64,
    )
    plan = dict(
        schema="quest-sdsc-student-name-cue-probe-plan-v1",
        task=worker.TASK,
        mode="probe",
        run_id="fixture",
        intent_id="a" * 32,
        code_sha256="1" * 64,
        protocol={
            k: getattr(binding, k)
            for k in (
                "science_file_sha256",
                "implementation_commit",
                "acceptance_commit",
                "protocol_sha256",
                "artifact_sha256",
            )
        },
        control_sha256={
            "tools/sdsc_student_name_cue_probe_audit.py": audit.sha(
                (ROOT / "tools/sdsc_student_name_cue_probe_audit.py").read_bytes()
            )
        },
        **worker.FLAGS,
    )
    report["plan_sha256"] = audit.sha(audit.canonical(plan))
    files["name-cue-report.json"] = audit.canonical(report)
    publication = dict(
        schema="quest-sdsc-student-name-cue-probe-publication-v1",
        task=worker.TASK,
        mode="probe",
        run_id="fixture",
        intent_id="a" * 32,
        job_id=report["job_id"],
        plan_sha256=report["plan_sha256"],
        code_sha256=plan["code_sha256"],
        passed=True,
        stage_complete=True,
        diagnostic_complete=True,
        preparation_complete=False,
        selected_checkpoint=None,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=[],
        files=[dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in sorted(files.items())],
        **worker.FLAGS,
    )
    return plan, publication, files, binding


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "external_hash",
        "promotion",
        "large",
        "file_bytes",
        "raw_inventory",
        "lineage",
        "auditor",
        "path",
        "duplicate",
    ],
)
def test_independent_publication_binding(mutation):
    plan, pub, files, binding = publication_fixture()
    if mutation == "promotion":
        pub["formal_initial_accepted"] = True
    elif mutation == "large":
        pub["large_files"] = [{"path": "checkpoint.pt"}]
    elif mutation == "file_bytes":
        files["name-cue-prompts.jsonl"] = b"bad"
    elif mutation == "raw_inventory":
        r = json.loads(files["name-cue-report.json"])
        r["raw_artifacts"].pop()
        files["name-cue-report.json"] = audit.canonical(r)
        pub["files"] = [dict(path=k, size=len(v), sha256=audit.sha(v)) for k, v in sorted(files.items())]
    elif mutation == "lineage":
        binding.implementation_commit = "9" * 40
    elif mutation == "auditor":
        plan["control_sha256"]["tools/sdsc_student_name_cue_probe_audit.py"] = "0" * 64
    elif mutation == "path":
        pub["files"][0]["path"] = "../escape"
    elif mutation == "duplicate":
        pub["files"].append(pub["files"][0])
    raw = audit.canonical(pub)
    expected = "0" * 64 if mutation == "external_hash" else audit.sha(raw)
    if mutation is None:
        report, found = audit.validate_publication(plan, raw, expected, files, binding)
        assert report["selected_checkpoint"] is None and found["large_files"] == []
    else:
        with pytest.raises((ValueError, KeyError)):
            audit.validate_publication(plan, raw, expected, files, binding)
