"""Synthetic publication fixtures exercise a lossless accepted-view boundary.

Fixture hashes replace the externally accepted publication only within tests;
these tests neither simulate scientific acceptance nor supply GPU evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from posttrain_circuits.artifacts import adapted_teacher_sft as view
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import TeacherIdentity
from posttrain_circuits.learning.contracts import PromptBatch
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.teacher.seeding import teacher_candidate_seed


def sealed(value):
    return {**value, "sha256": sha256_value(value)}


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from posttrain_circuits.artifacts import adapted_student_protocol as protocol

    block = protocol.adapted_teacher_config()
    identity = TeacherIdentity.from_mapping(block["teacher_identity"])
    config = {
        "adapted_teacher": copy.deepcopy(block),
        "teacher": {
            "model_name_or_path": identity.base_model_id,
            "model_revision": identity.base_revision,
            "tokenizer_name_or_path": identity.tokenizer_id,
            "tokenizer_revision": identity.tokenizer_revision,
        },
    }
    # The accepted consumer protocol is separately tested with real Git history.
    monkeypatch.setattr(protocol, "validate_student_protocol", lambda _config: None)
    prompts = [f"prompt-{index:03d}" for index in range(256)]
    rows = []
    for prompt in prompts:
        prompt_hash = sha256_value(prompt)
        for candidate in range(8):
            rows.append(
                {
                    "attempt_id": f"{prompt}:candidate-{candidate:04d}",
                    "prompt_id": prompt,
                    "prompt_identity_sha256": prompt_hash,
                    "candidate_index": candidate,
                    "raw_prompt_text": "raw " + prompt,
                    "model_facing_prompt_text": "model " + prompt,
                    "input_ids": [1, 2],
                    "response_ids": [3 + candidate, 0],
                    "response_text": "proof",
                    "response_token_mask": [True, True],
                    "behavior_logprobs": [-0.1, -0.2],
                    "logprob_status": "available",
                    "finish_reason": "eos",
                    "teacher_identity_sha256": identity.sha256,
                    "sampling_request_seed": 31415,
                    "actual_sampling_seed": teacher_candidate_seed(31415, prompt_hash, candidate),
                    "sampling_protocol_id": "teacher-demo-v2-prompt-identity",
                    "sampling_temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "verifier_reward": 1.0,
                    "verification_trace": {"reward": 1.0},
                    "accepted": True,
                    "prompt_protocol": identity.prompt_protocol,
                    "enable_thinking": False,
                    "chat_template_sha256": identity.chat_template_sha256,
                    "raw_prompt_sha256": sha256_value("raw " + prompt),
                    "model_facing_prompt_sha256": sha256_value("model " + prompt),
                    "tokenizer_fingerprint": identity.tokenizer_fingerprint,
                }
            )
    stage_sha = "c" * 64
    evidence = sealed({"external_evidence_bindings": {"formal_store": {"sha256": stage_sha}}})
    attestation = sealed(
        {
            "decision": "accepted",
            "reviewed_evidence_sha256": evidence["sha256"],
            "acceptance_commit": view.PRODUCER_HEAD,
        }
    )
    acceptance = sealed(
        {
            "formal_teacher_accepted": True,
            "original_128_token_readiness_pass_claim": False,
            "protocol_sha256": view.TEACHER_PROTOCOL_SHA256,
            "evidence_sha256": evidence["sha256"],
            "teacher_identity": identity.to_mapping(),
            "independent_attestation_sha256": attestation["sha256"],
        }
    )
    block["acceptance_sha256"] = acceptance["sha256"]
    config["adapted_teacher"] = copy.deepcopy(block)
    monkeypatch.setattr(protocol, "adapted_teacher_config", lambda: copy.deepcopy(block))
    common = {
        "teacher_identity": identity.to_mapping(),
        "protocol_sha256": view.TEACHER_PROTOCOL_SHA256,
        "evidence_sha256": stage_sha,
        "formal_teacher_accepted": False,
    }
    accepted_view = sealed({**common, "attempt_ids": [row["attempt_id"] for row in rows]})
    documents = {
        "teacher-store/attempts.jsonl": b"\n".join(raw(row) for row in rows) + b"\n",
        "teacher-store/accepted-view.json": raw(accepted_view),
        "acceptance/accepted-teacher.json": raw(acceptance),
        "acceptance/independent-attestation.json": raw(attestation),
        "acceptance/publication-review.json": raw({"fixture": "externally reviewed"}),
        "acceptance/audit-bindings.json": raw({"fixture": "audited"}),
        "acceptance-evidence.json": raw(evidence),
    }
    source = sealed(
        {
            **common,
            "ordered_prompt_ids": prompts,
            "attempt_count": 2048,
            "covered_prompts": 256,
            "accepted_view_sha256": accepted_view["sha256"],
            "files": [
                {
                    "path": name,
                    "size": len(documents["teacher-store/" + name]),
                    "sha256": hashlib.sha256(documents["teacher-store/" + name]).hexdigest(),
                }
                for name in ("attempts.jsonl", "accepted-view.json")
            ],
        }
    )
    documents["teacher-store/manifest.json"] = raw(source)
    documents["producer-report.json"] = raw(
        {
            "passed": True,
            "exit_code": 0,
            "job_id": "54496291",
            "science_git_head": view.PRODUCER_HEAD,
            "teacher_store_sha256": source["sha256"],
            "adapted_teacher_sha256": identity.teacher_checkpoint_sha256,
        }
    )
    for name, data in documents.items():
        path = tmp_path / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(
        view,
        "INPUT_FILES",
        {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in documents.items()},
    )
    monkeypatch.setattr(view, "STORE_SHA256", source["sha256"])
    monkeypatch.setattr(view, "EVIDENCE_SHA256", evidence["sha256"])
    return tmp_path / "inputs", config, rows, source, accepted_view, identity


def test_real_source_identity_conversion_preserves_all_measurements_and_cursor(fixture):
    root, config, rows, _, _, identity = fixture
    before = {name: (root / name).read_bytes() for name in view.INPUT_FILES}
    attempts, manifest = view.read_accepted_teacher_sft(root, config=config)
    assert len(attempts) == 2048
    for original, attempt in zip(rows, attempts, strict=True):
        converted = asdict(attempt)
        assert converted.pop("teacher_id") == view.ADAPTED_MODEL_ID
        assert converted.pop("teacher_revision") == identity.teacher_checkpoint_sha256
        expected = {key: value for key, value in original.items() if key != "teacher_identity_sha256"}
        assert converted == expected
    assert manifest["protocol_bindings"]["producer_science_head"] == view.PRODUCER_HEAD
    assert view.validate_adapted_sft_manifest(manifest, config) == identity
    assert {name: (root / name).read_bytes() for name in view.INPUT_FILES} == before
    source = TeacherDemoStateSource(attempts)
    batch = PromptBatch((rows[0]["prompt_id"],), (rows[0]["raw_prompt_text"],))
    first = source.get_batch(None, batch, 0).records[0]
    saved = source.state_dict()
    second = source.get_batch(None, batch, 1).records[0]
    source.load_state_dict(saved)
    resumed = source.get_batch(None, batch, 1).records[0]
    assert first.response_ids == rows[0]["response_ids"]
    assert second == resumed and second.response_ids == rows[1]["response_ids"]
    assert first.behavior_policy_revision == identity.teacher_checkpoint_sha256
    assert first.behavior_logprobs == rows[0]["behavior_logprobs"]


@pytest.mark.parametrize(
    "name",
    [
        "teacher-store/attempts.jsonl",
        "acceptance/accepted-teacher.json",
        "acceptance-evidence.json",
        "producer-report.json",
    ],
)
def test_modified_source_or_acceptance_is_rejected(fixture, name):
    root, config, *_ = fixture
    path = root / name
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="published bytes"):
        view.read_accepted_teacher_sft(root, config=config)


def test_extra_files_and_symlinks_are_rejected(fixture):
    root, config, *_ = fixture
    extra = root / "shadow.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="extra"):
        view.read_accepted_teacher_sft(root, config=config)
    extra.unlink()
    source = root / "producer-report.json"
    saved = root.parent / "saved-report.json"
    source.rename(saved)
    source.symlink_to(saved)
    with pytest.raises(ValueError, match="symlink"):
        view.read_accepted_teacher_sft(root, config=config)


@pytest.mark.parametrize("mutation", ["reorder", "tokens", "seed", "mask", "identity", "rejected"])
def test_candidate_contract_cannot_change_during_view_conversion(fixture, mutation):
    _, _, rows, source, accepted_view, identity = fixture
    changed = copy.deepcopy(rows)
    if mutation == "reorder":
        changed[0], changed[1] = changed[1], changed[0]
    elif mutation == "tokens":
        changed[0]["response_ids"] = [3] * 257
        changed[0]["behavior_logprobs"] = [-0.1] * 257
        changed[0]["response_token_mask"] = [True] * 257
    elif mutation == "seed":
        changed[0]["actual_sampling_seed"] += 1
    elif mutation == "mask":
        changed[0]["response_token_mask"][0] = False
    elif mutation == "identity":
        changed[0]["teacher_identity_sha256"] = "f" * 64
    else:
        changed[0].update(accepted=False, verifier_reward=0.0)
    with pytest.raises(ValueError):
        view._convert_rows(changed, source, accepted_view, identity)


def test_unaccepted_consumer_protocol_fails_before_reading_teacher_inputs(fixture, monkeypatch):
    from posttrain_circuits.artifacts import adapted_student_protocol as protocol

    _, config, *_ = fixture

    def refused(_config):
        raise ValueError("consumer protocol proposed")

    monkeypatch.setattr(protocol, "validate_student_protocol", refused)
    with pytest.raises(ValueError, match="consumer protocol proposed"):
        view.read_accepted_teacher_sft(Path("/does-not-exist"), config=config)


def test_legacy_identity_and_rehashed_producer_relabel_are_rejected(fixture):
    root, config, *_ = fixture
    _, manifest = view.read_accepted_teacher_sft(root, config=config)
    for field, value in (("producer_science_head", "a" * 40), ("teacher_acceptance_sha256", "b" * 64)):
        changed = copy.deepcopy(manifest)
        changed["protocol_bindings"][field] = value
        changed["sha256"] = sha256_value({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="original teacher provenance"):
            view.validate_adapted_sft_manifest(changed, config)
    changed = copy.deepcopy(manifest)
    changed["behavior_policy"]["id"] = "Qwen/Qwen3-8B"
    changed["sha256"] = sha256_value({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="relabeled"):
        view.validate_adapted_sft_manifest(changed, config)


def test_experiment_binding_names_learned_weights_and_preserves_base_ancestry(fixture):
    from posttrain_circuits.artifacts.config_bindings import bind_config
    from posttrain_circuits.core.config import compose_config
    from posttrain_circuits.experiments.protocols.specs import build_experiment_binding

    root, minimal, _, _, _, identity = fixture
    _, manifest = view.read_accepted_teacher_sft(root, config=minimal)
    config = compose_config(
        [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "task.num_examples=256",
            "state_source.num_candidates=8",
            "seed=42",
        ]
    )
    config["adapted_teacher"] = minimal["adapted_teacher"]
    kwargs = {
        "config": config,
        "config_binding": bind_config(
            config, input_artifact_hashes={"state_source.store_path": manifest["sha256"]}
        ),
        "implementation_commit": "f" * 40,
        "implementation_dirty": False,
        "model_resolved_revision": config["model"]["model_revision"],
        "tokenizer_resolved_revision": config["model"]["tokenizer_revision"],
        "tokenizer_fingerprint": identity.tokenizer_fingerprint,
        "teacher_resolved_revision": identity.teacher_checkpoint_sha256,
        "initial_checkpoint_sha256": "1" * 64,
        "train_dataset_sha256": "2" * 64,
        "validation_dataset_sha256": "3" * 64,
        "model_facing_prompt_schedule_sha256": "4" * 64,
        "probe_manifest_sha256": "5" * 64,
        "task_protocol": {"name": "proofgraph"},
        "optimizer_spec": {"class": "AdamW"},
        "scheduler_spec": {"class": "LambdaLR"},
        "resolved_batch_contract": {"global_batch_size": 64},
        "full_parameter_training": True,
        "teacher_demo_manifest": manifest,
    }
    result = build_experiment_binding(**kwargs)
    assert result.teacher_id == view.ADAPTED_MODEL_ID
    assert (
        result.teacher_requested_revision
        == result.teacher_resolved_revision
        == identity.teacher_checkpoint_sha256
    )
    assert config["teacher"]["model_revision"] == identity.base_revision
    assert result.max_completion_length == 128
    assert result.teacher_demo_manifest_sha256 == manifest["sha256"]
    with pytest.raises(ValueError, match="runtime revision"):
        build_experiment_binding(**{**kwargs, "teacher_resolved_revision": identity.base_revision})


def test_train_refuses_proposed_consumer_before_any_model_load(fixture, monkeypatch):
    from types import SimpleNamespace

    from posttrain_circuits.artifacts import adapted_student_protocol as protocol
    from posttrain_circuits.cli import train

    _, minimal, *_ = fixture
    config = {**minimal, "experiment": {"name": "canonical_sft"}, "output_root": "/tmp/fixture"}
    monkeypatch.setattr(
        train,
        "parse_cli",
        lambda *_: (SimpleNamespace(output=None, dry_run=False, confirm_production=True), config),
    )
    monkeypatch.setattr(train, "is_production_scale", lambda _config: True)
    monkeypatch.setattr(train, "enforce_production_guard", lambda *_args, **_kwargs: True)

    def refused(_config):
        raise ValueError("consumer protocol proposed")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model load preceded consumer admission")

    monkeypatch.setattr(protocol, "validate_student_protocol", refused)
    monkeypatch.setattr(train, "load_model_and_tokenizer", forbidden)
    with pytest.raises(ValueError, match="consumer protocol proposed"):
        train.main([])


@pytest.mark.parametrize(
    "field,value", [("temperature", 0.8), ("sampling_request_seed", 42), ("successful_candidates", 2047)]
)
def test_rehashed_derived_generation_cannot_change_the_measured_policy(fixture, field, value):
    root, config, *_ = fixture
    _, manifest = view.read_accepted_teacher_sft(root, config=config)
    manifest["teacher_demo_generation"][field] = value
    manifest["sha256"] = sha256_value({key: item for key, item in manifest.items() if key != "sha256"})
    with pytest.raises(ValueError, match="population/envelope/teacher identity"):
        view.validate_adapted_sft_manifest(manifest, config)
